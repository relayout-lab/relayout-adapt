"""Roll out a trained LeRobot policy from a fixed set of initial states.

    python -m relayout.rollout --policy outputs/act/checkpoints/last/pretrained_model \
        --init data/init/std_200.npz --out runs/eval_x --shard 0/4 --videos 5

--set overrides policy config fields at load time, e.g. ACT temporal ensembling:
    --set n_action_steps=1 --set temporal_ensemble_coeff=0.01

Writes one JSON line per episode to <out>/episodes_<shard>.jsonl. Besides success, each
line keeps the final gripper / object state so failures can be analysed later.
"""
import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import torch

from relayout import sim

MUG, PLATE = "red_coffee_mug_1", "plate_1"


def load_policy(path, device, overrides=()):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    cfg.device = device
    for kv in overrides:
        key, val = kv.split("=", 1)
        assert hasattr(cfg, key), key
        setattr(cfg, key, ast.literal_eval(val))
    cfg.__post_init__()  # re-run validation, e.g. temporal ensembling needs n_action_steps=1
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def to_batch(obs):
    po = sim.policy_obs(obs)
    batch = {k: torch.from_numpy(v).permute(2, 0, 1).float().div(255).unsqueeze(0)
             for k, v in po.items() if k.startswith("observation.images")}
    batch["observation.state"] = torch.from_numpy(po["observation.state"]).unsqueeze(0)
    batch["task"] = [sim.TASK_TEXT]
    return batch


def body_pos(env, name):
    obj = env.env.objects_dict[name]
    return env.sim.data.body_xpos[env.sim.model.body_name2id(obj.root_body)].copy()


def run_episode(env, policy, pre, post, state, max_steps, frames=None):
    obs = sim.reset_to(env, state)
    policy.reset()
    mug0 = body_pos(env, MUG)
    max_lift, done, t = 0.0, False, 0
    for t in range(1, max_steps + 1):
        with torch.inference_mode():
            action = post(policy.select_action(pre(to_batch(obs))))
        action = action.squeeze(0).float().cpu().numpy()
        obs, _, done, _ = env.step(action)
        max_lift = max(max_lift, float(body_pos(env, MUG)[2] - mug0[2]))
        if frames is not None:
            frames.append(sim.human_frame(obs))
        if done:
            break
    return {
        "success": bool(done), "steps": t,
        "max_mug_lift": round(max_lift, 4),
        "mug_start": mug0.round(4).tolist(),
        "mug_end": body_pos(env, MUG).round(4).tolist(),
        "plate_end": body_pos(env, PLATE).round(4).tolist(),
        "gripper_end": obs["robot0_gripper_qpos"].round(4).tolist(),
    }


def save_video(frames, path):
    import imageio
    imageio.mimwrite(path, frames, fps=sim.FPS, codec="libx264", quality=8, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-steps", type=int, default=sim.MAX_STEPS)
    ap.add_argument("--videos", type=int, default=0, help="save videos for the first N episodes of this shard")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a policy config field")
    args = ap.parse_args()

    torch.set_num_threads(1)
    k, n = map(int, args.shard.split("/"))
    data = np.load(args.init)
    idx = list(range(k, len(data["ids"]), n))
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    log = open(out / f"episodes_{k}.jsonl", "a")
    done_ids = {json.loads(line)["id"] for line in open(out / f"episodes_{k}.jsonl")}

    policy, pre, post = load_policy(args.policy, args.device, args.set)
    env = sim.make_env()
    for j, i in enumerate(idx):
        eid = str(data["ids"][i])
        if eid in done_ids:
            continue
        frames = [] if j < args.videos else None
        t0 = time.time()
        rec = run_episode(env, policy, pre, post, data["states"][i], args.max_steps, frames)
        rec = {"id": eid, **rec, "sec": round(time.time() - t0, 2), "policy": args.policy, "set": args.set}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if frames:
            save_video(frames, out / "videos" / f"{eid}_{'ok' if rec['success'] else 'fail'}.mp4")
    env.close()


if __name__ == "__main__":
    main()
