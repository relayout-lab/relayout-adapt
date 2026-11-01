"""Run the (3) policy on the robot-side and far bins of the shifted grid and record whether the elbow
(joint 4) reaches its lower limit, to check that the policy fails where the conversion fails.

    python -m analysis.policy_elbow [--policy outputs/ft_gen_s1/checkpoints/last/pretrained_model]
    # -> outputs/analysis/policy_elbow.jsonl (one line per episode) and a per-bin summary
"""
import argparse
import json
from pathlib import Path

import numpy as np

BINS = ["r12a180", "r12a225", "r8a180", "r4a0", "r8a0", "r12a90"]
LIFT = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="outputs/ft_gen_s1/checkpoints/last/pretrained_model")
    ap.add_argument("--init", default="data/init/shift_grid.npz")
    ap.add_argument("--out", default="outputs/analysis/policy_elbow.jsonl")
    args = ap.parse_args()
    import torch
    from relayout import sim
    from relayout.rollout import load_policy, run_episode
    policy, pre, post = load_policy(args.policy, "cuda" if torch.cuda.is_available() else "cpu")
    env = sim.make_env()
    data = np.load(args.init)
    ids = [str(i) for i in data["ids"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out, "w")
    res = {}
    for b in BINS:
        for i, eid in enumerate(ids):
            if not eid.startswith(b + "_"):
                continue
            margin = []

            def log(env, obs):  # run_episode's frame_fn, used here to record the elbow every step
                e = env.env
                j = e.robots[0].joint_indexes[3]
                margin.append(float(e.sim.data.qpos[j] - e.sim.model.jnt_range[j][0]))
                return None

            rec = run_episode(env, policy, pre, post, data["states"][i], sim.MAX_STEPS, frames=[], frame_fn=log)
            r = {"bin": b, "id": eid, "success": rec["success"], "lifted": rec["max_mug_lift"] > LIFT,
                 "elbow_margin_min": round(min(margin), 3), "steps_at_limit": int(sum(m < 0.01 for m in margin)), "steps": rec["steps"]}
            out.write(json.dumps(r) + "\n")
            out.flush()
            res.setdefault(b, []).append(r)
        rs = res.get(b, [])
        if not rs:
            continue
        hit = [r["elbow_margin_min"] < 0.01 for r in rs]
        fail_nl = [r for r in rs if not r["success"] and not r["lifted"]]
        print(f"{b:8s} n={len(rs)} success {np.mean([r['success'] for r in rs]):.2f}  elbow at limit {np.mean(hit):.2f}"
              f"  failed before lifting {len(fail_nl)} (at limit {sum(r['elbow_margin_min'] < 0.01 for r in fail_nl)})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
