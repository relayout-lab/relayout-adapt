"""Record videos for people: run policies from chosen initial states and render a wide view.

The view is the front camera moved to look the same way as the policy's camera but from further
back, so both plates stay in frame when the layout is shifted. The policy still sees its own
128 px cameras; only the recorded frames differ.

    python -m relayout.record --policy outputs/g2_gen_s1 --init data/init/shift_grid.npz \
        --ids r12a90_official_1 --out outputs/video/gen

With --source K instead of --policy, records source demo K converted to each chosen layout
(relayout.generate), plus the source demo replayed in its own layout (<out>/source_K.mp4).

Writes <out>/<id>_<ok|fail>.mp4 (H.264) and <out>/episodes.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from relayout import sim
from relayout.rollout import load_policy, run_episode, save_video

CAM = "frontview"


def wide_view(width, height, back):
    """frame_fn for run_episode: place the camera once per (re)built model, then render."""
    from robosuite.utils.transform_utils import quat2mat
    placed = {"model": None}

    def frame(env, obs):
        m = env.sim.model
        if placed["model"] is not m:
            c, a = m.camera_name2id(CAM), m.camera_name2id("agentview")
            m.cam_quat[c], m.cam_fovy[c] = m.cam_quat[a], m.cam_fovy[a]
            rot = quat2mat(np.r_[m.cam_quat[a][1:], m.cam_quat[a][0]])  # (w, x, y, z) -> (x, y, z, w)
            m.cam_pos[c] = m.cam_pos[a] + rot @ np.array([0.0, 0.0, back])  # cameras look along -z
            env.sim.forward()
            placed["model"] = m
        return env.sim.render(width=width, height=height, camera_name=CAM)[::-1]
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--source", type=int, help="record converted demos of this source demo instead of a policy")
    ap.add_argument("--hdf5", default="data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5")
    ap.add_argument("--init", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--back", type=float, default=0.25, help="how far behind the policy camera (m)")
    ap.add_argument("--max-steps", type=int, default=sim.MAX_STEPS)
    args = ap.parse_args()
    data = np.load(args.init)
    ids = [str(i) for i in data["ids"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = sim.make_env()
    frame_fn = wide_view(args.width, args.height, args.back)
    if args.source is None:
        policy, pre, post = load_policy(args.policy, "cuda")
    else:
        import h5py
        from types import SimpleNamespace
        from relayout import generate
        with h5py.File(args.hdf5) as f:
            demo = f["data"][f"demo_{args.source}"]
            src = generate.record_source(env, demo)
            frames = []
            env.reset()
            obs = env.set_init_state(demo["states"][0])
            for a in src["act"]:
                obs, _, _, _ = env.step(a)
                frames.append(frame_fn(env, obs))
        save_video(frames, out / f"source_{args.source}.mp4")
        gen_args = SimpleNamespace(extra_steps=40, place_tol=0.03, disturb_tol=0.02)
    with open(out / "episodes.jsonl", "a") as log:
        for eid in args.ids:
            frames = []
            state = data["states"][ids.index(eid)]
            if args.source is None:
                rec = run_episode(env, policy, pre, post, state, args.max_steps, frames, frame_fn)
            else:
                _, rec = generate.attempt(env, src, state, gen_args, frame_fn, frames)
                rec["success"] = rec["accepted"]
            rec = {"id": eid, "policy": args.policy, "source": args.source, **rec}
            save_video(frames, out / f"{eid}_{'ok' if rec['success'] else 'fail'}.mp4")
            log.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    env.close()


if __name__ == "__main__":
    main()
