"""Record videos for people: run policies from chosen initial states and render a wide view.

The view is the front camera moved to look the same way as the policy's camera but from further
back, so both plates stay in frame when the layout is shifted. The policy still sees its own
128 px cameras; only the recorded frames differ.

    python -m relayout.record --policy outputs/g2_gen_s1 --init data/init/shift_grid.npz \
        --ids r12a90_official_1 --out outputs/video/gen

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
    ap.add_argument("--policy", required=True)
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
    policy, pre, post = load_policy(args.policy, "cuda")
    env = sim.make_env()
    frame_fn = wide_view(args.width, args.height, args.back)
    with open(out / "episodes.jsonl", "a") as log:
        for eid in args.ids:
            frames = []
            rec = run_episode(env, policy, pre, post, data["states"][ids.index(eid)], args.max_steps, frames, frame_fn)
            rec = {"id": eid, "policy": args.policy, **rec}
            save_video(frames, out / f"{eid}_{'ok' if rec['success'] else 'fail'}.mp4")
            log.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    env.close()


if __name__ == "__main__":
    main()
