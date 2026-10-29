"""Replay official LIBERO demonstrations in simulation and save them as a LeRobotDataset.

Images are re-rendered by the same environment code used for evaluation and generation.
No-op actions are dropped and demos whose replay does not reach the goal are discarded.

    python -m relayout.convert_demos --hdf5 data/source/<task>_demo.hdf5 \
        --out data/lerobot/redmug_std --demos 0:50
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from relayout import dataset, sim


def is_noop(action, prev):
    return np.linalg.norm(action[:6]) < 1e-4 and prev is not None and action[6] == prev[6]


def replay(env, demo):
    actions = demo["actions"][()].astype(np.float32)
    env.reset()
    obs = env.set_init_state(demo["states"][0])
    frames, prev, done = [], None, False
    for a in actions:
        if is_noop(a, prev):
            continue
        frames.append({**sim.policy_obs(obs), "action": a})
        prev = a
        obs, _, done, _ = env.step(a)
        if done:
            break
    return frames, bool(done), len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/redmug_std")
    ap.add_argument("--demos", default="0:50", help="slice of demo indices, e.g. 0:45")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    lo, hi = map(int, args.demos.split(":"))
    env = sim.make_env()
    report = []
    with h5py.File(args.hdf5) as f:
        def episodes():
            for i in range(lo, hi):
                frames, ok, n = replay(env, f["data"][f"demo_{i}"])
                report.append({"demo": i, "replay_success": ok, "raw_steps": n, "kept": len(frames)})
                print(json.dumps(report[-1]), flush=True)
                if ok:
                    yield frames
        n = dataset.write(episodes(), args.out, args.repo_id, args.overwrite)
    Path(args.out, "convert_report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in report))
    print(f"saved {n}/{hi - lo} episodes to {args.out}")
    env.close()


if __name__ == "__main__":
    main()
