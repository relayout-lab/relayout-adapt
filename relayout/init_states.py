"""Build a fixed set of evaluation initial states and save it as .npz.

    python -m relayout.init_states --out data/init/std_200.npz --official 50 --seeded 150 --seed 1000

`official` takes LIBERO's pruned init states for the task; `seeded` samples new placements
from the task's BDDL regions with fixed seeds. Every method is evaluated on the same file.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from relayout import sim


def official_states(n):
    import torch
    from libero.libero import get_libero_path
    path = Path(get_libero_path("init_states")) / sim.TASK_BDDL.replace(".bddl", ".pruned_init")
    states = torch.load(path, weights_only=False)
    return np.asarray(states)[:n]


def seeded_states(n, seed):
    out = []
    for i in range(n):
        env = sim.make_env(seed=seed + i)
        env.reset()
        out.append(env.sim.get_state().flatten().copy())
        env.close()
    return np.stack(out) if out else np.zeros((0,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--official", type=int, default=50)
    ap.add_argument("--seeded", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()

    parts, ids = [], []
    if args.official:
        s = official_states(args.official)
        parts.append(s)
        ids += [f"official_{i}" for i in range(len(s))]
    if args.seeded:
        s = seeded_states(args.seeded, args.seed)
        parts.append(s)
        ids += [f"seed_{args.seed + i}" for i in range(len(s))]
    states = np.concatenate(parts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, states=states, ids=np.array(ids),
             meta=json.dumps({"task": sim.TASK_BDDL, **vars(args)}))
    print(f"saved {len(ids)} states {states.shape} to {args.out}")


if __name__ == "__main__":
    main()
