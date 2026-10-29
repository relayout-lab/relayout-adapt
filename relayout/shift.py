"""Make shifted-layout initial states: every object on the table is translated by the same offset.

This models a relative placement error between the robot and the work area (e.g. a mobile base
stopping a few cm off): the objects keep their arrangement, the robot and cameras stay put.

    python -m relayout.shift --base data/init/std_200.npz --out data/init/shift_grid.npz \
        --radii 0.04 0.08 0.12 --angles 8 --per 20

Ids are "r<cm>a<deg>_<base id>" so results can be grouped by offset and paired by base state.
With --random N, offsets are drawn uniformly in radius (0..--rmax) and angle instead, with ids
"g<j>_<base id>"; this is used for generation targets, from base states disjoint from evaluation.
LIBERO's init states hold objects a few cm above the table, so settling drops them ~4 cm.
States where an object slides sideways while settling or falls off the table are dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from relayout import sim


def object_joints(env):
    """qpos address of each table object's free joint, keyed by object name."""
    out = {}
    for name, obj in env.env.objects_dict.items():
        lo, _ = env.sim.model.get_joint_qpos_addr(obj.joints[-1])
        out[name] = lo
    return out


def shift_state(env, state, offset):
    """Return (shifted flattened state, max sideways drift, max drop) for a base state and an (dx, dy) offset."""
    env.reset()
    env.set_init_state(state)
    joints = object_joints(env)
    qpos = env.sim.data.qpos
    for addr in joints.values():
        qpos[addr:addr + 2] += offset
    env.sim.forward()
    shifted = env.sim.get_state().flatten().copy()
    target = {n: qpos[a:a + 3].copy() for n, a in joints.items()}
    for _ in range(sim.SETTLE_STEPS):
        env.step(sim.NOOP)
    d = [env.sim.data.qpos[a:a + 3] - target[n] for n, a in joints.items()]
    return shifted, float(max(np.linalg.norm(x[:2]) for x in d)), float(max(-x[2] for x in d))


def offsets(radii, angles):
    out = []
    for r in radii:
        for k in range(angles):
            th = 2 * np.pi * k / angles
            out.append((f"r{round(r * 100)}a{round(np.degrees(th))}", np.array([r * np.cos(th), r * np.sin(th)])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="npz of base initial states (e.g. data/init/std_200.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radii", type=float, nargs="+", default=[0.04, 0.08, 0.12])
    ap.add_argument("--angles", type=int, default=8)
    ap.add_argument("--per", type=int, default=20, help="base states per offset (the first N of --base)")
    ap.add_argument("--random", type=int, default=0, help="draw N random offsets instead of the grid")
    ap.add_argument("--rmax", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-drift", type=float, default=0.01, help="sideways motion while settling (m)")
    ap.add_argument("--max-drop", type=float, default=0.08, help="drop while settling (m); ~0.04 is normal")
    args = ap.parse_args()

    base = np.load(args.base)
    env = sim.make_env()
    states, ids, dropped = [], [], []
    if args.random:
        rng = np.random.default_rng(args.seed)
        offs = []
        for j in range(args.random):
            i = j % len(base["ids"])
            r, th = rng.uniform(0, args.rmax), rng.uniform(0, 2 * np.pi)
            off = np.array([r * np.cos(th), r * np.sin(th)])
            s, drift, drop = shift_state(env, base["states"][i], off)
            sid = f"g{j}_{base['ids'][i]}"
            if drift > args.max_drift or drop > args.max_drop:
                dropped.append({"id": sid, "drift": round(drift, 4), "drop": round(drop, 4)})
                continue
            states.append(s)
            ids.append(sid)
            offs.append(off.round(4).tolist())
        args.offsets = offs
    for tag, off in ([] if args.random else offsets(args.radii, args.angles)):
        for i in range(args.per):
            s, drift, drop = shift_state(env, base["states"][i], off)
            bid = str(base["ids"][i])
            if drift > args.max_drift or drop > args.max_drop:
                dropped.append({"id": f"{tag}_{bid}", "drift": round(drift, 4), "drop": round(drop, 4)})
                continue
            states.append(s)
            ids.append(f"{tag}_{bid}")
        print(tag, off.round(3).tolist(), "kept", sum(x.startswith(tag + "_") for x in ids), flush=True)
    env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, states=np.stack(states), ids=np.array(ids),
             meta=json.dumps({"task": sim.TASK_BDDL, "dropped": dropped, **vars(args)}))
    print(f"saved {len(ids)} states to {args.out} ({len(dropped)} dropped)")


if __name__ == "__main__":
    main()
