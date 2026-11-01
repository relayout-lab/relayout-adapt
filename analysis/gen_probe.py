"""Replay the second-round generation attempts and record why conversions fail.

Each attempt in data/lerobot/pool/<bin>/u*/generate_report.jsonl is replayed the same way as
relayout.generate (the shifted hand trajectory, tracked with the source actions plus a P correction),
and every step records:
  - tracking error to the target hand position (before / at / after the grasp)
  - mug lift and final mug tilt
  - joint-limit margin (smallest, which joint, lo/hi side, steps at the limit) and the smallest
    singular value of the 6x7 hand Jacobian (small = near a singular pose)
  - contacts: arm (not fingers) touching anything, fingers touching the table

    python -m analysis.gen_probe [bin ...] --workers 8     # -> outputs/analysis/gen_probe.jsonl
"""
import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

LIFT = 0.05  # m, same "lifted" threshold as relayout.allocate


def probe(job):
    from relayout import sim
    from relayout.generate import offset_schedule, track_action
    from relayout.rollout import MUG, PLATE, body_pos
    global _env, _src, _targets
    if "_env" not in globals():
        _env = sim.make_env()
        _src = np.load("data/source/sources_0_50.npz", allow_pickle=True)["sources"].item()
        _targets = {}
    b, rep = job
    if b not in _targets:
        z = np.load(f"data/alloc/targets/{b}.npz")
        _targets[b] = dict(zip(map(str, z["ids"]), z["states"]))
    env, src = _env, _src[rep["source"]]
    obs = sim.reset_to(env, _targets[b][rep["init"]])
    e = env.env
    m, d = e.sim.model, e.sim.data  # the reset rebuilds the sim, so take these after it
    jidx = e.robots[0].joint_indexes
    lo, hi = m.jnt_range[jidx, 0], m.jnt_range[jidx, 1]
    mug_bid = m.body_name2id(e.objects_dict[MUG].root_body)
    mug0, plate = body_pos(env, MUG), body_pos(env, PLATE)
    off, g = offset_schedule(src, (mug0 - src["mug0"])[:2], (plate - src["plate0"])[:2])
    T = len(src["act"])
    err, margin, smin, arm_touch, finger_table, lift, joint_at, side = [], [], [], [], [], [], [], []
    done = False
    for t in range(T + 40):
        k = min(t, T - 1)
        target = src["pos"][k + 1] + off[k + 1]
        a = track_action(src, k, target, obs)
        if t >= T:
            a[:6] = np.clip((target - obs["robot0_eef_pos"]) / 0.05, -1, 1).tolist() + [0, 0, 0]
        obs, _, done, _ = env.step(a)
        err.append(float(np.linalg.norm(target - obs["robot0_eef_pos"])))
        q = d.qpos[jidx]
        mj = np.minimum(q - lo, hi - q)  # per-joint margin to the nearer limit (rad)
        margin.append(float(mj.min()))
        joint_at.append(int(mj.argmin()))
        side.append("lo" if (q - lo)[mj.argmin()] < (hi - q)[mj.argmin()] else "hi")
        J = np.vstack([d.get_site_jacp("gripper0_grip_site").reshape(3, -1)[:, jidx],
                       d.get_site_jacr("gripper0_grip_site").reshape(3, -1)[:, jidx]])
        smin.append(float(np.linalg.svd(J, compute_uv=False)[-1]))
        lift.append(float(d.body_xpos[mug_bid][2] - mug0[2]))
        at = ft = False
        for i in range(d.ncon):
            c = d.contact[i]
            n1, n2 = m.geom_id2name(c.geom1) or "", m.geom_id2name(c.geom2) or ""
            for x, y in ((n1, n2), (n2, n1)):
                if x.startswith("robot0_") and not y.startswith(("robot0_", "gripper0_")):
                    at = True
                if x.startswith("gripper0_") and ("table" in y or "floor" in y):
                    ft = True
        arm_touch.append(at)
        finger_table.append(ft)
        if done:
            break
    zax = d.body_xmat[mug_bid].reshape(3, 3)[:, 2]
    n = len(err)
    pre = slice(0, min(g, n))
    return {"bin": b, "init": rep["init"], "source": rep["source"], "accepted": rep["accepted"], "reason": rep["reason"],
            "success_probe": bool(done), "steps": n, "grasp_step": g, "lifted": max(lift) > LIFT, "lift_max": round(max(lift), 3),
            "err_pre_max_cm": round(100 * max(err[pre]), 2) if g else None,
            "err_at_grasp_cm": round(100 * err[min(g, n - 1)], 2),
            "err_post_max_cm": round(100 * max(err[g:]), 2) if g < n else None,
            "joint_margin_min": round(min(margin), 3),
            "limit_joint_pre": int(joint_at[int(np.argmin(margin[pre]))]) + 1 if g else None,
            "limit_side_pre": side[int(np.argmin(margin[pre]))] if g else None,
            "steps_at_limit_pre": int(sum(x < 0.01 for x in margin[pre])), "joint_margin_pre_min": round(min(margin[pre]), 3) if g else None,
            "sigma_min_min": round(min(smin), 4), "sigma_min_pre_min": round(min(smin[pre]), 4) if g else None,
            "arm_touch": any(arm_touch), "arm_touch_pre": any(arm_touch[pre]), "finger_table": any(finger_table),
            "mug_tilt_end_deg": round(float(np.degrees(np.arccos(np.clip(zax[2], -1, 1)))), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="*", default=["r12a180", "r12a225", "r12a270", "r8a180", "r4a0", "r4a90", "r8a0", "r12a90"])
    ap.add_argument("--pool", default="data/lerobot/pool")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/analysis/gen_probe.jsonl")
    args = ap.parse_args()
    jobs = []
    for b in args.bins:
        for f in sorted(glob.glob(f"{args.pool}/{b}/u*/generate_report.jsonl")):
            jobs += [(b, json.loads(line)) for line in open(f)]
    print(len(jobs), "attempts", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as p, open(args.out, "w") as out:
        for i, r in enumerate(p.imap_unordered(probe, jobs)):
            out.write(json.dumps(r) + "\n")
            out.flush()
            if i % 25 == 0:
                print(i, r["bin"], r["accepted"], r["success_probe"], flush=True)


if __name__ == "__main__":
    main()
