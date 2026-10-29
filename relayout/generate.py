"""Convert source demos to new layouts (a simplified MimicGen, translation only).

Each source demo is replayed once in its own layout to record the end-effector pose per step
and the start poses of the mug and the plate. For a target layout, the demo is split at the
first gripper-close command: before it the end-effector path is shifted with the mug, after it
with the plate. The shift is blended in smoothly (no pauses), and the arm follows the shifted
path with the source action as feed-forward plus a proportional correction. An attempt is kept
only if the task succeeds, the mug ends within --place-tol of the plate centre and no other
object moved more than --disturb-tol.

    python -m relayout.generate --hdf5 data/source/<task>_demo.hdf5 --init data/init/gen_shift.npz \
        --out data/lerobot/redmug_gen --attempts 200 --seed 0

Writes the accepted episodes as a LeRobotDataset and one JSON line per attempt to
<out>/generate_report.jsonl (accepted or the reason it was rejected).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from relayout import dataset, sim
from relayout.convert_demos import is_noop
from relayout.rollout import MUG, PLATE, body_pos

POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC_POSE: action 1.0 = 5 cm / 0.5 rad per step


def _rot():
    from robosuite.utils import transform_utils as tu
    return tu


def record_source(env, demo):
    """Replay a demo; return end-effector poses (T+1), kept actions (T) and start object positions."""
    actions = demo["actions"][()].astype(np.float32)
    env.reset()
    obs = env.set_init_state(demo["states"][0])
    mug0, plate0 = body_pos(env, MUG), body_pos(env, PLATE)
    pos, quat, acts, prev, done = [obs["robot0_eef_pos"].copy()], [obs["robot0_eef_quat"].copy()], [], None, False
    for a in actions:
        if is_noop(a, prev):
            continue
        prev = a
        obs, _, done, _ = env.step(a)
        acts.append(a)
        pos.append(obs["robot0_eef_pos"].copy())
        quat.append(obs["robot0_eef_quat"].copy())
        if done:
            break
    return {"ok": bool(done), "pos": np.array(pos), "quat": np.array(quat), "act": np.array(acts),
            "mug0": mug0, "plate0": plate0}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def offset_schedule(src, mug_off, plate_off, blend=30, settle=10):
    """Per-step xy offset: 0 -> mug offset over the approach, mug -> plate offset after the grasp."""
    T = len(src["act"])
    close = np.flatnonzero(src["act"][:, 6] > 0)
    g = int(close[0]) if len(close) else T
    b1 = max(5, min(blend, int(0.7 * g)))
    out = np.zeros((T + 1, 3))
    for t in range(T + 1):
        w1 = smoothstep(t / b1)
        w2 = smoothstep((t - g - settle) / blend) if t > g + settle else 0.0
        out[t, :2] = (1 - w2) * w1 * mug_off + w2 * plate_off
    return out, g


def track_action(src, t, target_pos, obs, kp=1.0):
    tu = _rot()
    a = src["act"][t].astype(np.float64).copy()
    a[:3] += kp * (target_pos - obs["robot0_eef_pos"]) / POS_SCALE
    r_err = tu.quat2mat(src["quat"][t + 1]) @ tu.quat2mat(obs["robot0_eef_quat"]).T
    a[3:6] += kp * tu.quat2axisangle(tu.mat2quat(r_err)) / ROT_SCALE * 0.5
    a[:6] = np.clip(a[:6], -1, 1)
    return a.astype(np.float32)


def others(env):
    return {n: body_pos(env, n) for n in env.env.objects_dict if n != MUG}


def attempt(env, src, state, args):
    """Run one conversion from a target initial state. Returns (frames, info)."""
    obs = sim.reset_to(env, state)
    mug, plate = body_pos(env, MUG), body_pos(env, PLATE)
    mug_off, plate_off = (mug - src["mug0"])[:2], (plate - src["plate0"])[:2]
    off, g = offset_schedule(src, mug_off, plate_off)
    before = others(env)
    frames, done, T = [], False, len(src["act"])
    for t in range(T + args.extra_steps):
        k = min(t, T - 1)
        target = src["pos"][k + 1] + off[k + 1]
        a = track_action(src, k, target, obs)
        if t >= T:  # hold the last pose a little longer if the goal is not reached yet
            a[:6] = np.clip((target - obs["robot0_eef_pos"]) / POS_SCALE, -1, 1).tolist() + [0, 0, 0]
        frames.append({**sim.policy_obs(obs), "action": a})
        obs, _, done, _ = env.step(a)
        if done:
            break
    mug_end, plate_end = body_pos(env, MUG), body_pos(env, PLATE)
    place = float(np.linalg.norm((mug_end - plate_end)[:2]))
    disturb = max(float(np.linalg.norm((p - before[n])[:2])) for n, p in others(env).items())
    info = {"success": bool(done), "steps": len(frames), "grasp_step": g, "place_dist": round(place, 4),
            "disturb": round(disturb, 4), "mug_off": np.round(mug_off, 4).tolist(), "plate_off": np.round(plate_off, 4).tolist()}
    reason = None
    if not done:
        reason = "no_success"
    elif place > args.place_tol:
        reason = "place_far"
    elif disturb > args.disturb_tol:
        reason = "disturbed"
    info["accepted"] = reason is None
    info["reason"] = reason
    return frames, info


def load_sources(env, hdf5, demos, cache):
    if cache and Path(cache).exists():
        return np.load(cache, allow_pickle=True)["sources"].item()
    lo, hi = map(int, demos.split(":"))
    out = {}
    with h5py.File(hdf5) as f:
        for i in range(lo, hi):
            s = record_source(env, f["data"][f"demo_{i}"])
            if s["ok"]:
                out[i] = s
            print(f"source demo_{i}: {'ok' if s['ok'] else 'replay failed'}", flush=True)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, sources=np.array(out, dtype=object))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--demos", default="0:50")
    ap.add_argument("--source-cache", default="data/source/sources_0_50.npz")
    ap.add_argument("--init", required=True, help="npz of target initial states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-id", default="local/redmug_gen")
    ap.add_argument("--attempts", type=int, default=0, help="0 = one attempt per target state")
    ap.add_argument("--range", help="lo:hi, use only these target states (one generation unit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-steps", type=int, default=40)
    ap.add_argument("--place-tol", type=float, default=0.03)
    ap.add_argument("--disturb-tol", type=float, default=0.02)
    ap.add_argument("--no-write", action="store_true", help="only measure the acceptance rate")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    env = sim.make_env()
    sources = load_sources(env, args.hdf5, args.demos, args.source_cache)
    keys = sorted(sources)
    data = dict(np.load(args.init))
    if args.range:
        lo, hi = map(int, args.range.split(":"))
        data = {k: data[k][lo:hi] for k in ("ids", "states")}
    n = args.attempts or len(data["ids"])
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    report = []

    def episodes():
        for j in range(n):
            i = j % len(data["ids"])
            sk = keys[rng.integers(len(keys))]
            frames, info = attempt(env, sources[sk], data["states"][i], args)
            report.append({"attempt": j, "init": str(data["ids"][i]), "source": int(sk), **info})
            print(json.dumps(report[-1]), flush=True)
            if info["accepted"]:
                yield frames

    if args.no_write:
        out.mkdir(parents=True, exist_ok=True)
        for _ in episodes():
            pass
        kept = sum(r["accepted"] for r in report)
    else:
        kept = dataset.write(episodes(), str(out), args.repo_id, args.overwrite)
    (out / "generate_report.jsonl").write_text("".join(json.dumps(r) + "\n" for r in report))
    print(f"accepted {kept}/{n}")
    env.close()


if __name__ == "__main__":
    main()
