"""Decide where to spend a second round of demo generation, and build the resulting datasets.

The offsets are split into bins: 3 radius ranges (2-6, 6-10, 10-12 cm, named by the evaluated
radius 4, 8, 12 cm; 12 cm is the edge of the round-1 range) x 8 directions (every 45 deg, each
+-22.5 deg). A bin is named like the evaluation grid ("r12a180"). The budget is a number of
generation units (one unit = a fixed number of attempts). Three strategies spend the same budget:

  uniform   the same number of units in every bin
  failure   units proportional to the policy's failure rate in each bin
  agent     an LLM reads the per-bin numbers and writes the allocation as JSON (validated)

The agent only decides the allocation; the conversion method, the source demos and the unit size
are fixed. Its inputs come from a development evaluation set that does not overlap the final one.

    python -m relayout.allocate decide --dev runs/g2_gen_s*/eval_shiftdev_last \
        --gen-init data/init/gen_shift_400.npz --gen-report data/lerobot/redmug_gen/generate_report.jsonl \
        --budget 48 --out data/alloc/allocations.json
    python -m relayout.allocate targets --alloc data/alloc/allocations.json --base data/init/gen_base_200.npz \
        --out data/alloc/targets
    python -m relayout.allocate build --alloc data/alloc/allocations.json --pool data/lerobot/pool \
        --round1 data/lerobot/redmug_gen --stats-from data/lerobot/redmug_std --out data/lerobot
"""
import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np

from relayout.summarize import load

RADII = {4: (2, 6), 8: (6, 10), 12: (10, 12)}  # cm: evaluated radius -> range of the bin
ANGLES = tuple(range(0, 360, 45))  # deg, bin centres; each bin is +-22.5 deg
BINS = [f"r{r}a{a}" for r in RADII for a in ANGLES]
STRATEGIES = ("uniform", "failure", "agent")


def bin_of(offset):
    """Bin name of an (dx, dy) offset in metres, or None if it is outside every bin."""
    r = float(np.hypot(*offset)) * 100
    rc = next((c for c, (lo, hi) in RADII.items() if lo <= r <= hi), None)
    if rc is None:
        return None
    a = np.degrees(np.arctan2(offset[1], offset[0])) % 360
    ac = ANGLES[int(((a + 22.5) % 360) // 45)]
    return f"r{rc}a{ac}"


def bin_stats(dev_dirs, gen_init, gen_report):
    """Per-bin numbers the strategies see: policy success on the development set (all seeds pooled),
    how failures look, and the round-1 generation acceptance rate."""
    st = {b: {"trials": 0, "success": 0, "fail_not_lifted": 0, "fail_lifted": 0, "fail_place_cm": [],
              "gen_attempts": 0, "gen_accepted": 0} for b in BINS}
    for d in dev_dirs:
        for eid, r in load(d).items():
            s = st[eid.split("_")[0]]
            s["trials"] += 1
            if r["success"]:
                s["success"] += 1
            elif r["max_mug_lift"] > 0.05:
                s["fail_lifted"] += 1
                s["fail_place_cm"].append(100 * float(np.hypot(*np.subtract(r["mug_end"], r["plate_end"])[:2])))
            else:
                s["fail_not_lifted"] += 1
    init = np.load(gen_init)
    offs = dict(zip(map(str, init["ids"]), json.loads(str(init["meta"]))["offsets"]))
    for line in open(gen_report):
        r = json.loads(line)
        b = bin_of(offs[r["init"]])
        if b:
            st[b]["gen_attempts"] += 1
            st[b]["gen_accepted"] += r["accepted"]
    out = {}
    for b, s in st.items():
        fp = s.pop("fail_place_cm")
        out[b] = {**s, "success_rate": round(s["success"] / s["trials"], 3) if s["trials"] else None,
                  "gen_accept_rate": round(s["gen_accepted"] / s["gen_attempts"], 3) if s["gen_attempts"] else None,
                  "fail_lifted_median_mug_to_plate_cm": round(float(np.median(fp)), 1) if fp else None}
    return out


def largest_remainder(weights, total):
    w = np.asarray(weights, float)
    w = np.full(len(w), 1.0) if w.sum() <= 0 else w
    q = w / w.sum() * total
    n = np.floor(q).astype(int)
    for i in np.argsort(-(q - n), kind="stable")[: total - n.sum()]:
        n[i] += 1
    return n.tolist()


def uniform(stats, budget):
    return dict(zip(BINS, largest_remainder([1] * len(BINS), budget)))


def failure(stats, budget):
    # a bin without development trials counts as failing everywhere
    return dict(zip(BINS, largest_remainder([1 - (stats[b]["success_rate"] or 0) for b in BINS], budget)))


PROMPT = """You are allocating a fixed budget of demonstration-generation work for a robot policy.

Task: a robot arm picks up a red mug and puts it on a plate (LIBERO simulation). All objects on the
table can be shifted together by an offset from the standard layout. An ACT policy was fine-tuned on
{n_round1} converted demonstrations spread uniformly over offsets 0-12 cm. It still fails in some
regions. We will generate more demonstrations by converting the original demos to new layouts and
fine-tune again. Your job is only to decide how many generation units go to each offset bin.

Bins: "r<radius cm>a<direction deg>"; radius bins 4, 8, 12 cover 2-6, 6-10, 10-12 cm; directions every 45 deg
(+-22.5 deg; 0 deg = +x, 90 deg = +y). One unit = {unit} generation attempts in that bin; only
attempts that succeed in simulation become training demos (see the round-1 acceptance rate).
Total budget: exactly {budget} units, non-negative integers, every bin listed (0 allowed).

Per-bin numbers (policy evaluated on a development set, {seeds} seeds pooled; round-1 generation):
{table}

Columns: success_rate = policy success; fail_not_lifted / fail_lifted = failures before / after
lifting the mug; fail_lifted_median_mug_to_plate_cm = for lifted failures, final mug-to-plate
distance; gen_accept_rate = share of round-1 generation attempts accepted in that bin.

Reply with one JSON object and nothing else:
{{"allocation": {{"<bin>": <units>, ...}}, "reasoning": "<2-4 sentences>"}}"""


def agent(stats, budget, unit, cmd, n_round1, seeds, log_dir, tries=3):
    cols = ["success_rate", "trials", "fail_not_lifted", "fail_lifted", "fail_lifted_median_mug_to_plate_cm", "gen_accept_rate"]
    table = "bin\t" + "\t".join(cols) + "\n" + "\n".join(b + "\t" + "\t".join(str(stats[b][c]) for c in cols) for b in BINS)
    prompt = PROMPT.format(n_round1=n_round1, unit=unit, budget=budget, seeds=seeds, table=table)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "agent_prompt.txt").write_text(prompt)
    msg = prompt
    for k in range(tries):
        raw = subprocess.run(shlex.split(cmd), input=msg, capture_output=True, text=True, check=True).stdout
        (log_dir / f"agent_response_{k}.txt").write_text(raw)
        err = None
        try:
            obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            alloc = {b: int(obj["allocation"].get(b, 0)) for b in BINS}
            extra = set(obj["allocation"]) - set(BINS)
            if extra:
                err = f"unknown bins {sorted(extra)}"
            elif min(alloc.values()) < 0 or sum(alloc.values()) != budget:
                err = f"units must be non-negative and sum to {budget} (got {sum(alloc.values())})"
        except (AttributeError, ValueError, KeyError, TypeError) as e:
            err = f"could not parse the JSON object: {e}"
        if err is None:
            return alloc, obj.get("reasoning", "")
        msg = f"{prompt}\n\nYour previous reply was invalid: {err}. Reply again with the JSON object only."
    raise RuntimeError(f"agent gave no valid allocation after {tries} tries: {err}")


def decide(args):
    stats = bin_stats(args.dev, args.gen_init, args.gen_report)
    out = Path(args.out)
    n_round1 = sum(json.loads(line)["accepted"] for line in open(args.gen_report))
    alloc = {"uniform": uniform(stats, args.budget), "failure": failure(stats, args.budget)}
    reasoning = None
    if args.agent_cmd:
        alloc["agent"], reasoning = agent(stats, args.budget, args.unit, args.agent_cmd, n_round1, len(args.dev), out.parent)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": args.budget, "unit": args.unit, "agent_cmd": args.agent_cmd,
                               "agent_reasoning": reasoning, "allocations": alloc, "stats": stats}, indent=1))
    for name, a in alloc.items():
        print(name, " ".join(f"{b}:{a[b]}" for b in BINS if a[b]))
    if reasoning:
        print("agent:", reasoning)


def targets(args):
    """For each bin, enough shifted initial states for the largest allocation (+ margin for dropped states)."""
    from relayout import sim
    from relayout.shift import shift_state
    al = json.loads(Path(args.alloc).read_text())
    unit = al["unit"]
    need = {b: max(a[b] for a in al["allocations"].values()) for b in BINS}
    base = np.load(args.base)
    rng = np.random.default_rng(args.seed)
    env = sim.make_env()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    j = 0
    for b in BINS:
        n = need[b] * unit
        if n == 0:
            continue
        rc, ac = map(int, re.match(r"r(\d+)a(\d+)", b).groups())
        states, ids, offs = [], [], []
        while len(states) < n:
            r = rng.uniform(*RADII[rc]) / 100
            th = np.radians(ac + rng.uniform(-22.5, 22.5))
            off = np.array([r * np.cos(th), r * np.sin(th)])
            i = j % len(base["ids"])
            j += 1
            s, drift, drop = shift_state(env, base["states"][i], off)
            if drift > 0.01 or drop > 0.08:
                continue
            states.append(s)
            ids.append(f"{b}_{len(ids)}_{base['ids'][i]}")
            offs.append(off.round(4).tolist())
        np.savez(out / f"{b}.npz", states=np.stack(states), ids=np.array(ids),
                 meta=json.dumps({"bin": b, "offsets": offs, "base": args.base, "seed": args.seed}))
        print(b, len(ids), flush=True)
    env.close()


def build(args):
    """Round-1 demos + the first k units of each bin's pool, for every strategy."""
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from relayout.dataset import use_stats_from
    al = json.loads(Path(args.alloc).read_text())
    pool = Path(args.pool)
    for name in args.strategies or al["allocations"]:
        parts, n_new = [LeRobotDataset("local/round1", root=args.round1)], 0
        for b in BINS:
            for k in range(al["allocations"][name][b]):
                d = pool / b / f"u{k}"
                rep = d / "generate_report.jsonl"
                if not rep.exists():
                    raise FileNotFoundError(f"{rep}: unit not generated yet")
                if any(json.loads(line)["accepted"] for line in open(rep)):  # a unit with no accepted demo has no dataset
                    parts.append(LeRobotDataset(f"local/{b}_u{k}", root=d))
                    n_new += parts[-1].meta.total_episodes
        out = Path(args.out) / f"redmug_{name}"
        merge_datasets(parts, output_repo_id=f"local/redmug_{name}", output_dir=out)
        use_stats_from(out, args.stats_from)
        print(name, "round1", parts[0].meta.total_episodes, "+ new", n_new, "->", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decide")
    p.add_argument("--dev", nargs="+", required=True, help="eval dirs on the development set (one per seed)")
    p.add_argument("--gen-init", required=True)
    p.add_argument("--gen-report", required=True)
    p.add_argument("--budget", type=int, default=48, help="units")
    p.add_argument("--unit", type=int, default=10, help="attempts per unit")
    p.add_argument("--agent-cmd", default="claude -p", help="command that reads a prompt on stdin ('' to skip)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("targets")
    p.add_argument("--alloc", required=True)
    p.add_argument("--base", required=True, help="base states disjoint from evaluation (gen_base_200.npz)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", required=True)
    p = sub.add_parser("build")
    p.add_argument("--alloc", required=True)
    p.add_argument("--pool", required=True, help="<pool>/<bin>/u<k> datasets from relayout.generate")
    p.add_argument("--round1", required=True)
    p.add_argument("--stats-from", required=True)
    p.add_argument("--strategies", nargs="*")
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"decide": decide, "targets": targets, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
