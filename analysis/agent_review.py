"""Check the agent's allocation: whether its stated reasons hold on the numbers it saw, how many demos
it expected against how many were generated, and how each allocation did on the final evaluation.

Uses the files that `scripts/run.sh alloc` writes:
  data/alloc/allocations.json   development stats shown to the agent, its reasoning, the three allocations
  data/lerobot/pool/<bin>/u<k>  second-round generation, one unit per directory
  outputs/eval/<cond>_s<seed>_shift   final evaluation, cond = gen (the (3) policy), uniform, failure, agent

    python -m analysis.agent_review     # -> outputs/analysis/agent_review.json and a printed summary
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

LIFT = 0.05  # m, same "lifted" threshold as relayout.allocate


def episodes(eval_dir, cond):
    runs = []
    for d in sorted(glob.glob(f"{eval_dir}/{cond}_s*_shift")):
        eps = [json.loads(line) for f in glob.glob(f"{d}/episodes_*.jsonl") for line in open(f)]
        assert eps, d
        runs.append(eps)
    assert runs, f"no {eval_dir}/{cond}_s*_shift"
    return runs


def per_bin(runs, bins):
    """Success rate and share of failures before lifting the mug, per bin (mean over training runs)."""
    out = {}
    for b in bins:
        sr, nl = [], []
        for eps in runs:
            e = [x for x in eps if x["id"].split("_")[0] == b]
            sr.append(np.mean([x["success"] for x in e]))
            nl.append(np.mean([(not x["success"]) and x["max_mug_lift"] <= LIFT for x in e]))
        out[b] = {"success": float(np.mean(sr)), "not_lifted": float(np.mean(nl))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default="data/alloc/allocations.json")
    ap.add_argument("--pool", default="data/lerobot/pool")
    ap.add_argument("--eval", default="outputs/eval")
    ap.add_argument("--out", default="outputs/analysis/agent_review.json")
    args = ap.parse_args()
    al = json.loads(Path(args.alloc).read_text())
    st, A = al["stats"], al["allocations"]
    bins = list(st)
    target = sorted(bins, key=lambda b: -A["agent"][b])[:4]  # the four bins the agent gave the most units
    rest = [b for b in bins if b not in target]

    # 1. diagnosis, on the development numbers the agent saw
    nl_total = sum(st[b]["fail_not_lifted"] for b in bins)
    nl_target = sum(st[b]["fail_not_lifted"] for b in target)
    far = {b: st[b]["fail_lifted_median_mug_to_plate_cm"] for b in bins}
    claimed = re.search(r"\((\d+) of \d+", al.get("agent_reasoning", ""))
    diag = {"agent_units_target": sum(A["agent"][b] for b in target),
            "agent_units_zero_bins": sum(A["agent"][b] == 0 for b in bins),
            "claimed_units_target": int(claimed.group(1)) if claimed else None,
            "not_lifted_share_target": nl_target / nl_total if nl_total else None,
            "not_lifted_target": nl_target, "not_lifted_total": nl_total,
            "lifted_fail_cm_target": {b: far[b] for b in target},
            "lifted_fail_cm_rest_median": float(np.nanmedian([np.nan if far[b] is None else far[b] for b in rest]))}

    # 2. expected and generated demos; every strategy takes the first k units of a bin's pool
    def unit(b, u):
        rep = [json.loads(line) for line in open(f"{args.pool}/{b}/u{u}/generate_report.jsonl")]
        return sum(r["accepted"] for r in rep), len(rep)

    gen = {}
    for k, alloc in A.items():
        acc = {b: sum(unit(b, u)[0] for u in range(alloc[b])) for b in bins}
        att = {b: sum(unit(b, u)[1] for u in range(alloc[b])) for b in bins}
        gen[k] = {"demos": sum(acc.values()), "demos_target": sum(acc[b] for b in target),
                  "demos_rest": sum(acc[b] for b in rest),
                  "accept_target": sum(acc[b] for b in target) / max(1, sum(att[b] for b in target)),
                  "accept_rest": sum(acc[b] for b in rest) / max(1, sum(att[b] for b in rest)),
                  "per_bin": {b: {"units": alloc[b], "accepted": acc[b], "attempts": att[b],
                                  "expected": round(al["unit"] * alloc[b] * st[b]["gen_accept_rate"], 1)} for b in bins}}

    # 3. final evaluation on the shifted grid
    res = {c: per_bin(episodes(args.eval, c), bins) for c in ("gen", *A)}
    final = {c: {"target": float(np.mean([r[b]["success"] for b in target])),
                 "rest": float(np.mean([r[b]["success"] for b in rest])),
                 "not_lifted_target": float(np.mean([r[b]["not_lifted"] for b in target])),
                 "per_bin": r} for c, r in res.items()}
    eff = {}
    for c in A:
        for part in ("target", "rest"):
            gain = 100 * (final[c][part] - final["gen"][part])
            n = gen[c][f"demos_{part}"]
            eff[f"{c}_{part}"] = {"gain_pt": gain, "demos": n, "pt_per_100_demos": 100 * gain / n if n else None}

    # 4. were the weak bins on the development set also weak on the final evaluation?
    dev = np.array([st[b]["success_rate"] for b in bins])
    fin = np.array([res["gen"][b]["success"] for b in bins])
    check = {"pearson_r": float(np.corrcoef(dev, fin)[0, 1]),
             "worst4_dev": sorted(bins, key=lambda b: st[b]["success_rate"])[:4],
             "worst4_final": sorted(bins, key=lambda b: res["gen"][b]["success"])[:4]}

    out = {"target_bins": target, "diagnosis": diag, "generation": gen, "final": final, "efficiency": eff, "dev_vs_final": check}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"1. agent's four bins {target}: {diag['agent_units_target']} units (stated {diag['claimed_units_target']}),"
          f" {diag['agent_units_zero_bins']} bins get none")
    print(f"   failures before lifting in those bins: {nl_target}/{nl_total}")
    print("2. generation (expected = units x unit size x first-round acceptance)")
    for b in target:
        g = gen["agent"]["per_bin"][b]
        print(f"   {b:8s} {g['units']:2d} units  expected {g['expected']:5.1f}  generated {g['accepted']}/{g['attempts']}")
    for c in gen:
        print(f"   {c:8s} +{gen[c]['demos']} demos (four bins {gen[c]['demos_target']}, rest {gen[c]['demos_rest']})"
              f"  acceptance four bins {100 * gen[c]['accept_target']:.0f}% rest {100 * gen[c]['accept_rest']:.0f}%")
    print("3. final success (%): four bins / rest / failures before lifting in the four bins")
    for c in final:
        print(f"   {c:8s} {100 * final[c]['target']:5.1f} {100 * final[c]['rest']:5.1f} {100 * final[c]['not_lifted_target']:5.1f}")
    for k, v in eff.items():
        print(f"   {k:16s} {v['gain_pt']:+6.1f} pt from (3) with {v['demos']:4d} demos")
    print(f"4. (3) per bin, development vs final: r = {check['pearson_r']:.2f}")


if __name__ == "__main__":
    main()
