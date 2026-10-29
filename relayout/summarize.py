"""Summarize rollout results: success rate with a Wilson 95% interval.

    python -m relayout.summarize runs/<run>/eval_xxx [more eval dirs...]
"""
import json
import math
import sys
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def load(d):
    eps = {}
    for f in sorted(Path(d).glob("episodes_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            eps[r["id"]] = r
    return eps


def summarize(d):
    eps = load(d)
    n = len(eps)
    k = sum(r["success"] for r in eps.values())
    lo, hi = wilson(k, n)
    groups = {}
    for r in eps.values():
        g = groups.setdefault(r["id"].split("_")[0], [0, 0])
        g[0] += r["success"]
        g[1] += 1
    out = {"dir": str(d), "n": n, "success": k, "rate": round(k / n, 4) if n else None,
           "wilson95": [round(lo, 4), round(hi, 4)],
           "by_group": {g: f"{a}/{b}" for g, (a, b) in groups.items()},
           "lifted_but_failed": sum((not r["success"]) and r["max_mug_lift"] > 0.05 for r in eps.values())}
    Path(d, "summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d)))
