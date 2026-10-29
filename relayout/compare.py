"""Compare two conditions on the same initial states: success rates and an exact McNemar test.

Each condition is one or more eval dirs (one per seed); dirs are paired in order (seed k of A with
seed k of B) and episodes are paired by id, so every pair shares the same initial state.

    python -m relayout.compare --a runs/g2_std_s{1,2,3}/eval_shift_last \
        --b runs/g2_gen_s{1,2,3}/eval_shift_last --names std gen

Prints one JSON line per seed, one for all seeds pooled, and the pooled split by offset radius
("r<cm>" prefix of the id).
"""
import argparse
import json
from math import comb

from relayout.summarize import load, wilson


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(pairs, names):
    n = ka = kb = only_a = only_b = 0
    for a, b in pairs:
        n += 1
        ka += a
        kb += b
        only_a += a and not b
        only_b += b and not a
    out = {"n": n}
    for name, k in zip(names, (ka, kb)):
        lo, hi = wilson(k, n)
        out[name] = f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]" if n else None
    out.update({f"{names[0]}_only": only_a, f"{names[1]}_only": only_b,
                "p_mcnemar": float(f"{mcnemar_exact(only_a, only_b):.3g}")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--names", nargs=2, default=["a", "b"])
    args = ap.parse_args()
    if len(args.a) != len(args.b):
        ap.error("--a and --b need the same number of dirs (one per seed)")
    pooled = []
    for k, (da, db) in enumerate(zip(args.a, args.b)):
        ea, eb = load(da), load(db)
        ids = sorted(ea.keys() & eb.keys())
        pairs = [(ea[i]["success"], eb[i]["success"], i) for i in ids]
        pooled += pairs
        print(json.dumps({"seed": k + 1, **compare([p[:2] for p in pairs], args.names)}, ensure_ascii=False))
    print(json.dumps({"seed": "all", **compare([p[:2] for p in pooled], args.names)}, ensure_ascii=False))
    for r in sorted({p[2].split("a")[0] for p in pooled if p[2].startswith("r")}, key=lambda s: int(s[1:])):
        sub = [p[:2] for p in pooled if p[2].split("a")[0] == r]
        print(json.dumps({"seed": "all", "radius": r, **compare(sub, args.names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
