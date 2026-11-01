"""Summarise gen_probe.py and policy_elbow.py: robot-side bins against far bins.

"At the limit" means the elbow (joint 4) came within 0.01 rad of its lower limit before the grasp.

    python -m analysis.elbow_summary
"""
import argparse
import json

import numpy as np

ROBOT = ["r12a180", "r12a225", "r8a180"]  # the mug and plate move toward the robot base
FAR = {"gen": ["r4a0", "r4a90", "r8a0", "r12a90"], "policy": ["r4a0", "r8a0", "r12a90"]}


def at_limit(r):
    return r["limit_joint_pre"] == 4 and r["limit_side_pre"] == "lo" and r["joint_margin_pre_min"] < 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="outputs/analysis/gen_probe.jsonl")
    ap.add_argument("--policy", default="outputs/analysis/policy_elbow.jsonl")
    args = ap.parse_args()
    g = [json.loads(line) for line in open(args.gen)]
    print(f"replayed {len(g)}, success reproduced {sum(r['success_probe'] == (r['reason'] != 'no_success') for r in g)}")
    for name, bins in (("robot side", ROBOT), ("far", FAR["gen"])):
        fail = [r for r in g if r["bin"] in bins and r["reason"] == "no_success"]
        acc = [r for r in g if r["bin"] in bins and r["accepted"]]
        both = fail + acc
        print(f"generation, {name}: at limit  replay failed {sum(map(at_limit, fail))}/{len(fail)}"
              f"  accepted {sum(map(at_limit, acc))}/{len(acc)}")
        print(f"  hand error at grasp (median cm)  failed {np.median([r['err_at_grasp_cm'] for r in fail]) if fail else '-'}"
              f"  accepted {np.median([r['err_at_grasp_cm'] for r in acc]) if acc else '-'}")
        if any(map(at_limit, fail)):
            print(f"  failed and at limit: steps at limit before grasp (median) "
                  f"{np.median([r['steps_at_limit_pre'] for r in fail if at_limit(r)])}"
                  f" of {np.median([r['grasp_step'] for r in fail if at_limit(r)])} steps to the grasp")
        print(f"  smallest singular value (median) {np.median([r['sigma_min_pre_min'] for r in both]):.2f}"
              f"  before the grasp: arm contacts {sum(r['arm_touch_pre'] for r in both)}"
              f"  fingers on the table {sum(r['finger_table'] for r in both)}")
    p = [json.loads(line) for line in open(args.policy)]
    for name, bins in (("robot side", ROBOT), ("far", FAR["policy"])):
        rs = [r for r in p if r["bin"] in bins]
        nl = [r for r in rs if not r["success"] and not r["lifted"]]
        print(f"policy, {name}: at limit {sum(r['elbow_margin_min'] < 0.01 for r in rs)}/{len(rs)}"
              f"  failed before lifting {len(nl)} (at limit {sum(r['elbow_margin_min'] < 0.01 for r in nl)})")


if __name__ == "__main__":
    main()
