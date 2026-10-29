#!/bin/bash
# Generate the second-round pool: one small dataset per generation unit, <pool>/<bin>/u<k>.
# Every strategy takes the first k units of a bin, so strategies share the same attempts.
#   bash scripts/gen_pool.sh data/alloc/allocations.json data/alloc/targets data/lerobot/pool [jobs]
set -euo pipefail
ALLOC="$1" TARGETS="$2" POOL="$3" JOBS="${4:-8}"
HDF5=data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5
PY="${PYTHON:-python}"
"$PY" - "$ALLOC" <<'PY' > "$POOL.units"
import json, sys
al = json.load(open(sys.argv[1]))
for b in al["stats"]:
    for k in range(max(a[b] for a in al["allocations"].values())):
        print(b, k, al["unit"])
PY
mkdir -p "$POOL"
run() {
  b=$1 k=$2 u=$3
  [[ -f "$POOL/$b/u$k/generate_report.jsonl" ]] && return 0
  mkdir -p "$POOL/$b"
  "$PY" -m relayout.generate --hdf5 "$HDF5" --init "$TARGETS/$b.npz" --range "$((k * u)):$(((k + 1) * u))" \
    --out "$POOL/$b/u$k" --repo-id "local/${b}_u$k" --seed "$((1000 + k))" --overwrite > "$POOL/$b/u$k.log" 2>&1
  grep "^accepted" "$POOL/$b/u$k.log" | sed "s|^|$b u$k |"
}
export -f run; export TARGETS POOL HDF5 PY
xargs -P "$JOBS" -L 1 bash -c 'run "$@"' _ < "$POOL.units"
