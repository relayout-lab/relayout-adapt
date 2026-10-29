#!/bin/bash
# Run the experiment stage by stage on one machine.
#   bash scripts/run.sh <stage>     stages: prepare pi0 generate finetune eval alloc (or all)
# Settings (environment variables): SEEDS (default "1 2 3"), WORKERS (parallel evaluation, default 8),
# AGENT (allocation agent command, default "claude -p"; '' to skip), QUICK=1 (tiny sizes, to check that it runs).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5
SEEDS="${SEEDS:-1 2 3}"
WORKERS="${WORKERS:-8}"
AGENT="${AGENT-claude -p}"
if [[ "${QUICK:-0}" == 1 ]]; then
  DEMOS=0:4 STD="--official 2 --seeded 2" GRID="--radii 0.08 --angles 2 --per 2" NGEN=4 SKIP=2
  PI0_STEPS=200 FT_STEPS=100 BATCH=8 MAX_STEPS=40 BUDGET=4 UNIT=1 SEEDS="${SEEDS%% *}" WORKERS=2
else
  DEMOS=0:50 STD="--official 50 --seeded 150" GRID="" NGEN=200 SKIP=20
  PI0_STEPS=10000 FT_STEPS=5000 BATCH=32 MAX_STEPS=300 BUDGET=48 UNIT=10
fi
PI0=outputs/pi0/checkpoints/last/pretrained_model
COMMON=(--policy.device=cuda --policy.push_to_hub=false --batch_size="$BATCH" --env_eval_freq=0 --wandb.enable=false)

finetune() {  # finetune <dataset> <seed>: from pi0, same steps for every dataset
  [[ -d outputs/ft_$1_s$2 ]] && return 0
  lerobot-train --policy.path="$PI0" --dataset.repo_id="local/redmug_$1" --dataset.root="data/lerobot/redmug_$1" \
    --output_dir="outputs/ft_$1_s$2" --seed="$2" --steps="$FT_STEPS" "${COMMON[@]}"
}

evaluate() {  # evaluate <policy dir> <init npz> <out dir>: parallel shards, then the success rate
  for k in $(seq 0 $((WORKERS - 1))); do
    python -m relayout.rollout --policy "$1" --init "$2" --out "$3" --shard "$k/$WORKERS" --max-steps "$MAX_STEPS" \
      > /dev/null 2>&1 &
  done
  wait
  python -m relayout.summarize "$3" | cut -c1-120
}

dirs() {  # dirs <name> <tag>: one eval dir per seed
  for s in $SEEDS; do echo "outputs/eval/$1_s${s}_$2"; done
}

case "${1:?stage}" in
prepare)  # demos -> dataset, evaluation layouts, generation layouts
  python -m relayout.convert_demos --hdf5 "$SRC" --out data/lerobot/redmug_std --demos "$DEMOS"
  python -m relayout.init_states --out data/init/std_200.npz $STD --seed 1000
  python -m relayout.shift --base data/init/std_200.npz --out data/init/shift_grid.npz $GRID
  python -m relayout.shift --base data/init/std_200.npz --out data/init/shift_dev.npz $GRID --skip "$SKIP"
  python -m relayout.init_states --out data/init/gen_base_200.npz --official 0 --seeded "$NGEN" --seed 5000
  python -m relayout.shift --base data/init/gen_base_200.npz --out data/init/gen_shift_400.npz \
    --random $((2 * NGEN)) --rmax 0.12 --seed 7 ;;
pi0)  # initial policy
  lerobot-train --dataset.repo_id=local/redmug_std --dataset.root=data/lerobot/redmug_std --policy.type=act \
    --policy.chunk_size=50 --policy.n_action_steps=10 --output_dir=outputs/pi0 --seed=3 --steps="$PI0_STEPS" "${COMMON[@]}" ;;
generate)  # convert the demos to shifted layouts; normalise with pi0's statistics
  python -m relayout.generate --hdf5 "$SRC" --demos "$DEMOS" --source-cache "data/source/sources_${DEMOS/:/_}.npz" \
    --init data/init/gen_shift_400.npz --out data/lerobot/redmug_gen
  python -m relayout.dataset --use-stats-from data/lerobot/redmug_std data/lerobot/redmug_gen ;;
finetune)  # (2) source demos and (3) converted demos
  for s in $SEEDS; do finetune std "$s"; finetune gen "$s"; done ;;
eval)  # every policy on the shifted grid and the standard layout, then (3) vs (2)
  for s in $SEEDS; do
    for p in pi0:$PI0 std:outputs/ft_std_s$s/checkpoints/last/pretrained_model gen:outputs/ft_gen_s$s/checkpoints/last/pretrained_model; do
      evaluate "${p#*:}" data/init/shift_grid.npz "outputs/eval/${p%%:*}_s${s}_shift"
      evaluate "${p#*:}" data/init/std_200.npz "outputs/eval/${p%%:*}_s${s}_std"
    done
  done
  python -m relayout.compare --a $(dirs std shift) --b $(dirs gen shift) --names std gen ;;
alloc)  # challenge: split a second generation budget three ways, fine-tune and compare
  for s in $SEEDS; do evaluate "outputs/ft_gen_s$s/checkpoints/last/pretrained_model" data/init/shift_dev.npz "outputs/eval/gen_s${s}_dev"; done
  python -m relayout.allocate decide --dev $(dirs gen dev) --gen-init data/init/gen_shift_400.npz \
    --gen-report data/lerobot/redmug_gen/generate_report.jsonl --budget "$BUDGET" --unit "$UNIT" \
    --agent-cmd "$AGENT" --out data/alloc/allocations.json
  python -m relayout.allocate targets --alloc data/alloc/allocations.json --base data/init/gen_base_200.npz --out data/alloc/targets
  PYTHON=python bash scripts/gen_pool.sh data/alloc/allocations.json data/alloc/targets data/lerobot/pool "$WORKERS"
  python -m relayout.allocate build --alloc data/alloc/allocations.json --pool data/lerobot/pool \
    --round1 data/lerobot/redmug_gen --stats-from data/lerobot/redmug_std --out data/lerobot
  strategies=$(python -c "import json; print(*json.load(open('data/alloc/allocations.json'))['allocations'])")
  for st in $strategies; do
    for s in $SEEDS; do
      finetune "$st" "$s"
      evaluate "outputs/ft_${st}_s$s/checkpoints/last/pretrained_model" data/init/shift_grid.npz "outputs/eval/${st}_s${s}_shift"
    done
    python -m relayout.compare --a $(dirs gen shift) --b $(dirs "$st" shift) --names gen "$st" | grep '"all"' | head -1
  done ;;
all)
  for st in prepare pi0 generate finetune eval alloc; do bash "$0" "$st"; done ;;
*)
  echo "unknown stage: $1" >&2; exit 1 ;;
esac
