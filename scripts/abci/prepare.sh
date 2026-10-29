#!/bin/bash
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
# Replay the official demos into a LeRobotDataset and fix the evaluation initial states.
set -euo pipefail
source "$PAI_ROOT/relayout-adapt/scripts/abci/job_env.sh"
SRC="$PAI_ROOT/data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5"
python -m relayout.convert_demos --hdf5 "$SRC" --out "$PAI_ROOT/data/lerobot/redmug_std" --demos 0:50 --overwrite
python -m relayout.init_states --out "$PAI_ROOT/data/init/std_200.npz" --official 50 --seeded 150 --seed 1000
