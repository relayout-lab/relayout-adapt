#!/bin/bash
# Download the official LIBERO-90 demonstrations for the target task.
#   bash scripts/fetch_source.sh [out_dir]
set -euo pipefail
OUT="${1:-data/source}"
FILE=LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5
SHA=6d73840ba02f8f98c08668d948d1c731c660c6dd4fb0d6fc3f4a8a3bfa44f88c
mkdir -p "$OUT"
if [[ ! -f "$OUT/$FILE" ]]; then
  curl -fSL -o "$OUT/$FILE.part" \
    "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_90/$FILE"
  mv "$OUT/$FILE.part" "$OUT/$FILE"
fi
echo "$SHA  $OUT/$FILE" | sha256sum -c -
