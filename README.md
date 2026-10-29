# relayout-adapt

LIBERO の片付けタスク（赤いマグを左の皿に置く）で、物体の配置がずれた現場に方策を適応させる実験コード。
Physical AI 応用編1 最終課題（pai_final）。

1. 公式デモ 50 本で ACT を学習する（初期方策）
2. 配置をずらした現場で評価する
3. 既存デモを現場の配置に合わせて変換し、物理再生で成功したものだけを追加学習に使う

## 環境

Python 3.12、GPU 1 枚（描画は EGL）。

```bash
uv venv -p 3.12 .venv
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install -p .venv/bin/python "lerobot[libero,training]==0.6.1"
bash scripts/fetch_source.sh            # 公式デモ（LIBERO-90、722MB）を data/source/ へ
```

- `CMAKE_POLICY_VERSION_MINIMUM=3.5` は、依存の egl-probe が CMake 4 でビルドできない問題の回避。
- `import relayout` の時点で LIBERO の設定をリポジトリ内の `.libero/` に固定する。設定ファイルが無いと LIBERO は標準入力で質問して止まり、既存の `~/.libero` があると別の LIBERO を読むため。
- LIBERO の素材（約 400MB）は初回に `~/.cache/libero/assets` へ自動で取得される。

## 使い方

```bash
# 公式デモを再生し直して LeRobotDataset にする（再生で目標に届かないデモは除外）
python -m relayout.convert_demos --hdf5 data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5 \
    --out data/lerobot/redmug_std --demos 0:50

# 評価用の初期状態を固定する（公式 50 + seed 固定で 150）
python -m relayout.init_states --out data/init/std_200.npz --official 50 --seeded 150 --seed 1000

# ACT を学習する
lerobot-train --dataset.repo_id=local/redmug_std --dataset.root=data/lerobot/redmug_std \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=50 --policy.n_action_steps=10 \
    --output_dir=outputs/act_std --seed=1 --steps=30000 --batch_size=32 \
    --env_eval_freq=0 --wandb.enable=false

# 評価する（--shard k/n で分割して並列に回せる）
python -m relayout.rollout --policy outputs/act_std/checkpoints/last/pretrained_model \
    --init data/init/std_200.npz --out outputs/eval_std --videos 3
python -m relayout.summarize outputs/eval_std
```

画像はシミュレータが出す向きのまま（上下左右が反転した状態）で学習・評価に使い、人が見る動画だけ向きを直す。

## 構成

| ファイル | 内容 |
|---|---|
| `relayout/sim.py` | 環境の生成、観測の変換、初期状態へのリセット |
| `relayout/convert_demos.py` | 公式デモの再生と LeRobotDataset への変換 |
| `relayout/dataset.py` | LeRobotDataset への書き出し |
| `relayout/init_states.py` | 評価用の初期状態の作成 |
| `relayout/rollout.py` | 方策の評価（試行ごとの結果を jsonl に記録） |
| `relayout/summarize.py` | 成功率と 95% 信頼区間（Wilson） |

## ABCI

`scripts/abci/` に ABCI 3.0（PBS）用の環境構築とジョブスクリプトがある。ジョブはすべて GPU ノードで EGL 描画。
