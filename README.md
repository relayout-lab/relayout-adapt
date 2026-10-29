# relayout-adapt

物の配置がずれた現場に、模倣学習の方策を**デモの変換**で適応させる実験コード。
LIBERO の片付けタスク（赤いマグを左の皿に置く）で、机の上の物をまとめて数 cm ずらすと、標準の配置で学習した ACT 方策はほとんど失敗する。
手元のデモを新しい配置に変換して（MimicGen の簡略版）追加学習すると、ずれた配置でも成功するようになる。

Physical AI 応用編1 最終課題。

## 結果

固定した評価用の配置（ずらし量 4・8・12 cm × 8 方向 × 20 = 480 配置）と標準配置（200 配置）での成功率。
②・③ は同じ初期方策 π0 から同じ 5000 step だけ追加学習し、seed 1〜3 の平均。

| 条件 | 追加学習のデータ | ずらした配置 | 4 cm | 8 cm | 12 cm | 標準配置 |
|---|---|---|---|---|---|---|
| ① π0 のまま | — | 9% | 24% | 1% | 1% | 73% |
| ② 元のデモで追加学習 | 標準配置のデモ 37 本 | 8% | 23% | 0% | 0% | 69% |
| ③ 変換デモで追加学習 | ずらした配置へ変換したデモ 358 本 | **60%** | 72% | 59% | 48% | 78% |

- ③ と ② を同じ配置どうしで比べると、③ だけが成功したのが 775 試行、② だけが 29 試行（McNemar 検定 p ≈ 10⁻¹⁸⁹）
- ② は ① と差がない（p = 0.41）。効いたのは学習量ではなく、ずらした配置のデモという中身
- 標準配置でも ③ は落ちない（78%、② は 69%）

**生成予算の配分（挑戦）**：ずらし方を 24 区間に分け、追加で 480 回の生成をどこに使うかを、均等・失敗率に比例・LLM エージェントの 3 通りで決めて比べた（各 6 本の学習）。

| 配分 | 追加で作れたデモ | ずらした配置 | 標準配置 |
|---|---|---|---|
| ③（追加なし） | — | 60% | 78% |
| 均等 | 399 本 | **71%** | 90% |
| 失敗率に比例 | 372 本 | 63% | 80% |
| エージェント | 316 本 | 67% | 79% |

追加の生成は効く（③ 対 均等 p = 0.0003、学習 1 本を 1 標本とした Welch の t 検定）。
エージェントは苦手な区間（12 cm のロボット側）に予算を集中させ、その区間では均等と並んだが、ほかの区間で下がり、全体では均等を超えなかった（p = 0.19）。
集中先は生成の採用率が 28% と低く、作れたデモが最も少なかった。

## 方法

1. **初期方策 π0**：公式デモ 50 本のうち、再生して目標に届く 37 本で ACT を学習する（1 万 step）
2. **ずらした配置**：机の上の物（マグ・皿など 5 個）を同じだけ平行移動した初期状態を作る（ロボットとカメラは動かさない）。移動ロボットが数 cm ずれて止まった場面にあたる
3. **デモの変換**：元のデモを再生して手先の軌跡を記録し、最初にグリッパを閉じる所で 2 つに分ける。掴むまではマグのずれ、運ぶ所からは皿のずれだけ軌跡を平行移動し（なめらかに切り替える）、元の動作に位置と向きの補正を足して追従させる。シミュレータで成功し、皿の中心から 3 cm 以内に置け、周りの物を 2 cm 以上動かさなかったものだけ採用する（0〜12 cm のずれで採用率 90%）
4. **追加学習と評価**：π0 から同じ step 数で追加学習し、固定した配置で評価する。正規化の統計量は π0 のものにそろえる
5. **配分（挑戦）**：区間ごとの成功率・失敗の形・生成の採用率をエージェント（`claude -p`）に渡し、配分を JSON で答えさせる。エージェントに見せる評価は、最終評価と配置が重ならない開発用セットを使う

## 環境

Python 3.12、GPU 1 枚（描画は EGL）。

```bash
uv venv -p 3.12 .venv
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install -p .venv/bin/python "lerobot[libero,training]==0.6.1"
source .venv/bin/activate
bash scripts/fetch_source.sh            # 公式デモ（LIBERO-90、722MB）を data/source/ へ
```

- `CMAKE_POLICY_VERSION_MINIMUM=3.5` は、依存の egl-probe が CMake 4 でビルドできない問題の回避。
- `import relayout` の時点で LIBERO の設定をリポジトリ内の `.libero/` に固定する。設定ファイルが無いと LIBERO は標準入力で質問して止まり、既存の `~/.libero` があると別の LIBERO を読むため。
- LIBERO の素材（約 400MB）は初回に `~/.cache/libero/assets` へ自動で取得される。

## 再現手順

評価・生成は `--shard k/n` や並列実行で分けられる（下の例は 1 本で回す形）。時間の目安は GPU 1 枚で、学習 1 万 step が 10 分前後、評価 480 配置が並列 8 本で 10〜20 分。

```bash
SRC=data/source/LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_demo.hdf5

# 1. 公式デモを LeRobotDataset にし、評価用の初期状態を固定する
python -m relayout.convert_demos --hdf5 $SRC --out data/lerobot/redmug_std --demos 0:50
python -m relayout.init_states --out data/init/std_200.npz --official 50 --seeded 150 --seed 1000

# 2. 初期方策 π0
lerobot-train --dataset.repo_id=local/redmug_std --dataset.root=data/lerobot/redmug_std \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=50 --policy.n_action_steps=10 \
    --output_dir=outputs/pi0 --seed=3 --steps=10000 --batch_size=32 \
    --env_eval_freq=0 --wandb.enable=false
PI0=outputs/pi0/checkpoints/last/pretrained_model

# 3. ずらした配置（評価用 480）と、生成用の配置（評価と土台が重ならない 200 件を 0〜12 cm ずらす）
python -m relayout.shift --base data/init/std_200.npz --out data/init/shift_grid.npz
python -m relayout.init_states --out data/init/gen_base_200.npz --official 0 --seeded 200 --seed 5000
python -m relayout.shift --base data/init/gen_base_200.npz --out data/init/gen_shift_400.npz --random 400 --rmax 0.12 --seed 7

# 4. デモの変換（採用されたものだけ LeRobotDataset へ）。統計量は π0 の学習データにそろえる
python -m relayout.generate --hdf5 $SRC --init data/init/gen_shift_400.npz --out data/lerobot/redmug_gen
python -m relayout.dataset --use-stats-from data/lerobot/redmug_std data/lerobot/redmug_gen

# 5. ② と ③ を π0 から追加学習する（seed 1〜3）
for ds in std gen; do
  lerobot-train --policy.path=$PI0 --policy.device=cuda --policy.push_to_hub=false \
      --dataset.repo_id=local/redmug_$ds --dataset.root=data/lerobot/redmug_$ds \
      --output_dir=outputs/ft_${ds}_s1 --seed=1 --steps=5000 --batch_size=32 \
      --env_eval_freq=0 --wandb.enable=false
done

# 6. 評価と比較
python -m relayout.rollout --policy outputs/ft_gen_s1/checkpoints/last/pretrained_model \
    --init data/init/shift_grid.npz --out outputs/eval/gen_s1_shift
python -m relayout.summarize outputs/eval/gen_s1_shift
python -m relayout.compare --a outputs/eval/std_s1_shift --b outputs/eval/gen_s1_shift --names std gen
```

配分（挑戦）：

```bash
# 開発用の評価（最終評価と土台が重ならない配置）で ③ を評価してから、3 通りの配分を決める
python -m relayout.shift --base data/init/std_200.npz --out data/init/shift_dev.npz --skip 20
python -m relayout.rollout --policy outputs/ft_gen_s1/checkpoints/last/pretrained_model \
    --init data/init/shift_dev.npz --out outputs/eval/gen_s1_dev
python -m relayout.allocate decide --dev outputs/eval/gen_s1_dev \
    --gen-init data/init/gen_shift_400.npz --gen-report data/lerobot/redmug_gen/generate_report.jsonl \
    --budget 48 --agent-cmd "claude -p" --out data/alloc/allocations.json
# 区間ごとに生成し、配分ごとのデータセットを作る（あとは 5・6 と同じ）
python -m relayout.allocate targets --alloc data/alloc/allocations.json --base data/init/gen_base_200.npz --out data/alloc/targets
bash scripts/gen_pool.sh data/alloc/allocations.json data/alloc/targets data/lerobot/pool
python -m relayout.allocate build --alloc data/alloc/allocations.json --pool data/lerobot/pool \
    --round1 data/lerobot/redmug_gen --stats-from data/lerobot/redmug_std --out data/lerobot
```

`--agent-cmd` は標準入力でプロンプトを受け取り、答えを標準出力に返すコマンドなら何でもよい（`''` でエージェントを使わない）。
プロンプトと答えは `data/alloc/` に残る。

動画用の録画（方策を動かして広い視点で描く。`--source K` で変換デモを撮る）：

```bash
python -m relayout.record --policy outputs/ft_gen_s1/checkpoints/last/pretrained_model \
    --init data/init/shift_grid.npz --ids r12a90_official_1 --out outputs/video/gen
```

画像はシミュレータが出す向きのまま（上下左右が反転した状態）で学習・評価に使い、人が見る動画だけ向きを直す。

## 構成

| ファイル | 内容 |
|---|---|
| `relayout/sim.py` | 環境の生成、観測の変換、初期状態へのリセット |
| `relayout/convert_demos.py` | 公式デモの再生と LeRobotDataset への変換 |
| `relayout/dataset.py` | LeRobotDataset への書き出し、正規化の統計量の差し替え |
| `relayout/init_states.py` | 評価用の初期状態の作成 |
| `relayout/shift.py` | 物をまとめてずらした初期状態（格子・ランダム） |
| `relayout/generate.py` | デモの変換と採用判定 |
| `relayout/rollout.py` | 方策の評価（試行ごとの結果を jsonl に記録） |
| `relayout/summarize.py` | 成功率と 95% 信頼区間（Wilson） |
| `relayout/compare.py` | 同じ配置どうしの対比較（McNemar の正確検定） |
| `relayout/allocate.py` | 生成予算の区間ごとの配分（均等・失敗率比例・エージェント）とデータセットの組み立て |
| `relayout/record.py` | 動画用の録画 |
| `scripts/gen_pool.sh` | 区間ごとの生成を並列に回す |
| `scripts/abci/` | 計算機クラスタ（ABCI 3.0、PBS）用のジョブスクリプト。実験の大半はこれで回した |

## 使ったもの

- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（シミュレータと LIBERO-90 の公式デモ）
- [LeRobot](https://github.com/huggingface/lerobot)（ACT の学習）
- 変換の考え方は MimicGen（Mandlekar et al., CoRL 2023）を簡略にしたもの（物を基準にした区間の変換と、成功したものだけの採用）
