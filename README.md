# relayout-adapt

物の配置がずれた現場に、模倣学習の方策を**デモの変換**で適応させる（LIBERO・ACT）。
赤いマグを皿に置くタスクで、机の上の物をまとめて数 cm ずらすと、標準配置で学習した方策はほぼ失敗する。
手元のデモをずれた配置に変換して追加学習すると、成功するようになる。Physical AI 応用編1 最終課題。

## 結果

ずらした配置 480（4・8・12 cm × 8 方向 × 20）と標準配置 200 での成功率。②③ は同じ初期方策 π0 から同じ step 数で追加学習（3 seed 平均）。

| 条件 | ずらした配置 | 12 cm | 標準配置 |
|---|---|---|---|
| ① π0 のまま | 9% | 1% | 73% |
| ② 元のデモ 37 本で追加学習 | 8% | 0% | 69% |
| ③ 変換デモ 358 本で追加学習 | **60%** | **48%** | 78% |

③ は ② より明らかに高い（McNemar p ≈ 10⁻¹⁸⁹）。② は ① と差がなく、効いたのは学習量ではなくデモの中身。

**挑戦（生成予算の配分）**：追加で 480 回の生成を、均等・失敗率に比例・LLM エージェントの 3 通りで 24 区間に配分した。
ずらした配置で ③ 60% → 均等 **71%**・エージェント 67%・失敗率比例 63%（各 6 本の学習）。
エージェントは苦手な区間に集中させたが、そこは生成の採用率が低く、全体では均等を超えなかった。

## 方法

1. 公式デモ（再生で成功する 37 本）で ACT を学習し、初期方策 π0 とする
2. 机の上の物をまとめて平行移動した配置を作る（移動ロボットが数 cm ずれて止まった場面にあたる）
3. 元のデモを「掴むまで」と「運んで置くまで」に分け、それぞれマグ・皿のずれだけ手先の軌跡を動かして再生する。成功し、皿の中心から 3 cm 以内に置け、周りを動かさないものだけ採用する（MimicGen の簡略版、採用率 90%）
4. π0 から追加学習し、固定した配置で評価する

## 動かし方

Python 3.12、GPU 1 枚（描画は EGL）。

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv pip install "lerobot[libero,training]==0.6.1"
bash scripts/fetch_source.sh        # LIBERO-90 の公式デモ（722MB）

QUICK=1 bash scripts/run.sh all     # 小さい規模で一通り動くか確かめる

bash scripts/run.sh prepare         # デモの変換、評価用・生成用の配置
bash scripts/run.sh pi0             # 初期方策 π0
bash scripts/run.sh generate        # ずらした配置へのデモ変換
bash scripts/run.sh finetune        # ②③ の追加学習（seed 1〜3）
bash scripts/run.sh eval            # 評価と ③ 対 ② の比較
bash scripts/run.sh alloc           # 挑戦：生成予算の配分の比較（エージェントは claude -p）
```

結果は `outputs/eval/` に、試行ごとの記録（jsonl）と成功率（summary.json）で残る。
`SEEDS`・`WORKERS`（評価の並列数）・`AGENT`（配分を決めるコマンド）は環境変数で変えられる。

## 構成

| ファイル | 内容 |
|---|---|
| `relayout/shift.py` | 物をまとめてずらした初期状態 |
| `relayout/generate.py` | デモの変換と採用判定 |
| `relayout/rollout.py` / `summarize.py` / `compare.py` | 評価、成功率（Wilson 区間）、対比較（McNemar） |
| `relayout/allocate.py` | 生成予算の配分（均等・失敗率比例・エージェント） |
| `relayout/record.py` | 動画用の録画 |
| `scripts/run.sh` | 実験を段階ごとに回す |
| `scripts/abci/` | 計算機クラスタ（PBS）用のジョブ。実験の大半はこれで回した |

## 使ったもの・ライセンス

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)（シミュレータと公式デモ）、[LeRobot](https://github.com/huggingface/lerobot)（ACT）、MimicGen（Mandlekar et al., CoRL 2023）の考え方。
このリポジトリは MIT（`LICENSE`）。LIBERO と LeRobot はそれぞれのライセンスに従う。
