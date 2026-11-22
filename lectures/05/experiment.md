# 第5回 演習・宿題の実験手順（GitHub Codespaces）

演習と宿題（基礎・実装・発展）の比較の実験を、GitHub Codespaces で実行する手順です。実験の設計・仮説・結果は [answer.md](answer.md) に、質問・条件・自動チェックのデータは [conditions.toml](conditions.toml) に書きます。実行には比較用スクリプト [compare.py](compare.py) を使い、実装と発展の取得処理は [app/retrieval_lec05.py](../../app/retrieval_lec05.py) にあります。手順の中の「知らせる」は、Claude に伝えることです。

## 使うファイル

| もの | 場所 | 説明 |
|---|---|---|
| 比較用スクリプト | `lectures/05/compare.py` | 条件と質問の組を実行し、結果を保存して集計する。サブコマンドは `status`・`variants`・`run`・`summarize` |
| 条件のデータ | `lectures/05/conditions.toml` | 質問（q1〜q3・qa1・qa2）、条件（F・F214・V0・T・G・B1・I1・I2・A1・A2）、自動チェック |
| 取得処理 | `app/retrieval_lec05.py` | 実装（前後ページ）と発展（用語から始める Graph 検索）。upstream の `app/retrieval.py` は書き換えず、足したもの |
| エージェント | `app/agent.py`・`app/tools.py`・`app/retrieval.py` | この回では書き換えない。ランナーが、条件ごとに指示・ツールの説明文・取得処理を一時的に差し替えて実行する（`agent.override`） |
| テスト | `lectures/05/test_lec05.py`・`lectures/05/test_compare.py` | 取得処理（前後ページ・用語から始める Graph 検索）と、比較用スクリプトのテスト。API キーを使わない |
| 結果の置き場所 | `lectures/05/results/` | 1 回の実行ごとに JSON とトレースを置く（下の「結果の置き場所」） |
| 集計 | `lectures/05/results/summary.md` | `summarize` で作る比較の表（`summary.json` も同じ内容） |

## 条件の一覧と質問

### 条件

全部で 10 条件、質問ごとに 3 試行ずつ流して、合計 132 試行です。

| label | 名前 | 段 | 中身 | 質問 |
|---|---|---|---|---|
| F | `full` | 演習 | 全文投入（438 ページ） | q1〜q3 |
| F214 | `full_aligned` | 演習（追加） | 全文投入を 214 ページにそろえる | q1〜q3 |
| V0 | `vector` | 演習・基準 | 基本RAG（`search_materials` だけ。上位 5 件） | 全 5 問 |
| T | `tree` | 演習 | `--tree` と同じ（`browse_tree`） | q1〜q3 |
| G | `graph` | 演習・発展の基準 | `--graph` と同じ（`search_graph`） | 全 5 問 |
| B1 | `doc_hint` | 基礎 | `search_materials` の説明文だけを変える | 全 5 問 |
| I1 | `neighbors` | 実装 | 前後 1 ページを足す（名前・説明文・指示文は V0 と同じ） | 全 5 問 |
| I2 | `topk15` | 実装（対照） | `RAG_TOP_K` を 15 にする（前後なし。I1 の最大 15 ページと量をそろえる） | 全 5 問 |
| A1 | `term_graph` | 発展 | 用語から始める Graph 検索（章の上限 2） | 全 5 問 |
| A2 | `term_graph_cap3` | 発展（感度） | A1 の章の上限を 3 にする | 全 5 問 |

- 対象資料は、F だけ全 9 資料 438 ページ。F214 とほかの条件は、第1〜3回の本講義 214 ページ。
- V0 のツールは `search_materials` の 1 つだけ（Tree・Graph もツール 1 つ）。CLI の通常モードは、第3回の `make_quiz` と承認付きの `save_learning_log` も持つので、ツールが 3 つになる。
- 条件の違いは `variants` で確かめる（「最初の確認」）。

### 質問

| qid | 内容 | 流す条件 |
|---|---|---|
| q1 | 第2回の LoRA と QLoRA（第2回 p.52 の 1 ページで答えがそろう） | 全 10 条件 |
| q2 | 第3回のコンパクションとメモリの違い（複数ページ） | 全 10 条件 |
| q3 | 今回作った RAG の実測値（資料にない質問） | 全 10 条件 |
| qa1 | ファインチューニングと構造化出力の使い分け（第2回と第3回をまたぐ） | V0・G・B1・I1・I2・A1・A2 |
| qa2 | エージェントのメモリの危険と防ぎ方（第3回の連続したページ） | V0・G・B1・I1・I2・A1・A2 |

- F・F214・T は q1〜q3 だけ。
- 質問文・gold・自動チェックは、answer.md の「質問セット」と conditions.toml にある。

## 結果の置き場所

```text
lectures/05/results/
├── <model>/<variant>/<qid>/t<k>.json        1 回分の記録（回答、ツール呼び出し、usage、条件の sha256 など）
├── <model>/<variant>/<qid>/t<k>.trace.txt   CLI の --trace と同じ表示に、取得の内訳などを足したもの
├── variants/<variant>.json・.txt            条件ごとに、モデルに見える指示・ツールの説明文・取得処理・設定
├── cli/<名前>.txt                           演習の CLI と、新しい取得の検索だけの出力
└── summary.md・summary.json                 集計
```

- `<model>` は `gemini-3.5-flash-lite`。`--dry-run` の結果は、repo の外の一時フォルダの `<model>-dry-run` に入るので、混ざらない。
- `<variant>` は条件の名前（`full`・`vector`・`neighbors` など。上の表の「名前」）。`<qid>` は `q1`〜`q3`・`qa1`・`qa2`、`<k>` は試行の番号（1〜3）。

## 上書きを防ぐために

- **済みの試行は飛ばします。** `run` は、保存先に `t<k>.json` があればその実行を飛ばすので、同じコマンドを何度実行しても上書きされません。止まったときも、同じコマンドで続きから実行できます。
- **結果は、成功したときだけ保存されます。** API のエラーで止まった実行は `t<k>.json` が置かれないので、次に実行したときにやり直されます。全文投入が入力上限を超えた記録（`status=input_limit`）は、結果として保存されます。
- **やり直すときは、その `t<k>.json` と `t<k>.trace.txt` を消してから実行します。** 結果のファイルは、手で直しません（ランナーの記録と食い違うため）。
- **守るパス（`app`・`data/rag_data`・`lectures/05/compare.py`・`lectures/05/conditions.toml`・`pyproject.toml`・`uv.lock`）に変更があると、`run` は止まります。** 結果がどの commit のものか分からなくなるためです。これらのファイルは編集しません。
- **条件が途中で変わると止まります。** 保存した記録には、条件（指示・ツールの説明・取得処理・設定・対象資料）の sha256 が入っていて、いまの条件と違うと止まります。1 回目と 2 回目で条件が違うと、比べられないためです。
- **`AIE_MODEL` と `AIE_RAG_DB` は設定しません。** 結果は既定のモデル（`gemini-3.5-flash-lite`）で取ります。`AIE_MODEL` を設定すると、結果が別のフォルダに入って表が分かれます。`AIE_RAG_DB` を設定すると、同梱と違う検索 DB を使うことになるので、`run` が止まります。
- **`git add -A` は使いません。** 関係ないファイルが入らないよう、`lectures/05/results` だけを add します。
- **仮説は、結果を見る前に commit 済みです。** answer.md の「仮説（実行前に記録）」は、実行したあとで書き換えません。
- **GitHub の「Sync fork → Discard commits」は押しません。** fork の main に積んだコミット（前の回までの提出）が消えます。

## 準備

### 1. Codespaces を lec05 で開く

lec05 ブランチは作成して push してあります。演習の手順 1 にある `git switch -c lec05` は要りません。

- 新しく開く場合：GitHub で自分の fork を開き、ブランチを `lec05` に切り替えてから、「Code → Codespaces → Create codespace on lec05」を押す。
- 既に Codespace がある場合：そのまま開く。

どちらの場合も、開いたあとにターミナルで次を実行します（新しく開いたときは、すでに lec05 にいるので、`git switch` は何も変えません）。

```bash
git fetch origin
git switch lec05
git pull --no-rebase --no-edit
uv sync --frozen
[ -n "$GEMINI_API_KEY" ] && echo 設定済み || echo 未設定
[ -z "$AIE_MODEL$AIE_RAG_DB" ] && echo OK || echo "AIE_MODEL か AIE_RAG_DB が設定されている（unset してから進む）"
```

- `git switch lec05` で止まる場合は、前の回のブランチに変更が残っている。そのブランチで commit してから、もう一度実行する。
- `git pull --no-rebase --no-edit` は、lec05 を最新にするため。
- `uv sync --frozen` は、lec05 では sqlite-vec が増えるため。lec04 と行き来するたびに実行する。実行しないと、`sqlite_vec` の ImportError で止まる。
- API キーの値は表示しない。「未設定」のときは、下の「うまくいかないとき」を見る。
- 最後の行が OK 以外のときは、`unset AIE_MODEL AIE_RAG_DB` を実行してから進む。

### 2. 利用枠を確かめる

- AI Studio の rate limit の画面で、`gemini-3.5-flash-lite` の 1 分あたりのリクエスト数・1 分あたりのトークン数（TPM）・1 日あたりのリクエスト数を確かめる。埋め込み（`gemini-embedding-001`）の上限も、同じ画面で見る。
- TPM が 20 万未満なら、`run` に `--tpm-budget <その値>` を付ける（段 3・4 のコマンドも同じ）。全文投入は、1 回の入力が 10 万 tokens を超えるため。
- TPM が全文投入 1 回の入力より小さいと、全文投入は通らない。そのときは、止めて知らせる。

## 最初の確認（API キー不要）

```bash
uv run pytest -q                                                # 講座のテスト
uv run pytest -q lectures/05                                    # 取得処理とランナーのテスト
uv run pytest -q lectures/03                                    # 第3回のコードと compare.py のテスト
uv run python lectures/05/compare.py variants                   # 条件ごとの差分（どれも 1 か所だけ）
uv run python lectures/05/compare.py run --level all --dry-run  # API を使わない予行。出力は一時フォルダ
uv run python lectures/05/compare.py status                     # 予定・済み・残りの件数と、残りの推定リクエスト数
```

- 通らなければ、止めて知らせる。
- `lectures/05` と `lectures/03` のテストは、別々に実行する。1 つのコマンドにまとめると、別の回の `compare.py` と名前が衝突する。講座の `uv run pytest -q` は `tests/` だけが対象なので、これも別に実行する。
- `variants` は、条件ごとに「モデルに見える指示とツールの説明文、取得処理、設定」を `lectures/05/results/variants/` に保存し、基準との差分を表示する。差分がちょうど 1 か所でないときは、警告が出る。警告が出たら、先に進まずに知らせる。
  - V0 が基準。B1・I1・I2・A1 は V0 との差分、F214 は F との差分、A2 は A1 との差分。
  - T と G は、方式そのものが違うので、基準との差分は出ない。
- `--dry-run` は、偽のモデルと偽の埋め込みで、全部の経路を通す。最後に出力先が表示される。repo の `lectures/05/results/` には書かない。
- `status` は、条件 × 質問ごとの済み・予定・残りを表示する。この時点の合計は `0/132`。

## 実行の順番

最初に、全部の条件を 1 回ずつ（`--trials 1`）流して、すべてのレベルの結果をそろえます。そのあと、2・3 回目をまとめて足します。利用枠が途中で尽きても、段 3 までで、全レベルの 1 回目を提出できます。

**☆ は実験済み**です（結果は `lectures/05/results/` に置き、[answer.md](answer.md) に記入済み）。

| 段 | 状態 | 内容 | コマンド（`uv run python lectures/05/compare.py` の後） | LLM リクエストの目安 | 埋め込み API の目安 | 時間の目安 |
|---:|---|---|---|---:|---:|---:|
| 0 | ☆ 実験済み | 準備と確認（API なし） | 上の「準備」と「最初の確認」 | 0 | 0 | 約 5 分 |
| 1 | ☆ 実験済み | 演習の CLI（problem.md の STEP 1〜4）と、新しい取得の検索だけ | 下の「段 1」 | 約 15 | 約 20 | 約 5 分 |
| 2 | ☆ 実験済み | q1 で形を確かめる | `run --level exercise --question q1 --trials 1`、`summarize` | 約 15 | 約 6 | 約 3 分 |
| 3 | ☆ 実験済み | 全条件の 1 回目 | `run --level all --trials 1` | 約 120 | 約 100 | 20〜30 分 |
| 4 | ☆ 実験済み | 2・3 回目 | `run --level all --trials 3` | 約 260 | 約 210 | 40〜60 分 |
| 5 | ☆ 実験済み | 集計と受け渡し | `summarize`、キーの確認、add・commit・pull・push（下の「結果の保存と受け渡し」） | 0 | 0 | 約 5 分 |

- 実際の実行（2026-11-22）：段 4 の 114 回目で 1 日の上限（PerDay）に達して止まり、そこまでの結果を push した。上限が戻ったあと（日本時間 17 時以降）に、同じコマンドで残りの 18 回を流した。段 1〜4 を 1 日で流すと、gemini-3.5-flash-lite の無料枠の 1 日の上限（AI Studio の表示で 500 リクエスト）を超えた。

- 合計は、LLM 約 410 回、埋め込み約 340 リクエスト、約 1〜1.5 時間（429 で延びる）。
- 見積りの根拠：
  - 1 試行の LLM リクエストは、全文投入が 1、ベクトル系と Graph が約 3、Tree が約 7（目次を開くたびに 1 回増える）。
  - 内訳は、全文投入 18 試行 × 1、ベクトル系と Graph の 105 試行 × 約 3、Tree の 9 試行 × 約 7、段 1 の CLI 約 15。合計は約 410。
  - 埋め込みは、1 回の検索で 2 リクエスト（トークン数の確認と、埋め込み）。1 試行の検索は 1〜2 回なので、1 試行あたり約 3 リクエストで数えた。
  - 時間は、1 試行あたり 20〜40 秒と、試行の間の待ち 8 秒。
  - `status` の「残りの推定リクエスト」（約 335）は、ベクトル系と Graph を 2.5、Tree を 6 とした、生成だけの目安。この表より少なめに出る。
- 段 3 のあとで、途中の結果を受け渡してもよい（下の「結果の保存と受け渡し」と同じ手順）。
- 段 3 の `run --level all --trials 1` は、全 44 組（条件 × 質問）の 1 回目のうち、段 2 で済んだ 5 試行を飛ばして、残りを実行する。段 4 の `--trials 3` は、2・3 回目の 88 試行。
- 順番は、試行 → 質問 → 条件。条件の順は、conditions.toml の `order` を基本にして、試行 k では 3 × (k − 1) 個ずらして巡回させる（1 回目はそのまま、2 回目は 3 つ、3 回目は 6 つ）。時間帯や API の混み具合の影響と、条件の効果を分けるため。全文投入どうし（F と F214）は、巡回しても隣にならない。
- `--level`・`--variant`・`--question` で、実行する条件を絞れる（`--variant` には、名前か label を使う）。1 回に流す件数は `--max-runs`、試行の間の待ちは `--interval`（秒、既定 8）で変えられる。

## 段ごとの手順

### 段 1：演習の CLI と、新しい取得の検索だけ

problem.md の演習の STEP 1〜4 のコマンドを、課題文どおりに CLI で 1 回ずつ実行し、出力を残します。同じターミナルで、上から順に実行してください（`AIE_QUESTION` は、そのターミナルの中だけで有効です）。

```bash
mkdir -p lectures/05/results/cli
AIE_QUESTION='第2回の説明では、LoRAは何を固定し何を学習しますか？ QLoRAは何を加えますか？'
uv run python -m app.agent --full-context --trace "$AIE_QUESTION" 2>&1 | tee lectures/05/results/cli/step1_full_context.txt
uv run python -m app.retrieval "LoRA QLoRA 違い" 2>&1 | tee lectures/05/results/cli/step2_retrieval.txt
uv run python -m app.agent --trace "$AIE_QUESTION" 2>&1 | tee lectures/05/results/cli/step3_vector.txt
uv run python -m app.agent --tree --trace "$AIE_QUESTION" 2>&1 | tee lectures/05/results/cli/step4_tree.txt
uv run python -m app.agent --graph --trace "$AIE_QUESTION" 2>&1 | tee lectures/05/results/cli/step4_graph.txt
```

- `| tee` は、画面に出しながら、ファイルにも書く。`2>&1` は、エラーの表示も一緒に残す。
- STEP 3 で、保存の承認を聞かれたら、N と答える。CLI の通常モードは、第3回の `make_quiz` と承認付きの `save_learning_log` も持つため。保存を頼む質問ではないので、聞かれないのが普通。
- STEP 3 の記録は、ツール 3 つのまま使う。ランナーの V0（ツール 1 つ）より、入力トークンが多くなる。
- STEP 1（全文投入）は、1 回の入力が 10 万 tokens を超える。429 で止まったら、1 分待って再実行する。止まったときの出力は、再実行の前に、別の名前で残す。

  ```bash
  mv lectures/05/results/cli/step1_full_context.txt lectures/05/results/cli/step1_full_context_error.txt
  ```

続いて、新しい取得の検索だけを実行します（埋め込みの API だけを使い、回答は作りません）。

```bash
uv run python -m app.retrieval_lec05 --neighbors "LoRA QLoRA 違い" 2>&1 | tee lectures/05/results/cli/impl_neighbors_lora.txt
QA1='回答を決まったJSON形式で返させたいとき、第2回で学んだファインチューニングと第3回の構造化出力は、どう使い分ければよいですか。構造化出力はどうやって形式を守らせるのですか'
QA2='第3回の講義では、エージェントのメモリにはどんな危険があると説明されていますか。防ぐには何をすればよいですか'
uv run python -m app.retrieval_lec05 --brief "$QA1" 2>&1 | tee lectures/05/results/cli/adv_qa1_vector.txt
uv run python -m app.retrieval_lec05 --term-graph --brief "$QA1" 2>&1 | tee lectures/05/results/cli/adv_qa1_term_graph.txt
uv run python -m app.retrieval_lec05 --brief "$QA2" 2>&1 | tee lectures/05/results/cli/adv_qa2_vector.txt
uv run python -m app.retrieval_lec05 --term-graph --brief "$QA2" 2>&1 | tee lectures/05/results/cli/adv_qa2_term_graph.txt
```

- `--neighbors` は、検索で見つけたページ（hit）と、その前後 1 ページ（neighbor）を表示する。
- `--brief` は、1 ページ 1 行で表示する。`--term-graph` は、選んだ理由（確保・距離・補充）と、章の上限で外したページも表示する。
- この結果を見ても、質問と仮説は変えない。結果を見て都合のよい質問を選ぶのを避けるため。
- 段 1 と段 2 の間は、1 分以上あける。ランナーの TPM ガードは、CLI の全文投入（STEP 1）の分を知らないため。

### 段 2：q1 で形を確かめる

```bash
uv run python lectures/05/compare.py run --level exercise --question q1 --trials 1
uv run python lectures/05/compare.py summarize
```

- `run` の各行の status が `ok` になっているか見る。
- `lectures/05/results/summary.md` の表B・表C が空でないこと、「警告」の節に問題が出ていないことを見る。
- 空や error なら、止めて知らせる。

### 段 3・4：全条件の 1 回目、2・3 回目

```bash
uv run python lectures/05/compare.py run --level all --trials 1
uv run python lectures/05/compare.py run --level all --trials 3
```

- 段 3 が終わってから、段 4 を実行する。TPM が 20 万未満のときは、どちらにも `--tpm-budget <その値>` を付ける。
- 途中で止まったら、同じコマンドで続きから実行する（済みの試行は飛ばされる）。
- 止まった直後に `echo $?` で、終了コードを確かめる。

  | 終了コード | 意味 | 対処 |
  |---|---|---|
  | 0 | 正常に終わった | 次へ進む |
  | 1 | 止まった（設定の誤り・API エラー・未コミットの変更など） | 表示された理由を知らせる |
  | 2 | 1 日の上限（PerDay） | そこまでの結果を push し、日本時間 17 時以降に、同じコマンドを実行する |
  | 130 | Ctrl+C で止めた | 同じコマンドで続きから実行する |

- Ctrl+C で止めてもよい。続きから再開できる。

## 各段で確かめること

段が終わるたびに、次の表の「確かめること」を見ます。「止めて知らせる条件」に当たったら、次の段に進まずに知らせます。`summarize` のあとは、`lectures/05/results/summary.md` の「警告」の節も見ます（status が ok でない試行、再試行した試行、モデルに見えた指示が条件と違う試行、ツールの error を返した試行が並ぶ）。

| 段 | 確かめること | 止めて知らせる条件 |
|---:|---|---|
| 0 | 3 つの `pytest` が通る。`variants` に警告が出ない。`--dry-run` が最後まで通り、「実行 132 回」と出る。`status` の合計が `0/132` | どれかが通らない。`variants` に警告が出る |
| 1 | `lectures/05/results/cli/` に 10 個のファイルができた。STEP 1・3・4 の末尾に `[time]`・`[usage]`・`[cost]` が出た。STEP 3 で `search_materials`、STEP 4 の Tree で `browse_tree`、Graph で `search_graph` の `[tool call]` が出た。`--term-graph` の出力に「拾った用語」と「章の上限（2件）で外したページ」が出た | ファイルが空、またはエラーで終わった（429 は 1 分待って再実行。続くときだけ知らせる）。STEP 3 で `search_materials` が呼ばれなかった |
| 2 | `run` の各行が `ok`。`summary.md` の表B・表C が空でない。「警告」の節に、ok でない試行・指示が条件と違う試行・ツールの error が無い | 表が空。status が ok でない。「警告」の節に、上の項目が出た |
| 3 | `status` の「済み」が `44/132` に増え、最後の status が `ok`。F・F214 は `input_limit`（入力上限を超えた記録）でもよい。終了コード 0 で終わった | `ok`・`input_limit` 以外の status。429 や 503 の再試行が何度も続く |
| 4 | `status` の合計が `132/132`。`summarize` のあと、表A〜D に「未実行」が残らない | 132/132 に届かないまま止まった。「警告」の節に、ツールの error（検索 DB の失敗など）が出た |
| 5 | キーなし。`git status --short` に `lectures/05/results` 以外の行が無い。`git push` が通った | キーあり。results 以外の変更がある。push が拒否された |

## 利用枠

- Gemini API の無料枠には、1 分あたりと 1 日あたりの上限（リクエスト数・トークン数）がある。値は AI Studio の rate limit の画面で確かめる。上限はプロジェクトごとなので、API キーを増やしても増えない。
- 1 日の上限は、太平洋時間の 0 時（日本時間 17 時）に戻る。
- 1 分あたりの上限（429）に当たると、スクリプトが待ってから自動で再試行する（最大 5 回。待つ時間はエラーの本文に従い、無ければ 60 秒）。
- 1 日の上限（エラーの本文に `PerDay`）に当たると、「明日、同じコマンドで続きから」と表示して止まる（終了コード 2）。そこまでの結果を push しておき、日本時間 17 時以降に同じコマンドを実行する。
- 503 などのサーバー側のエラーは、30・60・120 秒待って、最大 3 回再試行する。それでも続くと止まる（終了コード 1）。
- 全文投入は、1 回の入力が 10 万 tokens を超える（F の prompt は約 23 万字、F214 は約 11 万字）。TPM の上限に近いので、スクリプトは、全文投入の前に、直近 60 秒の入力トークンが予算（既定 20 万）を超えないよう待つ。予算は `--tpm-budget` で変える。
- 段 1 の STEP 1（CLI の全文投入）は、このガードの外で動く。そのため、段 1 と段 2 の間は 1 分以上あける。
- 埋め込みは、1 回の検索で 2 リクエスト（トークン数の確認と、埋め込み）。生成とは別の上限に数えられる。
- 実行の間の待ちは `--interval`（秒、既定 8）で、1 回に流す件数は `--max-runs` で変えられる。

## 結果の保存と受け渡し

段 3 のあと（途中）と、段 5（最後）に、次の 2 つを順に実行して push します。

1 つ目は、集計とキーの確認です。まだ commit しません。

```bash
uv run python lectures/05/compare.py summarize
grep -rlEI "AIza[0-9A-Za-z_-]{30,}" lectures/05/results && echo "キーあり：止めて知らせる" || echo キーなし
git status --short
```

- `grep` は、結果に API キー（`AIza…` の形の文字列）が含まれていないかの確認。一致したファイルの名前だけを表示する（キーそのものは画面に出さない）。スクリプトも書き出す前に伏せ字にしているが、念のため確かめる。「キーあり」が出たら、2 つ目に進まずに知らせる。
- `git status --short` に、`lectures/05/results` 以外の行が出ていないかを見る。出ていたら、2 つ目に進まずに知らせる。

2 つ目は、add・commit・push です。1 つ目で「キーなし」と出て、`git status --short` に `lectures/05/results` 以外の行が無いときだけ実行します。

```bash
# キーなしで、git status に lectures/05/results 以外の行が無いときだけ実行する
! grep -rqEI "AIza[0-9A-Za-z_-]{30,}" lectures/05/results \
  && git add lectures/05/results \
  && git commit -m "Lecture 05: Add results" \
  && git pull --no-rebase --no-edit \
  && git push
```

- 先頭の `! grep` は、キーが見つかると、`!` で反転して失敗になり、`git add` 以降を実行しない。`&&` でつないであるので、どれか 1 つが失敗すると、その後のコマンドも実行されない（キーが見つかったのに、add・commit・push が動いてしまうことを防ぐ）。
- `git pull` で衝突したときも、そこで止まり、`git push` は実行されない。止めて知らせる。
- `git add -A` は使わない。関係ないファイルが入らないよう、`lectures/05/results` だけを add する。
- commit のメッセージは、PR のほかのコミットに合わせて英語にする（`Lecture 05: Add results`）。
- push したら、Claude に「push した」と伝える。こちらで pull して、answer.md に書き写す。
- 最後に、Codespace を停止する（左下の「Codespaces」→ Stop Current Codespace）。

## うまくいかないとき

<details><summary>症状と対処の一覧</summary>

| 症状 | 対処 |
|---|---|
| `sqlite_vec` の ImportError（`No module named 'sqlite_vec'`） | `uv sync --frozen` を忘れている。lec05 では sqlite-vec が増えるので、lec04 から切り替えたあとは必ず実行する |
| 「検索DBがありません」 | `echo "$AIE_RAG_DB"` が空か確かめ、設定されていたら `unset AIE_RAG_DB`（ランナーは、設定されていると止まる）。`ls data/rag_data/index.sqlite3` で、ファイルがあるかも確かめる。無ければ `git pull --no-rebase --no-edit` で lec05 を最新にし、それでも無ければ止めて知らせる |
| 429（レート制限）が続く | スクリプトが自動で待って再試行する（最大 5 回）。止まったら、同じコマンドで続きから実行する。続くときは、止めて知らせる（`--interval 20` のように待ちを長くしてもよい） |
| 終了コード 2（1 日の上限） | そこまでの結果を push し、日本時間 17 時以降に、同じコマンドで続きから実行する |
| 全文投入が入力上限で通らない | `status=input_limit` として結果に記録される。やり直さない（answer.md では「未実行」と書く）。ほかの条件は、そのまま進めてよい。push のあとで知らせる |
| 未コミットの変更があると言われて止まった（preflight） | `git status` で、`app/`・`data/rag_data/`・`lectures/05/compare.py`・`lectures/05/conditions.toml`・`pyproject.toml`・`uv.lock` を編集していないか見る。止めて知らせる（`git restore` や commit はしない）。`??` の行は、git が追跡していないファイルなので、中身を確かめずに消さない |
| 条件が変わったと言われた（条件の sha256 が、いまのコードと違う） | 途中で、条件かコードが変わっている。止めて知らせる（別の `--out` に書くかを、相談して決める） |
| STEP 3 で、保存の承認を聞かれた | N と答える。CLI の通常モードは、第3回の `save_learning_log`（承認付き）も持つため |
| 503（モデルが混み合っている）が続く | スクリプトが 30・60・120 秒待って再試行する。止まったら、時間をおいて同じコマンドで続きから実行する。それでも続くときは、モデルを変えずに知らせる（`AIE_MODEL` を変えると、結果が別のフォルダに入り、表が分かれるため） |
| 「環境変数 GEMINI_API_KEY が見つかりません」 | GitHub の Settings → Codespaces → Secrets で、`GEMINI_API_KEY` の Repository access に自分の fork が入っているかを確かめる。登録や変更のあとは、Codespace を開き直す（左下の「Codespaces」→「Stop Current Codespace」→ もう一度開く） |
| 質問が空だと言われる（段 1） | 別のターミナルを開くと、`AIE_QUESTION` が消えている。同じターミナルで、もう一度設定する |
| やり直したい実行がある | 「上書きを防ぐために」のとおり、その `t<k>.json` と `t<k>.trace.txt` を消してから、同じコマンドを実行する |

</details>
