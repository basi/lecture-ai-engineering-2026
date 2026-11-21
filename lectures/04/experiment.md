# 第4回 演習・宿題の実験手順（GitHub Codespaces）

演習と宿題（基礎・実装・発展）の実験を、GitHub Codespaces で実行する手順です。実験の設計・仮説・結果は [answer.md](answer.md) に、変種（V0〜V6・I0・I1）と実装のコードの判定のデータは [conditions.toml](conditions.toml) に書きます。実行には比較用スクリプト [compare.py](compare.py) を、集計には [report.py](report.py) を使います。手順の中の「知らせる」は、Claude に伝えることです（出力はそのまま貼ります）。

- 演習：公式の `app.judge` で、LLM に講師の評価セット（10 件）を採点させる（STEP 2）。基準を 1 か所直して、もう一度採点する（STEP 3）。
- 基礎：採点の設定（rubric・採点のモデル・出力の型・順序）を 1 か所ずつ変えた条件を、同じ 10 件で各 3 試行して比べる。
- 実装：自分の評価セット（10 件）に、自分のエージェントで回答を作り、ラベルを付けて採点する。
- 発展：SWE-bench Verified と τ²-bench の問題を読む。コードは書かず、API も使わない。

## 使うファイル

| もの | 場所 | 説明 |
|---|---|---|
| 比較用スクリプト | `lectures/04/compare.py` | 変種と評価セットの組を LLM に採点させ、結果を保存する（`run`）。実装の回答を自分のエージェントで作る（`generate`）。サブコマンドは `variants`・`status`・`run`・`generate`・`summarize` |
| 集計 | `lectures/04/report.py` | `summarize` が呼ぶ。`results/summary.md` と `summary.json` を作る |
| 条件のデータ | `lectures/04/conditions.toml` | 変種 V0〜V6（演習・基礎）と I0・I1（実装）、実装のコードの判定（m01〜m10） |
| rubric | `lectures/04/rubrics/baseline.md`・`data/eval/rubric.md` | 前者は講師の rubric のコピー（V0 などが使う）。後者は演習 STEP 3 で自分が直す rubric（V1・I0 が使う） |
| 評価セット | `data/eval/lec04_cases.jsonl`・`data/eval/my_cases.jsonl` | 演習と基礎は講師の 10 件（id は `01`〜`10`）。実装は自分の 10 件（id は `m01`〜`m10`。回答は `generate` で入る） |
| ラベル | `lectures/04/labels.jsonl`・`lectures/04/my_labels.jsonl` | 演習と基礎は `labels.jsonl`（セッション 1 で付けた）。実装は `my_labels.jsonl`（段 4 で付ける）。講師のラベルは `lectures/04/reference/labels_instructor.jsonl` |
| 公式の道具 | `app/judge.py`・`tools/annotate.py` | 演習 STEP 2・3 と実装の採点（`python -m app.judge`）、ラベル付けの画面（Streamlit） |
| テスト | `lectures/04/test_compare.py`・`test_report.py`・`test_lec04.py` | 比較用スクリプト・集計・実装のコードの判定のテスト。API キーを使わない |
| 結果の置き場所 | `lectures/04/results/` | 下の「結果の置き場所」 |
| 自分のメモ | [results/notes.md](results/notes.md) | 自分で書く（段 1・2・4・6）。Claude が answer.md に反映する |

### 変種（条件）の一覧

`--variant` には、名前（`native_reason_first`）でも label（`V5`）でも指定できます。

| label | 名前 | 段 | V0 から変えること |
|---|---|---|---|
| V0 | `baseline` | 演習 | なし。講師の rubric のまま（演習 STEP 2 と同じ） |
| V1 | `step3_policy` | 演習 | rubric を、STEP 3 で直した `data/eval/rubric.md` にする |
| V2 | `model_flash` | 基礎 | 採点のモデルを gemini-3.5-flash にする（既定は gemini-3.5-flash-lite） |
| V3 | `rubric_clarity` | 基礎 | 分かりやすさに 1 行足す（コードブロックを除いた本文が 450 字を超えたら fail） |
| V4 | `order_schema` | 基礎 | 出力の型のフィールド順だけを、判定→理由にする |
| V5 | `native_reason_first` | 基礎 | 構造化出力（NativeOutput）にし、順序を指示する文を外す（理由→判定）。V6 と組で比べる |
| V6 | `native_verdict_first` | 基礎 | V5 のスキーマの順だけを、判定→理由にする |
| I0 | `impl_step3` | 実装 | 自分の評価セットを、STEP 3 後の rubric で採点する |
| I1 | `impl_baseline` | 実装 | 自分の評価セットを、講師の rubric（STEP 3 前）で採点する（試行は 1 回だけ） |

### 結果の置き場所

```text
lectures/04/results/
├── judge/<variant>/<case_id>/t<k>.json     compare.py run の 1 回分の記録（判定と理由、生の出力、使用量、応答ヘッダなど）
├── judge/<variant>/request_sample.json     変種ごとに、最初の 1 件の送信の本文（API キーは入らない）
├── generate/<case_id>.json                 compare.py generate の 1 件分の記録（ツール呼び出し、承認、使用量）
├── cli/<名前>.txt・.json                    公式の app.judge の出力（compare_instructor・step2・step3・my_judge）
├── notes.md                                自分のメモ
└── summary.md・summary.json                集計（summarize が作る。手で直さない）
```

- `<variant>` は変種の名前（`baseline`・`step3_policy` など）で、`V0` などの label ではない。
- `<case_id>` は、演習と基礎が `01`〜`10`、実装が `m01`〜`m10`。`<k>` は試行の番号（1〜3。I1 は 1 だけ）。
- 実装の公式の採点の JSON は `lectures/04/my_judge_result.json`（`--out` で指定する）。

## 上書きを防ぐために

- **ラベルは、付けたらすぐ commit して凍結します。** LLM の判定を見る前に付けて commit したラベルだけを、「人のラベル」として使います。`run` は、`labels.jsonl`・`my_labels.jsonl` に未コミットの変更があると止まります。直したくなったときは、`labels.jsonl`・`my_labels.jsonl` を直さずに、別のファイルに保存して知らせてください（両方を報告します）。
- **annotate.py の「ラベルの保存先」を変え忘れないでください。** 実装のラベルを付けるときに、「評価セット」だけを `my_cases.jsonl` に変えて、保存先が `lectures/04/labels.jsonl` のままだと、演習のラベルが上書きされます（段 4 の 3）。
- **`generate` は、回答が入っていると止まります。** `--force` は使いません（回答を作り直すと、付けたラベルとずれます）。途中で止まったときの再開も、`--force` なしの同じコマンドです（済みの件は飛ばされます）。
- **`my_cases.jsonl` の id（`m01`〜`m10`）は変えません。** annotate.py の「作問」画面では追加しません（作問の画面は id を `01` から振るので、`conditions.toml` のコードの判定と対応しなくなります）。直すときは、エディタで直接直します。
- **採点した後に、評価セット（`my_cases.jsonl` の回答など）を書き換えません。** 書き換えると、`run` が「いまの条件と違います」で止まります。`summarize` も、書き換える前の回答への判定を集計から外し、`summary.md` に警告を出します。
- **既にある結果は飛ばします。** `run` は、保存先に `t<k>.json` があればその実行を飛ばすので、同じコマンドを何度実行しても上書きされません。止まったときも、同じコマンドで続きから実行できます。
- **結果は、実行が終わったときだけ置かれます。** API のエラーで止まった実行は `t<k>.json` が置かれないので、次に実行したときにやり直されます。
- **やり直したいとき**は、その `t<k>.json` を別のフォルダに移してから実行します。
- **実行中に、`compare.py`・`report.py`・`conditions.toml`・`rubrics/`・`app/judge.py`・`app/agent.py`・`app/tools.py`・`app/config.py`・`data/glossary.json`・`data/eval/` を書き換えないでください。** 未コミットの変更があると、`run` は止まります。結果がどの commit のものか分からなくなるためです。書き換えたときは、commit してから実行します（`--allow-dirty` は使いません）。
- **試行の途中で変種の中身が変わると止まります。** 1 回目と 2 回目で、rubric・指示・スキーマ・評価セット・モデルのどれかが違うと、比べられないためです。
- **`git add -A` は使いません。** `notebooks/` に講義資料があり、関係ないファイルが入るためです。add するときは、パスを指定します。
- **`AIE_MODEL` は設定しません。** `run` の採点のモデルは `conditions.toml` で決まりますが、公式の `app.judge`（STEP 2・3、実装の採点）と `generate` は `AIE_MODEL` を読みます。

## セッション 1（済んだこと）

ラベル付けと、講師のラベルとの比較までを、Codespaces で済ませました。Claude が仮説（answer.md）を書く前に、LLM の結果を見ないためです。

```bash
git switch main
git pull --no-rebase
git switch -c lec04
uv run pytest -q                                                    # 講座のテスト
uv run --with streamlit==1.66.1 streamlit run tools/annotate.py     # 10 件にラベルを付けて、Ctrl+C で止める
git add lectures/04/labels.jsonl
git commit -m "Lecture 04: Add my labels for the exercise cases"
git push -u origin lec04
mkdir -p lectures/04/results/cli
uv run python -m app.judge --compare lectures/04/reference/labels_instructor.jsonl | tee lectures/04/results/cli/compare_instructor.txt
git add lectures/04/results/cli/compare_instructor.txt
git commit -m "Lecture 04: Compare my labels with the instructor's"
git push
```

- 公式の `app.judge`（演習 STEP 2）は、まだ実行していません。段 2 で実行します。
- そのあと Claude が、比較用スクリプト・条件・評価セットの下書き（`my_cases.jsonl`）・この手順書・仮説（answer.md）を `lec04` に push しました。

## 準備

### 1. Codespaces を lec04 で開く

- 新しく開く場合：GitHub で自分の fork を開き、ブランチを `lec04` に切り替えてから、「Code → Codespaces → Create codespace on lec04」を押す。
- 既に Codespace がある場合：開く（セッション 1 で停止した Codespace を再開する）。

どちらの場合も、ターミナルで次を実行します。

```bash
git switch lec04
git pull --no-rebase --no-edit
```

- `git switch lec04` で止まる場合は、変更が残っている。`git status` を見て、知らせる。
- `git pull --no-rebase --no-edit` は、Claude が push した分を取り込むため。`--no-edit` で、マージのメッセージの編集画面（vi）を開かずに取り込む。
- `CONFLICT` と出たら、`git merge --abort` で元に戻してから知らせる（元に戻すので、自分の変更は消えない）。

### 2. 依存を確かめる

```bash
uv sync --frozen
```

- Codespace を作ったときに実行されている。lec04 では `pyproject.toml`・`uv.lock` を変えていないので、追加の導入は要らない（念のため実行する）。
- Streamlit は、ラベル付け（段 4）のときだけ、`uv run --with streamlit==1.66.1` で一時的に足す。初回は取得に 1〜2 分かかる。

### 3. API キーを確かめる

```bash
[ -n "$GEMINI_API_KEY" ] && echo 設定済み || echo 未設定
echo "${AIE_MODEL:-未設定}"
```

- `GEMINI_API_KEY` は、値を表示しない。「未設定」のときは、下の「うまくいかないとき」を見る。
- `AIE_MODEL` は「未設定」でよい。値が出たら、`unset AIE_MODEL` してから進める（上の「上書きを防ぐために」の最後）。

## 最初の確認（API キー不要）

```bash
uv run pytest -q                                      # 講座のテスト
uv run pytest -q lectures/04                          # 課題のコードと compare.py・report.py のテスト
uv run python lectures/04/compare.py variants         # 変種ごとの違い（API なし。保存はしない）
uv run python lectures/04/compare.py status           # 予定・済み・残りの件数と、残りの推定リクエスト数
```

- `uv run pytest -q` と `uv run pytest -q lectures/04` は、別々に実行する（同時に指定すると、他の回のテストとモジュール名がぶつかる）。どちらも failed が 0 件であること。**落ちたら、先に進まずに知らせる。**
- `variants` は、変種ごとに、モデル・rubric・指示の末尾・スキーマの順と、V0 からの変更の数を表示する。
  - V1 と I0 に「data/eval/rubric.md はまだ lectures/04/rubrics/baseline.md と同じです」と出るのは正常（段 2 の STEP 3 の後に消える）。
  - V5・V6 の「V0 からの変更」が 3 点・4 点と出るのも正常（V5・V6 は複数の変更の束。V5 と V6 の差だけを、順序の効果として読む）。V2〜V4 は 1 点。
- `status` は、予定 260 回（V0〜V6 と I0 が各 30 回、I1 が 10 回、`generate` が 10 件）、残りの推定リクエストの合計 280 と出る。
  - 最後の「ラベル」の欄で、`lectures/04/labels.jsonl` が「commit 済み」であること。`my_labels.jsonl` が「まだありません」なのは正常（段 4 で付ける）。

## 実行の順番

最初に、演習・基礎・実装を 1 試行ずつ流して、すべてのレベルの結果をそろえます。そのあと、2・3 試行目をまとめて足します。利用枠が途中で尽きても、段 2 で演習を、段 3 で基礎を、段 4 で全レベル（1 試行）を提出できます。表の「順」が、下の「段」の番号です。

**☆ は実験済み**です（結果は `lectures/04/results/` に置き、[answer.md](answer.md) に記入済み）。状態の列は、Claude が結果を書き終えたあとに付けます。

| 順 | 状態 | 内容 | コマンド（要点） | API の目安 | 時間の目安 |
|---:|---|---|---|---:|---:|
| 0 | ☆ | 準備と確認、1 件だけ試す（V0・V2・V4・V5・V6 の case 01）。**失敗したら止めて知らせる** | `git pull`・`uv sync`・`pytest`・`variants`・`status`、`run --variant V0,V2,V4,V5,V6 --case 01 --trials 1`、`summarize` | 5 | 約 10 分 |
| 1 | ☆ | 講師と違った項目のメモ | `results/cli/compare_instructor.txt` を見て、notes.md の 1 に書く | 0 | 約 10 分 |
| 2 | ☆ | 演習 STEP 2・3。ここまでで演習を提出できる | `app.judge`（STEP 2）→ rubric を 1 か所直して commit → `app.judge`（STEP 3） | 20 | 約 15 分 |
| 3 | ☆ | 基礎の 1 試行目。ここまでで基礎を提出できる | `run --level exercise,basic --trials 1`、`status`、`summarize` | 65 | 約 15 分 |
| 4 | | 実装（回答の生成・ラベル付け・採点）。ここまでで全レベルを提出できる | `generate`、ラベル付け、`app.judge`（my_cases）、`run --level impl --trials 1` | 約 60 | 約 40 分 |
| 5 | | 残りの試行（t2・t3）。3 試行がそろう | `run --level exercise,basic,impl` | 約 160 | 約 35 分 |
| 6 | | 集計と発展（問題と記事を読む） | `summarize`、ブラウザで読む | 0 | 約 30 分 |
| 7 | | 受け渡し | `grep`、`git add`・commit・pull・push | 0 | 約 5 分 |

- 合計は、LLM リクエストが約 310 回（うち gemini-3.5-flash が 30 回）、時間が約 2.5 時間。時間は、実行の間の待ち（既定 8 秒）を含めた見積もりで、429 で待つと延びる。
- 段 0 の 5 回は、段 3 で飛ばされる（段 3 は 70 回のうち、残りの 65 回）。
- `--level`・`--variant`・`--case` には、`,` で区切って複数を指定できる。`--level` と `--variant` を両方指定すると、両方に当てはまる変種だけを流す。
- `--level` の `exercise` は V0・V1、`basic` は V2〜V6、`impl` は I0・I1。段 3 で `--level exercise,basic` に限るのは、`impl` は段 4 の回答の生成とラベル付けの後でなければ流せないため。
- 段 5 は `--trials` を付けない。`--trials` は試行の回数の上限の意味で、付けないと、各変種の試行数（V0〜V6・I0 は 3、I1 は 1）まで流す。
- 提出するときは、その段の終わりで、下の「結果の保存と受け渡し」を行う（push まで）。
- 各段の終わりに `status` で「済み」が増えたことを確かめる。

## 段ごとの手順

### 段 0：1 件だけ試す（API 5 回）

準備と最初の確認は、上の 2 節です。ここでは、本番の API で、V0・V4 のツール出力、V5・V6 の構造化出力（NativeOutput）、gemini-3.5-flash（V2）が使えるかを、最初に確かめます。

```bash
uv run python lectures/04/compare.py run --variant V0,V2,V4,V5,V6 --case 01 --trials 1
uv run python lectures/04/compare.py summarize
ls lectures/04/results/judge/*/request_sample.json
```

- `run` は、最後に `実行 5 回（ok 5）` と出る。
- `lectures/04/results/summary.md` を Codespaces で開き（右上のプレビューのアイコンで見ると読みやすい）、「表E 順序の操作の確認」の V5・V6 の行で、`order_ok` が `true 1 / false 0 / null 0` であること。V0・V2・V4 はツール出力なので `null` になる（参考値。正常）。
- `request_sample.json` が、5 つの変種（`baseline`・`model_flash`・`order_schema`・`native_reason_first`・`native_verdict_first`）に 1 つずつあること。
- `summary.md` の「警告」に、まだ流していない変種の「記録が 1 件もありません」と、流した 5 変種の「欠け 29 件を集計から外しました（記録なし）」が出るのは正常（段 3 以降で減る）。
- 段 4 の前は、「自分のラベル `lectures/04/my_labels.jsonl` を読めません（ファイルがありません）」と「実装: generate の記録が使えない件があります」も正常（実装のラベルと回答は、段 4 で作る）。
- **失敗したら、ここで止めて知らせる。** エラーの文をそのまま知らせる。
  - 400（`API エラーで止めます（400）`）なら、V5・V6 を外して進める。
  - 404（`API エラーで止めます（404）`）なら、V2 を外して進める。
  - どちらも、外し方は「うまくいかないとき」に書いてある。外す前に知らせる。
  - `order_ok` が `true` でない、`request_sample.json` が無い、`invalid_output` や `error` が出た、のときも知らせる。

### 段 1：講師と違った項目のメモ（API なし）

```bash
cat lectures/04/results/cli/compare_instructor.txt
```

- 見るところ：基準別の一致率（自分と講師、10 件）、「判定が違う項目 N 件」（`#id 基準 自分=… 講師=…（講師のメモ: …）`）、「自分が『迷った』を付けた項目」。
- 違った項目ごとに、自分はどう読んで、その判定にしたかを、[results/notes.md](results/notes.md) の 1 に書く。違いが 0 件のときは、「迷った」を付けた項目と、迷った理由を書く。
  - ラベル付けのときに annotate.py のメモ欄へ理由を書いていれば、`lectures/04/labels.jsonl` の `note` に残っている（講師のラベルを見る前の記録）。書き直さず、notes.md にはそれを写すか、補足だけを書けばよい。
- 講師のラベルは正解ではない（人同士でも割れる）。
- notes.md は、段 2 の commit で一緒に add する。

### 段 2：演習 STEP 2・3（API 20 回）

**STEP 2**

```bash
mkdir -p lectures/04/results/cli
uv run python -m app.judge --out lectures/04/results/cli/step2.json | tee lectures/04/results/cli/step2.txt
```

- 出力の読み方は [problem.md](problem.md) の STEP 2 のとおり。先頭に `[model] gemini-3.5-flash-lite  temperature=0  rubric=data/eval/rubric.md` と出ること。
- 見るところ：基準別の一致率（分かりやすさが低いか）、要確認 N 件（LLM が unsure にした項目と理由）、不一致。
- `--out` で `results/cli/` に JSON を置く（既定の `lectures/04/judge_result.json` は使わない）。

**方針を決める**

1. 要確認の項目から、1 つ選ぶ。要確認が 0 件なら、STEP 1 で「迷った」を付けた項目を選び、基準の文言を 1 行変える。
   - 回の表示の項目は選ばない。回の表示はコード（`check_lecture_mark`）が判定するので、rubric を直しても判定は変わらない。
2. その項目を、合格にするか不合格にするかを決める（どちらでもよい）。
3. `data/eval/rubric.md` の該当する基準の下に、決めた方針を 1〜2 行で書き足し、最後の「まだ決めていないこと」から、その論点を消す。
   - **一度に直すのは 1 か所だけ。** 2 か所直すと、V1 が V0 との差を 1 か所で比べられなくなる。
   - 書き足す文に、出力の形式や、理由と判定の順序を変える指示を混ぜない。
4. 方針と理由を、notes.md の 2 に書く。

```bash
git diff --stat -- data/eval       # data/eval/rubric.md だけが、数行の変更になっていること
git add data/eval/rubric.md lectures/04/results/cli lectures/04/results/notes.md
git commit -m "Lecture 04: Decide the policy for STEP 3"
```

**STEP 3**

rubric を直して commit する前に、STEP 3 のコマンドを実行しない（STEP 2 と同じ rubric で採点することになる）。実行してしまったら、上書きされる前に `step3.json`・`step3.txt` の名前を `step2_rerun.json`・`step2_rerun.txt` に変えて残す（同じ rubric での 2 回目として使える）。

```bash
uv run python -m app.judge --out lectures/04/results/cli/step3.json | tee lectures/04/results/cli/step3.txt
```

- 見るところ：一致率、要確認の件数、選んだ項目の判定が STEP 2 からどう変わったか。
- STEP 2 と STEP 3 の変化は、Claude が `step2.txt` と `step3.txt` を比べて answer.md に書く。気づいたことがあれば、notes.md の 5 に書く。

### 段 3：基礎の 1 試行目（API 65 回、約 15 分）

```bash
uv run python lectures/04/compare.py run --level exercise,basic --trials 1
uv run python lectures/04/compare.py status
uv run python lectures/04/compare.py summarize
```

- V0 → V1 → … → V6 の順に、各 10 件を流す。段 0 で済んだ 5 件は飛ばされる。
- V1 は、段 2 で `data/eval/rubric.md` を commit していないと止まる（「うまくいかないとき」）。
- `status` で、V0〜V6 の「済み」が各 10 件になったこと。
- `summary.md` の「表E 順序の操作の確認」で、V5・V6 の `order_ok` を見る。`true 10 / false 0 / null 0` に近いこと。
  - **`false` が半数以上（5 件以上）のときは、先に進まずに知らせる。** `null` が出たときも知らせる（出力の形式が崩れた可能性）。
- `summary.md` の「表F 使用量と警告」で、「警告ヘッダのある記録」の数と、「警告ヘッダの例」を見る。出ていても、止めずに進める（結果は summary.md に残るので、Claude が読む）。思考トークン（`thoughts_tokens`）も、V2 と他で違うかを見る。

### 段 4：実装（API 約 60 回）

**1. 評価セットを確認・修正する**

`data/eval/my_cases.jsonl` をエディタで開き、10 件の質問（`question`）と、期待するふるまい（`reference.expected`）を確認・修正します。

- id（`m01`〜`m10`）は変えない。`answer` は空のままにする（`generate` が書く）。
- 期待するふるまいに、「第N回」や、段 2 で決めた STEP 3 の方針は書かない（採点器が別の基準と混同した例があるため）。
- **質問を大きく変えたときは、知らせる。** `conditions.toml` のコードの判定が、id ごとに書いてあるため。
- 直したことと、その理由を、notes.md の 3 に書く。
- 1 行が 1 件の JSON なので、文字列の中だけを直す。壊していないかは、次の `variants` で確かめる。

```bash
uv run python lectures/04/compare.py variants --variant I0     # 「評価セット: data/eval/my_cases.jsonl（10 件。回答が空 10 件 …）」と出ること
git add data/eval/my_cases.jsonl lectures/04/results/notes.md
git commit -m "Lecture 04: Finalize my eval cases"
```

**2. 回答を生成する**

```bash
uv run python lectures/04/compare.py generate
git add data/eval/my_cases.jsonl lectures/04/results/generate
git commit -m "Lecture 04: Generate answers with my agent"
```

- 約 30 リクエスト（1 件が 3 回前後）。1 件ごとに 8 秒待つ。自分のエージェント（`app/agent.py`）を動かし、承認が要る操作（`save_learning_log`）は自動で断る（承認はしない）。
- 全件が成功したときだけ、`my_cases.jsonl` の `answer` を書く。途中で失敗したら書かない（同じコマンドで続きから。済みの件は飛ばされる）。
- 見るところ：`[1/10] m01: ok 4.2s tool_calls=1 requests=2 approvals=0` のような行が 10 件。最後に「→ 回答を書き込みました（10 件）」と出ること。
- 配布の `app.judge --generate` は使わない。承認待ちが `DeferredToolRequests(...)` という文字列のまま回答に入る、途中で失敗すると全件が消える、ツール呼び出しが残らない、ため。
- `--force` は使わない。

**3. ラベルを付ける**

```bash
uv run --with streamlit==1.66.1 streamlit run tools/annotate.py
```

- 右下の「ブラウザーで開く」を押す（出ない場合は、下の「ポート」タブから 8501 を開く）。
- **最初に、サイドバーの「評価セット」を `data/eval/my_cases.jsonl`、「ラベルの保存先」を `lectures/04/my_labels.jsonl` の両方に変える（書き換えて Enter）。** 両方を変え終わるまで、ラジオボタンにも「保存して次へ」にも触らない。保存先を変え忘れると、演習のラベルが上書きされる。
- 「作問の保存先」と、画面の「作問（宿題・実装）」は触らない。画面は「ラベル付け」のままにする。表示がおかしいときは、ブラウザを再読み込みする。
- 5 つの基準を、1 件ずつ自分で選ぶ。未入力の既定値は「合格」なので、そのまま進めない。「回の表示」も、自分で見て選ぶ。
- 自分の知識ではなく、参照（用語集のエントリと期待するふるまい）と基準に照らして判定する。迷ったら「迷った」にチェックして、メモに迷った基準と理由を書く。
- ラベルを commit するまでは、`app.judge` と `run --level impl` を実行しない（LLM の判定を見る前に、ラベルを凍結するため）。
- 10 件目まで「保存して次へ」を押すと、「10 件すべて保存しました → lectures/04/my_labels.jsonl」と出る。Ctrl+C で止める。迷った件は、notes.md の 3 に書く。

```bash
git status --short lectures/04/labels.jsonl lectures/04/my_labels.jsonl    # my_labels.jsonl だけが ?? で出ること（labels.jsonl は出ない）
git add lectures/04/my_labels.jsonl
git commit -m "Lecture 04: Add my labels for my eval cases"
```

**4. 公式の採点**

```bash
uv run python -m app.judge --cases data/eval/my_cases.jsonl --labels lectures/04/my_labels.jsonl --out lectures/04/my_judge_result.json | tee lectures/04/results/cli/my_judge.txt
```

- API 10 回。STEP 3 の後の `data/eval/rubric.md` で、自分のラベルと比べる。
- 見るところ：基準別の一致率、要確認、不一致。

**5. 比較用スクリプトで採点する**

```bash
uv run python lectures/04/compare.py run --level impl --trials 1
uv run python lectures/04/compare.py status
```

- I0（STEP 3 後の rubric）と I1（講師の rubric）を、各 10 件。API 20 回。
- `status` で、I0・I1 の「済み」が各 10 件、`generate` が 10 件になったこと。

### 段 5：残りの試行（API 約 160 回、約 35 分）

```bash
uv run python lectures/04/compare.py run --level exercise,basic,impl
uv run python lectures/04/compare.py status
```

- t2・t3 を足す（V0〜V6 と I0 の各 20 件）。I1 は 1 試行だけなので、足されない。`--trials` は付けない。
- 試行を外側に回すので、t2（全変種）→ t3（全変種）の順に進む。止まっても、同じコマンドで続きから実行できる。
- 1 日の上限（終了コード 2）で止まったときは、「利用枠」を見る。
- `status` の合計が、済み 260・残り 0 になれば完了。

### 段 6：集計と発展（API なし、約 30 分）

```bash
uv run python lectures/04/compare.py summarize
```

- `lectures/04/results/summary.md` が最終版（手で直さない）。「警告」の欄を見る。

**発展：SWE-bench Verified の問題を読む**

読む順番：

1. まず `problem_statement` だけを読み、「自分なら解けるか」をメモする。
2. そのあと `patch`（正解）、`test_patch`、`FAIL_TO_PASS` を開く。
3. 元の PR は最後に開く（答えが見えるため）。instance_id の末尾の番号で開ける（例：https://github.com/sympy/sympy/pull/18199）。

| ID | 開く URL |
|---|---|
| django__django-11099 | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer/default/test?row=31 |
| astropy__astropy-12907 | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer/default/test?row=0 |
| sympy__sympy-18199 | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer/default/test?row=464 |
| sphinx-doc__sphinx-7590 | https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/viewer/default/test?row=392 |

**発展：τ²-bench の問題を読む**

論文の時点の版（v0.1.0）に固定したリンクです。読むところ：

- `user_scenario.instructions`：ユーザー役への指示。エージェントには見えない。
- `evaluation_criteria`：`actions`・`communicate_info`・`env_assertions`・`reward_basis`。
- 判断の根拠になるポリシー：同じフォルダの `policy.md`（telecom は `main_policy.md`）。

先に `user_scenario` とポリシーを読んで、自分ならどう対応するかを考え、そのあとで `evaluation_criteria` を見る。

| 問題 | 開く URL |
|---|---|
| airline 14 | https://github.com/sierra-research/tau2-bench/blob/37199f36924c8896f5e048360691f8476cd89ba1/data/tau2/domains/airline/tasks.json#L755-L862 |
| retail 24 | https://github.com/sierra-research/tau2-bench/blob/37199f36924c8896f5e048360691f8476cd89ba1/data/tau2/domains/retail/tasks.json#L2288-L2314 |
| telecom（データ通信） | https://github.com/sierra-research/tau2-bench/blob/37199f36924c8896f5e048360691f8476cd89ba1/data/tau2/domains/telecom/tasks.json#L2-L109 |
| telecom（引き継ぎ） | https://github.com/sierra-research/tau2-bench/blob/37199f36924c8896f5e048360691f8476cd89ba1/data/tau2/domains/telecom/tasks.json#L5876-L5959 |

- airline 14 は、あとで直された版もある（読み終わってから開く）：https://github.com/sierra-research/tau2-bench/blob/c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4/data/tau2/domains/airline/tasks.json#L795-L906

**発展：OpenAI の記事 3 本を読む**

ブラウザで開き、次の点を原文で確かめます。Claude は、この 3 本の本文を読めていません（取得が拒否された）。下の数字と日付は、検索の抜粋や別の資料から書いたものなので、原文と違っていたら、原文のとおりに書いてください。

| 記事 | 確かめること |
|---|---|
| [Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)（2024-08-13） | 93 人・1,699 件・38.3%・61.1%・68.3%。それぞれ、何の人数・件数・割合か |
| [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)（2026-02-23） | SWE-bench Verified での報告をやめた理由。138 件の監査（何を監査し、どうだったか） |
| [Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)（2026-07-08） | 撤回の文言（何の推奨を、なぜ撤回したか。原文の言い方） |

読んだ結果は、notes.md の 4 に書きます。SWE-bench Verified 4 件、τ²-bench 4 件、記事 3 本の欄が、雛形に並べてあります。

### 段 7：受け渡し

下の「結果の保存と受け渡し」のとおりに、`summarize` → キーの確認 → add → commit → pull → push を行い、Claude に知らせて、Codespace を停止します。

## 各段で確かめること

段が終わるたびに `status` を実行し、「済み」が増えたことを確かめます。段ごとの見どころは次のとおりです（比べた結果は answer.md に書きます）。

| 段 | 見るところ |
|---:|---|
| 0 | pytest が通る。`run` が `実行 5 回（ok 5）` で終わる。表E で V5・V6 の `order_ok` が `true 1 / false 0 / null 0`（V0・V2・V4 は `null` で正常）。`request_sample.json` が 5 つある |
| 1 | 講師と違った項目（`#id 基準 自分=… 講師=…`）と、講師のメモ。違いが 0 件のときは、「迷った」を付けた項目 |
| 2 | STEP 2：分かりやすさの一致率が低いか。要確認の項目と理由（0 件なら、STEP 1 で「迷った」を付けた項目）。STEP 3：選んだ項目の判定が、STEP 2 からどう変わったか。一致率と要確認の件数 |
| 3 | `status` で V0〜V6 の済みが各 10 件。表E：V5・V6 の `order_ok` がほぼ `true`（`false` が 5 件以上なら止めて知らせる）。表F：警告ヘッダの有無と例、`thoughts_tokens` |
| 4 | `generate`：`[n/10] m01: ok …` が 10 件。ラベルを付けた後の `git status --short` に、`lectures/04/labels.jsonl` が出ない。`my_judge.txt` の一致率・要確認・不一致。`status` で I0・I1・`generate` の済みが各 10 件 |
| 5 | `status` の合計が、済み 260・残り 0 |
| 6 | `summary.md` の「警告」の欄。notes.md の 4 に、発展の問題と記事の分が書けたか |
| 7 | `grep` が `キーなし`。`git status --short` に、意図しないファイルが無い。push できた |

## 利用枠

- Gemini API の無料枠には、1 分あたりと 1 日あたりの上限がある。値は AI Studio の rate limit の画面で確かめる。上限はプロジェクトごと、モデルごとに違う（V2 の gemini-3.5-flash は、他の変種と別の上限になる）。API キーを増やしても増えない。
- 1 日の上限は、太平洋時間の 0 時（日本時間 17 時）に戻る。
- 1 分あたりの上限（429）に当たると、スクリプトが待ってから自動で再試行する（本文の `retryDelay`、なければ 60 秒。最大 5 回）。503 などは、30・60・120 秒待って 3 回まで再試行する。
- 1 日の上限（エラーの本文に `PerDay`）に当たると、「1 日の上限（PerDay）に達しました。明日（日本時間 17 時以降）、同じコマンドで続きから実行してください。」と表示して止まる（終了コード 2。直後の `echo $?` で確かめられる）。そこまでの結果を push しておき、日本時間 17 時以降に同じコマンドを実行する。
- 実行の間の待ちは `--interval`（秒、既定 8）で、1 回に流す件数は `--max-runs` で変えられる（`run --max-runs 20` のように少しずつ流せる）。
- Ctrl+C で止めると、終了コード 130 で止まる。そのときに実行中だった 1 件は置かれないので、同じコマンドで続きから実行すると、やり直される。

## 結果の保存と受け渡し

段 7 のほか、1 日の上限などで途中で止まったときも、次の順で push します。

```bash
uv run python lectures/04/compare.py summarize
grep -rnEI "AIza[0-9A-Za-z_-]{30,}" lectures/04 data/eval || echo キーなし
git status --short
git add lectures/04/results lectures/04/my_judge_result.json
git commit -m "Lecture 04: Add results"
git pull --no-rebase --no-edit
git push
```

- `grep` は、結果に API キーが含まれていないかの確認。`キーなし` と出ること。ファイルが出たときは、commit せずに知らせる。スクリプトも書き出す前に伏せ字にしているが、念のため確かめる。`-I` は、`__pycache__` の中のバイナリを数えないため（テストが作るダミーの文字列に当たる）。
- `git status --short` で、add する前に、意図しないファイルが無いことを確かめる。`git add -A` は使わない。
- 段 4 の公式の採点（`app.judge`）の前に止まったときは、`lectures/04/my_judge_result.json` がまだ無いので、`git add` からその指定を外す。
- commit のメッセージは、PR のほかのコミットに合わせて英語にする。
- push したら、Claude に「push した」と伝える。こちらで pull して、answer.md に書き写す。
- 最後に、左下の「Codespaces」→「Stop Current Codespace」で、Codespace を停止する。

## うまくいかないとき

止まったときは、画面の出力（エラーの文）をそのまま Claude に知らせてください。

<details><summary>症状と対処の一覧</summary>

| 症状 | 対処 |
|---|---|
| 「結果を左右するファイル（compare.py・conditions.toml・rubric・app/・data/eval/ など）に、未コミットの変更があります」で止まった | 表示された行を `git status` で確かめる。意図した変更（`rubric.md` を直した、`my_cases.jsonl` を直した、など）なら、パスを指定して add して commit してから実行する。意図しない変更なら `git restore <ファイル>` で戻す。`??` の行は、git が追跡していないファイルなので、中身を確かめて、add するか、不要なら消す。`--allow-dirty` は使わない（結果がどの commit のものか分からなくなる） |
| 「lectures/04/my_labels.jsonl を commit してから実行してください」（`labels.jsonl` のこともある） | ラベルは、LLM の判定を見る前に commit して凍結する決まり。`git add lectures/04/my_labels.jsonl` → commit する。`labels.jsonl` と出たときは、セッション 1 で commit 済みのはずなので、`git status` で変更を確かめる（annotate.py で上書きしたなら、下の「保存先を変え忘れた」の行） |
| 「演習 STEP 3 で data/eval/rubric.md を直して commit してから実行してください」（V1 か I0 で止まる） | 段 2 の STEP 3 をしていない、または `rubric.md` を commit していない。`git log --oneline -3 -- data/eval/rubric.md` で commit があるかを確かめる |
| 段 0 で V5 が「API エラーで止めます（400）」 | **止めて知らせる**。知らせたあと、V5・V6 を外して進める（V0・V2・V4 は済んでいるので、飛ばされる）。段 3 は `run --level exercise,basic --variant V0,V1,V2,V3,V4 --trials 1`、段 5 は `run --level exercise,basic,impl --variant V0,V1,V2,V3,V4,I0,I1`（I0・I1 を忘れない）。順序の効果は、V4 だけで報告する |
| 段 0 で V2 が「API エラーで止めます（404）」（gemini-3.5-flash が見つからない） | **止めて知らせる**。知らせたあと、V2 を外して進める。段 0 は `run --variant V0,V4,V5,V6 --case 01 --trials 1` で続きから（V4・V5・V6 の確認が残っている）。段 3 は `run --level exercise,basic --variant V0,V1,V3,V4,V5,V6 --trials 1`、段 5 は `run --level exercise,basic,impl --variant V0,V1,V3,V4,V5,V6,I0,I1` |
| 「API エラー（429）。N 秒待って、再試行します」が続く | スクリプトが自動で待って再試行する（最大 5 回）。5 回を超えて止まったときは、数分おいて同じコマンドを実行する。続くときは `--interval 20` のように待ちを長くする。公式の `app.judge` は待たずに止まる（「レート制限に達しました」）ので、1 分待って再実行し、続くときは `--wait 2` を付ける |
| 「1 日の上限（PerDay）に達しました」で止まった（終了コード 2） | そこまでの結果を push し（上の「結果の保存と受け渡し」）、日本時間 17 時以降に、同じコマンドで続きから実行する |
| 503 が続く（「API エラーで止めます（503）」） | 30・60・120 秒待って 3 回再試行してから止まる。時間をおいて、同じコマンドで続きから実行する。それでも続くときは知らせる（採点のモデルを変えると、他の結果と比べられなくなるので、自分では `AIE_MODEL` を付けない） |
| `generate` が `pending_approvals` などで止まった（「N 件が ok になりませんでした: …」） | `my_cases.jsonl` は書かれていない（全件が ok のときだけ書く）。同じコマンドで続きから実行する（ok の件は飛ばされる）。同じ件が続けて ok にならないときは、`lectures/04/results/generate/<id>.json` の `status` と `error` を知らせる。`--force` は使わない |
| `generate` が「…には、もう回答が入っています」で止まった | 回答は既に入っている。作り直すとラベルとずれるので、`--force` は使わない。作り直したいときは知らせる |
| annotate.py で `IndexError` が出る、または表示がおかしい（前の評価セットの件が出る、件数がずれる） | ブラウザを再読み込みする。サイドバーの「評価セット」と「ラベルの保存先」を確かめて、もう一度入力して Enter を押す。評価セットが 10 件あることも確かめる（段 4 の 1 の `variants --variant I0`） |
| annotate.py の保存先を変え忘れて、保存してしまった（`git status --short` に `M lectures/04/labels.jsonl` が出る） | `git restore lectures/04/labels.jsonl` で、commit 済みの演習のラベルに戻す。そのうえで、保存先を直して、ラベル付けをやり直す |
| `summary.md` の「警告」に何か出た | 次は正常：「記録が 1 件もありません」（まだ流していない変種）、「欠け N 件を集計から外しました」の `（記録なし）`（まだ流していない試行）。段 4 の前の「自分のラベル `lectures/04/my_labels.jsonl` を読めません（ファイルがありません）」と、「generate の記録が使えない件があります」の `（記録なし）` も正常。次は知らせる：演習のラベル（`labels.jsonl`）の「ラベルを読めません」、「dry_run の記録が混ざっています」、「ラベルが実行後に変わりました」、「generate の記録の answer と評価セットの answer が違う件があります」、「同じ変種の記録で … がそろっていません」、`（invalid_output）`・`（error）` の欠け。`summary.md` は手で直さない（`summarize` で何度でも作り直せる） |
| 「…t1.json が、いまの条件と違います（違う項目: …）」で止まった | 同じ変種の結果に、違う条件の試行を混ぜないための停止。採点した後に、rubric・`conditions.toml`・評価セットのどれかを書き換えた。書き換えを `git restore` で戻してから、もう一度実行する。結果を取り直すなら、知らせる（自分では消さない） |
| 「環境変数 GEMINI_API_KEY が見つかりません」 | GitHub の Settings → Codespaces → Secrets で、`GEMINI_API_KEY` の Repository access に自分の fork が入っているかを確かめる。登録や変更のあとは、Codespace を開き直す（左下の「Codespaces」→「Stop Current Codespace」→ もう一度開く） |
| 「設定エラー: …」で止まった | `--variant`・`--level`・`--case` の打ち間違いなら、メッセージに使える名前が出る。打ち間違いでなければ、`conditions.toml` の問題なので知らせる |
| annotate.py の「作問（宿題・実装）」で、`my_cases.jsonl` に件を足してしまった（id が `01` から振られる） | 作問の画面は使わない。`git restore data/eval/my_cases.jsonl` で、commit 済みの状態に戻す（段 4 の 1 の commit の前なら、足した行をエディタで消す） |
| `my_cases.jsonl` の質問を大きく変えた | 知らせる。`conditions.toml` のコードの判定が id ごとに書いてあるため、質問と対応しなくなる |
| やり直したい実行がある | その `t<k>.json` を別のフォルダに移してから、同じコマンドを実行する |

</details>
