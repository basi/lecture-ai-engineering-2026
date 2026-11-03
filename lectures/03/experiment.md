# 第3回 演習・宿題の実験手順（GitHub Codespaces）

演習と宿題（基礎・実装・発展(a)・発展(b)）の比較の実験を、GitHub Codespaces で実行する手順です。実験の設計・仮説・結果は [answer.md](answer.md) に、質問・変種・シナリオ・自動チェックのデータは [conditions.toml](conditions.toml) に書きます。実行には比較用スクリプト [compare.py](compare.py) を使います。

## 使うファイル

| もの | 場所 | 説明 |
|---|---|---|
| 比較用スクリプト | `lectures/03/compare.py` | 変種と質問の組を実行し、結果を保存して集計する。サブコマンドは `status`・`variants`・`run`・`summarize` |
| 条件のデータ | `lectures/03/conditions.toml` | 質問（q01〜q10）、基礎の変種（V0 `baseline` と V1〜V7）、実装のシナリオ（s1〜s6）、自動チェック |
| エージェント（Pydantic AI 版） | `app/agent.py`・`app/tools.py` | 演習の `make_quiz` と、実装の `save_learning_log`（承認付き）を足したもの |
| エージェント（ADK 版） | `adk_agent/glossary_agent/` | 発展(a)。指示とツールは Pydantic AI 版と同じ |
| テスト | `lectures/03/test_lec03.py`・`lectures/03/test_compare.py` | 課題のコードと比較用スクリプトのテスト。API キーを使わない |
| 結果の置き場所 | `lectures/03/results/` | 1 回の実行ごとに JSON とトレースを置く（下の「結果の置き場所」） |
| 集計 | `lectures/03/results/summary.md` | `summarize` で作る比較の表 |

### 結果の置き場所

```text
lectures/03/results/
├── <model>/<engine>/<variant>/<qid>/t<k>.json        1 回分の記録（回答、ツール呼び出し、usage など）
├── <model>/<engine>/<variant>/<qid>/t<k>.trace.txt   CLI の --trace と同じ表示
├── <model>/adk|pi/baseline/<qid>/t<k>.raw.jsonl      ADK と Pi の生の出力（と t<k>.stderr.txt）
├── variants/pydantic/<variant>.json・.txt            変種ごとに、モデルに見える指示とツールの説明文
├── cli/<名前>.txt                                    演習の CLI の出力
└── summary.md・summary.json                          集計
```

- `<model>` は `gemini-3.5-flash-lite` など、`<engine>` は `pydantic`（演習のエージェント）・`ask`（第1回の素の LLM）・`adk`・`pi`。
- `<qid>` は `q01`〜`q10`（実装は `s1`〜`s6`）、`<k>` は試行の番号（1〜3）。

## 上書きを防ぐために

- **既にある結果は飛ばします。** `run` は、保存先に `t<k>.json` があればその実行を飛ばすので、同じコマンドを何度実行しても上書きされません。止まったときも、同じコマンドで続きから実行できます。
- **結果は、実行が終わったときだけ置かれます。** API のエラーで止まった実行は `t<k>.json` が置かれないので、次に実行したときにやり直されます。
- **やり直したいとき**は、その `t<k>.json` と、同じ名前のファイル（`.trace.txt` など）を別のフォルダに移してから実行します。
- **実行中に `app/`・`adk_agent/`・`compare.py`・`conditions.toml` を書き換えないでください。** 未コミットの変更があると、`run` は止まります。結果がどの commit のものか分からなくなるためです。書き換えたときは、commit してから実行します。
- **試行の途中で変種の中身が変わると止まります。** 1 回目と 2 回目で、モデルに見える指示やツールの説明文が違うと、比べられないためです。
- モデルを切り替えたとき（`AIE_MODEL`）は、結果が `<model>` ごとに別のフォルダに入るので、混ざりません。表も分かれます。

## 準備

### 1. Codespaces を lec03 で開く

lec03 ブランチは作成して push してあります。演習の手順1にある `git switch -c lec03` は要りません。

- 新しく開く場合：GitHub で自分の fork を開き、ブランチを `lec03` に切り替えてから、「Code → Codespaces → Create codespace on lec03」を押す。
- 既に Codespace がある場合：開いてから、ターミナルで次を実行する。

  ```bash
  git switch main
  git pull --no-rebase
  git fetch origin
  git switch lec03
  git pull --no-rebase
  ```

  - `git switch main` で止まる場合は、前の回のブランチに変更が残っている。前の回のブランチで commit してから、もう一度実行する。
  - 最後の `git pull --no-rebase` は、lec03 を最新にするため。

### 2. API キーを確かめる

```bash
[ -n "$GEMINI_API_KEY" ] && echo 設定済み || echo 未設定
```

- 値は表示しない。「未設定」のときは、下の「うまくいかないとき」を見る。

### 3. 依存を確かめる

- Codespace を作ったときに `uv sync --frozen` が実行されている。lec03 では `pyproject.toml` を変えていないので、追加の導入は要らない。
- ADK は、実行のときだけ `uv run --with google-adk==2.13.0` で一時的に足す（`pyproject.toml` と `uv.lock` は変わらない）。初回は ADK の取得に 1 分ほどかかる。

### 4. Pi を入れる（発展(b)）

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.5.0
pi --version
pi --list-models gemini
```

- 版を 1.5.0 に固定するのは、システムプロンプトを比べる版（v1.5.0）とそろえるため。
- `--list-models` の一覧に `gemini-3.5-flash-lite` があることを確かめる。

## 最初の確認（API キー不要）

```bash
uv run python -m app.agent --list-tools                         # 4 つのツールと、その説明文
uv run pytest -q                                                # 講座のテスト
uv run pytest -q lectures/03                                    # 課題のコードと compare.py のテスト
uv run python lectures/03/compare.py variants                   # 変種ごとの差分（どれも 1 か所だけ）
uv run python lectures/03/compare.py run --level all --dry-run  # API を使わない予行。出力は一時フォルダ
uv run python lectures/03/compare.py status                     # 予定・済み・残りの件数と、残りの推定リクエスト数
```

- `--list-tools` に、`make_quiz` と `save_learning_log` が出ることを確かめる。
- `variants` は、変種ごとに「モデルに見える指示とツールの説明文」を `lectures/03/results/variants/` に保存する。baseline との差分がちょうど 1 か所でないときは、警告が出る。警告が出たら、先に進まずに知らせる。
- `--dry-run` は、TestModel（API を呼ばない偽のモデル）で全部の経路を通す。最後に出力先が表示される。repo の `lectures/03/results/` には書かない。

## 実行の順番

最初に、全部の段を 1 回ずつ（`--trials 1`）流して、すべてのレベルの結果をそろえます。そのあと、2・3 回目をまとめて足します。利用枠が途中で尽きても、3 番までで基礎を、6 番までで全レベルを提出できます。

**☆ は実験済み**です（結果は `lectures/03/results/` に置き、[answer.md](answer.md) に記入済み）。

| 順 | 状態 | 内容 | コマンド（`uv run python lectures/03/compare.py` の後） | LLM リクエストの目安 | 時間の目安 |
|---:|---|---|---|---:|---:|
| 0 | | 最初の確認（上の節） | 上の 6 つ | 0 | 約 3 分 |
| 1 | | 演習：CLI の 3 つの質問と、素の LLM（`ask`） | 下の「演習の CLI」と `run --level exercise --trials 1` | 約 20 | 約 3 分 |
| 2 | | 基礎の基準（V0 `baseline`、10 問） | `run --level basic --variant baseline --trials 1` | 約 20 | 約 2 分 |
| 3 | | 基礎の変種 V1〜V3。ここまでで基礎を提出できる | `run --level basic --variant no_search_rule,query_many,ending_quiz --trials 1` | 約 30 | 約 3 分 |
| 4 | | 実装（s1〜s6） | `run --level impl --trials 1` | 約 20 | 約 2 分 |
| 5 | | 発展(a) ADK（q01〜q04） | 下の「adk run を手で 1 回」と `run --level adv-a --trials 1` | 約 15 | 約 4 分 |
| 6 | | 発展(b) Pi（q01・q02・q04） | 下の「Pi を手で 1 回」と `run --level adv-b --trials 1` | 約 25 | 約 5 分 |
| 7 | | 基礎の変種 V4・V5 | `run --level basic --trials 1` | 約 20 | 約 2 分 |
| 8 | | 2・3 回目（全部） | `run --level all --trials 3` | 約 270 | 約 25 分 |

- 1 巡で約 135、3 巡で約 400 リクエスト（再試行を含めると約 450）。時間は、実行の間の待ち（既定 8 秒）を含めた見積もりで、429 で待つと延びる（1 巡目の実測で直す）。
- `--variant`・`--question`・`--engine` には、名前を `,` で区切って複数指定できる。
- 7 番の `run --level basic --trials 1` は、V0〜V5 を順に回す。V0〜V3 は済んでいるので飛ばされ、V4・V5 だけが実行される。
- 1 番の `--level exercise` は、`ask`（q01・q02・q04）と、V0 の q01〜q03 を実行する。V0 の q01〜q03 は 2 番と同じ結果を使うので、2 番では残りの 7 問だけが実行される。
- 任意の変種 V6・V7 は、8 番のあとに `run --level basic --include-optional --trials 3` で足す。
- 各段の終わりに `status` で「済み」が増えたことを確かめ、下の「結果の保存と受け渡し」で push する（段ごとに push すると、途中の結果から answer.md に書き写せる）。

### 演習の CLI（1 番）

演習の手順2・3・5 のコマンドを、課題文どおりに CLI でも 1 回ずつ実行し、出力を残します。

```bash
mkdir -p lectures/03/results/cli
uv run python -m app.agent --list-tools | tee lectures/03/results/cli/list_tools.txt
uv run python -m app.agent "AIE2026の講座の宿題にある「3レベル」とは何ですか？" | tee lectures/03/results/cli/step2.txt
uv run python -m app.agent --trace "LoRAとQLoRAの違いは？" | tee lectures/03/results/cli/step3.txt
uv run python -m app.agent --trace "第2回の内容から確認問題を1問出して" | tee lectures/03/results/cli/step5.txt
uv run pytest -q | tee lectures/03/results/cli/pytest.txt
```

- 手順2：回答に「基礎・実装・発展」が出て、最後に関係する回（第1回）が示されているか。
- 手順3：`[tool call]` に `search_glossary` が `LoRA` と `QLoRA` で 2 回出ているか。`[usage]` の `requests` と `tool_calls` の値。
- 手順5：`[tool call] make_quiz({'lecture': 2})` のような行が出ているか。
  - **出ないときは、ここで止めて知らせる。** 課題文の対処（INSTRUCTIONS に「確認問題を求められたら make_quiz を使う」を足す）をすると、基礎の基準が変わるため。
- `| tee` は、画面に出しながら、ファイルにも書く。

### 発展(a)：adk run を手で 1 回（5 番）

課題文どおり、`adk run` の対話でも 1 回動かします。

```bash
PYTHONPATH=. uv run --with google-adk==2.13.0 adk run adk_agent/glossary_agent
```

- `[user]:` のあとに、演習と同じ質問（例：`AIE2026の講座の宿題にある「3レベル」とは何ですか？`）を入れて Enter を押す。終わるときは `exit` と入力する。
- 画面の出力は、選んでコピーし、`lectures/03/results/cli/adk_run.txt` に貼って残す（対話なので `tee` は付けない）。
- 対話の `adk run` は、会話を `adk_agent/glossary_agent/.adk/` に保存する。git では無視する設定（`adk_agent/.gitignore`）なので、commit されず、`compare.py` も止まらない。`compare.py` の ADK は `--in_memory` で動くので、このフォルダは使わない。
- そのあと、比較用スクリプトでは、まず 1 問だけ流して、記録の形を確かめる。

  ```bash
  uv run python lectures/03/compare.py run --level adv-a --question q01 --trials 1
  uv run python lectures/03/compare.py summarize
  ```

  - `summary.md` の ADK の行に、ツール呼び出しと requests が入っていれば、残りを `run --level adv-a --trials 1` で流す。
  - 入っていないときは、ADK の生の出力（`t1.raw.jsonl`）の形が想定と違う。そのまま push して知らせる（生の出力から、解析を直せる）。

### 発展(b)：Pi を手で 1 回（6 番）

Pi は、ファイルの書き換えやコマンドの実行を、確認を求めずに行います。必ず Codespaces の中で試します。

```bash
git status --short   # 何も出ない（作業ツリーに変更が無い）ことを確かめる
pi
```

- Pi の中で `/model` と入力し、Gemini のモデル（`gemini-3.5-flash-lite`）を選ぶ。API キーは `GEMINI_API_KEY` がそのまま使われる。
- 質問を 1 つ投げてみる（例：`data/glossary.json を読んで、AIE2026の宿題の「3レベル」を教えて。ファイルは変更しないで`）。
- 終わるときは `/quit` と入力する（入力欄が空のときの Ctrl+D でも終わる）。
- 終わったら、もう一度 `git status --short` を実行する。何か出たら、`git diff` で中身を見てから、`git restore <ファイル>` で戻す。
- 比較用スクリプトは、Pi を読み取り専用のツール（`read,grep,find,ls`）だけで、`data/` だけを複製した一時フォルダで動かす。repo のファイルは書き換えられない。`problem.md` や結果を読んで答えを写すこともない。ADK と同じく、まず 1 問だけ流して形を確かめる。

  ```bash
  uv run python lectures/03/compare.py run --level adv-b --question q01 --trials 1
  uv run python lectures/03/compare.py summarize
  ```

  - `summary.md` の Pi の行に、ツール呼び出し（`read`・`grep` など）と回答が入っていれば、残りを `run --level adv-b --trials 1` で流す。

## 各段で確かめること

段が終わるたびに `summarize` を実行し、`lectures/03/results/summary.md` の表に「未実行」以外の値が入ったことを確かめます。段ごとの見どころは次のとおりです（比べた結果は answer.md に書きます）。

| 順 | 見るところ |
|---:|---|
| 1 | `ask` の q01 の回答に「基礎・実装・発展」が無く、「初級・中級・上級」のような推測があるか（第1回の記録と同じか）。V0 の q01〜q03 で、検索と `make_quiz` が呼ばれたか |
| 2 | V0 の q02 で `search_glossary` が 2 回、requests が 2 前後か。q08 で `get_lecture_summary(4)` が呼ばれたか |
| 3 | 表A の、検索の回数・引数・requests が、V0 とどう違うか。V3 で、最後の第N回が無くなったか |
| 4 | s1 で保存されたファイルが 1 つ（JSON の保存されたファイルの欄）、s2 で 0、s5 で 1 か。s3・s4 で、保存が呼ばれていないか |
| 5・6 | 最初の 1 問で、ツール呼び出しと回答が表に入ったか（上の各節） |
| 7 | V1 と V5 のどちらで、検索が減ったか |

## 利用枠

- Gemini API の無料枠には、1 分あたりと 1 日あたりの上限がある。値は AI Studio の rate limit の画面で確かめる。上限はプロジェクトごとなので、API キーを増やしても増えない。
- 1 日の上限は、太平洋時間の 0 時（日本時間 17 時）に戻る。
- 1 分あたりの上限（429）に当たると、スクリプトが待ってから自動で再試行する。
- 1 日の上限（エラーの本文に `PerDay`）に当たると、「明日、同じコマンドで続きから」と表示して止まる（終了コード 2）。そこまでの結果を push しておき、日本時間 17 時以降に同じコマンドを実行する。
- 実行の間の待ちは `--interval`（秒、既定 8）で、1 回に流す件数は `--max-runs` で変えられる。

## 結果の保存と受け渡し

段ごとに、次の順で push します。

```bash
uv run python lectures/03/compare.py summarize
grep -rqF -- "$GEMINI_API_KEY" lectures/03/results && echo "NG: API キーが含まれています" || echo "OK"
git add lectures/03/results
git commit -m "Lecture 03: Record <段の名前> results"
git pull --no-rebase
git push
```

- `grep` は、結果に API キーが含まれていないかの確認（`GEMINI_API_KEY` が設定されている Codespace で実行する）。スクリプトも書き出す前に伏せ字にしているが、念のため確かめる。NG が出たら、commit せずに知らせる。
- `git add -A` は使わない。関係ないファイルが入らないよう、`lectures/03/results` だけを add する。
- commit のメッセージは、PR のほかのコミットに合わせて英語にする（例：`Lecture 03: Record exercise and baseline results`）。
- push したら、Claude に「push した」と伝える。こちらで pull して、answer.md に書き写す。

## うまくいかないとき

<details><summary>症状と対処の一覧</summary>

| 症状 | 対処 |
|---|---|
| 「レート制限に達しました」・429 で待ちが続く | スクリプトが自動で待って再試行する。続くときは `--interval 20` のように待ちを長くする |
| 1 日の上限で止まった（終了コード 2） | そこまでの結果を push し、日本時間 17 時以降に、同じコマンドで続きから実行する |
| 503（モデルが混み合っている）が続く | 時間をおいて再実行する。それでも続くときは、`AIE_MODEL=gemini-3.1-flash-lite` を付けて実行する（スライドの対処）。ただし、結果は別のモデルのフォルダに入り、表も分かれる。その巡は全部の条件を同じモデルでやり直すことになるので、切り替える前に知らせる |
| 「環境変数 GEMINI_API_KEY が見つかりません」 | GitHub の Settings → Codespaces → Secrets で、`GEMINI_API_KEY` の Repository access に自分の fork が入っているかを確かめる。登録や変更のあとは、Codespace を開き直す（左下の「Codespaces」→「Stop Current Codespace」→ もう一度開く） |
| 未コミットの変更があると言われて止まった | `git status` で、`app/`・`adk_agent/`・`compare.py`・`conditions.toml` の変更を確かめる。意図した変更なら commit してから実行する。意図しない変更なら `git restore <ファイル>` で戻す。`??` の行は、git が追跡していないファイル（実行で作られたものなど）なので、中身を確かめて、不要なら消す。`adk_agent/glossary_agent/.adk/` が出たときは、対話の `adk run` が作った会話のフォルダなので、`git pull --no-rebase` で lec03 を最新にすれば無視される（消さなくてよい） |
| `variants` で、差分が 1 か所でないと警告が出た | 先に進まずに知らせる（変種の定義を直す） |
| 手順5で `make_quiz` が呼ばれない | INSTRUCTIONS を書き換える前に知らせる（基礎の基準が変わるため） |
| ADK の初回が遅い | 初回は ADK の取得に 1 分ほどかかる。2 回目からは速い |
| ADK・Pi の行に、ツール呼び出しや回答が入らない | 生の出力（`t<k>.raw.jsonl`）の形が想定と違う。そのまま push して知らせる。生の出力から解析を直し、`summarize` だけで表を作り直せる |
| `pi: command not found` | 「Pi を入れる」の `npm install` をやり直す |
| Pi でモデルが見つからない | `pi --list-models gemini` で ID を確かめ、`--pi-model <ID>` を付けて実行する |
| やり直したい実行がある | その `t<k>.json` と、同じ名前のファイルを別のフォルダに移してから、同じコマンドを実行する |

</details>
