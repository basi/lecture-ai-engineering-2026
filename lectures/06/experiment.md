# 第6回 演習・宿題の実験手順（GitHub Codespaces）

演習（STEP 1〜4）と宿題（基礎・実装）は、`lectures/06/run.py` でまとめて 1 回流す。発展（Claude Code の MCP の設定との比較）は手元の Mac で済ませ、`answer.md` に書いた。仮説は、流す前に `answer.md` の「仮説と照合」に記録した。

## 使うファイル

| もの | 場所 | 説明 |
|---|---|---|
| 実行スクリプト | `lectures/06/run.py` | すべての項目を順に流し、結果を `lectures/06/results/` に保存する。ほかに `--list`（流す項目と、済んだかどうか）・`--scan`（秘密情報の走査）・`--summarize`（集計）がある |
| 宿題の実装 | `app/mcp_server.py` | 基礎：`get_lecture_summary` を公開する。実装：`make_quiz` を公開する。出題できない回は、理由と出題できる回を return で返す |
| 設定 | `app/mcp_servers.json` | STEP 3・4 のとおり、fetch と aws を足した |
| 比較用のサーバー | `lectures/06/quiz_server.py` | `make_quiz` の返し方だけを変えた 3 つの変種（`valueerror`・`toolerror`・`schema`）。ツールの名前・説明文・引数は `app/mcp_server.py` と同じ |
| 環境変数の確認用のサーバー | `lectures/06/env_probe.py` | 子プロセスに見える環境変数の名前と、目印があるかだけを返す。値は返さない |
| テスト | `lectures/06/test_lec06.py`・`lectures/06/test_run.py` | API キーもネットワークも使わない。`uv run pytest -q lectures/06` で流す |
| 結果の置き場所 | `lectures/06/results/` | 1 回の実行ごとに 1 ファイル（下の「結果の置き場所」） |

## 外へ送られるもの

MCP のサーバーは、エージェントの外のプログラムやサービスにつながる。この実行で外へ送るのは、次の 4 か所だけ。

| 送り先 | 送るもの | 送る項目 |
|---|---|---|
| Gemini API | 指示文、質問、ツールの一覧（名前・説明文・引数）、ツールの結果（用語集の文、fetch が取ったページの本文、AWS の検索結果） | 段 B・段 C |
| PyPI | `mcp-server-fetch==2026.11.11` と、それが使うパッケージの取得（`uvx` が行う） | fetch を最初に起動する項目（段 A の `step3_tools`） |
| fetch が取得するページ | ページの取得（STEP 3 の URL。ほかの項目でも、モデルが選んだ URL を取得することがある） | 設定に fetch がある、段 B の項目 |
| AWS Knowledge MCP Server | 接続とツールの一覧の取得。STEP 4 では、モデルが作った検索の語 | 設定が STEP 4 の、ツール一覧と CLI の項目（段 A の `step4_tools`・`basic_tools`・`impl_tools`、段 B の `step4`・`basic`・`impl`） |

秘密を外へ出さないために、run.py は次のようにする。

- 子プロセスには、名前に TOKEN・SECRET・PASSWORD・KEY・CREDENTIAL を含む環境変数を渡さない（`GITHUB_TOKEN` など）。
- `GEMINI_API_KEY` は、LLM を呼ぶ子プロセス（段 B）にだけ渡す。段 C は、run.py の中で LLM を呼ぶ。
- 結果は `--scan` で走査し、通ったときだけ commit する（下の「手順」の 3）。走査では、名前に TOKEN・KEY・SECRET・PASSWORD を含む環境変数の値と、Google API キー・GitHub トークンの形を探す。見つけても、値は表示しない。

## 第三者のサーバーについて

- fetch は第三者のコードで、自分のプログラムと同じユーザーで動く。
  - 版は problem.md と同じ `2026.11.11` に固定する。オプションは、problem.md の `--no-readability-js` のほかに足さない。
  - 子プロセスに環境変数を渡さなくても、同じユーザーのプロセスは、親や祖先のプロセスの環境変数を `/proc` から読める。段 A の `env_probe` で、これを確かめる。環境変数を絞っても、秘密は守れない。そのため、信頼できるサーバーだけを設定に書き、終わったら Codespace を停止する。
- aws は URL で接続するので、版を固定できない。そこで `--summarize` で、実行の間にツールの定義が変わっていないかを比べる。
- fetch の起動と AWS への接続は、Codespaces だけで行う。手元の Mac で起動したのは、自分で書いたサーバー（glossary・quiz_server・env_probe）だけ。
- 次のものは使わない。
  - glossary の `--http`（ポートを開かない）。
  - 副作用のあるツール（`save_learning_log`）の、MCP での公開。`app/mcp_agent.py` には、ツールを呼ぶ前に承認を求める仕組みがないため。

## 流す項目

全 38 項目を、段 A → 段 B → 段 C の順に流す。`uv run python lectures/06/run.py --list` で、項目の一覧と、済んだかどうかを見られる（何も起動しない）。

| 段 | 項目の数 | 内容 | API |
|---|---:|---|---|
| A | 14 | ツール一覧（状態ごと）、pytest（基礎と実装の時点）、`make_quiz` の直接の呼び出し（4 条件 × 第99・5・2回）、env_probe | 使わない |
| B | 9 | 講座の CLI（`app.mcp_agent --trace`）。STEP 2〜4 と基礎の質問を各 2 回、実装の質問（第99回）を 1 回 | 使う |
| C | 15 | 実装の比較。同じエージェントで、`make_quiz` の返し方 4 条件を比べる（下の表） | 使う |

- LLM へのリクエストは 70〜100 回の見込み。1 日の上限（500 回）に収まる。
- 時間は 15〜30 分の見込み。LLM を呼ぶ項目の間は、5 秒あける。

### 状態（流している間だけ、2 つのファイルを書き換える）

| 状態 | `app/mcp_server.py` | 使う項目 |
|---|---|---|
| S0 | 演習の開始時点（dccf422）。公開するのは `search_glossary` だけ | STEP 1〜4 |
| S1 | S0 に、基礎の 1 行（`get_lecture_summary`）を足したもの | 基礎 |
| S2 | HEAD（実装の `make_quiz` まで） | 実装、段 C |

| 設定 | `app/mcp_servers.json` |
|---|---|
| STEP 1 | glossary だけ（dccf422 の版） |
| STEP 3 | glossary と fetch |
| STEP 4 | glossary・fetch・aws（HEAD の版） |

- 終わるときは、2 つのファイルを HEAD の内容に戻し、sha256 で確かめる。Ctrl+C で止めたときや、ターミナルを閉じたときも同じ。
- Codespace ごと止まったときは、次に流したときに、途中の状態から自動で戻す。

### 段 C の条件と質問

| 条件 | 出題できない回のとき | 第99回のクイズを作って | 第5回のクイズを作って |
|---|---|---:|---:|
| return（`app/mcp_server.py` そのもの） | 理由の文を return で返す | 3 回 | 3 回 |
| valueerror | 同じ文で `raise ValueError` | 3 回 | 段 A だけ |
| toolerror | 同じ文で `raise ToolError` | 3 回 | 段 A だけ |
| schema | 引数の範囲を `Field(ge=1, le=13)` にする | 3 回 | 段 A だけ |

- 第99回は、存在しない回。第5回は、回はあるが、用語集に用語がない回。
- 条件の順は、試行ごとに入れ替える（試行 1 は return → valueerror → toolerror、試行 2 は valueerror → toolerror → return、試行 3 は toolerror → return → valueerror）。
- schema は、Gemini が引数の範囲を守らせる影響を見る条件。return と raise の比較とは分けて報告する。

## 手順

新しい Codespace のときは、作成後の準備（uv の導入と `uv sync`）が終わってから始める。

```bash
# 1. 確かめる（どちらも、何も出ないこと）
git status --short
git branch --list lec06
```

- `git status --short` に、`??`（追跡していないファイル）以外の行が出たら、そこで止める。自分の作業なら、commit するか、`git stash` で退避してから進む。
- `git branch --list lec06` に何か出たら、それは授業で作った lec06。`git branch -m lec06 lec06-class` で名前を変えてから進む。

```bash
# 2. lec06 に切り替えて、流す（15〜30 分）
git fetch origin && git switch lec06 && git pull --ff-only && uv sync --frozen
uv run python lectures/06/run.py
```

- 流している間に Codespace が止まっても、もう一度流せば、続きから流す（済んだ項目は飛ばす）。

```bash
# 3. 走査が通ったときだけ、結果を commit して push する
uv run python lectures/06/run.py --scan && git add lectures/06/results && git commit -m "Lecture 06: Add results" && git pull --no-rebase --no-edit && git push
```

push したら、Codespace を停止する。

### 2 の終わり方と、次にすること

| 最後に出る文 | 終了コード | 次にすること |
|---|---:|---|
| すべての項目が終わりました | 0 | 3 を実行する |
| 失敗した項目が N 件あります | 1 | `uv run python lectures/06/run.py` を、もう一度実行する（失敗した項目だけをやり直す）。それでも失敗が残ったら、3 を実行してから、失敗の一覧を知らせる |
| Gemini API の 1 日の上限に達したとみなして、止めます | 2 | 3 を実行する。日本時間 17 時（1 日の上限が戻る時刻）を過ぎてから、`uv run python lectures/06/run.py` と 3 を、もう一度実行する |
| Ctrl+C で止めました／終了の合図を受けて、止めます | 130 など | `uv run python lectures/06/run.py` を、もう一度実行する |
| 流す前の確認で止まった（ブランチ、commit していない変更、API キーなど） | 1 | 表示のとおりに直してから、`uv run python lectures/06/run.py` を、もう一度実行する（下の「困ったとき」） |

- もう一度流すと、済んだ項目は飛ばして、続きから流す。失敗した項目は、やり直す。

## 知らせてほしいとき

知らせるのは、次の 2 つのときだけ。

- 3 で push したとき。「push した」の一言でよい。失敗の一覧が出ていたら、その行も添える。
- 3 の `--scan` が「秘密情報らしき文字列が見つかりました」と表示したとき。このときは commit されない。表示された行を貼る（ファイルの名前と、形の種類または環境変数の名前だけが出る。値は出ない）。

## 困ったとき

| 表示・状況 | すること |
|---|---|
| origin/lec06 が見つかりません | `git fetch origin` のあと、もう一度流す |
| HEAD（…）が origin/lec06（…）と違います | `git switch lec06` と `git pull --ff-only` のあと、もう一度流す。授業で作った lec06 が残っているときは、先に名前を変える（手順の 1） |
| 環境変数 AIE_MODEL が … になっています | `unset AIE_MODEL` のあと、もう一度流す |
| app/ か lectures/06/ に、commit していない変更があります | `git status` で確かめる。自分の作業なら、commit するか `git stash` で退避してから、もう一度流す |
| … に、このスクリプトが作る状態とは違う変更があります | 表示のとおり、commit するか `git checkout HEAD -- <ファイル>` で捨ててから、もう一度流す |
| uv と uvx が見つかりません | Codespace の準備が終わるのを待ち、新しいターミナルで、もう一度流す |
| 環境変数 GEMINI_API_KEY が見つかりません | 表示のとおり、Codespaces のシークレットに登録し、Codespace を再起動してから、もう一度流す |
| HEAD の内容に戻せませんでした | 表示された `git checkout HEAD -- app/mcp_server.py app/mcp_servers.json` を実行してから、知らせる |
| 段 C で、エラーのトレースバックが画面に出る | 想定どおり（raise の条件で、サーバーが出す表示）。そのまま待つ |
| 同じ項目のまま、10 分以上進まない | Ctrl+C で止めて、もう一度流す。また止まったら知らせる |
| 3 の push だけが失敗した | `git pull --no-rebase --no-edit && git push` を、もう一度実行する。また失敗したら知らせる |

## 結果の置き場所

```text
lectures/06/results/
├── run.log                                実行の記録（画面に出る進行の行と同じ）
├── cli/<項目>.txt                         段 A のツール一覧と pytest、段 B の CLI の出力。見出しに、コマンド・状態・sha256・時刻・終了コード
├── direct/<条件>.json                     段 A の直接の呼び出し。ツールの定義、第99・5・2回の本文、モデルに渡る文
├── direct/<条件>.server.log               そのときのサーバーの stderr（ValueError の理由は、ここにだけ残る）
├── direct/<条件>.agent.server.log         偽のモデルで、エージェントを通したときのサーバーの stderr
├── security/env_probe.json                env_probe の結果（環境変数の名前と、真偽だけ）
├── impl/configs/<条件>.json               段 C の、条件ごとの設定
├── impl/<条件>/<質問>/t<試行>.json         段 C の 1 回分。メッセージ全体、ツールの呼び出し、例外、トークン数
├── impl/<条件>/<質問>/t<試行>.trace.txt    同じ回の trace。ツールの結果は全文で、モデルに渡る文も出す
├── …/<項目>.error.txt                     失敗した項目の記録（やり直して成功すると消える）
└── summary.md                             手元で --summarize が作る集計（Codespaces では作らない）
```

- 結果のファイルがある項目は、済みとして飛ばす。成功した結果だけを、本来の名前で保存する。

## 進み具合

| 段階 | 状態 |
|---|---|
| 準備（コード・テスト・仮説） | 済み |
| Codespaces での実行（段 A・B・C） | 済み（2026-11-27。全 38 項目、やり直し 0 回、走査も通った） |
| 結果の書き写し（answer.md） | 済み（`run.py --summarize` で `results/summary.md` も作った） |
