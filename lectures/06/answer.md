# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：
- STEP 2 の [tool call]：
- STEP 3 の [tool call]：
- STEP 4 の [tool call]：
- 気づき・疑問：

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：基礎・実装・発展（すべて）
- やったこと：
- 結果：
- 気づき・疑問：

### 発展：Claude Code の MCP の設定を、演習の設定ファイルと比べる

- 調べた日：2026-11-26
- 調べたソフトウェア：Claude Code（手元の版は 2.1.327）
- 参照した公式ドキュメント：
  - Connect Claude Code to tools via MCP：https://code.claude.com/docs/en/mcp
  - Configure permissions：https://code.claude.com/docs/en/permissions

#### 調べた 4 つのこと

| 調べること | 演習（`app/mcp_servers.json`） | Claude Code |
|---|---|---|
| 設定を書く場所 | `app/mcp_servers.json` の 1 か所。読むのは `app/mcp_agent.py` だけ | `claude mcp add` で登録する。書き込む先はスコープで決まる。local と user は `~/.claude.json`、project はプロジェクト直下の `.mcp.json` |
| コマンドで起動する書き方 | `"glossary": {"command": "uv", "args": ["run", "python", "-m", "app.mcp_server"]}` | `claude mcp add glossary -- uv run python -m app.mcp_server`。`--` より後ろが、そのままサーバーのコマンドと引数になる。`.mcp.json` に書くときは、演習と同じ `"command"` と `"args"`（`"type"` を書かなければ stdio） |
| URL で接続する書き方 | `"aws": {"url": "https://knowledge-mcp.global.api.aws"}` | `claude mcp add --transport http aws https://knowledge-mcp.global.api.aws`。`.mcp.json` では `{"type": "http", "url": "…"}`。`"type"` の無い `"url"` だけの項目は、エラーとして読み飛ばされる |
| 有効になる範囲 | このリポジトリのエージェント（`app/mcp_agent.py`）だけ | local（既定）：このプロジェクトで、自分だけ。project：このプロジェクトを使う全員（`.mcp.json` を commit して共有する）。user：自分のすべてのプロジェクト。同じ名前が複数のスコープにあれば、local → project → user の順に 1 つだけを使い、項目は混ぜない |

#### 4 つのほかに違いが目立ったこと

| | 演習 | Claude Code |
|---|---|---|
| ツールの名前 | `glossary_search_glossary`（サーバー名 + `_` + ツール名） | `mcp__glossary__search_glossary`。権限のルール（allow・deny）にも、この名前を書く |
| 使う前の承認 | ない。設定に書けば、次の起動からそのまま使う | 対話で使うときは、project スコープ（`.mcp.json`）のサーバーを使う前に承認を求める。リポジトリに入った設定で、そのリポジトリのサーバーを承認することもできない（信頼の確認を受け入れるまで、承認待ちのまま）。ただし `claude -p`（対話しない実行）では、承認を求めずに読み込む |
| 環境変数 | `"env"` に書いたものが、既定の数個（HOME・PATH など）に足される | `${VAR}` と `${VAR:-既定値}` を展開する。リモートのサーバーの `url` と `headers` では、一部の資格情報の変数（Claude 自身の `ANTHROPIC_API_KEY` など）を、設定されていても空として扱う |

#### 手元で確かめたこと

用語集サーバーを local スコープで一度だけ登録し、接続できることを確かめてから、すぐ削除した。fetch と AWS は登録していない。

```text
$ claude mcp add --scope local glossary -- .venv/bin/python -m app.mcp_server
Added stdio MCP server glossary with command: .venv/bin/python -m app.mcp_server to local config
File modified: ~/.claude.json [project: ~/Sites/basi/lecture-ai-engineering-2026]

$ claude mcp get glossary
glossary:
  Scope: Local config (private to you in this project)
  Status: ✔ Connected
  Type: stdio
  Command: .venv/bin/python
  Args: -m app.mcp_server
  Environment:

$ claude mcp remove glossary -s local
Removed MCP server glossary from local config
```

- `~/.claude.json` の、このプロジェクトの欄に、`"glossary": {"type": "stdio", "command": ".venv/bin/python", "args": ["-m", "app.mcp_server"], "env": {}}` と書かれた。演習の設定と同じ形に、`"type"` と空の `"env"` が足されている。
- 手元の Mac には uv が無いので、コマンドは `uv run python` ではなく `.venv/bin/python` にした。確かめたかったのは書き方と範囲なので、どちらでも同じ。
- `claude mcp get` は、登録の前にも、ほかに登録しているサーバーの名前を表示した。結果を残すときは、glossary の行だけにした。

#### 気づき・疑問

- 書き方は、どちらも `"mcpServers"` の JSON で、ほぼ同じ。違うのは、どこに置くかと、誰に効くか。演習の設定はリポジトリに入るので、Claude Code の project スコープに近い。
- Claude Code は、リポジトリに入っている設定（`.mcp.json`）を、すぐには信用しない。対話で使うときは、使う前に本人が承認する（`claude -p` は例外）。演習のエージェントには、この段がない。設定ファイルに書いたサーバーは、次の起動でそのまま起動する。演習の補足にあった「出所の分からないサーバーは、設定ファイルに書きません」を、Claude Code は仕組みでも支えている。
- URL の書き方は、演習では `"url"` だけでよいが、Claude Code の `.mcp.json` では `"type": "http"` が要る。演習の設定をそのまま `.mcp.json` に写すと、aws の項目は読み飛ばされる。
- 疑問：演習のエージェントに承認の段を足すなら、どこに置くのがよいか。第3回の `save_learning_log` で使った、呼ぶ前に利用者の承認を求める仕組み（requires_approval）を、MCP のツールにも使えるのか。

<details>
<summary>仮説と照合</summary>

### 仮説（実行前に記録）

2026-11-26、実行前に記録した。回数は、`lectures/06/run.py` で流す回数（段 B は各 2 回、段 C は各 3 回）に対するもの。

**演習**

1. STEP 1 の 1 つ目（`app.agent --list-tools`）は、search_materials・make_quiz・save_learning_log の 3 つを表示する。upstream の例（search_materials だけ）と違うのは、fork の `app/agent.py` に第3回のツールが残っているから。2 つ目は glossary_search_glossary の 1 つだけ。
2. STEP 2 は、2 回とも glossary_search_glossary を呼び、引数は「3レベル」を含む。モデル応答は 2 回。INSTRUCTIONS の 1 行目（答える前に、必ず search_glossary で用語集を確認する）と、用語集の別名「3レベル」があるから。
3. STEP 3 は、2 回とも fetch_fetch を、質問の URL で呼ぶ。INSTRUCTIONS の最後の行（Web ページの内容には、合うツールがあれば使う）があるから。また、2 回中 1 回以上は、search_glossary も呼ぶ。MCP は用語集にある語で、INSTRUCTIONS の 1 行目が「必ず」と言うから。
4. STEP 4 は、2 回とも `aws_` で始まる検索のツールを呼ぶ。fetch と aws のツールの説明文は、英語のまま一覧に出る。

**基礎**

1. エージェント側を変えずに、一覧に glossary_get_lecture_summary が増える（glossary のツールが 1 個から 2 個になる）。
2. 「第2回では何を学びますか？」で、2 回とも glossary_get_lecture_summary を `lecture=2` で呼ぶ。INSTRUCTIONS の「回の内容を聞かれたら get_lecture_summary を使う」と、説明文の例（「第3回では何を学ぶ？」）があるから。
3. `uv run pytest -q` は、基礎の時点（S1）でも、実装を足した時点（S2）でも、すべて通る。upstream のテストは、ツールが「含まれる」かで判定しているから。

**実装：段 A（API を使わずに決まる部分。ローカルのテストで確かめ済みで、Codespaces でも同じになるはず）**

1. return：第99回・第5回のどちらも、isError は false で、本文は理由の文そのもの。モデルにも、ツールの結果として全文が渡る。
2. ValueError：本文は `Error executing tool make_quiz` だけになり、理由の文はサーバーのログ（stderr）にしか残らない。モデルに渡るのは、そのあとに「Fix the errors and try again.」が付いた文。
3. ToolError：本文は `Error executing tool make_quiz: ` と理由の文。モデルには、そのあとに「Fix the errors and try again.」が付く。
4. schema：第99回は、関数に届く前に、サーバーが英語の検証エラーを返す（出題できる回の案内は無い）。第5回は範囲内なので、return と同じ文を返す。
5. raise の 2 条件と schema（第99回）は、同じツールが 2 回続けて失敗すると、UnexpectedModelBehavior で実行が止まる。

**実装：段 C（Gemini の反応）**

1. return（第99回）：3 回とも make_quiz(99) を 1 回だけ呼び、呼び直さない。回答では、第99回は無いことと、出題できる回（1, 2, 3, 4, 6）を伝え、利用者に選んでもらう。文の最後に「利用者に、この中から回を選んでもらってください」と書いたから。
2. ValueError（第99回）：理由が届かないので、3 回中 2 回以上は、別の回で呼び直すか、もう一度 99 で呼んで止まる。止まらずに答えた回も、第99回が無いことは伝えられない。
3. ToolError（第99回）：理由は届くので、3 回とも、第99回が無いことは伝える。ただし「Fix the errors and try again.」が付くので、3 回中 1 回以上は、利用者に聞かずに別の回で呼び直す。
4. schema（第99回）：Gemini の関数呼び出しはスキーマを守らせるモード（VALIDATED）なので、99 がサーバーに届くのは、3 回中 1 回以下。届かない回は、make_quiz を呼ばずに答えるか、範囲内の値に置き換えて呼ぶか、エラーになる。
5. return（第5回）：3 回とも make_quiz(5) を呼び、第5回の用語がまだ無いことと、出題できる回を伝える。呼び直さない。
6. raise の 2 条件は、呼び直しのぶん、return よりモデル応答の回数が多い（中央値で 1 回以上）。

**セキュリティ**

1. stdio のサーバー（子）に見える環境変数は、既定の数個（HOME・LOGNAME・PATH・SHELL・TERM・USER）のうち、クライアントにあるもの（HOME・PATH）だけで、目印（AIE_LEC06_CANARY）は無い。
2. Codespaces（Linux）では、子から親の `/proc/PID/environ` を読め、目印が見える。祖先の environ も開ける。環境変数を渡さないことは、秘密を守る境界にはならない。
3. `run.py --scan` は、results に秘密情報らしき文字列を見つけない。

### 照合

| # | 仮説 | 結果 | 照合 |
|---|---|---|---|
| 演習 1 | STEP 1 の 1 つ目は 3 つ、2 つ目は 1 つ | | |
| 演習 2 | STEP 2 は 2 回とも glossary_search_glossary（「3レベル」） | | |
| 演習 3 | STEP 3 は 2 回とも fetch_fetch。1 回以上 search_glossary も | | |
| 演習 4 | STEP 4 は 2 回とも `aws_` の検索 | | |
| 基礎 1 | 一覧に glossary_get_lecture_summary が増える | | |
| 基礎 2 | 2 回とも get_lecture_summary(2) | | |
| 基礎 3 | S1・S2 とも pytest がすべて通る | | |
| 段 A 1〜5 | 本文とモデルに渡る文、2 回の失敗で止まる | | |
| 段 C 1 | return：3 回とも呼び直さず、無い回と出題できる回を伝える | | |
| 段 C 2 | ValueError：2 回以上、呼び直すか止まる | | |
| 段 C 3 | ToolError：3 回とも無いと伝え、1 回以上は呼び直す | | |
| 段 C 4 | schema：99 が届くのは 1 回以下 | | |
| 段 C 5 | return（第5回）：3 回とも用語が無いことを伝える | | |
| 段 C 6 | raise はモデル応答の回数が多い | | |
| セキュリティ 1〜3 | 子に目印は無い／親からは読める／走査で見つからない | | |

</details>
