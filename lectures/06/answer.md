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
