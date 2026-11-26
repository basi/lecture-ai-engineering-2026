"""第6回: 用語集と各回の要点を MCP サーバーとして公開する。

使い方:
    uv run python -m app.mcp_agent --list-tools     # このサーバーが公開するツールを一覧する（APIキー不要）
    uv run python -m app.mcp_server --http          # Streamable HTTP で起動する（http://127.0.0.1:8000/mcp）

引数なしで実行すると stdio（標準入出力）で待ち受ける。これは MCP クライアント（app.mcp_agent など）が
サブプロセスとして起動する使い方なので、手で実行すると入力待ちのまま止まって見える。

第3回では tools.py の関数を agent.py に直接登録した。MCP サーバーにすると、同じ関数を
エージェントのプログラム（app/mcp_agent.py）からも、MCP に対応したほかのソフトウェアからも、
共通の手順で使えるようになる。
"""

import argparse

from mcp.server.mcpserver import MCPServer

from app import tools

# サーバーの名前と説明。クライアントに表示される（設定ファイル側のキー "glossary" とは別物）
server = MCPServer(
    "aie2026-glossary",
    instructions="AIE2026（AIエンジニアリング実践講座）の用語集と各回の要点を提供する。",
)

# 第3回と同じ関数を、そのまま MCP のツールとして公開する。docstring と型ヒントが説明文になる
server.tool()(tools.search_glossary)

# TODO(宿題・基礎): get_lecture_summary も公開しよう
server.tool()(tools.get_lecture_summary)

# 宿題・実装: 第3回の make_quiz を公開する。出題できない回は、理由と出題できる回を return で返す
LAST_LECTURE = 13  # 講座は全 13 回


def quiz_unavailable(lecture: int) -> str | None:
    """確認問題を出せない回なら、その理由と出題できる回をまとめた文を返す。出せる回なら None を返す。"""
    available = sorted({e["lecture"] for e in tools.load_glossary()})
    if not 1 <= lecture <= LAST_LECTURE:
        reason = f"第{lecture}回はありません。この講座は第1回〜第{LAST_LECTURE}回です。"
    elif lecture not in available:
        reason = f"第{lecture}回の用語は、まだ用語集にありません。"
    else:
        return None
    options = ", ".join(map(str, available))
    return f"{reason}確認問題を出せる回: {options}。利用者に、この中から回を選んでもらってください。"


@server.tool(description=tools.make_quiz.__doc__)  # 説明文は第3回の make_quiz と同じにする
def make_quiz(lecture: int) -> str:
    unavailable = quiz_unavailable(lecture)
    if unavailable is not None:
        # raise にせず return で返す。文がそのままモデルに渡り、モデルは利用者に回を選び直してもらえる
        return unavailable
    return tools.make_quiz(lecture)


def main() -> None:
    parser = argparse.ArgumentParser(description="用語集 MCP サーバー")
    parser.add_argument("--http", action="store_true", help="Streamable HTTP で起動する（既定は stdio）")
    args = parser.parse_args()
    if args.http:
        print("MCP サーバーを http://127.0.0.1:8000/mcp で起動します（Ctrl+C で停止）")
        server.run(transport="streamable-http")
    else:
        # stdio では標準出力がプロトコルの通信路になるので、print してはいけない
        server.run()


if __name__ == "__main__":
    main()
