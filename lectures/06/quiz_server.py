"""第6回・宿題（実装）: make_quiz が出題できない回を受け取ったときの扱いを比べるための、実験用の MCP サーバー。

app/mcp_server.py の make_quiz は、出題できない回（存在しない回・用語集にない回）を、理由と出題できる回の文にして
return で返す。このファイルは、そこだけを変えた 3 つの変種を作る。ほかのツールと、サーバーの名前・説明は
app/mcp_server.py と同じなので、違いは「範囲外のときの扱い」だけになる。

- valueerror: 同じ文で ValueError を raise する。クライアントに返るのは「Error executing tool make_quiz」だけで、
              文はサーバーの標準エラー出力（トレースバック）にしか出ない
- toolerror:  同じ文で ToolError を raise する。クライアントに返る本文は「Error executing tool make_quiz: （文）」になる
- schema:     引数に範囲（1〜13）を付ける。範囲外は、関数に届く前に、サーバーが入力の検証エラーを返す。
              範囲内で用語のない回（第5回など）は、app/mcp_server.py と同じ文を return する

使い方（実験のときだけ使う。app/mcp_servers.json には書かない）:
    uv run python lectures/06/quiz_server.py --variant valueerror
    uv run python lectures/06/quiz_server.py --variant toolerror
    uv run python lectures/06/quiz_server.py --variant schema

stdio（標準入出力）で待ち受けるので、手で実行すると入力待ちのまま止まって見える。
MCP クライアントがサブプロセスとして起動する使い方で、lectures/06/test_lec06.py もそのようにして確かめる。
"""

import argparse
import sys
from pathlib import Path
from typing import Annotated

# このファイルは、パスを指定して実行される（python lectures/06/quiz_server.py）。
# リポジトリ直下は import の探索先に入らないので、先に入れてから app を import する
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402
from pydantic import Field  # noqa: E402

from app import tools  # noqa: E402
from app.mcp_server import LAST_LECTURE, quiz_unavailable  # noqa: E402
from app.mcp_server import server as return_server  # noqa: E402

VARIANTS = ("valueerror", "toolerror", "schema")


def build_server(variant: str) -> MCPServer:
    """変種ごとの MCP サーバーを作る。ツールと登録の順は app/mcp_server.py と同じで、make_quiz だけが違う。"""
    if variant not in VARIANTS:
        raise ValueError(f"variant は {', '.join(VARIANTS)} のどれかにしてください: {variant!r}")
    server = MCPServer(return_server.name, instructions=return_server.instructions)
    server.tool()(tools.search_glossary)
    server.tool()(tools.get_lecture_summary)

    # どの変種も、関数の名前を make_quiz にする。名前が違うと、引数のスキーマの title（make_quizArguments）が変わり、
    # 範囲外のときの扱い以外にも違いが出てしまう。説明文は、どれも第3回の make_quiz と同じにする
    if variant == "valueerror":

        @server.tool(description=tools.make_quiz.__doc__)
        def make_quiz(lecture: int) -> str:
            unavailable = quiz_unavailable(lecture)
            if unavailable is not None:
                raise ValueError(unavailable)  # 想定外の失敗として扱われ、文はクライアントに渡らない
            return tools.make_quiz(lecture)

    elif variant == "toolerror":

        @server.tool(description=tools.make_quiz.__doc__)
        def make_quiz(lecture: int) -> str:
            unavailable = quiz_unavailable(lecture)
            if unavailable is not None:
                raise ToolError(unavailable)  # 想定した失敗として扱われ、文がクライアントに渡る
            return tools.make_quiz(lecture)

    else:  # schema

        @server.tool(description=tools.make_quiz.__doc__)
        def make_quiz(lecture: Annotated[int, Field(ge=1, le=LAST_LECTURE)]) -> str:
            unavailable = quiz_unavailable(lecture)  # 範囲内で用語のない回（第5回など）だけが、ここに来る
            if unavailable is not None:
                return unavailable
            return tools.make_quiz(lecture)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="make_quiz の範囲外の扱いを比べるための MCP サーバー（実験用）")
    parser.add_argument("--variant", choices=VARIANTS, required=True, help="出題できない回のときの扱い")
    args = parser.parse_args()
    # stdio では標準出力がプロトコルの通信路になるので、print してはいけない
    build_server(args.variant).run()


if __name__ == "__main__":
    main()
