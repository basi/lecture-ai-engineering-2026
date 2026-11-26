"""第6回・セキュリティの確認用: stdio で起動された MCP サーバー（子プロセス）が、親プロセスの環境変数を見られるかを調べる。

MCP クライアントは、stdio のサーバーを子プロセスとして起動する。このとき子に渡る環境変数は、
mcp の SDK が HOME・LOGNAME・PATH・SHELL・TERM・USER だけに絞る（API キーなどは渡らない）。
ところが、同じユーザーで動く子プロセスは、環境変数を受け継がなくても、親の /proc/PID/environ を読める（Linux）。
それを、無害な目印の変数（AIE_LEC06_CANARY）で確かめる。本物のキーは使わない:
確かめるのは目印だけで、本物のキーの値は、取り出さない・返さない・保存しない。

ツール report は、次のことだけを JSON 文字列で返す。環境変数の値は、返さない・保存しない。
- 自分（子）の環境変数の名前と、目印があるか
- /proc があるか（macOS にはない。ないときは parent と ancestors が null になる）
- 親の /proc/PID/environ を読めるか。読めたときは、目印があるか（親の環境変数の並びは、
  目印で始まる項目があるかを調べるためだけに読み込み、判定したらすぐ捨てる）
- 親の親からさかのぼった先祖のプロセスの /proc/PID/environ を、開けるか。
  中身は読まない（開いて、すぐ閉じるだけ）

使い方:
    python lectures/06/env_probe.py     # stdio で待ち受ける（MCP クライアントがサブプロセスとして起動する）
"""

import json
import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

CANARY_NAME = "AIE_LEC06_CANARY"  # 無害な目印の変数名。本物のキーの名前は使わない
MAX_LEVELS = 4  # 親の親から、先祖をさかのぼる段数の上限
PROC = Path("/proc")  # テストが、偽の /proc に差し替えて確かめられるように、定数にしておく

server = MCPServer("aie2026-env-probe")


def _parent_pid_of(pid: int) -> int | None:
    """/proc/<pid>/stat から、親の pid を読む。読めなければ None を返す。"""
    try:
        stat = (PROC / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # 形式は「pid (コマンド名) 状態 親のpid …」。コマンド名に空白や ) が入りうるので、最後の ) より後ろを読む
    fields = stat.rpartition(")")[2].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _parent_report(pid: int) -> dict:
    """親の /proc/<pid>/environ に目印があるかだけを調べる。値は返さない・保存しない。"""
    try:
        data = (PROC / str(pid) / "environ").read_bytes()
    except OSError:
        return {"pid": pid, "environ_readable": False, "has_canary": None}
    prefix = f"{CANARY_NAME}=".encode()
    has_canary = any(item.startswith(prefix) for item in data.split(b"\0"))
    return {"pid": pid, "environ_readable": True, "has_canary": has_canary}


def _environ_openable(pid: int) -> bool:
    """/proc/<pid>/environ を開けるかだけを調べる。中身は読まない（開いて、すぐ閉じる）。"""
    try:
        with open(PROC / str(pid) / "environ", "rb"):
            return True
    except OSError:
        return False


def _ancestors(parent_pid: int) -> list[dict]:
    """親の親から最大 MAX_LEVELS 段まで、先祖のプロセスの environ を開けるかを調べる。pid 1 まで来たら止める。"""
    found = []
    pid: int | None = parent_pid
    for level in range(1, MAX_LEVELS + 1):
        if pid is None or pid <= 1:
            break
        pid = _parent_pid_of(pid)
        if not pid:  # 読めなかった（None）か、親がいない（0）
            break
        found.append({"level": level, "pid": pid, "environ_openable": _environ_openable(pid)})
    return found


@server.tool()
def report() -> str:
    """このプロセスの環境と、親・先祖のプロセスの /proc を調べた結果を、JSON 文字列で返す。環境変数の値は返さない。"""
    proc_available = (PROC / "self").exists()
    parent_pid = os.getppid()
    return json.dumps(
        {
            "platform": sys.platform,
            "own_env_names": sorted(os.environ),
            "own_env_has_canary": CANARY_NAME in os.environ,
            "proc_available": proc_available,
            "parent": _parent_report(parent_pid) if proc_available else None,
            "ancestors": _ancestors(parent_pid) if proc_available else None,
        },
        ensure_ascii=False,
    )


def main() -> None:
    # stdio では標準出力がプロトコルの通信路になるので、print してはいけない
    server.run()


if __name__ == "__main__":
    main()
