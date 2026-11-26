"""第6回: Codespaces で 1 回だけ流す、実行スクリプト。結果は lectures/06/results/ に集める。

    uv run python lectures/06/run.py              # 全部を流す（途中で止まっても、同じコマンドで続きから流せる）
    uv run python lectures/06/run.py --list       # 流す項目と、済んだかどうかを表示する（何も起動しない）
    uv run python lectures/06/run.py --scan       # results/ に秘密情報らしき文字列がないかを調べる
    uv run python lectures/06/run.py --summarize  # results/summary.md を作る（結果を pull したあと、手元で使う）

流す項目は 3 段に分かれる。
- 段 A（API を使わない）: ツール一覧・pytest・サーバーの直接の呼び出し・環境変数の確認。
  uvx の初回取得と AWS への接続は、LLM を使う前に、ここで済ませる
- 段 B（LLM を使う CLI）: 演習と宿題の質問を、app.mcp_agent に聞く
- 段 C（LLM を使う、プロセス内の比較）: make_quiz の返し方 4 条件を、同じ質問で比べる

流している間は、app/mcp_server.py と app/mcp_servers.json を、演習・宿題の途中の状態に一時的に置き換える。
終わるとき（Ctrl+C・SIGTERM・SIGHUP で止まったときも）HEAD の内容に戻し、sha256 で確かめる。
このファイルの中では app.mcp_server を import しない（実行中にそのファイルを書き換えるため）。

終了コード: 0 完了／1 一部が失敗、または前提を満たさない／2 Gemini API の 1 日の上限／130 Ctrl+C
"""

import argparse
import asyncio
import contextlib
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic_core
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.shared.exceptions import MCPError
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import AgentRunError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

# このファイルは、パスを指定して実行される（python lectures/06/run.py）。リポジトリ直下は import の探索先に
# 入らないので、先に入れてから app を import する
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
for _path in (str(REPO), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import env_probe  # noqa: E402
from app.config import DEFAULT_MODEL, MODEL, get_api_key  # noqa: E402
from app.mcp_agent import build_mcp_agent, build_model, describe_servers  # noqa: E402

RUN_PY = Path(__file__).resolve()
RESULTS = REPO / "lectures/06/results"  # テストは、一時ディレクトリに差し替える

SERVER = "app/mcp_server.py"  # 実行中に、状態ごとの内容へ一時的に置き換えるファイル
CONFIG = "app/mcp_servers.json"
QUIZ_SERVER = "lectures/06/quiz_server.py"
BASE_COMMIT = "dccf422d6671eff654300cbb4f22a9b67165989d"  # 第6回の演習の開始時点（S0 と STEP 1 の内容はここから取る）
TODO_LINE = "# TODO(宿題・基礎): get_lecture_summary も公開しよう"
ENABLE_LINE = "server.tool()(tools.get_lecture_summary)"

VARIANTS = ("valueerror", "toolerror", "schema")  # lectures/06/quiz_server.py の変種（そこから import すると app.mcp_server を読むため、ここに写す）
CONDITIONS = ("return", *VARIANTS)  # make_quiz が出題できない回を、どう返すか
CANARY_VALUE = "lec06-canary"  # env_probe の目印の値。出力に出てはいけない
MODE = "legacy"  # app/mcp_agent.py の load_mcp_servers と同じ接続方式
DIRECT_LECTURES = (99, 5, 2)  # 直接の呼び出しで、make_quiz に渡す回

# 待ち時間と回数
LLM_INTERVAL = 5  # LLM を呼ぶ実行の間隔（秒）
RETRY_WAIT = 60  # やり直す前に待つ秒数
MAX_RETRIES = 3
RATE_LIMIT_STREAK_LIMIT = 3  # 429 がこの回数続いたら、1 日の上限とみなす
TIMEOUT_LIST_TOOLS = 180
TIMEOUT_TRACE = 300
TIMEOUT_PYTEST = 600
TIMEOUT_PROBE = 120

# 子プロセスに渡さない環境変数の名前（大文字小文字を区別しない）
ENV_SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")

# --scan が調べるもの
SCAN_NAME_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
SCAN_MIN_LENGTH = 8
SCAN_PATTERNS = {
    "Google API キーの形": re.compile(rb"AIza[0-9A-Za-z_\-]{35}"),
    "GitHub トークンの形": re.compile(rb"gh[opsu]_[0-9A-Za-z]{36,}"),
    "GitHub fine-grained トークンの形": re.compile(rb"github_pat_[0-9A-Za-z_]{22,}"),
}

# app/mcp_servers.json の HEAD の内容（problem.md の STEP 3・4 と同じ値）。起動時に確かめる
EXPECTED_SERVERS = {
    "glossary": {"command": "uv", "args": ["run", "python", "-m", "app.mcp_server"]},
    "fetch": {"command": "uvx", "args": ["mcp-server-fetch==2026.11.11", "--no-readability-js"]},
    "aws": {"url": "https://knowledge-mcp.global.api.aws"},
}

SERVER_STATES = {
    "S0": "演習の開始時点の app/mcp_server.py",
    "S1": "基礎の宿題を終えた app/mcp_server.py",
    "S2": "実装の宿題を終えた app/mcp_server.py（HEAD と同じ）",
}
CONFIG_STATES = {
    "STEP1": "STEP 1（glossary だけ）",
    "STEP3": "STEP 3（glossary と fetch）",
    "STEP4": "STEP 4（glossary・fetch・aws）",
}
CONFIG_SHORT = {"STEP1": "STEP 1", "STEP3": "STEP 3", "STEP4": "STEP 4"}

# 段 B の質問。problem.md の文と、一字一句同じにする（テストが確かめる）
QUESTIONS = {
    "step2": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "step3": "https://modelcontextprotocol.io/ の冒頭を日本語で3行に要約して",
    "step4": "AWSのドキュメントを検索して、Amazon S3 のバージョニングを2行で説明して",
    "basic": "第2回では何を学びますか？",
    "impl": "第99回のクイズを作って",
}
# 段 C の質問
IMPL_QUESTIONS = {"q99": "第99回のクイズを作って", "q5": "第5回のクイズを作って"}

STAGE_TITLES = {
    "A": "段 A（API を使わない）",
    "B": "段 B（LLM を使う CLI）",
    "C": "段 C（LLM を使う、プロセス内の比較。S2）",
}

OUTCOME_TEXT = {
    "ok": "成功",
    "stopped": "停止（UnexpectedModelBehavior）",
    "rate_limited": "失敗（レート制限）",
    "overloaded": "失敗（モデルが混み合っている）",
    "mcp_connect": "失敗（MCP サーバーに接続できない）",
    "error": "失敗",
}
RETRYABLE = ("rate_limited", "overloaded", "mcp_connect")  # 待ってやり直す分類

JST = dt.timezone(dt.timedelta(hours=9), "JST")


class StateError(Exception):
    """app/mcp_server.py・app/mcp_servers.json の扱いで、続けられないときのエラー。"""


class UnknownChange(StateError):
    """前回の途中の状態ではない変更が、残っている。"""


class DailyLimit(Exception):
    """Gemini API の 1 日の上限に達したとみなして、止めるときの例外。"""


# --- 時刻・表示・書き込み ---


def now() -> dt.datetime:
    """日本時間の現在時刻。"""
    return dt.datetime.now(JST)


def iso(moment: dt.datetime) -> str:
    return moment.isoformat(timespec="seconds")


def sleep(seconds: float) -> None:
    """待つ。テストは、これを差し替えて待たずに済ませる。"""
    time.sleep(seconds)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def secret_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """名前に TOKEN・KEY・SECRET・PASSWORD を含み、値が 8 文字以上の環境変数（名前と値）。"""
    source = os.environ if env is None else env
    return {
        name: value
        for name, value in source.items()
        if any(word in name.upper() for word in SCAN_NAME_WORDS) and len(value) >= SCAN_MIN_LENGTH
    }


def redact(text: str) -> str:
    """画面と run.log に出す文から、秘密の値を伏せる（念のため）。"""
    for value in secret_values().values():
        text = text.replace(value, "***")
    return text


def shown(path: Path) -> str:
    """表示用の短いパス（results/ からの相対）。results/ の外なら、そのままのパス。"""
    try:
        return f"{RESULTS.name}/{Path(path).relative_to(RESULTS).as_posix()}"
    except ValueError:
        return str(path)


def log(message: str) -> None:
    """進行を標準出力に 1 行で表示し、同じ行を results/run.log に追記する。"""
    line = f"{now():%Y-%m-%d %H:%M:%S} {redact(message)}".replace("\r", " ").replace("\n", " ")  # 例外の文に改行があっても、1 行にする
    print(line, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write(path: Path, data: bytes | str, *, mode_from: Path | None = None) -> None:
    """同じディレクトリの一時ファイルに書いてから、os.replace で置き換える（途中で止まっても、壊れたファイルを残さない）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if mode_from is not None and Path(mode_from).exists():
            shutil.copymode(mode_from, tmp)  # 置き換えても、元のファイルの権限を変えない
        else:
            os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def remove_temp_files() -> None:
    """前回、書き込みの途中で止まったときに残った一時ファイルを消す。"""
    for path in RESULTS.rglob(".*.tmp"):
        with contextlib.suppress(OSError):
            path.unlink()


def jsonable(value: Any) -> Any:
    return pydantic_core.to_jsonable_python(value, fallback=str)


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def read_text(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- git ---


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True)


def git_text(*args: str) -> str | None:
    """git の出力（前後の空白を除く）。失敗したら None。"""
    completed = git(*args)
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", "replace").strip()


def git_show(rev: str, path: str) -> bytes:
    """git show <rev>:<path> の内容（バイト列）。"""
    completed = git("show", f"{rev}:{path}")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise StateError(f"git show {rev[:7]}:{path} に失敗しました: {detail}")
    return completed.stdout


# --- 状態（app/mcp_server.py と app/mcp_servers.json を、一時的に置き換える） ---


@dataclass(frozen=True)
class Contents:
    """状態ごとのファイルの内容（バイト列）。"""

    head_server: bytes
    head_config: bytes
    servers: dict[str, bytes]  # S0・S1・S2
    configs: dict[str, bytes]  # STEP1・STEP3・STEP4


def build_contents(head_server: bytes, head_config: bytes, base_server: bytes, base_config: bytes) -> Contents:
    """HEAD と、第6回の開始時点（BASE_COMMIT）の内容から、すべての状態の内容を作る。git には触れない。"""
    # S1: S0 の TODO の行（ちょうど 1 つ）の直後に、1 行を足したもの。commit のハッシュには頼らない
    lines = base_server.decode("utf-8").splitlines(keepends=True)
    todo = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == TODO_LINE]
    if len(todo) != 1:
        raise StateError(f"{BASE_COMMIT[:7]} の {SERVER} に、次の行がちょうど 1 つあるはずですが、{len(todo)} 個ありました: {TODO_LINE}")
    index = todo[0]
    s1 = "".join(lines[: index + 1] + [ENABLE_LINE + "\n"] + lines[index + 1 :]).encode("utf-8")

    # STEP4 は HEAD の設定。problem.md の STEP 3・4 と同じ値かを確かめる
    try:
        head_servers = json.loads(head_config.decode("utf-8"))["mcpServers"]
    except (ValueError, KeyError) as e:
        raise StateError(f"HEAD の {CONFIG} を読めません: {e}") from e
    if head_servers != EXPECTED_SERVERS or list(head_servers) != list(EXPECTED_SERVERS):
        raise StateError(f"HEAD の {CONFIG} が、problem.md の STEP 4 と同じではありません（glossary・fetch・aws の順）")
    # STEP3: HEAD の glossary と fetch だけ
    step3 = {"mcpServers": {name: head_servers[name] for name in ("glossary", "fetch")}}

    return Contents(
        head_server=head_server,
        head_config=head_config,
        servers={"S0": base_server, "S1": s1, "S2": head_server},
        configs={"STEP1": base_config, "STEP3": dump_json(step3).encode("utf-8"), "STEP4": head_config},
    )


def load_contents() -> Contents:
    """git から、すべての状態の内容を取り出す。"""
    return build_contents(
        git_show("HEAD", SERVER),
        git_show("HEAD", CONFIG),
        git_show(BASE_COMMIT, SERVER),
        git_show(BASE_COMMIT, CONFIG),
    )


class Workspace:
    """app/mcp_server.py と app/mcp_servers.json を、状態ごとの内容に置き換え、HEAD に戻す。

    root は、リポジトリ直下。テストは、一時ディレクトリのコピーを渡す（本物のファイルは書き換えない）。
    """

    def __init__(self, root: Path, contents: Contents):
        self.root = Path(root)
        self.contents = contents
        self.server_path = self.root / SERVER
        self.config_path = self.root / CONFIG

    def _write(self, path: Path, data: bytes) -> None:
        try:
            if path.exists() and path.read_bytes() == data:
                return  # すでに同じ内容なら、書かない
            atomic_write(path, data, mode_from=path)
        except OSError as e:
            raise StateError(f"{path} を書き換えられませんでした: {e}") from e

    def apply(self, server: str | None, config: str | None) -> None:
        """状態に合わせて書き換える。None のものは、そのままにする。"""
        if server is not None:
            self._write(self.server_path, self.contents.servers[server])
        if config is not None:
            self._write(self.config_path, self.contents.configs[config])

    def restore(self) -> None:
        """HEAD の内容に戻し、戻ったことを sha256 で確かめる。確かめられなければ StateError。"""
        problems = []
        for path, data in ((self.server_path, self.contents.head_server), (self.config_path, self.contents.head_config)):
            try:
                self._write(path, data)  # 片方が失敗しても、もう片方は戻す
                if sha256_of(path) != sha256_bytes(data):
                    problems.append(f"{path.name} の sha256 が HEAD と違います")
            except (StateError, OSError) as e:
                problems.append(str(e))
        if problems:
            raise StateError("; ".join(problems))

    def digests(self) -> dict[str, str]:
        """いまの 2 つのファイルの sha256。"""
        return {SERVER: sha256_of(self.server_path), CONFIG: sha256_of(self.config_path)}


def reconcile_leftover(workspace: Workspace) -> list[str]:
    """前回の中断で、既知の途中の状態（S0・S1、STEP 1・STEP 3）のまま残っていたら、HEAD の内容に戻す。

    戻したものの説明を返す。既知でない変更が残っていたら、何も書き換えずに UnknownChange を投げる。
    """
    contents = workspace.contents
    checks = (
        (SERVER, workspace.server_path, contents.head_server, {"S0": contents.servers["S0"], "S1": contents.servers["S1"]}),
        (CONFIG, workspace.config_path, contents.head_config, {"STEP1": contents.configs["STEP1"], "STEP3": contents.configs["STEP3"]}),
    )
    to_restore = []
    for name, path, head, known in checks:
        try:
            current = path.read_bytes()
        except OSError:
            current = None
        if current == head:
            continue
        label = next((label for label, data in known.items() if data == current), None)
        if label is None:
            raise UnknownChange(
                f"{name} に、このスクリプトが作る状態とは違う変更があります。自分で直した内容でないか、確かめてください。\n"
                f"残す場合は commit してから、捨てる場合は git checkout HEAD -- {name} を実行してから、もう一度実行します。"
            )
        to_restore.append((name, path, head, label))
    messages = []
    for name, path, head, label in to_restore:
        workspace._write(path, head)
        messages.append(f"前回の実行が途中で止まったようです。{name} が途中の状態（{label}）のまま残っていたので、HEAD の内容に戻しました。")
    return messages


# --- 子プロセスに渡す環境変数 ---


def env_for(needs_api: bool, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """子プロセスに渡す環境変数。秘密らしい名前の変数は除き、LLM を呼ぶ実行にだけ GEMINI_API_KEY を足す。"""
    source = os.environ if base is None else base
    env = {name: value for name, value in source.items() if not any(word in name.upper() for word in ENV_SECRET_WORDS)}
    if needs_api and "GEMINI_API_KEY" in source:
        env["GEMINI_API_KEY"] = source["GEMINI_API_KEY"]
    env["PYTHONUNBUFFERED"] = "1"
    return env


def probe_env() -> dict[str, str]:
    """env_probe の確認で、--probe-client に渡す最小限の環境変数と、無害な目印。TOKEN や KEY の類は渡さない。"""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONUNBUFFERED": "1",
        env_probe.CANARY_NAME: CANARY_VALUE,
    }
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    return env


# --- 実行する項目の表 ---


@dataclass(frozen=True)
class Item:
    id: str
    stage: str  # A・B・C
    kind: str  # cli・pytest・direct・probe・impl
    server: str | None  # app/mcp_server.py の状態（S0・S1・S2）
    config: str | None  # app/mcp_servers.json の状態（STEP1・STEP3・STEP4）
    summary: str  # 内容
    argv: tuple[str, ...] | None  # コマンド（CLI と pytest）
    func: str | None  # 関数（direct・probe・impl）
    uses_api: bool
    timeout: int | None  # 秒
    condition: str | None = None
    question_id: str | None = None
    trial: int | None = None


_LIST_TOOLS = ("uv", "run", "python", "-m", "app.mcp_agent", "--list-tools")
_PYTEST = ("uv", "run", "pytest", "-q")

# 段 A の CLI: （id, 状態, 設定, 内容, コマンド, 種類, timeout）
_STAGE_A = (
    ("step1_agent_tools", "S0", "STEP1", "STEP 1: 第5回までのエージェントのツール一覧", ("uv", "run", "python", "-m", "app.agent", "--list-tools"), "cli", TIMEOUT_LIST_TOOLS),
    ("step1_tools", "S0", "STEP1", "STEP 1: MCP エージェントのツール一覧（glossary だけ）", _LIST_TOOLS, "cli", TIMEOUT_LIST_TOOLS),
    ("step3_tools", "S0", "STEP3", "STEP 3: ツール一覧（fetch を足す。uvx の初回取得）", _LIST_TOOLS, "cli", TIMEOUT_LIST_TOOLS),
    ("step4_tools", "S0", "STEP4", "STEP 4: ツール一覧（aws を足す。AWS に接続）", _LIST_TOOLS, "cli", TIMEOUT_LIST_TOOLS),
    ("basic_tools", "S1", "STEP4", "宿題・基礎: ツール一覧（get_lecture_summary が増える）", _LIST_TOOLS, "cli", TIMEOUT_LIST_TOOLS),
    ("impl_tools", "S2", "STEP4", "宿題・実装: ツール一覧（make_quiz が増える）", _LIST_TOOLS, "cli", TIMEOUT_LIST_TOOLS),
    ("basic_pytest", "S1", "STEP4", "宿題・基礎: pytest（upstream のテスト）", _PYTEST, "pytest", TIMEOUT_PYTEST),
    ("impl_pytest", "S2", "STEP4", "宿題・実装: pytest（upstream のテスト）", _PYTEST, "pytest", TIMEOUT_PYTEST),
    ("impl_pytest_lec06", "S2", "STEP4", "宿題・実装: pytest（lectures/06 のテスト）", (*_PYTEST, "lectures/06"), "pytest", TIMEOUT_PYTEST),
)

# 段 B: （id の前半, 状態, 設定, 内容, 質問の名前, 回数）
_STAGE_B = (
    ("step2", "S0", "STEP1", "STEP 2: 用語集を使う質問", "step2", 2),
    ("step3", "S0", "STEP3", "STEP 3: fetch を使う質問", "step3", 2),
    ("step4", "S0", "STEP4", "STEP 4: aws を使う質問", "step4", 2),
    ("basic", "S1", "STEP4", "宿題・基礎: get_lecture_summary を使う質問", "basic", 2),
    ("impl", "S2", "STEP4", "宿題・実装: 出題できない回を聞く質問", "impl", 1),
)


def impl_order() -> list[tuple[str, str, int]]:
    """段 C の実行の順（条件, 質問, 試行）。計 15 回。

    試行 1〜3 のそれぞれで、return・valueerror・toolerror を 1 つずつ回した順で q99。続いて schema の q99 を 3 回、return の q5 を 3 回。
    """
    first = ("return", "valueerror", "toolerror")
    order = []
    for trial in (1, 2, 3):
        shift = trial - 1
        for condition in first[shift:] + first[:shift]:
            order.append((condition, "q99", trial))
    order += [("schema", "q99", trial) for trial in (1, 2, 3)]
    order += [("return", "q5", trial) for trial in (1, 2, 3)]
    return order


def build_items() -> list[Item]:
    """流す項目の表。この順に実行する。"""
    items = []
    for id_, server, config, summary, argv, kind, timeout in _STAGE_A:
        items.append(Item(id_, "A", kind, server, config, summary, argv, None, False, timeout))
    for condition in CONDITIONS:
        summary = f"直接の呼び出し（{condition}）: make_quiz を 99・5・2 で呼び、FunctionModel で 2 回続けて呼ぶ"
        items.append(Item(f"direct_{condition}", "A", "direct", "S2", None, summary, None, "run_direct", False, None, condition=condition))
    items.append(Item("env_probe", "A", "probe", None, None, "env_probe: エージェントと同じクライアントの経路で、子の環境変数を調べる", None, "run_probe", False, TIMEOUT_PROBE))
    for prefix, server, config, summary, question_key, count in _STAGE_B:
        argv = ("uv", "run", "python", "-m", "app.mcp_agent", "--trace", QUESTIONS[question_key])
        for trial in range(1, count + 1):
            items.append(Item(f"{prefix}_t{trial}", "B", "cli", server, config, summary, argv, None, True, TIMEOUT_TRACE))
    for condition, question_id, trial in impl_order():
        summary = f"プロセス内の比較: 条件 {condition}／{IMPL_QUESTIONS[question_id]}／試行 {trial}"
        items.append(
            Item(f"{condition}_{question_id}_t{trial}", "C", "impl", "S2", None, summary, None, "run_impl", True, None,
                 condition=condition, question_id=question_id, trial=trial)
        )
    return items


def result_base(item: Item) -> Path:
    """項目の結果ファイルの、拡張子を除いたパス。"""
    if item.kind in ("cli", "pytest"):
        return RESULTS / "cli" / item.id
    if item.kind == "direct":
        return RESULTS / "direct" / item.condition
    if item.kind == "probe":
        return RESULTS / "security" / "env_probe"
    return RESULTS / "impl" / item.condition / item.question_id / f"t{item.trial}"


def done_path(item: Item) -> Path:
    """このファイルがあれば、その項目は済み。"""
    return Path(f"{result_base(item)}{'.txt' if item.kind in ('cli', 'pytest') else '.json'}")


def error_path(item: Item) -> Path:
    return Path(f"{result_base(item)}.error.txt")


def trace_path(item: Item) -> Path:
    return Path(f"{result_base(item)}.trace.txt")


def server_text(state: str | None) -> str:
    return f"{state}（{SERVER_STATES[state]}）" if state else "なし"


def config_text(state: str | None) -> str:
    return CONFIG_STATES[state] if state else "なし"


def state_short(item: Item) -> str:
    parts = [part for part in (item.server, CONFIG_SHORT.get(item.config)) if part]
    return "・".join(parts) or "-"


# --- CLI の結果の分類と保存 ---


def classify_cli(returncode: int | None, stdout: str, stderr: str) -> str:
    """CLI の実行結果を分ける: ok・rate_limited・overloaded・mcp_connect・stopped・error（timeout は returncode が None）。"""
    if returncode is None:
        return "error"
    if returncode == 0:
        return "ok"
    text = f"{stdout}\n{stderr}"
    if "レート制限に達しました" in text:
        return "rate_limited"
    if "混み合っています（503）" in text:
        return "overloaded"
    if "MCP サーバーに接続できませんでした" in text:
        return "mcp_connect"
    if "UnexpectedModelBehavior" in stderr:
        return "stopped"
    return "error"


def outcome_of(item: Item, returncode: int | None, stdout: str, stderr: str) -> str:
    """項目の結果の分類。pytest は、テストの文面が分類の文字列に当たらないよう、終了コードだけで分ける。"""
    if item.kind == "pytest":
        return "ok" if returncode == 0 else "error"
    return classify_cli(returncode, stdout, stderr)


def outcome_text(outcome: str, returncode: int | None) -> str:
    if outcome == "error" and returncode is None:
        return "失敗（timeout）"
    return OUTCOME_TEXT[outcome]


def execute(argv: Any, timeout: int | None, env: Mapping[str, str]) -> tuple[int | None, str, str]:
    """コマンドを実行し、（終了コード, stdout, stderr）を返す。timeout や起動の失敗のときの終了コードは None。

    パイプは通さず、一時ファイルへ直接書かせる。
    """
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out, tempfile.TemporaryFile(
        "w+", encoding="utf-8", errors="replace"
    ) as err:
        try:
            returncode: int | None = subprocess.run(argv, cwd=REPO, env=dict(env), stdout=out, stderr=err, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            returncode = None
        except OSError as e:  # コマンドを起動できなかった
            err.write(f"{type(e).__name__}: {e}\n")
            returncode = None
        out.seek(0)
        err.seek(0)
        return returncode, out.read(), err.read()


def cli_report(
    item: Item, *, digests: Mapping[str, str], head: str, started: dt.datetime, seconds: float,
    returncode: int | None, retries: int, outcome: str, stdout: str, stderr: str,
) -> str:
    """CLI の結果のファイルの中身。見出し、空行、stdout、（stderr が空でなければ）--- stderr --- と stderr。"""
    code = "なし（timeout または起動の失敗）" if returncode is None else str(returncode)
    heading = [
        f"$ {shlex.join(item.argv)}",
        f"# 状態: {server_text(item.server)}／設定: {config_text(item.config)}",
        f"# sha256: {SERVER}={digests[SERVER]} {CONFIG}={digests[CONFIG]}",
        f"# git HEAD: {head}",
        f"# 開始: {iso(started)}／所要: {seconds:.1f} 秒／終了コード: {code}／やり直し: {retries} 回／結果: {outcome_text(outcome, returncode)}",
    ]
    text = "\n".join(heading) + "\n\n" + stdout
    if stdout and not stdout.endswith("\n"):
        text += "\n"
    if stderr:
        text += "--- stderr ---\n" + stderr
        if not stderr.endswith("\n"):
            text += "\n"
    return text


# --- 段 A: 直接の呼び出し ---


def server_command(condition: str) -> tuple[str, list[str]]:
    """条件ごとの、MCP サーバーの起動コマンド。テストは、これを sys.executable で起動するものに差し替える。"""
    if condition == "return":
        return "uv", ["run", "python", "-m", "app.mcp_server"]
    return "uv", ["run", "python", QUIZ_SERVER, "--variant", condition]


def _text_of(result: Any) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


def two_calls_then_answer(messages: list, info: Any) -> ModelResponse:
    """glossary_make_quiz を第99回で呼び、結果が返ったら文章で答える偽のモデル（API は使わない）。

    通常の結果が返ったら（return の条件）、同じ呼び出しを繰り返さずに答える。失敗が続く条件では、2 回続けて呼ぶ。
    """
    returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
    calls = sum(
        isinstance(part, ToolCallPart) for message in messages if isinstance(message, ModelResponse) for part in message.parts
    )
    if returned or calls >= 2:
        return ModelResponse(parts=[TextPart("（偽のモデルの答え）")])
    return ModelResponse(parts=[ToolCallPart("glossary_make_quiz", {"lecture": 99})])


def message_facts(messages: list) -> dict:
    """残ったメッセージから、ツールの呼び出し・戻り・再試行の依頼・終了の理由・トークン数を取り出す。"""
    tool_calls, tool_returns, retry_prompts, finish_reasons = [], [], [], []
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    for message in messages:
        if isinstance(message, ModelResponse):
            finish_reasons.append(message.finish_reason)  # 取れなければ None
            usage["requests"] += 1
            usage["input_tokens"] += message.usage.input_tokens
            usage["output_tokens"] += message.usage.output_tokens
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                try:
                    args = part.args_as_dict()
                except ValueError:
                    args = part.args
                tool_calls.append({"tool": part.tool_name, "args": args})
            elif isinstance(part, ToolReturnPart):
                tool_returns.append({"tool": part.tool_name, "content": jsonable(part.content)})
            elif isinstance(part, RetryPromptPart):
                retry_prompts.append(
                    {"tool": part.tool_name, "content": jsonable(part.content), "sent_to_model": part.model_response()}
                )
    quiz_args = [
        call["args"].get("lecture") if isinstance(call["args"], dict) else None
        for call in tool_calls
        if str(call["tool"]).endswith("make_quiz")
    ]
    return {
        "tool_calls": tool_calls,
        "tool_returns": tool_returns,
        "retry_prompts": retry_prompts,
        "called_make_quiz": bool(quiz_args),
        "make_quiz_args": quiz_args,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def exception_facts(error: BaseException | None) -> dict | None:
    """例外の型・文と、その原因（__cause__）の型・文。"""
    if error is None:
        return None
    cause = error.__cause__
    return {
        "type": type(error).__name__,
        "message": getattr(error, "message", None) or str(error),
        "cause_type": type(cause).__name__ if cause is not None else None,
        "cause_message": (getattr(cause, "message", None) or str(cause)) if cause is not None else None,
    }


def dump_messages(messages: list) -> Any:
    """メッセージを、JSON にできる形にする。実際のモデルの応答に JSON にできないものがあっても、ほかの項目は保存できるようにする。"""
    try:
        return ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    except Exception as e:
        return {"error": f"メッセージを JSON にできませんでした: {type(e).__name__}: {e}"}


async def direct_record(condition: str) -> dict:
    """サーバーを直接つないで、ツールの定義と make_quiz の結果を集め、FunctionModel の実行も記録する。"""
    command, args = server_command(condition)
    log_dir = RESULTS / "direct"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{condition}.server.log", f"{condition}.agent.server.log"):
        (log_dir / name).unlink(missing_ok=True)  # fastmcp は追記で書くので、前回の分を消す

    transport = StdioTransport(command, args, keep_alive=False, log_file=log_dir / f"{condition}.server.log")
    async with Client(transport, mode=MODE) as client:
        tools = [
            {"name": tool.name, "description": tool.description, "inputSchema": tool.input_schema}
            for tool in await client.list_tools()
        ]
        calls = {}
        for lecture in DIRECT_LECTURES:
            result = await client.call_tool("make_quiz", {"lecture": lecture}, raise_on_error=False)
            calls[str(lecture)] = {"isError": result.is_error, "text": _text_of(result)}

    # エージェントを通すと、モデルには何が渡るか（API は使わない。モデルは偽物）
    agent_transport = StdioTransport(command, args, keep_alive=False, log_file=log_dir / f"{condition}.agent.server.log")
    agent = Agent(toolsets=[MCPToolset(Client(agent_transport, mode=MODE)).prefixed("glossary")])
    error = None
    output = None
    with capture_run_messages() as messages:
        try:
            output = (await agent.run("第99回のクイズを作って", model=FunctionModel(two_calls_then_answer))).output
        except UnexpectedModelBehavior as e:
            error = e
    facts = message_facts(messages)
    return {
        "tools": tools,
        "calls": calls,
        "agent": {
            "output": output,
            **{key: facts[key] for key in ("tool_calls", "tool_returns", "retry_prompts")},
            "exception": exception_facts(error),
        },
    }


# --- 段 C: プロセス内の比較 ---


def write_condition_config(condition: str) -> Path:
    """条件ごとの設定ファイルを書く。サーバー名はどれも glossary なので、ツール名はどれも glossary_make_quiz になる。"""
    command, args = server_command(condition)
    path = RESULTS / "impl" / "configs" / f"{condition}.json"
    atomic_write(path, dump_json({"mcpServers": {"glossary": {"command": command, "args": args}}}))
    return path


@dataclass
class Attempt:
    """段 C の 1 回の実行。kind は ok・stopped・agent_error・http・connect。"""

    kind: str
    messages: list
    result: Any = None
    error: BaseException | None = None


def attempt_impl(config_path: Path, question: str) -> Attempt:
    """設定ファイルのエージェントに、質問を 1 回だけ聞く。想定した失敗は Attempt にして返し、想定外の例外はそのまま投げる。"""
    agent = build_mcp_agent(config_path)
    model = build_model()
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(question, model=model)
        except ModelHTTPError as e:  # AgentRunError より先に捕まえる
            return Attempt("http", messages, error=e)
        except UnexpectedModelBehavior as e:
            return Attempt("stopped", messages, error=e)
        except AgentRunError as e:  # UsageLimitExceeded など
            return Attempt("agent_error", messages, error=e)
        except (MCPError, OSError) as e:
            return Attempt("connect", messages, error=e)
        except RuntimeError as e:
            if "Client failed to connect" in str(e):
                return Attempt("connect", messages, error=e)
            raise
    return Attempt("ok", messages, result=result)


def retry_delay(body: str) -> float:
    """429 の本文にある retryDelay（秒）。読めなければ 60 秒。"""
    match = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", body)
    return float(match.group(1)) if match else float(RETRY_WAIT)


def content_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(jsonable(content), ensure_ascii=False)


def format_trace(header: list[str], messages: list, *, usage: Mapping[str, int], error: BaseException | None) -> str:
    """段 C の trace.txt の中身。app/mcp_agent.py の print_trace と同じ形だが、戻りの文は全文で、モデルに渡る文も出す。"""
    lines = list(header)
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                lines.append(f"[tool call]   {part.tool_name}({part.args})")
            elif isinstance(part, ToolReturnPart):
                lines.append(f"[tool return] {part.tool_name} -> {part.content}")
            elif isinstance(part, RetryPromptPart):
                lines.append(f"[tool error]  {part.tool_name} -> {content_text(part.content)}")
                lines.append(f"[sent to model] {part.model_response()}")
            elif isinstance(part, TextPart):
                lines.append(f"[answer] {part.content}")
        if isinstance(message, ModelResponse) and message.finish_reason:
            lines.append(f"[finish] {message.finish_reason}")
    if error is not None:
        facts = exception_facts(error)
        lines.append(f"[stopped] {facts['type']}: {facts['message']}")
        if facts["cause_type"]:
            lines.append(f"[cause] {facts['cause_type']}: {facts['cause_message']}")
    lines.append(
        f"[usage] 入力 {usage['input_tokens']:,} tokens / 出力 {usage['output_tokens']:,} tokens / モデル応答 {usage['requests']} 回"
    )
    return "\n".join(lines) + "\n"


# --- 実行 ---


@dataclass(frozen=True)
class Failure:
    id: str
    reason: str
    path: Path


class Runner:
    """項目を実行し、結果を保存する。失敗した項目は failures に集め、1 日の上限に達したら DailyLimit を投げる。"""

    def __init__(self, workspace: Workspace | None = None, head: str | None = None):
        self.workspace = workspace
        self.head = head if head is not None else (git_text("rev-parse", "HEAD") or "不明")
        self.failures: list[Failure] = []
        self.rate_limit_streak = 0
        self.llm_runs = 0

    # 共通

    def apply_state(self, item: Item) -> None:
        if self.workspace is not None:
            self.workspace.apply(item.server, item.config)

    def pause_before_llm(self) -> None:
        """LLM を呼ぶ実行の前に、間をあける（最初の 1 回は、あけない）。"""
        if self.llm_runs > 0:
            sleep(LLM_INTERVAL)
        self.llm_runs += 1

    def note_rate_limit(self, *, limited: bool, answered: bool) -> bool:
        """429 が続いた回数を数える。モデルから応答が返ったら、数を戻す。1 日の上限とみなすなら True。"""
        if limited:
            self.rate_limit_streak += 1
        elif answered:
            self.rate_limit_streak = 0
        return self.rate_limit_streak >= RATE_LIMIT_STREAK_LIMIT

    def fail(self, item: Item, reason: str, text: str) -> None:
        """失敗の内容を .error.txt に保存して、失敗の一覧に入れる。先へ進む。"""
        path = error_path(item)
        atomic_write(path, text)
        self.failures.append(Failure(item.id, reason, path))
        log(f"{item.id}: 失敗（{reason}）→ {shown(path)}")

    def skip_if_done(self, item: Item) -> bool:
        if done_path(item).exists():
            log(f"{item.id}: 済み（飛ばす）")
            return True
        return False

    # 項目

    def run_item(self, item: Item) -> None:
        try:
            if item.kind in ("cli", "pytest"):
                self.run_cli(item)
            elif item.kind == "direct":
                self.run_direct(item)
            elif item.kind == "probe":
                self.run_probe(item)
            else:
                self.run_impl(item)
        except (DailyLimit, StateError):
            raise
        except Exception as e:  # 想定外の例外でも、ほかの項目は続ける
            self.fail(item, f"想定外のエラー: {type(e).__name__}: {e}", traceback.format_exc())

    def run(self, items: list[Item]) -> None:
        for stage, title in STAGE_TITLES.items():
            stage_items = [item for item in items if item.stage == stage]
            done = sum(done_path(item).exists() for item in stage_items)
            log(f"=== {title}: {len(stage_items)} 項目（済み {done}）===")
            if stage == "C" and done < len(stage_items):
                log("サーバー側のエラーの表示（トレースバックなど）が、画面に出ることがあります。想定どおりの動きです。")
            for item in stage_items:
                self.run_item(item)

    # CLI と pytest

    def run_cli(self, item: Item) -> None:
        if self.skip_if_done(item):
            return
        self.apply_state(item)
        log(f"{item.id}: 実行（{state_short(item)}）{shlex.join(item.argv)}")
        if item.uses_api:
            self.pause_before_llm()
        retries = 0
        while True:
            started = now()
            began = time.monotonic()
            returncode, stdout, stderr = execute(item.argv, item.timeout, env_for(item.uses_api))
            seconds = time.monotonic() - began
            outcome = outcome_of(item, returncode, stdout, stderr)
            limit_reached = item.uses_api and self.note_rate_limit(
                limited=outcome == "rate_limited", answered=outcome in ("ok", "stopped")
            )
            if outcome in RETRYABLE and not limit_reached and retries < MAX_RETRIES:
                retries += 1
                log(f"{item.id}: {outcome_text(outcome, returncode)}。{RETRY_WAIT} 秒待って、やり直します（{retries}/{MAX_RETRIES} 回目）")
                sleep(RETRY_WAIT)
                continue
            break
        digests = self.workspace.digests() if self.workspace is not None else {SERVER: sha256_of(REPO / SERVER), CONFIG: sha256_of(REPO / CONFIG)}
        text = cli_report(
            item, digests=digests, head=self.head, started=started, seconds=seconds,
            returncode=returncode, retries=retries, outcome=outcome, stdout=stdout, stderr=stderr,
        )
        if limit_reached:
            atomic_write(error_path(item), text)
            raise DailyLimit(f"{item.id}: レート制限が {RATE_LIMIT_STREAK_LIMIT} 回続きました")
        if outcome in ("ok", "stopped"):
            atomic_write(done_path(item), text)
            error_path(item).unlink(missing_ok=True)  # 前回の失敗の記録を消す
            log(f"{item.id}: {outcome_text(outcome, returncode)}（{seconds:.1f} 秒、やり直し {retries} 回）→ {shown(done_path(item))}")
        else:
            self.fail(item, outcome_text(outcome, returncode), text)

    # 段 A: 直接の呼び出しと env_probe

    def run_direct(self, item: Item) -> None:
        if self.skip_if_done(item):
            return
        self.apply_state(item)
        command, args = server_command(item.condition)
        log(f"{item.id}: 実行（{state_short(item)}）{shlex.join([command, *args])}")
        started = now()
        began = time.monotonic()
        record = asyncio.run(direct_record(item.condition))
        record = {
            "condition": item.condition,
            "server_command": [command, *args],
            "started_at": iso(started),
            "duration_s": round(time.monotonic() - began, 1),
            "git_head": self.head,
            "files_sha256": {SERVER: sha256_of(REPO / SERVER), QUIZ_SERVER: sha256_of(REPO / QUIZ_SERVER)},
            **record,
        }
        atomic_write(done_path(item), dump_json(record))
        error_path(item).unlink(missing_ok=True)
        log(f"{item.id}: 成功（{record['duration_s']} 秒）→ {shown(done_path(item))}")

    def run_probe(self, item: Item) -> None:
        if self.skip_if_done(item):
            return
        env = probe_env()
        argv = [sys.executable, str(RUN_PY), "--probe-client"]
        log(f"{item.id}: 実行（クライアントに渡す環境変数: {', '.join(sorted(env))}）")
        returncode, stdout, stderr = execute(argv, item.timeout, env)
        if CANARY_VALUE in stdout or CANARY_VALUE in stderr:
            # 値が出力に出た。内容は保存しない
            self.fail(item, "目印の値が出力に出ました", "env_probe の目印の値が、出力に含まれていました。出力は保存していません。")
            return
        lines = stdout.strip().splitlines()
        try:
            if returncode != 0 or not lines:
                raise ValueError(f"終了コード {returncode}")
            report = json.loads(lines[-1])
        except ValueError as e:
            self.fail(item, f"--probe-client の結果を読めません（{e}）", f"{e}\n\n--- stderr ---\n{stderr}")
            return
        record = {"client_env_names": sorted(env), "canary_name": env_probe.CANARY_NAME, "report": report}
        atomic_write(done_path(item), dump_json(record))
        error_path(item).unlink(missing_ok=True)
        log(f"{item.id}: 成功 → {shown(done_path(item))}")

    # 段 C

    def run_impl(self, item: Item) -> None:
        if self.skip_if_done(item):
            return
        self.apply_state(item)
        config_path = write_condition_config(item.condition)
        question = IMPL_QUESTIONS[item.question_id]
        log(f"{item.id}: 実行（条件 {item.condition}／{question}／試行 {item.trial}）")
        self.pause_before_llm()
        retries = 0
        while True:
            started = now()
            began = time.monotonic()
            attempt = attempt_impl(config_path, question)
            seconds = time.monotonic() - began
            action, wait, reason = self.judge(item, attempt, retries)
            if action == "retry":
                retries += 1
                log(f"{item.id}: {reason}。{wait:.0f} 秒待って、やり直します（{retries}/{MAX_RETRIES} 回目）")
                sleep(wait)
                continue
            break
        if action == "fail":
            self.fail(item, reason, f"{reason}\n\n{self.describe_error(attempt)}\n")
            return
        status = {"http": "http_error"}.get(attempt.kind, attempt.kind)
        self.save_impl(item, attempt, status, config_path, started, seconds, retries)
        log(f"{item.id}: {status}（{seconds:.1f} 秒、やり直し {retries} 回）→ {shown(done_path(item))}")

    @staticmethod
    def describe_error(attempt: Attempt) -> str:
        error = attempt.error
        return f"{type(error).__name__}: {error}" if error is not None else ""

    def judge(self, item: Item, attempt: Attempt, retries: int) -> tuple[str, float, str]:
        """（動作, 待つ秒数, 理由）。動作は save（保存する）・retry（待ってやり直す）・fail（失敗の一覧に入れる）。

        1 日の上限とみなすときは、DailyLimit を投げる。
        """
        if attempt.kind in ("ok", "stopped", "agent_error"):
            self.note_rate_limit(limited=False, answered=True)
            return "save", 0, attempt.kind
        retry_or_fail = "retry" if retries < MAX_RETRIES else "fail"
        if attempt.kind == "connect":
            return retry_or_fail, RETRY_WAIT, f"MCP サーバーに接続できません（{attempt.error}）"
        error = attempt.error  # http
        status, body = error.status_code, str(error.body)
        if status == 429:
            if "PerDay" in body:
                atomic_write(error_path(item), f"429（1 日の上限）\n\n{body}")
                raise DailyLimit(f"{item.id}: 1 日の上限に達しました（429 PerDay）")
            if self.note_rate_limit(limited=True, answered=False):
                atomic_write(error_path(item), f"429（レート制限）が {RATE_LIMIT_STREAK_LIMIT} 回続きました\n\n{body}")
                raise DailyLimit(f"{item.id}: レート制限が {RATE_LIMIT_STREAK_LIMIT} 回続きました")
            return retry_or_fail, retry_delay(body), "429（レート制限）"
        if status in (500, 502, 503, 504):
            return retry_or_fail, RETRY_WAIT, f"{status}（モデルの側のエラー）"
        if status == 400 and item.condition == "schema":
            self.note_rate_limit(limited=False, answered=True)
            return "save", 0, "http_error"  # schema の条件では、400 も観察する結果
        return "fail", 0, f"HTTP {status}"

    def save_impl(self, item: Item, attempt: Attempt, status: str, config_path: Path, started: dt.datetime, seconds: float, retries: int) -> None:
        facts = message_facts(attempt.messages)
        result = attempt.result
        usage = (
            {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens, "requests": result.usage.requests}
            if result is not None
            else facts["usage"]
        )
        http_error = None
        if attempt.kind == "http":
            http_error = {"status": attempt.error.status_code, "body": str(attempt.error.body)[:1000]}
        record = {
            "condition": item.condition,
            "question_id": item.question_id,
            "question": IMPL_QUESTIONS[item.question_id],
            "trial": item.trial,
            "model": MODEL,
            "started_at": iso(started),
            "duration_s": round(seconds, 1),
            "status": status,
            "output": result.output if result is not None else None,
            "usage": usage,
            "tool_calls": facts["tool_calls"],
            "tool_returns": facts["tool_returns"],
            "retry_prompts": facts["retry_prompts"],
            "called_make_quiz": facts["called_make_quiz"],
            "make_quiz_args": facts["make_quiz_args"],
            "finish_reasons": facts["finish_reasons"],
            "exception": exception_facts(attempt.error) if attempt.kind in ("stopped", "agent_error") else None,
            "http_error": http_error,
            "messages": dump_messages(attempt.messages),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
            "git_head": self.head,
            "retries": retries,
            "files_sha256": {SERVER: sha256_of(REPO / SERVER), QUIZ_SERVER: sha256_of(REPO / QUIZ_SERVER)},
        }
        header = [
            f"# 条件: {item.condition}／質問: {item.question_id}（{IMPL_QUESTIONS[item.question_id]}）／試行: {item.trial}",
            f"# 開始: {iso(started)}／status: {status}／所要: {seconds:.1f} 秒／やり直し: {retries} 回",
            *(f"# {line}" for line in describe_servers(config_path)),
            "",
        ]
        error = attempt.error if attempt.kind != "ok" else None
        atomic_write(trace_path(item), format_trace(header, attempt.messages, usage=usage, error=error))
        atomic_write(done_path(item), dump_json(record))  # json が、済みの印なので、最後に書く
        error_path(item).unlink(missing_ok=True)


# --- 前処理 ---


def check_branch() -> bool:
    """HEAD が origin/lec06 と一致するか。"""
    head = git_text("rev-parse", "HEAD")
    remote = git_text("rev-parse", "--verify", "-q", "origin/lec06")
    if remote is None:
        print("origin/lec06 が見つかりません。次のコマンドを実行してから、もう一度実行してください。\n  git fetch origin")
        return False
    if head != remote:
        print(
            f"HEAD（{(head or '不明')[:7]}）が origin/lec06（{remote[:7]}）と違います。次の 2 つを実行してから、もう一度実行してください。\n"
            "  git switch lec06\n"
            "  git pull --ff-only\n"
            "授業で作った lec06 が残っている場合は、先に git branch -m lec06 lec06-class で名前を変えてから、上の 2 つを実行します。"
        )
        return False
    return True


def check_base_commit() -> bool:
    """第6回の開始時点の commit が、HEAD の履歴にあるか。"""
    if git("merge-base", "--is-ancestor", BASE_COMMIT, "HEAD").returncode == 0:
        return True
    print(
        f"第6回の開始時点（{BASE_COMMIT[:7]}）が、いまのブランチの履歴にありません。\n"
        "git fetch origin のあとに git switch lec06、git pull --ff-only を実行してから、もう一度実行してください。"
    )
    return False


def check_model() -> bool:
    """モデルを、既定の gemini-3.5-flash-lite に固定する。"""
    chosen = os.getenv("AIE_MODEL")
    if chosen is not None and chosen != DEFAULT_MODEL:
        print(
            f"環境変数 AIE_MODEL が {chosen!r} になっています。結果をそろえるため、モデルは {DEFAULT_MODEL} に固定します。\n"
            "unset AIE_MODEL を実行してから、もう一度実行してください。"
        )
        return False
    return True


def check_clean() -> bool:
    """app/ と lectures/06/ に、commit していない変更（追跡しているファイル）がないか。results/ は未追跡なので対象外。"""
    if git("diff", "--quiet", "HEAD", "--", "app", "lectures/06").returncode == 0:
        return True
    print(
        "app/ か lectures/06/ に、commit していない変更があります（git status で確認できます）。\n"
        "commit するか、git stash で退避してから、もう一度実行してください。"
    )
    return False


def check_tools() -> bool:
    """uv・uvx と、GEMINI_API_KEY があるか。"""
    missing = [name for name in ("uv", "uvx") if shutil.which(name) is None]
    if missing:
        print(f"{' と '.join(missing)} が見つかりません。Codespaces のターミナルで実行してください。")
        return False
    try:
        get_api_key()
    except SystemExit as e:  # 案内の文は、get_api_key と同じ
        print(e.code)
        return False
    return True


def preflight() -> Contents | None:
    """流す前に確かめる。続けられないときは、理由を表示して None を返す。"""
    if not (check_branch() and check_base_commit() and check_model()):
        return None
    try:
        contents = load_contents()
    except StateError as e:
        print(e)
        return None
    try:
        for message in reconcile_leftover(Workspace(REPO, contents)):
            print(message)
    except UnknownChange as e:
        print(e)
        return None
    if not (check_clean() and check_tools()):
        return None
    return contents


# --- --list ---


def is_done(item: Item) -> bool:
    return done_path(item).exists()


def results_dir() -> str:
    """表示用の、結果の置き場所（リポジトリ直下からの相対）。"""
    try:
        return f"{RESULTS.relative_to(REPO).as_posix()}/"
    except ValueError:
        return f"{RESULTS}/"


def pad(text: str, width: int) -> str:
    """全角を 2 桁として数え、右に空白を足して、表の列の幅をそろえる。"""
    used = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
    return text + " " * max(0, width - used)


def what_runs(item: Item) -> str:
    """表の「コマンド（または関数）」の列。"""
    if item.argv:
        return shlex.join(item.argv)
    if item.kind == "direct":
        command, args = server_command(item.condition)
        return f"関数 {item.func}: stdio で {shlex.join([command, *args])} につなぐ"
    if item.kind == "probe":
        return f"関数 {item.func}: python lectures/06/run.py --probe-client を、最小限の環境変数で起動する"
    return f"関数 {item.func}: build_mcp_agent（configs/{item.condition}.json）→ run_sync（{IMPL_QUESTIONS[item.question_id]}）"


def list_main() -> int:
    """流す項目の表と、済んだかどうかを表示する。状態も results も変えず、何も起動しない。"""
    items = build_items()
    done = sum(is_done(item) for item in items)
    print(f"第6回の実行項目: 全 {len(items)} 項目のうち、済みは {done} 項目（結果は {results_dir()} に保存する）")
    print("[済]／[未] は、結果のファイルがあるか。API は、LLM を呼ぶ項目。1 項目の 2 行目は、コマンド（または関数）。")
    for stage, title in STAGE_TITLES.items():
        print(f"\n{title}")
        for item in (item for item in items if item.stage == stage):
            limit = f"{item.timeout} 秒" if item.timeout else "-"
            mark = "済" if is_done(item) else "未"
            print(f"  [{mark}] {pad(item.id, 19)} {pad(state_short(item), 11)} {pad('API' if item.uses_api else '-', 3)} {pad(limit, 6)} {item.summary}")
            print(f"        {what_runs(item)}")
    return 0


# --- --scan ---


def scan_results(results: Path, env: Mapping[str, str] | None = None) -> tuple[int, list[tuple[str, str]]]:
    """results/ 以下のすべてのファイル（バイナリも）に、秘密情報らしき文字列がないかを調べる。

    （調べたファイルの数, [(results/ からの相対パス, 環境変数の名前または形の種類), ...]）を返す。値は返さない。
    """
    values = {name: value.encode("utf-8") for name, value in secret_values(env).items()}
    files = sorted(path for path in Path(results).rglob("*") if path.is_file())
    findings = []
    for path in files:
        data = path.read_bytes()
        relative = f"{results.name}/{path.relative_to(results).as_posix()}"
        findings += [(relative, name) for name, value in values.items() if value in data]
        findings += [(relative, kind) for kind, pattern in SCAN_PATTERNS.items() if pattern.search(data)]
    return len(files), findings


def scan_report(results: Path | None = None, env: Mapping[str, str] | None = None) -> tuple[int, list[str]]:
    """--scan の表示する行と、終了コード。見つかった値は、表示しない。"""
    results = RESULTS if results is None else Path(results)
    if not results.is_dir():
        return 0, ["results がありません"]
    count, findings = scan_results(results, env)
    if findings:
        return 1, [f"秘密情報らしき文字列が見つかりました: {path}（{kind}）" for path, kind in findings]
    return 0, [f"走査: 秘密情報らしき文字列は見つかりませんでした（{count} ファイル）"]


def scan_main() -> int:
    code, lines = scan_report()
    print("\n".join(lines))
    return code


# --- --summarize ---


def cell(text: Any, limit: int | None = None) -> str:
    """表の 1 マスに入れる文。改行と連続する空白を 1 つの空白にし、| は \\| にする。"""
    text = " ".join(str(text).split())
    if limit is not None:
        text = text[:limit]
    return text.replace("|", "\\|")


def stdout_part(text: str) -> str:
    """CLI の結果のファイルから、stdout の部分（--- stderr --- より前）を返す。"""
    return text.split("\n--- stderr ---\n")[0]


def extract_aws_tools(text: str) -> dict[str, str]:
    """ツール一覧の出力から、[aws_ で始まるツールの部分（名前と、その本文）を取り出す。"""
    sections: dict[str, list[str]] = {}
    current = None
    for line in stdout_part(text).splitlines():
        heading = re.fullmatch(r"\[([A-Za-z0-9_.\-]+)\]", line)  # MCP のツール名には、. と - も使える
        if heading:
            current = heading.group(1) if heading.group(1).startswith("aws_") else None
            if current:
                sections[current] = [line]
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).rstrip() for name, lines in sections.items()}


def summary_tool_calls(items: list[Item]) -> list[str]:
    lines = ["## 1. 段 B: [tool call] の行（cli/*.txt）", ""]
    for item in items:
        if item.stage != "B":
            continue
        lines += [f"### {item.id}", ""]
        text = read_text(done_path(item))
        if text is None:
            lines += ["なし（結果のファイルがありません）", ""]
            continue
        heading = [line.removeprefix("# ") for line in text.splitlines() if line.startswith(("# 状態:", "# 開始:"))]
        calls = [line for line in stdout_part(text).splitlines() if line.startswith("[tool call]")]
        lines += [f"- {line}" for line in heading]
        lines += ["", "```text", *(calls or ["なし（[tool call] の行がありません）"]), "```", ""]
    return lines


def summary_direct() -> list[str]:
    lines = ["## 2. 段 A: 直接の呼び出し（条件 × lecture）", "", "| 条件 | lecture | isError | 本文（先頭 80 字） |", "|---|---|---|---|"]
    records = {condition: read_json(RESULTS / "direct" / f"{condition}.json") for condition in CONDITIONS}
    for condition, record in records.items():
        for lecture in DIRECT_LECTURES:
            call = (record or {}).get("calls", {}).get(str(lecture))
            if call is None:
                lines.append(f"| {condition} | {lecture} | なし | なし |")
            else:
                lines.append(f"| {condition} | {lecture} | {call['isError']} | {cell(call['text'], 80)} |")
    lines += ["", "### lecture=99 のとき、モデルに渡る文（FunctionModel の実行で残ったメッセージ）", ""]
    for condition, record in records.items():
        agent = (record or {}).get("agent")
        if agent is None:
            lines += [f"- {condition}: なし", ""]
        elif agent["retry_prompts"]:
            lines += [f"- {condition}（RetryPromptPart の model_response()）:", "", "```text", agent["retry_prompts"][0]["sent_to_model"], "```", ""]
        elif agent["tool_returns"]:
            lines += [f"- {condition}（ToolReturnPart の content）:", "", "```text", str(agent["tool_returns"][0]["content"]), "```", ""]
        else:
            lines += [f"- {condition}: なし（ツールの戻りも再試行の依頼もありません）", ""]
    return lines


def summary_impl() -> list[str]:
    lines = [
        "## 3. 段 C: 条件 × 質問 × 試行",
        "",
        "| 条件 | 質問 | 試行 | make_quiz の引数 | エラー数 | status | 回答（先頭 60 字） | 入力 tokens | 出力 tokens | リクエスト |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        for question_id in IMPL_QUESTIONS:
            for trial in (1, 2, 3):
                path = RESULTS / "impl" / condition / question_id / f"t{trial}.json"
                record = read_json(path)
                if record is None:
                    if any(c == condition and q == question_id and t == trial for c, q, t in impl_order()):
                        lines.append(f"| {condition} | {question_id} | {trial} | なし | なし | なし | なし | なし | なし | なし |")
                    continue
                usage = record["usage"]
                lines.append(
                    f"| {condition} | {question_id} | {trial} | {cell(record['make_quiz_args'])} | {len(record['retry_prompts'])} "
                    f"| {record['status']} | {cell(record['output'] or '', 60)} "
                    f"| {usage['input_tokens']} | {usage['output_tokens']} | {usage['requests']} |"
                )
    return lines + [""]


def summary_aws() -> list[str]:
    lines = ["## 4. aws_ ツールの定義（step4_tools・basic_tools・impl_tools の一致）", ""]
    found = {}
    for id_ in ("step4_tools", "basic_tools", "impl_tools"):
        text = read_text(RESULTS / "cli" / f"{id_}.txt")
        found[id_] = None if text is None else extract_aws_tools(text)
        lines.append(f"- {id_}: {'なし（結果のファイルがありません）' if text is None else f'{len(found[id_])} 個（{', '.join(found[id_]) or 'なし'}）'}")
    present = {id_: tools for id_, tools in found.items() if tools is not None}
    lines.append("")
    if len(present) < 2:
        lines += ["比べられません（結果が 2 つ以上ありません）。", ""]
    elif all(tools == next(iter(present.values())) for tools in present.values()):
        lines += [f"一致していました（{', '.join(present)}）。", ""]
    else:
        lines += ["一致しませんでした。差分（最初の結果との違い）:", ""]
        names = list(present)
        first = [line for body in present[names[0]].values() for line in body.splitlines()]
        for other in names[1:]:
            if present[other] == present[names[0]]:
                continue  # 最初の結果と同じものは、差分がない
            other_lines = [line for body in present[other].values() for line in body.splitlines()]
            diff = difflib.unified_diff(first, other_lines, fromfile=names[0], tofile=other, lineterm="")
            lines += [f"- {other}:", "", "```diff", *diff, "```", ""]
    return lines


def summary_probe() -> list[str]:
    lines = ["## 5. env_probe（子プロセスが見られる環境変数）", ""]
    record = read_json(RESULTS / "security" / "env_probe.json")
    if record is None:
        return lines + ["なし（結果のファイルがありません）", ""]
    report = record.get("report", {})
    parent = report.get("parent")
    ancestors = report.get("ancestors")
    if parent is None:
        parent_text = "なし（/proc がありません）"
    else:
        parent_text = f"environ を読めた: {parent.get('environ_readable')}／目印があった: {parent.get('has_canary')}"
    if ancestors is None:
        ancestors_text = "なし（/proc がありません）"
    else:
        ancestors_text = f"{len(ancestors)} 段（environ を開けた: {sum(bool(a.get('environ_openable')) for a in ancestors)} 段）"
    return lines + [
        f"- クライアントに渡した環境変数の名前: {', '.join(record.get('client_env_names', []))}",
        f"- 目印の名前: {record.get('canary_name')}",
        f"- 子（サーバー）の環境変数の名前: {', '.join(report.get('own_env_names') or [])}",
        f"- 子の環境に目印があったか: {report.get('own_env_has_canary')}",
        f"- /proc があるか: {report.get('proc_available')}（{report.get('platform')}）",
        f"- 親の /proc/PID/environ: {parent_text}",
        f"- 先祖のプロセス: {ancestors_text}",
        "",
    ]


def summary_text() -> str:
    """results/ の結果から、summary.md の中身を作る。"""
    items = build_items()
    parts = ["# 第6回 実行結果のまとめ", "", "`run.py --summarize` が、results/ の結果から作った（サーバーも API も使っていない）。", ""]
    for section in (summary_tool_calls(items), summary_direct(), summary_impl(), summary_aws(), summary_probe()):
        parts += section
    return "\n".join(parts).rstrip() + "\n"


def summarize_main() -> int:
    if not RESULTS.is_dir():
        print("results がありません。結果を pull してから、もう一度実行してください。")
        return 1
    path = RESULTS / "summary.md"
    atomic_write(path, summary_text())
    print(f"{shown(path)} を作りました。")
    return 0


# --- --probe-client（隠しオプション） ---


def probe_client_main() -> int:
    """env_probe を、エージェントと同じクライアントの経路で呼び、ツールの結果（JSON 文字列）を 1 行で出す。API キーは要らない。

    設定には env を書かない。子に渡る環境変数は、クライアント（MCPToolset・fastmcp の Client・StdioTransport）が決める。
    """
    config = {"mcpServers": {"probe": {"command": sys.executable, "args": ["lectures/06/env_probe.py"]}}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe_servers.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        result = build_mcp_agent(path).run_sync("probe", model=TestModel(call_tools=["probe_report"]))
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    if not returns:
        print("probe_report のツールの戻りがありません", file=sys.stderr)
        return 1
    print(" ".join(str(returns[0].content).split()))
    return 0


# --- 実行全体 ---


def _exit_on_signal(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def exit_on_signals() -> Iterator[None]:
    """SIGTERM・SIGHUP が来たら SystemExit を投げる。finally を通って、状態を HEAD に戻してから終わるため。"""
    previous = {}
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, _exit_on_signal)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def restore_head(workspace: Workspace) -> bool:
    """app/mcp_server.py と app/mcp_servers.json を HEAD の内容に戻し、sha256 で確かめる。"""
    try:
        workspace.restore()
    except StateError as e:
        log(f"HEAD の内容に戻せませんでした: {e}")
        print(f"次のコマンドで、戻してください。\n  git checkout HEAD -- {SERVER} {CONFIG}")
        return False
    log(f"{SERVER} と {CONFIG} を、HEAD の内容に戻しました（sha256 を確かめました）")
    return True


def run_all() -> int:
    contents = preflight()
    if contents is None:
        return 1
    return execute_all(Workspace(REPO, contents), build_items())


def execute_all(workspace: Workspace, items: list[Item], runner: Runner | None = None) -> int:
    """項目を順に流す。終わるとき（止めたときも）状態を HEAD に戻し、結果を走査して、終了コードを返す。"""
    runner = Runner(workspace) if runner is None else runner
    RESULTS.mkdir(parents=True, exist_ok=True)
    remove_temp_files()
    log(f"第6回の実行を始めます（全 {len(items)} 項目のうち、済み {sum(is_done(item) for item in items)} 項目。HEAD は {runner.head[:7]}）")

    exit_code = 0
    with exit_on_signals():
        try:
            runner.run(items)
        except DailyLimit as e:
            log(f"Gemini API の 1 日の上限に達したとみなして、止めます（{e}）。")
            log("結果を commit して push してから、日本時間 17 時（1 日の上限が戻る時刻）を過ぎて、同じコマンドをもう一度実行してください。済んだ項目は飛ばして、続きから流します。")
            exit_code = 2
        except KeyboardInterrupt:
            log("Ctrl+C で止めました。済んだ項目は残っています。同じコマンドをもう一度実行すると、続きから流します。")
            exit_code = 130
        except SystemExit as e:  # SIGTERM・SIGHUP
            log("終了の合図を受けて、止めます。同じコマンドをもう一度実行すると、続きから流せます。")
            exit_code = e.code if isinstance(e.code, int) else 1
        except StateError as e:
            log(f"続けられません: {e}")
            exit_code = 1
        finally:
            restored = restore_head(workspace)

    scan_code, scan_lines = scan_report()
    for line in scan_lines:
        log(line)
    if scan_code != 0:
        log("この結果は commit しないでください。上に出た、見つかったファイルを、知らせてください。")
    if runner.failures:
        log(f"失敗した項目が {len(runner.failures)} 件あります。")
        for failure in runner.failures:
            log(f"  - {failure.id}: {failure.reason}（{shown(failure.path)}）")
        log("同じコマンドをもう一度実行すると、失敗した項目だけをやり直します。それでも失敗が残ったら、結果を commit して push してから、上の一覧を知らせてください（.error.txt の中身を貼る必要はありません）。")
    elif exit_code == 0 and restored and scan_code == 0:
        log(f"すべての項目が終わりました。{results_dir()} を commit して push してください。")
    if exit_code == 0 and (runner.failures or scan_code != 0 or not restored):
        exit_code = 1
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第6回の実行スクリプト（Codespaces で 1 回だけ流し、結果を lectures/06/results/ に集める）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="流す項目と、済んだかどうかを表示して終わる（何も起動しない）")
    mode.add_argument("--scan", action="store_true", help="results/ に秘密情報らしき文字列がないかを調べて終わる")
    mode.add_argument("--summarize", action="store_true", help="results/summary.md を作って終わる（サーバーも API も使わない）")
    mode.add_argument("--probe-client", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO)
    if args.list:
        return list_main()
    if args.scan:
        return scan_main()
    if args.summarize:
        return summarize_main()
    if args.probe_client:
        return probe_client_main()
    return run_all()


if __name__ == "__main__":
    sys.exit(main())
