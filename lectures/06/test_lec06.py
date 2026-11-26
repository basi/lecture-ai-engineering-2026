"""第6回の宿題（実装）のテスト（LLM も API キーもネットワークも使わない）。

    uv run pytest -q lectures/06

app/mcp_server.py の make_quiz（出題できない回は、理由と出題できる回を return で返す）と、
lectures/06/quiz_server.py の変種（valueerror・toolerror・schema）を、MCP のクライアントとして呼んで確かめる。
- ツールの定義と呼び出しの結果: 同じプロセスの中でつなぐ（in-memory の）クライアントで確かめる
- モデルに渡る文: FunctionModel（呼び方を決めた偽のモデル）で確かめる
- stdio（サブプロセスとして起動）: サーバーの標準エラー出力と、環境変数の受け渡しだけを、実際に起動して確かめる
"""

import asyncio
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.server.mcpserver import MCPServer
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import env_probe
import quiz_server
from app import tools
from app.mcp_server import quiz_unavailable
from app.mcp_server import server as return_server

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODE = "legacy"  # app/mcp_agent.py の load_mcp_servers と同じ接続方式（実際のエージェントと同じ条件でつなぐ）
TOOL = "glossary_make_quiz"  # エージェントから見たツール名（設定ファイルのサーバー名 "glossary" が前に付く）
ERROR = "Error executing tool make_quiz"  # サーバーが、例外のときに返す本文（ToolError のときは、この後ろに ": 文" が続く）
CANARY = env_probe.CANARY_NAME
CANARY_VALUE = "lec06-canary"  # 目印の値。サーバーが返す JSON に、この値が出てはいけない


def _no_such_lecture(n: int) -> str:
    return f"第{n}回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。"


def _no_terms(n: int) -> str:
    return f"第{n}回の用語は、まだ用語集にありません。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。"


async def _list_tools(server):
    async with Client(server, mode=MODE) as client:
        return await client.list_tools()


def _tools_of(server) -> dict:
    """ツールの名前と定義の対応（登録の順）。"""
    return {t.name: t for t in asyncio.run(_list_tools(server))}


def _definitions(server) -> list[tuple]:
    """ツールごとの（名前, 説明文, 引数のスキーマ）を、登録の順に並べたもの。"""
    return [(t.name, t.description, t.input_schema) for t in asyncio.run(_list_tools(server))]


async def _call_make_quiz(server, lectures):
    async with Client(server, mode=MODE) as client:
        return {n: await client.call_tool("make_quiz", {"lecture": n}, raise_on_error=False) for n in lectures}


def _body(result) -> str:
    assert len(result.content) == 1
    return result.content[0].text


def _assert_quiz(result):
    """出題できたときの結果。エラーではなく、question と answer を持つ JSON が返る（問題は呼ぶたびに変わる）。"""
    assert result.is_error is False
    quiz = json.loads(_body(result))
    assert set(quiz) == {"question", "answer"}
    assert quiz["question"] and quiz["answer"]


# --- ツールの定義 ---


def test_server_lists_tools_in_registration_order():
    assert list(_tools_of(return_server)) == ["search_glossary", "get_lecture_summary", "make_quiz"]


def test_make_quiz_is_the_lecture03_function_as_it_is():
    """説明文と引数のスキーマが、第3回の make_quiz をそのまま登録したときと同じ（説明文は docstring から、引数は型ヒントから決まる）。"""
    reference = MCPServer("reference")
    reference.tool()(tools.make_quiz)
    mine, expected = _tools_of(return_server)["make_quiz"], _tools_of(reference)["make_quiz"]
    assert mine.description == expected.description
    assert mine.input_schema == expected.input_schema


@pytest.mark.parametrize("variant", ["valueerror", "toolerror"])
def test_raising_variants_define_the_same_tools_as_the_return_server(variant):
    assert _definitions(quiz_server.build_server(variant)) == _definitions(return_server)


def test_schema_variant_only_adds_the_range_to_the_lecture_argument():
    base, schema = _tools_of(return_server), _tools_of(quiz_server.build_server("schema"))
    assert list(schema) == list(base)
    assert [t.description for t in schema.values()] == [t.description for t in base.values()]
    for name in ("search_glossary", "get_lecture_summary"):
        assert schema[name].input_schema == base[name].input_schema
    lecture = schema["make_quiz"].input_schema["properties"]["lecture"]
    assert (lecture["minimum"], lecture["maximum"]) == (1, 13)
    # lecture に minimum と maximum が付く以外は、return 版と同じ
    expected = copy.deepcopy(base["make_quiz"].input_schema)
    expected["properties"]["lecture"].update(minimum=1, maximum=13)
    assert schema["make_quiz"].input_schema == expected


async def _identity(server):
    async with Client(server, mode=MODE) as client:
        return client.server_info.name, client.instructions


@pytest.mark.parametrize("variant", quiz_server.VARIANTS)
def test_variants_keep_the_server_name_and_instructions(variant):
    name, instructions = asyncio.run(_identity(return_server))
    assert name == "aie2026-glossary"
    assert instructions
    assert asyncio.run(_identity(quiz_server.build_server(variant))) == (name, instructions)


def test_build_server_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        quiz_server.build_server("nope")


# --- quiz_unavailable ---


def test_glossary_has_terms_only_for_the_expected_lectures():
    """このファイルの前提。用語集に第5回などの用語が増えたら、下の期待（出題できる回と、用語のない回）を直す。"""
    assert sorted({e["lecture"] for e in tools.load_glossary()}) == [1, 2, 3, 4, 6]


@pytest.mark.parametrize("lecture", [99, 0, -1, 14])
def test_quiz_unavailable_tells_that_the_lecture_does_not_exist(lecture):
    assert quiz_unavailable(lecture) == _no_such_lecture(lecture)


@pytest.mark.parametrize("lecture", [5, 13])
def test_quiz_unavailable_tells_that_the_lecture_has_no_terms(lecture):
    assert quiz_unavailable(lecture) == _no_terms(lecture)


@pytest.mark.parametrize("lecture", [1, 2, 3, 4, 6])
def test_quiz_unavailable_is_none_for_lectures_with_terms(lecture):
    assert quiz_unavailable(lecture) is None


# --- 呼び出しの結果（in-memory） ---


def test_return_server_returns_the_sentence_as_a_normal_result():
    results = asyncio.run(_call_make_quiz(return_server, [99, 5, 2]))
    assert (results[99].is_error, _body(results[99])) == (False, _no_such_lecture(99))
    assert (results[5].is_error, _body(results[5])) == (False, _no_terms(5))
    _assert_quiz(results[2])


def test_valueerror_hides_the_sentence_from_the_client():
    results = asyncio.run(_call_make_quiz(quiz_server.build_server("valueerror"), [99, 5, 2]))
    for n in (99, 5):
        assert results[n].is_error is True
        assert _body(results[n]) == ERROR  # 完全一致。理由も、出題できる回も、含まれない
    _assert_quiz(results[2])


def test_toolerror_passes_the_sentence_to_the_client_as_an_error():
    results = asyncio.run(_call_make_quiz(quiz_server.build_server("toolerror"), [99, 5, 2]))
    assert (results[99].is_error, _body(results[99])) == (True, f"{ERROR}: {_no_such_lecture(99)}")
    assert (results[5].is_error, _body(results[5])) == (True, f"{ERROR}: {_no_terms(5)}")
    _assert_quiz(results[2])


def test_schema_variant_rejects_out_of_range_with_a_validation_error():
    results = asyncio.run(_call_make_quiz(quiz_server.build_server("schema"), [99, 5, 2]))
    assert results[99].is_error is True
    body = _body(results[99])
    assert body.startswith(f"{ERROR}: 1 validation error for make_quizArguments")
    assert "validation error" in body
    assert "less than or equal to 13" in body
    assert "確認問題を出せる回" not in body  # 検証エラーには、出題できる回が書かれていない
    # 範囲内で用語のない回は、関数に届くので、return 版と同じ文が返る
    assert (results[5].is_error, _body(results[5])) == (False, _no_terms(5))
    _assert_quiz(results[2])


# --- モデルに渡る文（FunctionModel） ---


def call_make_quiz_99(messages, info):
    """結果を見ずに、毎回、第99回の確認問題を求める偽のモデル。"""
    return ModelResponse(parts=[ToolCallPart(TOOL, {"lecture": 99})])


def call_once_then_answer(messages, info):
    """第99回の確認問題を 1 回だけ求め、結果（成功でも失敗でも）を見たら、呼び直さずに文章で答える偽のモデル。"""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(TOOL, {"lecture": 99})])
    return ModelResponse(parts=[TextPart("第99回はありません。")])


def _parts(messages, kind):
    return [p for m in messages for p in m.parts if isinstance(p, kind)]


def _run_agent(server, model_function):
    """server のツールを持つエージェントを、FunctionModel で実行し、（結果, 例外, 残ったメッセージ）を返す。

    ツール名には glossary_ が付く（app/mcp_agent.py と同じ）。UnexpectedModelBehavior 以外の例外は、そのまま出す。
    """

    async def run():
        agent = Agent(toolsets=[MCPToolset(Client(server, mode=MODE)).prefixed("glossary")])
        with capture_run_messages() as messages:
            try:
                result = await agent.run("第99回の確認問題を出して", model=FunctionModel(model_function))
            except UnexpectedModelBehavior as e:
                return None, e, messages
            return result, None, messages

    return asyncio.run(run())


def test_model_gets_the_sentence_as_a_tool_return_from_the_return_server():
    result, error, messages = _run_agent(return_server, call_once_then_answer)
    assert error is None
    assert [p.content for p in _parts(messages, ToolReturnPart)] == [_no_such_lecture(99)]
    assert _parts(messages, RetryPromptPart) == []
    assert result.output == "第99回はありません。"


@pytest.mark.parametrize(
    ("variant", "body"),
    [("valueerror", ERROR), ("toolerror", f"{ERROR}: {_no_such_lecture(99)}")],
)
def test_two_failures_in_a_row_stop_the_run(variant, body):
    _, error, messages = _run_agent(quiz_server.build_server(variant), call_make_quiz_99)
    assert isinstance(error, UnexpectedModelBehavior)  # 同じツールが 2 回続けて失敗した（既定の max_retries は 1）
    assert len(_parts(messages, ToolCallPart)) == 2
    (retry,) = _parts(messages, RetryPromptPart)  # 残るのは、1 回目の失敗だけ
    assert retry.content == body
    assert retry.model_response() == f"{body}\n\nFix the errors and try again."  # モデルに渡る文。末尾が付く
    assert isinstance(error.__cause__, ModelRetry)
    assert error.__cause__.message == body  # 2 回目の失敗の本文


def test_two_failures_in_a_row_with_the_schema_variant_show_the_validation_error():
    _, error, messages = _run_agent(quiz_server.build_server("schema"), call_make_quiz_99)
    assert isinstance(error, UnexpectedModelBehavior)
    (retry,) = _parts(messages, RetryPromptPart)
    assert "validation error" in retry.content
    assert "less than or equal to 13" in retry.content
    assert "確認問題を出せる回" not in retry.content  # モデルは、出題できる回を知らされない


@pytest.mark.parametrize("variant", quiz_server.VARIANTS)
def test_model_that_answers_after_one_failure_finishes_the_run(variant):
    result, error, messages = _run_agent(quiz_server.build_server(variant), call_once_then_answer)
    assert error is None
    assert len(_parts(messages, RetryPromptPart)) == 1
    assert result.output == "第99回はありません。"


# --- stdio（サブプロセスとして起動する） ---


def test_valueerror_sentence_stays_in_the_server_log_over_stdio(tmp_path):
    log = tmp_path / "server.log"
    transport = StdioTransport(
        sys.executable,
        [str(HERE / "quiz_server.py"), "--variant", "valueerror"],
        cwd=str(ROOT),
        keep_alive=False,
        log_file=log,
    )

    async def call():
        async with Client(transport, mode=MODE) as client:
            return await client.call_tool("make_quiz", {"lecture": 99}, raise_on_error=False)

    result = asyncio.run(call())
    body = _body(result)
    assert result.is_error is True
    assert "第99回はありません" not in body  # クライアントには、文が渡らない
    assert body == ERROR
    # 文は、サーバーの標準エラー出力（トレースバック）にだけ出る。ログは狭い幅で折り返され、罫線（│）が入るので、除いて調べる
    assert "第99回はありません" in re.sub(r"[│\s]", "", log.read_text(encoding="utf-8"))


async def _report_over_stdio(transport) -> str:
    async with Client(transport, mode=MODE) as client:
        result = await client.call_tool("report", {}, raise_on_error=False)
    assert result.is_error is False
    return _body(result)


def test_env_is_not_inherited_by_the_stdio_server(monkeypatch):
    monkeypatch.setenv(CANARY, CANARY_VALUE)
    assert CANARY in os.environ and "GEMINI_API_KEY" in os.environ  # 親（このテスト）の環境には、どちらもある
    transport = StdioTransport(sys.executable, [str(HERE / "env_probe.py")], cwd=str(ROOT), keep_alive=False)
    text = asyncio.run(_report_over_stdio(transport))
    report = json.loads(text)
    # 子（サーバー）に渡るのは、絞られた環境変数だけ。目印も API キーも、渡らない
    assert report["own_env_has_canary"] is False
    assert CANARY not in report["own_env_names"]
    assert "GEMINI_API_KEY" not in report["own_env_names"]
    assert "PATH" in report["own_env_names"]
    assert CANARY_VALUE not in text  # 値は返さない
    # report["parent"] は、ここでは確かめない。monkeypatch で足した変数は、/proc/PID/environ には出ない
    # （そこにあるのは、プロセスが起動したときの環境）。Linux での確認は、次のテストで行う


# 親役のプロセスが実行するプログラム。env_probe を stdio のサーバーとして起動し、report の結果を標準出力に 1 行で出す
PARENT_CODE = """\
import asyncio
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main():
    transport = StdioTransport(sys.executable, [sys.argv[1]], keep_alive=False)
    async with Client(transport, mode="legacy") as client:
        result = await client.call_tool("report", {}, raise_on_error=False)
    print(result.content[0].text)


asyncio.run(main())
"""


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/PID/environ があるのは Linux だけ（macOS では parent が null になる）")
def test_child_can_read_the_parent_environ_on_linux():
    """同じユーザーの子プロセスは、環境変数を受け継がなくても、親の /proc/PID/environ を読める。

    目印は、親役のプロセスを起動するときの環境に入れる。このテスト自身のプロセスの環境に monkeypatch で足しても、
    /proc/PID/environ には出ないため（そこにあるのは、プロセスが起動したときの環境）。
    """
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/"), CANARY: CANARY_VALUE}
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT_CODE, str(HERE / "env_probe.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        out, err = parent.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        parent.kill()  # 親役と、その子のサーバーを残さない（親役が死ぬと、サーバーの標準入力が閉じて、サーバーも終わる）
        parent.communicate()
        raise
    assert parent.returncode == 0, err
    report = json.loads(out)
    assert report["proc_available"] is True
    assert report["own_env_has_canary"] is False  # 子には、目印は渡っていない
    assert report["parent"] == {"pid": parent.pid, "environ_readable": True, "has_canary": True}  # それでも、親の環境から見つかる
    assert CANARY_VALUE not in out  # 値は返さない
    ancestors = report["ancestors"]
    assert len(ancestors) <= 4
    assert [a["level"] for a in ancestors] == list(range(1, len(ancestors) + 1))
    assert all(set(a) == {"level", "pid", "environ_openable"} for a in ancestors)  # 中身は、返さない


# --- env_probe の /proc の読み方（偽の /proc で確かめる。macOS でも動く） ---


def _fake_proc(root: Path, processes: dict) -> Path:
    """偽の /proc を作る。processes は {pid: (親の pid, environ の中身)}。中身が None のプロセスは、environ を開けない。"""
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    for pid, (ppid, environ) in processes.items():
        (proc / str(pid)).mkdir()
        # コマンド名に空白や ) が入っても、親の pid を読めるかを確かめるため、わざと入れる
        (proc / str(pid) / "stat").write_text(f"{pid} (my (odd) name) S {ppid} {pid} {pid} 0 -1\n", encoding="utf-8")
        if environ is not None:
            (proc / str(pid) / "environ").write_bytes(environ)
    return proc


def _probe_in_process(monkeypatch, proc: Path, ppid: int) -> tuple[str, dict]:
    monkeypatch.setattr(env_probe, "PROC", proc)
    monkeypatch.setattr(env_probe.os, "getppid", lambda: ppid)
    text = env_probe.report()
    return text, json.loads(text)


def test_report_finds_the_canary_in_the_parent_environ_without_returning_values(tmp_path, monkeypatch):
    environ = f"HOME=/home/x\0{CANARY}={CANARY_VALUE}\0GEMINI_API_KEY=not-a-real-secret\0".encode()
    proc = _fake_proc(tmp_path / "with", {4242: (1, environ)})
    text, report = _probe_in_process(monkeypatch, proc, 4242)
    assert report["parent"] == {"pid": 4242, "environ_readable": True, "has_canary": True}
    assert CANARY_VALUE not in text
    assert "not-a-real-secret" not in text

    # 目印と名前が似ているだけの変数は、目印とは見なさない
    lookalike = f"X{CANARY}=1\0{CANARY}_2=1\0".encode()
    proc = _fake_proc(tmp_path / "without", {4242: (1, lookalike)})
    _, report = _probe_in_process(monkeypatch, proc, 4242)
    assert report["parent"] == {"pid": 4242, "environ_readable": True, "has_canary": False}


class _OpenOnly:
    """open して close するだけのファイルの代わり。中身を読もうとすると、失敗する。"""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        raise AssertionError(f"{self.path} の中身を読もうとした（{name}）")


def test_report_walks_up_to_four_ancestors_without_reading_them(tmp_path, monkeypatch):
    opened = []

    def open_only(path, mode="r"):
        """env_probe の open の代わり。開けるかどうかだけを答え、中身を読もうとしたら失敗させる。"""
        if not Path(path).exists():
            raise FileNotFoundError(path)
        opened.append(Path(path).parent.name)
        return _OpenOnly(path)

    monkeypatch.setattr(env_probe, "open", open_only, raising=False)  # env_probe の中の open だけを差し替える
    secret = f"{CANARY}={CANARY_VALUE}".encode()  # 先祖の environ に目印があっても、中身は読まないので、結果に出ない
    chain = {
        4242: (3000, b""),
        3000: (2000, secret),
        2000: (1500, None),  # environ を開けない
        1500: (1200, secret),
        1200: (900, b""),
        900: (1, b""),
    }
    proc = _fake_proc(tmp_path, chain)
    text, report = _probe_in_process(monkeypatch, proc, 4242)
    assert report["ancestors"] == [
        {"level": 1, "pid": 3000, "environ_openable": True},
        {"level": 2, "pid": 2000, "environ_openable": False},
        {"level": 3, "pid": 1500, "environ_openable": True},
        {"level": 4, "pid": 1200, "environ_openable": True},
    ]
    assert opened == ["3000", "1500", "1200"]  # 開いたのは、開ける先祖だけ。読んではいない
    assert CANARY_VALUE not in text


@pytest.mark.parametrize(
    ("ppid", "chain", "expected_pids"),
    [
        # 親（4242）の親が 3000（level 1）で、その親が pid 1（level 2）。pid 1 まで来たら、そこで止める
        (4242, {4242: (3000, b""), 3000: (1, b""), 1: (0, b"")}, [3000, 1]),
        # pid 1 の親が 0 でなくても、pid 1 から先はたどらない
        (4242, {4242: (3000, b""), 3000: (1, b""), 1: (777, b""), 777: (500, b"")}, [3000, 1]),
        (1, {1: (0, b"")}, []),  # 親が pid 1 なら、親の親はいない
    ],
)
def test_report_stops_at_pid_1(tmp_path, monkeypatch, ppid, chain, expected_pids):
    _, report = _probe_in_process(monkeypatch, _fake_proc(tmp_path, chain), ppid)
    assert [a["pid"] for a in report["ancestors"]] == expected_pids


def test_report_without_proc_or_with_an_unreadable_parent(tmp_path, monkeypatch):
    # /proc がないとき（macOS など）
    _, report = _probe_in_process(monkeypatch, tmp_path / "no-proc", 4242)
    assert report["proc_available"] is False
    assert report["parent"] is None
    assert report["ancestors"] is None

    # 親の environ を開けないとき
    proc = _fake_proc(tmp_path / "unreadable", {4242: (1, None)})
    _, report = _probe_in_process(monkeypatch, proc, 4242)
    assert report["parent"] == {"pid": 4242, "environ_readable": False, "has_canary": None}
