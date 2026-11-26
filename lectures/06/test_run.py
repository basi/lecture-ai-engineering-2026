"""lectures/06/run.py のテスト（LLM も API キーもネットワークも使わない）。

    uv run pytest -q lectures/06

run.py は、Codespaces で 1 回だけ流す実行スクリプト。実行中は app/mcp_server.py と app/mcp_servers.json を書き換え、
その状態で pytest も流す。そのとき、このテストも走るので、次のようにする。
- 状態の切り替えと戻しは、一時ディレクトリのコピーで確かめる（本物の 2 つのファイルは書き換えない。conftest が確かめる）
- subprocess は、スタブに差し替える。実際に起動するのは、ネットワークに出ない手元のサーバーと --probe-client だけ
- サーバーは uv ではなく sys.executable で起動する（uv に頼らない）
- sleep は差し替えて、待たない（conftest の waits）。結果は、一時ディレクトリに書く（conftest の isolated_results）
"""

import ast
import contextlib
import io
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest
from mcp.shared.exceptions import MCPError
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import env_probe
import quiz_server
import run
from app.config import api_error_message
from app.mcp_agent import print_tools, print_trace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEAD = "a" * 40  # 見出しに書く git HEAD の代わり（テストでは git を呼ばない）
ERROR = "Error executing tool make_quiz"  # ValueError のときに、クライアントに返る本文（ToolError のときは、後ろに ": 文" が続く）
FIX = "\n\nFix the errors and try again."  # RetryPromptPart.model_response() の末尾


def no_such_lecture(n: int) -> str:
    return f"第{n}回はありません。この講座は第1回〜第13回です。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。"


def no_terms(n: int) -> str:
    return f"第{n}回の用語は、まだ用語集にありません。確認問題を出せる回: 1, 2, 3, 4, 6。利用者に、この中から回を選んでもらってください。"


ITEMS = {item.id: item for item in run.build_items()}


def item_named(id_: str) -> run.Item:
    return ITEMS[id_]


def problem_md() -> str:
    return (HERE / "problem.md").read_text(encoding="utf-8")


def problem_md_configs() -> dict:
    """problem.md の STEP 3・STEP 4 の設定（json のコードブロック）。"""
    blocks = [json.loads(block) for block in re.findall(r"```json\n(.*?)```", problem_md(), re.S)]
    by_servers = {tuple(block["mcpServers"]): block for block in blocks}
    return {"STEP3": by_servers[("glossary", "fetch")], "STEP4": by_servers[("glossary", "fetch", "aws")]}


# --- フィクスチャ ---


@pytest.fixture(scope="module")
def contents():
    """git から取り出した、すべての状態の内容（読むだけ。書き換えない）。"""
    return run.load_contents()


@pytest.fixture
def sandbox(tmp_path, contents):
    """2 つのファイルを HEAD の内容で置いた、一時ディレクトリの Workspace（本物の app/ ではない）。"""
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / run.SERVER).write_bytes(contents.head_server)
    (root / run.CONFIG).write_bytes(contents.head_config)
    return run.Workspace(root, contents)


@pytest.fixture
def runner(sandbox):
    return run.Runner(sandbox, head=HEAD)


class FakeRun:
    """subprocess.run の代わり。呼ばれるたびに、用意した応答を順に、渡されたファイルへ書く。

    応答は（終了コード, stdout, stderr）。終了コードが None なら timeout を、応答が関数なら、その関数（argv, kwargs を受ける）を実行する。
    watch に渡したファイルは、呼ばれた時点の内容を seen に残す（実行の前に、状態が合っているかを確かめるため）。
    """

    def __init__(self, *responses, watch=()):
        self.responses = list(responses)
        self.watch = [Path(path) for path in watch]
        self.calls = []
        self.seen = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        self.seen.append([path.read_bytes() for path in self.watch])
        response = self.responses.pop(0)
        if callable(response):
            return response(argv, kwargs)
        returncode, stdout, stderr = response
        if returncode is None:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        for name, text in (("stdout", stdout), ("stderr", stderr)):
            kwargs[name].write(text)
            kwargs[name].flush()
        return subprocess.CompletedProcess(argv, returncode)


def install(monkeypatch, *responses, watch=()) -> FakeRun:
    fake = FakeRun(*responses, watch=watch)
    monkeypatch.setattr(run.subprocess, "run", fake)
    return fake


def python_server_command(condition: str) -> tuple[str, list[str]]:
    """uv の代わりに、sys.executable でサーバーを起動する（uv に頼らない）。"""
    if condition == "return":
        return sys.executable, ["-m", "app.mcp_server"]
    return sys.executable, [str(HERE / "quiz_server.py"), "--variant", condition]


@pytest.fixture
def python_servers(monkeypatch):
    monkeypatch.setattr(run, "server_command", python_server_command)
    monkeypatch.chdir(ROOT)  # -m app.mcp_server を探せるように、リポジトリ直下で起動する


def call_quiz_99_until_answered(messages, info):
    """第99回の make_quiz を呼ぶ。通常の結果が返ったら、文章で答える（失敗が続くあいだは、呼び続ける）。"""
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart("第99回はありません。")])
    return ModelResponse(parts=[ToolCallPart("glossary_make_quiz", {"lecture": 99})])


# --- 項目の表 ---


EXPECTED_STAGE_A = {
    # id: （状態, 設定, コマンド, timeout）
    "step1_agent_tools": ("S0", "STEP1", ["uv", "run", "python", "-m", "app.agent", "--list-tools"], 180),
    "step1_tools": ("S0", "STEP1", ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"], 180),
    "step3_tools": ("S0", "STEP3", ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"], 180),
    "step4_tools": ("S0", "STEP4", ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"], 180),
    "basic_tools": ("S1", "STEP4", ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"], 180),
    "impl_tools": ("S2", "STEP4", ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"], 180),
    "basic_pytest": ("S1", "STEP4", ["uv", "run", "pytest", "-q"], 600),
    "impl_pytest": ("S2", "STEP4", ["uv", "run", "pytest", "-q"], 600),
    "impl_pytest_lec06": ("S2", "STEP4", ["uv", "run", "pytest", "-q", "lectures/06"], 600),
}
EXPECTED_STAGE_B = {
    # id: （状態, 設定, 質問の名前）
    "step2_t1": ("S0", "STEP1", "step2"),
    "step2_t2": ("S0", "STEP1", "step2"),
    "step3_t1": ("S0", "STEP3", "step3"),
    "step3_t2": ("S0", "STEP3", "step3"),
    "step4_t1": ("S0", "STEP4", "step4"),
    "step4_t2": ("S0", "STEP4", "step4"),
    "basic_t1": ("S1", "STEP4", "basic"),
    "basic_t2": ("S1", "STEP4", "basic"),
    "impl_t1": ("S2", "STEP4", "impl"),
}


def test_the_table_has_38_items_in_the_order_a_b_c():
    items = run.build_items()
    assert len({item.id for item in items}) == len(items) == 38
    assert "".join(item.stage for item in items) == "A" * 14 + "B" * 9 + "C" * 15
    for item in items:
        assert (item.argv is None) == (item.func is not None)  # コマンドか関数の、どちらか
        assert item.uses_api == (item.stage in ("B", "C"))
        assert item.server in (None, *run.SERVER_STATES)
        assert item.config in (None, *run.CONFIG_STATES)


def test_stage_a_commands_match_the_specification():
    stage_a = {item.id: item for item in run.build_items() if item.stage == "A" and item.argv}
    assert list(stage_a) == list(EXPECTED_STAGE_A)
    for id_, (server, config, argv, timeout) in EXPECTED_STAGE_A.items():
        item = stage_a[id_]
        assert (item.server, item.config, list(item.argv), item.timeout) == (server, config, argv, timeout)


def test_stage_a_direct_calls_and_env_probe_come_after_the_commands():
    rest = [item for item in run.build_items() if item.stage == "A" and not item.argv]
    assert [item.id for item in rest] == ["direct_return", "direct_valueerror", "direct_toolerror", "direct_schema", "env_probe"]
    assert [(item.server, item.config) for item in rest[:4]] == [("S2", None)] * 4  # 直接の呼び出しは、S2 の状態で
    assert rest[-1].timeout == 120


def test_stage_b_questions_are_the_ones_in_problem_md():
    problem = problem_md()
    assert set(run.QUESTIONS) == {"step2", "step3", "step4", "basic", "impl"}
    for question in run.QUESTIONS.values():
        assert question in problem  # problem.md と、一字一句同じ
    stage_b = {item.id: item for item in run.build_items() if item.stage == "B"}
    assert list(stage_b) == list(EXPECTED_STAGE_B)
    for id_, (server, config, key) in EXPECTED_STAGE_B.items():
        item = stage_b[id_]
        assert (item.server, item.config) == (server, config)
        assert item.argv == ("uv", "run", "python", "-m", "app.mcp_agent", "--trace", run.QUESTIONS[key])
        assert item.timeout == 300 and item.uses_api


def test_stage_c_questions_and_order():
    assert run.IMPL_QUESTIONS == {"q99": "第99回のクイズを作って", "q5": "第5回のクイズを作って"}
    assert run.IMPL_QUESTIONS["q99"] in problem_md()
    order = [(item.condition, item.question_id, item.trial) for item in run.build_items() if item.stage == "C"]
    assert order == [
        ("return", "q99", 1), ("valueerror", "q99", 1), ("toolerror", "q99", 1),
        ("valueerror", "q99", 2), ("toolerror", "q99", 2), ("return", "q99", 2),
        ("toolerror", "q99", 3), ("return", "q99", 3), ("valueerror", "q99", 3),
        ("schema", "q99", 1), ("schema", "q99", 2), ("schema", "q99", 3),
        ("return", "q5", 1), ("return", "q5", 2), ("return", "q5", 3),
    ]
    assert all(item.server == "S2" and item.config is None for item in run.build_items() if item.stage == "C")


def test_server_commands_and_variants_match_quiz_server():
    assert run.VARIANTS == quiz_server.VARIANTS
    assert run.CONDITIONS == ("return", *quiz_server.VARIANTS)
    assert run.server_command("return") == ("uv", ["run", "python", "-m", "app.mcp_server"])
    for variant in quiz_server.VARIANTS:
        assert run.server_command(variant) == ("uv", ["run", "python", "lectures/06/quiz_server.py", "--variant", variant])


def test_condition_config_names_the_server_glossary():
    path = run.write_condition_config("valueerror")
    assert path == run.RESULTS / "impl/configs/valueerror.json"
    expected = {"mcpServers": {"glossary": {"command": "uv", "args": ["run", "python", "lectures/06/quiz_server.py", "--variant", "valueerror"]}}}
    assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert json.loads(run.write_condition_config("return").read_text(encoding="utf-8"))["mcpServers"]["glossary"]["args"] == ["run", "python", "-m", "app.mcp_server"]


def test_run_py_never_imports_the_files_it_rewrites():
    """run.py が app.mcp_server を import すると、実行中に書き換えたファイルを読み込んでしまう（quiz_server も、それを import する）。"""
    tree = ast.parse(Path(run.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
            imported |= {f"{node.module}.{alias.name}" for alias in node.names}
    assert not {name for name in imported if name.startswith("app.mcp_server") or name.split(".")[0] == "quiz_server"}


# --- 状態（S0・S1・S2 と、STEP 1・3・4） ---


def test_s1_is_s0_plus_one_line_and_matches_the_basic_homework_commit(contents):
    assert contents.servers["S0"] == run.git_show(run.BASE_COMMIT, run.SERVER)
    s0 = contents.servers["S0"].decode("utf-8").splitlines(keepends=True)
    s1 = contents.servers["S1"].decode("utf-8").splitlines(keepends=True)
    todo = [i for i, line in enumerate(s0) if line.rstrip("\n") == run.TODO_LINE]
    assert len(todo) == 1  # S0 に、ちょうど 1 回
    assert s1 == s0[: todo[0] + 1] + ["server.tool()(tools.get_lecture_summary)\n"] + s0[todo[0] + 1 :]  # S0 に 1 行足したもの
    # 基礎の宿題で、この 1 行を最初に足した commit の app/mcp_server.py と、一致する
    log = run.git("log", "-S", "server.tool()(tools.get_lecture_summary)", "--format=%H", "--", run.SERVER)
    oldest = log.stdout.decode("utf-8").split()[-1]  # 新しい順に並ぶので、最後が最も古い
    assert contents.servers["S1"] == run.git_show(oldest, run.SERVER)


def test_s0_does_not_register_get_lecture_summary_and_s1_has_no_make_quiz(contents):
    s0, s1, s2 = (contents.servers[name].decode("utf-8") for name in ("S0", "S1", "S2"))
    assert "tools.get_lecture_summary" not in s0
    assert "server.tool()(tools.get_lecture_summary)" in s1
    assert "make_quiz" not in s0 and "make_quiz" not in s1
    assert "make_quiz" in s2
    assert contents.servers["S2"] == contents.head_server


@pytest.mark.parametrize("count", [0, 2])
def test_build_contents_needs_exactly_one_todo_line(contents, count):
    base = ("x = 1\n" + (run.TODO_LINE + "\n") * count).encode("utf-8")
    with pytest.raises(run.StateError, match="ちょうど 1 つ"):
        run.build_contents(contents.head_server, contents.head_config, base, contents.configs["STEP1"])


def test_head_config_is_step4_of_problem_md_and_step3_has_only_glossary_and_fetch(contents):
    problem = problem_md_configs()
    assert {"mcpServers": run.EXPECTED_SERVERS} == problem["STEP4"]
    assert contents.configs["STEP4"] == contents.head_config
    assert json.loads(contents.head_config) == problem["STEP4"]
    assert list(json.loads(contents.head_config)["mcpServers"]) == ["glossary", "fetch", "aws"]
    step3 = json.loads(contents.configs["STEP3"])
    assert step3 == problem["STEP3"]
    assert list(step3["mcpServers"]) == ["glossary", "fetch"]
    assert contents.configs["STEP3"] == (json.dumps(step3, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert list(json.loads(contents.configs["STEP1"])["mcpServers"]) == ["glossary"]  # 演習の開始時点


def test_build_contents_rejects_a_head_config_that_differs_from_problem_md(contents):
    changed = json.loads(contents.head_config)
    changed["mcpServers"]["fetch"]["args"][0] = "mcp-server-fetch==0.0.0"
    with pytest.raises(run.StateError, match="STEP 4"):
        run.build_contents(contents.head_server, json.dumps(changed).encode("utf-8"), contents.servers["S0"], contents.configs["STEP1"])


# --- 状態の切り替えと戻し（一時ディレクトリのコピー） ---


def test_apply_switches_the_two_files_and_restore_brings_back_head(sandbox, contents):
    for server, config in [("S0", "STEP1"), ("S1", "STEP3"), ("S2", "STEP4"), ("S0", "STEP4")]:
        sandbox.apply(server, config)
        assert sandbox.server_path.read_bytes() == contents.servers[server]
        assert sandbox.config_path.read_bytes() == contents.configs[config]
        assert sandbox.digests() == {run.SERVER: run.sha256_bytes(contents.servers[server]), run.CONFIG: run.sha256_bytes(contents.configs[config])}
    sandbox.apply("S0", "STEP3")
    sandbox.restore()
    assert sandbox.server_path.read_bytes() == contents.head_server
    assert sandbox.config_path.read_bytes() == contents.head_config
    assert sorted(path.name for path in sandbox.server_path.parent.iterdir()) == ["mcp_server.py", "mcp_servers.json"]  # 一時ファイルが残らない


def test_apply_leaves_a_file_alone_when_its_state_is_none(sandbox, contents):
    sandbox.apply("S0", None)
    assert sandbox.server_path.read_bytes() == contents.servers["S0"]
    assert sandbox.config_path.read_bytes() == contents.head_config
    sandbox.apply(None, "STEP1")
    assert sandbox.server_path.read_bytes() == contents.servers["S0"]
    assert sandbox.config_path.read_bytes() == contents.configs["STEP1"]


def test_apply_keeps_the_file_mode(sandbox):
    os.chmod(sandbox.server_path, 0o640)
    sandbox.apply("S0", None)
    assert stat.S_IMODE(sandbox.server_path.stat().st_mode) == 0o640


def test_restore_reports_files_it_could_not_bring_back(sandbox, monkeypatch):
    sandbox.apply("S0", "STEP1")

    def broken(src, dst):
        raise OSError("読み取り専用です")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(run.StateError) as caught:
        sandbox.restore()
    assert "mcp_server.py" in str(caught.value) and "mcp_servers.json" in str(caught.value)  # 片方が失敗しても、もう片方も試した
    assert sorted(path.name for path in sandbox.server_path.parent.iterdir()) == ["mcp_server.py", "mcp_servers.json"]


def test_reconcile_does_nothing_when_both_files_are_at_head(sandbox):
    assert run.reconcile_leftover(sandbox) == []


@pytest.mark.parametrize(("server", "config"), [("S0", "STEP1"), ("S1", "STEP3"), ("S0", "STEP4"), ("S2", "STEP1"), ("S1", "STEP4")])
def test_reconcile_brings_back_a_known_leftover_from_an_interrupted_run(sandbox, contents, server, config):
    sandbox.apply(server, config)  # 中断を模す。途中の状態が、残っている
    messages = run.reconcile_leftover(sandbox)
    assert messages and all("HEAD の内容に戻しました" in message for message in messages)
    assert sandbox.server_path.read_bytes() == contents.head_server
    assert sandbox.config_path.read_bytes() == contents.head_config


def test_reconcile_says_which_file_it_brought_back(sandbox):
    sandbox.apply("S1", "STEP3")
    messages = run.reconcile_leftover(sandbox)
    assert "app/mcp_server.py" in messages[0] and "S1" in messages[0]
    assert "app/mcp_servers.json" in messages[1] and "STEP3" in messages[1]


@pytest.mark.parametrize("which", ["server", "config"])
def test_reconcile_stops_on_a_change_it_does_not_know(sandbox, which):
    path = sandbox.server_path if which == "server" else sandbox.config_path
    path.write_text(path.read_text(encoding="utf-8") + "\n# 自分で直した\n", encoding="utf-8")
    with pytest.raises(run.UnknownChange, match="自分で直した内容でないか"):
        run.reconcile_leftover(sandbox)


def test_reconcile_touches_nothing_when_one_file_is_unknown(sandbox):
    sandbox.apply("S0", None)  # こちらは、既知の途中の状態
    sandbox.config_path.write_text("{}", encoding="utf-8")  # こちらは、既知ではない
    before = (sandbox.server_path.read_bytes(), sandbox.config_path.read_bytes())
    with pytest.raises(run.UnknownChange, match="app/mcp_servers.json"):
        run.reconcile_leftover(sandbox)
    assert (sandbox.server_path.read_bytes(), sandbox.config_path.read_bytes()) == before  # どちらも、書き換えない


def test_reconcile_stops_when_a_file_is_missing(sandbox):
    sandbox.config_path.unlink()
    with pytest.raises(run.UnknownChange):
        run.reconcile_leftover(sandbox)


# --- 環境変数・書き込み・ログ ---


def test_env_for_drops_secrets_and_adds_the_api_key_only_for_llm_runs(monkeypatch):
    secrets = {"GITHUB_TOKEN": "t1", "GH_TOKEN": "t2", "FOO_SECRET": "s", "GEMINI_API_KEY": "k", "db_password": "p", "AWS_CREDENTIAL_FILE": "c", "Some_Key": "x"}
    for name, value in secrets.items():
        monkeypatch.setenv(name, f"{value}-value")
    monkeypatch.setenv("KEEP_ME", "1")
    plain = run.env_for(False)
    assert not secrets.keys() & plain.keys()  # 大文字小文字を区別せず、すべて消える
    assert plain["KEEP_ME"] == "1" and plain["PYTHONUNBUFFERED"] == "1" and "PATH" in plain
    with_api = run.env_for(True)
    assert with_api["GEMINI_API_KEY"] == "k-value"  # LLM を呼ぶ実行にだけ足す
    assert not (secrets.keys() - {"GEMINI_API_KEY"}) & with_api.keys()  # ほかは、消えたまま


def test_atomic_write_replaces_the_file_without_leaving_a_temporary_file(tmp_path):
    path = tmp_path / "sub" / "x.txt"
    run.atomic_write(path, "新しい\n")
    assert path.read_text(encoding="utf-8") == "新しい\n"
    assert [p.name for p in path.parent.iterdir()] == ["x.txt"]


def test_atomic_write_keeps_the_old_file_when_it_cannot_finish(tmp_path, monkeypatch):
    path = tmp_path / "x.txt"
    path.write_text("古い\n", encoding="utf-8")

    def broken(src, dst):
        raise OSError("途中で止まった")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        run.atomic_write(path, "新しい\n")
    assert path.read_text(encoding="utf-8") == "古い\n"  # 壊れたファイルを残さない
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]  # 一時ファイルも残さない


def test_remove_temp_files_clears_what_an_interrupted_write_left(tmp_path):
    (run.RESULTS / "cli").mkdir(parents=True)
    leftover = run.RESULTS / "cli" / ".step1_tools.txt.abc123.tmp"
    keep = run.RESULTS / "cli" / "step1_tools.txt"
    leftover.write_text("途中", encoding="utf-8")
    keep.write_text("済み", encoding="utf-8")
    run.remove_temp_files()
    assert not leftover.exists() and keep.exists()


def test_log_prints_and_appends_the_same_line_without_secrets(monkeypatch, capsys):
    monkeypatch.setenv("MY_API_TOKEN", "secret-value-123")
    run.log("項目 secret-value-123 を実行")
    run.log("二行目")
    out = capsys.readouterr().out
    lines = (run.RESULTS / "run.log").read_text(encoding="utf-8").splitlines()
    assert out.splitlines() == lines  # 画面と run.log は、同じ内容
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d 項目 \*\*\* を実行", lines[0])
    assert "secret-value-123" not in out and lines[1].endswith("二行目")


def test_log_keeps_each_entry_on_one_line(capsys):
    run.log("一行目\n二行目\r\n三行目")  # 例外の文に、改行があっても
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.endswith("一行目 二行目  三行目\n")
    assert (run.RESULTS / "run.log").read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_termination_signals_raise_system_exit_and_the_old_handler_comes_back(name):
    signum = getattr(signal, name, None)
    if signum is None:
        pytest.skip(f"{name} がない環境")
    before = signal.getsignal(signum)
    with run.exit_on_signals():
        with pytest.raises(SystemExit) as caught:
            signal.raise_signal(signum)
        assert caught.value.code == 128 + signum
    assert signal.getsignal(signum) == before


# --- CLI の結果の分類 ---


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "expected"),
    [
        (0, "回答\n", "", "ok"),
        (1, api_error_message(429) + "\n", "", "rate_limited"),
        (1, api_error_message(503) + "\n", "", "overloaded"),
        (1, "MCP サーバーに接続できませんでした: boom\napp/mcp_servers.json の内容を確認してください。\n", "", "mcp_connect"),
        (1, "", "Traceback (most recent call last):\n  ...\npydantic_ai.exceptions.UnexpectedModelBehavior: Tool exceeded max retries\n", "stopped"),
        (1, "何か\n", "ValueError: boom\n", "error"),
        (None, "", "", "error"),  # timeout
        (None, api_error_message(429) + "\n", "", "error"),  # timeout は、出力に関わらず error
    ],
)
def test_classify_cli_sorts_the_six_kinds(returncode, stdout, stderr, expected):
    assert run.classify_cli(returncode, stdout, stderr) == expected


def test_pytest_items_are_judged_by_the_exit_code_only():
    """pytest の失敗の表示に、テストの中の文面（レート制限など）が出ても、やり直しの対象にしない。"""
    item = item_named("impl_pytest")
    assert run.outcome_of(item, 1, api_error_message(429), "") == "error"
    assert run.outcome_of(item, 1, "", "UnexpectedModelBehavior") == "error"
    assert run.outcome_of(item, 0, "", "") == "ok"
    assert run.outcome_of(item_named("step2_t1"), 1, api_error_message(429), "") == "rate_limited"


# --- コマンドの実行（本物のプロセスで） ---


def child_env() -> dict:
    return {"PATH": os.environ["PATH"], "PYTHONUTF8": "1"}


def test_execute_runs_a_real_process_in_the_repo_and_reads_its_output_from_files():
    code = "import os, sys; print('標準出力', os.getcwd()); print('標準エラー', file=sys.stderr); sys.exit(3)"
    returncode, stdout, stderr = run.execute([sys.executable, "-S", "-c", code], 30, child_env())
    assert returncode == 3 and stderr == "標準エラー\n"
    assert Path(stdout.split(" ", 1)[1].strip()).resolve() == run.REPO.resolve()  # 作業ディレクトリは、リポジトリ直下


def test_execute_reports_a_timeout_with_the_output_so_far():
    code = "import time; print('始まった', flush=True); time.sleep(30)"
    returncode, stdout, stderr = run.execute([sys.executable, "-S", "-c", code], 3, child_env())  # -S: site を読まず、負荷が高くても、すぐ起動する
    assert returncode is None and stdout == "始まった\n"


def test_execute_reports_a_command_that_cannot_start():
    returncode, stdout, stderr = run.execute(["no-such-command-for-lec06"], 5, child_env())
    assert returncode is None and stdout == "" and "FileNotFoundError" in stderr


def test_execute_gives_the_child_only_the_environment_it_is_given(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t" * 20)
    code = "import os; print(sorted(name for name in os.environ if 'TOKEN' in name or name == 'GEMINI_API_KEY'))"
    assert run.execute([sys.executable, "-S", "-c", code], 30, run.env_for(False))[1] == "[]\n"
    assert run.execute([sys.executable, "-S", "-c", code], 30, run.env_for(True))[1] == "['GEMINI_API_KEY']\n"


# --- CLI の項目の実行（subprocess をスタブにして） ---


def test_a_rate_limited_run_is_repeated_after_a_wait_and_the_retry_is_recorded(runner, sandbox, contents, monkeypatch, waits):
    fake = install(monkeypatch, (1, api_error_message(429) + "\n", ""), (0, "回答\n[tool call]   x({})\n", ""))
    item = item_named("step2_t1")
    runner.run_item(item)
    assert len(fake.calls) == 2
    assert waits == [run.RETRY_WAIT]  # 最初の LLM の実行なので、間をあける 5 秒はなく、やり直す前の 60 秒だけ
    lines = run.done_path(item).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "$ " + shlex.join(item.argv)
    assert lines[0].startswith("$ uv run python -m app.mcp_agent --trace '")
    assert lines[1] == "# 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 1（glossary だけ）"
    s0, step1 = run.sha256_bytes(contents.servers["S0"]), run.sha256_bytes(contents.configs["STEP1"])
    assert lines[2] == f"# sha256: app/mcp_server.py={s0} app/mcp_servers.json={step1}"
    assert lines[3] == f"# git HEAD: {HEAD}"
    assert re.fullmatch(
        r"# 開始: \d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+09:00／所要: \d+\.\d 秒／終了コード: 0／やり直し: 1 回／結果: 成功", lines[4]
    )
    assert lines[5:] == ["", "回答", "[tool call]   x({})"]
    assert not run.error_path(item).exists() and runner.failures == []
    assert runner.rate_limit_streak == 0  # 成功したので、数を戻した


def test_three_rate_limits_in_a_row_stop_the_run_with_the_daily_limit(runner, monkeypatch, waits):
    limited = (1, api_error_message(429) + "\n", "")
    fake = install(monkeypatch, limited, limited, limited)
    item = item_named("step2_t1")
    with pytest.raises(run.DailyLimit):
        runner.run_item(item)
    assert len(fake.calls) == 3
    assert waits == [run.RETRY_WAIT, run.RETRY_WAIT]  # 3 回目で止めるので、そのあとは待たない
    assert run.error_path(item).exists() and not run.done_path(item).exists()


def test_rate_limits_are_counted_across_items_and_a_success_resets_the_count(runner, monkeypatch):
    limited = (1, api_error_message(429) + "\n", "")
    install(
        monkeypatch,
        limited, limited, (0, "回答\n", ""),  # step2_t1: 2 回続いたが、成功したので、数を戻す
        limited, limited, (1, "別のエラー\n", ""),  # step2_t2: 2 回続き、別の失敗で終わる（数は戻らない）
        limited,  # step3_t1: これで 3 回続いた
    )
    runner.run_item(item_named("step2_t1"))
    assert runner.rate_limit_streak == 0
    runner.run_item(item_named("step2_t2"))
    assert runner.rate_limit_streak == 2 and [f.id for f in runner.failures] == ["step2_t2"]
    with pytest.raises(run.DailyLimit):
        runner.run_item(item_named("step3_t1"))


@pytest.mark.parametrize(
    ("response", "outcome_line"),
    [
        ((1, api_error_message(503) + "\n", ""), "失敗（モデルが混み合っている）"),
        ((1, "MCP サーバーに接続できませんでした: x\n", ""), "失敗（MCP サーバーに接続できない）"),
    ],
)
def test_overloaded_and_mcp_connect_are_repeated_up_to_three_times_then_listed_as_failures(runner, monkeypatch, waits, response, outcome_line):
    fake = install(monkeypatch, *[response] * 4)
    item = item_named("step4_t1")
    runner.run_item(item)
    assert len(fake.calls) == 1 + run.MAX_RETRIES
    assert waits == [run.RETRY_WAIT] * run.MAX_RETRIES
    assert f"やり直し: {run.MAX_RETRIES} 回／結果: {outcome_line}" in run.error_path(item).read_text(encoding="utf-8")
    assert [f.id for f in runner.failures] == ["step4_t1"] and not run.done_path(item).exists()


def test_a_stopped_run_is_saved_under_the_real_name_and_not_repeated(runner, monkeypatch):
    stderr = "Traceback (most recent call last):\n  ...\npydantic_ai.exceptions.UnexpectedModelBehavior: Tool 'glossary_make_quiz' exceeded max retries count of 1\n"
    fake = install(monkeypatch, (1, "[server] glossary: コマンドで起動（uv run python -m app.mcp_server）\n", stderr))
    item = item_named("impl_t1")
    runner.run_item(item)
    assert len(fake.calls) == 1  # やり直さない
    text = run.done_path(item).read_text(encoding="utf-8")
    assert "終了コード: 1／やり直し: 0 回／結果: 停止（UnexpectedModelBehavior）" in text
    assert f"--- stderr ---\n{stderr}" in text
    assert not run.error_path(item).exists() and runner.failures == []


def test_an_error_is_saved_as_error_txt_and_listed_and_the_next_item_still_runs(runner, monkeypatch):
    install(monkeypatch, (1, "出力\n", "boom\n"), (0, "OK\n", ""))
    first, second = item_named("step1_tools"), item_named("step3_tools")
    runner.run_item(first)
    runner.run_item(second)
    assert not run.done_path(first).exists()  # 本来の名前では、保存しない
    error = run.error_path(first).read_text(encoding="utf-8")
    assert error.splitlines()[0] == "$ uv run python -m app.mcp_agent --list-tools"
    assert "終了コード: 1／やり直し: 0 回／結果: 失敗" in error and "boom" in error
    assert [(f.id, f.path) for f in runner.failures] == [("step1_tools", run.error_path(first))]
    assert run.done_path(second).exists()  # 先へ進んだ


def test_a_success_removes_the_old_error_file(runner, monkeypatch):
    item = item_named("step1_tools")
    install(monkeypatch, (1, "", "boom\n"))
    runner.run_item(item)
    assert run.error_path(item).exists()
    install(monkeypatch, (0, "一覧\n", ""))
    runner.run_item(item)
    assert run.done_path(item).exists() and not run.error_path(item).exists()


def test_a_timeout_is_an_error_and_a_failure_to_start_is_too(runner, monkeypatch):
    install(monkeypatch, (None, "", ""))
    runner.run_item(item_named("impl_pytest"))
    assert "終了コード: なし（timeout または起動の失敗）／やり直し: 0 回／結果: 失敗（timeout）" in run.error_path(item_named("impl_pytest")).read_text(encoding="utf-8")

    def cannot_start(argv, kwargs):
        raise FileNotFoundError("uv")

    install(monkeypatch, cannot_start)
    runner.run_item(item_named("step1_tools"))
    assert "FileNotFoundError: uv" in run.error_path(item_named("step1_tools")).read_text(encoding="utf-8")
    assert [f.id for f in runner.failures] == ["impl_pytest", "step1_tools"]


def test_pytest_items_are_never_repeated_even_if_the_output_looks_like_a_rate_limit(runner, monkeypatch, waits):
    fake = install(monkeypatch, (1, api_error_message(429) + "\n", ""))
    runner.run_item(item_named("basic_pytest"))
    assert len(fake.calls) == 1 and waits == []
    assert [f.id for f in runner.failures] == ["basic_pytest"]


def test_a_finished_item_is_skipped(runner, monkeypatch):
    item = item_named("step1_tools")
    run.atomic_write(run.done_path(item), "済み\n")
    fake = install(monkeypatch)  # 呼ばれたら、応答が足りなくて失敗する
    runner.run_item(item)
    assert fake.calls == [] and runner.failures == []
    assert run.done_path(item).read_text(encoding="utf-8") == "済み\n"


def test_a_command_runs_in_the_repo_with_files_for_output_and_the_right_environment(runner, sandbox, contents, monkeypatch, fake_api_key):
    monkeypatch.setenv("GITHUB_TOKEN", "t" * 20)
    fake = install(monkeypatch, (0, "x\n", ""), (0, "y\n", ""), watch=[sandbox.server_path, sandbox.config_path])
    runner.run_item(item_named("step3_tools"))  # API を使わない
    runner.run_item(item_named("basic_t1"))  # API を使う
    (argv1, kw1), (argv2, kw2) = fake.calls
    assert argv1 == ["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"]
    assert argv2 == ["uv", "run", "python", "-m", "app.mcp_agent", "--trace", "第2回では何を学びますか？"]
    assert kw1["cwd"] == kw2["cwd"] == run.REPO
    assert (kw1["timeout"], kw2["timeout"]) == (180, 300)
    for kw in (kw1, kw2):
        assert kw["stdout"] is not subprocess.PIPE and hasattr(kw["stdout"], "fileno")  # パイプを通さず、ファイルへ直接
        assert kw["stderr"] is not subprocess.PIPE and hasattr(kw["stderr"], "fileno")
        assert "GITHUB_TOKEN" not in kw["env"] and kw["env"]["PYTHONUNBUFFERED"] == "1"
    assert "GEMINI_API_KEY" not in kw1["env"] and kw2["env"]["GEMINI_API_KEY"] == fake_api_key
    # 実行の前に、状態が合っている
    assert fake.seen == [
        [contents.servers["S0"], contents.configs["STEP3"]],
        [contents.servers["S1"], contents.configs["STEP4"]],
    ]


def test_llm_runs_are_spaced_by_five_seconds_except_the_first(runner, monkeypatch, waits):
    install(monkeypatch, (0, "a\n", ""), (0, "b\n", ""), (0, "c\n", ""), (0, "d\n", ""))
    runner.run_item(item_named("step1_tools"))  # API を使わない項目は、間をあけない
    for id_ in ("step2_t1", "step2_t2", "step3_t1"):
        runner.run_item(item_named(id_))
    assert waits == [run.LLM_INTERVAL, run.LLM_INTERVAL]


# --- 実行全体（一時ディレクトリのコピーで。終わるとき、止めたときも、HEAD に戻る） ---


def is_at_head(sandbox, contents) -> bool:
    return sandbox.server_path.read_bytes() == contents.head_server and sandbox.config_path.read_bytes() == contents.head_config


def execute(sandbox, *ids):
    return run.execute_all(sandbox, [item_named(id_) for id_ in ids], run.Runner(sandbox, head=HEAD))


def test_execute_all_returns_zero_runs_in_the_intermediate_states_and_ends_at_head(sandbox, contents, monkeypatch, capsys):
    fake = install(monkeypatch, (0, "一覧\n", ""), (0, "一覧\n", ""), watch=[sandbox.server_path, sandbox.config_path])
    assert execute(sandbox, "step1_tools", "step3_tools") == 0
    assert fake.seen == [[contents.servers["S0"], contents.configs["STEP1"]], [contents.servers["S0"], contents.configs["STEP3"]]]
    assert is_at_head(sandbox, contents)
    out = capsys.readouterr().out
    assert "HEAD の内容に戻しました（sha256 を確かめました）" in out
    assert "走査: 秘密情報らしき文字列は見つかりませんでした（3 ファイル）" in out  # 結果が 2 つと、run.log
    assert out.splitlines() == (run.RESULTS / "run.log").read_text(encoding="utf-8").splitlines()  # 画面と run.log は、同じ内容


def test_execute_all_lists_the_failures_and_returns_one(sandbox, contents, monkeypatch, capsys):
    install(monkeypatch, (1, "", "boom\n"), (0, "OK\n", ""))
    assert execute(sandbox, "step1_tools", "step3_tools") == 1
    out = capsys.readouterr().out
    assert "失敗した項目が 1 件あります。" in out
    assert "  - step1_tools: 失敗（results/cli/step1_tools.error.txt）" in out
    assert "知らせてください" in out
    assert is_at_head(sandbox, contents)


def test_execute_all_returns_two_on_the_daily_limit_and_restores_head(sandbox, contents, monkeypatch, capsys):
    limited = (1, api_error_message(429) + "\n", "")
    install(monkeypatch, limited, limited, limited)
    assert execute(sandbox, "step2_t1") == 2
    assert "1 日の上限" in capsys.readouterr().out
    assert is_at_head(sandbox, contents)


def test_execute_all_returns_130_on_ctrl_c_and_restores_head(sandbox, contents, monkeypatch, capsys):
    def interrupt(argv, kwargs):
        raise KeyboardInterrupt

    fake = install(monkeypatch, interrupt, watch=[sandbox.server_path, sandbox.config_path])
    assert execute(sandbox, "step1_tools") == 130
    assert fake.seen == [[contents.servers["S0"], contents.configs["STEP1"]]]  # 途中の状態のときに止まった
    assert "同じコマンドをもう一度実行すると、続きから流します" in capsys.readouterr().out
    assert is_at_head(sandbox, contents)


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_execute_all_restores_head_when_it_gets_a_termination_signal(sandbox, contents, monkeypatch, name):
    signum = getattr(signal, name, None)
    if signum is None:
        pytest.skip(f"{name} がない環境")
    before = signal.getsignal(signum)

    def terminate(argv, kwargs):
        signal.raise_signal(signum)

    fake = install(monkeypatch, terminate, watch=[sandbox.server_path, sandbox.config_path])
    assert execute(sandbox, "step1_tools", "step3_tools") == 128 + signum
    assert len(fake.calls) == 1  # 1 つ目で止まり、2 つ目は流さない
    assert fake.seen == [[contents.servers["S0"], contents.configs["STEP1"]]]
    assert is_at_head(sandbox, contents)
    assert signal.getsignal(signum) == before


def test_execute_all_reports_a_secret_found_in_the_results_without_printing_it(sandbox, contents, monkeypatch, capsys):
    leaked = "AIza" + "q" * 35
    install(monkeypatch, (0, f"出力 {leaked}\n", ""))
    assert execute(sandbox, "step1_tools") == 1
    out = capsys.readouterr().out
    assert "秘密情報らしき文字列が見つかりました: results/cli/step1_tools.txt（Google API キーの形）" in out
    assert "commit しないでください" in out
    assert leaked not in out and leaked not in (run.RESULTS / "run.log").read_text(encoding="utf-8")


def test_execute_all_tells_how_to_restore_when_head_cannot_be_brought_back(sandbox, monkeypatch, capsys):
    def broken_restore():
        raise run.StateError("書けません")

    monkeypatch.setattr(sandbox, "restore", broken_restore)
    install(monkeypatch, (0, "一覧\n", ""))
    assert execute(sandbox, "step1_tools") == 1
    assert "git checkout HEAD -- app/mcp_server.py app/mcp_servers.json" in capsys.readouterr().out


def test_execute_all_continues_from_the_finished_items(sandbox, contents, monkeypatch):
    run.atomic_write(run.done_path(item_named("step1_tools")), "済み\n")
    fake = install(monkeypatch, (0, "一覧\n", ""))
    assert execute(sandbox, "step1_tools", "step3_tools") == 0
    assert [argv for argv, _ in fake.calls] == [["uv", "run", "python", "-m", "app.mcp_agent", "--list-tools"]]
    assert run.done_path(item_named("step1_tools")).read_text(encoding="utf-8") == "済み\n"


def test_execute_all_stops_when_the_state_cannot_be_written(sandbox, contents, monkeypatch, capsys):
    def broken_apply(server, config):
        raise run.StateError("app/mcp_server.py を書き換えられませんでした: 読み取り専用です")

    monkeypatch.setattr(sandbox, "apply", broken_apply)
    fake = install(monkeypatch)
    assert execute(sandbox, "step1_tools", "step3_tools") == 1
    assert fake.calls == []  # 状態を合わせられないので、何も流さない
    assert "続けられません: app/mcp_server.py を書き換えられませんでした" in capsys.readouterr().out
    assert is_at_head(sandbox, contents)


def test_run_all_stops_with_one_when_the_preflight_fails(monkeypatch):
    monkeypatch.setattr(run, "preflight", lambda: None)
    monkeypatch.setattr(run, "execute_all", lambda *args: pytest.fail("前処理が通らないのに、流そうとした"))
    assert run.run_all() == 1


def test_main_without_options_runs_everything_on_the_repo_workspace(monkeypatch, contents, tmp_path):
    captured = {}
    monkeypatch.setattr(run, "preflight", lambda: contents)
    monkeypatch.setattr(run, "execute_all", lambda workspace, items: captured.update(workspace=workspace, items=items) or 7)
    monkeypatch.chdir(tmp_path)
    assert run.main([]) == 7  # execute_all の終了コードが、そのまま返る
    assert Path.cwd().resolve() == run.REPO.resolve()  # 最初に、リポジトリ直下へ移る
    assert captured["workspace"].root == run.REPO and captured["workspace"].contents is contents
    assert captured["items"] == run.build_items()


# --- 前処理 ---


@pytest.fixture
def git_answers(monkeypatch):
    answers = {}
    monkeypatch.setattr(run, "git_text", lambda *args: answers.get(args))
    return answers


HEAD_KEY = ("rev-parse", "HEAD")
REMOTE_KEY = ("rev-parse", "--verify", "-q", "origin/lec06")


def test_check_branch_passes_when_head_is_origin_lec06(git_answers):
    git_answers.update({HEAD_KEY: "1" * 40, REMOTE_KEY: "1" * 40})
    assert run.check_branch() is True


def test_check_branch_tells_how_to_catch_up_when_they_differ(git_answers, capsys):
    git_answers.update({HEAD_KEY: "1" * 40, REMOTE_KEY: "2" * 40})
    assert run.check_branch() is False
    out = capsys.readouterr().out
    assert "git switch lec06" in out and "git pull --ff-only" in out
    assert "git branch -m lec06 lec06-class" in out  # 授業で作った lec06 が残っている場合


def test_check_branch_asks_for_a_fetch_when_origin_lec06_is_missing(git_answers, capsys):
    git_answers.update({HEAD_KEY: "1" * 40, REMOTE_KEY: None})
    assert run.check_branch() is False
    assert "git fetch origin" in capsys.readouterr().out


def test_check_base_commit_asks_git_whether_it_is_an_ancestor_of_head(monkeypatch, capsys):
    calls = []

    def fake_git(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, len(calls) - 1, b"", b"")  # 1 回目は 0（祖先）、2 回目は 1（祖先ではない）

    monkeypatch.setattr(run, "git", fake_git)
    assert run.check_base_commit() is True
    assert run.check_base_commit() is False
    assert calls[0] == ("merge-base", "--is-ancestor", "dccf422d6671eff654300cbb4f22a9b67165989d", "HEAD")
    assert len(run.BASE_COMMIT) == 40 and "第6回の開始時点（dccf422）" in capsys.readouterr().out


def test_check_model_pins_the_default_model(monkeypatch, capsys):
    monkeypatch.delenv("AIE_MODEL", raising=False)
    assert run.check_model() is True
    monkeypatch.setenv("AIE_MODEL", run.DEFAULT_MODEL)
    assert run.check_model() is True
    monkeypatch.setenv("AIE_MODEL", "gemini-3.5-flash")
    assert run.check_model() is False
    assert "unset AIE_MODEL" in capsys.readouterr().out


def test_check_clean_looks_at_tracked_files_under_app_and_lectures_06(monkeypatch, capsys):
    calls = []
    dirty = iter([False, True])

    def fake_git(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1 if next(dirty) else 0, b"", b"")

    monkeypatch.setattr(run, "git", fake_git)
    assert run.check_clean() is True
    assert run.check_clean() is False
    assert calls[0] == ("diff", "--quiet", "HEAD", "--", "app", "lectures/06")
    assert "commit していない変更" in capsys.readouterr().out


def test_check_tools_needs_uv_uvx_and_the_api_key(monkeypatch, capsys):
    monkeypatch.setattr(run.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert run.check_tools() is True  # conftest が、偽のキーを入れている
    monkeypatch.setattr(run.shutil, "which", lambda name: None if name == "uvx" else f"/usr/bin/{name}")
    assert run.check_tools() is False
    assert "uvx が見つかりません" in capsys.readouterr().out
    monkeypatch.setattr(run.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert run.check_tools() is False
    with pytest.raises(SystemExit) as expected:
        run.get_api_key()
    assert capsys.readouterr().out.endswith(f"{expected.value.code}\n")  # app/config.py の get_api_key と、同じ案内（値は含まれない）


def stub_checks(monkeypatch, contents, failing=None):
    for name in ("check_branch", "check_base_commit", "check_model", "check_clean", "check_tools"):
        monkeypatch.setattr(run, name, lambda name=name: name != failing)
    monkeypatch.setattr(run, "load_contents", lambda: contents)
    monkeypatch.setattr(run, "reconcile_leftover", lambda workspace: ["前回の途中の状態を戻しました"])


def test_preflight_returns_the_contents_and_tells_what_it_brought_back(monkeypatch, contents, capsys):
    stub_checks(monkeypatch, contents)
    assert run.preflight() is contents
    assert "前回の途中の状態を戻しました" in capsys.readouterr().out


@pytest.mark.parametrize("failing", ["check_branch", "check_base_commit", "check_model", "check_clean", "check_tools"])
def test_preflight_returns_none_when_a_check_fails(monkeypatch, contents, failing):
    stub_checks(monkeypatch, contents, failing)
    assert run.preflight() is None


def test_preflight_stops_on_an_unknown_change_and_on_a_broken_head_config(monkeypatch, contents, capsys):
    stub_checks(monkeypatch, contents)

    def unknown(workspace):
        raise run.UnknownChange("app/mcp_server.py に、知らない変更があります")

    monkeypatch.setattr(run, "reconcile_leftover", unknown)
    assert run.preflight() is None
    assert "知らない変更があります" in capsys.readouterr().out

    def broken():
        raise run.StateError("HEAD の設定が違います")

    monkeypatch.setattr(run, "load_contents", broken)
    assert run.preflight() is None
    assert "HEAD の設定が違います" in capsys.readouterr().out


def test_preflight_checks_the_clean_state_after_bringing_back_a_leftover(monkeypatch, contents):
    """前回の途中の状態を戻してから、未コミットの変更がないかを調べる（途中の状態が、未コミットの変更に見えないように）。"""
    order = []
    stub_checks(monkeypatch, contents)
    monkeypatch.setattr(run, "reconcile_leftover", lambda workspace: order.append("reconcile") or [])
    monkeypatch.setattr(run, "check_clean", lambda: order.append("clean") or True)
    assert run.preflight() is contents
    assert order == ["reconcile", "clean"]


# --- 段 A: 直接の呼び出し（uv ではなく sys.executable で、サーバーを起動する） ---


def tool_names(record):
    return [tool["name"] for tool in record["tools"]]


def run_direct(condition):
    runner = run.Runner(head=HEAD)
    item = item_named(f"direct_{condition}")
    runner.run_item(item)
    assert runner.failures == [], runner.failures
    return json.loads(run.done_path(item).read_text(encoding="utf-8"))


@pytest.mark.parametrize("condition", run.CONDITIONS)
def test_direct_call_records_the_tools_the_calls_and_the_fake_model_run(python_servers, condition):
    record = run_direct(condition)
    assert set(record) == {"condition", "server_command", "started_at", "duration_s", "git_head", "files_sha256", "tools", "calls", "agent"}
    assert record["condition"] == condition and record["git_head"] == HEAD
    assert record["server_command"] == [sys.executable, *python_server_command(condition)[1]]
    assert tool_names(record) == ["search_glossary", "get_lecture_summary", "make_quiz"]
    for tool in record["tools"]:
        assert set(tool) == {"name", "description", "inputSchema"} and tool["description"] and tool["inputSchema"]["type"] == "object"
    assert list(record["calls"]) == ["99", "5", "2"]
    # 第2回は、どの条件でも出題できる（問題は呼ぶたびに変わるので、形だけ確かめる）
    assert record["calls"]["2"]["isError"] is False
    assert set(json.loads(record["calls"]["2"]["text"])) == {"question", "answer"}
    assert set(record["agent"]) == {"output", "tool_calls", "tool_returns", "retry_prompts", "exception"}
    assert record["agent"]["tool_calls"][0] == {"tool": "glossary_make_quiz", "args": {"lecture": 99}}
    assert (run.RESULTS / "direct" / f"{condition}.server.log").exists()
    assert (run.RESULTS / "direct" / f"{condition}.agent.server.log").exists()


def test_direct_call_with_the_return_server_gets_the_sentence_as_a_normal_result(python_servers):
    record = run_direct("return")
    assert record["calls"]["99"] == {"isError": False, "text": no_such_lecture(99)}
    assert record["calls"]["5"] == {"isError": False, "text": no_terms(5)}
    agent = record["agent"]
    assert [r["content"] for r in agent["tool_returns"]] == [no_such_lecture(99)]  # 1 回で成功するので、呼び直さない
    assert len(agent["tool_calls"]) == 1 and agent["retry_prompts"] == []
    assert agent["exception"] is None and agent["output"]


def test_direct_call_with_the_valueerror_server_hides_the_sentence(python_servers):
    record = run_direct("valueerror")
    for lecture in ("99", "5"):
        assert record["calls"][lecture] == {"isError": True, "text": ERROR}  # 完全一致。理由も、出題できる回も、含まれない
    agent = record["agent"]
    assert len(agent["tool_calls"]) == 2 and agent["tool_returns"] == [] and agent["output"] is None
    assert agent["retry_prompts"] == [{"tool": "glossary_make_quiz", "content": ERROR, "sent_to_model": ERROR + FIX}]
    exception = agent["exception"]
    assert exception["type"] == "UnexpectedModelBehavior" and "exceeded max retries count of 1" in exception["message"]
    assert (exception["cause_type"], exception["cause_message"]) == ("ModelRetry", ERROR)  # 2 回目の失敗の本文
    # 文は、サーバーの標準エラー出力（トレースバック）にだけ出る。ログは狭い幅で折り返され、罫線（│）が入るので、除いて調べる
    log = (run.RESULTS / "direct" / "valueerror.server.log").read_text(encoding="utf-8")
    assert "第99回はありません" in re.sub(r"[│\s]", "", log)


def test_direct_call_with_the_toolerror_server_passes_the_sentence_as_an_error(python_servers):
    record = run_direct("toolerror")
    assert record["calls"]["99"] == {"isError": True, "text": f"{ERROR}: {no_such_lecture(99)}"}
    assert record["calls"]["5"] == {"isError": True, "text": f"{ERROR}: {no_terms(5)}"}
    agent = record["agent"]
    body = f"{ERROR}: {no_such_lecture(99)}"
    assert agent["retry_prompts"] == [{"tool": "glossary_make_quiz", "content": body, "sent_to_model": body + FIX}]
    assert (agent["exception"]["type"], agent["exception"]["cause_message"]) == ("UnexpectedModelBehavior", body)


def test_direct_call_with_the_schema_server_rejects_out_of_range_with_a_validation_error(python_servers):
    record = run_direct("schema")
    text = record["calls"]["99"]["text"]
    assert record["calls"]["99"]["isError"] is True
    assert text.startswith(f"{ERROR}: 1 validation error for make_quizArguments") and "less than or equal to 13" in text
    assert "確認問題を出せる回" not in text  # 検証エラーには、出題できる回が書かれていない
    assert record["calls"]["5"] == {"isError": False, "text": no_terms(5)}  # 範囲内で用語のない回は、return 版と同じ文
    sent = record["agent"]["retry_prompts"][0]["sent_to_model"]
    assert "validation error" in sent and "less than or equal to 13" in sent and sent.endswith(FIX)
    lecture = next(t for t in record["tools"] if t["name"] == "make_quiz")["inputSchema"]["properties"]["lecture"]
    assert (lecture["minimum"], lecture["maximum"]) == (1, 13)


def test_direct_call_clears_the_server_log_of_a_previous_run(python_servers):
    log = run.RESULTS / "direct" / "toolerror.server.log"
    log.parent.mkdir(parents=True)
    log.write_text("前回の分", encoding="utf-8")
    run_direct("toolerror")
    assert "前回の分" not in log.read_text(encoding="utf-8")


# --- 段 A: env_probe（run.py 自身を、最小限の環境で起動する） ---


def test_probe_client_does_not_hand_the_canary_to_the_server():
    env = run.probe_env()
    assert set(env) <= {"PATH", "HOME", "LANG", "PYTHONUNBUFFERED", env_probe.CANARY_NAME}  # TOKEN や KEY の類は、渡さない
    assert env[env_probe.CANARY_NAME] == run.CANARY_VALUE == "lec06-canary" and env["PYTHONUNBUFFERED"] == "1"
    assert env_probe.CANARY_NAME not in os.environ  # このテストの環境には、目印がない
    completed = subprocess.run(
        [sys.executable, str(run.RUN_PY), "--probe-client"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 1  # 結果は、1 行で出る
    report = json.loads(lines[0])
    assert report["own_env_has_canary"] is False and env_probe.CANARY_NAME not in report["own_env_names"]
    assert "GEMINI_API_KEY" not in report["own_env_names"] and "PATH" in report["own_env_names"]
    assert run.CANARY_VALUE not in completed.stdout + completed.stderr  # 値は、出力のどこにも出ない


def test_run_probe_saves_what_the_client_passed_and_the_report():
    runner = run.Runner(head=HEAD)
    item = item_named("env_probe")
    runner.run_item(item)
    assert runner.failures == []
    record = json.loads(run.done_path(item).read_text(encoding="utf-8"))
    assert run.done_path(item) == run.RESULTS / "security" / "env_probe.json"
    assert set(record) == {"client_env_names", "canary_name", "report"}
    assert record["client_env_names"] == sorted(run.probe_env())
    assert record["canary_name"] == env_probe.CANARY_NAME
    assert record["report"]["own_env_has_canary"] is False
    assert run.CANARY_VALUE not in run.done_path(item).read_text(encoding="utf-8")


def test_run_probe_does_not_save_when_the_canary_value_shows_up(runner, monkeypatch):
    install(monkeypatch, (0, json.dumps({"leak": run.CANARY_VALUE}) + "\n", ""))
    item = item_named("env_probe")
    runner.run_item(item)
    assert not run.done_path(item).exists()
    assert [f.reason for f in runner.failures] == ["目印の値が出力に出ました"]
    assert run.CANARY_VALUE not in run.error_path(item).read_text(encoding="utf-8")  # 値は、エラーのファイルにも書かない


def test_run_probe_runs_run_py_with_the_minimal_environment(runner, monkeypatch):
    fake = install(monkeypatch, (0, json.dumps({"own_env_has_canary": False}) + "\n", ""))
    runner.run_item(item_named("env_probe"))
    ((argv, kwargs),) = fake.calls
    assert argv == [sys.executable, str(run.RUN_PY), "--probe-client"]
    assert kwargs["cwd"] == run.REPO and kwargs["timeout"] == 120 and kwargs["env"] == run.probe_env()
    assert "GEMINI_API_KEY" not in kwargs["env"]


# --- 段 C: プロセス内の比較 ---

IMPL_KEYS = {
    "condition", "question_id", "question", "trial", "model", "started_at", "duration_s", "status", "output", "usage",
    "tool_calls", "tool_returns", "retry_prompts", "called_make_quiz", "make_quiz_args", "finish_reasons", "exception",
    "http_error", "messages", "config", "git_head", "retries", "files_sha256",
}


def run_impl(monkeypatch, id_, model_function=call_quiz_99_until_answered):
    monkeypatch.setattr(run, "build_model", lambda: FunctionModel(model_function))
    runner = run.Runner(head=HEAD)
    item = item_named(id_)
    runner.run_item(item)
    assert runner.failures == [], runner.failures
    record = json.loads(run.done_path(item).read_text(encoding="utf-8"))
    return record, run.trace_path(item).read_text(encoding="utf-8")


def test_impl_run_with_the_return_server_ends_with_status_ok(python_servers, monkeypatch):
    record, trace = run_impl(monkeypatch, "return_q99_t1")
    assert set(record) == IMPL_KEYS
    assert (record["condition"], record["question_id"], record["question"], record["trial"]) == ("return", "q99", "第99回のクイズを作って", 1)
    assert record["status"] == "ok" and record["model"] == run.MODEL and record["output"] == "第99回はありません。"
    assert record["called_make_quiz"] is True and record["make_quiz_args"] == [99]
    assert record["tool_calls"] == [{"tool": "glossary_make_quiz", "args": {"lecture": 99}}]
    assert record["tool_returns"] == [{"tool": "glossary_make_quiz", "content": no_such_lecture(99)}]
    assert record["retry_prompts"] == [] and record["exception"] is None and record["http_error"] is None
    assert set(record["usage"]) == {"input_tokens", "output_tokens", "requests"} and record["usage"]["requests"] == 2
    assert record["finish_reasons"] == [None, None]  # FunctionModel は、終了の理由を返さない（取れなければ null）
    assert (record["git_head"], record["retries"]) == (HEAD, 0)
    assert record["files_sha256"] == {run.SERVER: run.sha256_of(run.REPO / run.SERVER), run.QUIZ_SERVER: run.sha256_of(run.REPO / run.QUIZ_SERVER)}
    command, args = python_server_command("return")
    assert record["config"] == {"mcpServers": {"glossary": {"command": command, "args": args}}}
    assert json.loads((run.RESULTS / "impl/configs/return.json").read_text(encoding="utf-8")) == record["config"]
    assert [m["kind"] for m in record["messages"]] == ["request", "response", "request", "response"]
    assert run.done_path(item_named("return_q99_t1")) == run.RESULTS / "impl/return/q99/t1.json"
    # trace
    assert trace.splitlines()[0] == "# 条件: return／質問: q99（第99回のクイズを作って）／試行: 1"
    assert f"[tool call]   glossary_make_quiz({{'lecture': 99}})" in trace
    assert f"[tool return] glossary_make_quiz -> {no_such_lecture(99)}" in trace
    assert "[answer] 第99回はありません。" in trace and "[stopped]" not in trace
    assert "[server] glossary: コマンドで起動（" in trace
    assert re.search(r"\[usage\] 入力 [\d,]+ tokens / 出力 [\d,]+ tokens / モデル応答 2 回\n$", trace)


@pytest.mark.parametrize(
    ("condition", "body"),
    [("valueerror", ERROR), ("toolerror", f"{ERROR}: {no_such_lecture(99)}")],
)
def test_impl_run_stops_after_two_failures_in_a_row_and_records_the_exception(python_servers, monkeypatch, condition, body):
    record, trace = run_impl(monkeypatch, f"{condition}_q99_t1")
    assert set(record) == IMPL_KEYS
    assert record["status"] == "stopped" and record["output"] is None
    exception = record["exception"]
    assert exception["type"] == "UnexpectedModelBehavior" and "exceeded max retries count of 1" in exception["message"]
    assert (exception["cause_type"], exception["cause_message"]) == ("ModelRetry", body)
    assert record["make_quiz_args"] == [99, 99] and record["tool_returns"] == []
    assert record["retry_prompts"] == [{"tool": "glossary_make_quiz", "content": body, "sent_to_model": body + FIX}]
    assert record["usage"]["requests"] == 2 and record["http_error"] is None
    assert f"[tool error]  glossary_make_quiz -> {body}" in trace
    assert f"[sent to model] {body}{FIX}" in trace
    assert "[stopped] UnexpectedModelBehavior: " in trace and f"[cause] ModelRetry: {body}" in trace
    assert trace.splitlines()[-1].startswith("[usage] 入力 ") and trace.rstrip().endswith("モデル応答 2 回")


def test_impl_run_with_the_schema_server_shows_the_validation_error(python_servers, monkeypatch):
    record, trace = run_impl(monkeypatch, "schema_q99_t1")
    assert record["status"] == "stopped"
    sent = record["retry_prompts"][0]["sent_to_model"]
    assert "validation error" in sent and "less than or equal to 13" in sent and "確認問題を出せる回" not in sent
    assert "[sent to model] Error executing tool make_quiz: 1 validation error" in trace


def test_impl_files_are_written_only_by_replacing_and_the_json_is_the_mark_of_done(python_servers, monkeypatch):
    run_impl(monkeypatch, "return_q99_t1")
    folder = run.RESULTS / "impl/return/q99"
    assert sorted(path.name for path in folder.iterdir()) == ["t1.json", "t1.trace.txt"]  # 一時ファイルが残らない
    # 済みの項目は、飛ばす（モデルを作らない）
    monkeypatch.setattr(run, "build_model", lambda: pytest.fail("済みの項目で、モデルを作った"))
    runner = run.Runner(head=HEAD)
    runner.run_item(item_named("return_q99_t1"))
    assert runner.failures == []


# --- 段 C: 失敗の扱い（attempt_impl をスタブにして） ---


class RaisingAgent:
    def __init__(self, error):
        self.error = error

    def run_sync(self, question, *, model):
        raise self.error


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ModelHTTPError(429, "m", "busy"), "http"),
        (UnexpectedModelBehavior("止まった"), "stopped"),
        (UsageLimitExceeded("多すぎる"), "agent_error"),
        (MCPError(-32000, "接続が切れた"), "connect"),
        (OSError("起動できない"), "connect"),
        (RuntimeError("Client failed to connect: boom"), "connect"),
    ],
)
def test_attempt_impl_sorts_the_failures_it_knows(monkeypatch, error, kind):
    monkeypatch.setattr(run, "build_mcp_agent", lambda path: RaisingAgent(error))
    monkeypatch.setattr(run, "build_model", lambda: object())
    attempt = run.attempt_impl(Path("unused.json"), "質問")
    assert attempt.kind == kind and attempt.error is error and attempt.messages == []


def test_attempt_impl_does_not_swallow_other_errors(monkeypatch):
    monkeypatch.setattr(run, "build_model", lambda: object())
    for error in (RuntimeError("別の RuntimeError"), KeyError("x")):
        monkeypatch.setattr(run, "build_mcp_agent", lambda path, error=error: RaisingAgent(error))
        with pytest.raises(type(error)):
            run.attempt_impl(Path("unused.json"), "質問")


def ok_attempt(output="答え"):
    result = types.SimpleNamespace(output=output, usage=types.SimpleNamespace(input_tokens=10, output_tokens=2, requests=2))
    return run.Attempt("ok", [], result=result)


def http_attempt(status, body="x"):
    return run.Attempt("http", [], error=ModelHTTPError(status, "m", body))


def script(monkeypatch, *attempts):
    queue = list(attempts)
    calls = []

    def fake(config_path, question):
        calls.append((config_path, question))
        return queue.pop(0)

    monkeypatch.setattr(run, "attempt_impl", fake)
    return calls


def test_server_side_errors_are_repeated_after_60_seconds_up_to_three_times(runner, monkeypatch, waits):
    calls = script(monkeypatch, http_attempt(503), http_attempt(500), http_attempt(502), ok_attempt())
    runner.run_item(item_named("return_q99_t1"))
    assert len(calls) == 4 and waits == [run.RETRY_WAIT] * 3
    record = json.loads(run.done_path(item_named("return_q99_t1")).read_text(encoding="utf-8"))
    assert (record["status"], record["retries"]) == ("ok", 3)


def test_server_side_errors_that_go_on_become_a_failure(runner, monkeypatch, waits):
    calls = script(monkeypatch, *[http_attempt(504, "busy")] * 4)
    item = item_named("return_q99_t1")
    runner.run_item(item)
    assert len(calls) == 4 and waits == [run.RETRY_WAIT] * 3
    assert not run.done_path(item).exists() and "504" in run.error_path(item).read_text(encoding="utf-8")
    assert [f.id for f in runner.failures] == ["return_q99_t1"]


def test_a_429_without_perday_waits_for_the_retry_delay(runner, monkeypatch, waits):
    body = {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}]}}
    script(monkeypatch, http_attempt(429, body), ok_attempt())
    runner.run_item(item_named("return_q99_t1"))
    assert waits == [17.0]
    assert runner.rate_limit_streak == 0  # 成功したので、戻した


def test_a_429_with_perday_stops_at_once_as_the_daily_limit(runner, monkeypatch, waits):
    body = {"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    calls = script(monkeypatch, http_attempt(429, body))
    item = item_named("return_q99_t1")
    with pytest.raises(run.DailyLimit, match="PerDay"):
        runner.run_item(item)
    assert len(calls) == 1 and waits == []
    assert "PerDay" in run.error_path(item).read_text(encoding="utf-8") and not run.done_path(item).exists()


def test_three_429s_in_a_row_also_stop_a_stage_c_run(runner, monkeypatch, waits):
    script(monkeypatch, *[http_attempt(429, "slow down")] * 3)
    with pytest.raises(run.DailyLimit, match="3 回続きました"):
        runner.run_item(item_named("return_q99_t1"))
    assert waits == [run.RETRY_WAIT] * 2  # retryDelay が読めないので、60 秒


def test_429s_are_counted_across_the_cli_and_stage_c(runner, monkeypatch):
    install(monkeypatch, (1, api_error_message(429) + "\n", ""), (1, api_error_message(429) + "\n", ""), (1, "別のエラー\n", ""))
    runner.run_item(item_named("step2_t1"))  # 429 が 2 回続いて、別のエラーで終わった
    assert runner.rate_limit_streak == 2
    script(monkeypatch, http_attempt(429, "slow down"))
    with pytest.raises(run.DailyLimit):
        runner.run_item(item_named("return_q99_t1"))  # これで 3 回続いた


def test_a_400_is_a_result_only_for_the_schema_condition(runner, monkeypatch):
    body = "x" * 2000
    script(monkeypatch, http_attempt(400, body))
    schema = item_named("schema_q99_t1")
    runner.run_item(schema)
    record = json.loads(run.done_path(schema).read_text(encoding="utf-8"))
    assert record["status"] == "http_error" and record["http_error"] == {"status": 400, "body": body[:1000]}
    assert record["exception"] is None and record["output"] is None
    assert "[stopped] ModelHTTPError" in run.trace_path(schema).read_text(encoding="utf-8")
    assert runner.failures == []
    # ほかの条件の 400 は、失敗の一覧に入る
    script(monkeypatch, http_attempt(400, body))
    runner.run_item(item_named("valueerror_q99_t1"))
    assert [f.id for f in runner.failures] == ["valueerror_q99_t1"]
    assert not run.done_path(item_named("valueerror_q99_t1")).exists()


def test_other_http_statuses_are_failures(runner, monkeypatch, waits):
    calls = script(monkeypatch, http_attempt(403, "forbidden"))
    runner.run_item(item_named("return_q99_t1"))
    assert len(calls) == 1 and waits == [] and [f.reason for f in runner.failures] == ["HTTP 403"]


def test_connection_failures_are_repeated_up_to_three_times_then_listed(runner, monkeypatch, waits):
    failing = run.Attempt("connect", [], error=MCPError(-32000, "接続が切れた"))
    calls = script(monkeypatch, *[failing] * 4)
    item = item_named("return_q99_t1")
    runner.run_item(item)
    assert len(calls) == 4 and waits == [run.RETRY_WAIT] * 3
    assert "MCP サーバーに接続できません" in run.error_path(item).read_text(encoding="utf-8")
    assert [f.id for f in runner.failures] == ["return_q99_t1"]


def test_agent_errors_and_stops_are_results_and_are_not_repeated(runner, monkeypatch):
    stopped = run.Attempt("stopped", [], error=UnexpectedModelBehavior("止まった"))
    agent_error = run.Attempt("agent_error", [], error=UsageLimitExceeded("多すぎる"))
    calls = script(monkeypatch, stopped, agent_error)
    runner.run_item(item_named("valueerror_q99_t1"))
    runner.run_item(item_named("toolerror_q99_t1"))
    assert len(calls) == 2 and runner.failures == []
    statuses = [json.loads(run.done_path(item_named(i)).read_text(encoding="utf-8")) for i in ("valueerror_q99_t1", "toolerror_q99_t1")]
    assert [r["status"] for r in statuses] == ["stopped", "agent_error"]
    assert statuses[1]["exception"]["type"] == "UsageLimitExceeded"


def test_an_unexpected_exception_is_listed_with_its_traceback_and_the_next_item_runs(runner, monkeypatch):
    def fake(config_path, question):
        raise KeyError("想定外")

    monkeypatch.setattr(run, "attempt_impl", fake)
    runner.run_item(item_named("return_q99_t1"))
    error = run.error_path(item_named("return_q99_t1")).read_text(encoding="utf-8")
    assert "KeyError" in error and "Traceback" in error
    assert runner.failures[0].reason.startswith("想定外のエラー: KeyError")
    script(monkeypatch, ok_attempt())
    runner.run_item(item_named("return_q5_t1"))
    assert run.done_path(item_named("return_q5_t1")).exists()


def test_stage_c_runs_are_spaced_by_five_seconds_except_the_first(runner, monkeypatch, waits):
    script(monkeypatch, ok_attempt(), ok_attempt(), ok_attempt())
    for id_ in ("return_q99_t1", "return_q99_t2", "return_q99_t3"):
        runner.run_item(item_named(id_))
    assert waits == [run.LLM_INTERVAL, run.LLM_INTERVAL]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('{"error": {"details": [{"retryDelay": "17s"}]}}', 17.0),
        ("{'error': {'details': [{'@type': 'RetryInfo', 'retryDelay': '52s'}]}}", 52.0),
        ('"retryDelay": "0.5s"', 0.5),
        ("retryDelay が書かれていない", 60.0),
        ("retryDelay: 500ms", 60.0),  # 読めなければ、60 秒
    ],
)
def test_retry_delay_reads_the_seconds_from_the_body(body, expected):
    assert run.retry_delay(body) == expected


def test_a_result_is_saved_even_when_the_messages_cannot_be_turned_into_json(runner, monkeypatch):
    class Broken:
        @staticmethod
        def dump_python(messages, mode):
            raise ValueError("JSON にできない値がある")

    monkeypatch.setattr(run, "ModelMessagesTypeAdapter", Broken)
    script(monkeypatch, ok_attempt())
    item = item_named("return_q99_t1")
    runner.run_item(item)
    record = json.loads(run.done_path(item).read_text(encoding="utf-8"))
    assert record["messages"] == {"error": "メッセージを JSON にできませんでした: ValueError: JSON にできない値がある"}
    assert record["status"] == "ok" and runner.failures == []


# --- --scan ---


def plant(path: str, data: bytes | str) -> None:
    target = run.RESULTS / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))


def test_scan_finds_planted_secrets_in_text_and_binary_files_without_printing_them(monkeypatch, capsys):
    aiza, ghp, pat = "AIza" + "x" * 35, "ghp_" + "y" * 36, "github_pat_" + "z" * 22  # 形に合う、偽の文字列
    value = "tok-0123456789-abcdef"
    monkeypatch.setenv("MY_SERVICE_TOKEN", value)
    monkeypatch.setenv("SHORT_TOKEN", "short")  # 8 文字未満は、調べない
    monkeypatch.setenv("PLAIN_SETTING", "visible-value-123")  # 名前に TOKEN・KEY・SECRET・PASSWORD がないので、調べない
    plant("cli/a.txt", f"前 {aiza} 後")
    plant("cli/b.txt", ghp)
    plant("impl/x/y.json", f'{{"k": "{pat}"}}')
    plant("direct/d.bin", b"\x00\xff\xfe" + value.encode() + b"\x00\x01")
    plant("cli/clean.txt", "きれいなファイル short visible-value-123 AIza-too-short ghp_short")
    assert run.main(["--scan"]) == 1
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "秘密情報らしき文字列が見つかりました: results/cli/a.txt（Google API キーの形）",
        "秘密情報らしき文字列が見つかりました: results/cli/b.txt（GitHub トークンの形）",
        "秘密情報らしき文字列が見つかりました: results/direct/d.bin（MY_SERVICE_TOKEN）",
        "秘密情報らしき文字列が見つかりました: results/impl/x/y.json（GitHub fine-grained トークンの形）",
    ]
    for secret in (aiza, ghp, pat, value):
        assert secret not in out  # 値は、表示しない


def test_scan_reports_the_name_of_the_environment_variable_that_leaked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-but-long")
    plant("cli/leak.txt", "ここに not-a-real-key-but-long が出た")
    code, lines = run.scan_report()
    assert code == 1 and lines == ["秘密情報らしき文字列が見つかりました: results/cli/leak.txt（GEMINI_API_KEY）"]


def test_scan_of_clean_results_returns_zero_and_counts_the_files(capsys):
    plant("cli/a.txt", "きれい")
    plant("impl/b/c.json", "{}")
    assert run.main(["--scan"]) == 0
    assert capsys.readouterr().out == "走査: 秘密情報らしき文字列は見つかりませんでした（2 ファイル）\n"


def test_scan_without_results_says_so_and_returns_zero(capsys):
    assert not run.RESULTS.exists()
    assert run.main(["--scan"]) == 0
    assert capsys.readouterr().out == "results がありません\n"


# --- --list ---


def test_list_prints_the_table_without_starting_anything(monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise AssertionError("--list が、起動してはいけないものを呼びました")

    forbidden = [
        (run.subprocess, "run"), (run.subprocess, "Popen"), (run, "execute"),  # コマンドの実行
        (run, "build_mcp_agent"), (run, "build_model"), (run, "StdioTransport"), (run, "Client"), (run, "Agent"), (run, "MCPToolset"),  # MCP・エージェント
        (run, "git"), (run, "git_text"), (run, "get_api_key"),  # git・キー
    ]
    for owner, name in forbidden:
        monkeypatch.setattr(owner, name, boom)
    monkeypatch.delenv("GEMINI_API_KEY")  # キーも、
    monkeypatch.setenv("PATH", "")  # uv も、要らない
    paths = [run.REPO / run.SERVER, run.REPO / run.CONFIG]
    before = [path.read_bytes() for path in paths]
    assert run.main(["--list"]) == 0
    out = capsys.readouterr().out
    items = run.build_items()
    assert f"全 {len(items)} 項目のうち、済みは 0 項目（結果は {run.RESULTS}/ に保存する）" in out
    for item in items:
        assert f"[未] {item.id} " in out
    assert [path.read_bytes() for path in paths] == before  # 状態を変えない
    assert not run.RESULTS.exists()  # results も作らない


def test_list_shows_which_items_are_done_and_what_runs(capsys):
    run.atomic_write(run.done_path(item_named("step1_tools")), "済み\n")
    run.atomic_write(run.done_path(item_named("direct_return")), "{}\n")
    assert run.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "済みは 2 項目" in out
    assert "[済] step1_tools " in out and "[済] direct_return " in out and "[未] step3_tools " in out
    assert "uv run python -m app.mcp_agent --list-tools" in out
    assert "--trace '第2回では何を学びますか？'" in out
    assert "stdio で uv run python lectures/06/quiz_server.py --variant schema につなぐ" in out
    assert "run_sync（第5回のクイズを作って）" in out
    assert out.count("段 A（API を使わない）") == out.count("段 B（LLM を使う CLI）") == out.count("段 C（LLM を使う、プロセス内の比較。S2）") == 1


def test_the_modes_cannot_be_mixed():
    with pytest.raises(SystemExit):
        run.parse_args(["--list", "--scan"])


def test_main_moves_to_the_repo_first(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert run.main(["--scan"]) == 0
    assert Path.cwd().resolve() == run.REPO.resolve()


# --- --summarize ---


AWS_TOOLS = """\
[server] glossary: コマンドで起動（uv run python -m app.mcp_server）
[server] aws: URL に接続（https://knowledge-mcp.global.api.aws）
モデルに見えているツール: 3 個

[glossary_search_glossary]
  講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。
  引数: {"query": {"title": "Query", "type": "string"}}

[aws_search_documentation]
  Search AWS documentation.
  引数: {"search_phrase": {"type": "string"}}

[aws_read_documentation]
  Read AWS documentation%s.
  引数: {"url": {"type": "string"}}
"""


def fake_cli(id_: str, stdout: str, *, outcome: str = "ok", returncode: int = 0, stderr: str = "") -> None:
    item = item_named(id_)
    text = run.cli_report(
        item, digests={run.SERVER: "0" * 64, run.CONFIG: "1" * 64}, head=HEAD, started=run.now(), seconds=1.5,
        returncode=returncode, retries=0, outcome=outcome, stdout=stdout, stderr=stderr,
    )
    run.atomic_write(run.done_path(item), text)


def fake_direct(condition: str) -> None:
    """直接の呼び出しの結果（direct/<条件>.json）の、要約に使う部分だけを、偽物で作る。"""
    body = {"return": None, "valueerror": ERROR, "toolerror": f"{ERROR}: {no_such_lecture(99)}", "schema": f"{ERROR}: 1 validation error"}[condition]
    agent = {"output": None, "tool_calls": [], "exception": None, "retry_prompts": [], "tool_returns": []}
    if body is None:  # return: 通常の結果として、文が返る
        agent["tool_returns"] = [{"tool": "glossary_make_quiz", "content": no_such_lecture(99)}]
    else:  # raise と schema: 失敗として、文（の末尾に 1 行付いたもの）がモデルに渡る
        agent["retry_prompts"] = [{"tool": "glossary_make_quiz", "content": body, "sent_to_model": body + FIX}]
    calls = {
        "99": {"isError": body is not None, "text": no_such_lecture(99) if body is None else body},
        "5": {"isError": False, "text": no_terms(5)},
        "2": {"isError": False, "text": '{"question": "q", "answer": "a"}'},
    }
    plant(f"direct/{condition}.json", json.dumps({"condition": condition, "calls": calls, "agent": agent}, ensure_ascii=False))


def fake_impl(condition: str, question_id: str, trial: int, *, status: str = "ok", args=(99,), errors: int = 0, output: str | None = "回答") -> None:
    record = {
        "status": status, "make_quiz_args": list(args), "retry_prompts": [{}] * errors, "output": output,
        "usage": {"input_tokens": 1234, "output_tokens": 56, "requests": 2},
    }
    plant(f"impl/{condition}/{question_id}/t{trial}.json", json.dumps(record, ensure_ascii=False))


def sections(text: str) -> dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return {part.split("\n", 1)[0]: part for part in parts}


EXPECTED_HEADINGS = [
    "1. 段 B: [tool call] の行（cli/*.txt）",
    "2. 段 A: 直接の呼び出し（条件 × lecture）",
    "3. 段 C: 条件 × 質問 × 試行",
    "4. aws_ ツールの定義（step4_tools・basic_tools・impl_tools の一致）",
    "5. env_probe（子プロセスが見られる環境変数）",
]


def test_summarize_builds_summary_md_with_all_headings_from_fake_results(capsys):
    fake_cli("step2_t1", "回答\n\n--- trace ---\n[tool call]   glossary_search_glossary({'query': '3レベル'})\n[usage] 入力 1 tokens / 出力 1 tokens / モデル応答 2 回\n")
    fake_cli("step2_t2", "回答\n\n--- trace ---\n[usage] 入力 1 tokens / 出力 1 tokens / モデル応答 1 回\n")  # ツールは呼ばれなかった
    fake_cli("impl_t1", "", outcome="stopped", returncode=1, stderr="UnexpectedModelBehavior\n")
    for id_ in ("step4_tools", "basic_tools", "impl_tools"):
        fake_cli(id_, AWS_TOOLS % "")  # 3 つとも同じ
    for condition in run.CONDITIONS:
        fake_direct(condition)
    fake_impl("return", "q99", 1)
    fake_impl("valueerror", "q99", 1, status="stopped", args=(99, 99), errors=1, output=None)
    fake_impl("return", "q5", 3, args=(5,), output="あ" * 100)
    plant("security/env_probe.json", json.dumps({
        "client_env_names": ["HOME", "PATH"], "canary_name": env_probe.CANARY_NAME,
        "report": {"platform": "linux", "own_env_names": ["HOME", "PATH"], "own_env_has_canary": False, "proc_available": True,
                   "parent": {"pid": 1, "environ_readable": True, "has_canary": True},
                   "ancestors": [{"level": 1, "pid": 2, "environ_openable": True}, {"level": 2, "pid": 3, "environ_openable": False}]},
    }))
    assert run.main(["--summarize"]) == 0
    assert capsys.readouterr().out == "results/summary.md を作りました。\n"
    text = (run.RESULTS / "summary.md").read_text(encoding="utf-8")
    assert text.startswith("# 第6回 実行結果のまとめ\n")
    assert list(sections(text)) == EXPECTED_HEADINGS  # 見出しが、順にそろっている
    assert [p.name for p in run.RESULTS.iterdir() if p.name.startswith(".")] == []  # 一時ファイルが残らない

    first = sections(text)[EXPECTED_HEADINGS[0]]
    assert re.findall(r"^### (\w+)$", first, re.M) == list(EXPECTED_STAGE_B)  # 段 B の 9 項目すべて
    assert "[tool call]   glossary_search_glossary({'query': '3レベル'})" in first
    assert "なし（[tool call] の行がありません）" in first  # step2_t2
    assert "なし（結果のファイルがありません）" in first  # step3_t1 など、未実行のもの
    assert "- 状態: S0（演習の開始時点の app/mcp_server.py）／設定: STEP 1（glossary だけ）" in first

    second = sections(text)[EXPECTED_HEADINGS[1]]
    assert "| valueerror | 99 | True | Error executing tool make_quiz |" in second
    assert len(no_such_lecture(99)) <= 80 < len(f"{ERROR}: {no_such_lecture(99)}")  # このテストの前提。短い文は、そのまま表に入る
    assert f"| return | 99 | False | {no_such_lecture(99)} |" in second
    long_body = f"{ERROR}: {no_such_lecture(99)}"
    assert f"| toolerror | 99 | True | {long_body[:80]} |" in second  # 長い文は、先頭 80 字だけ
    assert long_body[:81] not in second.split("### ")[0]
    assert ERROR + FIX in second and long_body + FIX in second  # 99 のときに、モデルに渡る文（全文）
    assert "| schema | 5 | False | " in second

    third = sections(text)[EXPECTED_HEADINGS[2]]
    assert "| return | q99 | 1 | [99] | 0 | ok | 回答 | 1234 | 56 | 2 |" in third
    assert "| valueerror | q99 | 1 | [99, 99] | 1 | stopped |  | 1234 | 56 | 2 |" in third
    assert f"| return | q5 | 3 | [5] | 0 | ok | {'あ' * 60} | 1234 | 56 | 2 |" in third  # 回答は、先頭 60 字
    assert "| toolerror | q99 | 1 | なし | なし | なし | なし | なし | なし | なし |" in third  # 未実行
    assert len([line for line in third.splitlines() if line.startswith("|")]) == 2 + 15  # ヘッダと区切りの行、と、15 回分

    fourth = sections(text)[EXPECTED_HEADINGS[3]]
    assert "一致していました（step4_tools, basic_tools, impl_tools）" in fourth
    assert "- step4_tools: 2 個（aws_search_documentation, aws_read_documentation）" in fourth

    fifth = sections(text)[EXPECTED_HEADINGS[4]]
    assert f"- 目印の名前: {env_probe.CANARY_NAME}" in fifth and "- 子の環境に目印があったか: False" in fifth
    assert "- 親の /proc/PID/environ: environ を読めた: True／目印があった: True" in fifth
    assert "- 先祖のプロセス: 2 段（environ を開けた: 1 段）" in fifth


def test_summarize_shows_the_difference_when_the_aws_tools_do_not_match():
    fake_cli("step4_tools", AWS_TOOLS % "")
    fake_cli("basic_tools", AWS_TOOLS % "")
    fake_cli("impl_tools", AWS_TOOLS % "（変わった）")
    text = run.summary_text()
    fourth = sections(text)[EXPECTED_HEADINGS[3]]
    assert "一致しませんでした。差分（最初の結果との違い）:" in fourth and "```diff" in fourth
    assert "- impl_tools:\n\n```diff\n--- step4_tools\n+++ impl_tools" in fourth
    assert "- basic_tools:\n\n```diff" not in fourth  # 最初の結果（step4_tools）と同じものは、差分を出さない
    assert "-  Read AWS documentation." in fourth and "+  Read AWS documentation（変わった）." in fourth


def test_summarize_writes_the_word_none_for_what_has_no_result():
    run.RESULTS.mkdir(parents=True)
    assert run.main(["--summarize"]) == 0
    text = (run.RESULTS / "summary.md").read_text(encoding="utf-8")
    assert list(sections(text)) == EXPECTED_HEADINGS  # 結果がなくても、見出しはそろう
    assert "なし" in text and "| なし |" in text
    assert "比べられません" in sections(text)[EXPECTED_HEADINGS[3]]
    assert "なし（結果のファイルがありません）" in sections(text)[EXPECTED_HEADINGS[4]]


def test_summarize_without_results_asks_to_pull_first(capsys):
    assert not run.RESULTS.exists()
    assert run.main(["--summarize"]) == 1
    assert "results がありません" in capsys.readouterr().out and not run.RESULTS.exists()


def test_extract_aws_tools_takes_only_the_aws_sections_and_stops_before_stderr():
    text = AWS_TOOLS % "" + "\n--- stderr ---\n[aws_not_a_tool]\n警告\n"
    tools = run.extract_aws_tools(text)
    assert list(tools) == ["aws_search_documentation", "aws_read_documentation"]
    assert tools["aws_search_documentation"] == '[aws_search_documentation]\n  Search AWS documentation.\n  引数: {"search_phrase": {"type": "string"}}'
    assert "glossary" not in "".join(tools.values())


FAKE_AWS_SERVER = '''\
from mcp.server.mcpserver import MCPServer

server = MCPServer("fake-aws")


@server.tool()
def search_documentation(search_phrase: str) -> str:
    """AWS のドキュメントを検索する。

    search_phrase に、検索する語を入れる。
    """
    return "x"


@server.tool(name="read.docs-v2")  # MCP のツール名には、. と - も使える
def read_docs(url: str) -> str:
    """ドキュメントを読む。"""
    return "y"


if __name__ == "__main__":
    server.run()
'''


def test_extract_aws_tools_reads_what_print_tools_really_prints(python_servers, tmp_path):
    """app/mcp_agent.py の print_tools が出す形（見出し・説明・引数の行）から、aws_ のツールを、1 つずつ取り出せる。

    標準出力だけを捕まえる（capsys は標準エラー出力も差し替え、サーバーの標準エラー出力の受け渡しに必要な fileno を無くす）。
    """
    server = tmp_path / "fake_aws.py"
    server.write_text(FAKE_AWS_SERVER, encoding="utf-8")  # ネットワークに出ない、手元の偽のサーバー
    command, args = python_server_command("return")
    config = tmp_path / "servers.json"
    config.write_text(
        json.dumps({"mcpServers": {"glossary": {"command": command, "args": args}, "aws": {"command": sys.executable, "args": [str(server)]}}}),
        encoding="utf-8",
    )
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_tools(run.build_mcp_agent(config))
    tools = run.extract_aws_tools(printed.getvalue())
    assert list(tools) == ["aws_search_documentation", "aws_read.docs-v2"]  # glossary のツールは入らず、. や - を含む名前も、別のツールになる
    assert tools["aws_search_documentation"] == (
        "[aws_search_documentation]\n  AWS のドキュメントを検索する。\n  \n  search_phrase に、検索する語を入れる。\n"
        '  引数: {"search_phrase": {"title": "Search Phrase", "type": "string"}}'
    )
    assert tools["aws_read.docs-v2"] == '[aws_read.docs-v2]\n  ドキュメントを読む。\n  引数: {"url": {"title": "Url", "type": "string"}}'


def test_summary_picks_up_the_tool_call_lines_that_print_trace_really_prints(python_servers):
    agent = run.build_mcp_agent(run.write_condition_config("return"))
    result = agent.run_sync("第99回のクイズを作って", model=FunctionModel(call_quiz_99_until_answered))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print(result.output)
        print_trace(result)  # app/mcp_agent.py の --trace の表示
    fake_cli("impl_t1", printed.getvalue())
    first = sections(run.summary_text())[EXPECTED_HEADINGS[0]]
    assert "[tool call]   glossary_make_quiz({'lecture': 99})" in first
