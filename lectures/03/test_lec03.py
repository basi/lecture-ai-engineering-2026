"""第3回の課題のテスト（LLM も API キーも使わない）。

    uv run pytest -q lectures/03

演習の make_quiz と、宿題（実装）の save_learning_log・承認の流れを確かめる。
モデルは TestModel（指定したツールを呼ぶだけの偽モデル）に差し替える。
"""

import builtins
import json
import random
import re
import sys
from datetime import datetime

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import tools
from app.agent import agent, ask_approval, run_with_approval


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """保存先を一時フォルダに差し替える（本物の learning_log/ を汚さない）。まだ作られていない。"""
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", path)
    return path


@pytest.fixture
def fixed_now(monkeypatch):
    """日時を 2026-11-03 12:00:00 に固定する。同じ題名で続けて保存すると、ファイル名がぶつかる。"""

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 3, 12, 0, 0)

    monkeypatch.setattr(tools, "datetime", FixedDatetime)


def _terms_of(lecture):
    return {t["term"]: t["definition"] for t in tools.load_glossary() if t["lecture"] == lecture}


def _save_model():
    """save_learning_log だけを呼ぶ偽モデル（引数はダミー）。"""
    return TestModel(call_tools=["save_learning_log"])


def _tool_returns(result):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]


# --- make_quiz（演習） ---


def test_make_quiz_asks_about_a_term_of_that_lecture():
    terms = _terms_of(2)
    assert len(terms) > 1
    asked = set()
    for seed in range(30):  # 出題は乱数で決まるので、何度か試して、毎回第2回の用語か確かめる
        random.seed(seed)
        quiz = json.loads(tools.make_quiz(2))
        assert set(quiz) == {"question", "answer"}
        term = quiz["question"].removeprefix("「").removesuffix("」の意味は？")
        assert term in terms, quiz["question"]
        assert quiz["answer"] == terms[term]
        asked.add(term)
    assert len(asked) > 1  # 毎回同じ用語ではなく、乱数で選ばれている


def test_make_quiz_gives_the_same_term_with_the_same_seed():
    random.seed(0)
    first = tools.make_quiz(2)
    random.seed(0)
    assert tools.make_quiz(2) == first  # 比較用スクリプトは、seed で出題をそろえる


def test_make_quiz_without_terms_lists_the_lectures_it_can_use():
    available = sorted({t["lecture"] for t in tools.load_glossary()})
    message = tools.make_quiz(0)
    assert "第0回の用語は用語集にありません" in message
    assert f"出題できる回: {', '.join(map(str, available))}" in message
    for lecture in available:  # 一覧に出た回は、実際に出題できる
        assert json.loads(tools.make_quiz(lecture))["question"]


# --- save_learning_log（宿題・実装） ---


def test_save_learning_log_writes_a_markdown_file(log_dir):
    message = tools.save_learning_log("LoRAのメモ", "低ランクの行列だけを学習する。")

    (saved,) = log_dir.iterdir()  # フォルダは、無ければ作られる
    assert re.fullmatch(r"\d{8}-\d{6}-LoRAのメモ\.md", saved.name)
    assert saved.read_text(encoding="utf-8") == "# LoRAのメモ\n\n低ランクの行列だけを学習する。\n"
    assert saved.name in message  # 返り値に、保存したファイル名が入っている


def test_save_learning_log_never_writes_outside_the_log_dir(tmp_path, log_dir):
    readme = tmp_path / "README.md"  # 題名の ../README.md が指す先
    readme.write_text("元の内容", encoding="utf-8")
    titles = [
        "../README.md",
        "../../README.md",
        "..\\README.md",
        "a/../../README",
        str(tmp_path / "outside"),  # 絶対パス
        "..",
        "\x00\x1b[31m",  # 制御文字
    ]
    for title in titles:
        tools.save_learning_log(title, "本文")

    saved = [p for p in tmp_path.rglob("*") if p.is_file() and p != readme]
    assert len(saved) == len(titles)
    for path in saved:
        assert path.parent == log_dir
        assert path.name.count(".") == 1 and path.name.endswith(".md")  # ".." は残らない
    assert readme.read_text(encoding="utf-8") == "元の内容"


def test_save_learning_log_does_not_overwrite_existing_files(log_dir, fixed_now):
    for i in (1, 2, 3):  # 日時が同じなので、3回とも同じ名前になるところ
        tools.save_learning_log("メモ", f"{i}回目")

    texts = {p.name: p.read_text(encoding="utf-8") for p in log_dir.iterdir()}
    assert set(texts) == {"20261103-120000-メモ.md", "20261103-120000-メモ-2.md", "20261103-120000-メモ-3.md"}
    assert "1回目" in texts["20261103-120000-メモ.md"]  # 最初のファイルは書き換わらない
    assert "2回目" in texts["20261103-120000-メモ-2.md"]
    assert "3回目" in texts["20261103-120000-メモ-3.md"]


def test_save_learning_log_handles_empty_and_long_titles(log_dir):
    tools.save_learning_log("", "本文")
    tools.save_learning_log("!!!", "本文")  # 使える文字が1つも無い
    tools.save_learning_log("あ" * 300, "本文")

    names = [p.name for p in log_dir.iterdir()]
    assert sum(re.fullmatch(r"\d{8}-\d{6}-log(-2)?\.md", n) is not None for n in names) == 2  # 空なら "log"
    assert max(len(n) for n in names) < 80  # 長すぎる題名は切られる


def test_save_learning_log_refuses_a_link_that_points_outside(tmp_path, log_dir, fixed_now):
    log_dir.mkdir()
    outside = tmp_path / "outside.txt"
    (log_dir / "20261103-120000-メモ.md").symlink_to(outside)  # 保存するはずの名前に、外を指すリンクを置く

    message = tools.save_learning_log("メモ", "本文")

    assert not outside.exists()
    assert "保存しませんでした" in message


# --- 承認の流れ（宿題・実装） ---


def test_save_learning_log_waits_for_approval(log_dir):
    result = agent.run_sync("LoRAを学習ログに保存して", model=_save_model())

    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not list(log_dir.glob("*"))  # 承認する前は、何も保存されていない


def test_approved_call_saves_the_log(log_dir):
    asked = []

    def approve(call):
        asked.append(call)
        return True

    result = run_with_approval(agent, "LoRAを学習ログに保存して", decide=approve, model=_save_model())

    assert [c.tool_name for c in asked] == ["save_learning_log"]  # 承認を聞かれたのは1回
    assert isinstance(result.output, str)  # 続きから実行されて、文章の答えが返る
    assert len(list(log_dir.glob("*.md"))) == 1
    (returned,) = _tool_returns(result)
    assert returned.outcome == "success"


def test_denied_call_saves_nothing_and_tells_the_model(log_dir):
    result = run_with_approval(agent, "LoRAを学習ログに保存して", decide=lambda call: False, model=_save_model())

    assert isinstance(result.output, str)
    assert not list(log_dir.glob("*"))  # ファイルはできない
    (returned,) = _tool_returns(result)
    assert returned.outcome == "denied"  # 拒否されたことが、ツールの戻り値に記録される
    assert "断りました" in returned.content


def test_usage_is_the_total_of_both_runs(log_dir):
    question = "LoRAを学習ログに保存して"
    once = agent.run_sync(question, model=_save_model())  # 承認待ちで止まる1回目だけ
    total = run_with_approval(agent, question, decide=lambda call: True, model=_save_model())

    assert once.usage.requests == 1
    assert total.usage.requests == 2  # 1回目（保存を呼ぶ）+ 2回目（承認のあとに答える）
    assert total.usage.input_tokens > once.usage.input_tokens


def test_second_run_continues_the_first_conversation(log_dir):
    result = run_with_approval(agent, "LoRAを学習ログに保存して", decide=lambda call: True, model=_save_model())

    calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
    assert [c.tool_name for c in calls] == ["save_learning_log"]  # 1回目の呼び出しも、会話に残っている


def test_no_approval_is_asked_when_no_tool_needs_it(log_dir):
    def never(call):
        raise AssertionError("承認が必要なツールは呼ばれていないので、聞かれないはず")

    result = run_with_approval(agent, "LoRAとは？", decide=never, model=TestModel(call_tools=["search_glossary"]))

    assert isinstance(result.output, str)


def test_run_with_approval_stops_after_max_rounds(log_dir):
    def always_save(messages, info):  # 断られても、また保存を求め続けるモデル
        return ModelResponse(parts=[ToolCallPart("save_learning_log", {"title": "t", "content": "c"})])

    asked = []
    result = run_with_approval(
        agent,
        "保存して",
        decide=lambda call: asked.append(call) or False,  # 聞かれた呼び出しを記録して、拒否する
        max_rounds=2,
        model=FunctionModel(always_save),
    )

    assert len(asked) == 2  # 2回聞いて、あきらめる
    assert isinstance(result.output, DeferredToolRequests)
    assert not list(log_dir.glob("*"))


# --- ask_approval（CLI で利用者に聞く） ---


@pytest.mark.parametrize(
    ("typed", "expected"),
    [("y", True), ("yes", True), (" YES ", True), ("n", False), ("", False), ("はい", False), ("ok", False)],
)
def test_ask_approval_accepts_only_y_and_yes(monkeypatch, capsys, typed, expected):
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": prompts.append(prompt) or typed)

    call = ToolCallPart("save_learning_log", {"title": "LoRAのメモ", "content": "本文"})
    assert ask_approval(call) is expected

    assert "[y/N]" in prompts[0]
    shown = capsys.readouterr().out
    assert "save_learning_log" in shown and "LoRAのメモ" in shown  # ツール名と引数を見せてから聞く


def test_ask_approval_denies_when_there_is_no_input(monkeypatch):
    def end_of_input(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", end_of_input)

    assert ask_approval(ToolCallPart("save_learning_log", {"title": "t", "content": "c"})) is False


def test_cli_asks_for_approval_and_shows_the_total_usage(monkeypatch, capsys, log_dir):
    monkeypatch.setattr(agent_module, "build_model", _save_model)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
    monkeypatch.setattr(sys, "argv", ["app.agent", "--trace", "LoRAを学習ログに保存して"])

    agent_module.main()

    shown = capsys.readouterr().out
    assert "[承認待ち] save_learning_log" in shown
    assert "[tool call]   save_learning_log" in shown
    assert "requests=2" in shown  # [usage] は2回の合計
    assert len(list(log_dir.glob("*.md"))) == 1
