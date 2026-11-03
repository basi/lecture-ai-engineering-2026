"""比較用スクリプト compare.py のテスト（LLM も API キーも使わない）。

    uv run pytest -q lectures/03

変種の当て方、予行（--dry-run）と再開、秘密の置き換え、再試行（429・503・400）、
adk と pi の出力の解析、未コミットの変更の確認、チェックと表の形を確かめる。
モデルは FunctionModel / TestModel、外部コマンドは関数の差し替えで偽物にする。
"""

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import compare

MODEL = compare.default_model()


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """time.sleep は実際には待たない（待つ秒数だけ記録する）。API キーは偽物にする。"""
    waited = []
    monkeypatch.setattr(compare.time, "sleep", waited.append)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return waited


def run(*argv):
    return compare.main([str(a) for a in argv])


def real_run(out, *extra, level="basic", variant="baseline", question="q01", extra_args=()):
    """API を呼ぶ側の経路（--dry-run なし）で 1 条件を 1 回だけ実行する。モデルは各テストで差し替える。"""
    return run("run", "--level", level, "--variant", variant, "--question", question, "--trials", 1,
               "--allow-dirty", "--interval", 0, "--out", out, *extra, *extra_args)


def result_path(out, engine="pydantic", variant="baseline", qid="q01", trial=1, model=MODEL, suffix=".json"):
    return Path(out) / model / engine / variant / qid / f"t{trial}{suffix}"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def use_model(monkeypatch, model):
    monkeypatch.setattr(compare, "make_pydantic_model", lambda cond, ctx: model)


def text_model(text="LoRAは低ランク行列だけを学習する手法です。\n第2回"):
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


def failing_then(errors, final="答えです。\n第1回"):
    """errors を 1 つずつ出し、そのあとは final を返すモデル。呼ばれた回数は calls に数える。"""
    calls = []

    def fn(messages, info):
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return ModelResponse(parts=[TextPart(final)])

    return FunctionModel(fn), calls


def http_error(status, body=None, headers=None):
    return ModelHTTPError(status, "gemini-test", body or {"error": {"code": status}}, headers=headers)


RATE_PER_MINUTE = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"},
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
]}}
RATE_PER_DAY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
]}}


# --- 変種の当て方 ---


def test_every_variant_changes_exactly_one_place():
    cfg = compare.load_config()
    diffs = compare.variant_diffs(cfg)

    assert diffs["baseline"]["diff"] == []
    for name, info in diffs.items():
        if name != "baseline":
            assert info["hunks"] == 1, f"{name}: 差分が {info['hunks']} か所"
    assert len({info["effective"].sha256 for info in diffs.values()}) == len(diffs)  # 変種どうしで中身が違う
    assert "save_learning_log" in "\n".join(diffs["impl"]["diff"])  # 実装の変種だけ、ツールが 1 つ増える


def test_variants_change_the_copy_not_the_app():
    from app.agent import INSTRUCTIONS

    cfg = compare.load_config()
    quiet = compare.build_effective(cfg, "no_search_rule")
    many = compare.build_effective(cfg, "query_many")

    assert "必ず search_glossary で用語集を確認する" not in quiet.instructions
    assert "まとめて入れる" in {t.name: t for t in many.tools}["search_glossary"].description
    original = compare.registered_tools()["search_glossary"].description
    assert "1つだけ入れる" in original  # app/ の Tool は書き換わっていない
    assert "必ず search_glossary で用語集を確認する" in INSTRUCTIONS
    assert [t.name for t in compare.build_effective(cfg, "baseline").tools] == ["search_glossary", "get_lecture_summary", "make_quiz"]
    assert [t.requires_approval for t in compare.build_effective(cfg, "impl").tools] == [False, False, False, True]


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"op": "delete_line", "old": "存在しない行"}, "a\nb\n"),  # 0 か所
        ({"op": "delete_line", "old": "a"}, "a\nb\na\n"),  # 2 か所
        ({"op": "delete_line", "old": "a"}, "ab\nb\n"),  # 行の一部は、一致とみなさない
        ({"op": "replace_line", "old": "x", "new": "y"}, "a\nb\n"),
        ({"op": "replace_text", "old": "zz", "new": "y"}, "a\nb\n"),
        ({"op": "replace_text", "old": "a", "new": "y"}, "aa\nb\n"),  # 文字列が 2 か所
    ],
)
def test_a_change_must_match_exactly_one_place(change, text):
    with pytest.raises(compare.ConfigError, match="ちょうど 1 か所"):
        compare.apply_change(text, change, "テスト")


def test_apply_change_operations():
    text = "a\nb\nc\n"
    assert compare.apply_change(text, {"op": "delete_line", "old": "b"}, "x") == "a\nc\n"
    assert compare.apply_change(text, {"op": "replace_line", "old": "b", "new": "B"}, "x") == "a\nB\nc\n"
    assert compare.apply_change(text, {"op": "replace_text", "old": "b\nc", "new": ""}, "x") == "a\n\n"


def test_a_wrong_change_stops_before_any_api_call(tmp_path, monkeypatch, capsys):
    def never(cond, ctx):
        raise AssertionError("API を呼ぶ前に止まるはず")

    monkeypatch.setattr(compare, "make_pydantic_model", never)
    broken = compare.CONDITIONS_FILE.read_text(encoding="utf-8").replace(
        "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。", "- 存在しない行")
    conditions = tmp_path / "conditions.toml"
    conditions.write_text(broken, encoding="utf-8")

    assert run("run", "--level", "basic", "--conditions", conditions, "--allow-dirty", "--out", tmp_path / "out") == 1

    assert "ちょうど 1 か所" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_variants_command_saves_files_and_warns_when_a_variant_changes_two_places(tmp_path, capsys):
    out = tmp_path / "out"
    assert run("variants", "--out", out) == 0
    assert (out / "variants" / "pydantic" / "no_search_rule.txt").exists()
    saved = load(out / "variants" / "pydantic" / "query_many.json")
    assert saved["sha256"] and saved["instructions"] and [t["name"] for t in saved["tools"]]
    capsys.readouterr()

    two_places = compare.CONDITIONS_FILE.read_text(encoding="utf-8").replace(
        'changes = [\n  { target = "instructions", op = "replace_line", old = "- 回答の最後に、関係する回（第N回）を示す。", new = "- 回答の最後に、確認問題を1問出す。" },',
        'changes = [\n  { target = "instructions", op = "replace_line", old = "- 回答の最後に、関係する回（第N回）を示す。", new = "- 回答の最後に、確認問題を1問出す。" },\n'
        '  { target = "tool:make_quiz", op = "replace_text", old = "確認問題と答えを返す。", new = "確認問題を返す。" },')
    assert two_places != compare.CONDITIONS_FILE.read_text(encoding="utf-8")
    conditions = tmp_path / "conditions.toml"
    conditions.write_text(two_places, encoding="utf-8")

    assert run("variants", "--conditions", conditions, "--out", tmp_path / "out2") == 1
    assert "差分が 2 か所" in capsys.readouterr().out


# --- 実行する条件の組み立て ---


def test_build_plan_levels_and_dedup():
    cfg = compare.load_config()
    plan = compare.build_plan(cfg)
    keys = [c.key for c in plan]

    assert len(keys) == len(set(keys)) == 46  # 演習と基礎で共有する baseline は 1 回だけ
    counts = {e: sum(1 for c in plan if c.engine == e) for e in compare.ENGINES}
    assert counts == {"ask": 3, "pydantic": 36, "adk": 4, "pi": 3}
    assert [c.qid for c in plan if c.engine == "ask"] == ["q01", "q02", "q04"]
    assert [c.qid for c in plan if c.level == "exercise" and c.engine == "pydantic"] == ["q01", "q02", "q03"]
    assert [c.variant for c in plan if c.level == "basic"][0] == "baseline"
    order = list(dict.fromkeys(c.variant for c in plan if c.level == "basic"))
    assert order == ["baseline", "no_search_rule", "query_many", "ending_quiz", "no_notfound_rule", "doc_no_when"]
    assert {c.variant for c in plan}.isdisjoint({"no_hint_rule", "quiz_doc_no_when"})  # 任意の変種は、既定では含めない

    with_optional = compare.build_plan(cfg, include_optional=True)
    assert [c.key for c in with_optional if c.optional] == [
        ("pydantic", "no_hint_rule", "q09"), ("pydantic", "quiz_doc_no_when", "q03"), ("pydantic", "quiz_doc_no_when", "q10")]
    assert [c.approvals for c in plan if c.variant == "impl"] == [
        ("approve",), ("deny",), ("deny",), ("deny",), ("approve", "deny"), ("approve",)]
    only = compare.build_plan(cfg, levels=["adv-a"], questions=["q02", "q03"])
    assert [c.key for c in only] == [("adk", "baseline", "q02"), ("adk", "baseline", "q03")]


# --- 予行（--dry-run）、再開、集計 ---


def test_dry_run_all_levels_resume_and_summarize(tmp_path, capsys):
    out = tmp_path / "out"
    log_dir = compare.REPO / "learning_log"
    before = sorted(log_dir.rglob("*")) if log_dir.exists() else None
    assert run("run", "--level", "all", "--dry-run", "--trials", 1, "--out", out) == 0

    model_dir = out / f"{MODEL}-dry-run"  # 予行は別のフォルダに書く（本物の結果と混ざらない）
    plan = compare.build_plan(compare.load_config())
    found = {tuple(p.relative_to(model_dir).parts[:3]) for p in model_dir.glob("*/*/*/t1.json")}
    assert found == {c.key for c in plan}
    assert not (out / MODEL).exists()
    for cond in plan:
        base = model_dir / cond.engine / cond.variant / cond.qid
        assert (base / "t1.trace.txt").exists()
        assert (base / "t1.raw.jsonl").exists() == (cond.engine in ("adk", "pi"))
        assert (base / "t1.stderr.txt").exists() == (cond.engine in ("adk", "pi"))
        assert not list(base.glob("*.tmp"))
    # 承認でファイルができ、拒否で denied になる
    approved = load(model_dir / "pydantic/impl/s1/t1.json")
    assert [a["decision"] for a in approved["approvals"]] == ["approve"]
    assert len(approved["saved_files"]) == 1 and approved["tool_calls"][0]["outcome"] == "success"
    denied = load(model_dir / "pydantic/impl/s2/t1.json")
    assert [a["decision"] for a in denied["approvals"]] == ["deny"]
    assert denied["saved_files"] == [] and denied["tool_calls"][0]["outcome"] == "denied"
    # 記録する項目
    record = load(model_dir / "pydantic/baseline/q02/t1.json")
    for key in ("variant_sha256", "seed", "git", "started_at", "duration_s", "status", "attempts", "answer", "instructions",
                "tools_sha256", "tool_calls", "usage", "messages"):
        assert key in record, key
    assert record["seed"] == "q02:t1" and record["usage"]["requests"] == 2 and "commit" in record["git"]
    assert record["instructions"].startswith("あなたはAIエンジニアリング講座")
    assert "[model]" in (model_dir / "pydantic/baseline/q02/t1.trace.txt").read_text(encoding="utf-8")
    assert (log_dir.exists() and sorted(log_dir.rglob("*")) == before) or (not log_dir.exists() and before is None)  # learning_log/ は汚さない

    # 2 回目の実行では、何も増えず、何も書き直されない
    snapshot = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    assert run("run", "--level", "all", "--dry-run", "--trials", 1, "--out", out) == 0
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == snapshot

    # 試行を増やすと、足りない分だけ増える
    assert run("run", "--level", "all", "--dry-run", "--trials", 2, "--out", out) == 0
    assert len(list(model_dir.glob("*/*/*/t2.json"))) == len(plan)

    capsys.readouterr()
    assert run("summarize", "--out", out, "--trials", 2) == 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    for needle in ("表A", "表B", "表C", "V0 baseline", "V5 doc_no_when", "実装 impl", "ask/baseline", "adk/baseline", "pi/baseline",
                   "(参考) V0 baseline", "<details>", f"## モデル: {MODEL}-dry-run"):
        assert needle in summary, needle
    assert "未実行" not in summary
    assert load(out / "summary.json")["models"][f"{MODEL}-dry-run"]["basic"]["questions"][0] == "q01"


def test_dry_run_without_out_writes_to_a_temp_folder(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "auto"
    folder.mkdir()
    monkeypatch.setattr(compare.tempfile, "mkdtemp", lambda prefix="": str(folder))

    assert run("run", "--level", "exercise", "--engine", "ask", "--dry-run", "--trials", 1) == 0

    assert f"保存先: {folder}" in capsys.readouterr().out
    assert len(list(folder.glob("*/ask/baseline/q*/t1.json"))) == 3


def test_max_runs_stops_and_the_next_run_continues(tmp_path):
    out = tmp_path / "out"
    args = ["run", "--level", "exercise", "--dry-run", "--trials", 1, "--out", out]
    assert run(*args, "--max-runs", 2) == 0
    assert len(list(out.glob("*/*/*/*/t1.json"))) == 2
    assert run(*args) == 0
    assert len(list(out.glob("*/*/*/*/t1.json"))) == 6


def test_a_changed_variant_stops_the_run(tmp_path, capsys):
    out = tmp_path / "out"
    args = ["run", "--dry-run", "--engine", "pydantic", "--variant", "baseline", "--question", "q01", "--out", out]
    assert run(*args, "--trials", 1) == 0
    path = result_path(out, model=f"{MODEL}-dry-run")
    record = load(path)
    record["variant_sha256"] = "0" * 64  # 途中でコードが変わった、という状況
    path.write_text(json.dumps(record), encoding="utf-8")

    assert run(*args, "--trials", 2) == 1

    assert "sha256" in capsys.readouterr().err
    assert not result_path(out, trial=2, model=f"{MODEL}-dry-run").exists()


@pytest.mark.parametrize(("option", "value"), [("--question", "q99"), ("--variant", "baselin"), ("--engine", "gpt")])
def test_a_misspelled_filter_is_reported_with_the_valid_names(tmp_path, capsys, option, value):
    assert run("run", "--dry-run", option, value, "--out", tmp_path) == 1
    err = capsys.readouterr().err
    assert value in err and "使えるもの" in err
    assert not list(tmp_path.iterdir())


def test_summarize_without_results_says_so(tmp_path, capsys):
    assert run("summarize", "--out", tmp_path / "nothing") == 1
    assert "集計できる結果がありません" in capsys.readouterr().out


# --- 秘密の置き換え ---


def test_mask_replaces_api_keys_everywhere(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-value")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google-value")
    shaped = "AIza" + "x" * 35
    data = {"a": "key=secret-gemini-value", "list": ["secret-google-value", {"deep": f"x {shaped} y"}], "n": 3, "secret-gemini-value": 1}

    masked = compare.mask(data)

    text = json.dumps(masked, ensure_ascii=False)
    for secret in ("secret-gemini-value", "secret-google-value", shaped):
        assert secret not in text
    assert masked["a"] == "key=***" and masked["n"] == 3


def test_saved_files_never_contain_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-value")
    shaped = "AIza" + "y" * 35
    output = compare.TrialOutput({"answer": "secret-gemini-value", "x": [shaped]}, "trace secret-gemini-value", raw=shaped, stderr="secret-gemini-value")

    compare.save_trial(tmp_path / "t1", output)

    files = sorted(tmp_path.iterdir())
    assert [p.name for p in files] == ["t1.json", "t1.raw.jsonl", "t1.stderr.txt", "t1.trace.txt"]  # .tmp が残らない
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "secret-gemini-value" not in text and shaped not in text
        assert "***" in text


# --- 再試行（pydantic エンジン） ---


def test_rate_limit_is_retried_after_the_advised_delay(tmp_path, monkeypatch, sleeps):
    model, calls = failing_then([http_error(429, RATE_PER_MINUTE)])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 0

    assert sleeps == [7.0] and len(calls) == 2
    record = load(result_path(tmp_path))
    assert record["status"] == "ok" and record["answer"].startswith("答えです")
    assert [a.get("action", "ok") for a in record["attempts"]] == ["retry", "ok"]
    assert record["attempts"][0]["status"] == 429 and record["attempts"][0]["wait_s"] == 7.0


def test_rate_limit_uses_the_retry_after_header_then_a_default_of_60_seconds(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, failing_then([http_error(429, headers={"Retry-After": "12"})])[0])
    assert real_run(tmp_path / "a") == 0
    use_model(monkeypatch, failing_then([http_error(429)])[0])
    assert real_run(tmp_path / "b") == 0
    assert sleeps == [12.0, 60.0]


def test_rate_limit_gives_up_after_five_retries(tmp_path, monkeypatch, sleeps, capsys):
    use_model(monkeypatch, failing_then([http_error(429)] * 10)[0])

    assert real_run(tmp_path) == 1

    assert sleeps == [60.0] * 5
    assert not result_path(tmp_path).exists()
    assert "続きから" in capsys.readouterr().err


def test_a_daily_limit_stops_with_exit_code_2_and_saves_nothing(tmp_path, monkeypatch, sleeps, capsys):
    model, calls = failing_then([http_error(429, RATE_PER_DAY)])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 2

    assert sleeps == [] and len(calls) == 1
    assert not result_path(tmp_path).exists()
    assert "明日" in capsys.readouterr().err


def test_503_is_retried_after_30_60_120_seconds(tmp_path, monkeypatch, sleeps):
    model, calls = failing_then([http_error(503), http_error(503)])
    use_model(monkeypatch, model)
    assert real_run(tmp_path / "ok") == 0
    assert sleeps == [30.0, 60.0] and len(load(result_path(tmp_path / "ok"))["attempts"]) == 3


def test_503_that_keeps_failing_stops_and_suggests_another_model(tmp_path, monkeypatch, sleeps, capsys):
    use_model(monkeypatch, failing_then([http_error(503)] * 10)[0])

    assert real_run(tmp_path) == 1

    assert sleeps == [30.0, 60.0, 120.0]
    assert not result_path(tmp_path).exists()
    assert "AIE_MODEL" in capsys.readouterr().err


def test_a_timeout_without_a_status_is_retried_like_503(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, failing_then([ModelAPIError("gemini-test", "ReadTimeout")])[0])
    assert real_run(tmp_path) == 0
    assert sleeps == [30.0]


def test_400_stops_without_retrying(tmp_path, monkeypatch, sleeps, capsys):
    model, calls = failing_then([http_error(400, {"error": {"message": "API key not valid"}})])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 1

    assert sleeps == [] and len(calls) == 1
    assert not result_path(tmp_path).exists()
    assert "API エラー" in capsys.readouterr().err


def test_usage_limit_and_unexpected_model_behavior_are_results(tmp_path, monkeypatch):
    def loop(messages, info):  # 検索を呼び続けて、終わらないモデル
        return ModelResponse(parts=[ToolCallPart("search_glossary", {"query": "LoRA"})])

    use_model(monkeypatch, FunctionModel(loop))
    monkeypatch.setattr(compare, "REQUEST_LIMIT", 3)
    assert real_run(tmp_path / "limit") == 0
    record = load(result_path(tmp_path / "limit"))
    assert record["status"] == "usage_limit_exceeded" and record["error"]["type"] == "UsageLimitExceeded"
    assert record["answer"] == "" and len(record["tool_calls"]) >= 1  # 途中までの会話は残る

    def broken(messages, info):
        raise UnexpectedModelBehavior("想定外の応答")

    use_model(monkeypatch, FunctionModel(broken))
    assert real_run(tmp_path / "broken") == 0
    assert load(result_path(tmp_path / "broken"))["status"] == "unexpected_model_behavior"


def test_make_quiz_gets_the_same_term_for_the_same_question_and_trial(tmp_path, monkeypatch):
    """出題は乱数で決まるので、質問と試行ごとに seed を固定する（変種どうしで同じ語が出る）。"""
    import random

    from app import tools

    def ask_for_a_quiz(messages, info):  # 1 回目は make_quiz を呼び、結果が戻ったら答える
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("make_quiz", {"lecture": 2})])
        return ModelResponse(parts=[TextPart("出題しました。")])

    use_model(monkeypatch, FunctionModel(ask_for_a_quiz))
    assert real_run(tmp_path, question="q03") == 0
    assert real_run(tmp_path, question="q03", variant="quiz_doc_no_when", extra_args=["--include-optional"]) == 0

    random.seed("q03:t1")
    expected = tools.make_quiz(2)
    for variant in ("baseline", "quiz_doc_no_when"):
        record = load(result_path(tmp_path, variant=variant, qid="q03"))
        assert record["tool_calls"][0]["result"] == expected, variant  # 変種どうしで、同じ語が出る


def test_the_instructions_the_model_saw_are_recorded(tmp_path, monkeypatch):
    use_model(monkeypatch, text_model())
    assert real_run(tmp_path, variant="no_search_rule") == 0
    record = load(result_path(tmp_path, variant="no_search_rule"))
    assert "必ず search_glossary で用語集を確認する" not in record["instructions"]
    assert record["instructions_match"] is True  # 最初の request に入っていた指示が、設定した変種と同じ
    assert record["tools"] and record["variant_sha256"] == compare.build_effective(compare.load_config(), "no_search_rule").sha256


def test_instructions_match_is_false_when_the_variant_did_not_reach_the_model(tmp_path, monkeypatch):
    from app.agent import agent

    real_override = agent.override
    # 指示の上書きだけが効かなかった、という状況にする
    monkeypatch.setattr(agent, "override", lambda **kw: real_override(**{k: v for k, v in kw.items() if k != "instructions"}))
    use_model(monkeypatch, text_model())

    assert real_run(tmp_path, variant="no_search_rule") == 0

    record = load(result_path(tmp_path, variant="no_search_rule"))
    assert record["instructions_match"] is False
    assert "必ず search_glossary で用語集を確認する" in record["instructions"]  # 実際に見えた指示を記録している


# --- ask エンジン ---


def flaky_client(errors):
    class Flaky:
        calls = 0

        def __init__(self):
            self.models = self

        def generate_content(self, *, model, contents):
            Flaky.calls += 1
            if Flaky.calls <= len(errors):
                raise errors[Flaky.calls - 1]
            usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=2, thoughts_token_count=3, total_token_count=6)
            return SimpleNamespace(text="回答", usage_metadata=usage)

    return Flaky


def test_ask_engine_retries_a_rate_limit_and_counts_tokens(tmp_path, monkeypatch, sleeps):
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [{"retryDelay": "5s"}]}}
    monkeypatch.setattr(compare, "FakeGenAIClient", flaky_client([genai_errors.ClientError(429, body)]))

    assert run("run", "--dry-run", "--engine", "ask", "--question", "q01", "--trials", 1, "--out", tmp_path) == 0

    assert sleeps == [5.0]
    record = load(result_path(tmp_path, engine="ask", model=f"{MODEL}-dry-run"))
    assert record["usage"] == {"requests": 1, "tool_calls": 0, "input_tokens": 1, "output_tokens": 5, "candidates_tokens": 2,
                               "thoughts_tokens": 3, "total_tokens": 6}
    assert "[metrics]" in result_path(tmp_path, engine="ask", model=f"{MODEL}-dry-run", suffix=".trace.txt").read_text(encoding="utf-8")


def test_ask_engine_daily_limit_stops_with_exit_code_2(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "FakeGenAIClient", flaky_client([genai_errors.ClientError(429, RATE_PER_DAY)]))
    assert run("run", "--dry-run", "--engine", "ask", "--question", "q01", "--trials", 1, "--out", tmp_path) == 2
    assert not result_path(tmp_path, engine="ask", model=f"{MODEL}-dry-run").exists()


# --- adk と pi の出力の解析 ---


def event(**fields):
    return json.dumps({"author": "glossary_agent", "id": "e", **fields}, ensure_ascii=False)


def test_parse_adk_jsonl_handles_log_lines_mixed_keys_and_thoughts():
    text = "\n".join([
        "Log setup complete: /tmp/agents_log/agent.log",  # JSON ではない行
        "To access latest log: tail -F /tmp/agents_log/agent.latest.log",
        event(content={"role": "model", "parts": [{"functionCall": {"id": "c1", "name": "search_glossary", "args": {"query": "LoRA"}}}]},
              usageMetadata={"promptTokenCount": 100, "candidatesTokenCount": 10, "thoughtsTokenCount": 5, "totalTokenCount": 115}),  # camelCase
        event(content={"role": "user", "parts": [{"function_response": {"id": "c1", "name": "search_glossary", "response": {"result": "R1"}}}]}),  # snake_case
        '{"壊れた JSON": ',
        "[1, 2]",
        event(content={"role": "model", "parts": [{"text": "考え中", "thought": True}, {"text": "最終的な"}, {"text": "回答\n第2回"}]},
              usage_metadata={"prompt_token_count": 200, "candidates_token_count": 20, "thoughts_token_count": 7}, session_id="s", node_path="a@1"),
    ])

    parsed = compare.parse_adk_jsonl(text)

    assert parsed["answer"] == "最終的な回答\n第2回"  # thought のパートは除く
    assert [(c["name"], c["args"], c["result"], c["outcome"]) for c in parsed["tool_calls"]] == [
        ("search_glossary", {"query": "LoRA"}, "R1", "success")]
    assert parsed["usage"] == {"requests": 2, "tool_calls": 1, "input_tokens": 300, "output_tokens": 42, "candidates_tokens": 30,
                               "thoughts_tokens": 12, "total_tokens": 115}
    assert len(parsed["skipped_lines"]) == 4 and parsed["errors"] == []


def test_parse_adk_jsonl_reads_the_error_event_adk_really_prints():
    line = ('{"author": "glossary_agent", "session_id": "b5861e67", "node_path": "glossary_agent@1", "id": "b29c5ff8", '
            '"errorCode": "ValueError", "errorMessage": "No API key was provided. Please pass a valid API key.", '
            '"invocationId": "e-79a9", "nodeInfo": {"path": "glossary_agent@1"}, "timestamp": 1793711088.79}')

    parsed = compare.parse_adk_jsonl("Log setup complete: /tmp/x.log\n" + line + "\n")

    assert parsed["answer"] == "" and parsed["usage"]["requests"] == 0
    assert parsed["errors"] == [{"code": "ValueError", "message": "No API key was provided. Please pass a valid API key."}]


def test_the_adk_sample_has_the_shape_adk_prints():
    parsed = compare.parse_adk_jsonl(compare.ADK_SAMPLE_JSONL)
    assert parsed["usage"]["requests"] == 2 and [c["name"] for c in parsed["tool_calls"]] == ["search_glossary"] * 2
    assert parsed["answer"].endswith("第2回") and all(c["result"] for c in parsed["tool_calls"])


def test_parse_pi_jsonl():
    parsed = compare.parse_pi_jsonl("npm notice の行など\n" + compare.PI_SAMPLE_JSONL)

    assert parsed["session"]["cwd"] == "/tmp/pi-work" and parsed["answer"] == "（予行）Pi の回答です。"
    assert parsed["usage"] == {"requests": 2, "tool_calls": 1, "input_tokens": 2700, "output_tokens": 120, "cache_read_tokens": 0, "total_tokens": 2820}
    call = parsed["tool_calls"][0]
    assert (call["name"], call["args"], call["outcome"], call["response_index"]) == ("grep", {"pattern": "LoRA", "path": "data"}, "success", 1)
    assert "LoRA" in call["result"] and parsed["skipped_lines"] == ["npm notice の行など"]

    failed = "\n".join(json.dumps(e) for e in [
        {"type": "turn_start"},
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "delayMs": 2000, "errorMessage": "429 RESOURCE_EXHAUSTED"},
        {"type": "tool_execution_start", "toolCallId": "t", "toolName": "read", "args": {"path": "x"}},
        {"type": "tool_execution_end", "toolCallId": "t", "toolName": "read", "result": {"content": [{"type": "text", "text": "ENOENT"}]}, "isError": True},
        {"type": "message_end", "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": "429 RESOURCE_EXHAUSTED PerDay"}},
        {"type": "auto_retry_end", "success": False, "finalError": "429"},
    ])
    parsed = compare.parse_pi_jsonl(failed)
    assert parsed["answer"] == "" and parsed["auto_retries"][0]["errorMessage"] == "429 RESOURCE_EXHAUSTED"
    assert parsed["tool_calls"][0]["outcome"] == "failed" and len(parsed["errors"]) == 2


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier", "daily_limit"),
        ("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinute retryDelay: '30s'", "retry"),
        ("503 UNAVAILABLE The model is overloaded", "retry"),
        ("Error: Command timed out after 180s", "retry"),
        ("No API key was provided", "stop"),
        ("400 INVALID_ARGUMENT", "stop"),
        ("404 NOT_FOUND models/xyz is not found", "stop"),
    ],
)
def test_errors_of_external_commands_are_classified_like_api_errors(text, action):
    assert compare.judge(compare.failure_from_text(text), {})[0] == action
    assert compare.judge(compare.failure_from_text("429 retryDelay: '30s'"), {}) == ("retry", 30.0)


def test_adk_engine_runs_the_command_retries_and_keeps_the_raw_output(tmp_path, monkeypatch, sleeps):
    replies = [("", "Error: 429 RESOURCE_EXHAUSTED retryDelay: '9s'", 1), (compare.ADK_SAMPLE_JSONL, "stderr の警告", 0)]
    seen = []

    def fake_run(cmd, *, cwd, env):
        seen.append((cmd, cwd, env["PYTHONPATH"]))
        return replies.pop(0)

    monkeypatch.setattr(compare, "run_subprocess", fake_run)
    monkeypatch.setattr(compare.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert run("run", "--level", "adv-a", "--question", "q02", "--trials", 1, "--allow-dirty", "--interval", 0, "--out", tmp_path) == 0

    question = compare.load_config()["questions"]["q02"]["text"]
    assert sleeps == [9.0] and len(seen) == 2
    assert seen[0] == (compare.adk_command(question), compare.REPO, str(compare.REPO))
    assert seen[0][0][:8] == ["uv", "run", "--with", "google-adk==2.13.0", "adk", "run", "--in_memory", "--jsonl"]
    record = load(result_path(tmp_path, engine="adk", qid="q02"))
    assert record["status"] == "ok" and record["exit_code"] == 0 and record["usage"]["requests"] == 2
    assert [a.get("action", "ok") for a in record["attempts"]] == ["retry", "ok"]
    assert result_path(tmp_path, engine="adk", qid="q02", suffix=".raw.jsonl").read_text(encoding="utf-8") == compare.ADK_SAMPLE_JSONL
    assert result_path(tmp_path, engine="adk", qid="q02", suffix=".stderr.txt").read_text(encoding="utf-8") == "stderr の警告"


def test_adk_engine_stops_on_a_missing_key_and_on_a_daily_limit(tmp_path, monkeypatch, sleeps, capsys):
    monkeypatch.setattr(compare.shutil, "which", lambda name: f"/usr/bin/{name}")
    args = ("run", "--level", "adv-a", "--question", "q01", "--trials", 1, "--allow-dirty", "--interval", 0, "--out", tmp_path)

    monkeypatch.setattr(compare, "run_subprocess", lambda cmd, *, cwd, env: ("", "Error: No API key was provided.", 1))
    assert run(*args) == 1
    assert sleeps == []

    monkeypatch.setattr(compare, "run_subprocess", lambda cmd, *, cwd, env: ("", "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel", 1))
    assert run(*args) == 2
    assert not result_path(tmp_path, engine="adk").exists()
    assert "明日" in capsys.readouterr().err


def test_pi_engine_runs_in_a_temp_folder_that_has_only_data(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *, cwd, env):
        seen.update(cmd=cmd, cwd=Path(cwd), top=sorted(p.name for p in Path(cwd).iterdir()), data=sorted(p.name for p in (Path(cwd) / "data").iterdir()))
        return compare.PI_SAMPLE_JSONL, "", 0

    monkeypatch.setattr(compare, "run_subprocess", fake_run)
    monkeypatch.setattr(compare.shutil, "which", lambda name: "/usr/bin/pi")
    monkeypatch.setattr(compare, "pi_version", lambda: "1.5.0")

    assert run("run", "--level", "adv-b", "--question", "q01", "--trials", 1, "--allow-dirty", "--interval", 0,
               "--pi-model", "gemini-test", "--out", tmp_path) == 0

    cmd = seen["cmd"]
    assert cmd[:6] == ["pi", "--mode", "json", "--no-session", "--no-context-files", "--tools"] and cmd[6] == "read,grep,find,ls"
    assert cmd[cmd.index("--provider") + 1] == "google" and cmd[cmd.index("--model") + 1] == "gemini-test"
    assert cmd[-1] == compare.load_config()["questions"]["q01"]["text"]
    assert seen["top"] == ["data"] and "glossary.json" in seen["data"]  # problem.md や結果は見えない
    assert seen["cwd"] != compare.REPO
    record = load(result_path(tmp_path, engine="pi", model="gemini-test"))
    assert record["pi_version"] == "1.5.0" and record["model"] == "gemini-test" and record["usage"]["requests"] == 2


def test_pi_missing_prints_how_to_install_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare.shutil, "which", lambda name: None)
    assert run("run", "--level", "adv-b", "--trials", 1, "--allow-dirty", "--out", tmp_path) == 1
    assert "npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.5.0" in capsys.readouterr().err


# --- 未コミットの変更の確認 ---


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)


def test_git_state_looks_only_at_the_files_that_affect_the_run(tmp_path):
    repo = tmp_path / "repo"
    for name in ("app/agent.py", "adk_agent/glossary_agent/agent.py", "lectures/03/compare.py", "lectures/03/conditions.toml", "README.md"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("元\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    clean = compare.git_state(repo)
    assert clean["dirty"] is False and len(clean["commit"]) == 40

    (repo / "README.md").write_text("関係ない変更\n", encoding="utf-8")
    (repo / "lectures/03/answer.md").write_text("関係ない新規ファイル\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is False

    (repo / "app/agent.py").write_text("変更\n", encoding="utf-8")
    state = compare.git_state(repo)
    assert state["dirty"] is True and any("app/agent.py" in line for line in state["dirty_files"])

    git(repo, "checkout", "--", "app/agent.py")
    (repo / "adk_agent/glossary_agent/new.py").write_text("新規\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is True  # 追加しただけの新規ファイルも、未コミットの変更

    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "more")
    assert compare.git_state(repo)["dirty"] is False

    (repo / "lectures/03/conditions.toml").write_text("直した\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is True


def test_git_state_ignores_the_session_folder_that_adk_run_creates(tmp_path):
    """対話の adk run は、会話を adk_agent/glossary_agent/.adk/ に保存する。adk_agent/.gitignore が無視するので、確認は止まらない。"""
    repo = tmp_path / "repo"
    for name in ("app/agent.py", "adk_agent/glossary_agent/agent.py", "lectures/03/compare.py", "lectures/03/conditions.toml"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("元\n", encoding="utf-8")
    real_ignore = compare.REPO / "adk_agent" / ".gitignore"  # 本物の .gitignore を、そのまま commit しておく
    (repo / "adk_agent/.gitignore").write_text(real_ignore.read_text(encoding="utf-8"), encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    assert compare.git_state(repo)["dirty"] is False

    session = repo / "adk_agent/glossary_agent/.adk/session.db"
    session.parent.mkdir(parents=True)
    session.write_text("会話の中身\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is False  # .adk/ は無視される

    (repo / "adk_agent/glossary_agent/new.py").write_text("新規\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is True  # ほかの新しいファイルは、今までどおり未コミットの変更


def test_preflight_refuses_a_dirty_tree_unless_allowed_or_dry_run(tmp_path):
    cfg = compare.load_config()
    git_info = {"commit": "abc", "dirty": True, "dirty_files": [" M app/agent.py"]}
    ctx = compare.Context(cfg=cfg, out=tmp_path, model="m", pi_model="m", git=git_info)
    plan = compare.build_plan(cfg, levels=["exercise"], engines=["ask"])

    with pytest.raises(compare.StopRun) as stopped:
        compare.preflight(ctx, plan, SimpleNamespace(allow_dirty=False))
    assert stopped.value.code == 1 and "app/agent.py" in stopped.value.message and "--allow-dirty" in stopped.value.message

    compare.preflight(ctx, plan, SimpleNamespace(allow_dirty=True))
    compare.preflight(dataclasses.replace(ctx, dry_run=True), plan, SimpleNamespace(allow_dirty=False))


def test_preflight_explains_untracked_files_only_when_there_are_some(tmp_path):
    cfg = compare.load_config()
    plan = compare.build_plan(cfg, levels=["exercise"], engines=["ask"])
    note = "?? は git が追跡していないファイル（実行で作られたものなど）。中身を確かめて、不要なら消してください。"

    def stop_message(dirty_files):
        git_info = {"commit": "abc", "dirty": True, "dirty_files": dirty_files}
        ctx = compare.Context(cfg=cfg, out=tmp_path, model="m", pi_model="m", git=git_info)
        with pytest.raises(compare.StopRun) as stopped:
            compare.preflight(ctx, plan, SimpleNamespace(allow_dirty=False))
        assert stopped.value.code == 1  # ?? の行があっても、確認は厳しいまま（止まる）
        return stopped.value.message

    only_untracked = stop_message(["?? adk_agent/glossary_agent/.adk/"])  # 実際に起きた表示
    assert note in only_untracked and "?? adk_agent/glossary_agent/.adk/" in only_untracked
    mixed = stop_message([" M app/agent.py", "?? adk_agent/glossary_agent/new.py"])
    assert note in mixed and " M app/agent.py" in mixed
    assert note not in stop_message([" M app/agent.py"])


def test_run_without_an_api_key_stops_with_instructions(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert real_run(tmp_path) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err


# --- チェック ---


def record(**fields):
    base = {"engine": "pydantic", "answer": "", "tool_calls": [], "approvals": [], "saved_files": [], "outside_files": [], "usage": {}}
    return {**base, **fields}


def call(name, args, response_index=1):
    return {"name": name, "args": args, "response_index": response_index, "result": "戻り値", "outcome": "success"}


def passes(check, **fields):
    return compare.evaluate_check(check, record(**fields))


def test_answer_checks():
    assert passes({"kind": "answer_contains_all", "values": ["基礎", "実装"]}, answer="基礎と実装") is True
    assert passes({"kind": "answer_contains_all", "values": ["基礎", "発展"]}, answer="基礎と実装") is False
    assert passes({"kind": "answer_contains_any", "values": ["発展", "実装"]}, answer="基礎と実装") is True
    assert passes({"kind": "answer_not_contains", "values": ["初級", "上級"]}, answer="基礎と実装") is True
    assert passes({"kind": "answer_not_contains", "values": ["初級", "上級"]}, answer="初級です") is False
    not_in_glossary = compare.checks_for(compare.load_config(), "q04")[0]  # 「用語集にない」と伝える
    assert not_in_glossary["kind"] == "answer_regex"
    for sentence in ("用語集にはありません。", "RLHFは用語集には見つかりませんでした。", "用語集には載っていません。", "用語集にはない用語です。"):
        assert passes(not_in_glossary, answer=sentence) is True, sentence
    assert passes(not_in_glossary, answer="用語集に出ています。ありません。") is False  # 「。」をまたいだ一致は数えない
    assert passes(not_in_glossary, answer="RLHFは人間のフィードバックによる強化学習です。") is False
    assert passes({"kind": "answer_not_regex", "pattern": "```"}, answer="コードは書きません") is True
    assert passes({"kind": "answer_not_regex", "pattern": "```"}, answer="```python\n```") is False


def test_ending_lecture_looks_at_the_last_two_non_empty_lines():
    check = {"kind": "ending_lecture"}
    assert passes(check, answer="本文\n\n第2回\n\n") is True
    assert passes(check, answer="第2回の話\n本文\n\n中間\n最後") is False  # 最後の 2 行より前
    assert passes(check, answer="本文\n関係する回: 第３回") is True  # 全角の数字も
    assert passes(check, answer="本文\n関係する回: 第2回\n\n以上です。") is True  # 最後の 1 行ではなく、最後の 2 行
    assert passes({"kind": "ending_lecture", "lecture": 2}, answer="本文\n第2回") is True
    assert passes({"kind": "ending_lecture", "lecture": 2}, answer="本文\n第3回") is False
    assert passes({"kind": "ending_lecture", "expect": False}, answer="本文\n用語集にはありません") is True
    assert passes({"kind": "ending_lecture", "expect": False}, answer="本文\n第1回") is False


def test_tool_checks():
    calls = [call("search_glossary", {"query": "LoRA"}), call("search_glossary", {"query": "QLoRA"}), call("make_quiz", {"lecture": 2}, 2)]
    assert passes({"kind": "tool_called", "name": "search_glossary"}, tool_calls=calls) is True
    assert passes({"kind": "tool_called", "name": "search_glossary", "args_contains": '"query":"QLoRA"'}, tool_calls=calls) is True
    assert passes({"kind": "tool_called", "name": "search_glossary", "args_contains": '"query":"RAG"'}, tool_calls=calls) is False
    assert passes({"kind": "tool_called", "name": "make_quiz", "args_contains": '"lecture":2', "after": "search_glossary"}, tool_calls=calls) is True
    assert passes({"kind": "tool_called", "name": "search_glossary", "after": "make_quiz"}, tool_calls=calls) is False
    same_step = [call("search_glossary", {"query": "LoRA"}), call("make_quiz", {"lecture": 2})]  # 同じ応答で一緒に呼んだ
    assert passes({"kind": "tool_called", "name": "make_quiz", "after": "search_glossary"}, tool_calls=same_step) is False
    assert passes({"kind": "tool_not_called", "name": "save_learning_log"}, tool_calls=calls) is True
    assert passes({"kind": "tool_count", "name": "search_glossary", "min": 2, "max": 2}, tool_calls=calls) is True
    assert passes({"kind": "tool_count", "name": "search_glossary", "max": 1}, tool_calls=calls) is False
    returns = [{"name": "search_glossary", "args": {}, "result": '[{"term": "PEFT"}, {"term": "LoRA"}]'}]
    assert passes({"kind": "tool_result_contains", "name": "search_glossary", "values": ['"term": "LoRA"']}, tool_calls=returns) is True
    assert passes({"kind": "tool_result_contains", "name": "search_glossary", "values": ['"term": "量子化"']}, tool_calls=returns) is False


def test_count_checks_and_applicable_engines():
    assert passes({"kind": "requests_at_most", "value": 4}, usage={"requests": 3}) is True
    assert passes({"kind": "requests_at_most", "value": 2}, usage={"requests": 3}) is False
    assert passes({"kind": "approvals_count", "value": 2}, approvals=[{}, {}]) is True
    assert passes({"kind": "approvals_count", "value": 1}, approvals=[{}, {}]) is False
    assert passes({"kind": "files_saved", "value": 0}, saved_files=[]) is True
    assert passes({"kind": "files_saved", "max": 1}, saved_files=[{}, {}]) is False
    assert passes({"kind": "files_outside_none"}, outside_files=["README.md"]) is False
    assert passes({"kind": "files_outside_none"}, outside_files=[]) is True
    only_tools = {"kind": "tool_called", "name": "search_glossary", "engines": ["pydantic", "adk"]}
    assert compare.evaluate_check(only_tools, record(engine="ask")) is None  # ツールを使わないエンジンには当てはまらない
    assert compare.evaluate_check(only_tools, record(engine="pi")) is None


def test_every_check_in_conditions_toml_can_be_evaluated():
    cfg = compare.load_config()
    for qid in [*cfg["questions"], *cfg["scenarios"]]:
        for check in compare.checks_for(cfg, qid):
            for engine in compare.ENGINES:
                assert compare.evaluate_check(check, record(engine=engine)) in (True, False, None), (qid, check)


# --- 集計の表 ---


def write_record(out, engine, variant, qid, trial, model=MODEL, **fields):
    rec = {**record(schema=1, engine=engine, variant=variant, qid=qid, trial=trial), "status": "ok", "duration_s": 1.0, **fields}
    path = result_path(out, engine, variant, qid, trial, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return path


def test_summary_tables_show_signatures_checks_and_ranges(tmp_path):
    two_searches = [call("search_glossary", {"query": "LoRA"}), call("search_glossary", {"query": "QLoRA"})]
    one_search = [call("search_glossary", {"query": "LoRA"})]
    answers = ["4ビットに量子化する。\n第2回", "4ビットに量子化する。\n第2回", "違いは学習の方法です。\n第2回"]
    for trial, (calls, requests, tokens, seconds, answer) in enumerate([
        (two_searches, 2, 100, 3.0, answers[0]), (two_searches, 2, 200, 5.0, answers[1]), (one_search, 3, 300, 9.0, answers[2])], start=1):
        write_record(tmp_path, "pydantic", "baseline", "q02", trial, tool_calls=calls, answer=answer, duration_s=seconds,
                     usage={"requests": requests, "tool_calls": len(calls), "input_tokens": tokens, "output_tokens": tokens // 10})
    write_record(tmp_path, "pydantic", "no_search_rule", "q02", 1, answer="第2回", usage={"requests": 1, "tool_calls": 0})

    assert run("summarize", "--out", tmp_path, "--level", "basic") == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "search_glossary×2 (2/3) / search_glossary (1/3)｜tc 2,2,1｜req 2,2,3" in summary  # 表A
    assert "ツールなし (1/1)｜tc 0｜req 1 [1/3回]" in summary  # 試行が足りないものは、そう書く
    assert "未実行" in summary  # 流す予定なのに、まだ結果が無いもの
    assert "| q02 | 4ビット（4bit）に触れる | 2/3 | 0/1 |" in summary  # 表B
    assert "| q02 | 検索をちょうど 2 回呼ぶ | 2/3 | 0/1 |" in summary
    assert "| V0 baseline | q02 | 3 | 200 (100–300) | 20 (10–30) | 5.0 (3.0–9.0) | 2 (2–3) |" in summary  # 表C
    assert summary.count("<summary>V0 baseline / q02（3 回）</summary>") == 1
    assert "違いは学習の方法です。" in summary


def test_summarize_parses_adk_and_pi_again_from_the_raw_output(tmp_path):
    for engine, sample in (("adk", compare.ADK_SAMPLE_JSONL), ("pi", compare.PI_SAMPLE_JSONL)):
        path = write_record(tmp_path, engine, "baseline", "q01", 1, answer="", usage={}, status="no_answer")  # 古い解析の結果（回答なし）
        path.with_name("t1.raw.jsonl").write_text(sample, encoding="utf-8")

    assert run("summarize", "--out", tmp_path) == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "（予行）LoRAは低ランク行列だけを学習する手法で" in summary  # adk
    assert "（予行）Pi の回答です。" in summary  # pi
    assert "search_glossary×2 (1/1)｜tc 2｜req 2" in summary and "grep (1/1)｜tc 1｜req 2" in summary
    assert "no_answer" not in summary


def test_status_counts_planned_done_and_remaining(tmp_path, capsys):
    for qid in ("q01", "q02"):
        write_record(tmp_path, "pydantic", "baseline", qid, 1)

    assert run("status", "--out", tmp_path, "--level", "basic", "--variant", "baseline", "--trials", 3) == 0

    lines = capsys.readouterr().out.splitlines()
    row = next(line for line in lines if "pydantic/baseline" in line)
    assert row.split()[2:] == ["10", "30", "2", "28", "70"]  # 条件・予定・済み・残り・残りの推定リクエスト（28 × 2.5）
    assert any("合計" in line and "30" in line for line in lines)
