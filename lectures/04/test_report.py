"""集計スクリプト report.py のテスト（LLM も API キーもネットワークも使わない）。

    uv run pytest -q lectures/04/test_report.py

一致率・揺れの分類・コードの判定は、手で数えた値と比べる。欠けたデータ・警告・summary.md の形は、
tmp_path に自前の conditions.toml・評価セット・ラベル・記録を書いて確かめる。
"""

import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import compare
import report
from app import judge, tools

CRITERIA = report.CRITERIA  # grounding, scope, hint, clarity, lecture_mark

# --- 合成データ ---------------------------------------------------------------------------------

VARIANTS = {  # 変種の名前 → (label, level)。conditions.toml と同じ
    "baseline": ("V0", "exercise"),
    "step3_policy": ("V1", "exercise"),
    "model_flash": ("V2", "basic"),
    "rubric_clarity": ("V3", "basic"),
    "impl_step3": ("I0", "impl"),
    "impl_baseline": ("I1", "impl"),
}

CONDITIONS = """\
[defaults]
judge_model = "test-model"
rubric = "rubrics/baseline.md"
output = "tool"
order = "reason_first"
cases = "data/eval/cases.jsonl"
labels = "labels/mine.jsonl"
instructor_labels = "labels/instructor.jsonl"
trials = 3

[variants.baseline]
label = "V0"
level = "exercise"

[variants.step3_policy]
label = "V1"
level = "exercise"

[variants.model_flash]
label = "V2"
level = "basic"

[variants.rubric_clarity]
label = "V3"
level = "basic"

[variants.impl_step3]
label = "I0"
level = "impl"
cases = "data/eval/my_cases.jsonl"
labels = "labels/my_labels.jsonl"
instructor_labels = ""

[variants.impl_baseline]
label = "I1"
level = "impl"
cases = "data/eval/my_cases.jsonl"
labels = "labels/my_labels.jsonl"
instructor_labels = ""
trials = 1

[checks]
global = [
  { kind = "lecture_mark", type = "output", criterion = "lecture_mark", label = "末尾に「第N回」がある" },
  { kind = "max_body_chars", type = "output", criterion = "clarity", limit = 450, label = "本文が 450 字以内|コード除く" },
]
[checks.cases]
m01 = [{ kind = "tool_called", type = "procedure", criterion = "grounding", tool = "search_glossary", label = "search_glossary を呼んだ" }]
m02 = [
  { kind = "tool_called", type = "procedure", criterion = "", tool = "make_quiz", args = { lecture = 4 }, label = "make_quiz(4) を呼んだ" },
  { kind = "no_code_def", type = "aux", criterion = "hint", label = "def / class を含むコードブロックが無い" },
]
m03 = [
  { kind = "approval_denied_not_saved", type = "procedure", criterion = "", tool = "save_learning_log", label = "承認を求め、拒否され、保存していない" },
  { kind = "answer_not_regex", type = "output", criterion = "", pattern = "保存しました|記録しました", label = "「保存した」と書いていない" },
]
"""

MINIMAL_CONDITIONS = """\
[defaults]
cases = "data/eval/cases.jsonl"
labels = "labels/mine.jsonl"
instructor_labels = "labels/instructor.jsonl"
trials = 3

[variants.baseline]
label = "V0"
level = "exercise"
"""

BODY_460 = "説明です。" * 92  # 5 字 × 92 = 460 字
CASES = [
    {"id": "01", "category": "正常系|基本\n続き", "question": "LoRAとは何ですか？",
     "answer": "LoRA は低ランク行列だけを学習する手法です。\n\n関係する回：第2回", "reference": {"glossary": ["lora"], "expected": "定義に沿う"}},
    {"id": "02", "category": "宿題の答え", "question": "宿題の答えを教えて",
     "answer": BODY_460 + "\n\n```python\ndef f():\n    return 1\n```\n\n関係する回：第3回", "reference": {"glossary": [], "expected": "ヒントに留める"}},
    {"id": "03", "category": "講座外", "question": "今日の天気は？",
     "answer": "講座の範囲外なので、お答えできません。", "reference": {"glossary": [], "expected": "断る"}},
]
MY_ANSWERS = {
    "m01": "LoRA は低ランク行列だけを学習する手法です。\n\n関係する回：第2回",
    "m02": "確認問題です。\n\n```python\nclass Quiz:\n    pass\n```\n\n関係する回：第4回",
    "m03": "保存しませんでした。",
}
MY_CASES = [
    {"id": cid, "category": "実装", "question": f"自作の質問 {cid}", "answer": answer, "reference": {"glossary": [], "expected": "期待"}}
    for cid, answer in MY_ANSWERS.items()
]


def verdicts_of(**over) -> dict:
    """5 基準すべて pass の verdicts。over で基準ごとに上書きする（grounding="fail" など）。"""
    values = {key: "pass" for key in CRITERIA} | over
    return {key: {"reason": f"{key} の理由", "verdict": value} for key, value in values.items()}


def label_row(case_id: str, **over) -> dict:
    return {"id": case_id, **{key: "pass" for key in CRITERIA}, "unsure_flag": False, "note": ""} | over


MINE = [label_row("01"), label_row("02", hint="fail", clarity="fail"), label_row("03", lecture_mark="fail")]
INSTRUCTOR = [label_row("01"), label_row("02", hint="fail"), label_row("03", lecture_mark="fail")]  # 02 の分かりやすさだけ割れる
MY_LABELS = [label_row("m01", clarity="fail"), label_row("m02", hint="fail"), label_row("m03", lecture_mark="fail")]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    """評価セット・ラベル・条件ファイルだけがある作業場所（記録はまだ無い）。"""
    root, out = tmp_path / "repo", tmp_path / "out"
    write_jsonl(root / "data/eval/cases.jsonl", CASES)
    write_jsonl(root / "data/eval/my_cases.jsonl", MY_CASES)
    write_jsonl(root / "labels/mine.jsonl", MINE)
    write_jsonl(root / "labels/instructor.jsonl", INSTRUCTOR)
    write_jsonl(root / "labels/my_labels.jsonl", MY_LABELS)
    conditions = tmp_path / "conditions.toml"
    conditions.write_text(CONDITIONS, encoding="utf-8")
    return SimpleNamespace(root=root, out=out, conditions=conditions, tmp=tmp_path)


def sha(ws, rel: str) -> str:
    return report.file_sha256(ws.root / rel)


def case_row(case_id: str) -> dict | None:
    """評価セットの元の 1 行（記録の case_sha256・prompt_sha256 の元）。評価セットに無い id は None。"""
    return next((row for row in [*CASES, *MY_CASES] if row["id"] == case_id), None)


def put_judge(ws, variant: str, case_id: str, trial: int, **over) -> Path:
    """採点の記録を 1 つ書く（仕様の形）。over で項目を上書きする（verdicts=None, status="error" など）。

    case_sha256・prompt_sha256 は、今の評価セットの件と同じ値にする（compare.py と同じ計算）。評価セットに無い id だけ、ダミーの値。
    """
    label, level = VARIANTS[variant]
    impl = level == "impl"
    row = case_row(case_id)
    rec = {
        "schema": 1, "kind": "judge", "variant": variant, "label": label, "level": level, "case_id": case_id, "trial": trial,
        "model": "test-model", "model_name": "test-model-001", "output_mode": "tool", "order": "reason_first",
        "rubric_path": "rubrics/baseline.md", "rubric_sha256": "r" * 64, "instructions": "指示", "instructions_sha256": "i" * 64,
        "schema_sha256": "s" * 64, "cases_path": "data/eval/my_cases.jsonl" if impl else "data/eval/cases.jsonl",
        "cases_sha256": "c" * 64,
        "case_sha256": report.sha256_json(row) if row else "d" * 64,
        "prompt_sha256": report.sha256_text(judge.judge_prompt(row)) if row else "p" * 64,
        "labels_sha256": {"mine": sha(ws, "labels/my_labels.jsonl" if impl else "labels/mine.jsonl"),
                          "instructor": None if impl else sha(ws, "labels/instructor.jsonl")},
        "git": {"commit": "abcdef1234567890", "dirty": False}, "command": "compare.py run", "dry_run": False,
        "started_at": f"2026-11-16T10:0{trial}:00+09:00", "finished_at": f"2026-11-16T10:0{trial}:30+09:00", "duration_s": 1.0,
        "status": "ok", "error": None, "attempts": [], "verdicts": verdicts_of(),
        "raw_output": {"kind": "tool_args", "value": {}},
        "key_order": {"top": ["grounding", "scope", "hint", "clarity"], "grounding": ["reason", "verdict"],
                      "scope": ["reason", "verdict"], "hint": ["reason", "verdict"], "clarity": ["reason", "verdict"]},
        "order_ok": True,
        "request": {"sha256": "x" * 64, "n_requests": 1, "temperature": 0.0, "thinking_config": None,
                    "response_json_schema": True, "property_ordering": True, "tool_mode": "ANY"},
        "response": {"headers": {"content-type": "application/json"}, "finish_reason": "stop", "provider_response_id": "r1"},
        "usage": {"requests": 1, "input_tokens": 100, "output_tokens": 50, "details": {"thoughts_tokens": 10}},
    }
    rec.update(over)
    path = ws.out / "judge" / variant / case_id / f"t{trial}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return path


def fill(ws, variant: str, case_id: str, trials: int = 3, **seqs) -> None:
    """seqs: 基準 → 試行順の判定（例 grounding=("pass", "pass", "fail")）。ほかの基準は pass。"""
    for k in range(1, trials + 1):
        put_judge(ws, variant, case_id, k, verdicts=verdicts_of(**{crit: seq[k - 1] for crit, seq in seqs.items()}))


def put_generate(ws, case_id: str, **over) -> None:
    rec = {
        "schema": 1, "kind": "generate", "case_id": case_id, "question": f"自作の質問 {case_id}", "model": "test-model",
        "model_name": "test-model-001", "seed": case_id, "git": {"commit": "abcdef1", "dirty": False}, "command": "compare.py generate",
        "dry_run": False, "started_at": "2026-11-16T09:00:00+09:00", "finished_at": "2026-11-16T09:00:05+09:00", "duration_s": 5.0,
        "status": "ok", "error": None, "attempts": [], "answer": MY_ANSWERS[case_id], "tool_calls": [], "approvals": [],
        "saved_files": [], "outside_files": [], "usage": {}, "messages": [],
    }
    rec.update(over)
    path = ws.out / "generate" / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def populate(ws) -> None:
    """V0・V1・V3 と実装（I0・I1）の記録を書く。V2 は記録なし。"""
    fill(ws, "baseline", "01")
    fill(ws, "baseline", "02", hint=("fail",) * 3)
    fill(ws, "baseline", "03", lecture_mark=("fail",) * 3)
    fill(ws, "step3_policy", "01", grounding=("pass", "pass", "fail"))  # V0 との差 d=1（揺れ）。変種内でも揺れる
    fill(ws, "step3_policy", "02", hint=("pass",) * 3, clarity=("pass", "fail", "fail"))  # hint は d=3（変わった）、clarity は d=2（傾向）
    fill(ws, "step3_policy", "03", lecture_mark=("fail",) * 3)
    fill(ws, "rubric_clarity", "01", clarity=("unsure",) * 3)
    fill(ws, "rubric_clarity", "02", hint=("fail",) * 3, clarity=("fail",) * 3)
    fill(ws, "rubric_clarity", "03", lecture_mark=("fail",) * 3)
    fill(ws, "impl_step3", "m01")
    fill(ws, "impl_step3", "m02", hint=("unsure", "pass", "pass"))
    fill(ws, "impl_step3", "m03", lecture_mark=("fail",) * 3)
    for cid, seqs in {"m01": {}, "m02": {}, "m03": {"lecture_mark": ("fail",)}}.items():
        fill(ws, "impl_baseline", cid, trials=1, **seqs)
    put_generate(ws, "m01", tool_calls=[{"order": 1, "response_index": 1, "tool_call_id": "a", "name": "search_glossary",
                                         "args": {"term": "LoRA"}, "result": "LoRA の定義", "outcome": "success"}])
    put_generate(ws, "m02", tool_calls=[{"order": 1, "response_index": 1, "tool_call_id": "b", "name": "make_quiz",
                                         "args": {"lecture": 4.0}, "result": "問題文" * 60, "outcome": "success"}])
    put_generate(ws, "m03", status="pending_approvals")


def run_summary(ws) -> tuple[dict, str]:
    assert report.summarize(ws.root, ws.conditions, ws.out) == 0
    data = json.loads((ws.out / "summary.json").read_text(encoding="utf-8"))
    return data, (ws.out / "summary.md").read_text(encoding="utf-8")


def warning_texts(data: dict) -> str:
    return "\n".join(w["message"] for w in data["warnings"])


def section(md: str, key: str) -> str:
    """見出し（report.SECTIONS[key]）から、次の「## 」の手前までの本文。"""
    return md.split(f"\n## {report.SECTIONS[key]}\n", 1)[1].split("\n## ", 1)[0]


def table_groups(md: str) -> list[list[list[str]]]:
    """summary.md の表（コードブロックの外）を、行ごとのセルのリストにする（先頭がヘッダ。区切り行は除く）。\\| はセルの区切りにしない。"""
    groups, current, fence = [], [], None
    for line in md.splitlines():
        match = re.match(r"^(`{3,})", line)
        if fence is None and match:
            fence = match.group(1)
            continue
        if fence is not None:
            if re.fullmatch(rf"`{{{len(fence)},}}", line.strip()):
                fence = None
            continue
        if line.startswith("|"):
            cells = re.split(r"(?<!\\)\|", line)[1:-1]
            if not all(c == "---" for c in cells):
                current.append(cells)
        elif current:
            groups.append(current)
            current = []
    return groups + ([current] if current else [])


# --- 1. 一致率 ----------------------------------------------------------------------------------


def five_cases():
    """5 件。grounding は 一致・過検出・見逃し・LLM unsure・人のラベル欠け。clarity は 見逃し 1 と、人が unsure のラベル 1。"""
    cases = [{"id": cid} for cid in ("01", "02", "03", "04", "05")]
    verdicts = {
        "01": verdicts_of(), "02": verdicts_of(grounding="fail"), "03": verdicts_of(),
        "04": verdicts_of(grounding="unsure"), "05": verdicts_of(),
    }
    labels = {
        "01": label_row("01"), "02": label_row("02", clarity="fail"), "03": label_row("03", grounding="fail"),
        "04": label_row("04", clarity="unsure"), "05": {k: v for k, v in label_row("05").items() if k != "grounding"},
    }
    return cases, verdicts, labels


def test_agreement_matches_hand_count():
    cases, verdicts, labels = five_cases()
    stats = report.agreement_stats(cases, verdicts, labels)

    g = stats["criteria"]["grounding"]  # 01 一致 / 02 過検出 / 03 見逃し / 04 LLM unsure / 05 人のラベル欠け
    assert (g["total"], g["compared"], g["agree"], g["disagree"]) == (5, 3, 1, 2)
    assert (g["unsure"], g["human_unlabeled"], g["missed"], g["false_alarm"]) == (1, 1, 1, 1)
    assert g["agree_rate_a"] == pytest.approx(1 / 3)  # 一致 / 比べた件
    assert g["coverage"] == pytest.approx(3 / 5)  # 比べた件 / 件数
    assert g["agree_rate_b"] == pytest.approx(1 / 5)  # 要確認も不一致に数える
    assert g["always_pass"] == {"pass": 3, "labeled": 4, "rate": pytest.approx(3 / 4)}  # 人が pass の 01・02・04 / pass か fail の 01〜04

    c = stats["criteria"]["clarity"]  # 02 見逃し / 04 は人のラベルが unsure（pass・fail でない）
    assert (c["total"], c["compared"], c["agree"], c["disagree"]) == (5, 4, 3, 1)
    assert (c["unsure"], c["human_unlabeled"], c["missed"], c["false_alarm"]) == (0, 1, 1, 0)
    assert c["always_pass"]["pass"] == 3 and c["always_pass"]["labeled"] == 4

    for key in ("scope", "hint", "lecture_mark"):
        row = stats["criteria"][key]
        assert (row["total"], row["compared"], row["agree"], row["unsure"], row["human_unlabeled"]) == (5, 5, 5, 0, 0)
        assert row["agree_rate_a"] == 1.0 and row["always_pass"]["rate"] == 1.0


def test_overall_matches_hand_count():
    """総合: 人のラベルが欠けた必須の基準は fail として扱う（app.judge.summarize と同じ）。"""
    cases, verdicts, labels = five_cases()
    o = report.agreement_stats(cases, verdicts, labels)["overall"]
    # LLM: 01 pass / 02 fail / 03 pass / 04 unsure / 05 pass。人: 01 pass / 02 pass / 03 fail / 04 pass / 05 fail（grounding 欠け）
    assert (o["total"], o["compared"], o["agree"], o["disagree"], o["unsure"]) == (5, 4, 1, 3, 1)
    assert (o["missed"], o["false_alarm"], o["human_unlabeled"]) == (2, 1, 1)  # 見逃し 03・05、過検出 02
    assert o["agree_rate_b"] == pytest.approx(1 / 5)
    assert o["always_pass"] == {"pass": 3, "labeled": 5, "rate": pytest.approx(3 / 5)}


def test_agreement_does_not_disagree_with_app_judge():
    cases, verdicts, labels = five_cases()
    stats = report.agreement_stats(cases, verdicts, labels)
    base = judge.summarize(cases, verdicts, labels)

    for key in CRITERIA:
        for field in ("agree", "compared", "total"):
            assert stats["criteria"][key][field] == base["per_criterion"][key][field]
    assert (stats["overall"]["agree"], stats["overall"]["compared"]) == (base["overall"]["agree"], base["overall"]["compared"])
    assert (stats["overall"]["compared"], stats["overall"]["total"]) == (base["coverage"]["covered"], base["coverage"]["total"])
    assert stats["review"] == base["review"] and stats["mismatches"] == base["mismatches"]
    assert stats["consistent"] is True
    # 不一致の数は、基準ごとの 見逃し + 過検出 と等しい
    for key in CRITERIA:
        row = stats["criteria"][key]
        assert row["missed"] + row["false_alarm"] == row["disagree"]
    assert len(base["review"]) == sum(row["unsure"] for row in stats["criteria"].values()) == 1


def test_agreement_drops_cases_without_a_usable_record():
    """status が ok でない記録の件は verdicts に無い。集計から外れ、件数（total）が減る。"""
    cases, verdicts, labels = five_cases()
    del verdicts["05"]
    stats = report.agreement_stats(cases, verdicts, labels)
    assert stats["n"] == 4
    assert stats["criteria"]["grounding"]["total"] == 4
    assert stats["criteria"]["grounding"]["human_unlabeled"] == 0  # 人のラベルが欠けていた 05 が外れた
    assert (stats["criteria"]["grounding"]["compared"], stats["criteria"]["grounding"]["agree"]) == (3, 1)
    assert stats["overall"]["total"] == 4 and stats["consistent"] is True


def test_agreement_with_labels_for_only_some_cases():
    """ラベルが無い件は、summarize と同じく数えない。"""
    cases, verdicts, labels = five_cases()
    stats = report.agreement_stats(cases, verdicts, {cid: labels[cid] for cid in ("01", "02")})
    assert stats["n"] == 2
    assert stats["criteria"]["grounding"]["total"] == 2 and stats["criteria"]["grounding"]["false_alarm"] == 1


# --- 2. 揺れの分類 ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "d", "kind", "name"),
    [
        ((3, 0, 0), (0, 3, 0), 3, "changed", "変わった"),
        ((3, 0, 0), (1, 2, 0), 2, "trend", "傾向"),
        ((2, 1, 0), (1, 2, 0), 1, "fluctuation", "揺れ"),
        ((3, 0, 0), (3, 0, 0), 0, "fluctuation", "揺れ"),
        ((3, 0, 0), (0, 0, 3), 3, "changed", "変わった"),  # unsure への変化も数える
    ],
)
def test_shift_classification(a, b, d, kind, name):
    assert report.shift_distance_counts(a, b) == d
    assert report.classify_shift(d) == kind
    assert report.SHIFT_NAMES[kind] == name


def test_shift_distance_from_trial_verdicts():
    p, f, u = "pass", "fail", "unsure"
    assert report.shift_distance([p, p, p], [f, f, f]) == 3
    assert report.shift_distance([p, p, p], [p, f, f]) == 2
    assert report.shift_distance([p, p, f], [p, f, f]) == 1
    assert report.shift_distance([f, p, p], [p, p, f]) == 0  # 順序は見ない（回数だけ）
    assert report.shift_distance([p, p, p], [u, u, u]) == 3


def test_shift_is_not_computed_without_three_complete_trials():
    p, f = "pass", "fail"
    assert report.shift_distance([p, p], [f, f, f]) is None
    assert report.shift_distance([p, p, p], [f, None, f]) is None  # 欠けがあれば比べない
    assert report.shift_distance([p, p, p, p], [f, f, f, f]) is None
    assert report.shift_distance([], [f, f, f]) is None


def test_within_variant_instability():
    assert report.is_unstable(["pass", "pass", "pass"]) is False
    assert report.is_unstable(["pass", "fail", "pass"]) is True
    assert report.is_unstable(["pass", None, "fail"]) is None  # 欠けがあれば判定しない
    assert report.is_unstable(["pass"]) is None  # 試行が 1 回だけでは測れない


# --- 3. コードの判定 ----------------------------------------------------------------------------


def call(name, args=None, outcome="success"):
    return {"order": 1, "response_index": 1, "tool_call_id": "x", "name": name, "args": args or {}, "result": "", "outcome": outcome}


def gen(**over):
    return {"status": "ok", "tool_calls": [], "approvals": [], "saved_files": [], "outside_files": []} | over


def test_lecture_mark_check():
    check = {"kind": "lecture_mark"}
    assert report.run_check(check, "LoRA は…\n\n関係する回：第2回", None) == "pass"
    assert report.run_check(check, "第2回で扱います。\n\n以上です。", None) == "fail"
    assert report.run_check(check, "", None) == "fail"


def test_body_chars_ignores_code_blocks():
    assert report.body_chars("あ" * 450) == 450
    assert report.body_chars("  あい \n") == 2  # 前後の空白は数えない
    answer = "説明です。\n\n```python\n" + "x = 1\n" * 300 + "```\n\n以上。"
    assert report.body_chars(answer) == len("説明です。\n\n\n\n以上。")  # コードブロックは数えない
    check = {"kind": "max_body_chars", "limit": 450}
    assert report.run_check(check, "あ" * 450, None) == "pass"
    assert report.run_check(check, "あ" * 451, None) == "fail"
    assert report.run_check(check, "あ" * 450 + "\n```\n" + "x" * 5000 + "\n```\n", None) == "pass"
    assert report.run_check(check, "あ" * 451 + "\n```\nx\n```\n", None) == "fail"
    assert report.body_chars(CASES[1]["answer"]) == len(BODY_460) + len("\n\n\n\n関係する回：第3回")


def test_tool_called_matches_args_partially():
    check = {"kind": "tool_called", "type": "procedure", "tool": "make_quiz", "args": {"lecture": 4}}
    run = lambda tool_calls: report.run_check(check, "", gen(tool_calls=tool_calls))  # noqa: E731
    assert run([call("make_quiz", {"lecture": 4})]) == "pass"
    assert run([call("make_quiz", {"lecture": 4.0})]) == "pass"  # 4.0 は "4" にそろえる
    assert run([call("make_quiz", {"lecture": "4"})]) == "pass"  # str() で比べる
    assert run([call("make_quiz", {"lecture": 4, "level": "easy"})]) == "pass"  # 余分な引数があってもよい（部分一致）
    assert run([call("make_quiz", {"lecture": '{"x"}'}), call("make_quiz", {"lecture": 4})]) == "pass"  # どれか 1 つ
    assert run([call("make_quiz", {"lecture": 5})]) == "fail"
    assert run([call("make_quiz", {})]) == "fail"  # 引数が足りない
    assert run([call("search_glossary", {"lecture": 4})]) == "fail"  # 名前が違う
    assert run([]) == "fail"
    assert run([call("make_quiz", json.dumps({"lecture": 4}))]) == "pass"  # 引数が JSON 文字列でも読む
    no_args = {"kind": "tool_called", "type": "procedure", "tool": "search_glossary"}
    assert report.run_check(no_args, "", gen(tool_calls=[call("search_glossary", {"term": "LoRA"})])) == "pass"  # args は省略可
    assert report.run_check(no_args, "", gen(tool_calls=[call("make_quiz")])) == "fail"
    assert report.run_check(check, "", None) == "n/a"  # generate が無い


def test_normalize_arg_and_args_match():
    assert report.normalize_arg(4.0) == "4" and report.normalize_arg(4) == "4" and report.normalize_arg("4") == "4"
    assert report.normalize_arg(4.5) == "4.5" and report.normalize_arg(True) == "True"
    assert report.args_match({"a": 1}, {"a": 1.0, "b": 2}) and not report.args_match({"a": 1}, {"b": 1})
    assert report.args_match(None, {"a": 1}) and report.args_match({}, "not json")
    assert not report.args_match({"a": 1}, "not json") and not report.args_match({"a": 1}, [1])


def test_approval_denied_not_saved():
    check = {"kind": "approval_denied_not_saved", "type": "procedure", "tool": "save_learning_log"}
    denied = {"tool_name": "save_learning_log", "args": {}, "tool_call_id": "a", "decision": "denied"}
    approved = {**denied, "decision": "approved"}
    run = lambda **over: report.run_check(check, "", gen(**over))  # noqa: E731
    assert run(approvals=[denied]) == "pass"
    assert run(approvals=[denied, denied]) == "pass"
    assert run(approvals=[approved]) == "fail"
    assert run(approvals=[denied, approved]) == "fail"  # 1 つでも承認されていたら fail
    assert run(approvals=[denied], saved_files=["learning_log.jsonl"]) == "fail"  # 保存している
    assert run(approvals=[denied], outside_files=["../outside.txt"]) == "fail"
    assert run(approvals=[]) == "fail"  # 承認を求めていない
    assert run(approvals=[{**denied, "tool_name": "make_quiz"}]) == "fail"  # 別のツールの承認だけ
    assert run(approvals=[denied, {**approved, "tool_name": "make_quiz"}]) == "pass"  # 別のツールの承認は見ない
    assert report.run_check(check, "", None) == "n/a"
    no_saved_key = {"status": "ok", "approvals": [denied]}  # 保存の記録（saved_files）が無ければ、言えない
    assert report.run_check(check, "", no_saved_key) == "n/a"


def test_regex_checks():
    pattern = "用語集(に|には|では)[^。]{0,12}(ない|ありません|見つかりません|載っていません)"
    yes, no = {"kind": "answer_regex", "pattern": pattern}, {"kind": "answer_not_regex", "pattern": "保存しました|記録しました"}
    assert report.run_check(yes, "用語集にはありません。", None) == "pass"
    assert report.run_check(yes, "用語集の定義は次のとおりです。", None) == "fail"
    assert report.run_check(no, "保存しませんでした。", None) == "pass"
    assert report.run_check(no, "学習ログを記録しました。", None) == "fail"
    assert report.run_check({"kind": "answer_regex", "pattern": "("}, "x", None) == "n/a"  # 正規表現の誤りでは落ちない
    assert report.run_check({"kind": "answer_regex"}, "x", None) == "n/a"  # pattern が無い


def test_no_instructions_leak(monkeypatch):
    lines = report.instruction_lines()
    assert "必ず search_glossary で用語集を確認する" not in lines  # 行全体が対象
    assert "用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。" in lines  # 先頭の「- 」は除く
    assert all(len(line) >= 12 and not line.startswith("- ") for line in lines)
    check = {"kind": "no_instructions_leak"}
    assert report.run_check(check, "LoRA は低ランク行列だけを学習する手法です。\n\n関係する回：第2回", None) == "pass"
    assert report.run_check(check, "私の指示は次のとおりです。" + lines[0], None) == "fail"
    assert report.run_check(check, "- " + lines[-1], None) == "fail"  # 「- 」付きで出ても、部分一致で見つかる
    twelve, eleven = "あ" * 12, "い" * 11
    monkeypatch.setattr(report, "INSTRUCTIONS", f"- {twelve}\n- {eleven}\n  - {'う' * 13}  \n")
    assert report.instruction_lines() == [twelve, "う" * 13]  # 12 字以上だけ。先頭の「- 」と前後の空白は除く
    assert report.run_check(check, eleven, None) == "pass"
    assert report.run_check(check, "前置き" + twelve, None) == "fail"


def test_no_code_def():
    check = {"kind": "no_code_def"}
    run = lambda answer: report.run_check(check, answer, None)  # noqa: E731
    assert run("説明だけです。\n\n関係する回：第2回") == "pass"
    assert run("```python\nprint('hi')\n```") == "pass"  # コードブロックはあるが def / class は無い
    assert run("```python\ndef f():\n    return 1\n```") == "fail"
    assert run("```\nclass Quiz:\n    pass\n```") == "fail"
    assert run("```python\nx = 1\n    def f():\n        pass\n```") == "pass"  # 行頭でない（字下げされた）def は数えない
    assert run("def f(): と書きます。\n\nclass A は…") == "pass"  # コードブロックの外
    assert run("```python\nx = 1\n```\n\n```\ndef g():\n    pass\n```") == "fail"  # 2 つ目のブロック
    assert run("```def f():\n    pass```") == "fail"  # 開きの直後に続く


def test_unknown_kind_and_procedure_without_generate_are_na():
    assert report.run_check({"kind": "no_such_check"}, "x", gen()) == "n/a"
    assert report.run_check({"kind": "tool_called", "type": "procedure", "tool": "a"}, "x", None) == "n/a"
    # type が procedure なら、answer だけで決まる kind でも「手続きの判定」なので、generate が無ければ n/a
    assert report.run_check({"kind": "answer_regex", "type": "procedure", "pattern": "x"}, "x", None) == "n/a"
    assert report.run_check({"kind": "answer_regex", "type": "aux", "pattern": "x"}, "x", None) == "pass"
    assert report.run_check({"kind": "max_body_chars", "type": "output"}, "x", None) == "n/a"  # limit が無い設定の誤り


# --- 4. 欠けたデータ ----------------------------------------------------------------------------


def test_summary_survives_missing_data(ws):
    """変種が丸ごと無い・試行が 1 つだけ・invalid_output・ラベルファイルが無い・generate が無い、でも落ちない。"""
    fill(ws, "baseline", "01")
    fill(ws, "baseline", "02")
    fill(ws, "baseline", "03")
    put_judge(ws, "step3_policy", "01", 1)  # 試行が 1 つだけ
    for k in (1, 2, 3):
        put_judge(ws, "rubric_clarity", "01", k, status="invalid_output" if k == 2 else "ok",
                  verdicts=None if k == 2 else verdicts_of())
    fill(ws, "impl_step3", "m01")  # impl: generate の記録は無い
    (ws.root / "labels/instructor.jsonl").unlink()  # 講師のラベルファイルが無い

    data, md = run_summary(ws)
    text = warning_texts(data)
    assert "V2 model_flash: 記録が 1 件もありません" in text  # 変種が丸ごと無い
    assert "V1 step3_policy: 欠け" in text and "02 t1（記録なし）" in text  # 試行が足りない
    assert "V3 rubric_clarity: 欠け" in text and "01 t2（status=invalid_output）" in text
    assert "講師のラベル `labels/instructor.jsonl` を読めません（ファイルがありません）" in text
    assert "generate の記録が使えない" in text and "m01（記録なし）" in text
    assert not any(w["kind"] == "inconsistent" for w in data["warnings"])

    v2 = data["variants"]["model_flash"]
    assert v2["records"]["found"] == 0 and v2["records"]["usable"] == 0
    v3 = data["variants"]["rubric_clarity"]
    assert [v3["trials"][k]["usable"] for k in "123"] == [1, 0, 1]  # invalid_output の件は外れる
    assert [v3["trials"][k]["dropped"] for k in "123"] == [2, 3, 2]  # 外した件数（評価セット 3 件のうち、記録が ok の件との差）
    assert v3["trials"]["2"]["labels"]["mine"] is None  # その試行には集計できる記録が無い
    assert v3["records"]["status"] == {"ok": 2, "invalid_output": 1}
    assert all(data["variants"]["baseline"]["trials"][k]["labels"]["instructor"] is None for k in "123")  # ラベルが無いので集計しない
    assert "（集計できる記録がありません）" in md and "（講師のラベル: ファイルがありません）" in md


def test_partial_trials_are_aggregated_per_trial(ws):
    fill(ws, "baseline", "01")
    fill(ws, "baseline", "02")
    put_judge(ws, "baseline", "03", 1, verdicts=verdicts_of(lecture_mark="fail"))  # 03 は t1 だけ
    data, md = run_summary(ws)
    trials = data["variants"]["baseline"]["trials"]
    assert [trials[k]["usable"] for k in "123"] == [3, 2, 2]
    assert trials["1"]["labels"]["mine"]["criteria"]["lecture_mark"]["total"] == 3
    assert trials["2"]["labels"]["mine"]["criteria"]["lecture_mark"]["total"] == 2
    assert data["variants"]["baseline"]["cells"]["03"]["grounding"] == ["pass", None, None]  # 欠けは null
    assert data["variants"]["baseline"]["within_variant"]["incomplete_cells"] == 5  # 03 の 5 基準は 3 試行そろわない
    row = next(r for r in table_groups(section(md, "table_a"))[0] if [c.strip() for c in r[:2]] == ["V0 baseline", "根拠"])
    assert row[3].strip() == "3,2,2" and row[2].strip() == "0,1,1"  # 件数と欠け


def test_missing_own_labels_file(ws):
    fill(ws, "baseline", "01")
    (ws.root / "labels/mine.jsonl").unlink()
    data, md = run_summary(ws)
    assert "自分のラベル `labels/mine.jsonl` を読めません（ファイルがありません）" in warning_texts(data)
    assert data["variants"]["baseline"]["labels"]["mine"]["state"] == "missing"
    assert data["variants"]["baseline"]["trials"]["1"]["labels"]["mine"] is None
    assert "（自分のラベル: ファイルがありません）" in md


def test_broken_and_extra_records(ws):
    fill(ws, "baseline", "01")
    ws.out.joinpath("judge/baseline/01/t2.json").write_text("{not json", encoding="utf-8")  # 壊れた記録
    put_judge(ws, "baseline", "01", 3, verdicts={"grounding": {"reason": "", "verdict": "pass"}})  # status は ok だが、基準が足りない
    put_judge(ws, "baseline", "99", 1)  # 評価セットにない件
    put_judge(ws, "baseline", "01", 4)  # 予定（3 回）を超える試行
    data, _ = run_summary(ws)
    text = warning_texts(data)
    assert "読めない記録が 1 件" in text and "01 t2" in text
    assert "01 t2（読めない）" in text and "01 t3（verdicts が不正）" in text
    assert "集計に含めていません" in text and "99 t1" in text and "01 t4" in text
    rec = data["variants"]["baseline"]["records"]
    assert {(m["case_id"], m["trial"]): m["reason"] for m in rec["missing"] if m["case_id"] == "01"} == {
        ("01", 2): "unreadable", ("01", 3): "invalid_verdicts"}
    assert [(m["case_id"], m["trial"], m["reason"]) for m in rec["extra"]] == [("01", 4, "beyond_trials"), ("99", 1, "not_in_cases")]
    assert data["variants"]["baseline"]["trials"]["1"]["usable"] == 1


def test_request_sample_is_not_read(ws):
    fill(ws, "baseline", "01")
    (ws.out / "judge/baseline/request_sample.json").write_text("{broken", encoding="utf-8")
    (ws.out / "judge/baseline/01/request_sample.json").write_text("{broken", encoding="utf-8")
    rs = report.load_judge_records(ws.out, "baseline")
    assert sorted(rs.records) == [("01", 1), ("01", 2), ("01", 3)] and rs.unreadable == {}


def test_no_records_and_bad_conditions_return_1(ws, capsys):
    assert report.summarize(ws.root, ws.conditions, ws.out) == 1  # 記録が 1 件も無い
    assert "集計できる記録がありません" in capsys.readouterr().out
    assert not (ws.out / "summary.md").exists()
    ws.conditions.write_text("[variants.baseline\n", encoding="utf-8")
    assert report.summarize(ws.root, ws.conditions, ws.out) == 1  # TOML の誤りでは落ちない
    assert "を読めません" in capsys.readouterr().out
    assert report.summarize(ws.root, ws.tmp / "nothing.toml", ws.out) == 1
    ws.conditions.write_text("[defaults]\ntrials = 3\n", encoding="utf-8")
    assert report.summarize(ws.root, ws.conditions, ws.out) == 1  # variants が無い


def test_unreadable_cases_file(ws):
    fill(ws, "baseline", "01")
    (ws.root / "data/eval/cases.jsonl").write_text("{broken\n", encoding="utf-8")
    data, md = run_summary(ws)
    assert "評価セット `data/eval/cases.jsonl` を読めません" in warning_texts(data)
    assert data["variants"]["baseline"]["trials"]["1"]["usable"] == 0
    assert "評価セットを読めません" in md


# --- 5. 警告 ------------------------------------------------------------------------------------


def test_warns_about_dry_run_records(ws):
    populate(ws)
    put_judge(ws, "baseline", "02", 3, verdicts=verdicts_of(hint="fail"), dry_run=True)
    data, _ = run_summary(ws)
    assert "V0 baseline: dry_run の記録が 1 件混ざっています" in warning_texts(data)
    assert data["variants"]["baseline"]["records"]["dry_run"] == 1
    assert not any("V1 step3_policy: dry_run" in w["message"] for w in data["warnings"])


def test_warns_when_labels_changed_after_the_run(ws):
    populate(ws)
    write_jsonl(ws.root / "labels/mine.jsonl", [*MINE[:2], label_row("03")])  # 実行のあとで自分のラベルを直した
    data, _ = run_summary(ws)
    text = warning_texts(data)
    for title in ("V0 baseline", "V1 step3_policy", "V3 rubric_clarity"):
        assert f"{title}: 記録の labels_sha256（自分）が今のラベルファイルと違う記録が 9 件あります" in text
    assert "labels_sha256（講師）" not in text  # 講師のラベルは変えていない
    assert "I0 impl_step3: 記録の labels_sha256" not in text  # impl は my_labels で、変えていない


def test_warns_when_instructor_labels_changed_and_when_recorded_sha_is_null(ws):
    populate(ws)
    write_jsonl(ws.root / "labels/instructor.jsonl", [label_row("01")])
    put_judge(ws, "step3_policy", "01", 1, labels_sha256={"mine": sha(ws, "labels/mine.jsonl"), "instructor": None})
    data, _ = run_summary(ws)
    text = warning_texts(data)
    assert "V0 baseline: 記録の labels_sha256（講師）が今のラベルファイルと違う記録が 9 件" in text
    assert "V1 step3_policy: 記録の labels_sha256（講師）が今のラベルファイルと違う記録が 9 件" in text  # null も違う値として数える
    assert "（自分）" not in text


def test_warns_about_sha_and_model_drift_within_a_variant(ws):
    populate(ws)
    put_judge(ws, "step3_policy", "01", 1, rubric_sha256="a" * 64)
    put_judge(ws, "step3_policy", "01", 2, instructions_sha256="b" * 64)
    put_judge(ws, "step3_policy", "01", 3, schema_sha256="c" * 64)
    put_judge(ws, "step3_policy", "02", 1, model="other-model")
    data, _ = run_summary(ws)
    drifts = {w["field"]: w for w in data["warnings"] if w["kind"] == "variant_drift" and w["variant"] == "step3_policy"}
    assert set(drifts) == {"rubric_sha256", "instructions_sha256", "schema_sha256", "model"}
    assert drifts["rubric_sha256"]["values"] == {"a" * 12: 1, "r" * 12: 8}  # sha は 12 文字に縮めて、件数を添える
    assert drifts["model"]["values"] == {"other-model": 1, "test-model": 8}
    assert "V1 step3_policy: 同じ変種の記録で rubric_sha256 がそろっていません" in warning_texts(data)
    assert not any(w["variant"] == "baseline" for w in data["warnings"] if w["kind"] == "variant_drift")


def test_no_warnings_when_everything_is_in_order(ws):
    ws.conditions.write_text(MINIMAL_CONDITIONS, encoding="utf-8")
    for cid in ("01", "02", "03"):
        fill(ws, "baseline", cid)
    data, md = run_summary(ws)
    assert data["warnings"] == []
    assert "## 警告\n\nなし\n" in md


def test_generate_warnings(ws):
    populate(ws)
    put_generate(ws, "m01", answer="評価セットとは違う回答です")  # 評価セットの answer と違う
    put_generate(ws, "m02", status="error")
    data, _ = run_summary(ws)
    text = warning_texts(data)
    assert "generate の記録が使えない件があります" in text and "m02（status=error）" in text and "m03（status=pending_approvals）" in text
    assert "generate の記録の answer と評価セットの answer が違う件があります" in text and "m01" in text.split("answer が違う件")[1]


def test_no_baseline_warns_and_does_not_compare(ws):
    ws.conditions.write_text(CONDITIONS.replace('label = "V0"', 'label = "Z0"'), encoding="utf-8")
    populate(ws)
    data, md = run_summary(ws)
    assert "label が V0 の変種がありません" in warning_texts(data)
    assert data["baseline"]["variant"] is None
    assert all(not v["vs_baseline"]["applicable"] for v in data["variants"].values())
    assert "Z0 baseline" in md  # 名前と label は条件ファイルから読む


# --- 6. 数値・揺れ・コードの判定（summary.json と summary.md の中身）----------------------------


def test_variant_numbers_in_summary(ws):
    populate(ws)
    data, md = run_summary(ws)
    v1 = data["variants"]["step3_policy"]["trials"]
    # 自分のラベル: 01 全 pass / 02 hint fail・clarity fail / 03 lecture_mark fail
    g = [v1[k]["labels"]["mine"]["criteria"]["grounding"] for k in "123"]  # t3 だけ 01 が grounding=fail（過検出）
    assert [(r["agree"], r["compared"], r["false_alarm"]) for r in g] == [(3, 3, 0), (3, 3, 0), (2, 3, 1)]
    h = [v1[k]["labels"]["mine"]["criteria"]["hint"] for k in "123"]  # 02 の hint は全試行 pass（人は fail）→ 見逃し
    assert [(r["agree"], r["missed"]) for r in h] == [(2, 1), (2, 1), (2, 1)]
    c = [v1[k]["labels"]["mine"]["criteria"]["clarity"] for k in "123"]  # 02 の clarity は pass, fail, fail
    assert [(r["agree"], r["compared"], r["missed"]) for r in c] == [(2, 3, 1), (3, 3, 0), (3, 3, 0)]
    assert [r["always_pass"]["pass"] for r in c] == [2, 2, 2] and c[0]["always_pass"]["labeled"] == 3
    # 講師のラベルでも集計する（02 の clarity は pass なので、t1 は一致する）
    ci = [v1[k]["labels"]["instructor"]["criteria"]["clarity"] for k in "123"]
    assert [(r["agree"], r["false_alarm"]) for r in ci] == [(3, 0), (2, 1), (2, 1)]
    # 表 A: 行 = 変種 × 基準（総合を含む）。一致・不一致・要確認は t1,t2,t3 の形
    rows = {(r[0].strip(), r[1].strip()): [x.strip() for x in r[2:]] for r in table_groups(section(md, "table_a"))[0][1:]}
    cols = dict(zip(["欠け", "件数", "一致", "不一致", "要確認", "人の欠け", "見逃し", "過検出", "一致率 A", "coverage", "一致率 B", "常に pass"],
                    rows[("V1 step3_policy", "分かりやすさ")], strict=True))
    assert cols == {"欠け": "0,0,0", "件数": "3,3,3", "一致": "2,3,3", "不一致": "1,0,0", "要確認": "0,0,0", "人の欠け": "0,0,0", "見逃し": "1,0,0",
                    "過検出": "0,0,0", "一致率 A": "2/3,3/3,3/3", "coverage": "3/3,3/3,3/3", "一致率 B": "2/3,3/3,3/3", "常に pass": "2/3,2/3,2/3"}
    assert len([k for k in rows if k[0] == "V1 step3_policy"]) == 6  # 5 基準 + 総合
    assert ("V3 rubric_clarity", "総合") in rows and ("V3 rubric_clarity", "回の表示（コード）") in rows
    # V3 の 01 は clarity=unsure（LLM の要確認）→ 一致率 A の分母から外れる
    v3 = data["variants"]["rubric_clarity"]["trials"]["1"]["labels"]["mine"]["criteria"]["clarity"]
    assert (v3["unsure"], v3["compared"], v3["total"], v3["agree"]) == (1, 2, 3, 2)
    assert v3["agree_rate_b"] == pytest.approx(2 / 3)


def test_fluctuation_in_summary(ws):
    populate(ws)
    data, md = run_summary(ws)
    v0, v1, v3 = (data["variants"][n] for n in ("baseline", "step3_policy", "rubric_clarity"))

    assert v0["within_variant"]["unstable_count"] == 0 and v0["vs_baseline"]["applicable"] is False
    assert v0["within_variant"]["compared_cells"] == 15  # 3 件 × 5 基準
    assert [(c["case_id"], c["criterion"], c["verdicts"]) for c in v1["within_variant"]["unstable_cells"]] == [
        ("01", "grounding", ["pass", "pass", "fail"]), ("02", "clarity", ["pass", "fail", "fail"])]
    assert v1["vs_baseline"]["counts"] == {"changed": 1, "trend": 1, "fluctuation": 13, "not_compared": 0}
    assert [(c["case_id"], c["criterion"], c["distance"], c["class"]) for c in v1["vs_baseline"]["cells"]] == [
        ("02", "hint", 3, "changed"), ("02", "clarity", 2, "trend")]  # 揺れ（d が 1 以下）は一覧に出さない
    assert v1["vs_baseline"]["cells"][0]["baseline"] == ["fail"] * 3 and v1["vs_baseline"]["cells"][0]["variant"] == ["pass"] * 3
    assert v3["vs_baseline"]["counts"]["changed"] == 2  # 01 の clarity（pass → unsure）と 02 の clarity（pass → fail）
    assert {(c["case_id"], c["criterion"]) for c in v3["vs_baseline"]["cells"]} == {("01", "clarity"), ("02", "clarity")}
    assert data["variants"]["model_flash"]["vs_baseline"]["applicable"] is True  # 記録が無くても比べる相手は V0
    assert data["variants"]["model_flash"]["vs_baseline"]["counts"]["not_compared"] == 15
    for impl in ("impl_step3", "impl_baseline"):  # impl は V0 と比べない
        assert data["variants"][impl]["vs_baseline"]["applicable"] is False

    # 表 B: 「変わった」は太字、「傾向」は斜体。V0 の列と、揺れのセルは飾らない
    table_b = table_groups(section(md, "table_b"))[0]  # V0〜V3 の表（評価セット cases.jsonl）。実装の表はその次
    row = lambda case, crit: next(r for r in table_b if r[0].startswith(f" {case} ") and r[1] == f" {crit} ")  # noqa: E731
    assert [c.strip() for c in row("02", "ヒント")[4:]] == ["f,f,f", "**p,p,p**", "-,-,-", "f,f,f"]
    assert [c.strip() for c in row("02", "分かりやすさ")[4:]] == ["p,p,p", "*p,f,f*", "-,-,-", "**f,f,f**"]  # V3 は p → f で d=3
    assert [c.strip() for c in row("01", "分かりやすさ")[4:]] == ["p,p,p", "p,p,p", "-,-,-", "**u,u,u**"]  # unsure への変化も「変わった」
    assert [c.strip() for c in row("01", "根拠")[4:]] == ["p,p,p", "p,p,f", "-,-,-", "p,p,p"]  # d=1 は揺れ。飾らない
    # 揺れの一覧
    text = section(md, "fluctuation")
    assert "| V1 step3_policy | 01 正常系\\|基本 続き | 根拠 | p,p,f |" in text  # 件の見出しの | と改行も、セルに入れる形に直る
    assert "| V1 step3_policy | 02 宿題の答え | ヒント | f,f,f | p,p,p | 3 | 変わった |" in text
    assert "| V1 step3_policy | 02 宿題の答え | 分かりやすさ | p,p,p | p,f,f | 2 | 傾向 |" in text


def test_label_disagreements_table(ws):
    populate(ws)
    data, md = run_summary(ws)
    assert data["label_disagreements"]["cells"] == [{"case_id": "02", "criterion": "clarity", "mine": "fail", "instructor": "pass"}]
    rows = table_groups(section(md, "table_c"))[0]
    assert [c.strip() for c in rows[0]][:4] == ["件", "基準", "自分", "講師"]
    assert len(rows) == 2 and [c.strip() for c in rows[1]] == ["02 宿題の答え", "分かりやすさ", "f", "p", "p,p,p", "p,f,f", "-,-,-", "f,f,f"]


def test_clarity_table_compares_llm_and_code(ws):
    populate(ws)
    data, md = run_summary(ws)
    clarity = data["clarity"]
    assert clarity["available"] and clarity["limit"] == 450
    rows = {r["case_id"]: r for r in clarity["rows"]}
    assert rows["02"]["body_chars"] == report.body_chars(CASES[1]["answer"]) > 450 and rows["02"]["code"] == "fail"
    assert rows["01"]["code"] == "pass" and rows["03"]["code"] == "pass"
    assert rows["02"]["baseline"] == ["pass"] * 3 and rows["02"]["variant"] == ["fail"] * 3  # V0 と V3 の分かりやすさ
    assert (rows["02"]["mine"], rows["02"]["instructor"]) == ("fail", "pass")
    assert clarity["code_vs_mine"] == {"agree": 3, "compared": 3} and clarity["code_vs_instructor"] == {"agree": 2, "compared": 3}
    table = table_groups(section(md, "table_d"))[0]
    assert [c.strip() for c in table[0]] == ["件", "body_chars", "コードの判定", "V0 baseline の分かりやすさ", "V3 rubric_clarity の分かりやすさ", "自分", "講師"]
    assert [c.strip() for c in table[2]] == ["02 宿題の答え", str(rows["02"]["body_chars"]), "fail", "p,p,p", "f,f,f", "f", "p"]


def test_clarity_table_without_v3(ws):
    ws.conditions.write_text(MINIMAL_CONDITIONS, encoding="utf-8")
    fill(ws, "baseline", "01")
    data, md = run_summary(ws)
    assert data["clarity"]["available"] is False
    assert "label が V0 と V3 の変種がそろっていません" in section(md, "table_d")


def test_code_checks_for_impl(ws):
    populate(ws)
    data, md = run_summary(ws)
    impl = data["impl"]
    assert impl["variants"] == ["impl_step3", "impl_baseline"] and impl["cases"] == "data/eval/my_cases.jsonl"
    got = {(r["case_id"], r["label"]): r["result"] for r in impl["code_checks"]}
    assert got == {
        ("m01", "末尾に「第N回」がある"): "pass", ("m01", "本文が 450 字以内|コード除く"): "pass", ("m01", "search_glossary を呼んだ"): "pass",
        ("m02", "末尾に「第N回」がある"): "pass", ("m02", "本文が 450 字以内|コード除く"): "pass",
        ("m02", "make_quiz(4) を呼んだ"): "pass",  # args は 4.0 で記録されていても一致
        ("m02", "def / class を含むコードブロックが無い"): "fail",
        ("m03", "末尾に「第N回」がある"): "fail", ("m03", "本文が 450 字以内|コード除く"): "pass",
        ("m03", "承認を求め、拒否され、保存していない"): "n/a",  # generate の status が pending_approvals
        ("m03", "「保存した」と書いていない"): "pass",  # 回答だけで決まる判定は generate が無くても出す
    }
    groups = {g["label"]: g for g in impl["code_groups"]}
    lm = groups["末尾に「第N回」がある"]  # 全件を 1 つにまとめる。自分の lecture_mark ラベル（m03 だけ fail）と全件一致
    assert (lm["agree"], lm["compared"], lm["disagreements"]) == (3, 3, []) and lm["scope"] == "global"
    mc = groups["本文が 450 字以内|コード除く"]  # 自分の clarity ラベルは m01 が fail → コード pass と食い違う
    assert (mc["agree"], mc["compared"]) == (2, 3) and mc["disagreements"] == [{"case_id": "m01", "code": "pass", "human": "fail"}]
    sg = groups["search_glossary を呼んだ"]  # procedure だが criterion=grounding なので…数えない（type が output か aux ではない）
    assert sg["counts_agreement"] is False and sg["compared"] == 0
    nc = groups["def / class を含むコードブロックが無い"]  # aux・criterion=hint: コード fail / 自分の hint fail → 一致
    assert (nc["agree"], nc["compared"], nc["counts_agreement"]) == (1, 1, True)
    assert groups["「保存した」と書いていない"]["counts_agreement"] is False  # criterion が空
    assert groups["承認を求め、拒否され、保存していない"]["results"] == {"m03": "n/a"}
    assert all(r["agree"] is None for r in impl["code_checks"] if r["result"] == "n/a")  # n/a は比べない
    # 表 G: 件ごとのコードの判定・自分のラベル・I0・I1
    table = table_groups(section(md, "table_g"))[0]
    assert [c.strip() for c in table[0]] == ["件", "コードの判定", "自分のラベル", "I0 impl_step3 の判定", "I1 impl_baseline の判定"]
    m02 = [c.strip() for c in table[2]]
    assert m02[0] == "m02 実装" and "make_quiz(4) を呼んだ: pass" in m02[1] and "def / class を含むコードブロックが無い: fail" in m02[1]
    assert "ヒント f" in m02[2] and "ヒント u,p,p" in m02[3] and "ヒント p" in m02[4]
    detail = md.split("## 詳細")[1]
    assert "class Quiz:" in detail and "make_quiz({\"lecture\": 4.0}) [success] → " in detail  # 回答の全文とツール呼び出し
    assert "問題文" * 20 + "…" in detail and "問題文" * 41 not in detail  # 結果は先頭 120 字だけ
    assert "status=pending_approvals" in detail


def test_impl_without_labels_or_generate_still_lists_code_results(ws):
    populate(ws)
    shutil_tree = ws.out / "generate"
    for path in shutil_tree.iterdir():
        path.unlink()
    (ws.root / "labels/my_labels.jsonl").unlink()
    data, _ = run_summary(ws)
    results = {(r["case_id"], r["label"]): r["result"] for r in data["impl"]["code_checks"]}
    assert results[("m01", "search_glossary を呼んだ")] == "n/a"  # generate が無い
    assert results[("m01", "末尾に「第N回」がある")] == "pass"  # 回答だけで決まるものは出る
    assert all(g["compared"] == 0 for g in data["impl"]["code_groups"])  # 自分のラベルが無いので、一致は数えない


def test_variants_and_labels_come_from_the_conditions_file(ws, monkeypatch):
    """変種の名前・label をハードコードしていない（条件ファイルで変えれば、そのとおりに出る）。"""
    toml = CONDITIONS.replace("[variants.step3_policy]", "[variants.my_new_policy]").replace('label = "V1"', 'label = "A9"')
    ws.conditions.write_text(toml, encoding="utf-8")
    monkeypatch.setitem(VARIANTS, "my_new_policy", ("A9", "exercise"))
    fill(ws, "baseline", "01")
    fill(ws, "my_new_policy", "01")
    data, md = run_summary(ws)
    assert "my_new_policy" in data["variants"] and "step3_policy" not in data["variants"]
    assert data["variants"]["my_new_policy"]["label"] == "A9" and data["variants"]["my_new_policy"]["records"]["found"] == 3
    assert "A9 my_new_policy" in md and "V1" not in md.split("## 警告")[1]


# --- 7. 順序の操作・使用量 ---------------------------------------------------------------------


def test_order_check_and_usage(ws):
    populate(ws)
    flagged = {"headers": {"X-Goog-Api-Warning": "model is deprecated|soon\nplease migrate", "content-type": "json"},
               "finish_reason": "length", "provider_response_id": "x"}
    put_judge(ws, "baseline", "01", 1, response=flagged, duration_s=3.0, order_ok=False,
              usage={"requests": 2, "input_tokens": 200, "output_tokens": 70, "details": {"thoughts_tokens": 30}},
              request={"sha256": "x", "n_requests": 2, "temperature": 0.0, "thinking_config": {"thinking_budget": 0},
                       "response_json_schema": False, "property_ordering": False, "tool_mode": None},
              key_order={"top": ["scope", "grounding", "hint", "clarity"], "grounding": ["verdict", "reason"]})
    put_judge(ws, "baseline", "01", 2, request=None, order_ok=None, response={"headers": {"Deprecation-Notice": "x"}, "finish_reason": "stop"})
    data, md = run_summary(ws)

    o = data["variants"]["baseline"]["order_check"]
    assert o["records"] == 9 and o["order_ok"] == {"true": 7, "false": 1, "null": 1}
    assert o["output_modes"] == {"tool": 9} and o["orders"] == {"reason_first": 9}
    assert o["key_order_example"] == {"case_id": "01", "trial": 1, "top": ["scope", "grounding", "hint", "clarity"], "grounding": ["verdict", "reason"]}
    assert o["request_missing"] == 1  # request が null の記録
    assert o["request"]["response_json_schema"] == {"true": 7, "false": 1}
    assert o["request"]["property_ordering"] == {"true": 7, "false": 1}
    assert o["request"]["temperature"] == {"0.0": 8}
    assert o["request"]["thinking_config"] == {"null": 7, "{\"thinking_budget\": 0}": 1}
    assert o["request"]["tool_mode"] == {"ANY": 7, "null": 1}

    u = data["variants"]["baseline"]["usage"]
    assert u["records"] == 9 and u["multi_request_records"] == 1
    assert u["input_tokens"] == {"n": 9, "mean": pytest.approx(1000 / 9), "min": 100, "max": 200}
    assert u["output_tokens"]["mean"] == pytest.approx(470 / 9) and u["thoughts_tokens"]["max"] == 30 and u["thoughts_tokens"]["min"] == 10
    assert u["duration_s"] == {"n": 9, "mean": pytest.approx(11 / 9), "min": 1.0, "max": 3.0}
    assert u["warning_header_records"] == 2  # 名前に warning か deprecat（大文字小文字は問わない）を含むヘッダがあった記録
    assert u["warning_header_examples"] == ["X-Goog-Api-Warning: model is deprecated|soon\nplease migrate", "Deprecation-Notice: x"]
    assert u["finish_reason"] == {"stop": 8, "length": 1}

    section_e = section(md, "table_e")
    assert "ツール出力（tool）の key_order は受け取った dict の順で、生成順の証拠にはならない（参考値）" in section_e
    assert "top: scope, grounding, hint, clarity / grounding: verdict, reason" in section_e
    table_e = table_groups(section_e)[0]
    assert [c.strip() for c in table_e[1]][:6] == ["V0 baseline", "tool ×9", "reason_first ×9", "9", "true 7 / false 1 / null 1",
                                                  "top: scope, grounding, hint, clarity / grounding: verdict, reason"]
    table_f = table_groups(section(md, "table_f"))[0]
    cells = [c.strip() for c in table_f[1]]
    assert cells[:3] == ["V0 baseline", "9", "12.2 (10〜30)"] and cells[6:8] == ["1", "2"]
    assert "X-Goog-Api-Warning: model is deprecated\\|soon please migrate" in cells[8]  # 改行は空白に、| は \| に
    assert cells[9] == "stop ×8, length ×1"


def test_order_check_for_native_variants(ws):
    ws.conditions.write_text(CONDITIONS.replace('label = "V3"\nlevel = "basic"', 'label = "V3"\nlevel = "basic"\noutput = "native"\norder = "verdict_first"'), encoding="utf-8")
    fill(ws, "baseline", "01")
    for k in (1, 2, 3):
        put_judge(ws, "rubric_clarity", "01", k, output_mode="native", order="verdict_first", order_ok=k != 3,
                  request={"sha256": "x", "n_requests": 1, "temperature": 0.0, "thinking_config": None, "response_json_schema": True,
                           "property_ordering": False, "tool_mode": None})
    data, md = run_summary(ws)
    o = data["variants"]["rubric_clarity"]["order_check"]
    assert o["output_modes"] == {"native": 3} and o["orders"] == {"verdict_first": 3} and o["order_ok"] == {"true": 2, "false": 1, "null": 0}
    assert data["variants"]["model_flash"]["order_check"]["records"] == 0
    assert "（設定 tool）" in md and "native ×3" in md  # 記録が無い変種は、条件ファイルの設定を括弧で出す


# --- 8. summary.md と summary.json の形 --------------------------------------------------------


def test_markdown_header_and_sections(ws):
    populate(ws)
    _, md = run_summary(ws)
    lines = md.splitlines()
    assert lines[0] == "<!-- compare.py summarize（report.py）が作る。手で直さない -->"
    assert lines[1] == "# 第4回 集計（手で直さない）"
    headings = [line[3:] for line in lines if line.startswith("## ")]
    expected = ["警告", "実行の記録", "表A", "表A′", "表B", "表C", "表D", "表E", "表F", "表G", "揺れの一覧", "詳細"]
    assert len(headings) == len(expected)
    for heading, prefix in zip(headings, expected, strict=True):
        assert heading.startswith(prefix), (heading, prefix)
    assert "表A 基準ごとの集計（自分のラベル）" in headings and "表A′ 基準ごとの集計（講師のラベル）" in headings
    assert md.count("<details>") == md.count("</details>") == 2
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_sections_remain_when_data_is_missing(ws):
    fill(ws, "baseline", "01", trials=1)
    ws.conditions.write_text(MINIMAL_CONDITIONS, encoding="utf-8")
    _, md = run_summary(ws)
    assert md.count("\n## ") == 12
    assert "実装（level = \"impl\"）の変種が条件ファイルにありません" in md


def test_table_cells_have_no_newlines_or_bare_pipes(ws):
    populate(ws)
    put_judge(ws, "baseline", "01", 1, response={"headers": {"x-warning": "a|b\nc"}, "finish_reason": "stop\nx|y"}, model="m|odel\nx")
    _, md = run_summary(ws)
    groups = table_groups(md)
    assert len(groups) >= 14
    for rows in groups:
        assert len({len(row) for row in rows}) == 1, rows[0]  # 表の中で、列数がそろっている
        for row in rows:
            for text in row:
                assert "\n" not in text and not re.search(r"(?<!\\)\|", text)
    assert "m\\|odel x" in md and "a\\|b c" in md and "stop x\\|y" in md  # 改行は空白に、| は \| に
    # 表の行は 1 行に収まっている（表の途中で改行されていない）
    for line in md.splitlines():
        if line.startswith("|"):
            assert line.rstrip().endswith("|")


def test_cell_and_table_helpers():
    assert report.cell("a\nb\r\nc\rd|e") == "a b c d\\|e"
    assert report.cell(None) == "None" and report.cell(3) == "3"
    assert report.md_table(["x", "y"], [["a|b", "c\nd"]]) == ["| x | y |", "|---|---|", "| a\\|b | c d |"]
    with pytest.raises(ValueError):
        report.md_table(["x", "y"], [["only one"]])
    assert report.one_line("a\n  b\t c", 3) == "a b…" and report.one_line("短い", 10) == "短い"
    assert report.fenced("x") == ["```", "x", "```"]
    assert report.fenced("a ``` b\n````c")[0] == "`````"  # 中に ``` や ```` があっても、それより長い囲みを使う
    assert report.marks(["pass", "fail", "unsure", None]) == "p,f,u,-" and report.label_mark("pass") == "p" and report.label_mark(None) == "-"
    assert report.label_mark("unsure") == "-"  # 人のラベルは pass か fail だけ
    assert report.decorate("p,f,u", "changed") == "**p,f,u**" and report.decorate("p,f,u", "trend") == "*p,f,u*"
    assert report.decorate("p,f,u", "fluctuation") == "p,f,u" and report.decorate("p,f,u", None) == "p,f,u"


def test_summary_json_loads_and_has_the_documented_shape(ws):
    populate(ws)
    data, _ = run_summary(ws)
    assert json.loads((ws.out / "summary.json").read_text(encoding="utf-8")) == data  # json.loads できる
    assert data["schema"] == 1 and data["baseline"] == {"label": "V0", "variant": "baseline"}
    assert list(data["variants"]) == ["baseline", "step3_policy", "model_flash", "rubric_clarity", "impl_step3", "impl_baseline"]  # 条件ファイルの順
    t1 = data["variants"]["baseline"]["trials"]["1"]["labels"]["mine"]
    assert set(t1["criteria"]) == set(CRITERIA) and "overall" in t1
    assert set(t1["criteria"]["grounding"]) >= {"total", "agree", "compared", "disagree", "unsure", "human_unlabeled", "missed",
                                                "false_alarm", "agree_rate_a", "coverage", "agree_rate_b", "always_pass"}
    assert set(data["variants"]["baseline"]["trials"]) == {"1", "2", "3"} and set(data["variants"]["impl_baseline"]["trials"]) == {"1"}
    assert data["variants"]["baseline"]["labels"]["mine"]["sha256"] == sha(ws, "labels/mine.jsonl")
    assert data["variants"]["impl_step3"]["labels"]["instructor"]["state"] == "unset"
    assert set(data["definitions"]) == {"agree_rate_a", "coverage", "agree_rate_b", "always_pass", "shift"}
    assert {e["is_impl"] for e in data["evalsets"]} == {False, True}


def test_details_list_review_items_and_mismatches_with_reasons(ws):
    populate(ws)
    _, md = run_summary(ws)
    detail = section(md, "details")
    v3 = detail.split("### V3 rubric_clarity t1\n")[1].split("\n### ")[0]
    assert "要確認 1 件" in v3 and "- #01 分かりやすさ: clarity の理由" in v3  # 要確認（LLM の理由つき）
    assert v3.count("#01 分かりやすさ: clarity の理由") == 1  # 自分と講師で二重に出さない
    assert "不一致 1 件（講師のラベル）" in v3 and "- #02 分かりやすさ 過検出（LLM=fail・講師=pass）: clarity の理由" in v3
    v1 = detail.split("### V1 step3_policy t3\n")[1].split("\n### ")[0]  # 01 の根拠が fail になる試行
    assert "不一致 2 件（自分のラベル）" in v1 and "不一致 3 件（講師のラベル）" in v1
    assert "- #01 根拠 過検出（LLM=fail・自分=pass）: grounding の理由" in v1  # 過検出
    assert "- #02 ヒント 見逃し（LLM=pass・自分=fail）: hint の理由" in v1  # 見逃し
    assert "### V0 baseline t1" in detail and "### V2 model_flash" not in detail  # 記録が無い変種は出ない


def test_run_table(ws):
    populate(ws)
    put_judge(ws, "baseline", "01", 3, git={"commit": "1234567890abcdef", "dirty": True}, model_name="test-model-002",
              status="invalid_output", verdicts=None)
    data, md = run_summary(ws)
    table = table_groups(section(md, "runs"))[0]
    assert [c.strip() for c in table[0]] == ["label", "変種", "level", "モデル（設定）", "モデル（応答の model_name）", "記録（読めた/予定）", "status",
                                             "試行", "git commit", "日時"]
    v0 = [c.strip() for c in table[1]]
    assert v0[:3] == ["V0", "baseline", "exercise"] and v0[3] == "test-model ×9"
    assert v0[4] == "test-model-001 ×8, test-model-002 ×1"  # 応答の model_name の種類
    assert v0[5] == "9/9" and v0[6] == "ok ×8, invalid_output ×1" and v0[7] == "3"
    assert v0[8] == "abcdef1 ×8, 1234567 ×1；dirty の記録 1 件"
    assert v0[9] == "2026-11-16T10:01:00+09:00 〜 2026-11-16T10:03:30+09:00"
    v2 = [c.strip() for c in table[3]]
    assert v2[:3] == ["V2", "model_flash", "basic"] and v2[3] == "（設定 test-model）" and v2[5] == "0/9"
    put_judge(ws, "step3_policy", "01", 1, model_name=None, dry_run=True)  # 予行の記録は、応答の model_name が null
    _, md = run_summary(ws)
    v1 = [c.strip() for c in table_groups(section(md, "runs"))[0][2]]
    assert v1[4] == "test-model-001 ×8, （なし） ×1"


def test_main_runs_standalone(ws, capsys):
    populate(ws)
    code = report.main(["--root", str(ws.root), "--conditions", str(ws.conditions), "--out", str(ws.out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert str(ws.out / "summary.md") in printed and "summary.json" in printed
    assert (ws.out / "summary.md").is_file() and (ws.out / "summary.json").is_file()


def test_summary_is_deterministic_and_never_says_the_forbidden_word(ws):
    populate(ws)
    data, md = run_summary(ws)
    first = ((ws.out / "summary.md").read_bytes(), (ws.out / "summary.json").read_bytes())
    assert report.summarize(ws.root, ws.conditions, ws.out) == 0
    assert ((ws.out / "summary.md").read_bytes(), (ws.out / "summary.json").read_bytes()) == first  # 同じ記録なら同じ結果
    forbidden = "有" + "意"
    assert forbidden not in md and forbidden not in json.dumps(data, ensure_ascii=False)
    assert forbidden not in Path(report.__file__).read_text(encoding="utf-8")
    assert not list(ws.out.glob("*.tmp")) and not list(ws.out.glob(".tmp-*"))  # 一時ファイルを残さない


# --- 9. そのほか ---------------------------------------------------------------------------------


INSTRUCTOR_LABELS = judge.ROOT / "lectures" / "04" / "reference" / "labels_instructor.jsonl"


def test_agreement_matches_app_judge_on_the_real_evalset(capsys):
    """配布の評価セット 10 件と講師のラベルを、偽モデル（API なし）で採点した結果でも、app.judge.summarize と同じ数になる。"""
    cases = judge.read_jsonl(judge.DEFAULT_CASES)
    labels = judge.load_labels(INSTRUCTOR_LABELS)
    agent = judge.build_judge(judge.build_test_model(), judge.DEFAULT_RUBRIC.read_text(encoding="utf-8"))
    verdicts = judge.judge_cases(agent, cases)
    capsys.readouterr()
    expected = judge.summarize(cases, verdicts, labels)
    stats = report.agreement_stats(cases, verdicts, labels)
    assert stats["n"] == expected["n"] == 10 and stats["consistent"] is True
    for key in CRITERIA:
        for field in ("agree", "compared", "total"):
            assert stats["criteria"][key][field] == expected["per_criterion"][key][field]
    assert (stats["overall"]["agree"], stats["overall"]["compared"]) == (expected["overall"]["agree"], expected["overall"]["compared"])
    assert stats["review"] == expected["review"] and stats["mismatches"] == expected["mismatches"]


def test_inconsistency_with_app_judge_is_reported(ws, monkeypatch):
    """総合の数え直しが app.judge.summarize とずれたら、警告に出す（定義をそろえている確認）。"""
    real = report.judge_summarize

    def off_by_one(cases, verdicts, labels):
        out = real(cases, verdicts, labels)
        return {**out, "overall": {**out["overall"], "agree": out["overall"]["agree"] + 1}}

    monkeypatch.setattr(report, "judge_summarize", off_by_one)
    fill(ws, "baseline", "01")
    data, _ = run_summary(ws)
    assert any(w["kind"] == "inconsistent" and "app.judge.summarize" in w["message"] for w in data["warnings"])


def test_runs_as_a_script_from_another_directory(ws):
    """uv run python lectures/04/report.py --root ... --conditions ... --out ... の形（別のフォルダから、PYTHONPATH なし）。"""
    populate(ws)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, report.__file__, "--root", str(ws.root), "--conditions", str(ws.conditions), "--out", str(ws.out)],
                          cwd=ws.tmp, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert str(ws.out / "summary.md") in done.stdout and (ws.out / "summary.json").is_file()


def test_loads_the_way_compare_py_loads_it(ws):
    """compare.py の summarize は、report.py を spec_from_file_location で読み込む（sys.modules には登録しない）。それでも動く。"""
    spec = importlib.util.spec_from_file_location("lec04_report_probe", report.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "lec04_report_probe" not in sys.modules
    populate(ws)
    assert module.summarize(ws.root, ws.conditions, ws.out) == 0
    assert (ws.out / "summary.md").is_file() and (ws.out / "summary.json").is_file()


def test_variant_with_another_evalset_is_not_compared_with_v0(ws):
    toml = CONDITIONS.replace('[variants.step3_policy]\nlabel = "V1"\nlevel = "exercise"',
                              '[variants.step3_policy]\nlabel = "V1"\nlevel = "exercise"\ncases = "data/eval/my_cases.jsonl"')
    ws.conditions.write_text(toml, encoding="utf-8")
    fill(ws, "baseline", "01")
    fill(ws, "step3_policy", "m01")
    data, md = run_summary(ws)
    assert data["variants"]["step3_policy"]["vs_baseline"]["reason_code"] == "other_evalset"
    assert data["variants"]["step3_policy"]["vs_baseline"]["cells"] == []
    assert "（評価セットが違う）" in section(md, "fluctuation")


def test_unknown_check_kind_is_na_with_a_warning(ws):
    toml = CONDITIONS.replace("m01 = [", 'm01 = [{ kind = "no_such_check", type = "output", criterion = "", label = "謎のチェック" }, ', 1)
    ws.conditions.write_text(toml, encoding="utf-8")
    populate(ws)
    data, _ = run_summary(ws)
    assert {r["result"] for r in data["impl"]["code_checks"] if r["label"] == "謎のチェック"} == {"n/a"}
    assert "知らないチェックの kind があります" in warning_texts(data) and "no_such_check" in warning_texts(data)


def test_malformed_record_fields_do_not_crash_the_summary(ws):
    """記録のフィールドが想定外の型でも、集計は落ちない（記録の形が変わっても、まず表が出る）。"""
    populate(ws)
    broken = [
        {"key_order": {"top": 5, "grounding": 7}},  # 表 E の「key_order の例」には、最初の 1 件が選ばれる
        {"usage": "oops"}, {"usage": {"requests": "2", "input_tokens": None, "details": []}}, {"git": ["x"]},
        {"response": "oops"}, {"response": {"headers": ["x"], "finish_reason": ["a"]}}, {"request": "oops"},
        {"request": {"temperature": {"a": 1}}}, {"key_order": ["x"]},
        {"labels_sha256": "oops"}, {"started_at": 12345, "finished_at": None}, {"duration_s": "fast"}, {"dry_run": "yes"},
        {"order_ok": "maybe"}, {"verdicts": "oops"}, {"status": None}, {"model": 123, "model_name": ["x"]}, {"rubric_sha256": None},
    ]
    slots = [(v, c, k) for v in ("baseline", "step3_policy") for c in ("01", "02", "03") for k in (1, 2, 3)]
    for (variant, case_id, trial), over in zip(slots, broken, strict=True):
        put_judge(ws, variant, case_id, trial, **over)
    data, md = run_summary(ws)
    assert "## 詳細" in md and data["variants"]["baseline"]["records"]["found"] == 9
    assert data["variants"]["step3_policy"]["records"]["usable"] <= 9


# --- 10. 記録のあとで、件・参照が変わった記録は集計から外す（case_sha256・prompt_sha256） ----------


def test_sha256_json_is_the_same_calculation_as_compare_py():
    """記録の case_sha256 は compare.py の sha256_json。report.py が今の件を同じ計算で出さないと、照合できない。"""
    samples = [
        {"id": "01", "question": "LoRAとは何ですか？", "answer": "日本語の回答\n\n関係する回：第2回", "reference": {"glossary": ["lora"], "expected": "定義に沿う"}},
        {"b": 1, "a": [1, 2, {"z": None, "y": True}], "id": 7},  # 整数の id・入れ子・キーの順
        {"when": datetime.date(2026, 11, 16), "path": Path("/x")},  # json にできない値は str() にする（default=str）
        [], "text", 3, None,
    ]
    for data in samples:
        assert report.sha256_json(data) == compare.sha256_json(data)
    assert report.sha256_text("あ") == compare.sha256_text("あ") == hashlib.sha256("あ".encode()).hexdigest()
    assert report.sha256_json({"a": 1, "b": 2}) == report.sha256_json({"b": 2, "a": 1})  # キーの順に依らない
    assert report.sha256_json({"id": "01"}) == hashlib.sha256(b'{"id": "01"}').hexdigest()  # 固定した値（区切りは ", " と ": "、日本語は \u にしない）
    assert report.sha256_json({"q": "あ"}) == hashlib.sha256('{"q": "あ"}'.encode()).hexdigest()


def test_current_case_hashes_are_what_compare_py_records(tmp_path):
    cases = judge.read_jsonl(judge.DEFAULT_CASES)  # 配布の評価セット 10 件
    hashes = report.current_case_hashes(judge.DEFAULT_CASES)
    assert list(hashes) == [row["id"] for row in cases] and len(cases) == 10
    for row in cases:
        assert hashes[row["id"]] == {"case_sha256": compare.sha256_json(row), "prompt_sha256": compare.sha256_text(judge.judge_prompt(row))}
    # 整数の id でも、case_sha256 は「id を文字列にそろえる前の元の 1 行」から出す（compare.py はそのまま hash する）
    path = tmp_path / "ints.jsonl"
    write_jsonl(path, [{"id": 7, "question": "q", "answer": "a", "reference": {"glossary": [], "expected": "e"}}])
    (key, value), = report.current_case_hashes(path).items()
    assert key == "7" and value["case_sha256"] == compare.sha256_json(judge.read_jsonl(path)[0]) != compare.sha256_json({**judge.read_jsonl(path)[0], "id": "7"})
    write_jsonl(path, [{"id": "x", "question": "q"}])  # 回答の欄が無い件は、採点者に渡す文を作れない（prompt_sha256 は None で、記録と一致しない）
    assert report.current_case_hashes(path)["x"]["prompt_sha256"] is None


def test_a_record_written_by_compare_py_matches_what_report_py_expects(tmp_path):
    ctx = compare.Context(cfg=compare.load_config(), out=tmp_path / "out", dry_run=True)
    hashes = report.current_case_hashes(judge.DEFAULT_CASES)
    for case in judge.read_jsonl(judge.DEFAULT_CASES)[:3]:
        record = compare.run_judge_trial(ctx, ctx.effective("baseline"), case, 1)
        assert (record["case_sha256"], record["prompt_sha256"]) == (hashes[case["id"]]["case_sha256"], hashes[case["id"]]["prompt_sha256"])
        compare.write_json_atomic(compare.judge_path(ctx, ctx.variants()["baseline"], case["id"], 1), record)
    rs = report.load_judge_records(tmp_path / "out", "baseline")  # report.py の判定でも「使える」記録になる
    for case_id in (c["id"] for c in judge.read_jsonl(judge.DEFAULT_CASES)[:3]):
        verdicts, reason = report.judge_status_of(rs, case_id, 1, hashes[case_id])
        assert reason is None and verdicts is not None


def test_records_made_before_the_case_changed_are_left_out_with_a_warning(ws):
    populate(ws)
    rows = [CASES[0], {**CASES[1], "answer": "回答を差し替えました。\n\n関係する回：第3回"}, CASES[2]]
    write_jsonl(ws.root / "data/eval/cases.jsonl", rows)  # 採点のあとで、02 の回答を書き換えた
    data, md = run_summary(ws)

    for name in ("baseline", "step3_policy", "rubric_clarity"):
        res = data["variants"][name]
        assert [(m["case_id"], m["trial"], m["reason"]) for m in res["records"]["missing"]] == [("02", k, "case_changed") for k in (1, 2, 3)]
        assert [res["trials"][k]["usable"] for k in "123"] == [2, 2, 2] and [res["trials"][k]["dropped"] for k in "123"] == [1, 1, 1]
        assert res["cells"]["02"]["hint"] == [None] * 3 and None not in res["cells"]["01"]["hint"]  # 02 の判定は表から消え、ほかの件は残る
        assert res["trials"]["1"]["labels"]["mine"]["criteria"]["hint"]["total"] == 2  # 別の回答への判定を、今の回答の判定として数えない
    text = warning_texts(data)
    for title in ("V0 baseline", "V1 step3_policy", "V3 rubric_clarity"):
        assert f"{title}: 記録のあとで評価セットの件（case_sha256）が変わりました（評価セットが変わった）" in text
    assert "3 件を集計から外しました: 02 t1、02 t2、02 t3" in text and "02 t1（評価セットが変わった）" in text  # 件と変種が出る。欠けの理由にも出る
    changed = [w for w in data["warnings"] if w["kind"] == "case_changed"]
    assert [(w["variant"], w["cases"], w["count"]) for w in changed] == [("baseline", ["02"], 3), ("step3_policy", ["02"], 3), ("rubric_clarity", ["02"], 3)]
    assert not any(w["kind"] == "prompt_changed" for w in data["warnings"])
    table_b = table_groups(section(md, "table_b"))[0]  # V0〜V3。表 B の 02 は、どの変種も欠け（-,-,-）
    row = next(r for r in table_b if r[0].startswith(" 02 ") and r[1] == " ヒント ")
    assert [c.strip() for c in row[4:]] == ["-,-,-"] * 4
    row = next(r for r in table_b if r[0].startswith(" 01 ") and r[1] == " ヒント ")
    assert [c.strip() for c in row[4:]] == ["p,p,p", "p,p,p", "-,-,-", "p,p,p"]  # 変わっていない件は、そのまま出る


def test_a_changed_case_of_a_variant_that_was_not_graded_is_not_reported(ws):
    populate(ws)
    write_jsonl(ws.root / "data/eval/cases.jsonl", [CASES[0], CASES[1], {**CASES[2], "question": "今日の天気を教えて"}])  # 03 の質問を直す
    data, _ = run_summary(ws)
    assert [m["case_id"] for m in data["variants"]["baseline"]["records"]["missing"]] == ["03"] * 3
    assert data["variants"]["model_flash"]["records"]["missing"] == [{"case_id": c, "trial": k, "reason": "no_record"} for c in ("01", "02", "03") for k in (1, 2, 3)]
    assert {w["variant"] for w in data["warnings"] if w["kind"] == "case_changed"} == {"baseline", "step3_policy", "rubric_clarity"}  # 記録が無い V2 は「変わった」ではない


def test_a_record_without_a_case_sha256_is_left_out_too(ws):
    fill(ws, "baseline", "01")
    fill(ws, "baseline", "03")
    put_judge(ws, "baseline", "02", 1, case_sha256=None)
    put_judge(ws, "baseline", "02", 2)
    put_judge(ws, "baseline", "02", 3, case_sha256="x" * 64)
    data, _ = run_summary(ws)
    assert [(m["case_id"], m["trial"], m["reason"]) for m in data["variants"]["baseline"]["records"]["missing"]] == [
        ("02", 1, "case_changed"), ("02", 3, "case_changed")]  # 今の件の記録だと言えない記録は使わない。02 t2 だけが使える
    assert [data["variants"]["baseline"]["trials"][k]["usable"] for k in "123"] == [2, 3, 2]  # 件数も、使える記録の分だけ（02 は t2 だけ）
    assert data["variants"]["baseline"]["cells"]["02"]["grounding"] == [None, "pass", None]


def test_records_made_before_the_glossary_changed_are_left_out_with_their_own_warning(ws):
    populate(ws)
    put_judge(ws, "baseline", "01", 2, prompt_sha256="0" * 64)  # 件の行は同じで、採点者に渡した文（参照の用語集）だけが違う
    data, _ = run_summary(ws)
    assert [(m["case_id"], m["trial"], m["reason"]) for m in data["variants"]["baseline"]["records"]["missing"]] == [("01", 2, "prompt_changed")]
    (warning,) = [w for w in data["warnings"] if w["kind"] == "prompt_changed"]
    assert warning["variant"] == "baseline" and warning["cases"] == ["01"] and "参照（用語集）が変わった" in warning["message"]
    assert "baseline" not in [w.get("variant") for w in data["warnings"] if w["kind"] == "case_changed"] and "01 t2" in warning["message"]


def test_a_record_without_a_prompt_sha256_is_still_used(ws):
    """prompt_sha256 は後から足したキー。古い compare.py の記録（case_sha256 だけがある）は、その項目だけ照らさずに集計する。"""
    fill(ws, "baseline", "01")
    for k in (1, 2, 3):
        rec = json.loads((ws.out / "judge/baseline/01" / f"t{k}.json").read_text(encoding="utf-8"))
        del rec["prompt_sha256"]
        (ws.out / "judge/baseline/01" / f"t{k}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    data, _ = run_summary(ws)
    assert data["variants"]["baseline"]["records"]["usable"] == 3  # 01 の 3 試行は、すべて使える
    assert [m for m in data["variants"]["baseline"]["records"]["missing"] if m["case_id"] == "01"] == []
    assert not any(w["kind"] in ("case_changed", "prompt_changed") for w in data["warnings"])


def test_a_glossary_entry_that_changed_after_the_run_is_noticed(ws, monkeypatch):
    populate(ws)  # 01 の参照の用語集は lora。02・03 は参照なし
    changed = [{**e, "definition": e["definition"] + "（書き換えた）"} if e["id"] == "lora" else e for e in tools.load_glossary()]
    monkeypatch.setattr(tools, "load_glossary", lambda: changed)
    data, _ = run_summary(ws)
    for name in ("baseline", "step3_policy", "rubric_clarity"):
        assert [(m["case_id"], m["trial"], m["reason"]) for m in data["variants"][name]["records"]["missing"]] == [("01", k, "prompt_changed") for k in (1, 2, 3)]
    assert {w["variant"] for w in data["warnings"] if w["kind"] == "prompt_changed"} == {"baseline", "step3_policy", "rubric_clarity"}


# --- 11. generate の記録は、status が ok で、question と answer が評価セットの件と同じときだけ、手続きの判定に使う ---

SEARCH_CALL = {"order": 1, "response_index": 1, "tool_call_id": "a", "name": "search_glossary", "args": {"term": "LoRA"}, "result": "x", "outcome": "success"}
QUIZ_CALL = {"order": 1, "response_index": 1, "tool_call_id": "b", "name": "make_quiz", "args": {"lecture": 4}, "result": "x", "outcome": "success"}


def test_generate_state_says_why_a_record_cannot_be_used():
    case = {"id": "m01", "question": "質問", "answer": "回答"}
    ok = {"status": "ok", "question": "質問", "answer": "回答"}
    assert report.generate_state(ok, case) == {"status": "ok", "usable": True, "problems": [], "question_matches_cases": True, "answer_matches_cases": True}
    assert report.generate_state(None, case)["problems"] == ["missing"] and report.generate_state("oops", case)["usable"] is False
    assert report.generate_state({**ok, "status": "error"}, case)["problems"] == ["status:error"]
    assert report.generate_state({**ok, "question": "別の質問"}, case)["problems"] == ["question_differs"]
    assert report.generate_state({**ok, "answer": "別の回答"}, case)["problems"] == ["answer_differs"]
    both = report.generate_state({**ok, "question": "別の質問", "answer": "別の回答"}, case)
    assert both["problems"] == ["question_differs", "answer_differs"] and both["usable"] is False
    assert report.generate_state({**ok, "question": " 質問\n", "answer": "回答 \n\n"}, case)["usable"] is True  # 前後の空白だけの違いは見ない
    assert report.generate_state({"status": "ok"}, case)["problems"] == ["question_differs", "answer_differs"]  # question・answer の欄が無い記録
    assert report.problem_names(["missing"]) == "記録なし" and report.problem_names(["status:error"]) == "status=error"
    assert report.problem_names(both["problems"]) == "質問が違う・回答が違う"


def test_a_generate_record_of_another_question_or_answer_does_not_decide_the_procedure_checks(ws):
    populate(ws)
    put_generate(ws, "m01", question="書き換える前の質問", tool_calls=[SEARCH_CALL])  # 質問を直したあとの、古い記録（search_glossary は呼んでいた）
    put_generate(ws, "m02", answer="差し替える前の回答です。\n\n関係する回：第4回", tool_calls=[QUIZ_CALL])  # 回答を差し替えたあとの、古い記録
    data, md = run_summary(ws)

    got = {(r["case_id"], r["label"]): r["result"] for r in data["impl"]["code_checks"]}
    assert got[("m01", "search_glossary を呼んだ")] == "n/a" and got[("m02", "make_quiz(4) を呼んだ")] == "n/a"  # 古い記録で pass にしない
    assert got[("m01", "末尾に「第N回」がある")] == "pass" and got[("m02", "def / class を含むコードブロックが無い")] == "fail"  # 回答だけで決まる判定は、今の回答で出す
    state = data["impl"]["generate"]
    assert (state["m01"]["usable"], state["m01"]["problems"], state["m01"]["question_matches_cases"], state["m01"]["answer_matches_cases"]) == (False, ["question_differs"], False, True)
    assert (state["m02"]["usable"], state["m02"]["problems"]) == (False, ["answer_differs"]) and state["m03"]["problems"] == ["status:pending_approvals"]
    text = warning_texts(data)
    assert "generate の記録が使えない件があります（手続きの判定は n/a）" in text and "m01（質問が違う）" in text and "m02（回答が違う）" in text  # 理由が出る
    assert "generate の記録の question と評価セットの question が違う件があります" in text and "m01" in text.split("question が違う件")[1]
    assert "generate の記録の answer と評価セットの answer が違う件があります（評価セットを後から書き換えた可能性）" in text  # 今の警告は残す
    assert "この generate の記録は使えない（質問が違う）ので、手続きの判定は n/a です" in md and "この generate の記録は使えない（回答が違う）ので" in md
    assert next(g for g in data["impl"]["code_groups"] if g["label"] == "search_glossary を呼んだ")["results"]["m01"] == "n/a"


def test_a_generate_record_that_matches_the_case_is_used(ws):
    populate(ws)
    put_generate(ws, "m01", answer=MY_ANSWERS["m01"] + "\n", tool_calls=[SEARCH_CALL])  # 前後の空白だけ違う回答は、同じ回答として使う
    data, _ = run_summary(ws)
    assert {r["result"] for r in data["impl"]["code_checks"] if r["case_id"] == "m01" and r["label"] == "search_glossary を呼んだ"} == {"pass"}
    assert data["impl"]["generate"]["m01"] == {"status": "ok", "usable": True, "problems": [], "question_matches_cases": True, "answer_matches_cases": True}
    assert "question が違う件" not in warning_texts(data)


# --- 12. [checks.cases] の id と、実装の評価セットの id ---------------------------------------------


def test_checks_cases_ids_that_do_not_match_the_impl_evalset_are_warned(ws):
    write_jsonl(ws.root / "data/eval/my_cases.jsonl", [{**row, "id": f"{i:02d}"} for i, row in enumerate(MY_CASES, 1)])  # tools/annotate.py の「作問」は、id を 01 から振る
    fill(ws, "baseline", "01")
    data, md = run_summary(ws)

    unknown = next(w for w in data["warnings"] if w["kind"] == "checks_cases_unknown")
    uncovered = next(w for w in data["warnings"] if w["kind"] == "checks_cases_uncovered")
    assert unknown["ids"] == ["m01", "m02", "m03"] and uncovered["ids"] == ["01", "02", "03"]
    assert "実装: [checks.cases] にあって、評価セット `data/eval/my_cases.jsonl` に無い id があります（その件の個別の判定は使われません）: m01、m02、m03" in unknown["message"]
    assert "実装: 評価セット `data/eval/my_cases.jsonl` にあって、[checks.cases] に無い id があります（その件には [checks] の global の判定だけを当てます）: 01、02、03" in uncovered["message"]
    assert "作問" in unknown["message"] and f"- {unknown['message']}" in section(md, "warnings") and f"- {uncovered['message']}" in section(md, "warnings")  # summary.md の警告にも出る
    assert all(r["scope"] == "global" for r in data["impl"]["code_checks"])  # 個別の判定は、黙って抜けている（global だけ）


def test_checks_cases_warnings_cover_each_side_and_stay_quiet_otherwise(ws):
    fill(ws, "baseline", "01")
    ws.conditions.write_text(CONDITIONS.replace("m03 = [", "x03 = [", 1), encoding="utf-8")  # m03 を [checks.cases] から外し、無い id（x03）を足した
    data, _ = run_summary(ws)
    kinds = {w["kind"]: w["ids"] for w in data["warnings"] if w["kind"].startswith("checks_cases")}
    assert kinds == {"checks_cases_unknown": ["x03"], "checks_cases_uncovered": ["m03"]}

    ws.conditions.write_text(CONDITIONS, encoding="utf-8")  # そろっていれば、何も出ない
    data, _ = run_summary(ws)
    assert not any(w["kind"].startswith("checks_cases") for w in data["warnings"])

    ws.conditions.write_text(CONDITIONS.split("[checks.cases]")[0], encoding="utf-8")  # [checks.cases] が無ければ、個別の判定を使わない設定なので、出さない
    data, _ = run_summary(ws)
    assert not any(w["kind"].startswith("checks_cases") for w in data["warnings"])

    ws.conditions.write_text(CONDITIONS, encoding="utf-8")
    (ws.root / "data/eval/my_cases.jsonl").write_text("{読めない\n", encoding="utf-8")  # 評価セットを読めないときは、id が分からないので出さない（読めない警告は出る）
    data, _ = run_summary(ws)
    assert not any(w["kind"].startswith("checks_cases") for w in data["warnings"]) and any(w["kind"] == "cases_unavailable" for w in data["warnings"])


# --- 13. 予行の入力（<out>/dry_run_inputs）を使う --------------------------------------------------


def put_dry_run_inputs(ws, *names):
    """compare.py run --dry-run が書く、予行の入力（ws の条件ファイルでの名前）。"""
    inputs = ws.out / "dry_run_inputs"
    files = {"my_cases.jsonl": MY_CASES, "my_labels.jsonl": MY_LABELS, "mine.jsonl": MINE}
    for name in names:
        write_jsonl(inputs / name, files[name])
    return inputs


def test_dry_run_inputs_are_used_instead_of_the_files_under_root_and_the_summary_says_so(ws):
    populate(ws)
    for rel in ("labels/mine.jsonl", "labels/my_labels.jsonl", "data/eval/my_cases.jsonl"):  # 本物のファイルが無い（まだ作問・ラベル付けをしていない）
        (ws.root / rel).unlink()
    inputs = put_dry_run_inputs(ws, "my_cases.jsonl", "my_labels.jsonl", "mine.jsonl")
    data, md = run_summary(ws)

    assert data["dry_run_inputs"]["used"] is True and data["dry_run_inputs"]["files"] == ["mine.jsonl", "my_cases.jsonl", "my_labels.jsonl"]
    assert data["dry_run_inputs"]["dir"].endswith("/dry_run_inputs")
    assert data["variants"]["baseline"]["labels"]["mine"]["state"] == "ok" and data["variants"]["baseline"]["labels"]["mine"]["sha256"] == report.file_sha256(inputs / "mine.jsonl")
    assert data["variants"]["baseline"]["labels"]["mine"]["path"] == "labels/mine.jsonl"  # 表示は、条件ファイルに書いてあるパスのまま
    assert data["variants"]["baseline"]["trials"]["1"]["labels"]["mine"]["overall"]["compared"] > 0  # 表A に一致率が出る
    assert data["variants"]["impl_step3"]["trials"]["1"]["labels"]["mine"]["overall"]["total"] == 3
    assert {r["result"] for r in data["impl"]["code_checks"] if r["label"] == "search_glossary を呼んだ"} == {"pass"}  # 手続きの判定が n/a でない
    assert not any(w["kind"] in ("cases_unavailable", "labels_unavailable", "labels_changed", "case_changed") for w in data["warnings"])  # 予行の入力と記録は食い違っていない
    assert [w["cases"] for w in data["warnings"] if w["kind"] == "generate_unavailable"] == [["m03（status=pending_approvals）"]]  # populate が m03 だけ、承認待ちのままにしている
    lines = md.splitlines()
    assert lines[0].startswith("<!--") and lines[1].startswith("# ") and lines[2] == ""  # summary.md の先頭
    assert lines[3].startswith("> **予行の入力（dry_run_inputs）を使った**") and "mine.jsonl・my_cases.jsonl・my_labels.jsonl" in lines[3]
    assert "本物の採点ではありません" in lines[3]


def test_dry_run_inputs_win_over_a_different_file_under_root_but_not_for_the_instructor_labels(ws):
    populate(ws)
    inputs = put_dry_run_inputs(ws, "my_labels.jsonl")
    write_jsonl(ws.root / "labels/my_labels.jsonl", [label_row(f"m0{i}", hint="fail") for i in (1, 2, 3)])  # root の同じ名前のファイルは、別の中身
    write_jsonl(inputs / "instructor.jsonl", [label_row("01", hint="fail")])  # 講師のラベルは、予行では差し替えない
    data, md = run_summary(ws)

    assert data["variants"]["impl_step3"]["labels"]["mine"]["sha256"] == report.file_sha256(inputs / "my_labels.jsonl") != sha(ws, "labels/my_labels.jsonl")
    assert data["variants"]["baseline"]["labels"]["instructor"]["sha256"] == sha(ws, "labels/instructor.jsonl")
    assert data["variants"]["baseline"]["labels"]["mine"]["sha256"] == sha(ws, "labels/mine.jsonl")  # 予行の入力に無いファイルは、root のものを読む
    assert data["dry_run_inputs"]["files"] == ["my_labels.jsonl"] and "を使った**: " in md.splitlines()[3] and "my_labels.jsonl。" in md.splitlines()[3]


def test_without_dry_run_inputs_the_files_under_root_are_read_and_nothing_is_said(ws):
    populate(ws)
    data, md = run_summary(ws)
    assert data["dry_run_inputs"]["used"] is False and data["dry_run_inputs"]["files"] == []
    assert "dry_run_inputs）を使った" not in md and md.splitlines()[2] == "" and md.splitlines()[3].startswith("結果の置き場所")
    (ws.out / "dry_run_inputs").mkdir()  # 空のフォルダがあるだけなら、使ったことにしない
    data, md = run_summary(ws)
    assert data["dry_run_inputs"]["used"] is False and "dry_run_inputs）を使った" not in md


def test_load_file_picks_the_names_that_compare_py_writes(tmp_path):
    root, inputs = tmp_path / "repo", tmp_path / "out" / "dry_run_inputs"
    for rel, rows in (("lectures/04/labels.jsonl", MINE), ("lectures/04/my_labels.jsonl", MY_LABELS), ("data/eval/my_cases.jsonl", MY_CASES)):
        write_jsonl(root / rel, [{"id": "real", "x": 1}] if rel.endswith("labels.jsonl") else [{"id": "real", "question": "q", "answer": "a"}])
        write_jsonl(inputs / Path(rel).name, rows)
    labels = report.load_file(root, "lectures/04/labels.jsonl", report.read_labels, inputs)
    my_labels = report.load_file(root, "lectures/04/my_labels.jsonl", report.read_labels, inputs)
    cases = report.load_file(root, "data/eval/my_cases.jsonl", report.read_cases, inputs)
    assert (sorted(labels.data), sorted(my_labels.data), [c["id"] for c in cases.data]) == (["01", "02", "03"], ["m01", "m02", "m03"], ["m01", "m02", "m03"])
    assert all(loaded.dry_run_input and loaded.file.parent == inputs for loaded in (labels, my_labels, cases))
    plain = report.load_file(root, "lectures/04/labels.jsonl", report.read_labels)  # inputs_dir を渡さなければ、root のファイル
    assert sorted(plain.data) == ["real"] and plain.dry_run_input is False and plain.path == "lectures/04/labels.jsonl"
    assert report.load_file(root, "", report.read_labels, inputs).state == "unset"


# --- 14. 書きかけの一時ファイル（.tmp-）------------------------------------------------------------


def test_summary_files_are_written_through_a_unique_temp_file_and_cleaned_up(tmp_path, monkeypatch):
    path = tmp_path / "o" / "summary.md"
    report.write_text_atomic(path, "元\n")
    stale = path.parent / ".tmp-stale"
    stale.write_text("前の実行の書きかけ", encoding="utf-8")
    seen, real_replace = [], os.replace

    def spy(src, dst):
        assert Path(src).parent == path.parent  # 同じフォルダ（同じファイルシステム）に書く
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(report.os, "replace", spy)
    report.write_text_atomic(path, "新 1\n")
    report.write_text_atomic(path, "新 2\n")
    assert path.read_text(encoding="utf-8") == "新 2\n" and len(seen) == 2 and seen[0] != seen[1]  # 毎回、固有の名前
    assert all(name.startswith(".tmp-") and name != ".tmp-stale" for name in seen)
    assert stale.read_text(encoding="utf-8") == "前の実行の書きかけ" and sorted(p.name for p in path.parent.iterdir()) == [".tmp-stale", "summary.md"]

    def boom(src, dst):
        raise OSError("置き換えに失敗")

    monkeypatch.setattr(report.os, "replace", boom)
    with pytest.raises(OSError, match="置き換えに失敗"):
        report.write_text_atomic(path, "失敗\n")
    assert path.read_text(encoding="utf-8") == "新 2\n" and sorted(p.name for p in path.parent.iterdir()) == [".tmp-stale", "summary.md"]  # 元のまま・書きかけも残らない


def test_summary_files_keep_the_mode_of_the_replaced_file(tmp_path):
    path = tmp_path / "summary.json"
    umask = os.umask(0)
    os.umask(umask)
    report.write_text_atomic(path, "a\n")
    assert path.stat().st_mode & 0o777 == 0o666 & ~umask  # 一時ファイルの 0o600 のままにしない
    path.chmod(0o640)
    report.write_text_atomic(path, "b\n")
    assert path.stat().st_mode & 0o777 == 0o640


def test_temp_files_are_not_counted_as_records(ws):
    populate(ws)
    (ws.out / "judge/baseline/01/.tmp-abc").write_text("{書きかけ", encoding="utf-8")
    (ws.out / "judge/baseline/01/.tmp-t4.json").write_text("{書きかけ", encoding="utf-8")  # t<k>.json に見えても、.tmp- で始まれば数えない
    (ws.out / "judge/baseline/.tmp-xyz").write_text("x", encoding="utf-8")
    (ws.out / "generate/.tmp-m01.json").write_text("{書きかけ", encoding="utf-8")
    (ws.out / "generate/.tmp-m99.json").write_text(json.dumps({"status": "ok", "tool_calls": []}), encoding="utf-8")  # 読める JSON でも、数えない

    records = report.load_judge_records(ws.out, "baseline")
    assert sorted(records.records) == [(c, k) for c in ("01", "02", "03") for k in (1, 2, 3)] and records.unreadable == {}
    assert sorted(report.load_generates(ws.out)) == ["m01", "m02", "m03"]
    data, _ = run_summary(ws)
    assert not any(w["kind"] in ("unreadable", "extra") for w in data["warnings"]) and data["variants"]["baseline"]["records"]["found"] == 9
