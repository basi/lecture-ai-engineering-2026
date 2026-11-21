#!/usr/bin/env python3
"""第4回: compare.py が保存した記録（judge と generate）を集計し、summary.md と summary.json を作る。

compare.py の summarize サブコマンドは、このモジュールの summarize() を呼ぶだけ。単独でも動く:
    uv run python lectures/04/report.py --root . --conditions lectures/04/conditions.toml --out lectures/04/results

読むもの（root はリポジトリ直下。評価セットとラベルのパスは root からの相対）:
    conditions.toml                             変種（label・level・cases・labels・instructor_labels・trials）と [checks]
    <out>/judge/<variant>/<case_id>/t<k>.json   採点の記録（status が ok で、記録の case_sha256・prompt_sha256 が今の件と同じものだけ集計する。
                                                request_sample.json と、書きかけの .tmp- のファイルは読まない）
    <out>/generate/<case_id>.json               回答の生成の記録（実装の「手続き」のチェックに使う。status が ok で、
                                                question と answer が評価セットの件と同じものだけ）
    <out>/dry_run_inputs/                       あれば、そこにある labels.jsonl・my_labels.jsonl・my_cases.jsonl（compare.py run --dry-run が書く
                                                予行の仮のラベルと生成済みの評価セット）を、root の同じ名前のファイルより先に使う
書くもの: <out>/summary.md（手で直さない）と <out>/summary.json（answer.md への転記用）。

定義（app/judge.py の summarize() をそのまま使い、自前では数え直さない）:
    一致率 A  = agree / compared   LLM が unsure の項目と、人のラベルが pass / fail でない項目は分母から外す
    coverage  = compared / total
    一致率 B  = agree / total      要確認も不一致に数える
    常に pass = 人のラベルが pass の件数 / pass か fail の件数（全部 pass と答える採点器の一致率）
    ok でない記録の件は「欠け」として数え、その件は集計から外す（外した件数も表に出す）。
    記録のあとで件（評価セットの行）や参照の用語集が変わった記録（case_sha256・prompt_sha256 が今と違う）も、同じく欠けとして外し、警告に出す。
揺れの 3 段階（セル = 件 × 基準）:
    変種内の揺れ ... 3 試行の判定が全部同じでないセル
    V0 との比較  ... 両方 3 試行そろったセルで、pass・fail・unsure の回数の差の最大値 d を出す。
                    d = 3 は「変わった」、d = 2 は「傾向」、d が 1 以下は「揺れ」

summary.json の形（キーは英語。値は数と文字列だけ）:
    variants.<name>.trials.<k>.labels.<mine|instructor>.criteria.<criterion> / .overall
        total, agree, compared, disagree, unsure, human_unlabeled, missed, false_alarm,
        agree_rate_a, coverage, agree_rate_b, always_pass{pass, labeled, rate}
    variants.<name>.cells.<case_id>.<criterion>   試行順の判定（欠けは null）
    variants.<name>.within_variant / vs_baseline  揺れ（変種内・V0 との比較）
    variants.<name>.order_check / usage           表 E・表 F
    label_disagreements / clarity / impl          表 C・表 D・表 G（impl.generate.<id> = generate の記録を手続きの判定に使えるか: usable・problems）
    dry_run_inputs                                予行の入力（<out>/dry_run_inputs）を使ったか（used・dir・files）
"""

# 注意: from __future__ import annotations は付けない。compare.py の summarize は、このファイルを
# importlib.util.spec_from_file_location で読み込み、sys.modules には登録しない。その読み込み方だと、
# 注釈が文字列になった @dataclass は、定義の時点（dataclasses が sys.modules を引く）で失敗する。

import argparse
import contextlib
import hashlib
import json
import os
import re
import statistics
import sys
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # 「python lectures/04/report.py」で動かすと、app/ が見つからないため

from app.agent import INSTRUCTIONS  # noqa: E402
from app.judge import (  # noqa: E402
    ALL_CRITERIA,
    LLM_CRITERIA,
    NAMES,
    REQUIRED,
    check_lecture_mark,
    judge_prompt,
    load_labels,
    overall,
    read_jsonl,
)
from app.judge import summarize as judge_summarize  # noqa: E402

DEFAULT_CONDITIONS = HERE / "conditions.toml"
DEFAULT_OUT = HERE / "results"

CRITERIA = [key for key, _ in ALL_CRITERIA]  # 根拠・範囲・ヒント・分かりやすさ・回の表示（最後だけコードで判定）
LLM_KEYS = [key for key, _ in LLM_CRITERIA]
VERDICTS = ("pass", "fail", "unsure")
PASS_FAIL = ("pass", "fail")
MARK = {"pass": "p", "fail": "f", "unsure": "u"}
MISSING = "-"
OVERALL = "overall"  # 基準の行に並べる「総合」のキー
OVERALL_NAME = "総合"
LABEL_OWNERS = ("mine", "instructor")
OWNER_NAMES = {"mine": "自分", "instructor": "講師"}
BASELINE_LABEL = "V0"  # 基準の変種（label がこれのもの）
CLARITY_LABEL = "V3"  # 分かりやすさの基準を書き足した変種（表 D）
COMPARE_TRIALS = 3  # V0 との比較は、両方がこの回数そろったセルだけ
CHANGED_DISTANCE = 3
TREND_DISTANCE = 2
SHIFT_NAMES = {"changed": "変わった", "trend": "傾向", "fluctuation": "揺れ"}
SKIP_NAMES = {"no_baseline": "V0 なし", "is_baseline": "基準", "impl": "対象外（impl）", "other_evalset": "評価セットが違う", "single_trial": "試行 1 回"}
GENERATE_KINDS = {"tool_called", "approval_denied_not_saved"}  # generate の記録が要るチェック
DRIFT_FIELDS = ("rubric_sha256", "instructions_sha256", "schema_sha256", "model")
REQUEST_FIELDS = ("response_json_schema", "property_ordering", "temperature", "thinking_config", "tool_mode")
STATE_NAMES = {"ok": "読めました", "unset": "設定なし", "missing": "ファイルがありません", "error": "読めません"}
REASON_NAMES = {
    "no_record": "記録なし", "unreadable": "読めない", "invalid_verdicts": "verdicts が不正",
    "case_changed": "評価セットが変わった", "prompt_changed": "参照（用語集）が変わった",
}
CHANGED_REASONS = ("case_changed", "prompt_changed")  # 記録のあとで、件・参照が変わった（今の件の記録ではない）
PROBLEM_NAMES = {"missing": "記録なし", "question_differs": "質問が違う", "answer_differs": "回答が違う"}  # generate の記録を使えない理由
TMP_PREFIX = ".tmp-"  # compare.py と同じ。書きかけの一時ファイルの名前の頭（結果として数えない）
DRY_RUN_INPUTS = "dry_run_inputs"  # compare.py run --dry-run が、仮のラベルと生成済みの評価セットを置くフォルダ（<out> の下）
RESULT_PREVIEW = 120  # ツールの結果は先頭だけ見せる（字）
LIST_LIMIT = 8  # 警告に並べる件数の上限（全件は summary.json にある）

TITLE = "第4回 集計（手で直さない）"
HEADER_COMMENT = "<!-- compare.py summarize（report.py）が作る。手で直さない -->"
SECTIONS = {
    "warnings": "警告",
    "runs": "実行の記録",
    "table_a": "表A 基準ごとの集計（自分のラベル）",
    "table_a2": "表A′ 基準ごとの集計（講師のラベル）",
    "table_b": "表B 判定の一覧",
    "table_c": "表C 自分と講師でラベルが割れたセル",
    "table_d": "表D V3 の LLM とコード",
    "table_e": "表E 順序の操作の確認",
    "table_f": "表F 使用量と警告",
    "table_g": "表G 実装",
    "fluctuation": "揺れの一覧",
    "details": "詳細",
}


class ConfigError(Exception):
    """conditions.toml が読めない、または形が違う。"""


# ---------------------------------------------------------------------------
# 小さな部品（表示・比率・ファイル）
# ---------------------------------------------------------------------------


def ratio(num: int, den: int) -> float | None:
    """割合。分母が 0 なら None。"""
    return num / den if den else None


def cell(value) -> str:
    """Markdown の表のセルに入れる文字列にする。改行は空白に、| は \\| に置き換える。"""
    return re.sub(r"\r\n|\r|\n", " ", str(value)).replace("|", "\\|")


def md_table(headers: list[str], rows: list[list]) -> list[str]:
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"表の列数が合いません: {len(row)} != {len(headers)}: {row}")
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return lines


def one_line(text, limit: int | None = None) -> str:
    """空白（改行を含む）をつめて 1 行にする。limit を超える分は … で切る。"""
    line = re.sub(r"\s+", " ", str(text)).strip()
    return line[:limit] + "…" if limit and len(line) > limit else line


def fenced(text: str, lang: str = "") -> list[str]:
    """コードブロックにする。text の中にある ``` より長い囲みを使うので、中身で壊れない。"""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{lang}", *text.splitlines(), fence]


def listing(items: list[str], limit: int = LIST_LIMIT) -> str:
    shown = "、".join(items[:limit])
    return shown + (f"（ほか {len(items) - limit} 件）" if len(items) > limit else "")


def marks(seq) -> str:
    """試行順の判定を p,f,u の列にする。欠け（None）は -。"""
    return ",".join(MARK.get(v, MISSING) for v in seq)


def label_mark(value) -> str:
    """人のラベルを p / f にする。pass・fail 以外（欠け）は -。"""
    return MARK[value] if value in PASS_FAIL else MISSING


def decorate(text: str, shift: str | None) -> str:
    """V0 との比較で「変わった」は太字、「傾向」は斜体にする。"""
    if shift == "changed":
        return f"**{text}**"
    if shift == "trend":
        return f"*{text}*"
    return text


def criterion_name(key: str) -> str:
    if key == OVERALL:
        return OVERALL_NAME
    return NAMES[key] + ("（コード）" if key == CRITERIA[-1] else "")


def case_title(case: dict) -> str:
    category = case.get("category")
    return f"{case['id']} {category}" if category else str(case["id"])


def kind_key(value) -> str:
    """値の種類を数えるときのキー。文字列はそのまま、真偽・数・null・dict は JSON の書き方にする。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def fmt_counts(counts: dict, empty: str = MISSING) -> str:
    if not counts:
        return empty
    return ", ".join(f"{k} ×{n}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def fmt_model_counts(counts: dict) -> str:
    """応答の model_name の種類。予行（dry_run）や失敗で応答が無い記録は null なので、「（なし）」と書く。"""
    return fmt_counts({("（なし）" if key == "null" else key): n for key, n in counts.items()})


def number_stats(values: list) -> dict | None:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    return {"n": len(nums), "mean": statistics.fmean(nums), "min": min(nums), "max": max(nums)}


def fmt_stats(stats: dict | None, mean_digits: int = 1, range_digits: int = 0) -> str:
    """平均と最小〜最大。例: 123.4 (100〜150)"""
    if not stats:
        return MISSING
    return f"{stats['mean']:.{mean_digits}f} ({stats['min']:.{range_digits}f}〜{stats['max']:.{range_digits}f})"


def parse_time(text) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def time_range(records: list[dict]) -> tuple[str | None, str | None]:
    """started_at の最小と finished_at の最大（元の文字列のまま）。"""

    def pick(name: str, choose):
        found = [(t, r[name]) for r in records if name in r and (t := parse_time(r[name])) is not None]
        return choose(found)[1] if found else None

    return pick("started_at", min), pick("finished_at", max)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data) -> str:
    """キーの順を並べ替えて hash する（同じ内容なら同じ値）。

    compare.py の sha256_json と同じ計算（sort_keys・ensure_ascii=False・default=str）。記録の case_sha256 と、今の評価セットの件を
    照らすために、同じ計算をここにも置く（test_report.py が、両者が同じ値を返すことを固定している）。計算を変えるときは、両方を直す。
    """
    return sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))


def file_mode_for(path: Path) -> int:
    """書くファイルの権限。置き換えるファイルがあればその権限、なければ普通の新規ファイルと同じ（0o666 から umask を引いたもの）。"""
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def write_text_atomic(path: Path, text: str) -> None:
    """固有の名前の一時ファイル（同じフォルダの .tmp-…）に書いてから、os.replace で置く（途中で止まっても、半端な summary を残さない）。

    どんな終わり方でも、残った一時ファイルは消す。compare.py と同じ方式。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=TMP_PREFIX, delete=False) as f:
            tmp = Path(f.name)
            f.write(text)
        with contextlib.suppress(OSError):
            os.chmod(tmp, file_mode_for(path))
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def display_path(path: Path, root: Path) -> str:
    """root の下なら相対、そうでなければそのまま（summary に環境ごとの絶対パスを残さないため）。"""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# コードの判定（実装の評価セットだけに当てる）
# ---------------------------------------------------------------------------


def body_chars(answer: str) -> int:
    """回答の本文の字数。コードブロック（``` で囲んだ部分）は数えない。V3 の比較にも使う。"""
    return len(re.sub(r"```.*?```", "", answer, flags=re.S).strip())


def pass_if(ok: bool) -> str:
    return "pass" if ok else "fail"


def normalize_arg(value) -> str:
    """引数の値を str() で比べる。4.0 は "4" にそろえる。"""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def args_match(want: dict | None, got) -> bool:
    """want の各キーが、got（呼び出しの引数。dict か JSON 文字列）に同じ値であれば True（部分一致）。want が空なら引数は問わない。"""
    if not want:
        return True
    if isinstance(got, str):
        try:
            got = json.loads(got)
        except ValueError:
            return False
    if not isinstance(got, dict):
        return False
    return all(key in got and normalize_arg(got[key]) == normalize_arg(value) for key, value in want.items())


def instruction_lines() -> list[str]:
    """INSTRUCTIONS の各行から、先頭の「- 」と前後の空白を除いた、12 文字以上のものを取り出す。"""
    lines = []
    for raw in INSTRUCTIONS.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if len(line) >= 12:
            lines.append(line)
    return lines


def leaked_instruction_lines(answer: str) -> list[str]:
    """回答の中にそのまま出てしまっている INSTRUCTIONS の行。"""
    return [line for line in instruction_lines() if line in answer]


def has_code_def(answer: str) -> bool:
    """``` で囲んだブロックの中に、行頭が `def ` か `class ` の行があるか。"""
    for block in re.findall(r"```(.*?)```", answer, flags=re.S):
        if any(line.startswith(("def ", "class ")) for line in block.splitlines()):
            return True
    return False


def check_tool_called(check: dict, generate: dict) -> str:
    calls = generate.get("tool_calls")
    if not isinstance(calls, list):
        return "n/a"
    want = check.get("args")
    hit = any(
        isinstance(call, dict) and call.get("name") == check["tool"] and args_match(want, call.get("args"))
        for call in calls
    )
    return pass_if(hit)


def check_approval_denied_not_saved(check: dict, generate: dict) -> str:
    """その tool の承認がすべて denied で、保存したファイルも（外への書き込みも）無ければ pass。"""
    approvals, saved, outside = generate.get("approvals"), generate.get("saved_files"), generate.get("outside_files")
    if not isinstance(approvals, list) or saved is None or outside is None:
        return "n/a"  # 保存したかどうかを記録から言えない
    mine = [a for a in approvals if isinstance(a, dict) and a.get("tool_name") == check["tool"]]
    return pass_if(bool(mine) and all(a.get("decision") == "denied" for a in mine) and not saved and not outside)


def run_check(check: dict, answer: str, generate: dict | None) -> str:
    """コードの判定を 1 つ当てる。pass / fail / n/a を返す。

    n/a: generate の記録が要るのに使えない（無い・status が ok でない・question か answer が評価セットの件と違う。
    code_check_rows が、使えない記録は渡さない）、知らない kind、設定の誤り。
    answer は評価セットのファイルの answer を使う。
    """
    kind = check.get("kind")
    answer = str(answer or "")
    if (kind in GENERATE_KINDS or check.get("type") == "procedure") and generate is None:
        return "n/a"
    try:
        if kind == "lecture_mark":
            return check_lecture_mark(answer)
        if kind == "max_body_chars":
            return pass_if(body_chars(answer) <= int(check["limit"]))
        if kind == "tool_called":
            return check_tool_called(check, generate)
        if kind == "approval_denied_not_saved":
            return check_approval_denied_not_saved(check, generate)
        if kind == "answer_regex":
            return pass_if(re.search(check["pattern"], answer) is not None)
        if kind == "answer_not_regex":
            return pass_if(re.search(check["pattern"], answer) is None)
        if kind == "no_instructions_leak":
            return pass_if(not leaked_instruction_lines(answer))
        if kind == "no_code_def":
            return pass_if(not has_code_def(answer))
    except (KeyError, TypeError, ValueError, re.error):
        return "n/a"
    return "n/a"


def checks_for_case(checks_cfg: dict, case_id: str) -> list[tuple[str, dict]]:
    """[checks] の global を全件に、cases の分をその件に当てる。(scope, check) のリスト。"""
    checks = [("global", c) for c in checks_cfg.get("global") or [] if isinstance(c, dict)]
    per_case = (checks_cfg.get("cases") or {}).get(case_id) or []
    return checks + [("case", c) for c in per_case if isinstance(c, dict)]


def check_label(check: dict) -> str:
    return str(check.get("label") or check.get("kind") or "（無名のチェック）")


def counts_agreement(check: dict) -> bool:
    """type が output か aux で criterion が空でないチェックだけ、自分のラベルとの一致を数える。"""
    return check.get("type") in ("output", "aux") and bool(check.get("criterion"))


def code_check_rows(checks_cfg: dict, cases: list[dict], generates: dict[str, dict], labels: dict | None) -> list[dict]:
    """件ごと・チェックごとの判定。generate は、status が ok で、question と answer が評価セットの件と同じものだけ使う（違えば手続きの判定は n/a）。"""
    rows = []
    for case in cases:
        gen = generates.get(case["id"])
        gen = gen if generate_state(gen, case)["usable"] else None
        for scope, check in checks_for_case(checks_cfg, case["id"]):
            result = run_check(check, case.get("answer", ""), gen)
            criterion = str(check.get("criterion") or "")
            human = (labels or {}).get(case["id"], {}).get(criterion) if criterion else None
            human = human if human in PASS_FAIL else None
            comparable = counts_agreement(check) and result in PASS_FAIL and human is not None
            rows.append(
                {
                    "case_id": case["id"], "scope": scope, "kind": check.get("kind"), "type": check.get("type") or "",
                    "criterion": criterion, "label": check_label(check), "result": result, "human": human,
                    "counts_agreement": counts_agreement(check), "agree": (result == human) if comparable else None,
                }
            )
    return rows


def group_code_checks(rows: list[dict]) -> list[dict]:
    """判定ごと（global は全件を 1 つに、cases は件ごと）の一致数・比べた数・食い違った件。"""
    groups: dict[tuple, dict] = {}
    for row in rows:
        key = (row["scope"], row["case_id"] if row["scope"] == "case" else "", row["label"])
        group = groups.setdefault(
            key,
            {
                "scope": row["scope"], "case_id": row["case_id"] if row["scope"] == "case" else None,
                "label": row["label"], "kind": row["kind"], "type": row["type"], "criterion": row["criterion"],
                "counts_agreement": row["counts_agreement"], "results": {}, "agree": 0, "compared": 0, "disagreements": [],
            },
        )
        group["results"][row["case_id"]] = row["result"]
        if row["agree"] is not None:
            group["compared"] += 1
            group["agree"] += int(row["agree"])
            if not row["agree"]:
                group["disagreements"].append({"case_id": row["case_id"], "code": row["result"], "human": row["human"]})
    return list(groups.values())


# ---------------------------------------------------------------------------
# 一致率（app/judge.py の summarize() を土台にする）
# ---------------------------------------------------------------------------


def criterion_row(
    name: str, *, total: int, agree: int, compared: int, unsure: int, human_unlabeled: int,
    missed: int, false_alarm: int, human_pass: int, human_labeled: int,
) -> dict:
    return {
        "name": name, "total": total, "agree": agree, "compared": compared, "disagree": compared - agree,
        "unsure": unsure, "human_unlabeled": human_unlabeled, "missed": missed, "false_alarm": false_alarm,
        "agree_rate_a": ratio(agree, compared), "coverage": ratio(compared, total), "agree_rate_b": ratio(agree, total),
        "always_pass": {"pass": human_pass, "labeled": human_labeled, "rate": ratio(human_pass, human_labeled)},
    }


def overall_row(ids: list[str], verdicts: dict[str, dict], labels: dict[str, dict], base: dict) -> tuple[dict, bool]:
    """総合の行。app.judge.summarize と同じ定義で数え直し（見逃し・過検出・常に pass が summarize に無いため）、
    summarize の overall・coverage と食い違っていないかも返す。"""
    agree = compared = unsure = missed = false_alarm = human_pass = unlabeled = 0
    for cid in ids:
        judged = {k: verdicts[cid][k]["verdict"] for k in CRITERIA}
        human = {k: labels[cid].get(k) for k in REQUIRED}
        unlabeled += int(any(h not in PASS_FAIL for h in human.values()))
        judge_overall = overall(judged)
        # 人のラベルが pass / fail でない必須の基準は fail として扱う（summarize と同じ）
        human_overall = overall({k: (h if h in PASS_FAIL else "fail") for k, h in human.items()})
        human_pass += int(human_overall == "pass")
        if judge_overall == "unsure":
            unsure += 1
            continue
        compared += 1
        agree += int(judge_overall == human_overall)
        missed += int((judge_overall, human_overall) == ("pass", "fail"))
        false_alarm += int((judge_overall, human_overall) == ("fail", "pass"))
    row = criterion_row(
        OVERALL_NAME, total=len(ids), agree=agree, compared=compared, unsure=unsure, human_unlabeled=unlabeled,
        missed=missed, false_alarm=false_alarm, human_pass=human_pass, human_labeled=len(ids),
    )
    consistent = (agree, compared) == (base["overall"]["agree"], base["overall"]["compared"]) and (
        compared, len(ids)) == (base["coverage"]["covered"], base["coverage"]["total"])
    return row, consistent


def agreement_stats(cases: list[dict], verdicts: dict[str, dict], labels: dict[str, dict]) -> dict:
    """1 回の試行（verdicts: 件 → 基準 → {reason, verdict}）を、1 つのラベル（labels）と比べる。

    verdicts に無い件（status が ok でない記録）は集計から外す。agree・compared・total・review・mismatches は
    app.judge.summarize の値をそのまま使い、見逃し・過検出・要確認・人のラベルの欠け・一致率 B・常に pass を足す。
    """
    used = [c for c in cases if c["id"] in verdicts]
    base = judge_summarize(used, verdicts, labels)
    ids = [c["id"] for c in used if c["id"] in labels]
    unsure_by = Counter(r["criterion"] for r in base["review"])
    missed_by = Counter(m["criterion"] for m in base["mismatches"] if m["judge"] == "pass")  # LLM pass・人 fail
    false_alarm_by = Counter(m["criterion"] for m in base["mismatches"] if m["judge"] == "fail")  # LLM fail・人 pass
    criteria = {}
    for key in CRITERIA:
        human = [labels[cid].get(key) for cid in ids]
        labeled = [h for h in human if h in PASS_FAIL]
        per = base["per_criterion"][key]
        criteria[key] = criterion_row(
            per["name"], total=per["total"], agree=per["agree"], compared=per["compared"], unsure=unsure_by[key],
            human_unlabeled=len(human) - len(labeled), missed=missed_by[key], false_alarm=false_alarm_by[key],
            human_pass=labeled.count("pass"), human_labeled=len(labeled),
        )
    total_row, consistent = overall_row(ids, verdicts, labels, base)
    return {
        "n": base["n"], "criteria": criteria, "overall": total_row, "consistent": consistent,
        "review": base["review"], "mismatches": base["mismatches"],
    }


# ---------------------------------------------------------------------------
# 揺れの 3 段階
# ---------------------------------------------------------------------------


def verdict_counts(seq) -> tuple[int, int, int]:
    """試行順の判定から、(pass, fail, unsure) の回数を数える。"""
    return tuple(list(seq).count(v) for v in VERDICTS)


def is_complete(seq) -> bool:
    """比べられるだけ試行がそろっている（3 回分あり、欠けが無い）。"""
    return len(seq) == COMPARE_TRIALS and all(v in VERDICTS for v in seq)


def shift_distance_counts(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """(pass, fail, unsure) の回数どうしの差の最大値 d。"""
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def shift_distance(a, b) -> int | None:
    """2 つの試行順の判定（V0 と変種）の d。どちらかが 3 試行そろっていなければ None（比べない）。"""
    if not (is_complete(a) and is_complete(b)):
        return None
    return shift_distance_counts(verdict_counts(a), verdict_counts(b))


def classify_shift(d: int) -> str:
    """d = 3 は changed（変わった）、d = 2 は trend（傾向）、d が 1 以下は fluctuation（揺れ）。"""
    if d >= CHANGED_DISTANCE:
        return "changed"
    if d >= TREND_DISTANCE:
        return "trend"
    return "fluctuation"


def is_unstable(seq) -> bool | None:
    """変種内の揺れ: 3 試行の判定が全部同じでなければ True。欠けがあって比べられなければ None。"""
    if len(seq) < 2 or any(v not in VERDICTS for v in seq):
        return None
    return len(set(seq)) > 1


# ---------------------------------------------------------------------------
# 読み込み（条件・評価セット・ラベル・記録）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    level: str
    cases: str
    labels: str
    instructor_labels: str
    trials: int
    output: str
    order: str
    judge_model: str
    rubric: str

    @property
    def title(self) -> str:
        return f"{self.label} {self.name}"


def load_conditions(path: Path) -> dict:
    try:
        cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path} を読めません: {e}") from e
    if not isinstance(cfg.get("variants"), dict) or not cfg["variants"]:
        raise ConfigError(f"{path} に [variants.<name>] がありません")
    return cfg


def load_variants(cfg: dict) -> list[Variant]:
    """[variants.<name>] を、[defaults] の値で補って読む。名前と label は TOML から読む（条件ファイルの順）。"""
    defaults = cfg.get("defaults") or {}
    variants = []
    for name, table in cfg["variants"].items():
        merged = {**defaults, **(table or {})}
        try:
            trials = int(merged.get("trials", 3))
        except (TypeError, ValueError):
            raise ConfigError(f"variants.{name}: trials は整数です: {merged.get('trials')!r}") from None
        variants.append(
            Variant(
                name=name, label=str(merged.get("label") or name), level=str(merged.get("level") or ""),
                cases=str(merged.get("cases") or ""), labels=str(merged.get("labels") or ""),
                instructor_labels=str(merged.get("instructor_labels") or ""), trials=trials,
                output=str(merged.get("output") or ""), order=str(merged.get("order") or ""),
                judge_model=str(merged.get("judge_model") or ""), rubric=str(merged.get("rubric") or ""),
            )
        )
    return variants


@dataclass
class Loaded:
    """評価セットまたはラベルのファイルを読んだ結果。state は ok / unset（設定が空）/ missing（ファイルが無い）/ error。"""

    path: str  # conditions.toml に書いてある、root からの相対パス（読んだ場所ではない）
    state: str
    data: object = None  # 評価セットは list[dict]、ラベルは {id: 行}
    sha256: str | None = None
    detail: str = ""
    file: Path | None = None  # 実際に読んだファイル（予行の入力があれば、<out>/dry_run_inputs の中）
    dry_run_input: bool = False  # 予行の入力（<out>/dry_run_inputs）を使った

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    def describe(self) -> str:
        text = STATE_NAMES[self.state]
        return f"{text}: {self.detail}" if self.detail else text

    def info(self) -> dict:
        n = len(self.data) if self.ok else None
        return {"path": self.path, "state": self.state, "sha256": self.sha256, "n": n, "detail": self.detail}


def load_file(root: Path, rel: str, reader, inputs_dir: Path | None = None) -> Loaded:
    """root/rel を読む。inputs_dir（<out>/dry_run_inputs）に同じ名前のファイルがあれば、そちらを先に使う（compare.py が書く名前と同じ）。"""
    if not rel:
        return Loaded(rel, "unset")
    path, dry = Path(root) / rel, False
    if inputs_dir is not None and (Path(inputs_dir) / Path(rel).name).is_file():
        path, dry = Path(inputs_dir) / Path(rel).name, True
    if not path.is_file():
        return Loaded(rel, "missing")
    try:
        return Loaded(rel, "ok", reader(path), file_sha256(path), file=path, dry_run_input=dry)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return Loaded(rel, "error", detail=f"{type(e).__name__}: {e}", file=path, dry_run_input=dry)


def read_cases(path: Path) -> list[dict]:
    """評価セット。id は文字列にそろえる（記録のフォルダ名と照らすため）。"""
    return [{**row, "id": str(row["id"])} for row in read_jsonl(path)]


def read_labels(path: Path) -> dict[str, dict]:
    return {str(cid): row for cid, row in load_labels(path).items()}


@dataclass
class EvalSet:
    """同じ評価セットとラベルを使う変種のまとまり。"""

    index: int
    cases: Loaded
    mine: Loaded
    instructor: Loaded
    variants: list[Variant]
    case_hashes: dict = field(default_factory=dict)  # 件の id → {"case_sha256", "prompt_sha256"}（記録と照らす、今の評価セットの値）

    @property
    def case_list(self) -> list[dict]:
        return self.cases.data if self.cases.ok else []

    def labels_of(self, who: str) -> Loaded:
        return self.mine if who == "mine" else self.instructor

    @property
    def is_impl(self) -> bool:
        return bool(self.variants) and all(v.level == "impl" for v in self.variants)


def current_case_hashes(path: Path) -> dict[str, dict]:
    """今の評価セットの各件（id → hash）。compare.py が記録に残す値と同じ計算で出す。

    case_sha256 = 元の 1 行（id を文字列にそろえる前）の sha256_json、prompt_sha256 = 採点者に渡す文（judge_prompt）の sha256。
    質問・回答の欄が無い件の prompt_sha256 は None（記録の値と一致しない）。
    """
    out = {}
    for row in read_jsonl(path):
        try:
            prompt = sha256_text(judge_prompt(row))
        except (KeyError, TypeError, AttributeError):
            prompt = None
        out[str(row["id"])] = {"case_sha256": sha256_json(row), "prompt_sha256": prompt}
    return out


def load_evalsets(root: Path, variants: list[Variant], inputs_dir: Path | None = None) -> list[EvalSet]:
    """inputs_dir があれば、評価セットと自分のラベルは、そこにある同じ名前のファイルを先に使う（講師のラベルは、予行では差し替えない）。"""
    groups: dict[tuple, list[Variant]] = {}
    for v in variants:
        groups.setdefault((v.cases, v.labels, v.instructor_labels), []).append(v)
    evalsets = []
    for i, ((cases, labels, instructor), vs) in enumerate(groups.items()):
        loaded = load_file(root, cases, read_cases, inputs_dir)
        hashes: dict = {}
        if loaded.ok:
            try:
                hashes = current_case_hashes(loaded.file)
            except (OSError, ValueError, KeyError, TypeError):
                hashes = {}
        evalsets.append(
            EvalSet(
                index=i, cases=loaded, mine=load_file(root, labels, read_labels, inputs_dir),
                instructor=load_file(root, instructor, read_labels), variants=vs, case_hashes=hashes,
            )
        )
    return evalsets


@dataclass
class RecordSet:
    """<out>/judge/<variant>/ の記録。"""

    variant: str
    exists: bool = False
    records: dict[tuple[str, int], dict] = field(default_factory=dict)  # (件, 試行) → 記録。status は問わない
    unreadable: dict[tuple[str, int], str] = field(default_factory=dict)  # 読めなかった記録 → 理由


TRIAL_FILE = re.compile(r"t(\d+)\.json")


def read_json_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON の最上位が object ではありません")
    return data


def load_judge_records(out_dir: Path, variant: str) -> RecordSet:
    """<out>/judge/<variant>/<case_id>/t<k>.json を読む。request_sample.json など t<k>.json 以外は読まない。"""
    base = Path(out_dir) / "judge" / variant
    rs = RecordSet(variant=variant, exists=base.is_dir())
    if not rs.exists:
        return rs
    for case_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(case_dir.iterdir()):
            match = TRIAL_FILE.fullmatch(path.name)
            if not match:
                continue
            key = (case_dir.name, int(match.group(1)))
            try:
                rs.records[key] = read_json_object(path)
            except (OSError, ValueError) as e:
                rs.unreadable[key] = f"{type(e).__name__}: {e}"
    return rs


def load_generates(out_dir: Path) -> dict[str, dict]:
    """<out>/generate/<case_id>.json。読めないものは無いものとして扱う。書きかけの .tmp- のファイルは、結果として数えない。"""
    records = {}
    base = Path(out_dir) / "generate"
    if base.is_dir():
        for path in sorted(base.glob("*.json")):
            if path.name.startswith(TMP_PREFIX):
                continue
            try:
                records[path.stem] = read_json_object(path)
            except (OSError, ValueError):
                continue
    return records


def clean_verdicts(rec: dict) -> dict | None:
    """記録の verdicts を、app.judge.summarize に渡せる形にそろえる。5 基準がそろっていなければ None（欠け扱い）。"""
    raw = rec.get("verdicts")
    if not isinstance(raw, dict):
        return None
    out = {}
    for key in CRITERIA:
        item = raw.get(key)
        allowed = VERDICTS if key in LLM_KEYS else PASS_FAIL  # 回の表示はコードが決めるので unsure は無い
        if not isinstance(item, dict) or item.get("verdict") not in allowed:
            return None
        out[key] = {"reason": str(item.get("reason") or ""), "verdict": item["verdict"]}
    return out


@dataclass
class VariantData:
    variant: Variant
    evalset: EvalSet
    recordset: RecordSet
    usable: dict[int, dict[str, dict]] = field(default_factory=dict)  # 試行 → 件 → verdicts（ok の記録だけ）
    missing: list[dict] = field(default_factory=list)  # 欠け: {case_id, trial, reason}
    extra: list[dict] = field(default_factory=list)  # 集計に含めなかった記録（評価セットにない件・予定を超える試行）
    cells: dict[tuple[str, str], list[str | None]] = field(default_factory=dict)  # (件, 基準) → 試行順の判定

    @property
    def records(self) -> list[dict]:
        """読めた記録すべて（status は問わない）。件と試行の順。"""
        return [self.recordset.records[key] for key in sorted(self.recordset.records)]

    @property
    def trial_range(self) -> range:
        return range(1, self.variant.trials + 1)


def judge_status_of(rs: RecordSet, case_id: str, trial: int, now: dict | None = None) -> tuple[dict | None, str | None]:
    """(使える verdicts, 欠けの理由)。

    now は今の件の {"case_sha256", "prompt_sha256"}（EvalSet.case_hashes）。None なら照らさない。
    case_sha256 が記録の値と違う（記録に無いときも）なら、別の件への判定なので使わない（理由は case_changed）。
    prompt_sha256（採点者に渡した文。参照の用語集を含む）は、記録にあるときだけ照らす。違えば、別の参照への判定なので使わない
    （理由は prompt_changed）。後から足したキーなので、無い記録（古い compare.py のもの）は、この項目だけ照らさない。
    """
    rec = rs.records.get((case_id, trial))
    if rec is None:
        return None, ("unreadable" if (case_id, trial) in rs.unreadable else "no_record")
    if rec.get("status") != "ok":
        return None, f"status:{rec.get('status')}"
    if now is not None:
        if rec.get("case_sha256") != now["case_sha256"]:
            return None, "case_changed"
        if rec.get("prompt_sha256") is not None and rec["prompt_sha256"] != now["prompt_sha256"]:
            return None, "prompt_changed"
    verdicts = clean_verdicts(rec)
    return (verdicts, None) if verdicts else (None, "invalid_verdicts")


def build_variant_data(v: Variant, es: EvalSet, rs: RecordSet) -> VariantData:
    vd = VariantData(variant=v, evalset=es, recordset=rs, usable={k: {} for k in range(1, v.trials + 1)})
    cases = es.case_list
    for case in cases:
        for k in vd.trial_range:
            verdicts, reason = judge_status_of(rs, case["id"], k, es.case_hashes.get(case["id"]))
            if verdicts is not None:
                vd.usable[k][case["id"]] = verdicts
            elif es.cases.ok:
                vd.missing.append({"case_id": case["id"], "trial": k, "reason": reason})
        for crit in CRITERIA:
            vd.cells[(case["id"], crit)] = [
                vd.usable[k].get(case["id"], {}).get(crit, {}).get("verdict") for k in vd.trial_range
            ]
    known = {c["id"] for c in cases}
    for case_id, k in sorted(rs.records):
        if case_id not in known:
            vd.extra.append({"case_id": case_id, "trial": k, "reason": "not_in_cases"})
        elif not 1 <= k <= v.trials:
            vd.extra.append({"case_id": case_id, "trial": k, "reason": "beyond_trials"})
    return vd


# ---------------------------------------------------------------------------
# 分析（数値を作る。summary.json はこの結果をそのまま書く）
# ---------------------------------------------------------------------------


@dataclass
class Analysis:
    root: Path
    conditions: Path
    out_dir: Path
    cfg: dict
    variants: list[Variant]
    evalsets: list[EvalSet]
    vdata: dict[str, VariantData]
    generates: dict[str, dict]
    dry_inputs: list[str] = field(default_factory=list)  # 使った予行の入力（<out>/dry_run_inputs の中のファイル名）。使わなければ空
    warnings: list[dict] = field(default_factory=list)
    results: dict = field(default_factory=dict)

    @property
    def baseline(self) -> VariantData | None:
        for v in self.variants:
            if v.label == BASELINE_LABEL:
                return self.vdata[v.name]
        return None

    @property
    def primary(self) -> EvalSet | None:
        """V0 の評価セット。V0 が無ければ、impl でない最初の変種の評価セット。"""
        if self.baseline:
            return self.baseline.evalset
        return next((es for es in self.evalsets if not es.is_impl), None)

    @property
    def impl_evalset(self) -> EvalSet | None:
        """実装（level = impl）の評価セット。コードの判定と表 G は、最初のものだけに当てる。"""
        return next((es for es in self.evalsets if es.is_impl), None)

    @property
    def n_records(self) -> int:
        return sum(len(vd.recordset.records) for vd in self.vdata.values())

    def variant_by_label(self, label: str) -> Variant | None:
        return next((v for v in self.variants if v.label == label), None)


def count_by(records: list[dict], name: str) -> dict[str, int]:
    return dict(Counter(kind_key(r.get(name)) for r in records))


def commit_summary(records: list[dict]) -> tuple[dict[str, int], int]:
    """git の commit（短縮）ごとの記録数と、未コミットの変更があった（dirty）記録の数。"""
    commits: Counter = Counter()
    dirty = 0
    for r in records:
        git = r.get("git")
        if isinstance(git, dict):
            commits[str(git.get("commit") or MISSING)[:7]] += 1
            dirty += int(git.get("dirty") is True)
    return dict(commits), dirty


def run_info(records: list[dict]) -> dict:
    commits, dirty = commit_summary(records)
    first, last = time_range(records)
    return {
        "configured_models": count_by(records, "model"), "response_models": count_by(records, "model_name"),
        "output_modes": count_by(records, "output_mode"), "orders": count_by(records, "order"),
        "commits": commits, "dirty_records": dirty, "first_started_at": first, "last_finished_at": last,
    }


def drift_of(records: list[dict]) -> dict[str, dict[str, int]]:
    """同じ変種の記録で、値がそろっていないフィールド → {値（sha は 12 文字）: 件数}。"""
    out = {}
    for name in DRIFT_FIELDS:
        counts = Counter(str(r[name]) for r in records if r.get(name) not in (None, ""))
        if len(counts) > 1:
            out[name] = {(v[:12] if name.endswith("_sha256") else v): n for v, n in counts.items()}
    return out


def labels_drift(records: list[dict], es: EvalSet) -> dict[str, int]:
    """記録の labels_sha256 が今のラベルファイルの sha256 と違う記録の数（ラベルが実行後に変わった）。"""
    out = {}
    for who in LABEL_OWNERS:
        now = es.labels_of(who).sha256
        n = sum(
            1 for r in records
            if isinstance(r.get("labels_sha256"), dict) and who in r["labels_sha256"] and r["labels_sha256"][who] != now
        )
        if n:
            out[who] = n
    return out


def key_list(value) -> list:
    """key_order の 1 項目（キー名のリスト）。リストでなければ空にする。"""
    return list(value) if isinstance(value, (list, tuple)) else []


def order_check(records: list[dict]) -> dict:
    """表 E: 順序の操作（理由を先に書かせる／判定を先に書かせる）が、記録の上でどうなっていたか。"""
    order_ok = Counter(kind_key(r.get("order_ok")) for r in records)
    example = None
    for r in records:
        key_order = r.get("key_order")
        if isinstance(key_order, dict) and "top" in key_order:
            example = {"case_id": r.get("case_id"), "trial": r.get("trial"), "top": key_list(key_order.get("top")),
                       "grounding": key_list(key_order.get("grounding"))}
            break
    requests = [r["request"] for r in records if isinstance(r.get("request"), dict)]
    return {
        "records": len(records), "output_modes": count_by(records, "output_mode"), "orders": count_by(records, "order"),
        "order_ok": {key: order_ok.get(key, 0) for key in ("true", "false", "null")},
        "key_order_example": example, "request_missing": len(records) - len(requests),
        "request": {f: dict(Counter(kind_key(req.get(f)) for req in requests)) for f in REQUEST_FIELDS},
    }


def warning_headers_of(response: dict) -> list[str]:
    """応答ヘッダのうち、名前に warning か deprecat（大文字小文字は問わない）を含むものを「名前: 値」にする。"""
    headers = response.get("headers")
    if not isinstance(headers, dict):
        return []
    return [f"{k}: {v}" for k, v in headers.items() if "warning" in str(k).lower() or "deprecat" in str(k).lower()]


def usage_summary(records: list[dict]) -> dict:
    """表 F: トークン・秒・リクエスト数・応答ヘッダの警告・finish_reason。status は問わず全記録を数える。"""
    usages = [r["usage"] for r in records if isinstance(r.get("usage"), dict)]
    details = [u["details"] for u in usages if isinstance(u.get("details"), dict)]
    responses = [r["response"] for r in records if isinstance(r.get("response"), dict)]
    flagged = [found for found in map(warning_headers_of, responses) if found]  # 警告ヘッダがあった記録ごとの「名前: 値」
    return {
        "records": len(records),
        "thoughts_tokens": number_stats([d.get("thoughts_tokens") for d in details]),
        "output_tokens": number_stats([u.get("output_tokens") for u in usages]),
        "input_tokens": number_stats([u.get("input_tokens") for u in usages]),
        "duration_s": number_stats([r.get("duration_s") for r in records]),
        "multi_request_records": sum(1 for u in usages if isinstance(u.get("requests"), int) and u["requests"] >= 2),
        "warning_header_records": len(flagged), "warning_header_examples": [h[0] for h in flagged[:3]],
        "finish_reason": dict(Counter(kind_key(resp.get("finish_reason")) for resp in responses)),
    }


def within_variant(vd: VariantData) -> dict:
    """変種内の揺れ: 3 試行の判定が全部同じでないセル。試行が欠けて比べられないセルは数だけ出す。"""
    if vd.variant.trials < 2:
        return {"applicable": False, "reason_code": "single_trial", "reason": "試行が 1 回だけなので、揺れは測れません",
                "compared_cells": 0, "incomplete_cells": 0, "unstable_count": 0, "unstable_cells": []}
    compared, incomplete, unstable = 0, 0, []
    for (case_id, crit), seq in vd.cells.items():
        verdict = is_unstable(seq)
        if verdict is None:
            incomplete += 1
            continue
        compared += 1
        if verdict:
            unstable.append({"case_id": case_id, "criterion": crit, "verdicts": seq})
    return {"applicable": True, "reason_code": None, "reason": None, "compared_cells": compared, "incomplete_cells": incomplete,
            "unstable_count": len(unstable), "unstable_cells": unstable}


def versus_baseline(vd: VariantData, base: VariantData | None) -> dict:
    """V0 との比較: 両方 3 試行そろったセルの d を出し、変わった・傾向のセルを一覧にする（揺れは数だけ）。"""

    def skip(code: str, reason: str) -> dict:
        return {"applicable": False, "reason_code": code, "reason": reason, "compared_cells": 0, "counts": {}, "cells": []}

    if base is None:
        return skip("no_baseline", f"label が {BASELINE_LABEL} の変種がありません")
    if vd is base:
        return skip("is_baseline", "基準の変種です")
    if vd.variant.level == "impl":
        return skip("impl", "実装（impl）は比べる相手がないので、並べて出すだけです")
    if vd.evalset.cases.path != base.evalset.cases.path:
        return skip("other_evalset", "V0 と評価セットが違います")
    counts: Counter = Counter()
    cells = []
    for key, seq in vd.cells.items():
        d = shift_distance(base.cells.get(key, []), seq)
        if d is None:
            counts["not_compared"] += 1
            continue
        kind = classify_shift(d)
        counts[kind] += 1
        if kind != "fluctuation":
            cells.append({"case_id": key[0], "criterion": key[1], "distance": d, "class": kind,
                          "baseline": base.cells[key], "variant": seq})
    compared = sum(counts[k] for k in SHIFT_NAMES)
    return {"applicable": True, "reason_code": None, "reason": None, "compared_cells": compared,
            "counts": {k: counts.get(k, 0) for k in (*SHIFT_NAMES, "not_compared")}, "cells": cells}


def variant_result(an: Analysis, vd: VariantData) -> dict:
    v, es = vd.variant, vd.evalset
    records = vd.records
    expected = len(es.case_list)
    trials = {}
    for k in vd.trial_range:
        verdicts = vd.usable[k]
        labels = {}
        for who in LABEL_OWNERS:
            loaded = es.labels_of(who)
            labels[who] = agreement_stats(es.case_list, verdicts, loaded.data) if verdicts and loaded.ok else None
            if labels[who] and not labels[who]["consistent"]:
                an.warnings.append({
                    "kind": "inconsistent", "variant": v.name,
                    "message": f"{v.title} t{k}: 総合の数え直しが app.judge.summarize の値と食い違いました（report.py の不具合の可能性）"})
        trials[str(k)] = {"expected": expected, "usable": len(verdicts), "dropped": expected - len(verdicts), "labels": labels}
    cells: dict[str, dict] = {}
    for (case_id, crit), seq in vd.cells.items():
        cells.setdefault(case_id, {})[crit] = seq
    return {
        "label": v.label, "level": v.level, "name": v.name, "evalset": es.index,
        "settings": {"judge_model": v.judge_model, "rubric": v.rubric, "output": v.output, "order": v.order,
                     "cases": v.cases, "labels": v.labels, "instructor_labels": v.instructor_labels, "trials": v.trials},
        "labels": {"mine": es.mine.info(), "instructor": es.instructor.info()},
        "records": {
            "found": len(records), "expected": expected * v.trials, "usable": sum(len(t) for t in vd.usable.values()),
            "status": count_by(records, "status"), "dry_run": sum(int(r.get("dry_run") is True) for r in records),
            "unreadable": [{"case_id": c, "trial": k, "error": e} for (c, k), e in sorted(vd.recordset.unreadable.items())],
            "missing": vd.missing, "extra": vd.extra,
        },
        "run": run_info(records), "trials": trials, "cells": cells,
        "within_variant": within_variant(vd), "vs_baseline": versus_baseline(vd, an.baseline),
        "order_check": order_check(records), "usage": usage_summary(records),
    }


def label_disagreements(an: Analysis) -> dict:
    """表 C: 自分と講師でラベルが割れたセル（app.judge.compare_labels と同じ定義: 値が違うもの）。"""
    es = an.primary
    if es is None or not (es.mine.ok and es.instructor.ok):
        return {"available": False, "cells": []}
    mine, instructor = es.mine.data, es.instructor.data
    cells = []
    for case in es.case_list:
        cid = case["id"]
        if cid not in mine or cid not in instructor:
            continue
        for crit in CRITERIA:
            if mine[cid].get(crit) != instructor[cid].get(crit):
                cells.append({"case_id": cid, "criterion": crit, "mine": mine[cid].get(crit), "instructor": instructor[cid].get(crit)})
    return {"available": True, "evalset": es.index, "cells": cells}


def body_limit(cfg: dict) -> int | None:
    """[checks] の global にある max_body_chars の limit（表 D の「コードの判定」の基準）。"""
    for check in (cfg.get("checks") or {}).get("global") or []:
        if isinstance(check, dict) and check.get("kind") == "max_body_chars":
            try:
                return int(check["limit"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def clarity_comparison(an: Analysis) -> dict:
    """表 D: V3（分かりやすさの基準に字数を書き足した変種）の LLM と、コード（本文の字数）を並べる。"""
    v3_variant = an.variant_by_label(CLARITY_LABEL)
    v0, v3 = an.baseline, (an.vdata[v3_variant.name] if v3_variant else None)
    if v3 is None or v0 is None:
        return {"available": False, "reason": f"label が {BASELINE_LABEL} と {CLARITY_LABEL} の変種がそろっていません", "rows": []}
    if v0.evalset.cases.path != v3.evalset.cases.path or not v0.evalset.cases.ok:
        return {"available": False, "reason": "評価セットを読めない、または V0 と V3 で評価セットが違います", "rows": []}
    limit = body_limit(an.cfg)
    es = v0.evalset
    rows = []
    for case in es.case_list:
        cid = case["id"]
        chars = body_chars(str(case.get("answer") or ""))
        rows.append({
            "case_id": cid, "body_chars": chars, "code": pass_if(chars <= limit) if limit is not None else None,
            "baseline": v0.cells[(cid, "clarity")], "variant": v3.cells[(cid, "clarity")],
            "mine": (es.mine.data.get(cid, {}).get("clarity") if es.mine.ok else None),
            "instructor": (es.instructor.data.get(cid, {}).get("clarity") if es.instructor.ok else None),
        })
    return {"available": True, "reason": None, "limit": limit, "baseline": v0.variant.name, "variant": v3.variant.name,
            "rows": rows, "code_vs_mine": code_vs_labels(rows, "mine"), "code_vs_instructor": code_vs_labels(rows, "instructor")}


def code_vs_labels(rows: list[dict], who: str) -> dict:
    compared = [r for r in rows if r["code"] in PASS_FAIL and r[who] in PASS_FAIL]
    return {"agree": sum(int(r["code"] == r[who]) for r in compared), "compared": len(compared)}


def impl_result(an: Analysis) -> dict | None:
    """表 G: 実装の評価セットに、コードの判定を当てる。"""
    es = an.impl_evalset
    if es is None:
        return None
    cases = es.case_list
    labels = es.mine.data if es.mine.ok else None
    rows = code_check_rows(an.cfg.get("checks") or {}, cases, an.generates, labels)
    generate = {case["id"]: generate_state(an.generates.get(case["id"]), case) for case in cases}
    human = {case["id"]: {k: (labels[case["id"]].get(k) if labels and case["id"] in labels else None) for k in CRITERIA} for case in cases}
    return {
        "variants": [v.name for v in es.variants], "evalset": es.index, "cases": es.cases.path, "labels": es.mine.path,
        "case_ids": [c["id"] for c in cases], "human_labels": human, "generate": generate,
        "code_checks": rows, "code_groups": group_code_checks(rows),
    }


def generate_state(gen, case: dict) -> dict:
    """generate の記録の状態。usable = 手続きの判定に使える（status が ok で、question と answer が評価セットの件と同じ）。

    回答を差し替えたり、質問を直したりしたあとの古い記録で、tool_called などを pass にしないため。problems は使えない理由
    （missing・status:<値>・question_differs・answer_differs）。question・answer の比べ方は、前後の空白を無視する。
    """
    if not isinstance(gen, dict):
        return {"status": "missing", "usable": False, "problems": ["missing"], "question_matches_cases": None, "answer_matches_cases": None}
    status = gen.get("status")
    if status != "ok":
        return {"status": status, "usable": False, "problems": [f"status:{status}"], "question_matches_cases": None, "answer_matches_cases": None}
    question_same = str(gen.get("question") or "").strip() == str(case.get("question") or "").strip()
    answer_same = str(gen.get("answer") or "").strip() == str(case.get("answer") or "").strip()
    problems = [name for name, same in (("question_differs", question_same), ("answer_differs", answer_same)) if not same]
    return {"status": status, "usable": not problems, "problems": problems,
            "question_matches_cases": question_same, "answer_matches_cases": answer_same}


def problem_names(problems: list[str]) -> str:
    """generate の記録を使えない理由を、警告に書く形にする。"""
    return "・".join("status=" + p[len("status:"):] if p.startswith("status:") else PROBLEM_NAMES.get(p, p) for p in problems)


def add_warning(an: Analysis, kind: str, message: str, **extra) -> None:
    an.warnings.append({"kind": kind, "message": one_line(message), **extra})


def collect_file_warnings(an: Analysis) -> None:
    """評価セットとラベルのファイルが読めない、という警告。同じファイルは 1 回だけ。"""
    seen: set = set()
    for es in an.evalsets:
        names = "・".join(v.title for v in es.variants)
        if not es.cases.ok:
            add_warning(an, "cases_unavailable", f"評価セット `{es.cases.path or '（未設定）'}` を読めません（{es.cases.describe()}）。{names} は集計できません",
                        path=es.cases.path, state=es.cases.state)
        for who in LABEL_OWNERS:
            loaded = es.labels_of(who)
            unset_is_normal = who == "instructor"  # 講師のラベルは、実装では最初から無い
            if loaded.ok or (loaded.state == "unset" and unset_is_normal) or (who, loaded.path) in seen:
                continue
            seen.add((who, loaded.path))
            add_warning(an, "labels_unavailable", f"{OWNER_NAMES[who]}のラベル `{loaded.path or '（未設定）'}` を読めません（{loaded.describe()}）。{names} の一致率は出せません",
                        owner=who, path=loaded.path, state=loaded.state)


def collect_variant_warnings(an: Analysis) -> None:
    for v in an.variants:
        vd = an.vdata[v.name]
        records = vd.records
        if not records:
            add_warning(an, "no_records", f"{v.title}: 記録が 1 件もありません（judge/{v.name}/）", variant=v.name)
            continue
        dry = sum(int(r.get("dry_run") is True) for r in records)
        if dry:
            add_warning(an, "dry_run", f"{v.title}: dry_run の記録が {dry} 件混ざっています（本物の採点ではありません）", variant=v.name, count=dry)
        for name, counts in drift_of(records).items():
            add_warning(an, "variant_drift", f"{v.title}: 同じ変種の記録で {name} がそろっていません（{fmt_counts(counts)}）", variant=v.name, field=name, values=counts)
        for who, n in labels_drift(records, vd.evalset).items():
            add_warning(an, "labels_changed", f"{v.title}: 記録の labels_sha256（{OWNER_NAMES[who]}）が今のラベルファイルと違う記録が {n} 件あります（ラベルが実行後に変わりました）",
                        variant=v.name, owner=who, count=n)
        if vd.missing:
            items = [f"{m['case_id']} t{m['trial']}（{REASON_NAMES.get(m['reason'], m['reason'].replace('status:', 'status='))}）" for m in vd.missing]
            add_warning(an, "missing", f"{v.title}: 欠け {len(vd.missing)} 件を集計から外しました: {listing(items)}", variant=v.name, count=len(vd.missing))
        for reason, what in (("case_changed", "評価セットの件（case_sha256）"), ("prompt_changed", "採点者に渡した文（prompt_sha256。参照の用語集を含む）")):
            changed = [m for m in vd.missing if m["reason"] == reason]
            if changed:
                items = [f"{m['case_id']} t{m['trial']}" for m in changed]
                add_warning(an, reason, f"{v.title}: 記録のあとで{what}が変わりました（{REASON_NAMES[reason]}）。別の件への判定を今の件の判定にしないよう、{len(items)} 件を集計から外しました: {listing(items)}",
                            variant=v.name, count=len(items), cases=sorted({m["case_id"] for m in changed}))
        if vd.recordset.unreadable:
            items = [f"{c} t{k}" for c, k in sorted(vd.recordset.unreadable)]
            add_warning(an, "unreadable", f"{v.title}: 読めない記録が {len(items)} 件あります: {listing(items)}", variant=v.name, count=len(items))
        if vd.extra:
            items = [f"{m['case_id']} t{m['trial']}" for m in vd.extra]
            add_warning(an, "extra", f"{v.title}: 評価セットにない件・予定の試行数を超える記録が {len(items)} 件あります（集計に含めていません）: {listing(items)}",
                        variant=v.name, count=len(items))


def collect_checks_cases_warnings(an: Analysis, es: EvalSet, checks_cfg: dict) -> None:
    """[checks.cases] の id と、実装の評価セットの id が合っているか。

    tools/annotate.py の「作問」は id を 01 から振るので、conditions.toml の m01〜と合わないと、その件の個別の判定が黙って抜ける。
    """
    table = checks_cfg.get("cases")
    if not es.cases.ok or not isinstance(table, dict) or not table:
        return
    ids = [c["id"] for c in es.case_list]
    unknown = [key for key in table if key not in ids]
    uncovered = [i for i in ids if i not in table]
    hint = "（tools/annotate.py の「作問」は id を 01 から振ります。conditions.toml の [checks.cases] の id か、評価セットの id をそろえてください）"
    if unknown:
        add_warning(an, "checks_cases_unknown", f"実装: [checks.cases] にあって、評価セット `{es.cases.path}` に無い id があります（その件の個別の判定は使われません）: {listing(unknown)} {hint}", ids=unknown)
    if uncovered:
        add_warning(an, "checks_cases_uncovered", f"実装: 評価セット `{es.cases.path}` にあって、[checks.cases] に無い id があります（その件には [checks] の global の判定だけを当てます）: {listing(uncovered)} {hint}", ids=uncovered)


def collect_impl_warnings(an: Analysis) -> None:
    es = an.impl_evalset
    if es is None:
        return
    for other in an.evalsets:
        if other.is_impl and other is not es:
            names = "・".join(v.title for v in other.variants)
            add_warning(an, "impl_evalset_ignored", f"実装の評価セットが複数あります。コードの判定は最初の `{es.cases.path}` だけに当てます（{names} は並べて出すだけ）")
    checks_cfg = an.cfg.get("checks") or {}
    absent, differs, question_differs = [], [], []
    for case in es.case_list:
        state = generate_state(an.generates.get(case["id"]), case)
        needs_generate = any(c.get("type") == "procedure" or c.get("kind") in GENERATE_KINDS for _, c in checks_for_case(checks_cfg, case["id"]))
        if needs_generate and not state["usable"]:
            absent.append(f"{case['id']}（{problem_names(state['problems'])}）")
        if state["answer_matches_cases"] is False:
            differs.append(case["id"])
        if state["question_matches_cases"] is False:
            question_differs.append(case["id"])
    if absent:
        add_warning(an, "generate_unavailable", f"実装: generate の記録が使えない件があります（手続きの判定は n/a）: {listing(absent)}", cases=absent)
    if differs:
        add_warning(an, "generate_answer_differs", f"実装: generate の記録の answer と評価セットの answer が違う件があります（評価セットを後から書き換えた可能性）: {listing(differs)}", cases=differs)
    if question_differs:
        add_warning(an, "generate_question_differs", f"実装: generate の記録の question と評価セットの question が違う件があります（質問を後から書き換えた可能性。その件の手続きの判定は n/a）: {listing(question_differs)}", cases=question_differs)
    collect_checks_cases_warnings(an, es, checks_cfg)
    kinds = {c.get("kind") for case in es.case_list for _, c in checks_for_case(checks_cfg, case["id"])}
    unknown = sorted(str(k) for k in kinds - {"lecture_mark", "max_body_chars", "answer_regex", "answer_not_regex", "no_instructions_leak", "no_code_def", *GENERATE_KINDS})
    if unknown:
        add_warning(an, "unknown_check", f"実装: 知らないチェックの kind があります（n/a にしました）: {'、'.join(unknown)}", kinds=unknown)


def analyze(root: Path, cfg: dict, conditions: Path, out_dir: Path) -> Analysis:
    variants = load_variants(cfg)
    inputs_dir = Path(out_dir) / DRY_RUN_INPUTS  # compare.py run --dry-run が書く（本物の results には無い）。あれば、そこにあるファイルを先に使う
    evalsets = load_evalsets(root, variants, inputs_dir if inputs_dir.is_dir() else None)
    evalset_of = {v.name: es for es in evalsets for v in es.variants}
    an = Analysis(
        root=Path(root), conditions=Path(conditions), out_dir=Path(out_dir), cfg=cfg, variants=variants, evalsets=evalsets,
        vdata={v.name: build_variant_data(v, evalset_of[v.name], load_judge_records(out_dir, v.name)) for v in variants},
        generates=load_generates(out_dir),
        dry_inputs=sorted({loaded.file.name for es in evalsets for loaded in (es.cases, es.mine) if loaded.dry_run_input and loaded.file}),
    )
    if an.baseline is None:
        add_warning(an, "no_baseline", f"label が {BASELINE_LABEL} の変種がありません（V0 との比較はしません）")
    collect_file_warnings(an)
    collect_variant_warnings(an)
    collect_impl_warnings(an)
    an.results = build_results(an)
    return an


def build_results(an: Analysis) -> dict:
    variants = {v.name: variant_result(an, an.vdata[v.name]) for v in an.variants}  # inconsistent の警告もここで足される
    evalsets = [
        {"index": es.index, "cases": es.cases.info(), "labels": {who: es.labels_of(who).info() for who in LABEL_OWNERS},
         "variants": [v.name for v in es.variants], "case_ids": [c["id"] for c in es.case_list], "is_impl": es.is_impl,
         "human_labels": {who: ({cid: {k: row.get(k) for k in CRITERIA} for cid, row in es.labels_of(who).data.items()}
                                if es.labels_of(who).ok else None) for who in LABEL_OWNERS}}
        for es in an.evalsets
    ]
    return {
        "schema": 1, "kind": "judge_summary", "conditions": display_path(an.conditions, an.root), "out_dir": display_path(an.out_dir, an.root),
        "baseline": {"label": BASELINE_LABEL, "variant": an.baseline.variant.name if an.baseline else None},
        "definitions": {
            "agree_rate_a": "agree / compared（LLM が unsure の項目と、人のラベルが pass・fail でない項目は分母から外す。app.judge.summarize の定義）",
            "coverage": "compared / total", "agree_rate_b": "agree / total（要確認も不一致に数える）",
            "always_pass": "人のラベルが pass の件数 / pass か fail の件数",
            "shift": f"両方 {COMPARE_TRIALS} 試行そろったセルで、pass・fail・unsure の回数の差の最大値 d。"
                     f"d = {CHANGED_DISTANCE} は changed（変わった）、d = {TREND_DISTANCE} は trend（傾向）、それ以下は fluctuation（揺れ）",
        },
        "dry_run_inputs": {"used": bool(an.dry_inputs), "dir": f"{display_path(an.out_dir, an.root)}/{DRY_RUN_INPUTS}", "files": an.dry_inputs},
        "warnings": an.warnings, "evalsets": evalsets, "variants": variants,
        "label_disagreements": label_disagreements(an), "clarity": clarity_comparison(an), "impl": impl_result(an),
    }


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------


def trial_entry(res: dict, k: int) -> dict:
    return res["trials"][str(k)]


def stat_of(entry: dict, who: str, crit: str) -> dict | None:
    """試行の集計から、持ち主 who の基準 crit の行を取り出す。集計できていなければ None。"""
    stats = (entry.get("labels") or {}).get(who)
    if not stats:
        return None
    return stats["overall"] if crit == OVERALL else stats["criteria"][crit]


def series(rows: list[dict | None], fn) -> str:
    """試行ごとの値を "t1,t2,t3" の形に並べる。集計できなかった試行は -。"""
    return ",".join(fn(r) if r else MISSING for r in rows)


def agreement_cells(rows: list[dict | None]) -> list[str]:
    return [
        series(rows, lambda r: str(r["total"])), series(rows, lambda r: str(r["agree"])),
        series(rows, lambda r: str(r["disagree"])), series(rows, lambda r: str(r["unsure"])),
        series(rows, lambda r: str(r["human_unlabeled"])), series(rows, lambda r: str(r["missed"])),
        series(rows, lambda r: str(r["false_alarm"])), series(rows, lambda r: f"{r['agree']}/{r['compared']}"),
        series(rows, lambda r: f"{r['compared']}/{r['total']}"), series(rows, lambda r: f"{r['agree']}/{r['total']}"),
        series(rows, lambda r: f"{r['always_pass']['pass']}/{r['always_pass']['labeled']}"),
    ]


def render_intro(an: Analysis) -> list[str]:
    base = an.baseline.variant.title if an.baseline else "なし"
    return [
        f"結果の置き場所: `{display_path(an.out_dir, an.root)}`　条件: `{display_path(an.conditions, an.root)}`　基準の変種: {base}",
        "",
        "読み方: 判定は p = pass・f = fail・u = unsure（LLM が決めず、人が見る要確認）・`-` = 欠け。試行は t1,t2,t3 の順に並べる。"
        f"V0 との比較は、同じ評価セットの V0 と両方 {COMPARE_TRIALS} 試行そろったセルだけで、pass・fail・unsure それぞれの回数の差の最大値 d を見る"
        f"（d = {CHANGED_DISTANCE} は「変わった」、d = {TREND_DISTANCE} は「傾向」、d が 1 以下は「揺れ」）。"
        "試行が 3 回だけなので、結果の違いは件数で読み、確率の言葉は使わない。",
    ]


def render_dry_run_notice(an: Analysis) -> list[str]:
    """予行の入力（<out>/dry_run_inputs）を使ったときだけ、summary.md の先頭に書く注意。"""
    if not an.dry_inputs:
        return []
    files = "・".join(an.dry_inputs)
    return [
        f"> **予行の入力（dry_run_inputs）を使った**: `{display_path(an.out_dir, an.root)}/{DRY_RUN_INPUTS}` の {files}。"
        "本物のラベル・評価セットではなく、予行の仮のラベルと生成済みの評価セットで集計しています（本物の採点ではありません）。",
        "",
    ]


def render_warnings(an: Analysis) -> list[str]:
    return [f"- {w['message']}" for w in an.warnings] or ["なし"]


def render_runs(an: Analysis) -> list[str]:
    headers = ["label", "変種", "level", "モデル（設定）", "モデル（応答の model_name）", "記録（読めた/予定）", "status", "試行", "git commit", "日時"]
    rows = []
    for v in an.variants:
        res = an.results["variants"][v.name]
        run, rec = res["run"], res["records"]
        commits = fmt_counts(run["commits"]) + (f"；dirty の記録 {run['dirty_records']} 件" if run["dirty_records"] else "")
        span = f"{run['first_started_at']} 〜 {run['last_finished_at']}" if run["first_started_at"] else MISSING
        rows.append([v.label, v.name, v.level, fmt_counts(run["configured_models"]) if rec["found"] else f"（設定 {v.judge_model}）",
                     fmt_model_counts(run["response_models"]), f"{rec['found']}/{rec['expected']}", fmt_counts(rec["status"]),
                     v.trials, commits, span])
    return md_table(headers, rows)


AGREEMENT_HEADERS = ["変種", "基準", "欠け", "件数", "一致", "不一致", "要確認", "人の欠け", "見逃し", "過検出",
                     "一致率 A", "coverage", "一致率 B", "常に pass"]
AGREEMENT_NOTE = [
    "t1,t2,t3 は試行順。欠け = ok でない記録の件を集計から外した数。件数 = 集計した件（ラベルがある件）。"
    "要確認 = LLM が unsure にした件、人の欠け = 人のラベルが pass・fail でない件。見逃し = LLM pass・人 fail、過検出 = LLM fail・人 pass。"
    "一致率 A = 一致 / 比べた件（要確認と人の欠けは外す）、coverage = 比べた件 / 件数、一致率 B = 一致 / 件数（要確認も不一致に数える）、"
    "常に pass = 人のラベルが pass の件 / pass か fail の件。",
    "",
]


def render_agreement(an: Analysis, who: str) -> list[str]:
    """表 A（自分のラベル）と表 A′（講師のラベル）。講師の表には、講師のラベルがある変種だけを並べる。"""
    note = AGREEMENT_NOTE if who == "mine" else ["読み方は表A と同じ（ラベルが講師のもの）。講師のラベルがある変種だけを並べる。", ""]
    return note + render_agreement_table(an, who)


def render_agreement_table(an: Analysis, who: str) -> list[str]:
    rows = []
    for v in an.variants:
        if who == "instructor" and not v.instructor_labels:
            continue
        res = an.results["variants"][v.name]
        entries = [trial_entry(res, k) for k in range(1, v.trials + 1)]
        info = res["labels"][who]
        if not any(e["usable"] for e in entries):
            rows.append([v.title, "（集計できる記録がありません）"] + [MISSING] * (len(AGREEMENT_HEADERS) - 2))
        elif info["state"] != "ok":
            rows.append([v.title, f"（{OWNER_NAMES[who]}のラベル: {STATE_NAMES[info['state']]}）"] + [MISSING] * (len(AGREEMENT_HEADERS) - 2))
        else:
            dropped = ",".join(str(e["dropped"]) for e in entries)
            for crit in (*CRITERIA, OVERALL):
                rows.append([v.title, criterion_name(crit), dropped, *agreement_cells([stat_of(e, who, crit) for e in entries])])
    if not rows:
        return [f"{OWNER_NAMES[who]}のラベルがある変種はありません。"]
    return md_table(AGREEMENT_HEADERS, rows)


def shift_map(res: dict) -> dict[tuple[str, str], str]:
    return {(c["case_id"], c["criterion"]): c["class"] for c in res["vs_baseline"]["cells"]}


def case_lookup(an: Analysis, v: Variant) -> dict[str, dict]:
    """変種の評価セットの件を id で引く（評価セットが複数あっても、件の見出しを取り違えない）。"""
    return {c["id"]: c for c in an.vdata[v.name].evalset.case_list}


def render_judgment_table(an: Analysis, es: EvalSet) -> list[str]:
    show_instructor = es.instructor.state != "unset"
    headers = ["件", "基準", "自分"] + (["講師"] if show_instructor else []) + [v.title for v in es.variants]
    rows = []
    for case in es.case_list:
        cid = case["id"]
        for crit in CRITERIA:
            row = [case_title(case), criterion_name(crit), label_mark(es.mine.data.get(cid, {}).get(crit)) if es.mine.ok else MISSING]
            if show_instructor:
                row.append(label_mark(es.instructor.data.get(cid, {}).get(crit)) if es.instructor.ok else MISSING)
            for v in es.variants:
                res = an.results["variants"][v.name]
                text = marks(res["cells"][cid][crit])
                row.append(decorate(text, shift_map(res).get((cid, crit))))
            rows.append(row)
    return md_table(headers, rows)


def render_judgments(an: Analysis) -> list[str]:
    out = ["p = pass・f = fail・u = unsure・`-` = 欠け（人のラベルが無いときも `-`）。変種の列は試行順。"
           "V0 との比較で **太字** は「変わった」（d = 3）、*斜体* は「傾向」（d = 2）。"]
    ordered = sorted(an.evalsets, key=lambda es: (es is not an.primary, es.is_impl, es.index))
    for es in ordered:
        title = f"評価セット `{es.cases.path}`" + ("（実装）" if es.is_impl else "")
        out += ["", f"### {title}", ""]
        if not es.cases.ok:
            out.append(f"評価セットを読めません（{es.cases.describe()}）。")
        else:
            out += render_judgment_table(an, es)
    return out


def render_label_disagreements(an: Analysis) -> list[str]:
    info = an.results["label_disagreements"]
    if not info["available"]:
        return ["自分と講師のラベルがそろっていないので、比べられません。"]
    if not info["cells"]:
        return ["自分と講師でラベルが割れたセルはありません。"]
    es = an.primary
    headers = ["件", "基準", "自分", "講師"] + [v.title for v in es.variants]
    rows = []
    cases = {c["id"]: c for c in es.case_list}
    for c in info["cells"]:
        rows.append([case_title(cases[c["case_id"]]), criterion_name(c["criterion"]), label_mark(c["mine"]), label_mark(c["instructor"])]
                    + [marks(an.results["variants"][v.name]["cells"][c["case_id"]][c["criterion"]]) for v in es.variants])
    return [f"割れたセル {len(info['cells'])} 件（値が違うもの。app.judge --compare と同じ定義）。", ""] + md_table(headers, rows)


def render_clarity(an: Analysis) -> list[str]:
    info = an.results["clarity"]
    if not info["available"]:
        return [info["reason"]]
    limit = info["limit"]
    v0, v3 = an.vdata[info["baseline"]].variant, an.vdata[info["variant"]].variant
    rule = f"{limit} 字以内なら pass" if limit is not None else "（[checks] に max_body_chars が無いので、コードの判定は出せません）"
    out = [f"body_chars = 回答の本文の字数（コードブロックを除く）。コードの判定は本文が {rule}。分かりやすさの判定は試行順。", ""]
    headers = ["件", "body_chars", "コードの判定", f"{v0.title} の分かりやすさ", f"{v3.title} の分かりやすさ", "自分", "講師"]
    cases = case_lookup(an, v0)
    rows = [[case_title(cases[r["case_id"]]), r["body_chars"], r["code"] or MISSING, marks(r["baseline"]), marks(r["variant"]),
             label_mark(r["mine"]), label_mark(r["instructor"])] for r in info["rows"]]
    out += md_table(headers, rows)
    mine, instructor = info["code_vs_mine"], info["code_vs_instructor"]
    out += ["", f"コードの判定と分かりやすさのラベルの一致: 自分 {mine['agree']}/{mine['compared']}、講師 {instructor['agree']}/{instructor['compared']}。"]
    return out


def render_order(an: Analysis) -> list[str]:
    out = ["注記: ツール出力（tool）の key_order は受け取った dict の順で、生成順の証拠にはならない（参考値）。native は、モデルが書いた JSON 文字列の順。", ""]
    headers = ["変種", "output_mode", "order", "記録数", "order_ok", "key_order の例（最初の 1 件）", "request なし",
               "response_json_schema", "property_ordering", "temperature", "thinking_config", "tool_mode"]
    rows = []
    for v in an.variants:
        o = an.results["variants"][v.name]["order_check"]
        example = o["key_order_example"]
        shown = f"top: {', '.join(map(str, example['top'])) or MISSING} / grounding: {', '.join(map(str, example['grounding'])) or MISSING}" if example else MISSING
        modes = fmt_counts(o["output_modes"]) if o["records"] else f"（設定 {v.output}）"
        orders = fmt_counts(o["orders"]) if o["records"] else f"（設定 {v.order}）"
        rows.append([v.title, modes, orders, o["records"], f"true {o['order_ok']['true']} / false {o['order_ok']['false']} / null {o['order_ok']['null']}", shown,
                     o["request_missing"], *[fmt_counts(o["request"][f]) for f in REQUEST_FIELDS]])
    return out + md_table(headers, rows)


def render_usage(an: Analysis) -> list[str]:
    out = ["各列は 平均 (最小〜最大)。status を問わず、読めた記録すべてを数える。警告ヘッダ = 応答ヘッダの名前に warning か deprecat（大文字小文字は問わない）を含むもの。", ""]
    headers = ["変種", "記録数", "thoughts_tokens", "output_tokens", "input_tokens", "duration_s", "requests が 2 以上", "警告ヘッダのある記録", "警告ヘッダの例", "finish_reason"]
    rows = []
    for v in an.variants:
        u = an.results["variants"][v.name]["usage"]
        rows.append([v.title, u["records"], fmt_stats(u["thoughts_tokens"]), fmt_stats(u["output_tokens"]), fmt_stats(u["input_tokens"]),
                     fmt_stats(u["duration_s"], 2, 2), u["multi_request_records"], u["warning_header_records"],
                     "; ".join(one_line(h, 80) for h in u["warning_header_examples"]) or MISSING, fmt_counts(u["finish_reason"])])
    return out + md_table(headers, rows)


def render_impl(an: Analysis) -> list[str]:
    impl = an.results["impl"]
    es = an.impl_evalset
    if impl is None or es is None:
        return ["実装（level = \"impl\"）の変種が条件ファイルにありません。"]
    variants = [an.vdata[name].variant for name in impl["variants"]]
    out = [f"評価セット `{impl['cases']}`、自分のラベル `{impl['labels'] or '（未設定）'}`。コードの判定は pass / fail / n/a（generate の記録が使えない件の手続きの判定）。"
           "I0 などの判定は p,f,u を試行順に並べたもの。", ""]
    by_case: dict[str, list[dict]] = {}
    for row in impl["code_checks"]:
        by_case.setdefault(row["case_id"], []).append(row)
    headers = ["件", "コードの判定", "自分のラベル"] + [f"{v.title} の判定" for v in variants]
    rows = []
    for case in es.case_list:
        cid = case["id"]
        code = "；".join(f"{r['label']}: {r['result']}" for r in by_case.get(cid, [])) or MISSING
        mine = "／".join(f"{NAMES[k]} {label_mark(impl['human_labels'][cid][k])}" for k in CRITERIA)
        judged = ["／".join(f"{NAMES[k]} {marks(an.results['variants'][v.name]['cells'][cid][k])}" for k in CRITERIA) for v in variants]
        rows.append([case_title(case), code, mine, *judged])
    out += md_table(headers, rows)
    out += ["", "### output・aux の判定と、自分のラベルとの一致（判定ごと）", ""]
    group_rows = []
    for g in impl["code_groups"]:
        results = "、".join(f"{cid} {res}" for cid, res in g["results"].items())
        if g["counts_agreement"]:
            agree = f"{g['agree']}/{g['compared']}"
            diffs = "、".join(f"{d['case_id']}（コード {d['code']}・自分 {d['human']}）" for d in g["disagreements"]) or "なし"
        else:
            agree, diffs = "対象外", MISSING
        group_rows.append([g["label"], g["type"] or MISSING, g["kind"], NAMES.get(g["criterion"], g["criterion"]) or MISSING, results, agree, diffs])
    out += md_table(["判定", "type", "kind", "基準", "判定（件ごと）", "一致/比べた", "食い違った件"], group_rows) if group_rows else ["コードの判定が [checks] にありません。"]
    return out


def render_fluctuation(an: Analysis) -> list[str]:
    out = [f"セル = 件 × 基準。変種内の揺れ = 3 試行の判定が全部同じでないセル（欠けがあって比べられないセルは数えない）。"
           f"V0 との比較は、両方 {COMPARE_TRIALS} 試行そろったセルだけ。", ""]
    headers = ["変種", "変種内: 比べたセル", "変種内: 揺れたセル", "変種内: 比べられないセル", "V0 比: 比べたセル", "変わった", "傾向", "揺れ", "V0 比: 比べられないセル"]
    rows = []
    for v in an.variants:
        res = an.results["variants"][v.name]
        w, b = res["within_variant"], res["vs_baseline"]
        counts = b["counts"]
        within = [w["compared_cells"], w["unstable_count"], w["incomplete_cells"]] if w["applicable"] else [f"（{SKIP_NAMES[w['reason_code']]}）", MISSING, MISSING]
        versus = [b["compared_cells"], *(counts[k] for k in (*SHIFT_NAMES, "not_compared"))] if b["applicable"] else [f"（{SKIP_NAMES[b['reason_code']]}）"] + [MISSING] * 4
        rows.append([v.title, *within, *versus])
    out += md_table(headers, rows)
    unstable = [[v.title, case_title(case_lookup(an, v)[c["case_id"]]), criterion_name(c["criterion"]), marks(c["verdicts"])]
                for v in an.variants for c in an.results["variants"][v.name]["within_variant"]["unstable_cells"]]
    out += ["", "### 変種内で揺れたセル", ""]
    out += md_table(["変種", "件", "基準", "判定（試行順）"], unstable) if unstable else ["なし"]
    shifted = [[v.title, case_title(case_lookup(an, v)[c["case_id"]]), criterion_name(c["criterion"]), marks(c["baseline"]), marks(c["variant"]), c["distance"], SHIFT_NAMES[c["class"]]]
               for v in an.variants for c in an.results["variants"][v.name]["vs_baseline"]["cells"]]
    out += ["", "### V0 と比べて「変わった」「傾向」のセル", ""]
    out += md_table(["変種", "件", "基準", "V0", "変種", "d", "分類"], shifted) if shifted else ["なし"]
    return out


def render_review_and_mismatches(an: Analysis) -> list[str]:
    out = ["<details>", "<summary>変種 × 試行ごとの要確認（LLM の理由）と不一致（見逃し・過検出と LLM の理由）</summary>", ""]
    any_item = False
    for v in an.variants:
        res = an.results["variants"][v.name]
        for k in range(1, v.trials + 1):
            labels = trial_entry(res, k).get("labels") or {}
            review = (labels.get("mine") or labels.get("instructor") or {}).get("review") or []  # 要確認は LLM が決めたことなので、どちらのラベルでも同じ件
            mismatches = [(who, labels[who]["mismatches"]) for who in LABEL_OWNERS if labels.get(who) and labels[who]["mismatches"]]
            if not review and not mismatches:
                continue
            any_item = True
            out += [f"### {v.title} t{k}", ""]
            if review:
                out += [f"要確認 {len(review)} 件", ""] + [f"- #{r['id']} {r['name']}: {one_line(r['reason'])}" for r in review] + [""]
            for who, items in mismatches:
                out += [f"不一致 {len(items)} 件（{OWNER_NAMES[who]}のラベル）", ""]
                out += [f"- #{m['id']} {m['name']} {m['kind']}（LLM={m['judge']}・{OWNER_NAMES[who]}={m['human']}）: {one_line(m['reason'])}" for m in items] + [""]
    if not any_item:
        out += ["要確認も不一致もありません（または集計できる記録がありません）。", ""]
    return out + ["</details>", ""]


def render_impl_details(an: Analysis) -> list[str]:
    out = ["<details>", "<summary>実装（impl）の各件の質問・回答の全文・ツール呼び出し・承認</summary>", ""]
    es = an.impl_evalset
    if es is None:
        return out + ["実装の変種がありません。", "", "</details>", ""]
    for case in es.case_list:
        gen = an.generates.get(case["id"])
        state = generate_state(gen, case)
        status = "記録なし" if gen is None else f"status={gen.get('status')}"
        out += [f"### {case_title(case)}", "", f"- 質問: {one_line(case.get('question', ''))}", f"- generate: {status}", "- 回答（評価セットの answer）:", ""]
        out += fenced(str(case.get("answer") or ""), "text") + [""]
        if isinstance(gen, dict):
            calls = [c for c in gen.get("tool_calls") or [] if isinstance(c, dict)]
            lines = [f"{c.get('order', i)}. {c.get('name')}({json.dumps(c.get('args'), ensure_ascii=False)}) [{c.get('outcome')}] → {one_line(c.get('result', ''), RESULT_PREVIEW)}"
                     for i, c in enumerate(calls, 1)]
            out += ["- ツール呼び出し:", ""] + (fenced("\n".join(lines), "text") if lines else ["（なし）"]) + [""]
            approvals = [a for a in gen.get("approvals") or [] if isinstance(a, dict)]
            lines = [f"{a.get('tool_name')}({json.dumps(a.get('args'), ensure_ascii=False)}) → {a.get('decision')}" for a in approvals]
            out += ["- 承認:", ""] + (fenced("\n".join(lines), "text") if lines else ["（なし）"]) + [""]
            if not state["usable"]:
                out += [f"（この generate の記録は使えない（{problem_names(state['problems'])}）ので、手続きの判定は n/a です）", ""]
    return out + ["</details>", ""]


def render_details(an: Analysis) -> list[str]:
    return render_review_and_mismatches(an) + [""] + render_impl_details(an)


def render_markdown(an: Analysis) -> str:
    sections = [
        ("warnings", render_warnings(an)), ("runs", render_runs(an)), ("table_a", render_agreement(an, "mine")),
        ("table_a2", render_agreement(an, "instructor")), ("table_b", render_judgments(an)),
        ("table_c", render_label_disagreements(an)), ("table_d", render_clarity(an)), ("table_e", render_order(an)),
        ("table_f", render_usage(an)), ("table_g", render_impl(an)), ("fluctuation", render_fluctuation(an)),
        ("details", render_details(an)),
    ]
    out = [HEADER_COMMENT, f"# {TITLE}", "", *render_dry_run_notice(an), *render_intro(an), ""]
    for key, body in sections:
        out += [f"## {SECTIONS[key]}", "", *body, ""]
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def summarize(root: Path, conditions: Path, out_dir: Path) -> int:
    """summary.md と summary.json を out_dir に書き、書いたパスを表示して 0 を返す。

    読める記録が 1 件も無いときは、表示だけして 1 を返す（ファイルは書かない）。compare.py の summarize から呼ばれる。
    """
    root, conditions, out_dir = Path(root), Path(conditions), Path(out_dir)
    try:
        cfg = load_conditions(conditions)
        an = analyze(root, cfg, conditions, out_dir)
    except ConfigError as e:
        print(e)
        return 1
    if an.n_records == 0:
        print(f"{out_dir / 'judge'} に、集計できる記録がありません（judge/<variant>/<case_id>/t<k>.json）。")
        return 1
    md_path, json_path = out_dir / "summary.md", out_dir / "summary.json"
    write_text_atomic(md_path, render_markdown(an))
    write_text_atomic(json_path, json.dumps(an.results, ensure_ascii=False, indent=2) + "\n")
    print(f"集計しました: {md_path}（と {json_path.name}）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="第4回: judge・generate の記録を集計して、summary.md と summary.json を作る")
    parser.add_argument("--root", type=Path, default=REPO, help="リポジトリ直下（評価セットとラベルのパスの基準）")
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS, help="条件ファイル（conditions.toml）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="記録の置き場所（judge/ と generate/ がある）。summary もここに書く")
    args = parser.parse_args(argv)
    return summarize(args.root, args.conditions, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
