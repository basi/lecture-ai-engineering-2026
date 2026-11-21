#!/usr/bin/env python3
"""第4回 宿題: 採点の条件（rubric・採点のモデル・出力の型・順序）を変えて、LLM-as-a-Judge の採点を比べる実行スクリプト。

何を比べるか（変種 V0〜V6・I0・I1）は conditions.toml に書いてある。
このスクリプトは、それを読んで評価セットの全件を LLM に採点させ、1 件 1 回の結果を results/ に保存する。
一致率などの集計は report.py（summarize）が行う。

使い方（リポジトリの直下で）:
    uv run python lectures/04/compare.py variants                       # 変種ごとの違い（モデル・rubric・指示の末尾・スキーマの順）を表示（API なし）
    uv run python lectures/04/compare.py status                         # 済み／全体と、残りのリクエストの見積もり（API なし）
    uv run python lectures/04/compare.py run --dry-run --trials 1       # 偽のモデルで予行（API なし。結果は一時フォルダ）
    uv run python lectures/04/compare.py run --level exercise --trials 1  # 演習の 2 変種を 1 回ずつ
    uv run python lectures/04/compare.py generate --dry-run             # 実装: 回答の生成を予行（API なし）
    uv run python lectures/04/compare.py generate                       # 実装: 自分の評価セットの回答を、自分のエージェントで作る
    uv run python lectures/04/compare.py summarize                      # report.py で集計する

条件の数え方: 1 条件 = 評価セットの全件 × trials 回。試行番号を外側に回す（全条件の t1 → t2 → t3）。
保存先: results/judge/<variant>/<case_id>/t<k>.json（あれば「済み」。上書きしない。成功したときだけ、最後に書く）。
        書くときは、同じフォルダに .tmp- で始まる固有の名前の一時ファイルを作り、os.replace で置く（残った一時ファイルは消す。
        .tmp- のファイルは結果として数えない）。置く瞬間にもうあれば（同時に 2 つ動かした場合など）、置き換えずに警告して飛ばす。
        止まっても、同じコマンドで続きから実行できる。変種ごとの最初の送信の本文は results/judge/<variant>/request_sample.json。
        回答の生成は results/generate/<case_id>.json（status・dry_run・質問・モデル・agent_sha256 が今と同じときだけ使い回す）。
        予行（--dry-run）は一時フォルダに書く。

実行の前に、API を呼ばずに確かめること（どれかに当たれば終了コード 1 で止まる）:
    1. conditions.toml の誤り（知らないキー・label の重なり・changes の old が rubric にちょうど 1 か所ではない、など）。
       [checks.cases] の id と実装の評価セットの id が合わないときは、止めずに警告を表示する
    2. 結果を左右するファイルとラベルの未コミットの変更（--allow-dirty と --dry-run では飛ばす）
    3. require_rubric_changed の変種で、data/eval/rubric.md が baseline.md のままではないか
    4. ラベルに評価セットの全 id がそろっているか。実装の評価セットの全件に回答が入っているか
    5. 同じ変種の、保存済みのすべての記録（--case で絞っても全件）と、rubric・指示・スキーマ・評価セット・モデルが食い違っていないか。
       今の評価セットにある件については、件（case_sha256）と採点者に渡す文（prompt_sha256。参照の用語集を含む）も（途中で変わった可能性）
    6. 選んだ変種の評価セット（と generate の cases）の id が、一意で、英数字・_・- の 1〜40 文字か（結果のファイル名に使うため）

終了コード: 0 = 正常 / 1 = 止まった（設定の誤り・API エラー・未コミットの変更など）/ 2 = 1 日の上限（PerDay）/ 130 = Ctrl+C。

秘密: 書き出す前に、GEMINI_API_KEY・GOOGLE_API_KEY の値と AIza… の形の文字列を *** に置き換える。
      送信のヘッダ（API キーが入る）は保存しない。環境変数の一覧は記録しない。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import difflib
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import typing
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # 「python lectures/04/compare.py」で動かすと、app/ が見つからないため
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # pydantic-ai の初回バナー（9 行）を出さない

from pydantic import BaseModel, Field, create_model  # noqa: E402
from pydantic_ai import Agent, NativeOutput, StructuredDict, capture_run_messages  # noqa: E402
from pydantic_ai.exceptions import (  # noqa: E402
    AgentRunError,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (  # noqa: E402
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.tools import ToolDefinition  # noqa: E402
from pydantic_ai.usage import UsageLimits  # noqa: E402

from app.judge import (  # noqa: E402
    LLM_CRITERIA,
    CriterionVerdict,
    JudgeResult,
    Verdict,
    _rule_based_judge,
    build_judge,
    build_test_model,
    check_lecture_mark,
    judge_prompt,
    load_labels,
    read_jsonl,
)

CONDITIONS_REL = "lectures/04/conditions.toml"
RESULTS_REL = "lectures/04/results"
DEFAULT_GENERATE_CASES = "data/eval/my_cases.jsonl"
AGENT_FILES = ("app/agent.py", "app/tools.py", "data/glossary.json")  # 回答の生成を左右するファイル。中身をこの順に連結して agent_sha256 にする

LEVELS = ("exercise", "basic", "impl")
OUTPUTS = ("tool", "native")
ORDERS = ("reason_first", "verdict_first")
CRITERIA = [key for key, _ in LLM_CRITERIA]  # grounding, scope, hint, clarity（LLM が判定する基準）
VERDICTS = list(typing.get_args(Verdict))  # pass, fail, unsure
ORDER_KEYS = {"reason_first": ["reason", "verdict"], "verdict_first": ["verdict", "reason"]}
ORDER_NAMES = {"reason_first": "理由→判定", "verdict_first": "判定→理由"}

DEFAULT_KEYS = ("judge_model", "rubric", "output", "order", "cases", "labels", "instructor_labels", "trials")
REQUIRED_DEFAULT_KEYS = ("judge_model", "rubric", "output", "order", "cases", "labels", "trials")
VARIANT_KEYS = (*DEFAULT_KEYS, "label", "level", "description", "changes", "require_rubric_changed")
NATIVE_KEYS = ("instructions_suffix",)
CHANGE_KEYS = ("target", "op", "old", "new")

REQUEST_LIMIT = 10  # generate の 1 件あたりの LLM 呼び出しの上限（ループの暴走を防ぐ。承認の 2 回目の run と合計）
RATE_LIMIT_RETRIES = 5  # 429 の再試行の回数
RATE_LIMIT_DEFAULT_WAIT = 60  # 429 で待ち時間が分からないときの秒数
SERVER_ERROR_WAITS = (30, 60, 120)  # 503・500・タイムアウトの再試行の前に待つ秒数
# GoogleProvider が自分で作る client（providers/google.py の _create_google_httpx2_client）と同じ timeout
GOOGLE_HTTP_TIMEOUT = 20 * 60
GOOGLE_CONNECT_TIMEOUT = 5
JUDGE_REQUESTS = 1  # 採点 1 回あたりのリクエスト数の見積もり
GENERATE_REQUESTS = 3  # 回答の生成 1 件あたりのリクエスト数の見積もり（承認待ちで 2 回に分かれることがある）
GUARDED_PATHS = [
    "lectures/04/compare.py",
    "lectures/04/report.py",
    "lectures/04/conditions.toml",
    "lectures/04/rubrics/",
    "app/judge.py",
    "app/agent.py",
    "app/tools.py",
    "app/config.py",
    "data/glossary.json",
    "data/eval/",
]
GENERATE_GUARDED_PATHS = ["app", "data/glossary.json"]  # + 評価セット（--cases）
DAILY_LIMIT_MESSAGE = (
    "1 日の上限（PerDay）に達しました。明日（日本時間 17 時以降）、同じコマンドで続きから実行してください。"
    "（済みの結果は飛ばされます）"
)


class ConfigError(Exception):
    """conditions.toml などの誤り。API を呼ぶ前に止める。"""


class StopRun(Exception):
    """実行を止める。code は終了コード。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# 条件（conditions.toml）の読み込みと検証
# ---------------------------------------------------------------------------


def conditions_path(arg: str | Path | None = None) -> Path:
    return Path(arg) if arg else ROOT / CONDITIONS_REL


def load_config(path: str | Path | None = None) -> dict:
    """conditions.toml を読み、形を確かめる。"""
    path = conditions_path(path)
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path} を読めません: {e}") from e
    validate_config(cfg)
    return cfg


def check_keys(where: str, table: dict, allowed: tuple[str, ...]) -> None:
    for key in table:
        if key not in allowed:
            raise ConfigError(f"{where}: 知らないキーです: {key!r}（使えるキー: {', '.join(allowed)}）")


def check_values(where: str, table: dict) -> None:
    """defaults と variants に共通のキーの、値の型と選択肢を確かめる。"""
    for key in ("judge_model", "rubric", "cases", "labels"):
        if key in table and (not isinstance(table[key], str) or not table[key].strip()):
            raise ConfigError(f"{where}.{key}: 空でない文字列にしてください: {table[key]!r}")
    if "instructor_labels" in table and not isinstance(table["instructor_labels"], str):
        raise ConfigError(f"{where}.instructor_labels: 文字列にしてください（使わないときは空文字）: {table['instructor_labels']!r}")
    if "output" in table and table["output"] not in OUTPUTS:
        raise ConfigError(f"{where}.output: {' か '.join(OUTPUTS)} のどちらかです: {table['output']!r}")
    if "order" in table and table["order"] not in ORDERS:
        raise ConfigError(f"{where}.order: {' か '.join(ORDERS)} のどちらかです: {table['order']!r}")
    if "trials" in table and (isinstance(table["trials"], bool) or not isinstance(table["trials"], int) or table["trials"] < 1):
        raise ConfigError(f"{where}.trials: 1 以上の整数にしてください: {table['trials']!r}")


def checks_cases_warnings(cfg: dict, root: Path) -> list[str]:
    """[checks.cases] の id と、実装（level = impl）の評価セットの id が合っているかを確かめ、合わないものを警告の文にして返す。

    tools/annotate.py の「作問」は id を 01 から振るので、conditions.toml の m01〜と合わないと、その件の個別の判定が黙って抜ける。
    評価セットがまだ無い・読めないとき、[checks.cases] が無いときは、何も返さない（止めない）。
    """
    checks = cfg.get("checks")
    table = checks.get("cases") if isinstance(checks, dict) else None
    impl = [v for v in resolve_variants(cfg).values() if v.level == "impl"]
    if not isinstance(table, dict) or not table or not impl:
        return []
    try:
        ids = list(dict.fromkeys(str(c["id"]) for c in read_jsonl(root / impl[0].cases)))
    except (OSError, ValueError, KeyError, TypeError):
        return []
    unknown = [key for key in table if key not in ids]
    uncovered = [i for i in ids if i not in table]
    hint = "（tools/annotate.py の「作問」は id を 01 から振ります。conditions.toml の [checks.cases] の id か、評価セットの id をそろえてください）"
    messages = []
    if unknown:
        messages.append(f"[checks.cases] にあって、評価セット {impl[0].cases} に無い id があります（その件の個別の判定は使われません）: {', '.join(unknown)} {hint}")
    if uncovered:
        messages.append(f"評価セット {impl[0].cases} にあって、[checks.cases] に無い id があります（その件には [checks] の global の判定だけを当てます）: {', '.join(uncovered)} {hint}")
    return messages


def validate_config(cfg: dict, root: Path | None = None) -> None:
    """[defaults]・[native]・[variants.*] を検証する（[checks] の中身は report.py が使うので、検証しない）。

    rubric の読み込みも行い、changes の old が対象の rubric にちょうど 1 か所だけ一致するかも確かめる。
    [checks.cases] の id が実装の評価セットの id と合わないときだけ、警告を表示する（止めない）。
    """
    root = Path(root) if root else ROOT
    defaults = cfg.get("defaults")
    if not isinstance(defaults, dict):
        raise ConfigError("[defaults] がありません")
    check_keys("defaults", defaults, DEFAULT_KEYS)
    for key in REQUIRED_DEFAULT_KEYS:
        if key not in defaults:
            raise ConfigError(f"defaults.{key} がありません")
    check_values("defaults", defaults)

    native = cfg.get("native", {})
    if not isinstance(native, dict):
        raise ConfigError("[native] は表（テーブル）にしてください")
    check_keys("native", native, NATIVE_KEYS)
    if "instructions_suffix" in native and (not isinstance(native["instructions_suffix"], str) or not native["instructions_suffix"].strip()):
        raise ConfigError("native.instructions_suffix: 空でない文字列にしてください")

    variants = cfg.get("variants")
    if not isinstance(variants, dict) or "baseline" not in variants:
        raise ConfigError("variants.baseline がありません（V0。すべての比較の基準）")
    owners: dict[str, str] = {}
    for name, spec in variants.items():
        where = f"variants.{name}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where}: 表（テーブル）にしてください")
        check_keys(where, spec, VARIANT_KEYS)
        for key in ("label", "level"):
            if key not in spec:
                raise ConfigError(f"{where}.{key} がありません")
        label = spec["label"]
        if not isinstance(label, str) or not label.strip():
            raise ConfigError(f"{where}.label: 空でない文字列にしてください: {label!r}")
        if label in owners:
            raise ConfigError(f"{where}.label {label!r} が {owners[label]} と重なっています（label は変種ごとに別にしてください）")
        owners[label] = name
        if spec["level"] not in LEVELS:
            raise ConfigError(f"{where}.level: {', '.join(LEVELS)} のどれかです: {spec['level']!r}")
        if "description" in spec and not isinstance(spec["description"], str):
            raise ConfigError(f"{where}.description: 文字列にしてください")
        if "require_rubric_changed" in spec and not isinstance(spec["require_rubric_changed"], bool):
            raise ConfigError(f"{where}.require_rubric_changed: true か false にしてください")
        check_values(where, spec)
        changes = spec.get("changes", [])
        if not isinstance(changes, list):
            raise ConfigError(f"{where}.changes: 配列にしてください")
        for i, change in enumerate(changes, 1):
            if not isinstance(change, dict):
                raise ConfigError(f"{where}.changes[{i}]: 表（{{ target = ..., op = ..., old = ..., new = ... }}）にしてください")
            check_keys(f"{where}.changes[{i}]", change, CHANGE_KEYS)
            if change.get("target") != "rubric":
                raise ConfigError(f"{where}.changes[{i}].target は 'rubric' です: {change.get('target')!r}")
            if change.get("op") != "replace_text":
                raise ConfigError(f"{where}.changes[{i}].op は 'replace_text' です: {change.get('op')!r}")
            if not isinstance(change.get("old"), str) or not change["old"]:
                raise ConfigError(f"{where}.changes[{i}].old: 空でない文字列にしてください")
            if not isinstance(change.get("new"), str):
                raise ConfigError(f"{where}.changes[{i}].new: 文字列にしてください")

    for v in resolve_variants(cfg).values():
        where = f"variants.{v.name}"
        if v.output == "native" and not native.get("instructions_suffix"):
            raise ConfigError(f"{where}: output = 'native' には [native].instructions_suffix が必要です")
        apply_changes(read_rubric(root, v.rubric), v.changes, where)  # rubric が読めること、old が 1 か所だけ一致すること
    for message in checks_cases_warnings(cfg, root):
        print(f"警告: {message}", file=sys.stderr)


@dataclass(frozen=True)
class Variant:
    """conditions.toml の 1 変種を、[defaults] と重ねて解決したもの。"""

    name: str
    label: str
    level: str
    description: str
    judge_model: str
    rubric: str
    output: str
    order: str
    cases: str
    labels: str
    instructor_labels: str
    trials: int
    changes: tuple
    require_rubric_changed: bool


def resolve_variants(cfg: dict) -> dict[str, Variant]:
    defaults = cfg["defaults"]
    resolved = {}
    for name, spec in cfg["variants"].items():
        merged = {**defaults, **{k: v for k, v in spec.items() if k in DEFAULT_KEYS}}
        resolved[name] = Variant(
            name=name, label=spec["label"], level=spec["level"], description=spec.get("description", ""),
            judge_model=merged["judge_model"], rubric=merged["rubric"], output=merged["output"], order=merged["order"],
            cases=merged["cases"], labels=merged["labels"], instructor_labels=merged.get("instructor_labels", ""),
            trials=merged["trials"], changes=tuple(spec.get("changes", [])),
            require_rubric_changed=bool(spec.get("require_rubric_changed", False)),
        )
    return resolved


# ---------------------------------------------------------------------------
# 変種を当てた結果: rubric・モデルに見える instructions・出力の型とスキーマ
# ---------------------------------------------------------------------------


def read_rubric(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"rubric を読めません: {rel}（{e.strerror or e}）") from e


def apply_changes(text: str, changes, where: str) -> str:
    """changes（replace_text）を順に当てる。old がちょうど 1 か所に一致しなければ ConfigError。"""
    for i, change in enumerate(changes, 1):
        old, new = change["old"], change["new"]
        count = text.count(old)
        if count != 1:
            raise ConfigError(
                f"{where}: changes[{i}] の old が rubric に {count} か所あります（ちょうど 1 か所である必要があります）: {old[:50]!r}"
            )
        text = text.replace(old, new)
    return text


def swapped_judge_result() -> type[BaseModel]:
    """出力の型のフィールド順だけを verdict→reason にした JudgeResult（V4）。

    クラス名・docstring・各 Field の description・型は配布コード（app/judge.py）の CriterionVerdict・JudgeResult から写す。
    送るスキーマの違いは「CriterionVerdict の properties と required の順」だけになる（$defs のキー名も CriterionVerdict のまま）。
    """
    fields = CriterionVerdict.model_fields
    names = ["verdict", *(n for n in fields if n != "verdict")]
    criterion = create_model(
        "CriterionVerdict", __doc__=CriterionVerdict.__doc__,
        **{n: (fields[n].annotation, Field(description=fields[n].description)) for n in names},
    )
    return create_model(
        "JudgeResult", __doc__=JudgeResult.__doc__,
        **{n: (criterion, Field(description=f.description)) for n, f in JudgeResult.model_fields.items()},
    )


def tool_output_type(order: str) -> type[BaseModel] | None:
    """ツール出力で run_sync に渡す型。配布コードの既定（JudgeResult）のままなら None。"""
    return None if order == "reason_first" else swapped_judge_result()


def tool_schema(order: str) -> dict:
    return (tool_output_type(order) or JudgeResult).model_json_schema()


def native_schema(order: str) -> dict:
    """NativeOutput に渡す、インラインのスキーマ（$ref なし）。各基準の properties・required・propertyOrdering を order の順にする。"""
    keys = ORDER_KEYS[order]
    fields = CriterionVerdict.model_fields
    leaves = {
        "reason": {"type": "string", "description": fields["reason"].description},
        "verdict": {"type": "string", "enum": list(VERDICTS), "description": fields["verdict"].description},
    }

    def criterion(description: str) -> dict:
        return {
            "type": "object", "description": description,
            "properties": {k: copy.deepcopy(leaves[k]) for k in keys},
            "required": list(keys), "propertyOrdering": list(keys),
        }

    return {
        "type": "object",
        "properties": {k: criterion(JudgeResult.model_fields[k].description) for k in CRITERIA},
        "required": list(CRITERIA), "propertyOrdering": list(CRITERIA),
    }


def native_output(schema: dict) -> NativeOutput:
    # StructuredDict は渡した dict に title を書き込むので、複製を渡す（schema_sha256 を変えないため）
    return NativeOutput(StructuredDict(copy.deepcopy(schema), name="JudgeResult"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data) -> str:
    """キーの順を並べ替えて hash する（同じ内容なら同じ値）。"""
    return sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class _Probed(Exception):
    """モデルに見える instructions を取るための、偽のモデルからの中断。"""


def seen_instructions(agent: Agent, output_type=None) -> str:
    """モデルに実際に渡る instructions を、API を呼ばずに取り出す（偽のモデルが instructions を見て中断する）。

    pydantic-ai は instructions の前後の空白を落とすので、文字列を組み立てた結果ではなく、渡る文字列そのものを使う。
    """
    seen: dict = {}

    def probe(messages, info):
        seen["instructions"] = info.instructions
        raise _Probed

    kwargs = {"output_type": output_type} if output_type is not None else {}
    try:
        agent.run_sync("probe", model=FunctionModel(probe), **kwargs)
    except _Probed:
        pass
    return seen.get("instructions") or ""


@dataclass
class Effective:
    """変種を当てた結果。採点の Agent を作るのに必要なものと、記録・比較のための hash。"""

    variant: Variant
    rubric_base: str  # 変更を当てる前の rubric（ファイルの中身）
    rubric: str  # 変更を当てた後
    rubric_sha256: str
    instructions: str  # モデルに見える instructions
    instructions_sha256: str
    schema: dict  # 出力スキーマ（手元で組み立てたもの）
    schema_sha256: str
    output_type: Any = None  # ツール出力で、配布コードの既定と違う型を使うときだけ
    native_instructions: str | None = None  # native のとき、Agent に渡す instructions（rubric + 接尾辞）

    def tail(self) -> str:
        """instructions のうち、rubric の後ろに付く部分（出力の形式の文）。"""
        body = self.rubric.strip()
        return self.instructions[len(body):] if self.instructions.startswith(body) else self.instructions


def build_effective(cfg: dict, name: str, root: Path | None = None) -> Effective:
    root = Path(root) if root else ROOT
    v = resolve_variants(cfg)[name]
    base = read_rubric(root, v.rubric)
    rubric = apply_changes(base, v.changes, f"variants.{name}")
    probe_model = FunctionModel(lambda messages, info: ModelResponse(parts=[]))
    if v.output == "native":
        schema = native_schema(v.order)
        native_instructions = rubric + cfg["native"]["instructions_suffix"]
        agent = Agent(probe_model, output_type=native_output(schema), instructions=native_instructions, model_settings={"temperature": 0.0})
        output_type = None
        instructions = seen_instructions(agent)
    else:
        schema = tool_schema(v.order)
        native_instructions = None
        output_type = tool_output_type(v.order)
        instructions = seen_instructions(build_judge(probe_model, rubric), output_type)
    return Effective(
        variant=v, rubric_base=base, rubric=rubric, rubric_sha256=sha256_text(rubric),
        instructions=instructions, instructions_sha256=sha256_text(instructions),
        schema=schema, schema_sha256=sha256_text(json.dumps(schema, ensure_ascii=False)),  # 順序を残して hash する
        output_type=output_type, native_instructions=native_instructions,
    )


def build_agent(eff: Effective, model) -> tuple[Agent, Any]:
    """採点の Agent と、run_sync に渡す output_type（None なら Agent の既定）を返す。"""
    if eff.variant.output == "native":
        agent = Agent(model, output_type=native_output(eff.schema), instructions=eff.native_instructions, model_settings={"temperature": 0.0})
        return agent, None
    return build_judge(model, eff.rubric), eff.output_type  # 配布コードと同じ instructions になる


CHANGE_ITEMS = ("モデル", "rubric の本文", "出力", "順序", "instructions の末尾")


def changed_items(base: Effective, other: Effective) -> list[str]:
    """base から other への変更点を、モデル・rubric の本文・出力・順序・instructions の末尾の 5 項目で数える。"""
    a, b = base.variant, other.variant
    checks = (
        a.judge_model != b.judge_model, base.rubric != other.rubric, a.output != b.output,
        a.order != b.order, base.tail() != other.tail(),
    )
    return [name for name, changed in zip(CHANGE_ITEMS, checks, strict=True) if changed]


# ---------------------------------------------------------------------------
# 秘密の除去と、書き込み
# ---------------------------------------------------------------------------

KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{35}")


def mask(value):
    """文字列（dict・list の中も含む）から、API キーの値と AIza… の形の文字列を *** に置き換える。"""
    secrets = sorted((v for v in (os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY")) if v), key=len, reverse=True)

    def clean(text: str) -> str:
        for secret in secrets:
            text = text.replace(secret, "***")
        return KEY_PATTERN.sub("***", text)

    def walk(x):
        if isinstance(x, str):
            return clean(x)
        if isinstance(x, dict):
            return {(clean(k) if isinstance(k, str) else k): walk(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [walk(v) for v in x]
        return x

    return walk(value)


TMP_PREFIX = ".tmp-"  # 書きかけの一時ファイルの名前の頭。t<k>.json・<id>.json などの探し方に当たらず、結果として数えない


def file_mode_for(path: Path) -> int:
    """書くファイルの権限。置き換えるファイルがあればその権限、なければ普通の新規ファイルと同じ（0o666 から umask を引いたもの）。

    NamedTemporaryFile は 0o600 で作るので、そのまま置くと、利用者のファイルの権限が変わってしまう。
    """
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def write_text_atomic(path: Path, text: str, *, exclusive: bool = False) -> bool:
    """固有の名前の一時ファイル（同じフォルダの .tmp-…）に書いてから、os.replace で置く（途中で止まっても、半端なファイルが残らない）。

    exclusive=True のときは、path がもうあれば置き換えず、False を返す（同時に動いた別の実行の結果を守る）。
    どんな終わり方でも、残った一時ファイルは消す。置けたら True。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=TMP_PREFIX, delete=False) as f:
            tmp = Path(f.name)
            f.write(mask(text))
        with contextlib.suppress(OSError):
            os.chmod(tmp, file_mode_for(path))
        if not exclusive:
            os.replace(tmp, path)
            return True
        try:
            os.link(tmp, path)  # path がもうあれば FileExistsError（作ると確かめるが 1 つの操作なので、同時に動いても取り違えない）
        except FileExistsError:
            return False
        except OSError:  # ハードリンクを作れないファイルシステムでは、確かめてから置き換える（こちらは、ごくわずかに隙間がある）
            if path.exists():
                return False
            os.replace(tmp, path)
        return True
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, data, *, exclusive: bool = False) -> bool:
    return write_text_atomic(path, json.dumps(mask(data), ensure_ascii=False, indent=2, default=str) + "\n", exclusive=exclusive)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """app.judge.write_jsonl と同じ形（1 行 1 件、ensure_ascii=False）で、原子的に書く。"""
    write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def rel_path(path: Path | str, root: Path | None = None) -> str:
    root = root or ROOT
    path = Path(path)
    return path.relative_to(root).as_posix() if path.is_absolute() and path.is_relative_to(root) else str(path)


# ---------------------------------------------------------------------------
# git: 結果がどの commit のものかを確かにする。ラベルは LLM の判定を見る前に凍結する
# ---------------------------------------------------------------------------


GIT_TIMEOUT = 60  # git が 1 回の確認で使える秒数（入力待ちなどで、止まったままにならないように）


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """git を読み取りだけに使う（status・rev-parse・ls-files）。index を書き換えず、入力を待たず、止まったままにならない。"""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL, timeout=GIT_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, 124, "", f"git が {GIT_TIMEOUT} 秒で終わりませんでした")
    except OSError as e:  # git が入っていない
        return subprocess.CompletedProcess(["git", *args], 127, "", str(e))


def git_state(repo: Path | None = None, paths=GUARDED_PATHS) -> dict:
    """commit と、実行に効くファイル（paths）の未コミットの変更（未追跡のファイルを含む）。"""
    repo = repo or ROOT
    head = run_git(repo, "rev-parse", "HEAD")
    status = run_git(repo, "status", "--porcelain", "--", *paths)
    dirty_files = [line for line in status.stdout.splitlines() if line.strip()] if status.returncode == 0 else []
    if status.returncode != 0 and head.returncode == 0:  # git のリポジトリなのに status が失敗したときは、きれいとは言えない
        dirty_files = [f"?? （git status が失敗しました: {status.stderr.strip()[:100]}）"]
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def dirty_message(what: str, dirty_files: list[str]) -> str:
    files = "\n".join(f"  {line}" for line in dirty_files)
    untracked = "\n?? は git が追跡していないファイルです。" if any(line.startswith("??") for line in dirty_files) else "\n"
    return f"{what}に、未コミットの変更があります。\n{files}{untracked}結果がどの commit のものかを確かにするため、commit してから実行してください（確認を省くときは --allow-dirty）。"


def git_path_problem(repo: Path, rel: str) -> str | None:
    """ファイルが commit 済みでなければ、その理由を返す。"missing"（無い）・"untracked"（git が追跡していない）・"dirty"（未コミットの変更）。"""
    if not (repo / rel).is_file():
        return "missing"
    if run_git(repo, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return "untracked"
    status = run_git(repo, "status", "--porcelain", "--", rel)
    return "dirty" if status.stdout.strip() else None


# ---------------------------------------------------------------------------
# API エラーの分類と、再試行
# ---------------------------------------------------------------------------


class ApiFailure(Exception):
    """API の失敗を、共通の形にしたもの。"""

    def __init__(self, status: int | None, text: str, retry_after: float | None = None, transport: bool = False):
        super().__init__(text)
        self.status = status
        self.text = text
        self.retry_after = retry_after
        self.transport = transport  # 接続の失敗・タイムアウト（status が無い）


def retry_delay_from_text(text: str) -> float | None:
    """429 の本文にある RetryInfo の retryDelay（例: '34s'）を秒で返す。"""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
    return float(match.group(1)) if match else None


def classify_failure(failure: ApiFailure, counters: dict) -> tuple[str, float | None]:
    """('retry', 待つ秒数)・('daily_limit', None)・('stop', None) のどれかを返す。

    429: 本文に PerDay があれば日の上限。なければ、待って最大 5 回まで再試行する（待つ秒数は 本文の retryDelay → ヘッダの retry_after → 60 秒）。
    503・500 などと、status の無い失敗（タイムアウト・接続）: 30・60・120 秒で再試行する。
    400〜404 などは、設定の誤りなので止める。
    """
    text = failure.text
    rate_limited = failure.status == 429 or "RESOURCE_EXHAUSTED" in text or "PerDay" in text
    if rate_limited and "PerDay" in text:
        return "daily_limit", None
    if rate_limited:
        counters["rate"] = counters.get("rate", 0) + 1
        if counters["rate"] > RATE_LIMIT_RETRIES:
            return "stop", None
        wait = retry_delay_from_text(text) or failure.retry_after or RATE_LIMIT_DEFAULT_WAIT
        return "retry", float(wait)
    server_side = (failure.status is not None and failure.status >= 500) or (failure.status is None and failure.transport)
    if server_side:
        counters["server"] = counters.get("server", 0) + 1
        if counters["server"] > len(SERVER_ERROR_WAITS):
            return "stop", None
        return "retry", float(SERVER_ERROR_WAITS[counters["server"] - 1])
    return "stop", None


def call_with_retries(ctx: Context, fn, attempts: list):
    """fn() を実行する。ApiFailure が出たら、classify_failure に従って待って再試行する。記録は attempts に足す。"""
    counters: dict = {}
    while True:
        try:
            value = fn()
        except ApiFailure as failure:
            action, wait = classify_failure(failure, counters)
            attempts.append({"attempt": len(attempts) + 1, "status": failure.status, "error": failure.text[:500], "action": action, "wait_s": wait})
            if action == "daily_limit":
                raise StopRun(2, DAILY_LIMIT_MESSAGE) from failure
            if action == "stop":
                hint = ""
                if failure.status is None or failure.status >= 500:
                    hint = "\n503 などが続くときは、しばらく待ってから同じコマンドで続きから実行してください。"
                raise StopRun(1, f"API エラーで止めます（{failure.status}）: {failure.text[:300]}{hint}\n同じコマンドで続きから実行できます。") from failure
            ctx.wait(wait, f"API エラー（{failure.status}）")
            continue
        attempts.append({"attempt": len(attempts) + 1, "ok": True})
        return value


def api_failure_from(e: ModelAPIError) -> ApiFailure:
    """pydantic-ai の API エラーを ApiFailure にする（ModelHTTPError 以外は、接続の失敗・タイムアウト）。"""
    return ApiFailure(getattr(e, "status_code", None), str(e), getattr(e, "retry_after", None), transport=not isinstance(e, ModelHTTPError))


# ---------------------------------------------------------------------------
# HTTP の記録と、本物のモデル（GoogleModel）
# ---------------------------------------------------------------------------


class HttpRecorder:
    """GoogleModel が送った HTTP の記録。1 件ずつ順番に呼ぶので、呼ぶ前に reset() して、その試行の分だけ集める。

    送信のヘッダ（API キーが入る）は取らない。応答のヘッダだけを取る。
    """

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[dict] = []

    def reset(self) -> None:
        self.requests = []
        self.responses = []

    async def on_request(self, request) -> None:
        try:
            body = request.content
        except Exception:  # noqa: BLE001  本文を読めない送り方のとき（ストリームなど）
            body = b""
        self.requests.append({"method": request.method, "url": str(request.url), "body": body})

    async def on_response(self, response) -> None:
        await response.aread()  # 本文を読んでから、ヘッダを取る
        self.responses.append({"status_code": response.status_code, "headers": dict(response.headers)})

    def first_body(self) -> dict | None:
        if not self.requests:
            return None
        try:
            body = json.loads(self.requests[0]["body"])
        except ValueError:
            return None
        return body if isinstance(body, dict) else None


def contains_key(data, names: set[str]) -> bool:
    if isinstance(data, dict):
        return any(k in names or contains_key(v, names) for k, v in data.items())
    if isinstance(data, list):
        return any(contains_key(v, names) for v in data)
    return False


def request_summary(recorder: HttpRecorder) -> dict | None:
    """送信の要点（temperature・thinkingConfig・response_json_schema の有無・propertyOrdering の有無・toolConfig の mode）。"""
    if not recorder.requests:
        return None
    body = recorder.first_body() or {}
    generation = body.get("generationConfig") or body.get("generation_config") or {}
    tool_config = body.get("toolConfig") or body.get("tool_config") or {}
    calling = tool_config.get("functionCallingConfig") or tool_config.get("function_calling_config") or {}
    return {
        "sha256": hashlib.sha256(recorder.requests[0]["body"]).hexdigest(),
        "n_requests": len(recorder.requests),
        "temperature": generation.get("temperature"),
        "thinking_config": generation.get("thinkingConfig", generation.get("thinking_config")),
        "response_json_schema": "responseJsonSchema" in generation or "response_json_schema" in generation,
        "property_ordering": contains_key(body, {"propertyOrdering", "property_ordering"}),
        "tool_mode": calling.get("mode"),
    }


def make_http_client(recorder: HttpRecorder, *, transport=None):
    """GoogleProvider が自分で作る client と同じもの（httpx2.AsyncClient、timeout 20 分・connect 5 秒）に、記録のフックを付ける。

    transport はテスト用（httpx2.MockTransport）。
    """
    import httpx2

    timeout = httpx2.Timeout(GOOGLE_HTTP_TIMEOUT, connect=GOOGLE_CONNECT_TIMEOUT)
    if transport is not None:
        return httpx2.AsyncClient(
            transport=transport, timeout=timeout,
            event_hooks={"request": [recorder.on_request], "response": [recorder.on_response]},
        )
    from pydantic_ai._http import create_async_httpx2_client

    client = create_async_httpx2_client(timeout=timeout)
    hooks = client.event_hooks
    client.event_hooks = {"request": [*hooks["request"], recorder.on_request], "response": [*hooks["response"], recorder.on_response]}
    return client


# ---------------------------------------------------------------------------
# 実行の状態（Context）
# ---------------------------------------------------------------------------


@dataclass
class Context:
    cfg: dict
    out: Path
    root: Path = field(default_factory=lambda: ROOT)
    dry_run: bool = False
    interval: float = 0.0
    git: dict = field(default_factory=dict)
    argv: list = field(default_factory=list)
    transport: Any = None  # テスト用: GoogleModel の HTTP を受け取る httpx2.MockTransport
    recorder: HttpRecorder = field(default_factory=HttpRecorder)
    inputs: dict = field(default_factory=dict)  # 予行で、評価セットとラベルを差し替えるファイル（root からの相対パス → 実際のパス）
    _effective: dict = field(default_factory=dict)
    _models: dict = field(default_factory=dict)
    _http_client: Any = None
    _cases: dict = field(default_factory=dict)

    def effective(self, name: str) -> Effective:
        if name not in self._effective:
            self._effective[name] = build_effective(self.cfg, name, self.root)
        return self._effective[name]

    def variants(self) -> dict[str, Variant]:
        return resolve_variants(self.cfg)

    def path(self, rel: str) -> Path:
        return self.inputs.get(rel) or (self.root / rel)

    def cases(self, v: Variant) -> list[dict]:
        path = self.path(v.cases)
        if path not in self._cases:
            if not path.is_file():
                raise ConfigError(f"評価セットがありません: {v.cases}")
            self._cases[path] = read_jsonl(path)
        return self._cases[path]

    def cases_sha256(self, v: Variant) -> str | None:
        return sha256_file(self.path(v.cases))

    def labels_sha256(self, v: Variant) -> dict:
        instructor = sha256_file(self.path(v.instructor_labels)) if v.instructor_labels else None
        return {"mine": sha256_file(self.path(v.labels)), "instructor": instructor}

    def google_model(self, name: str):
        """本物のモデル。HTTP を記録する client を共有する（1 件ずつ順番に呼ぶ）。"""
        if name not in self._models:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            from app.config import get_api_key

            if self._http_client is None:
                self._http_client = make_http_client(self.recorder, transport=self.transport)
            self._models[name] = GoogleModel(name, provider=GoogleProvider(api_key=get_api_key(), http_client=self._http_client))
        return self._models[name]

    def wait(self, seconds: float, why: str) -> None:
        print(f"  … {why}。{seconds:.0f} 秒待って、再試行します", flush=True)
        time.sleep(seconds)


# ---------------------------------------------------------------------------
# 採点（judge）の 1 回の実行
# ---------------------------------------------------------------------------


def native_test_model(order: str) -> FunctionModel:
    """予行用: native の出力（TextPart の JSON）を返す偽のモデル。判定は build_test_model と同じ規則（_rule_based_judge）。"""
    keys = ORDER_KEYS[order]

    def respond(messages, info):
        # _rule_based_judge は info.output_tools[0] を読むが、native では空。名前だけの偽の出力ツールを渡して判定だけ借りる
        borrowed = dataclasses.replace(info, output_tools=[ToolDefinition(name="final_result")])
        args = _rule_based_judge(messages, borrowed).parts[0].args_as_dict()
        out = {c: {k: args[c][k] for k in keys} for c in CRITERIA}
        return ModelResponse(parts=[TextPart(json.dumps(out, ensure_ascii=False))])

    return FunctionModel(respond, model_name="test")


def make_judge_model(ctx: Context, eff: Effective):
    """予行は偽のモデル（API を呼ばない）。本番は GoogleModel（HTTP を記録する client つき）。"""
    if ctx.dry_run:
        return native_test_model(eff.variant.order) if eff.variant.output == "native" else build_test_model()
    return ctx.google_model(eff.variant.judge_model)


def args_dict(part) -> dict | str:
    try:
        return part.args_as_dict()
    except Exception:  # noqa: BLE001  引数が JSON として読めないとき
        return part.args if isinstance(part.args, str) else str(part.args)


def as_text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def last_response(messages) -> ModelResponse | None:
    return next((m for m in reversed(messages) if isinstance(m, ModelResponse)), None)


def response_text(response: ModelResponse | None) -> str:
    return "".join(p.content for p in response.parts if isinstance(p, TextPart)) if response is not None else ""


OUTPUT_TOOL_NAME = "final_result"  # pydantic-ai の出力ツールの名前（output_type が 1 つのとき）
FINAL_RESULT_PROCESSED = "Final result processed."  # 採用された出力ツールの ToolReturnPart の内容（pydantic-ai 2.57.0 の _tool_execution._FINAL_RESULT_PROCESSED）


def adopted_tool_call(messages, response: ModelResponse | None) -> ToolCallPart | None:
    """応答の ToolCallPart のうち、pydantic-ai が最終の出力として採用したもの（result.output の元）。

    1 つの応答に final_result が 2 つあるとき、採用されるのは、最初に検証を通った 1 つだけ。最後の呼び出しとは限らない。
    最終の ModelRequest にある、内容が "Final result processed." の ToolReturnPart の tool_call_id で見つける。
    見つからなければ（失敗した実行など）、最初の final_result。
    """
    calls = [p for p in response.parts if isinstance(p, ToolCallPart)] if response is not None else []
    if not calls:
        return None
    last_request = next((m for m in reversed(messages) if isinstance(m, ModelRequest)), None)
    if last_request is not None:
        processed = {p.tool_call_id for p in last_request.parts if isinstance(p, ToolReturnPart) and p.content == FINAL_RESULT_PROCESSED}
        for call in calls:
            if call.tool_call_id in processed:
                return call
    return next((c for c in calls if c.tool_name == OUTPUT_TOOL_NAME), calls[0])


def expected_key_order(order: str) -> dict:
    return {"top": list(CRITERIA), **{c: list(ORDER_KEYS[order]) for c in CRITERIA}}


def key_order_of(data) -> dict | None:
    """出力の JSON の、キーの順（最上位と各基準）。dict でなければ None。"""
    if not isinstance(data, dict):
        return None
    return {"top": list(data), **{c: (list(data[c]) if isinstance(data.get(c), dict) else None) for c in CRITERIA}}


def validate_native_text(text: str) -> tuple[dict | None, dict | None, str | None]:
    """native の生の出力（JSON 文字列）を確かめる。戻り値は (verdicts, key_order, 問題)。

    4 基準がそろい、各基準に reason（文字列）と verdict（pass / fail / unsure）があること。だめなら verdicts は None。
    """
    try:
        data = json.loads(text)
    except ValueError as e:
        return None, None, f"JSON として読めません: {e}"
    if not isinstance(data, dict):
        return None, None, "JSON の最上位が object ではありません"
    problems, verdicts = [], {}
    for c in CRITERIA:
        item = data.get(c)
        if not isinstance(item, dict):
            problems.append(f"{c} がありません")
            continue
        reason, verdict = item.get("reason"), item.get("verdict")
        if not isinstance(reason, str):
            problems.append(f"{c}.reason が文字列ではありません")
        if verdict not in VERDICTS:
            problems.append(f"{c}.verdict が {'/'.join(VERDICTS)} のどれでもありません: {verdict!r}")
        if isinstance(reason, str) and verdict in VERDICTS:
            verdicts[c] = {"reason": reason, "verdict": verdict}
    return (None if problems else verdicts), key_order_of(data), ("; ".join(problems) or None)


def usage_of(result, messages) -> dict:
    """使用量。失敗したときは、応答ごとの使用量を足す。思考トークンは details の thoughts_tokens。"""
    if result is not None:
        usage = result.usage  # 2.57.0 ではプロパティ（result.usage() ではない）
        return {"requests": usage.requests, "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "details": dict(usage.details)}
    requests = input_tokens = output_tokens = 0
    details: Counter = Counter()
    for m in messages:
        if isinstance(m, ModelResponse):
            requests += 1
            input_tokens += m.usage.input_tokens
            output_tokens += m.usage.output_tokens
            details.update(m.usage.details)
    return {"requests": requests, "input_tokens": input_tokens, "output_tokens": output_tokens, "details": dict(details)}


def judge_command(eff: Effective) -> str:
    v = eff.variant
    if v.output == "native":
        return (
            "Agent(model, output_type=NativeOutput(StructuredDict(<スキーマ>, name='JudgeResult')), instructions=<rubric + [native].instructions_suffix>, "
            f"model_settings={{'temperature': 0.0}}).run_sync(judge_prompt(case))（variant={v.name}）"
        )
    extra = ", output_type=<フィールド順を入れ替えた JudgeResult>" if eff.output_type is not None else ""
    return f"app.judge.build_judge(model, <rubric>).run_sync(judge_prompt(case){extra})（variant={v.name}）"


def judge_dir(ctx: Context, v: Variant) -> Path:
    return ctx.out / "judge" / v.name


def judge_path(ctx: Context, v: Variant, case_id, k: int) -> Path:
    return judge_dir(ctx, v) / str(case_id) / f"t{k}.json"


def run_judge_trial(ctx: Context, eff: Effective, case: dict, k: int) -> dict:
    """1 件を 1 回採点して、記録（dict）を返す。API の失敗は再試行し、止めるときは StopRun。invalid_output は記録として返す。"""
    v = eff.variant
    prompt = judge_prompt(case)
    attempts: list = []

    def attempt() -> dict:
        ctx.recorder.reset()
        agent, output_type = build_agent(eff, make_judge_model(ctx, eff))
        kwargs = {"output_type": output_type} if output_type is not None else {}
        result, status, error = None, "ok", None
        started = time.perf_counter()
        with capture_run_messages() as captured:
            try:
                result = agent.run_sync(prompt, **kwargs)
            except ModelAPIError as e:
                raise api_failure_from(e) from e
            except UnexpectedModelBehavior as e:
                status, error = "invalid_output", {"type": type(e).__name__, "message": str(e)}
            except (AgentRunError, UserError) as e:
                status, error = "error", {"type": type(e).__name__, "message": str(e)}
        duration = time.perf_counter() - started
        messages = result.all_messages() if result is not None else list(captured)
        return {
            "result": result, "status": status, "error": error, "duration": duration, "messages": messages,
            "request": request_summary(ctx.recorder), "http": list(ctx.recorder.responses), "bodies": list(ctx.recorder.requests),
        }

    started_at = now_iso()
    got = call_with_retries(ctx, attempt, attempts)
    result, messages, status, error = got["result"], got["messages"], got["status"], got["error"]
    response = last_response(messages)

    # 生の出力と、判定の取り出し
    verdicts, key_order, order_ok, raw_output = None, None, None, None
    if v.output == "native":
        text = response_text(response)
        raw_output = {"kind": "native_text", "value": text} if response is not None else None
        if result is not None:
            verdicts, key_order, problem = validate_native_text(text)
            if problem:
                status, error = "invalid_output", {"type": "InvalidOutput", "message": problem}
        else:
            key_order = key_order_of(_loads_or_none(text))
        if key_order is not None:
            order_ok = key_order == expected_key_order(v.order)
    else:
        call = adopted_tool_call(messages, response)  # verdicts は result.output（採用された呼び出し）から取るので、生の出力も同じ呼び出しにする
        args = args_dict(call) if call is not None else None
        raw_output = {"kind": "tool_args", "value": args} if args is not None else None
        key_order = key_order_of(args)
        if result is not None:
            verdicts = {c: {"reason": getattr(result.output, c).reason, "verdict": getattr(result.output, c).verdict} for c in CRITERIA}
    if verdicts is not None:
        verdicts["lecture_mark"] = {
            "reason": "末尾（最後の行）に「第N回」があるかをコードで判定",
            "verdict": check_lecture_mark(case["answer"]),
        }

    first_request = next((m for m in messages if isinstance(m, ModelRequest) and m.instructions), None)
    seen = first_request.instructions if first_request is not None else eff.instructions
    http = None
    if not ctx.dry_run:
        details = response.provider_details if response is not None and isinstance(response.provider_details, dict) else {}
        http = {
            "headers": got["http"][-1]["headers"] if got["http"] else None,
            "finish_reason": response.finish_reason if response is not None else None,
            "provider_response_id": response.provider_response_id if response is not None else None,
            "provider_finish_reason": details.get("finish_reason"),
        }
        write_request_sample(ctx, v, got["bodies"])
    record = {
        "schema": 1, "kind": "judge", "variant": v.name, "label": v.label, "level": v.level,
        "case_id": case["id"], "trial": k,
        "model": v.judge_model, "model_name": response.model_name if response is not None and not ctx.dry_run else None,
        "output_mode": v.output, "order": v.order,
        "rubric_path": v.rubric, "rubric_sha256": eff.rubric_sha256,
        "instructions": seen, "instructions_sha256": sha256_text(seen), "instructions_match": seen == eff.instructions,
        "schema_sha256": eff.schema_sha256,
        "cases_path": v.cases, "cases_sha256": ctx.cases_sha256(v), "case_sha256": sha256_json(case),
        "prompt_sha256": sha256_text(prompt),  # 採点者に渡した文（質問・回答・参照）。参照の用語集のエントリが変わると変わる
        "labels_sha256": ctx.labels_sha256(v),
        "git": {"commit": ctx.git.get("commit"), "dirty": ctx.git.get("dirty")},
        "command": judge_command(eff), "compare_args": ctx.argv, "dry_run": ctx.dry_run,
        "started_at": started_at, "finished_at": now_iso(), "duration_s": round(got["duration"], 3),
        "status": status, "error": error, "attempts": attempts,
        "verdicts": verdicts, "raw_output": raw_output, "key_order": key_order, "order_ok": order_ok,
        "request": None if ctx.dry_run else got["request"], "response": http,
        "usage": usage_of(result, messages),
    }
    return record


def _loads_or_none(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def write_request_sample(ctx: Context, v: Variant, bodies: list[dict]) -> None:
    """変種ごとに、最初の 1 件の送信の本文（JSON の全文）を残す。ヘッダ（API キーが入る）は保存しない。"""
    path = judge_dir(ctx, v) / "request_sample.json"
    if path.exists() or not bodies:
        return
    try:
        body = json.loads(bodies[0]["body"])
    except ValueError:
        return
    write_json_atomic(path, body)


# ---------------------------------------------------------------------------
# 回答の生成（実装）: 評価セットの質問を、自分のエージェント（app.agent）に投げて answer を埋める
# ---------------------------------------------------------------------------


def make_decider(plan: tuple | list, log: list):
    """承認待ちの呼び出しごとに、plan の答え（足りない分は deny）を返す。記録は log に足す。generate は常に拒否する。"""
    state = {"n": 0}

    def decide(call) -> bool:
        decision = plan[state["n"]] if state["n"] < len(plan) else "deny"
        state["n"] += 1
        log.append({
            "index": state["n"], "tool_call_id": call.tool_call_id, "tool_name": call.tool_name,
            "args": args_dict(call), "decision": "approved" if decision == "approve" else "denied",
        })
        return decision == "approve"

    return decide


def collect_tool_calls(messages) -> list[dict]:
    """会話から、ツール呼び出しを順に取り出す（何回目の応答か、名前、引数、戻り値の全文、outcome）。"""
    calls, by_id, response_index = [], {}, 0
    for message in messages:
        if isinstance(message, ModelResponse):
            response_index += 1
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    entry = {
                        "order": len(calls) + 1, "response_index": response_index, "tool_call_id": part.tool_call_id,
                        "name": part.tool_name, "args": args_dict(part), "result": None, "outcome": "pending",
                    }
                    calls.append(entry)
                    by_id[part.tool_call_id] = entry
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                entry = by_id.get(getattr(part, "tool_call_id", None))
                if entry is None:
                    continue
                if isinstance(part, ToolReturnPart):
                    entry["result"], entry["outcome"] = as_text(part.content), part.outcome
                elif isinstance(part, RetryPromptPart):
                    entry["result"], entry["outcome"] = as_text(part.content), "retry"
    return calls


def generation_model(ctx: Context):
    """予行は TestModel（API を呼ばない。承認が要る save_learning_log も呼ぶ）。本番は app.agent.build_model()。"""
    if ctx.dry_run:
        from pydantic_ai.models.test import TestModel

        return TestModel(call_tools=["search_glossary", "make_quiz", "save_learning_log"])
    from app import agent as agent_module

    return agent_module.build_model()


def generation_model_name() -> str:
    from app.config import MODEL

    return MODEL


def agent_sha256(root: Path) -> str:
    """回答の生成を左右するファイル（app/agent.py・app/tools.py・data/glossary.json）の中身を、この順に連結した sha256。

    generate の記録に残し、記録を使い回す前に今の値と比べる。無いファイルは、空として連結する。
    """
    data = b"".join((root / rel).read_bytes() if (root / rel).is_file() else b"" for rel in AGENT_FILES)
    return hashlib.sha256(data).hexdigest()


def run_generate_case(ctx: Context, case: dict) -> dict:
    """1 件の質問を自分のエージェントに投げて、記録（dict）を返す。承認が要るツールは、拒否して記録する。"""
    from app import agent as agent_module
    from app import tools

    case_id, question = case["id"], case["question"]
    attempts: list = []

    def attempt() -> dict:
        random.seed(case_id)  # make_quiz の出題を、実行のたびにそろえる
        with tempfile.TemporaryDirectory(prefix="lec04-generate-") as root:
            old_log_dir = tools.LOG_DIR
            tools.LOG_DIR = Path(root) / "learning_log"  # 本物の learning_log/ を汚さない
            approvals: list = []
            try:
                with capture_run_messages() as captured:
                    started = time.perf_counter()
                    result, status, error = None, "ok", None
                    try:
                        result = agent_module.run_with_approval(
                            agent_module.agent, question, decide=make_decider((), approvals),
                            model=generation_model(ctx), usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
                        )
                    except UsageLimitExceeded as e:
                        status, error = "usage_limit_exceeded", {"type": type(e).__name__, "message": str(e)}
                    except ModelAPIError as e:
                        raise api_failure_from(e) from e
                    except UnexpectedModelBehavior as e:
                        status, error = "unexpected_model_behavior", {"type": type(e).__name__, "message": str(e)}
                    except (AgentRunError, UserError) as e:
                        status, error = "error", {"type": type(e).__name__, "message": str(e)}
                    duration = time.perf_counter() - started
                partial = list(captured)
            finally:
                tools.LOG_DIR = old_log_dir
            log_dir = Path(root) / "learning_log"
            saved = [{"name": p.relative_to(log_dir).as_posix(), "content": p.read_text(encoding="utf-8")}
                     for p in sorted(log_dir.rglob("*")) if p.is_file()] if log_dir.exists() else []
            outside = [p.relative_to(root).as_posix() for p in sorted(Path(root).rglob("*"))
                       if p.is_file() and log_dir not in p.parents]
        return {"result": result, "status": status, "error": error, "duration": duration, "approvals": approvals,
                "saved": saved, "outside": outside, "partial": partial}

    started_at = now_iso()
    got = call_with_retries(ctx, attempt, attempts)
    result, status, error = got["result"], got["status"], got["error"]
    answer, messages_json, usage, response = "", ModelMessagesTypeAdapter.dump_json(got["partial"]), {}, last_response(got["partial"])
    if result is not None:
        messages_json = result.all_messages_json()
        usage = dataclasses.asdict(result.usage)
        response = last_response(result.all_messages())
        if isinstance(result.output, str):
            answer = result.output
            if not answer.strip():  # 空の回答を、評価セットに書き込まない（回答が無い件と区別できなくなる）
                status, error = "error", {"type": "EmptyAnswer", "message": "モデルの回答が空でした"}
        else:
            status = "pending_approvals"  # max_rounds までに承認待ちが終わらなかった
    messages = result.all_messages() if result is not None else got["partial"]
    return {
        "schema": 1, "kind": "generate", "case_id": case_id, "question": question,
        "model": "test" if ctx.dry_run else generation_model_name(),
        "model_name": response.model_name if response is not None and not ctx.dry_run else None,
        "agent_sha256": agent_sha256(ctx.root),  # 記録を使い回してよいかの確認に使う（エージェントのコード・ツール・用語集が変わったら、作り直す）
        "seed": case_id, "git": {"commit": ctx.git.get("commit"), "dirty": ctx.git.get("dirty")},
        "command": "app.agent.run_with_approval(agent, <質問>, decide=<常に拒否>, model=build_model(), usage_limits=UsageLimits(request_limit=10))",
        "compare_args": ctx.argv, "dry_run": ctx.dry_run,
        "started_at": started_at, "finished_at": now_iso(), "duration_s": round(got["duration"], 3),
        "status": status, "error": error, "attempts": attempts, "answer": answer,
        "tool_calls": collect_tool_calls(messages), "approvals": got["approvals"],
        "saved_files": got["saved"], "outside_files": got["outside"],
        "usage": usage, "messages": json.loads(messages_json),
    }


def generate_path(ctx: Context, case_id) -> Path:
    return ctx.out / "generate" / f"{case_id}.json"


def check_no_answers(cases: list[dict], shown: str, force: bool) -> None:
    """回答が 1 件でも入っている評価セットは、--force が無ければ止める（付けたラベルとのずれを防ぐ）。"""
    done = [c["id"] for c in cases if str(c.get("answer", "")).strip()]
    if done and not force:
        raise StopRun(
            1,
            f"{shown} には、もう回答が入っています（{len(done)} 件）。作り直すと、付けたラベルとずれます。\n"
            "それでも作り直すときは --force を付けてください（既存のラベルと、results/judge の結果は取り直しになります）。",
        )


CASE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,40}")  # 結果のファイル名（judge/<変種>/<id>/・generate/<id>.json）に使う id の形


def check_case_ids(cases: list[dict], shown: str) -> None:
    """評価セットの id が、結果のファイル名に使える形で、重なっていないことを確かめる（API を呼ぶ前に止める）。

    重なると、後の試行が同じ t1.json を上書きする。/ や .. を含む id は、結果の置き場所の外を指しうる。
    大文字小文字だけが違う id は、大文字小文字を区別しないファイルシステム（macOS・Windows）では同じファイルになるので、重なりに数える。
    """
    ids = [c.get("id") if isinstance(c, dict) else None for c in cases]
    bad = [repr(i) for i in ids if not (isinstance(i, str) and CASE_ID_PATTERN.fullmatch(i))]
    if bad:
        raise StopRun(
            1,
            f"{shown} の id に、使えないものがあります: {', '.join(dict.fromkeys(bad))}\n"
            "id は結果のファイル名に使うので、英数字・_・- だけの 1〜40 文字の文字列にしてください。",
        )
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(i.casefold(), []).append(i)
    repeated = sorted({i for group in groups.values() if len(group) > 1 for i in group})
    if repeated:
        raise StopRun(
            1,
            f"{shown} の id が重なっています: {', '.join(repeated)}（大文字小文字だけが違う id も、同じファイルになることがあるので重なりに数えます）\n"
            "重なると、後の試行が同じ結果のファイルを上書きします。id は 1 件ずつ別にしてください。",
        )


def read_generate_record(path: Path) -> dict | None:
    """既存の generate の記録。読めない（JSON でない・object でない）ときは None。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def generate_reuse_problems(existing: dict | None, case: dict, *, dry_run: bool, model: str, agent: str) -> list[str]:
    """既存の generate の記録を使い回せない理由（使い回せるなら空）。

    使い回すのは、status が ok・dry_run が今の実行と同じ・質問・モデル・agent_sha256 が今と同じ、のすべてが合うときだけ。
    予行（TestModel）の記録を本番の評価セットに書いたり、別のモデル・別のエージェントの回答を混ぜたりしないため。
    """
    if existing is None:
        return ["記録を読めません"]
    problems = []
    if existing.get("status") != "ok":
        problems.append(f"status が ok ではありません（{existing.get('status')}）")
    if existing.get("dry_run") is not dry_run:
        recorded = existing.get("dry_run")
        problems.append("予行（dry_run）の記録です" if recorded is True else "本番の記録です" if recorded is False else "dry_run の記録がありません")
    if existing.get("question") != case["question"]:
        problems.append("質問が評価セットと違います")
    if existing.get("model") != model:
        problems.append(f"モデルが違います（記録 {existing.get('model')} / 今 {model}）")
    if existing.get("agent_sha256") != agent:
        problems.append(
            "エージェント（app/agent.py・app/tools.py・data/glossary.json）が変わりました" if existing.get("agent_sha256")
            else "記録に agent_sha256 がありません（古い記録）"
        )
    return problems


def generate_cases(ctx: Context, cases_path: Path, *, force: bool, shown: str | None = None) -> int:
    """cases_path の全件の回答を作る。全件が ok になったときだけ、cases_path の answer を埋めて書く（1 件でも失敗すれば書かない）。"""
    shown = shown or rel_path(cases_path, ctx.root)
    cases = read_jsonl(cases_path)
    check_case_ids(cases, shown)
    check_no_answers(cases, shown, force)
    agent_now = agent_sha256(ctx.root)
    model_now = "test" if ctx.dry_run else generation_model_name()
    records: dict[str, dict] = {}
    executed = 0
    for n, case in enumerate(cases, 1):
        case_id, path = case["id"], generate_path(ctx, case["id"])
        shown_path = rel_path(path, ctx.root)
        if path.is_file() and not force:
            existing = read_generate_record(path)
            problems = generate_reuse_problems(existing, case, dry_run=ctx.dry_run, model=model_now, agent=agent_now)
            if not problems:
                records[case_id] = existing
                print(f"[{n}/{len(cases)}] {case_id}: 済み（{shown_path} を使います）", flush=True)
                continue
            if ctx.dry_run and existing is not None and existing.get("dry_run") is False:
                raise StopRun(1, f"{shown_path} は本番の記録です。予行の記録で置き換えないため止めます（予行は、別の --out に書いてください）。")
            print(f"[{n}/{len(cases)}] {case_id}: 作り直します（{shown_path} は使えません: {'・'.join(problems)}）", flush=True)
        if executed and ctx.interval > 0:
            time.sleep(ctx.interval)
        if not ctx.dry_run:
            print(f"  → {case_id} を生成します", flush=True)
        record = run_generate_case(ctx, case)
        write_json_atomic(path, record)
        executed += 1
        records[case_id] = record
        usage = record["usage"] or {}
        print(
            f"[{n}/{len(cases)}] {case_id}: {record['status']} {record['duration_s']:.1f}s tool_calls={len(record['tool_calls'])} "
            f"requests={usage.get('requests')} approvals={len(record['approvals'])}",
            flush=True,
        )
    failed = [cid for cid, r in records.items() if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} 件が ok になりませんでした: {', '.join(failed)}。{shown} は書きません。", file=sys.stderr)
        for cid in failed:
            r = records[cid]
            print(f"  {cid}: {r['status']} {(r['error'] or {}).get('message', '')[:200]}", file=sys.stderr)
        print("詳しくは results/generate/<id>.json を見てください。同じコマンドで、済みの件は飛ばして続きから実行できます。", file=sys.stderr)
        return 1
    for case in cases:
        case["answer"] = records[case["id"]]["answer"]
    write_jsonl_atomic(cases_path, cases)
    print(f"\n→ 回答を書き込みました（{len(cases)} 件）: {shown}。記録: {ctx.out / 'generate'}")
    return 0


# ---------------------------------------------------------------------------
# run: 試行番号を外側に回して、全変種を実行する
# ---------------------------------------------------------------------------


def csv_list(values) -> list[str] | None:
    """--level a b や --level a,b のどちらの書き方でも、リストにする。"""
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    return [item for value in values for item in value.split(",") if item]


def parse_levels(values) -> tuple[str, ...]:
    items = csv_list(values)
    if not items or "all" in items:
        return LEVELS
    for item in items:
        if item not in LEVELS:
            raise ConfigError(f"--level {item!r} は知りません（使えるもの: all, {', '.join(LEVELS)}）")
    return tuple(item for item in LEVELS if item in items)


def select_variants(cfg: dict, args) -> list[Variant]:
    """--level と --variant（名前か label）で絞り込む。打ち間違いは、使える名前を示して止める。順番は conditions.toml の順。"""
    resolved = resolve_variants(cfg)
    levels = parse_levels(getattr(args, "level", None))
    wanted = csv_list(getattr(args, "variant", None))
    if wanted:
        known = {w.lower() for v in resolved.values() for w in (v.name, v.label)}
        for item in wanted:
            if item.lower() not in known:
                choices = ", ".join(f"{v.name}（{v.label}）" for v in resolved.values())
                raise ConfigError(f"--variant {item!r} は知りません（使えるもの: {choices}）")
    selected = []
    for v in resolved.values():
        if v.level not in levels:
            continue
        if wanted and not any(item.lower() in (v.name.lower(), v.label.lower()) for item in wanted):
            continue
        selected.append(v)
    return selected


def check_variant_case_ids(ctx: Context, variants: list[Variant]) -> None:
    """選んだ変種が使う評価セット（同じファイルは 1 回）の id が、一意で安全か。API を呼ぶ前に確かめる。選んでいない変種の評価セットは読まない。"""
    seen: set[Path] = set()
    for v in variants:
        path = ctx.path(v.cases)
        if path not in seen:
            seen.add(path)
            check_case_ids(ctx.cases(v), v.cases)


def check_case_filter(ctx: Context, wanted: list[str] | None, variants: list[Variant], *, skip_missing: bool = False) -> None:
    """--case の id が、選んだ変種（variants）の評価セットにあるか。選んでいない変種の評価セットは、読まない・要求しない。

    skip_missing: 評価セットのファイルが無い変種を、数えずに飛ばす（status 用。run は、無ければ ConfigError で止める）。
    """
    if not wanted:
        return
    known: list[str] = []
    for v in variants:
        try:
            known += [str(c["id"]) for c in ctx.cases(v)]
        except ConfigError:
            if not skip_missing:
                raise
    known = list(dict.fromkeys(known))
    for item in wanted:
        if item not in known:
            raise ConfigError(f"--case {item!r} は知りません（使えるもの: {', '.join(known)}）")


def trials_of(v: Variant, override: int | None) -> int:
    """--trials は上限として扱う（I1 のように trials = 1 の変種に、余分な試行を作らない）。"""
    return min(override, v.trials) if override else v.trials


def build_plan(ctx: Context, variants: list[Variant], *, cases: list[str] | None = None, trials: int | None = None) -> list[tuple[int, Variant, dict]]:
    """実行する (試行番号, 変種, 件) を、実行する順に返す。試行番号を外側に回す。変種は conditions.toml の順、件は評価セットの順。"""
    per_variant = {v.name: [c for c in ctx.cases(v) if not cases or str(c["id"]) in cases] for v in variants}
    plan = []
    for k in range(1, max((trials_of(v, trials) for v in variants), default=0) + 1):
        for v in variants:
            if k > trials_of(v, trials):
                continue
            plan += [(k, v, case) for case in per_variant[v.name]]
    return plan


def trial_done(ctx: Context, v: Variant, case_id, k: int) -> bool:
    return judge_path(ctx, v, case_id, k).exists()


def saved_records(ctx: Context, v: Variant) -> list[Path]:
    """変種の保存済みの記録（<out>/judge/<variant>/<case_id>/t<k>.json）すべて。request_sample.json や、書きかけの .tmp- のファイルは数えない。"""
    base = judge_dir(ctx, v)
    if not base.is_dir():
        return []
    return sorted(
        p for folder in base.iterdir() if folder.is_dir()
        for p in folder.iterdir() if p.is_file() and re.fullmatch(r"t\d+\.json", p.name)
    )


def is_inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def resolve_out(args, dry_run: bool, root: Path | None = None) -> Path:
    root = root or ROOT
    default = root / RESULTS_REL
    if getattr(args, "out", None):
        out = Path(args.out)
    elif dry_run:
        return Path(tempfile.mkdtemp(prefix="lec04-dry-run-"))
    else:
        return default
    if dry_run and is_inside(out, default):
        raise StopRun(1, f"--dry-run の結果は {RESULTS_REL} に書けません（本物の結果と混ざるため）。--out を省くと一時フォルダに書きます。")
    return out


def make_context(args, cfg: dict, *, dry_run: bool = False, guard_paths=GUARDED_PATHS, with_git: bool = True) -> Context:
    return Context(
        cfg=cfg, out=resolve_out(args, dry_run), root=ROOT, dry_run=dry_run,
        interval=0.0 if dry_run else float(getattr(args, "interval", 0) or 0),
        argv=list(getattr(args, "_argv", None) or []), git=git_state(ROOT, guard_paths) if with_git else {},
    )


def dummy_labels(ids: list) -> list[dict]:
    return [
        {"id": i, "grounding": "pass", "scope": "pass", "hint": "pass", "clarity": "pass", "lecture_mark": "pass",
         "unsure_flag": False, "note": "予行用の仮のラベル（全部 pass）"}
        for i in ids
    ]


def prepare_dry_run(ctx: Context, variants: list[Variant]) -> None:
    """予行の入力を、一時フォルダ（<out>/dry_run_inputs/）に用意する。ユーザーのファイルには書かない。

    - 実装の評価セット: my_cases.jsonl のコピー。回答が無ければ、予行の generate（TestModel）で埋める
    - ラベル: 全部 pass の仮のラベル（本物のラベルは読まない）
    """
    work = ctx.out / "dry_run_inputs"
    work.mkdir(parents=True, exist_ok=True)
    for rel in dict.fromkeys(v.cases for v in variants if v.level == "impl"):
        src, dst = ctx.root / rel, work / Path(rel).name
        if not src.is_file():
            raise ConfigError(f"評価セットがありません: {rel}")
        if not dst.exists():
            shutil.copyfile(src, dst)
        ctx.inputs[rel] = dst
        if any(not str(c.get("answer", "")).strip() for c in read_jsonl(dst)):
            print(f"予行: {rel} のコピーに回答が無いので、先に予行の generate（TestModel）で埋めます", flush=True)
            if generate_cases(ctx, dst, force=True, shown=f"{rel}（予行のコピー）") != 0:
                raise StopRun(1, "予行の generate が失敗しました")
        ctx._cases.pop(dst, None)
    for rel, cases_rel in dict.fromkeys((v.labels, v.cases) for v in variants):
        dst = work / Path(rel).name
        ids = [c["id"] for c in ctx.cases(next(v for v in variants if v.labels == rel and v.cases == cases_rel))]
        if not dst.exists():
            write_text_atomic(dst, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dummy_labels(ids)))
        ctx.inputs[rel] = dst


def current_prompt_sha256(case: dict) -> str | None:
    """今の件を採点者に渡す文（judge_prompt）の sha256。質問・回答の欄が無い件は None（照合しない）。"""
    try:
        return sha256_text(judge_prompt(case))
    except (KeyError, TypeError, AttributeError):
        return None


def check_saved_records(ctx: Context, variants: list[Variant]) -> None:
    """選んだ変種の、保存済みのすべての記録を、今の条件と照らす。食い違いがあれば止める（API を呼ぶ前）。

    変種全体の値（rubric・instructions・schema・評価セットのファイル・モデル・dry_run）は、どの記録でも同じでなければならない。
    件ごとの値（case_sha256・prompt_sha256）は、今の評価セットにある件の記録だけ照らす。
    """
    for v in variants:
        eff = ctx.effective(v.name)
        whole = {
            "rubric_sha256": eff.rubric_sha256, "instructions_sha256": eff.instructions_sha256, "schema_sha256": eff.schema_sha256,
            "cases_sha256": ctx.cases_sha256(v), "model": v.judge_model, "dry_run": ctx.dry_run,
        }
        per_case = {str(c["id"]): {"case_sha256": sha256_json(c), "prompt_sha256": current_prompt_sha256(c)} for c in ctx.cases(v)}
        for path in saved_records(ctx, v):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise StopRun(1, f"{path} を読めません: {e}\n壊れた記録に、続きの試行を混ぜないため止めます。その変種の結果（{rel_path(judge_dir(ctx, v), ctx.root)}）を確かめてください。") from e
            if not isinstance(record, dict):
                raise StopRun(1, f"{path} の最上位が object ではありません。その変種の結果（{rel_path(judge_dir(ctx, v), ctx.root)}）を確かめてください。")
            case_id = path.parent.name
            expected = {**whole, **per_case.get(case_id, {})}
            differs = [key for key, value in expected.items() if record.get(key) != value]
            if differs:
                raise StopRun(
                    1,
                    f"{path} が、いまの条件と違います（違う項目: {', '.join(differs)}。{v.label} {v.name} / {case_id}）。\n"
                    "同じ変種の結果に、違う条件の試行を混ぜないため止めます。途中で rubric・conditions.toml・評価セット・用語集のどれかが変わった可能性があります。\n"
                    f"取り直すなら、その変種の結果（{rel_path(judge_dir(ctx, v), ctx.root)}）を手で退避してから、もう一度実行してください。",
                )


def preflight(ctx: Context, variants: list[Variant], args) -> None:
    """API を呼ぶ前に、止めるべきことを確かめる（上の docstring の 2〜5。1 は load_config、6 は cmd_run の最初）。"""
    labels_rels = list(dict.fromkeys(v.labels for v in variants))
    # 2. commit の確認
    if ctx.dry_run:
        print("予行（--dry-run）: 未コミットの変更の確認・STEP 3 の確認・本物のラベルの確認は飛ばします（仮のラベルを使います）")
    elif args.allow_dirty:
        print("--allow-dirty: 未コミットの変更の確認は飛ばします（結果の git.dirty には、そのまま記録されます）")
    else:
        if ctx.git["dirty"]:
            raise StopRun(1, dirty_message("結果を左右するファイル（compare.py・conditions.toml・rubric・app/・data/eval/ など）", ctx.git["dirty_files"]))
        for rel in labels_rels:
            problem = git_path_problem(ctx.root, rel)
            if problem == "missing":
                raise StopRun(1, f"{rel} がありません。tools/annotate.py でラベルを付けて、commit してから実行してください（ラベルは LLM の判定を見る前に凍結します）")
            if problem:
                why = "git が追跡していません" if problem == "untracked" else "未コミットの変更があります"
                raise StopRun(1, f"{rel} を commit してから実行してください（ラベルは LLM の判定を見る前に凍結します。{why}）")
    # 3. require_rubric_changed の確認
    if not ctx.dry_run:
        baseline = ctx.effective("baseline").rubric_base
        for v in variants:
            if v.require_rubric_changed and ctx.effective(v.name).rubric_base == baseline:
                raise StopRun(
                    1,
                    f"{v.label} {v.name}: {v.rubric} が {ctx.variants()['baseline'].rubric} と同じです。"
                    f"演習 STEP 3 で {v.rubric} を直して commit してから実行してください",
                )
    # 4. ラベルに評価セットの全 id がそろっていること。実装の評価セットの全件に回答が入っていること
    for v in variants:
        cases = ctx.cases(v)
        if v.level == "impl":
            empty = [c["id"] for c in cases if not str(c.get("answer", "")).strip()]
            if empty:
                raise StopRun(1, f"{v.cases} の {', '.join(empty)} に回答がありません。先に compare.py generate を実行してください")
        if ctx.dry_run:
            continue
        path = ctx.path(v.labels)
        if not path.is_file():
            raise StopRun(1, f"{v.labels} がありません。tools/annotate.py でラベルを付けてください")
        try:
            labels = load_labels(path)
        except (ValueError, KeyError) as e:
            raise StopRun(1, f"{v.labels} を読めません: {e}") from e
        missing = [str(c["id"]) for c in cases if c["id"] not in labels]
        if missing:
            raise StopRun(1, f"{v.labels} に評価セットの全 id がそろっていません（足りない id: {', '.join(missing)}）")
    # 5. 混ざりの確認: 同じ変種の保存済みのすべての記録と、中身が食い違っていないこと（--case で絞っても、全件を見る。labels_sha256 の違いは止めない）
    check_saved_records(ctx, variants)
    if ctx.dry_run:
        return
    if not os.environ.get("GEMINI_API_KEY"):
        raise StopRun(1, "環境変数 GEMINI_API_KEY が見つかりません（Codespaces secret に登録して、Codespaces を再起動してください）。")


def cmd_run(args) -> int:
    cfg = load_config(args.conditions)
    variants = select_variants(cfg, args)
    if not variants:
        print("条件に一致する変種がありません（--level / --variant を確かめてください）", file=sys.stderr)
        return 1
    ctx = make_context(args, cfg, dry_run=args.dry_run)
    check_variant_case_ids(ctx, variants)
    check_case_filter(ctx, csv_list(args.case), variants)
    if ctx.dry_run:
        print(f"予行（--dry-run）: API は呼びません。結果は {ctx.out} に書きます（{RESULTS_REL} と {DEFAULT_GENERATE_CASES} には書きません）", flush=True)
        prepare_dry_run(ctx, variants)
    plan = build_plan(ctx, variants, cases=csv_list(args.case), trials=args.trials)
    if not plan:
        print("条件に一致する実行がありません（--variant / --case を確かめてください）", file=sys.stderr)
        return 1
    preflight(ctx, variants, args)

    todo = [(k, v, case) for k, v, case in plan if not trial_done(ctx, v, case["id"], k)]
    total = len(plan)
    mode = "予行" if ctx.dry_run else "本番"
    if not ctx.dry_run:
        print(f"[model] 採点のモデル: {', '.join(dict.fromkeys(v.judge_model for v in variants))}  temperature=0  interval={ctx.interval:g} 秒", flush=True)
    print(f"{mode}: 変種 {len(variants)} 件、{total} 回（変種 × 評価セットの件 × 試行）のうち、済み {total - len(todo)} 回、これから {len(todo)} 回", flush=True)
    executed, counts = 0, Counter()
    for k, v, case in todo:
        if args.max_runs is not None and executed >= args.max_runs:
            print(f"--max-runs {args.max_runs} に達したので止めます。残り {len(todo) - executed} 回は、同じコマンドで続きから実行できます。")
            break
        if executed and ctx.interval > 0:
            time.sleep(ctx.interval)  # API を呼んだ後は interval 秒待つ（最後の後は待たない）
        record = run_judge_trial(ctx, ctx.effective(v.name), case, k)
        path = judge_path(ctx, v, case["id"], k)
        executed += 1
        position = f"[{total - len(todo) + executed}/{total}] {v.label} {v.name} {case['id']} t{k}"
        if not write_json_atomic(path, record, exclusive=True):  # 置く瞬間にもうあった（同時に別の実行が書いた）ので、置き換えない
            print(f"{position}: 警告: {rel_path(path, ctx.root)} がもうあるので、置き換えずに飛ばします（同時に別の実行が書いた可能性があります。この回の結果は保存していません）", file=sys.stderr, flush=True)
            continue
        counts[record["status"]] += 1
        usage = record["usage"]
        print(f"{position}: {record['status']} {record['duration_s']:.1f}s tokens={usage['input_tokens']}/{usage['output_tokens']}", flush=True)
    breakdown = "・".join(f"{status} {n}" for status, n in counts.items()) or "なし"
    print(f"\n実行 {executed} 回（{breakdown}）。保存先: {ctx.out}")
    if counts.get("invalid_output") or counts.get("error"):
        print("invalid_output・error は、結果として保存しました（出力の形式が崩れた回として数えます。再実行では飛ばされます）。")
    if ctx.dry_run:
        print(f"（予行の結果です。本物の採点ではありません。集計の動作確認: uv run python lectures/04/compare.py summarize --out {ctx.out}）")
    elif executed:
        print("集計するには: uv run python lectures/04/compare.py summarize")
    return 0


# ---------------------------------------------------------------------------
# generate / summarize / status / variants
# ---------------------------------------------------------------------------


def cmd_generate(args) -> int:
    cases_rel = rel_path(Path(args.cases), ROOT)
    guard = [cases_rel, *GENERATE_GUARDED_PATHS]
    ctx = make_context(args, {}, dry_run=args.dry_run, guard_paths=guard)
    cases_path = Path(args.cases) if Path(args.cases).is_absolute() else ROOT / args.cases
    if not cases_path.is_file():
        raise StopRun(1, f"評価セットがありません: {args.cases}（tools/annotate.py の「作問」で作ります）")
    check_case_ids(read_jsonl(cases_path), cases_rel)  # API を呼ぶ前に（結果のファイル名に使う id が、重なっていない・安全であること）
    if ctx.dry_run:
        work = ctx.out / "dry_run_inputs"
        work.mkdir(parents=True, exist_ok=True)
        copy_path = work / cases_path.name
        shutil.copyfile(cases_path, copy_path)
        print(f"予行（--dry-run）: API は呼びません。{cases_rel} のコピー（{copy_path}）に書きます。記録は {ctx.out / 'generate'}", flush=True)
        cases_path, shown = copy_path, f"{cases_rel}（予行のコピー）"
    else:
        shown = cases_rel
        check_no_answers(read_jsonl(cases_path), shown, args.force)
        if args.allow_dirty:
            print("--allow-dirty: 未コミットの変更の確認は飛ばします")
        elif ctx.git["dirty"]:
            raise StopRun(1, dirty_message("結果を左右するファイル（評価セット・app/・data/glossary.json）", ctx.git["dirty_files"]))
        if not os.environ.get("GEMINI_API_KEY"):
            raise StopRun(1, "環境変数 GEMINI_API_KEY が見つかりません（Codespaces secret に登録して、Codespaces を再起動してください）。")
        print(f"[model] {generation_model_name()}（app.config.MODEL。AIE_MODEL で切り替え）  評価セット: {cases_rel}", flush=True)
    code = generate_cases(ctx, cases_path, force=args.force, shown=shown)
    if code == 0 and not ctx.dry_run:
        print("次の手順: ① git add / commit で回答を固定 → ② tools/annotate.py でラベルを付け、commit（ラベルの凍結）→ ③ compare.py run --level impl")
    return code


def cmd_summarize(args) -> int:
    report_file = HERE / "report.py"
    if not report_file.is_file():
        print(f"report.py がありません（{report_file}）", file=sys.stderr)
        return 1
    spec = importlib.util.spec_from_file_location("lec04_report", report_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # exec_module の前に登録する（dataclass が、定義のとき自分のモジュールを sys.modules から引くため）
    added = str(HERE) not in sys.path
    if added:
        sys.path.insert(0, str(HERE))  # report.py が、同じフォルダのファイルを import しても動くように（読み込む間だけ）
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)  # 読み込みに失敗したモジュールを残さない
        raise
    finally:
        if added:
            sys.path.remove(str(HERE))
    out = Path(args.out) if args.out else ROOT / RESULTS_REL
    return int(module.summarize(ROOT, conditions_path(args.conditions), out))


def display_width(text: str) -> int:
    """端末での表示幅（全角の文字は 2）。表の桁をそろえるため。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def print_table(rows: list[tuple]) -> None:
    widths = [max(display_width(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(str(cell) + " " * (width - display_width(str(cell))) for cell, width in zip(r, widths, strict=True)).rstrip())


NO_EVALSET = "評価セットがありません"


def cmd_status(args) -> int:
    cfg = load_config(args.conditions)
    ctx = make_context(args, cfg, with_git=False)
    variants = select_variants(cfg, args)
    check_case_filter(ctx, csv_list(args.case), variants, skip_missing=True)  # 選んだ変種の評価セットだけを見る。無い変種は、下の表で「評価セットがありません」と出す
    rows = [("レベル", "変種", "件数", "試行", "予定", "済み", "残り", "残りの推定リクエスト")]
    total_planned = total_done = total_requests = 0
    missing_sets: dict[str, None] = {}
    for v in variants:
        try:
            all_cases = ctx.cases(v)
        except ConfigError:  # 評価セットのファイルが無い（実装の my_cases.jsonl は、作問するまで無い）
            missing_sets[v.cases] = None
            rows.append((v.level, f"{v.label} {v.name}", NO_EVALSET, "-", "-", "-", "-", "-"))
            continue
        cases = [c for c in all_cases if not csv_list(args.case) or str(c["id"]) in csv_list(args.case)]
        trials = trials_of(v, args.trials)
        planned = len(cases) * trials
        done = sum(trial_done(ctx, v, c["id"], k) for c in cases for k in range(1, trials + 1))
        requests = (planned - done) * JUDGE_REQUESTS
        rows.append((v.level, f"{v.label} {v.name}", len(cases), trials, planned, done, planned - done, requests))
        total_planned, total_done, total_requests = total_planned + planned, total_done + done, total_requests + requests
    impl = [v for v in variants if v.level == "impl"]
    if impl:  # 回答の生成（実装）
        try:
            cases = ctx.cases(impl[0])
        except ConfigError:
            missing_sets[impl[0].cases] = None
            rows.append(("impl", "generate（回答の生成）", NO_EVALSET, "-", "-", "-", "-", "-"))
        else:
            ok = 0
            for c in cases:
                record = read_generate_record(generate_path(ctx, c["id"]))
                ok += int(record is not None and record.get("status") == "ok")
            requests = (len(cases) - ok) * GENERATE_REQUESTS
            rows.append(("impl", "generate（回答の生成）", len(cases), "-", len(cases), ok, len(cases) - ok, requests))
            total_planned, total_done, total_requests = total_planned + len(cases), total_done + ok, total_requests + requests
    rows.append(("合計", "", "", "", total_planned, total_done, total_planned - total_done, total_requests))
    print(f"結果の置き場所: {ctx.out}")
    print(f"残りの推定リクエスト = 残りの実行回数 × {JUDGE_REQUESTS}（採点）、generate は 1 件 × {GENERATE_REQUESTS}。無料枠の 1 日の上限は AI Studio の画面で確かめる")
    print_table(rows)
    for rel in missing_sets:
        print(f"{NO_EVALSET}: {rel}（tools/annotate.py の「作問」で作ります）")
    print("\nラベル（run の前に commit して凍結する）:")
    for rel in dict.fromkeys(v.labels for v in variants):
        problem = git_path_problem(ctx.root, rel)
        print(f"  {rel}: " + {None: "commit 済み", "missing": "まだありません", "untracked": "あるが、git が追跡していない", "dirty": "未コミットの変更がある"}[problem])
    return 0


def tail_lines(text: str, n: int = 3) -> list[str]:
    return [line for line in text.splitlines() if line.strip()][-n:]


def schema_order_text(eff: Effective) -> str:
    """スキーマの順。tool は $defs.CriterionVerdict の properties、native は各基準の properties（と propertyOrdering の有無）。"""
    if eff.variant.output == "native":
        criteria = eff.schema["properties"]
        orders = {" → ".join(item["properties"]) for item in criteria.values()}
        with_ordering = all("propertyOrdering" in item for item in criteria.values()) and "propertyOrdering" in eff.schema
        top = " → ".join(eff.schema["properties"])
        return f"各基準: {' / '.join(sorted(orders))}{'（propertyOrdering 付き）' if with_ordering else '（propertyOrdering なし）'}／最上位: {top}"
    return f"$defs.CriterionVerdict: {' → '.join(eff.schema['$defs']['CriterionVerdict']['properties'])}"


def cmd_variants(args) -> int:
    cfg = load_config(args.conditions)
    ctx = make_context(args, cfg, with_git=False)
    variants = select_variants(cfg, args)
    base = ctx.effective("baseline")
    defaults = ctx.variants()["baseline"]
    base_lines = base.rubric.splitlines()
    for v in variants:
        eff = ctx.effective(v.name)
        print(f"=== {v.label} {v.name}（{v.level}）: {v.description} ===")
        print(f"  モデル: {v.judge_model}")
        print(f"  出力: {v.output}（{'ツール出力' if v.output == 'tool' else 'NativeOutput'}）  順序: {v.order}（{ORDER_NAMES[v.order]}）  temperature: 0.0  試行: {v.trials} 回")
        print(f"  rubric: {v.rubric}（sha256 {eff.rubric_sha256[:8]}）" + (f"  changes: {len(v.changes)} 件" if v.changes else ""))
        if v.cases != defaults.cases or v.labels != defaults.labels:
            try:
                cases = ctx.cases(v)
            except ConfigError:  # 評価セットのファイルが無い（実装の my_cases.jsonl は、作問するまで無い）
                print(f"  評価セット: {v.cases}（{NO_EVALSET}。tools/annotate.py の「作問」で作ります）  ラベル: {v.labels}")
            else:
                empty = sum(1 for c in cases if not str(c.get("answer", "")).strip())
                print(f"  評価セット: {v.cases}（{len(cases)} 件" + (f"。回答が空 {empty} 件: 先に compare.py generate" if empty else "") + f"）  ラベル: {v.labels}")
        if v.require_rubric_changed and eff.rubric_base == base.rubric_base:
            print(f"  ※ {v.rubric} はまだ {defaults.rubric} と同じです（演習 STEP 3 で直して commit するまで、run はこの変種で止まります）")
        diff = list(difflib.unified_diff(base_lines, eff.rubric.splitlines(), defaults.rubric, f"{v.label} {v.name} の rubric", lineterm=""))
        if diff:
            print("  rubric の baseline.md からの差分:\n" + "\n".join("    " + line for line in diff))
        print("  instructions の末尾 3 行:")
        for line in tail_lines(eff.instructions):
            print(f"    | {line}")
        print(f"  スキーマの順: {schema_order_text(eff)}")
        if v.name == "baseline":
            print("  V0 からの変更: なし（基準）")
        else:
            items = changed_items(base, eff)
            print(f"  V0 からの変更: {len(items)} 点" + (f"（{'・'.join(items)}）" if items else ""))
        print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--conditions", help=f"条件のファイル（既定: {CONDITIONS_REL}）")
    filters.add_argument("--level", nargs="+", metavar="LEVEL", help="exercise basic impl（カンマか空白で区切って複数可。既定: all）")
    filters.add_argument("--variant", nargs="+", metavar="VARIANT", help="変種の名前か label（V0 など。複数可。絞り込み）")

    selecting = argparse.ArgumentParser(add_help=False, parents=[filters])
    selecting.add_argument("--case", nargs="+", metavar="ID", help="評価セットの id（複数可。絞り込み）。例: 01 m03")
    selecting.add_argument("--trials", type=int, help="試行の回数の上限（既定: 各変種の trials。それより多くはしない）")
    selecting.add_argument("--out", help=f"結果の置き場所（既定: {RESULTS_REL}。--dry-run は一時フォルダ）")

    parser = argparse.ArgumentParser(description="採点の条件を変えて、LLM-as-a-Judge の採点を比べる（実行）。集計は summarize（report.py）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("variants", parents=[filters], help="変種ごとの違い（モデル・rubric・指示の末尾・スキーマの順）を表示（API なし）")
    p.set_defaults(func=cmd_variants)

    p = sub.add_parser("status", parents=[selecting], help="済み／全体の実行回数と、残りのリクエストの見積もり（API なし）")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", parents=[selecting], help="評価セットの全件を LLM に採点させる（既にある結果は飛ばす）")
    p.add_argument("--interval", type=float, default=8, help="API を呼んだ後に待つ秒数（既定 8。予行では待たない）")
    p.add_argument("--max-runs", type=int, help="この実行で行う回数の上限（試しに少しだけ流すとき）")
    p.add_argument("--dry-run", action="store_true", help="API を呼ばずに予行する（偽のモデル。結果は一時フォルダ）")
    p.add_argument("--allow-dirty", action="store_true", help="未コミットの変更があっても実行する（ラベルの commit の確認も飛ばす。ラベルの中身の確認は飛ばさない）")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("generate", help="実装: 評価セットの回答を、自分のエージェント（app.agent）で作る")
    p.add_argument("--cases", default=DEFAULT_GENERATE_CASES, help=f"評価セット（既定: {DEFAULT_GENERATE_CASES}）")
    p.add_argument("--force", action="store_true", help="回答が入っていても作り直す（既存のラベルとずれる）")
    p.add_argument("--interval", type=float, default=8, help="1 件ごとに待つ秒数（既定 8。予行では待たない）")
    p.add_argument("--dry-run", action="store_true", help="API を呼ばずに予行する（TestModel。一時フォルダのコピーに書く）")
    p.add_argument("--allow-dirty", action="store_true", help="未コミットの変更があっても実行する")
    p.add_argument("--out", help=f"記録の置き場所（既定: {RESULTS_REL}。--dry-run は一時フォルダ）")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("summarize", help="report.py の summarize で集計する")
    p.add_argument("--out", help=f"結果の置き場所（既定: {RESULTS_REL}）")
    p.add_argument("--conditions", help=f"条件のファイル（既定: {CONDITIONS_REL}）")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("trials", "max_runs"):
        if getattr(args, name, None) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} は 1 以上にしてください")
    args._argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        return args.func(args)
    except StopRun as e:
        print(e.message, file=sys.stderr)
        return e.code
    except ConfigError as e:
        print(f"設定エラー: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました。同じコマンドで続きから実行できます。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
