#!/usr/bin/env python3
"""第5回: 検索の方式と取得処理を変えて、講義資料に答える RAG エージェントの動きを比べる実行・集計スクリプト。

何を比べるか（質問・条件・自動チェック）は conditions.toml に書いてある。
このスクリプトは、それを読んで実行し、結果を results/ に保存し、集計する。

使い方（リポジトリの直下で）:
    uv run python lectures/05/compare.py status                          # 条件×質問ごとの済み・予定と、残りの LLM リクエストの目安（API なし）
    uv run python lectures/05/compare.py variants                        # 条件ごとに、モデルに見える指示・ツール・取得処理・設定を保存し、基準との差分を表示（API なし）
    uv run python lectures/05/compare.py run --level all --dry-run       # 偽のモデルと偽の埋め込みで予行（API なし。結果は一時フォルダ）
    uv run python lectures/05/compare.py run --level exercise --trials 1 # 演習の条件を 1 回ずつ
    uv run python lectures/05/compare.py run --level all --trials 3      # 2・3 回目を足す（既にある結果は飛ばす）
    uv run python lectures/05/compare.py summarize                       # results/summary.md と summary.json を作る

共通のオプション: --out（結果の置き場所）・--conditions・--level（exercise|basic|impl|adv|all。, 区切りで複数可）・
    --variant（条件の名前か label。, 区切りで複数可）・--question・--trials（既定 3）。
    run だけ: --interval（既定 8 秒）・--max-runs・--dry-run・--allow-dirty（開発用）・--tpm-budget（既定 200000。0 で無効）。
順番は、試行番号（外）→ 質問 → 条件。条件の順は conditions.toml の order を、試行ごとに order_shift 個ずつ巡回させる
（時間帯や API の混み具合の影響と、条件の効果を区別しやすくするため）。止まっても、同じコマンドで続きから実行できる。

保存先: <out>/<model>/<variant>/<qid>/t<k>.json と t<k>.trace.txt。条件の見え方は <out>/variants/<variant>.json と .txt。
        t<k>.json があれば「済み」。成功したとき（status が ok・input_limit・usage_limit・unexpected）だけ、
        同じフォルダの .tmp- で始まる一時ファイルに書いてから os.replace で置く（trace → json の順。json は、あれば置き換えない）。
        API の失敗で止まったものは保存しない。予行（--dry-run）は <model>-dry-run のフォルダに書くので、本物の結果とは混ざらない。
        STEP 1〜4 の CLI の出力は <out>/cli/<名前>.txt に tee で置く（ランナーは作らない）。

実行の前に、API を呼ばずに確かめること（どれかに当たれば終了コード 1 で止まる）:
    1. app/・data/rag_data/・compare.py・conditions.toml・pyproject.toml・uv.lock の未コミットの変更（--allow-dirty と --dry-run では飛ばす）
    2. 条件の解決（conditions.toml の誤り・--variant や --question の綴り違い）
    3. 保存済みの記録の条件の sha256（条件・app/ のコード・データ）と質問の文が、今と違わないか（違えば、別の --out に書く）
    4. 環境変数 GEMINI_API_KEY（予行では不要）
    5. 環境変数 AIE_RAG_DB が設定されていないか（同梱の検索 DB を使うため）
    6. 検索を使う条件があれば、検索 DB が開けて、chunks が 214 行・3 資料あるか

終了コード: 0 = 正常 / 1 = 止まった（設定の誤り・API エラー・未コミットの変更など）/ 2 = 1 日の上限（PerDay）/ 130 = Ctrl+C。

秘密: 書き出す前に、GEMINI_API_KEY・GOOGLE_API_KEY の値と AIza… の形の文字列を *** に置き換える。
      環境変数の一覧は記録しない。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import difflib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import types
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # 「python lectures/05/compare.py」で動かすと、app/ が見つからないため
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # pydantic-ai の初回バナー（9 行）を出さない

from pydantic_ai import Tool, capture_run_messages  # noqa: E402
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded  # noqa: E402
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
from pydantic_ai.usage import RunUsage, UsageLimits  # noqa: E402

CONDITIONS_FILE = HERE / "conditions.toml"
RESULTS_REL = "lectures/05/results"
DEFAULT_OUT = HERE / "results"

LEVELS = ("exercise", "basic", "impl", "adv")
MODES = ("full", "vector", "tree", "graph")
ROLES = ("required", "bonus", "flag", "reference")
OK_STATUSES = ("ok", "input_limit", "usage_limit", "unexpected")  # 結果として保存する status

# conditions.toml の settings で書き換えてよい設定（名前 → その設定を持つモジュール）。試行の間だけ書き換えて、終われば戻す
SETTING_MODULES = {
    "RAG_TOP_K": "app.retrieval",
    "GRAPH_START_K": "app.retrieval",
    "GRAPH_HOPS": "app.retrieval",
    "GRAPH_MAX_RESULTS": "app.retrieval",
    "NEIGHBOR_WINDOW": "app.retrieval_lec05",
    "TERM_GRAPH_TOP_K": "app.retrieval_lec05",
    "TERM_GRAPH_SECTION_CAP": "app.retrieval_lec05",
    "TERM_GRAPH_VECTOR_POOL": "app.retrieval_lec05",
    "TERM_GRAPH_HUB_TERMS": "app.retrieval_lec05",
}
RETRIEVAL_LEC05 = "app.retrieval_lec05"  # TRACE（取得の内訳）を持つモジュール
# 実行結果を左右するコード。ファイル全体の sha256 を、条件の sha256 に入れる（関数単位の追跡が、呼び出しの先まで届かないため）
CODE_ROOT = REPO
CODE_FILES = ("app/agent.py", "app/tools.py", "app/retrieval.py", "app/config.py")  # 全条件
LEC05_FILE = "app/retrieval_lec05.py"  # 取得処理か設定でこのモジュールを使う条件だけ

TOP_KEYS = ("defaults", "questions", "variants", "levels", "fragments")
DEFAULT_KEYS = ("trials", "interval", "request_limit", "tpm_budget", "order", "order_shift")
QUESTION_KEYS = ("text", "gold", "acceptable", "note", "checks")
CHECK_KEYS = ("label", "kind", "role", "pattern", "values", "name", "args_contains", "after", "min", "max", "value", "note")
VARIANT_KEYS = ("label", "level", "description", "mode", "base", "documents", "retrieval", "changes", "settings", "request_limit", "questions")
LEVEL_KEYS = ("reference", "description")
CHANGE_KEYS = ("target", "op", "old", "new")

RATE_LIMIT_RETRIES = 5  # 429 の再試行の回数
RATE_LIMIT_DEFAULT_WAIT = 60  # 429 で待ち時間が分からないときの秒数
SERVER_ERROR_WAITS = (30, 60, 120)  # 503・500・タイムアウト・接続の失敗の再試行の前に待つ秒数
TPM_WINDOW = 60  # 入力トークンの予算を数える直近の秒数
TPM_CHARS_PER_TOKEN = 0.6  # 既存の記録が無いときの、prompt の文字数 → 入力トークンの見積もり
GIT_TIMEOUT = 60  # git が 1 回の確認で使える秒数（入力待ちなどで、止まったままにならないように）
MAX_PAGE_RANGE = 20  # これより広いページの範囲（p.1-50 など）は、引用として数えない
GUARDED_PATHS = ["app", "data/rag_data", "lectures/05/compare.py", "lectures/05/conditions.toml", "pyproject.toml", "uv.lock"]
# 残りの LLM リクエスト数の目安（1 回の実行あたり）
ESTIMATED_REQUESTS = {"full": 1, "vector": 2.5, "tree": 6, "graph": 2.5}
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


# app/ のモジュールは、使うときに読み込む（compare.py を import しただけでは、読み込まない）


def app_agent():
    from app import agent

    return agent


def app_retrieval():
    from app import retrieval

    return retrieval


def app_tools():
    from app import tools

    return tools


def app_config():
    from app import config

    return config


def setting_module(name: str):
    """設定 name を持つモジュール。読み込めなければ ConfigError。"""
    module_name = SETTING_MODULES[name]
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001  モジュールが無い・壊れている
        raise ConfigError(f"設定 {name} を持つ {module_name} を読み込めません: {type(e).__name__}: {e}") from e


# ---------------------------------------------------------------------------
# ページの一覧（data/rag_data/lectures.json。全 9 資料 438 ページ）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PageIndex:
    titles: dict  # 資料 ID → 資料名
    pages: dict  # 資料 ID → {ページ番号: page_id}
    page_ids: frozenset
    chars: dict  # page_id → 本文の字数


def doc_lecture(document_id: str) -> int | None:
    """資料 ID（lec02-main など）の回。orientation などは None。"""
    match = re.fullmatch(r"lec(\d{2})-(?:main|exercise)", document_id)
    return int(match.group(1)) if match else None


def doc_kind(document_id: str) -> str:
    """main（本講義）・exercise（演習）・orientation・other。"""
    if document_id.endswith("-main"):
        return "main"
    if document_id.endswith("-exercise"):
        return "exercise"
    return "orientation" if document_id == "orientation" else "other"


@lru_cache(maxsize=4)
def _read_page_index(path: str, mtime_ns: int) -> PageIndex:
    documents = json.loads(Path(path).read_text(encoding="utf-8"))
    titles, pages, chars = {}, {}, {}
    for document in documents:
        titles[document["id"]] = document["title"]
        pages[document["id"]] = {}
        for section in document["content"]:
            for page in section["content"]:
                pages[document["id"]][page["page"]] = page["id"]
                chars[page["id"]] = len(page["content"])
    return PageIndex(titles, pages, frozenset(chars), chars)


def page_index() -> PageIndex:
    source = Path(app_config().RAG_SOURCE)
    return _read_page_index(str(source), source.stat().st_mtime_ns)


# ---------------------------------------------------------------------------
# チェック: 集計のときに判定する（種類は CHECKS に足せる）
# ---------------------------------------------------------------------------


def normalize_answer(text: str | None) -> str:
    """回答を、全角・半角・大文字・小文字・空白の違いをなくした形にする（パターンは、この形に re.search する）。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").casefold())


def args_json(args) -> str:
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


def calls_named(rec: dict, name: str) -> list[dict]:
    return [c for c in rec.get("tool_calls", []) if c.get("name") == name]


def in_range(n: int, check: dict) -> bool:
    if "value" in check:
        return n == check["value"]
    return check.get("min", 0) <= n <= check.get("max", 10**9)


def check_tool_called(check: dict, rec: dict) -> bool:
    calls = rec.get("tool_calls", [])
    after = check.get("after")
    if after:
        firsts = [c for c in calls if c.get("name") == after]
        if not firsts:
            return False
        first = firsts[0].get("response_index", firsts[0].get("order", 0))
        calls = [c for c in calls if c.get("response_index", c.get("order", 0)) > first]
    needle = check.get("args_contains")
    return any(c.get("name") == check["name"] and (needle is None or needle in args_json(c.get("args"))) for c in calls)


def answer_matches(check: dict, rec: dict) -> bool:
    return re.search(check["pattern"], normalize_answer(rec["answer"])) is not None


CHECKS = {
    "answer_regex": answer_matches,
    "answer_not_regex": lambda c, r: not answer_matches(c, r),
    "answer_contains_all": lambda c, r: all(normalize_answer(v) in normalize_answer(r["answer"]) for v in c["values"]),
    "answer_contains_any": lambda c, r: any(normalize_answer(v) in normalize_answer(r["answer"]) for v in c["values"]),
    "answer_not_contains": lambda c, r: not any(normalize_answer(v) in normalize_answer(r["answer"]) for v in c["values"]),
    "tool_called": check_tool_called,
    "tool_not_called": lambda c, r: not calls_named(r, c["name"]),
    "tool_count": lambda c, r: in_range(len(calls_named(r, c["name"])), c),
    "tool_result_contains": lambda c, r: all(v in "\n".join(str(x.get("result") or "") for x in calls_named(r, c["name"])) for v in c["values"]),
    "requests_at_most": lambda c, r: (r.get("usage") or {}).get("requests", 10**9) <= c["value"],
}
REGEX_KINDS = ("answer_regex", "answer_not_regex")
VALUES_KINDS = ("answer_contains_all", "answer_contains_any", "answer_not_contains", "tool_result_contains")


def check_label(check: dict) -> str:
    return check["label"]


def check_role(check: dict) -> str:
    return check.get("role", "required")


def checks_for(cfg: dict, qid: str) -> list[dict]:
    return list(cfg["questions"][qid].get("checks", []))


def evaluate_check(check: dict, rec: dict) -> bool:
    """チェックを判定する（True = 条件を満たす）。"""
    rec = {**rec, "answer": rec.get("answer") or ""}
    return bool(CHECKS[check["kind"]](check, rec))


VERDICTS = ("correct", "review", "wrong")  # 自動の正答・要確認・自動の誤答
VERDICT_NAMES = {"correct": "自動の正答", "review": "要確認", "wrong": "自動の誤答"}


def score_trial(checks: list[dict], rec: dict) -> dict:
    """1 回の試行のチェックの結果。

    results: label → 通ったか。
    verdict: 自動の判定を 3 つに分ける（required が無ければ None）。
      correct（自動の正答）: required がすべて一致し、flag が一致しない。
      review（要確認）: required がすべて一致し、flag も一致した。目視で決める。正答にも誤答にも数えない。
      wrong（自動の誤答）: required のどれかが一致しない。flag は問わない。
    correct: verdict が correct か（required が無ければ None）。
    flagged: flag のチェックが 1 つでも通らなかった（= flag のパターンに一致した）。flag が無ければ None。
    """
    results = {check_label(c): evaluate_check(c, rec) for c in checks}
    roles = {check_label(c): check_role(c) for c in checks}
    required = [results[label] for label, role in roles.items() if role == "required"]
    flags = [results[label] for label, role in roles.items() if role == "flag"]
    flagged = any(not ok for ok in flags) if flags else None
    if not required:
        verdict = None
    elif not all(required):
        verdict = "wrong"
    elif flagged:
        verdict = "review"
    else:
        verdict = "correct"
    return {"results": results, "roles": roles, "verdict": verdict, "correct": None if verdict is None else verdict == "correct", "flagged": flagged}


# ---------------------------------------------------------------------------
# 条件（conditions.toml）の読み込みと検証
# ---------------------------------------------------------------------------


def load_config(path: Path | str | None = None) -> dict:
    """conditions.toml を読み、形を確かめる（条件の当て方・取得処理の import・スキーマまで確かめる）。"""
    path = Path(path) if path else CONDITIONS_FILE
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


FRAGMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
FRAGMENT_REFERENCE = re.compile(r"<<([A-Za-z0-9_]+)>>")


def expand_fragments(text: str, fragments: dict, where: str, _stack: tuple = ()) -> str:
    """正規表現の中の <<名前>> を、[fragments] の部品（(?:…) で包んだもの）に置き換える。部品の中の <<名前>> も置き換える（入れ子）。"""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in fragments:
            raise ConfigError(f"{where}: 知らない部品 <<{name}>> です（使えるもの: {', '.join(fragments) or 'なし'}）")
        if name in _stack:
            raise ConfigError(f"{where}: 部品 <<{name}>> が循環しています（{' → '.join((*_stack, name))}）")
        return "(?:" + expand_fragments(fragments[name], fragments, where, (*_stack, name)) + ")"

    return FRAGMENT_REFERENCE.sub(replace, text)


def validate_fragments(fragments) -> dict:
    """[fragments]（チェックの正規表現に使う共通の部品）を検証する。部品は、展開して正規表現として読めること。"""
    require_table("fragments", fragments)
    for name, text in fragments.items():
        if not FRAGMENT_NAME.fullmatch(name):
            raise ConfigError(f"fragments.{name}: 名前は、大文字・数字・_ だけにしてください（例: NEG_SURU）")
        if not isinstance(text, str) or not text:
            raise ConfigError(f"fragments.{name}: 空でない文字列にしてください")
    for name, text in fragments.items():
        try:
            re.compile(expand_fragments(text, fragments, f"fragments.{name}", (name,)))
        except re.error as e:
            raise ConfigError(f"fragments.{name} を正規表現として読めません: {e}") from e
    return fragments


def require_table(where: str, value) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: 表（テーブル）にしてください")
    return value


def require_str_list(where: str, value, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{where}: 空でない文字列の配列にしてください: {value!r}")
    if not allow_empty and not value:
        raise ConfigError(f"{where}: 1 つ以上書いてください")
    if len(value) != len(set(value)):
        raise ConfigError(f"{where}: 同じものが重なっています: {value!r}")
    return value


def require_int(where: str, value, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where}: {minimum} 以上の整数にしてください: {value!r}")
    return value


def validate_defaults(defaults) -> None:
    require_table("defaults", defaults)
    check_keys("defaults", defaults, DEFAULT_KEYS)
    for key in DEFAULT_KEYS:
        if key not in defaults:
            raise ConfigError(f"defaults.{key} がありません")
    require_int("defaults.trials", defaults["trials"], minimum=1)
    require_int("defaults.request_limit", defaults["request_limit"], minimum=1)
    require_int("defaults.tpm_budget", defaults["tpm_budget"])
    require_int("defaults.order_shift", defaults["order_shift"])
    interval = defaults["interval"]
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or interval < 0:
        raise ConfigError(f"defaults.interval: 0 以上の数にしてください: {interval!r}")
    require_str_list("defaults.order", defaults["order"], allow_empty=False)


def validate_check(where: str, check) -> None:
    require_table(where, check)
    check_keys(where, check, CHECK_KEYS)
    for key in ("label", "kind"):
        if not isinstance(check.get(key), str) or not check[key]:
            raise ConfigError(f"{where}.{key} がありません")
    if check["kind"] not in CHECKS:
        raise ConfigError(f"{where}: 知らないチェックの種類です: {check['kind']!r}（種類は {sorted(CHECKS)}）")
    if check_role(check) not in ROLES:
        raise ConfigError(f"{where}.role: {', '.join(ROLES)} のどれかです: {check.get('role')!r}")
    kind = check["kind"]
    if kind in REGEX_KINDS:
        if not isinstance(check.get("pattern"), str) or not check["pattern"]:
            raise ConfigError(f"{where}: {kind} には pattern が必要です")
        try:
            re.compile(check["pattern"])
        except re.error as e:
            raise ConfigError(f"{where}.pattern を正規表現として読めません: {e}") from e
    if kind in VALUES_KINDS:
        require_str_list(f"{where}.values", check.get("values"), allow_empty=False)
    if kind.startswith("tool_") and not isinstance(check.get("name"), str):
        raise ConfigError(f"{where}: {kind} には name が必要です")
    if kind == "requests_at_most":
        require_int(f"{where}.value", check.get("value"), minimum=1)


def validate_questions(questions, fragments: dict | None = None) -> None:
    """質問と、そのチェックを検証する。チェックの pattern の <<名前>> は、ここで [fragments] の部品に展開する（questions を書き換える）。"""
    require_table("questions", questions)
    if not questions:
        raise ConfigError("[questions] がありません")
    known = page_index().page_ids
    for qid, item in questions.items():
        where = f"questions.{qid}"
        require_table(where, item)
        check_keys(where, item, QUESTION_KEYS)
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ConfigError(f"{where}.text がありません")
        gold = require_str_list(f"{where}.gold", item.get("gold", []))
        acceptable = require_str_list(f"{where}.acceptable", item.get("acceptable", []))
        for key, ids in (("gold", gold), ("acceptable", acceptable)):
            missing = [page_id for page_id in ids if page_id not in known]
            if missing:
                raise ConfigError(f"{where}.{key}: data/rag_data/lectures.json にないページです: {', '.join(missing)}")
        overlap = sorted(set(gold) & set(acceptable))
        if overlap:
            raise ConfigError(f"{where}: gold と acceptable に同じページがあります: {', '.join(overlap)}")
        checks = item.get("checks", [])
        if not isinstance(checks, list):
            raise ConfigError(f"{where}.checks: 配列にしてください")
        labels = []
        for i, check in enumerate(checks, 1):
            if isinstance(check, dict) and isinstance(check.get("pattern"), str):
                check["pattern"] = expand_fragments(check["pattern"], fragments or {}, f"{where}.checks[{i}].pattern")
            validate_check(f"{where}.checks[{i}]", check)
            labels.append(check_label(check))
        if len(labels) != len(set(labels)):
            raise ConfigError(f"{where}: チェックの label が重なっています（表で区別できません）")


def resolve_spec(cfg: dict, name: str, _stack: tuple = ()) -> dict:
    """条件 name を、base をたどって解決した設定（mode・documents・request_limit・retrieval・changes・settings）にする。

    mode・documents・request_limit は、自分に書いてあればそれ、なければ base のもの。
    retrieval と settings は、キーごとに base のものへ重ねる。changes は、base のものの後に自分のものを足す。
    """
    if name in _stack:
        raise ConfigError(f"variants.{name}: base が循環しています（{' → '.join((*_stack, name))}）")
    spec = cfg["variants"][name]
    parent = resolve_spec(cfg, spec["base"], (*_stack, name)) if spec.get("base") else {}
    return {
        "mode": spec.get("mode", parent.get("mode")),
        "documents": spec.get("documents", parent.get("documents")),
        "request_limit": spec.get("request_limit", parent.get("request_limit", cfg["defaults"]["request_limit"])),
        "retrieval": {**parent.get("retrieval", {}), **spec.get("retrieval", {})},
        "changes": [*parent.get("changes", []), *spec.get("changes", [])],
        "settings": {**parent.get("settings", {}), **spec.get("settings", {})},
    }


def validate_variant_shape(cfg: dict, name: str, spec) -> None:
    where = f"variants.{name}"
    require_table(where, spec)
    check_keys(where, spec, VARIANT_KEYS)
    for key in ("label", "level"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ConfigError(f"{where}.{key} がありません")
    if spec["level"] not in LEVELS:
        raise ConfigError(f"{where}.level: {', '.join(LEVELS)} のどれかです: {spec['level']!r}")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ConfigError(f"{where}.description: 文字列にしてください")
    if "mode" in spec and spec["mode"] not in MODES:
        raise ConfigError(f"{where}.mode: {', '.join(MODES)} のどれかです: {spec['mode']!r}")
    if "mode" not in spec and not spec.get("base"):
        raise ConfigError(f"{where}: mode か base が必要です")
    if spec.get("base") and spec["base"] not in cfg["variants"]:
        raise ConfigError(f"{where}.base: 条件 {spec['base']!r} が variants にありません")
    if "documents" in spec:
        documents = spec["documents"]
        if documents != "all":
            require_str_list(f"{where}.documents", documents, allow_empty=False)
            unknown = [d for d in documents if d not in page_index().titles]
            if unknown:
                raise ConfigError(f"{where}.documents: data/rag_data/lectures.json にない資料です: {', '.join(unknown)}")
    if "request_limit" in spec:
        require_int(f"{where}.request_limit", spec["request_limit"], minimum=1)
    retrieval = spec.get("retrieval", {})
    require_table(f"{where}.retrieval", retrieval)
    for tool_name, target in retrieval.items():
        if not isinstance(target, str) or target.count(":") != 1 or not all(target.split(":")):
            raise ConfigError(f"{where}.retrieval.{tool_name}: 'module:attr' の形にしてください: {target!r}")
    settings = spec.get("settings", {})
    require_table(f"{where}.settings", settings)
    for key, value in settings.items():
        if key not in SETTING_MODULES:
            raise ConfigError(f"{where}.settings.{key}: 書き換えられる設定ではありません（使えるもの: {', '.join(SETTING_MODULES)}）")
        require_int(f"{where}.settings.{key}", value)
    changes = spec.get("changes", [])
    if not isinstance(changes, list):
        raise ConfigError(f"{where}.changes: 配列にしてください")
    for i, change in enumerate(changes, 1):
        require_table(f"{where}.changes[{i}]", change)
        check_keys(f"{where}.changes[{i}]", change, CHANGE_KEYS)
        if change.get("op") not in ("delete_line", "replace_line", "replace_text"):
            raise ConfigError(f"{where}.changes[{i}].op は delete_line・replace_line・replace_text のどれかです: {change.get('op')!r}")
        if not isinstance(change.get("old"), str) or not change["old"] or "target" not in change:
            raise ConfigError(f"{where}.changes[{i}]: target と old が必要です")
        if change["op"] != "delete_line" and not isinstance(change.get("new"), str):
            raise ConfigError(f"{where}.changes[{i}]: {change['op']} には new が必要です")
    questions = require_str_list(f"{where}.questions", spec.get("questions"), allow_empty=False)
    for qid in questions:
        if qid not in cfg["questions"]:
            raise ConfigError(f"{where}.questions: 質問 {qid} が questions にありません")


def validate_variants(cfg: dict) -> None:
    variants = cfg.get("variants")
    require_table("variants", variants)
    if not variants:
        raise ConfigError("[variants] がありません")
    for name, spec in variants.items():
        validate_variant_shape(cfg, name, spec)
    owners: dict[str, str] = {}
    for name, spec in variants.items():
        label = spec["label"]
        if label in owners:
            raise ConfigError(f"variants.{name}.label {label!r} が {owners[label]} と重なっています（label は条件ごとに別にしてください）")
        owners[label] = name
    for name, spec in variants.items():
        if spec["label"] in variants and spec["label"] != name:
            raise ConfigError(f"variants.{name}.label {spec['label']!r} が、別の条件の名前と同じです（--variant で区別できません）")
    for name in variants:
        resolved = resolve_spec(cfg, name)  # base の循環はここで見つかる
        where = f"variants.{name}"
        if resolved["mode"] is None:
            raise ConfigError(f"{where}: mode が決まりません（base をたどっても mode がありません）")
        if resolved["documents"] is not None and resolved["mode"] != "full":
            raise ConfigError(f"{where}.documents は mode = 'full' の条件だけに書けます")
        if resolved["mode"] == "full" and resolved["retrieval"]:
            raise ConfigError(f"{where}.retrieval: mode = 'full' の条件には、差し替えるツールがありません")


def validate_levels(cfg: dict) -> None:
    levels = cfg.get("levels", {})
    require_table("levels", levels)
    for level, spec in levels.items():
        where = f"levels.{level}"
        if level not in LEVELS:
            raise ConfigError(f"{where}: レベルは {', '.join(LEVELS)} のどれかです")
        require_table(where, spec)
        check_keys(where, spec, LEVEL_KEYS)
        for name in require_str_list(f"{where}.reference", spec.get("reference", [])):
            if name not in cfg["variants"]:
                raise ConfigError(f"{where}.reference: 条件 {name!r} が variants にありません")
            if cfg["variants"][name]["level"] == level:
                raise ConfigError(f"{where}.reference: {name!r} は、このレベルの条件です（reference は、比べる相手の行です）")


def validate_order(cfg: dict) -> None:
    order = cfg["defaults"]["order"]
    names = list(cfg["variants"])
    unknown = [n for n in order if n not in cfg["variants"]]
    if unknown:
        raise ConfigError(f"defaults.order に、variants にない名前があります: {', '.join(unknown)}")
    missing = [n for n in names if n not in order]
    if missing:
        raise ConfigError(f"defaults.order に、条件が足りません（全部の条件を 1 回ずつ書いてください）: {', '.join(missing)}")
    modes = [resolve_spec(cfg, n)["mode"] for n in order]
    for i, name in enumerate(order):
        following = (i + 1) % len(order)
        if len(order) > 1 and modes[i] == "full" and modes[following] == "full":
            raise ConfigError(
                f"defaults.order で、mode = 'full' の条件 {name} と {order[following]} が隣り合っています"
                "（入力トークンの予算を使い切るので、間に別の mode の条件を入れてください。先頭と末尾も隣り合わせです）"
            )


def validate_config(cfg: dict) -> None:
    """conditions.toml を検証する。API は呼ばない。

    知らないキー・値の型・label の重なり・gold と acceptable が lectures.json にあるか・両者の重なり・base の循環・
    order の網羅と full の隣り合わせ・settings の名前・changes の old（ちょうど 1 か所）・
    retrieval の module:attr が import でき、引数のスキーマが元のツールと同じか、を確かめる。
    チェックの pattern の <<名前>>（[fragments] の部品）は、展開して cfg に書き戻す。
    """
    require_table("conditions.toml", cfg)
    check_keys("conditions.toml", cfg, TOP_KEYS)
    for key in ("defaults", "questions", "variants"):
        if key not in cfg:
            raise ConfigError(f"[{key}] がありません")
    validate_defaults(cfg["defaults"])
    fragments = validate_fragments(cfg.get("fragments", {}))
    validate_questions(cfg["questions"], fragments)
    validate_variants(cfg)
    validate_levels(cfg)
    validate_order(cfg)
    for name in cfg["variants"]:
        build_effective(cfg, name)  # changes・retrieval・settings の当て方の誤りは、ここで ConfigError になる


# ---------------------------------------------------------------------------
# 変種を当てた結果: モデルに見える指示・ツール・取得処理・設定・対象資料・上限
# ---------------------------------------------------------------------------


def apply_change(text: str, change: dict, where: str) -> str:
    """changes の 1 件を text に当てる。old がちょうど 1 か所に一致しなければ ConfigError。"""
    op, old, new = change["op"], change["old"], change.get("new", "")
    if op in ("delete_line", "replace_line"):
        lines = text.split("\n")
        hits = [i for i, line in enumerate(lines) if line == old]
        if len(hits) != 1:
            raise ConfigError(f"{where}: 行「{old}」に一致する行が {len(hits)} か所です（ちょうど 1 か所である必要があります）")
        if op == "delete_line":
            del lines[hits[0]]
        else:
            lines[hits[0]] = new
        return "\n".join(lines)
    count = text.count(old)
    if count != 1:
        raise ConfigError(f"{where}: 文字列「{old}」が {count} か所に一致します（ちょうど 1 か所である必要があります）")
    return text.replace(old, new)


def apply_changes(text: str, changes: list[dict], target: str, where: str) -> str:
    for change in changes:
        if change["target"] == target:
            text = apply_change(text, change, f"{where} の {target}")
    return text


def registered_tools() -> dict:
    """agent に登録済みの Tool を、名前で引ける dict にする（agent.toolsets は public）。"""
    from pydantic_ai.toolsets import FunctionToolset

    agent = app_agent().agent
    return {t.name: t for ts in agent.toolsets if isinstance(ts, FunctionToolset) for t in ts.tools.values()}


def mode_defaults(mode: str) -> tuple[str, list, str | None]:
    """mode の既定の指示・ツール・prompt のひな形。app.agent.main() の --full-context・（なし）・--tree・--graph と同じ。"""
    agent = app_agent()
    tools = app_tools()
    if mode == "full":
        return agent.FULL_CONTEXT_INSTRUCTIONS, [], agent.FULL_CONTEXT_PROMPT
    if mode == "vector":
        available = registered_tools()
        if "search_materials" not in available:
            raise ConfigError("agent に search_materials が登録されていません")
        return agent.INSTRUCTIONS, [available["search_materials"]], None  # make_quiz・save_learning_log は入れない
    if mode == "tree":
        return agent.TREE_INSTRUCTIONS, [Tool(tools.browse_tree)], None  # CLI の override(tools=[tools.browse_tree]) と同じ定義
    return agent.GRAPH_INSTRUCTIONS, [Tool(tools.search_graph)], None


def load_callable(target: str, where: str):
    """'module:attr' の関数を読み込む。"""
    module_name, attr = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001  モジュールが無い・壊れている
        raise ConfigError(f"{where}: {module_name} を読み込めません: {type(e).__name__}: {e}") from e
    function = getattr(module, attr, None)
    if not callable(function):
        raise ConfigError(f"{where}: {module_name} に関数 {attr} がありません")
    return function


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of(data) -> str:
    return sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))


@lru_cache(maxsize=64)
def _file_sha256(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path) -> str | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return _file_sha256(str(path), stat.st_mtime_ns, stat.st_size)


def data_hashes() -> dict:
    """結果を左右するデータ（資料 JSON・目次・グラフ・検索 DB）の sha256。"""
    config = app_config()
    return {
        "lectures.json": file_sha256(config.RAG_SOURCE),
        "tree.json": file_sha256(config.RAG_TREE),
        "graph.json": file_sha256(config.RAG_GRAPH),
        "index.sqlite3": file_sha256(app_retrieval().RAG_DB),
    }


def source_of(obj) -> str:
    """関数・クラスのソース。取れなければ名前（アドレスは入れない。hash が実行ごとに変わらないように）。"""
    obj = inspect.unwrap(obj)
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return f"{getattr(obj, '__module__', '')}:{getattr(obj, '__qualname__', type(obj).__name__)}"


def code_names(code: types.CodeType):
    yield from code.co_names
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from code_names(const)


def referenced_code(function) -> dict[str, str]:
    """関数と、その下で使う同じモジュールの関数・クラス・定数の「名前 → sha256」（取得処理の hash に使う）。

    関数・クラスは inspect.getsource の sha256、定数（文字列・数・タプル・dict・正規表現）は repr の sha256。
    設定（SETTING_MODULES）と、実行ごとに変わるもの（TRACE などのリスト）は入れない。
    """
    function = inspect.unwrap(function)
    module_name = function.__module__
    found: dict[str, str] = {}
    queue = [function]
    while queue:
        item = queue.pop()
        key = f"{item.__module__}:{item.__qualname__}"
        if key in found:
            continue
        found[key] = sha256_text(source_of(item))
        code = getattr(item, "__code__", None)
        if code is None:
            continue
        for name in code_names(code):
            value = item.__globals__.get(name)
            if value is None or name in SETTING_MODULES:
                continue
            if callable(value):
                value = inspect.unwrap(value)
            if isinstance(value, types.FunctionType) and value.__module__ == module_name:
                queue.append(value)
            elif inspect.isclass(value) and value.__module__ == module_name:
                found.setdefault(f"{value.__module__}:{value.__qualname__}", sha256_text(source_of(value)))
            elif isinstance(value, (str, int, float, bool, tuple, frozenset, dict, re.Pattern)):
                stable = sorted(value, key=repr) if isinstance(value, frozenset) else value  # frozenset の repr の順は、実行ごとに変わる
                found[f"{module_name}:{name}"] = sha256_text(repr(stable))
    return found


RETRIEVAL_CORE = ("search", "embed_text", "search_graph", "browse_tree", "get_all_materials")


def retrieval_core_hashes() -> dict[str, str]:
    """どの条件でも使う、app.retrieval の検索・埋め込み・Tree・Graph・全文投入の関数の sha256。"""
    retrieval = app_retrieval()
    return {f"app.retrieval:{name}": sha256_text(source_of(getattr(retrieval, name))) for name in RETRIEVAL_CORE}


def function_name(function) -> str:
    function = inspect.unwrap(function)
    return f"{function.__module__}:{function.__qualname__}"


def tool_definition(tool) -> dict:
    d = tool.tool_def
    return {"name": d.name, "description": d.description, "parameters_json_schema": d.parameters_json_schema, "kind": d.kind}


@dataclass
class Effective:
    """条件を当てた結果。モデルに見える指示と、ツール（Tool）。app/ のファイルは書き換えない。"""

    name: str
    label: str
    level: str
    mode: str
    instructions: str
    tools: list
    tool_defs: list
    retrieval: dict  # ツール名 → {"function": "module:qualname", "code": {名前: sha256}}
    settings: dict  # 条件が書き換える設定
    settings_effective: dict  # 試行の間に効く、許可された設定すべての値
    documents: Any  # mode = full のときの対象資料（"all" か資料 ID の配列）。それ以外は None
    request_limit: int
    prompt_template: str | None
    changes: list
    base: str | None
    code_files: dict  # コードのファイル（app/agent.py など）→ ファイル全体の sha256。sha256 に入っている
    uses_lec05: bool  # 取得処理か設定で app.retrieval_lec05 を使うか
    sha256: str


def uses_lec05_module(retrieval: dict, settings: dict) -> bool:
    """取得処理（ツールの関数）か設定（settings）で、app.retrieval_lec05 を使う条件か。"""
    uses = any(info["function"].startswith(RETRIEVAL_LEC05 + ":") for info in retrieval.values())
    return uses or any(SETTING_MODULES[name] == RETRIEVAL_LEC05 for name in settings)


def code_file_hashes(uses_lec05: bool) -> dict:
    """結果を左右するコードのファイル全体の sha256。agent・tools・retrieval・config は全条件、retrieval_lec05 は使う条件だけ。"""
    rels = [*CODE_FILES, *([LEC05_FILE] if uses_lec05 else [])]
    return {rel: file_sha256(Path(CODE_ROOT) / rel) for rel in rels}


def effective_settings(overrides: dict) -> dict:
    """許可された設定すべての値（条件が書き換えるものは、その値）。モジュールが無い設定は None。"""
    values = {}
    for name, module_name in SETTING_MODULES.items():
        if name in overrides:
            values[name] = overrides[name]
            continue
        try:
            values[name] = getattr(importlib.import_module(module_name), name, None)
        except Exception:  # noqa: BLE001  実装のモジュールがまだ無い（その設定を使う条件は、検証で止まる）
            values[name] = None
    return values


def build_effective(cfg: dict, name: str) -> Effective:
    """条件 name を、mode の既定の指示・ツールに当てる。順番は base → mode → retrieval → changes → settings → documents → request_limit。"""
    where = f"variants.{name}"
    spec = resolve_spec(cfg, name)
    raw = cfg["variants"][name]
    instructions, tools, template = mode_defaults(spec["mode"])
    tools = list(tools)

    retrieval_info = {}
    for tool_name, target in spec["retrieval"].items():
        index = next((i for i, t in enumerate(tools) if t.name == tool_name), None)
        if index is None:
            raise ConfigError(f"{where}.retrieval: この条件のツールに {tool_name} がありません（ツール: {', '.join(t.name for t in tools) or 'なし'}）")
        base_tool = tools[index]
        function = load_callable(target, f"{where}.retrieval.{tool_name}")
        # 関数だけを dataclasses.replace で替えても、function_schema が元の関数のままで、元の関数が呼ばれ続ける。作り直す
        replaced = Tool(function, name=base_tool.name, description=base_tool.description, requires_approval=base_tool.requires_approval)
        if replaced.tool_def.parameters_json_schema != base_tool.tool_def.parameters_json_schema:
            raise ConfigError(
                f"{where}.retrieval.{tool_name}: {target} の引数のスキーマが、元のツールと違います"
                f"（元: {json.dumps(base_tool.tool_def.parameters_json_schema, ensure_ascii=False)}／"
                f"差し替え: {json.dumps(replaced.tool_def.parameters_json_schema, ensure_ascii=False)}）"
            )
        tools[index] = replaced

    allowed = {"instructions", *(f"tool:{t.name}" for t in tools)}
    for change in spec["changes"]:
        if change["target"] not in allowed:
            raise ConfigError(f"{where}: target {change['target']!r} は {sorted(allowed)} のどれかにしてください")
    instructions = apply_changes(instructions, spec["changes"], "instructions", where)
    for i, tool in enumerate(tools):
        description = apply_changes(tool.description or "", spec["changes"], f"tool:{tool.name}", where)
        if description != (tool.description or ""):
            tools[i] = dataclasses.replace(tool, description=description)  # 説明文だけの変更。関数とスキーマは変わらない

    tool_defs = [tool_definition(t) for t in tools]
    for tool in tools:
        retrieval_info[tool.name] = {"function": function_name(tool.function), "code": referenced_code(tool.function)}
    for key in spec["settings"]:
        setting_module(key)  # 設定を持つモジュールが読み込めること
        if not hasattr(setting_module(key), key):
            raise ConfigError(f"{where}.settings.{key}: {SETTING_MODULES[key]} にその設定がありません")
    settings = dict(spec["settings"])
    settings_effective = effective_settings(settings)
    documents = (spec["documents"] or "all") if spec["mode"] == "full" else None
    uses_lec05 = uses_lec05_module(retrieval_info, settings)
    code_files = code_file_hashes(uses_lec05)
    # retrieval_lec05 を使わない条件（V0 など）の sha256 は、そのファイルの設定の既定値が変わっても変わらない
    settings_in_sha = {k: v for k, v in settings_effective.items() if uses_lec05 or SETTING_MODULES[k] != RETRIEVAL_LEC05}
    sha = sha256_of({
        "mode": spec["mode"], "instructions": instructions, "tools": tool_defs, "retrieval": retrieval_info,
        "retrieval_core": retrieval_core_hashes(), "code_files": code_files, "settings": settings_in_sha, "documents": documents,
        "prompt_template": template, "request_limit": spec["request_limit"], "data": data_hashes(),
    })
    return Effective(
        name=name, label=raw["label"], level=raw["level"], mode=spec["mode"], instructions=instructions, tools=tools,
        tool_defs=tool_defs, retrieval=retrieval_info, settings=settings, settings_effective=settings_effective,
        documents=documents, request_limit=spec["request_limit"], prompt_template=template, changes=spec["changes"],
        base=raw.get("base"), code_files=code_files, uses_lec05=uses_lec05, sha256=sha,
    )


def scope_documents(eff_documents, mode: str) -> list[str]:
    """条件がモデルに見せる資料の ID（full は渡す資料、それ以外は検索の対象 RAG_DOCUMENT_IDS）。"""
    if mode == "full":
        return list(page_index().titles) if eff_documents in (None, "all") else list(eff_documents)
    return list(app_retrieval().RAG_DOCUMENT_IDS)


def render_effective(eff: Effective) -> str:
    """モデルに見える指示とツール定義と、取得処理・設定・対象資料・上限を、差分を取りやすい文字列にする。"""
    lines = ["[instructions]", *eff.instructions.rstrip("\n").split("\n"), "", "[tools]"]
    if not eff.tool_defs:
        lines.append("- （ツールなし）")
    for d in eff.tool_defs:
        lines.append(f"- {d['name']} ({d['kind']})")
        lines += [("    " + line).rstrip() for line in (d["description"] or "").split("\n")]
        props = d["parameters_json_schema"].get("properties", {})
        lines.append(f"    引数: {json.dumps(props, ensure_ascii=False)}")
    lines += ["", "[取得処理]"]
    if not eff.retrieval:
        lines.append("- （ツールなし）")
    for tool_name, info in eff.retrieval.items():
        lines.append(f"- {tool_name}: {info['function']}")
    lines += ["", "[設定]"]
    lines += [f"{key} = {value}" for key, value in sorted(eff.settings_effective.items())]
    index = page_index()
    ids = scope_documents(eff.documents, eff.mode)
    lines += ["", "[対象資料]（全文投入で渡す資料）" if eff.mode == "full" else "[対象資料]（検索の対象 RAG_DOCUMENT_IDS）"]
    total = 0
    for document_id in ids:
        count = len(index.pages.get(document_id, {}))
        total += count
        lines.append(f"- {document_id}: {count} ページ（{index.titles.get(document_id, '?')}）")
    lines.append(f"合計 {total} ページ")
    lines += ["", "[上限]", f"request_limit = {eff.request_limit}"]
    return "\n".join(lines) + "\n"


def diff_baseline(cfg: dict, name: str) -> str | None:
    """条件の差分を取る基準。base があれば base、なければ mode が full なら full・vector なら vector。tree・graph は基準なし。"""
    spec = cfg["variants"][name]
    if spec.get("base"):
        return spec["base"]
    mode = resolve_spec(cfg, name)["mode"]
    if mode in ("full", "vector") and mode in cfg["variants"]:
        return mode
    return None


def variant_diffs(cfg: dict) -> dict[str, dict]:
    """条件ごとに、基準との unified diff と、差分の塊（hunk）の数を返す。"""
    effective = {name: build_effective(cfg, name) for name in cfg["variants"]}
    texts = {name: render_effective(eff) for name, eff in effective.items()}
    result = {}
    for name in cfg["variants"]:
        baseline = diff_baseline(cfg, name)
        diff = []
        if baseline is not None:
            diff = list(difflib.unified_diff(texts[baseline].splitlines(), texts[name].splitlines(), baseline, name, lineterm=""))
        result[name] = {
            "effective": effective[name],
            "text": texts[name],
            "baseline": baseline,
            "diff": diff,
            "hunks": sum(1 for line in diff if line.startswith("@@")),
            "changes": len(cfg["variants"][name].get("changes", [])),
        }
    return result


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


TMP_PREFIX = ".tmp-"  # 書きかけの一時ファイルの名前の頭。t<k>.json などの探し方に当たらず、結果として数えない


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


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# git: 結果がどの commit のものかを確かにする
# ---------------------------------------------------------------------------


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """git を読み取りだけに使う（status・rev-parse）。index を書き換えず、入力を待たず、止まったままにならない。"""
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
    repo = repo or REPO
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


# ---------------------------------------------------------------------------
# API エラーの分類と、再試行
# ---------------------------------------------------------------------------


class ApiFailure(Exception):
    """API の失敗を、共通の形にしたもの。source は、どこで起きたか（model・embedding・tool_return）。"""

    def __init__(self, status: int | None, text: str, retry_after: float | None = None, transport: bool = False, source: str = "model"):
        super().__init__(text)
        self.status = status
        self.text = text
        self.retry_after = retry_after
        self.transport = transport  # 接続の失敗・タイムアウト（status が無い）
        self.source = source


class EmbeddingApiFailure(Exception):
    """埋め込み API の失敗。ツールの try/except を素通りして、run を打ち切るための例外。

    ValueError・FileNotFoundError・sqlite3.Error・google.genai の APIError のどれも継承しない
    （app/tools.py の search_materials などは、それらを捕まえて、モデルへの戻り値の error にしてしまうため）。
    元のエラーの status と本文（retryDelay や PerDay を含む）を持つ。
    """

    def __init__(self, status: int | None, text: str, *, transport: bool = False):
        super().__init__(text)
        self.status = status
        self.text = text
        self.transport = transport


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
            attempts.append({
                "attempt": len(attempts) + 1, "source": failure.source, "status": failure.status,
                "error": failure.text[:500], "action": action, "wait_s": wait,
            })
            if action == "daily_limit":
                raise StopRun(2, DAILY_LIMIT_MESSAGE) from failure
            if action == "stop":
                hint = ""
                if failure.status is None or failure.status >= 500:
                    hint = "\n503 などが続くときは、しばらく待ってから同じコマンドで続きから実行してください。"
                raise StopRun(
                    1, f"API エラーで止めます（{failure.status}。{failure.source}）: {failure.text[:300]}{hint}\n同じコマンドで続きから実行できます。"
                ) from failure
            ctx.wait(wait, f"API エラー（{failure.status}。{failure.source}）")
            continue
        attempts.append({"attempt": len(attempts) + 1, "ok": True})
        return value


def api_failure_from(e: ModelAPIError) -> ApiFailure:
    """pydantic-ai の API エラーを ApiFailure にする（ModelHTTPError 以外は、接続の失敗・タイムアウト）。"""
    return ApiFailure(
        getattr(e, "status_code", None), str(e), getattr(e, "retry_after", None),
        transport=not isinstance(e, ModelHTTPError), source="model",
    )


def is_transport_error(e: BaseException) -> bool:
    module = type(e).__module__.split(".")[0]
    return module in ("httpx", "httpx2", "httpcore") or isinstance(e, (TimeoutError, ConnectionError))


# 全文投入で、入力が上限を超えたときの 400 の本文（Gemini: "The input token count (N) exceeds the maximum number of tokens allowed (M)."）
INPUT_LIMIT_PATTERN = re.compile(
    r"input token count|exceeds? the maximum number of tokens|maximum number of tokens|token count[^.]{0,60}exceed|"
    r"too many tokens|input is too long|context (?:length|window)",
    re.I,
)


def is_input_limit(e: ModelAPIError) -> bool:
    return isinstance(e, ModelHTTPError) and e.status_code in (400, 413) and INPUT_LIMIT_PATTERN.search(f"{e} {e.body}") is not None


# ---------------------------------------------------------------------------
# 実行の状態（Context）と、実行する条件（Condition）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """実行する 1 つの条件（変種 × 質問）。試行番号は別。"""

    level: str
    variant: str
    label: str
    mode: str
    qid: str
    text: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.variant, self.qid)


@dataclass
class TrialOutput:
    record: dict
    trace: str


@dataclass
class Context:
    cfg: dict
    out: Path
    model: str
    dry_run: bool = False
    interval: float = 0.0
    tpm_budget: int = 0
    trials: int = 3
    git: dict = field(default_factory=dict)
    argv: list = field(default_factory=list)
    recent_inputs: list = field(default_factory=list)  # TPM ガード用: (試行を終えた time.monotonic(), その試行の入力トークン)
    _effective: dict = field(default_factory=dict)
    _google_model: Any = None

    def effective(self, name: str) -> Effective:
        if name not in self._effective:
            self._effective[name] = build_effective(self.cfg, name)
        return self._effective[name]

    def model_dir(self) -> str:
        """予行の結果は <model>-dry-run のフォルダに書く（本物の結果と混ざらないように）。"""
        return f"{self.model}-dry-run" if self.dry_run else self.model

    def google_model(self):
        """本物のモデル（app.agent.build_model）。1 回作って使い回す。"""
        if self._google_model is None:
            self._google_model = app_agent().build_model()
        return self._google_model

    def wait(self, seconds: float, why: str) -> None:
        print(f"  … {why}。{seconds:.0f} 秒待って、再試行します", flush=True)
        time.sleep(seconds)


def trial_stem(ctx: Context, cond: Condition, k: int) -> Path:
    return ctx.out / ctx.model_dir() / cond.variant / cond.qid / f"t{k}"


def sibling(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def trial_done(ctx: Context, cond: Condition, k: int) -> bool:
    return sibling(trial_stem(ctx, cond, k), ".json").exists()


def existing_trials(ctx: Context, cond: Condition) -> list[Path]:
    """その条件の、済みの試行の t<k>.json（書きかけの .tmp- のファイルは数えない）。"""
    folder = trial_stem(ctx, cond, 1).parent
    return sorted(p for p in folder.glob("t*.json") if re.fullmatch(r"t\d+\.json", p.name)) if folder.is_dir() else []


def save_trial(stem: Path, output: TrialOutput) -> bool:
    """trace を先に置き、最後に t<k>.json を置く（json があれば「済み」。json は、あれば置き換えない）。置けたら True。"""
    json_path = sibling(stem, ".json")
    if json_path.exists():  # 同時に動いた別の実行が、先に置いた
        return False
    write_text_atomic(sibling(stem, ".trace.txt"), output.trace)
    return write_json_atomic(json_path, output.record, exclusive=True)


# ---------------------------------------------------------------------------
# 設定・埋め込み・対象資料の差し替え（試行の間だけ。終われば戻す）
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def patched_attr(module, name: str, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextlib.contextmanager
def patched_settings(settings: dict):
    """conditions.toml の settings を、試行の間だけ書き換える（例外が起きても戻す）。"""
    with contextlib.ExitStack() as stack:
        for name, value in settings.items():
            stack.enter_context(patched_attr(setting_module(name), name, value))
        yield


class EmbeddingMeter:
    """質問の埋め込みを呼んだ回数と、かかった秒数。"""

    def __init__(self):
        self.calls = 0
        self.seconds = 0.0


def fake_embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """予行用の埋め込み。テキストの sha256 から作る、768 次元の長さ 1 のベクトル（同じ文なら、いつも同じ）。"""
    digest = b"".join(hashlib.sha256(f"{i}\n{task_type}\n{text}".encode()).digest() for i in range(24))  # 32 バイト × 24 = 768 バイト
    return app_retrieval().normalized_vector([byte / 127.5 - 1.0 for byte in digest])


@contextlib.contextmanager
def patched_embedding(ctx: Context, meter: EmbeddingMeter):
    """app.retrieval.embed_text を、呼んだ回数と秒を数える包みにする。

    google.genai の APIError と、接続・タイムアウトのエラーは、EmbeddingApiFailure にして送出する。
    ツールの try/except を素通りして、run を打ち切れる（ツールが error の戻り値にしてしまうと、失敗に気づけない）。
    予行は、fake_embed_text を包む。
    """
    from google.genai import errors as genai_errors

    retrieval = app_retrieval()
    real = fake_embed_text if ctx.dry_run else retrieval.embed_text

    def embed_text(text, task_type):
        started = time.perf_counter()
        try:
            return real(text, task_type)
        except genai_errors.APIError as e:
            raise EmbeddingApiFailure(e.code, str(e)) from e
        except Exception as e:
            if is_transport_error(e):
                raise EmbeddingApiFailure(None, f"{type(e).__name__}: {e}", transport=True) from e
            raise
        finally:
            meter.calls += 1
            meter.seconds += time.perf_counter() - started

    embed_text.__wrapped__ = real  # hash を取るとき（inspect.unwrap）、元の関数のソースを見る
    with patched_attr(retrieval, "embed_text", embed_text):
        yield


class SubsetSource:
    """app.retrieval.RAG_SOURCE の代役。選んだ資料だけの JSON を、read_text で返す。"""

    def __init__(self, text: str):
        self.text = text

    def read_text(self, encoding: str = "utf-8", errors: str | None = None) -> str:
        return self.text


def build_materials(eff: Effective) -> str:
    """全文投入で渡す資料の文字列。"all" は get_all_materials() そのまま。

    配列のときは、RAG_SOURCE を「選んだ資料だけの JSON を返す代役」に一時的に替えて、同じ get_all_materials() を呼ぶ
    （書式を写し取らないので、app/retrieval.py の書式が変われば、そのとおりに変わる）。
    """
    retrieval = app_retrieval()
    if eff.documents == "all":
        return retrieval.get_all_materials()
    wanted = set(eff.documents)
    documents = json.loads(retrieval.RAG_SOURCE.read_text(encoding="utf-8"))
    subset = [document for document in documents if document["id"] in wanted]
    with patched_attr(retrieval, "RAG_SOURCE", SubsetSource(json.dumps(subset, ensure_ascii=False))):
        return retrieval.get_all_materials()


def full_prompt(eff: Effective, materials: str, question: str) -> str:
    return eff.prompt_template.format(materials=materials, question=question)


def materials_info(eff: Effective, materials: str, prompt: str) -> dict:
    return {
        "documents": scope_documents(eff.documents, eff.mode),
        "pages": len(re.findall(r"^ページ: PDF p\.", materials, re.M)),
        "prompt_chars": len(prompt),
        "materials_chars": len(materials),
        "materials_sha256": sha256_text(materials),
        "prompt_sha256": sha256_text(prompt),
    }


# ---------------------------------------------------------------------------
# TPM ガード（mode = full のとき）: 直近 60 秒の入力トークンが予算を超えないよう、待つ
# ---------------------------------------------------------------------------


def recorded_input_tokens(ctx: Context, cond: Condition) -> list[int]:
    tokens = []
    for path in existing_trials(ctx, cond):
        try:
            value = (json.loads(path.read_text(encoding="utf-8")).get("usage") or {}).get("input_tokens")
        except (OSError, ValueError):
            continue
        if isinstance(value, int) and value > 0:
            tokens.append(value)
    return tokens


def tpm_estimate(ctx: Context, cond: Condition, prompt_chars: int) -> tuple[int, str]:
    """この試行の入力トークンの見積もり。その条件の既存の記録の中央値。無ければ prompt の文字数 × 0.6。"""
    recorded = recorded_input_tokens(ctx, cond)
    if recorded:
        return int(statistics.median(recorded)), "records"
    return int(prompt_chars * TPM_CHARS_PER_TOKEN), "chars"


def tpm_wait(ctx: Context, estimate: int) -> float:
    """直近 60 秒に終えた試行の入力トークンの和 + estimate が予算を超えるなら、超えなくなるまで待つ。待った秒数の合計を返す。

    見積もりだけで予算を超えるときは、待っても減らないので、待たずに実行して警告する。tpm_budget = 0 で無効。
    """
    if ctx.tpm_budget <= 0:
        return 0.0
    if estimate > ctx.tpm_budget:
        print(f"  警告: この試行の入力の見積もり {estimate:,} トークンが、予算 {ctx.tpm_budget:,} トークン（60 秒）を超えます。待たずに実行します", flush=True)
        return 0.0
    waited = 0.0
    while True:
        now = time.monotonic()
        ctx.recent_inputs = [(t, n) for t, n in ctx.recent_inputs if now - t < TPM_WINDOW]
        used = sum(n for _, n in ctx.recent_inputs)
        if used + estimate <= ctx.tpm_budget:
            return waited
        remaining, wait = used, TPM_WINDOW
        for finished, tokens in sorted(ctx.recent_inputs):  # 古い順に、60 秒が過ぎて数えなくなるのを待つ
            remaining -= tokens
            if remaining + estimate <= ctx.tpm_budget:
                wait = max(finished + TPM_WINDOW - now, 0.0)
                break
        print(
            f"  … 直近 {TPM_WINDOW} 秒の入力 {used:,} トークン + この試行の見積もり {estimate:,} が予算 {ctx.tpm_budget:,} を超えるので、{wait:.0f} 秒待ちます",
            flush=True,
        )
        time.sleep(wait)
        waited += wait


def guard_tpm(ctx: Context, cond: Condition, eff: Effective) -> dict:
    """mode = full の試行の前に呼ぶ。待った秒数などを、記録用の dict で返す。"""
    if ctx.tpm_budget <= 0:
        return {"budget": ctx.tpm_budget, "estimate": None, "estimate_source": None, "waited_s": 0.0}
    prompt = full_prompt(eff, build_materials(eff), cond.text)
    estimate, source = tpm_estimate(ctx, cond, len(prompt))
    waited = tpm_wait(ctx, estimate)
    return {"budget": ctx.tpm_budget, "estimate": estimate, "estimate_source": source, "waited_s": round(waited, 3)}


# ---------------------------------------------------------------------------
# ツールの戻り値の error の分類
# ---------------------------------------------------------------------------

SETUP_ERROR_PATTERN = re.compile(
    r"検索DBがありません|no such (?:table|column|module|function)|unable to open database|disk image is malformed|"
    r"file is not a database|database is locked|readonly database|not authorized|vec0|sqlite|"
    r"\[Errno \d+\]|No such file or directory|Permission denied",
    re.I,
)


def tool_error_message(result) -> str | None:
    """ツールの戻り値（JSON の文字列）の error の文。error が無い・JSON でなければ None。"""
    if not isinstance(result, str):
        return None
    try:
        data = json.loads(result)
    except ValueError:
        return None
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"]
    return None


def classify_error_message(message: str) -> tuple[str, int | None]:
    """ツールの戻り値の error を分ける。('api', status)・('setup', None)・('model', None)。

    api: app.config.api_error_message が作る文（429・503 の文、400 系の「APIキーかモデル名に問題があります（N）」、その他の「APIエラー（N）」）。
    setup: 検索 DB が無い・開けない（sqlite のエラー、ファイルの読み書きのエラー）。
    model: それ以外（空の検索語・目次ノードが無い・埋め込み入力の上限など。モデルが引数を直せば変わるもの）。
    """
    from app.config import api_error_message

    if message == api_error_message(429):
        return "api", 429
    if message == api_error_message(503):
        return "api", 503
    for status in (400, 500):
        prefix = api_error_message(status, "").split(str(status), 1)[0]  # 「APIキーかモデル名に問題があります（」「APIエラー（」
        match = re.match(re.escape(prefix) + r"(\d{3})）", message)
        if match:
            return "api", int(match.group(1))
    if SETUP_ERROR_PATTERN.search(message):
        return "setup", None
    return "model", None


def classify_tool_errors(tool_calls: list[dict]) -> list[dict]:
    """試行の後に、ツールの戻り値の error を見る。api は ApiFailure（試行ごとやり直す）、setup は StopRun、model は記録だけ。"""
    errors = []
    for call in tool_calls:
        message = tool_error_message(call.get("result"))
        if message is None:
            continue
        kind, status = classify_error_message(message)
        if kind == "api":
            raise ApiFailure(status, message, source="tool_return")
        if kind == "setup":
            raise StopRun(
                1,
                f"検索の準備ができていません（{call.get('name')} の戻り値の error: {message}）。\n"
                "data/rag_data/index.sqlite3 があるか、AIE_RAG_DB が設定されていないかを確かめてください。",
            )
        errors.append({"order": call.get("order"), "name": call.get("name"), "kind": "model", "error": message})
    return errors


# ---------------------------------------------------------------------------
# 予行（--dry-run）のモデル: mode ごとの台本どおりに動く FunctionModel（API を呼ばない）
# ---------------------------------------------------------------------------


def parse_json(text):
    try:
        return json.loads(text) if isinstance(text, str) else None
    except ValueError:
        return None


def first_user_text(messages) -> str:
    from pydantic_ai.messages import UserPromptPart

    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                return part.content if isinstance(part.content, str) else str(part.content)
    return ""


def tool_returns(messages) -> list:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def citation_text(page: dict) -> str:
    """予行の回答に書く引用（「第N回・PDF p.N」）。"""
    lecture = page.get("lecture")
    if lecture is None and page.get("document_id"):
        lecture = doc_lecture(page["document_id"])
    return f"第{lecture}回・PDF p.{page.get('page')}" if lecture is not None else f"PDF p.{page.get('page')}"


def script_answer(pages: list[dict]) -> str:
    if not pages:
        return "（予行）資料から確認できませんでした。"
    return f"（予行）取得した最後のページを引用します。\n\n根拠: {citation_text(pages[-1])}"


def script_full_answer(prompt: str) -> str:
    """全文投入の台本: 資料の最後のページを引用して答える。"""
    documents = re.findall(r"^資料: (\S+) /", prompt, re.M)
    pages = re.findall(r"^ページ: PDF p\.(\d+)", prompt, re.M)
    if not documents or not pages:
        return "（予行）資料から確認できませんでした。"
    lecture = doc_lecture(documents[-1])
    cite = f"第{lecture}回・PDF p.{pages[-1]}" if lecture is not None else f"{documents[-1]}・PDF p.{pages[-1]}"
    return f"（予行）資料を読みました。最後のページを引用します。\n\n根拠: {cite}"


def script_tree_step(question: str, data: dict | None, returned: int):
    """Tree の台本の次の 1 手。root → 講義 → 章 → ページの順にたどる。戻り値は (呼ぶ node_id, None) か (None, 回答)。"""
    if data is None or "node" not in data or returned >= 8:
        return None, "（予行）目次から確認できませんでした。"
    node, children = data["node"], data.get("children", [])
    if node.get("kind") == "page":
        return None, script_answer([node])
    if node.get("kind") == "collection":  # root: 質問の「第N回」の本講義があればそれ、なければ最初の本講義
        match = re.search(r"第(\d+)回", unicodedata.normalize("NFKC", question))
        wanted = f"lec{int(match.group(1)):02d}-main" if match else None
        chosen = next((c for c in children if c["id"] == wanted), None) or next((c for c in children if c["id"].endswith("-main")), None)
    else:  # 講義・章: 章があれば最初の章、なければ最初のページ
        chosen = next((c for c in children if c.get("kind") == "section"), None) or next((c for c in children if c.get("kind") == "page"), None)
    if chosen is None:
        return None, "（予行）目次から確認できませんでした。"
    return chosen["id"], None


def script_model(mode: str) -> FunctionModel:
    """mode ごとの台本どおりに動く偽のモデル。vector は検索を 1 回呼び、取得した最後のページを引用して答える。"""

    def respond(messages, info):
        returns = tool_returns(messages)
        question = first_user_text(messages)
        tool_name = info.function_tools[0].name if info.function_tools else None
        data = parse_json(returns[-1].content) if returns else None
        if mode == "full":
            return ModelResponse(parts=[TextPart(script_full_answer(question))])
        if mode == "tree":
            if not returns:
                return ModelResponse(parts=[ToolCallPart(tool_name, {})])  # node_id を省略 = 目次の最上位
            node_id, answer = script_tree_step(question, data, len(returns))
            if answer is not None:
                return ModelResponse(parts=[TextPart(answer)])
            return ModelResponse(parts=[ToolCallPart(tool_name, {"node_id": node_id})])
        if not returns:  # vector・graph: 検索を 1 回
            return ModelResponse(parts=[ToolCallPart(tool_name, {"query": question})])
        return ModelResponse(parts=[TextPart(script_answer([entry["page"] for entry in result_pages(data)["full"]]))])

    return FunctionModel(respond, model_name=f"dry-run-{mode}")


def make_model(ctx: Context, eff: Effective):
    """予行は台本のモデル（API を呼ばない）。本番は app.agent.build_model()。"""
    return script_model(eff.mode) if ctx.dry_run else ctx.google_model()


# ---------------------------------------------------------------------------
# 1 試行の実行
# ---------------------------------------------------------------------------


def args_dict(part) -> dict | str:
    try:
        return part.args_as_dict()
    except Exception:  # noqa: BLE001  引数が JSON として読めないとき
        return part.args if isinstance(part.args, str) else str(part.args)


def as_text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


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


def make_decider(log: list):
    """承認待ちの呼び出しを、すべて拒否する。記録は log に足す（比較用のツールに承認が要るものは無いが、念のため）。"""

    def decide(call) -> bool:
        log.append({
            "index": len(log) + 1, "tool_call_id": call.tool_call_id, "tool_name": call.tool_name,
            "args": args_dict(call), "decision": "denied",
        })
        return False

    return decide


def response_costs(messages) -> list:
    """応答ごとの費用（USD）。pydantic-ai 2.57 は、応答を受け取るたびに ModelResponse.usage.cost に入れる（料金が分からなければ None）。"""
    return [m.usage.cost for m in messages if isinstance(m, ModelResponse)]


def usage_of(result, messages) -> RunUsage:
    """試行の使用量。cost は、すべての応答の費用が分かるときだけ（1 つでも None なら None = 算定不可）。

    pydantic-ai の run の usage.cost は、一部の応答だけ費用が分かるとき、分かった分だけの合計になる（None を 0 として足す）ので、そのまま使わない。
    成功したときは result.usage を写し（result 側は書き換えない）、失敗したときは、応答ごとの使用量を足す。
    """
    if result is not None:
        usage = copy.deepcopy(result.usage)  # 2.57.0 ではプロパティ（result.usage() ではない）
    else:
        usage = RunUsage()
        for m in messages:
            if isinstance(m, ModelResponse):
                usage.requests += 1
                usage.incr(m.usage)
    if any(cost is None for cost in response_costs(messages)):
        usage.cost = None
    return usage


def elide_materials(value, materials: str):
    """messages の中の資料の本文を、<資料>…(N 字, sha256=…)…</資料> に置き換える（1 件が数百 KB になるので、git に入れない）。"""
    note = f"…({len(materials)} 字, sha256={sha256_text(materials)})…"

    def walk(x):
        if isinstance(x, str):
            if materials not in x:
                return x
            return x.replace("<資料>\n" + materials + "\n</資料>", f"<資料>{note}</資料>").replace(materials, note)
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(value)


def lec05_module(eff: Effective):
    """取得の内訳（TRACE）を持つ app.retrieval_lec05。この条件が使わなければ None（読み込まない）。"""
    return importlib.import_module(RETRIEVAL_LEC05) if eff.uses_lec05 else None


def run_attempt(ctx: Context, cond: Condition, eff: Effective) -> dict:
    """1 回の試み。API の失敗は ApiFailure で送出する（call_with_retries が待って再試行する）。"""
    agent_module = app_agent()
    agent = agent_module.agent
    meter, approvals = EmbeddingMeter(), []
    lec05 = lec05_module(eff)
    result, status, error, materials, prompt = None, "ok", None, None, None
    with contextlib.ExitStack() as stack:
        stack.enter_context(patched_settings(eff.settings))
        if lec05 is not None:
            lec05.TRACE.clear()
        stack.enter_context(patched_embedding(ctx, meter))
        captured = stack.enter_context(capture_run_messages())
        model = make_model(ctx, eff)
        limits = UsageLimits(request_limit=eff.request_limit)
        started = time.perf_counter()
        try:
            if eff.mode == "full":
                materials = build_materials(eff)  # 資料の組み立てから測る（CLI の [time] と同じ）
                prompt = full_prompt(eff, materials, cond.text)
                # run_full_context は SystemExit するので呼ばない。同じ override の中で、同じ prompt を投げる
                with agent.override(tools=[], instructions=eff.instructions):
                    result = agent.run_sync(prompt, model=model, usage_limits=limits)
            else:
                with agent.override(instructions=eff.instructions, tools=eff.tools):
                    result = agent_module.run_with_approval(
                        agent, cond.text, decide=make_decider(approvals), model=model, usage_limits=limits
                    )
        except UsageLimitExceeded as e:
            status, error = "usage_limit", {"type": type(e).__name__, "message": str(e)}
        except UnexpectedModelBehavior as e:
            status, error = "unexpected", {"type": type(e).__name__, "message": str(e)}
        except EmbeddingApiFailure as e:
            raise ApiFailure(e.status, e.text, transport=e.transport, source="embedding") from e
        except ModelAPIError as e:
            if eff.mode == "full" and is_input_limit(e):  # 入力が上限を超えた。再試行しても同じなので、結果として残す
                status, error = "input_limit", {"type": type(e).__name__, "message": str(e)[:1000]}
            else:
                raise api_failure_from(e) from e
        duration = time.perf_counter() - started
        messages = list(result.all_messages()) if result is not None else list(captured)
        trace_entries = copy.deepcopy(list(lec05.TRACE)) if lec05 is not None else []
    if result is not None and not isinstance(result.output, str):
        status, error = "unexpected", {"type": "NonTextOutput", "message": f"回答が文字列ではありません: {type(result.output).__name__}"}
    tool_calls = collect_tool_calls(messages)
    tool_errors = classify_tool_errors(tool_calls)  # api は ApiFailure（試行ごとやり直す）、setup は StopRun
    return {
        "result": result, "status": status, "error": error, "duration": duration, "messages": messages,
        "tool_calls": tool_calls, "tool_errors": tool_errors, "approvals": approvals, "meter": meter,
        "retrieval_trace": trace_entries, "materials": materials, "prompt": prompt,
    }


def run_command(eff: Effective) -> str:
    if eff.mode == "full":
        return (
            f"agent.run_sync(FULL_CONTEXT_PROMPT.format(materials=get_all_materials(), question=<質問>), model=build_model(), "
            f"usage_limits=UsageLimits(request_limit={eff.request_limit})) を agent.override(tools=[], instructions=FULL_CONTEXT_INSTRUCTIONS) の中で実行"
            f"（variant={eff.name}。資料: {eff.documents if eff.documents == 'all' else ', '.join(eff.documents)}）"
        )
    return (
        f"app.agent.run_with_approval(agent, <質問>, model=build_model(), usage_limits=UsageLimits(request_limit={eff.request_limit}))"
        f" を agent.override(instructions=<条件>, tools=<条件>) の中で実行（variant={eff.name}）"
    )


def package_versions() -> dict:
    import importlib.metadata as metadata

    versions = {}
    for name in ("pydantic-ai-slim", "google-genai", "sqlite-vec"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def trial_usage_fields(usage: RunUsage) -> dict:
    return {
        "usage": dataclasses.asdict(usage),
        "cost_usd": float(usage.cost) if usage.cost is not None else None,  # 分からないときは null（0 にしない）
        "thoughts_tokens": int(usage.details.get("thoughts_tokens", 0) or 0),
    }


def response_rows(messages) -> list[dict]:
    rows = []
    for message in messages:
        if isinstance(message, ModelResponse):
            rows.append({
                "index": len(rows) + 1, "input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens,
                "thoughts_tokens": int(message.usage.details.get("thoughts_tokens", 0) or 0), "finish_reason": message.finish_reason,
                "cost_usd": float(message.usage.cost) if message.usage.cost is not None else None,  # 応答ごとの費用。分からなければ null
            })
    return rows


def build_record(ctx: Context, cond: Condition, eff: Effective, k: int, got: dict, attempts: list, started_at: str, elapsed: float, tpm: dict | None) -> dict:
    result, messages, status = got["result"], got["messages"], got["status"]
    answer = result.output if result is not None and isinstance(result.output, str) else ""
    first_request = next((m for m in messages if isinstance(m, ModelRequest) and m.instructions), None)
    seen = first_request.instructions if first_request is not None else eff.instructions
    last_response = next((m for m in reversed(messages) if isinstance(m, ModelResponse)), None)
    usage = usage_of(result, messages)
    message_data = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
    if got["materials"]:
        message_data = elide_materials(message_data, got["materials"])
    record = {
        "schema": 1, "lecture": 5,
        "variant": eff.name, "label": eff.label, "level": eff.level, "mode": eff.mode, "variant_sha256": eff.sha256,
        "qid": cond.qid, "question": cond.text, "question_sha256": sha256_text(cond.text), "trial": k,
        "position": trial_position(ctx.cfg, cond, k),  # この試行（k）での、条件の位置（順を巡回させるので、試行ごとに違う）
        "model": ctx.model, "model_name": last_response.model_name if last_response is not None and not ctx.dry_run else None,
        "dry_run": ctx.dry_run, "command": run_command(eff), "compare_args": ctx.argv,
        "git": {"commit": ctx.git.get("commit"), "dirty": ctx.git.get("dirty")},
        "versions": package_versions(), "data_sha256": data_hashes(), "code_sha256": eff.code_files,
        "started_at": started_at, "finished_at": now_iso(), "duration_s": round(got["duration"], 3), "elapsed_s": round(elapsed, 3),
        "attempts": attempts, "status": status, "error": got["error"], "answer": answer,
        # モデルに実際に見えた指示（最初の request の instructions）。設定した条件と食い違えば、instructions_match が false
        "instructions": seen, "instructions_sha256": sha256_text(seen), "instructions_match": seen.strip() == eff.instructions.strip(),
        "tools": eff.tool_defs,
        "retrieval": {name: {"function": info["function"], "code_sha256": sha256_of(info["code"])} for name, info in eff.retrieval.items()},
        "settings": eff.settings, "settings_effective": eff.settings_effective, "request_limit": eff.request_limit,
        "tpm": tpm,
        "tool_calls": got["tool_calls"], "tool_errors": got["tool_errors"], "approvals": got["approvals"],
        "embedding": {"calls": got["meter"].calls, "seconds": round(got["meter"].seconds, 3)},
        "retrieval_trace": got["retrieval_trace"],
        **trial_usage_fields(usage),
        "responses": response_rows(messages), "messages": message_data,
    }
    if got["materials"] is not None:
        record["materials"] = materials_info(eff, got["materials"], got["prompt"])
    # 記録のときにも入れる。集計（summarize）は、tool_calls と answer から計算し直す
    scope = (record.get("materials") or {}).get("documents") or scope_documents(None, eff.mode)
    record["retrieved"] = analyze_retrieval(got["tool_calls"])
    record["citations"] = parse_citations(answer, scope)
    return record


def format_retrieval_trace(entry: dict) -> list[str]:
    """TRACE の 1 呼び出しを、1〜数行にする。"""
    tool = entry.get("tool")
    if tool == "neighbors":
        return [
            f"[neighbors] query={entry.get('query')!r} window={entry.get('window')} hits={entry.get('hits')} "
            f"neighbors={entry.get('neighbors')} pages={entry.get('pages')}"
        ]
    if tool == "term_graph":
        lines = [f"[term_graph] query={entry.get('query')!r} params={json.dumps(entry.get('params'), ensure_ascii=False)}"]
        lines.append("  用語: " + ("、".join(t.get("title", "?") for t in entry.get("terms", [])) or "なし")
                     + "／回の指定: " + ("、".join(f"第{n}回" for n in entry.get("lectures", [])) or "なし"))
        for item in entry.get("selected", []):
            distance = item.get("distance")
            lines.append(f"  選択: {item.get('page_id')} d={distance:.3f} {item.get('reason')}" if isinstance(distance, (int, float)) else f"  選択: {item.get('page_id')} {item.get('reason')}")
        skipped = entry.get("skipped_by_cap", [])
        lines.append("  章の上限で外した: " + (", ".join(str(s.get("page_id")) for s in skipped) or "なし"))
        return lines
    return [f"[{tool}] {json.dumps(entry, ensure_ascii=False, default=str)}"]


def build_trace(ctx: Context, cond: Condition, eff: Effective, got: dict, elapsed: float) -> str:
    """CLI の --trace と同じ表示（print_trace）に、条件・質問・取得の内訳を足した文章。"""
    lines = [f"[model] {ctx.model}", f"[variant] {eff.label} {eff.name}", f"[質問] {cond.qid}"]
    if got["prompt"] is not None:
        lines.append(f"[full-context] 入力{len(got['prompt']):,}文字")
    result, status = got["result"], got["status"]
    answer = result.output if result is not None and isinstance(result.output, str) else ""
    lines.append(answer if result is not None else f"[{status}] {(got['error'] or {}).get('message', '')}")
    # print_trace が表示する使用量も、記録と同じ（費用が一部しか分からないときは、算定不可）にする
    shim = SimpleNamespace(all_messages=lambda: got["messages"], usage=usage_of(result, got["messages"]))
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            app_agent().print_trace(shim, got["duration"])
    except Exception as e:  # noqa: BLE001  trace の表示の誤りで、保存できる結果を失わない
        buffer.write(f"\n[trace] 表示に失敗しました: {type(e).__name__}: {e}\n")
    lines.append(buffer.getvalue().rstrip("\n"))
    if got["retrieval_trace"]:
        lines += ["", "--- 取得の内訳 ---"]
        for entry in got["retrieval_trace"]:
            lines += format_retrieval_trace(entry)
    if got["tool_errors"]:
        lines += ["", "--- ツールの error（モデルが引数を直せば変わるもの） ---"]
        lines += [f"[{e['name']}] {e['error']}" for e in got["tool_errors"]]
    return "\n".join(lines).rstrip("\n") + "\n"


def run_trial(ctx: Context, cond: Condition, k: int) -> TrialOutput:
    """1 回の試行: TPM ガード（full のみ）→ 実行（再試行つき）→ 記録と trace。保存は呼び出し側。"""
    eff = ctx.effective(cond.variant)
    tpm = guard_tpm(ctx, cond, eff) if eff.mode == "full" else None
    attempts: list = []
    started_at = now_iso()
    clock = time.perf_counter()
    got = call_with_retries(ctx, lambda: run_attempt(ctx, cond, eff), attempts)
    elapsed = time.perf_counter() - clock
    record = build_record(ctx, cond, eff, k, got, attempts, started_at, elapsed, tpm)
    trace = build_trace(ctx, cond, eff, got, elapsed)
    ctx.recent_inputs.append((time.monotonic(), record["usage"]["input_tokens"]))
    return TrialOutput(record, trace)


# ---------------------------------------------------------------------------
# 解析 1: ツールの戻り値から、どのページを、どう取得したか（集計のときに、tool_calls から計算し直す）
# ---------------------------------------------------------------------------


def nested_pages(data) -> list[dict]:
    """入れ子の中から、page_id と content を持つ dict を拾う（形が分からない戻り値用）。"""
    found = []
    if isinstance(data, dict):
        if "page_id" in data and "content" in data:
            found.append(data)
        else:
            for value in data.values():
                found += nested_pages(value)
    elif isinstance(data, list):
        for value in data:
            found += nested_pages(value)
    return found


def hit_ranks(rows: list[dict]) -> dict[int, int]:
    """role を持つ戻り値（neighbors）で、hit の順位（distance の昇順）。行の位置 → 順位。"""
    hits = [i for i, row in enumerate(rows) if row.get("role") == "hit"]
    ordered = sorted(hits, key=lambda i: (rows[i].get("distance") is None, rows[i].get("distance") or 0.0, i))
    return {i: rank for rank, i in enumerate(ordered, 1)}


def result_pages(content) -> dict:
    """ツールの戻り値（JSON の文字列か、読み込んだ dict）から、全文で取得したページ・先頭文だけのページ・error を取り出す。

    形（キー）で判定する: results（検索。role があれば hit と neighbor を分ける）／node と children（Tree。node がページなら全文を開いた、
    children のページは先頭文だけ）／seeds と expanded（Graph）／error／それ以外（入れ子から page_id と content を持つ dict を拾う）。
    JSON でない戻り値（再試行の指示の文など）でも落ちない。
    """
    data = parse_json(content) if isinstance(content, str) else content
    out = {"kind": "none", "full": [], "preview": [], "error": None, "node": None}
    if not isinstance(data, (dict, list)):
        return out
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        out.update(kind="error", error=data["error"])
        return out
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = [row for row in data["results"] if isinstance(row, dict)]
        ranks = hit_ranks(rows) if any("role" in row for row in rows) else None
        out["kind"] = "results"
        for i, row in enumerate(rows):
            role = row.get("role")
            rank = (ranks.get(i) if ranks is not None else i + 1)
            out["full"].append({
                "page": row, "via": role or "search", "rank": rank, "distance": row.get("distance"), "role": role, "hops": None,
            })
        return out
    if isinstance(data, dict) and isinstance(data.get("node"), dict):
        node = data["node"]
        out.update(kind="tree", node={"id": node.get("id"), "kind": node.get("kind"), "title": node.get("title")})
        if node.get("kind") == "page":
            out["full"].append({"page": {**node, "page_id": node.get("id")}, "via": "tree_open", "rank": None, "distance": None, "role": None, "hops": None})
        for child in data.get("children") or []:
            if isinstance(child, dict) and child.get("kind") == "page":
                out["preview"].append({"page": {**child, "page_id": child.get("id")}, "via": "tree_preview"})
        return out
    if isinstance(data, dict) and ("seeds" in data or "expanded" in data):
        out["kind"] = "graph"
        for name, via in (("seeds", "seed"), ("expanded", "expanded")):
            for rank, row in enumerate([r for r in data.get(name) or [] if isinstance(r, dict)], 1):
                out["full"].append({
                    "page": row, "via": via, "rank": rank, "distance": row.get("distance"), "role": None, "hops": row.get("hops"),
                    "path": row.get("path"),
                })
        return out
    out["kind"] = "other"
    for row in nested_pages(data):
        out["full"].append({"page": row, "via": "other", "rank": None, "distance": row.get("distance"), "role": row.get("role"), "hops": row.get("hops")})
    return out


def analyze_retrieval(tool_calls: list[dict]) -> dict:
    """tool_calls から、取得したページ（全文／先頭文だけ）・字数・Tree でたどった目次を、記録の retrieved の形にする。"""
    full, preview, tree_path, errors = [], [], [], []
    return_chars = 0
    for call in tool_calls or []:
        content = call.get("result")
        return_chars += len(content) if isinstance(content, str) else 0
        parsed = result_pages(content)
        if parsed["error"] is not None:
            errors.append({"call": call.get("order"), "name": call.get("name"), "error": parsed["error"]})
        if parsed["kind"] == "tree":
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            tree_path.append({"call": call.get("order"), "node_id": args.get("node_id") or parsed["node"]["id"], **parsed["node"]})
        for entry in parsed["full"]:
            page = entry["page"]
            row = {
                "page_id": page.get("page_id") or page.get("id"), "via": entry["via"], "rank": entry["rank"], "distance": entry["distance"],
                "role": entry["role"], "hops": entry["hops"], "chars": len(page.get("content") or ""), "call": call.get("order"),
            }
            if entry.get("path"):
                row["path"] = entry["path"]
            full.append(row)
        for entry in parsed["preview"]:
            page = entry["page"]
            preview.append({"page_id": page.get("page_id"), "via": entry["via"], "chars": len(page.get("content") or ""), "call": call.get("order")})
    unique: dict[str, int] = {}
    for row in full:
        unique.setdefault(row["page_id"], row["chars"])
    return {
        "full_text": full, "preview": preview, "tree_path": tree_path, "errors": errors,
        "chars_full_text": sum(row["chars"] for row in full), "chars_unique": sum(unique.values()), "tool_return_chars": return_chars,
    }


def full_text_ids(retrieved: dict) -> list[str]:
    """全文で取得したページの ID（重複なし。取得した順）。"""
    return list(dict.fromkeys(row["page_id"] for row in retrieved.get("full_text", []) if row.get("page_id")))


def preview_only_ids(retrieved: dict) -> list[str]:
    """先頭文だけ見たページ（全文を開いていないもの）の ID。"""
    opened = set(full_text_ids(retrieved))
    return [p for p in dict.fromkeys(row["page_id"] for row in retrieved.get("preview", [])) if p not in opened]


def pages_in_scope(documents: list[str]) -> set[str]:
    """資料 ID のページすべての page_id。"""
    index = page_index()
    return {page_id for document in documents for page_id in index.pages.get(document, {}).values()}


def gold_retrieval(rec: dict, retrieved: dict, gold: list[str]) -> dict:
    """gold の取得率。全文で取得したページ ∩ gold ÷ gold。mode = full は、gold が対象資料に入っているかで判定する。gold が空なら rate は None。"""
    if rec.get("mode") == "full":
        scope = pages_in_scope((rec.get("materials") or {}).get("documents") or [])
        have = {g for g in gold if g in scope}
    else:
        ids = set(full_text_ids(retrieved))
        have = {g for g in gold if g in ids}
    ranks = {}
    for g in gold:
        found = [r["rank"] for r in retrieved.get("full_text", []) if r.get("page_id") == g and r.get("rank") is not None and r.get("via") in ("search", "hit", "seed")]
        ranks[g] = min(found) if found else None
    return {
        "gold": list(gold), "hit": [g for g in gold if g in have], "total": len(gold),
        "rate": (len(have) / len(gold)) if gold else None, "ranks": ranks,
    }


# ---------------------------------------------------------------------------
# 解析 2: 回答の引用（「第N回・PDF p.N」）を、ページに対応づける
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}"  # 4 桁以上（2026 年など）は、ページ番号として拾わない
_RANGE_SEP = r"[-–—~〜]"
_BARE = rf"{_NUM}(?:\s*{_RANGE_SEP}\s*{_NUM})?(?![A-Za-z\d.]|\s*[年月日回件個倍%％秒分時人行問章節つ])"  # p. の無い数字（列挙の 2 つ目以降）
# p.52・PDF p.52・pp.51-52・p.13、39、40・p.13・39・40
P_CITATION = re.compile(
    rf"(?<![A-Za-z])(?:pdf\s*)?pp?\.?\s*(?P<first>{_NUM}(?:\s*{_RANGE_SEP}\s*(?:pdf\s*)?(?:pp?\.?\s*)?{_NUM})?)(?![A-Za-z\d])(?P<rest>(?:\s*[、,・]\s*{_BARE})*)",
    re.I,
)  # 範囲の終わりの p.（p.13–p.15）は、あってもなくてもよい
PAGE_WORD_CITATION = re.compile(rf"(?<![A-Za-z\d.])(?P<list>{_BARE}(?:\s*[、,・]\s*{_BARE})*)\s*(?:ページ|頁)")  # 52ページ・13、39、40ページ
LECTURE_CUE = re.compile(r"第(\d+)回")
DOCUMENT_ID_CUE = re.compile(r"orientation|lec\d{2}-(?:main|exercise)")
KIND_CUES = (("演習", "exercise"), ("本講義", "main"), ("本編", "main"))
CUE_WINDOW = 40  # 引用の直前の、手がかりを探す字数（まず、同じ行の中）
CUE_WINDOW_LINES = 80  # 同じ行に手がかりが無いとき、改行をまたいで探す字数


def split_pages(text: str) -> list[tuple[str, str]]:
    """'13、39、40' や '51–52' を、[('13', '13'), ('39', '39'), ('51', '52')] のように（始まり, 終わり）に分ける。"""
    pairs = []
    for part in re.split(r"\s*[、,・]\s*", text.strip()):
        bounds = re.findall(r"\d+", part)  # 範囲の終わりの p. は読み飛ばす
        pairs.append((bounds[0], bounds[-1]))
    return pairs


def citation_matches(text: str) -> list[tuple[int, str, list[tuple[str, str]]]]:
    """正規化した回答から、引用を位置の順に探す。(開始位置, raw の文字列, [(始まり, 終わり) ...])。"""
    found, taken = [], []
    for pattern, group in ((P_CITATION, None), (PAGE_WORD_CITATION, "list")):
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < b and a < end for a, b in taken):  # すでに拾った引用と重なるもの（p.52ページ など）は飛ばす
                continue
            taken.append((start, end))
            numbers = match.group(group) if group else match.group("first") + match.group("rest")
            found.append((start, match.group(0).strip(), split_pages(numbers)))
    return sorted(found, key=lambda item: item[0])


def document_cues(window: str, titles: dict[str, str]) -> tuple[set[str] | None, str | None]:
    """引用の直前の窓から、資料の手がかりを探す。戻り値は (候補の資料の集合, 手がかりの種類)。手がかりが無ければ (None, None)。

    資料 ID と資料名は、その資料を決める（いちばん近いもの）。「第N回」は、その回の資料（「演習」「本講義」があれば、その種類だけ）。
    「演習」「本講義」だけなら、その種類の資料すべて。集合は、モデルに見えた資料（scope）に絞る前のもの。
    """
    exact = [(m.end(), m.group(0)) for m in DOCUMENT_ID_CUE.finditer(window)]
    for document_id, title in titles.items():
        position = window.rfind(title)
        if position >= 0:
            exact.append((position + len(title), document_id))
    if exact:
        return {max(exact)[1]}, "id"
    kind = None
    for word, label in KIND_CUES:
        position = window.rfind(word)
        if position >= 0 and (kind is None or position > kind[0]):
            kind = (position, label)
    lectures = [int(m.group(1)) for m in LECTURE_CUE.finditer(window)]
    if lectures:
        docs = {d for d in titles if doc_lecture(d) == lectures[-1]}
        if kind is not None:
            docs = {d for d in docs if doc_kind(d) == kind[1]}
        return docs, "lecture"
    if kind is not None:
        return {d for d in titles if doc_kind(d) == kind[1]}, "kind"
    return None, None


def parse_citations(answer: str, scope: list[str]) -> list[dict]:
    """回答の引用を、ページごとに 1 件の dict にする。raw の文字列は必ず残す。

    NFKC でそろえてから、p.N・PDF p.N・pp.N-M（範囲）・p.N、M・p.N・M（列挙）・N ページを探す（4bit・2026年・p値 などは拾わない）。
    資料の手がかり（第N回・資料 ID・資料名・演習・本講義）は、まず同じ行の直前 40 字から探す。同じ行に無ければ、改行をまたいで直前 80 字まで探す
    （「第3回の本講義では」と書いた次の行の「- PDF p.13」のため）。それでも無ければ、直前の引用の資料を引き継ぐ（cue_window: line・lines・carry）。
    scope は、モデルに見えた資料の ID（214 ページの条件は第1〜3回の本講義、全文投入は渡した資料）。
    status:
      ok         — 1 つのページに決まった
      ambiguous  — 手がかりはあるが、複数のページが残った（全文投入の「第N回」が本講義と演習の両方に同じ番号のページがあるときなど）。page_ids に全部残す
      unresolved — 手がかりが無く、複数の資料に同じ番号のページがある
      invalid    — 存在しないページ（対象外の資料を含む）
      wide_range — 20 ページを超える範囲（引用として数えない）
    """
    text = unicodedata.normalize("NFKC", answer or "")
    index = page_index()
    in_scope = [d for d in scope if d in index.titles]
    titles = {d: unicodedata.normalize("NFKC", index.titles[d]) for d in index.titles}
    citations, carried = [], None
    for start, raw, pairs in citation_matches(text):
        line_start = text.rfind("\n", 0, start) + 1
        docs, cue = document_cues(text[max(line_start, start - CUE_WINDOW):start], titles)
        cue_window = "line" if docs is not None else None
        if docs is None:  # 同じ行に手がかりが無いときは、改行をまたいで、直前 80 字まで探す
            docs, cue = document_cues(text[max(0, start - CUE_WINDOW_LINES):start], titles)
            cue_window = "lines" if docs is not None else None
        if docs is not None:
            carried = docs
        else:  # それでも無ければ、直前の引用の資料を引き継ぐ
            docs, cue = carried, ("carry" if carried is not None else None)
            cue_window = "carry" if carried is not None else None
        candidates_docs = [d for d in in_scope if docs is None or d in docs]
        for first, last in pairs:
            a, b = int(first), int(last)
            if b < a or b - a + 1 > MAX_PAGE_RANGE:
                citations.append({
                    "raw": raw, "page": None, "docs": candidates_docs, "page_ids": [], "cue": cue, "cue_window": cue_window,
                    "status": "wide_range" if b >= a else "invalid", "detail": f"範囲 {first}-{last}",
                })
                continue
            for number in range(a, b + 1):
                page_ids = [index.pages[d][number] for d in candidates_docs if number in index.pages[d]]
                if not page_ids:
                    status = "invalid"
                elif len(page_ids) == 1:
                    status = "ok"
                else:
                    status = "ambiguous" if cue is not None else "unresolved"
                citations.append({
                    "raw": raw, "page": number, "docs": [d for d in candidates_docs if number in index.pages[d]],
                    "page_ids": page_ids, "cue": cue or ("existence" if status == "ok" else None), "cue_window": cue_window,
                    "status": status, "detail": None,
                })
    return citations


def citation_metrics(citations: list[dict], gold: list[str], acceptable: list[str], retrieved_ids) -> dict:
    """引用の指標。ok と ambiguous を数える（ambiguous は、候補に gold・acceptable があれば、その引用として数える）。

    pages: 引用したページ（候補のまとまり）の数。gold・acceptable・other: そのうち、gold・acceptable・どちらでもないもの。
    grid: gold ページごとの、取得（全文）× 引用の 2×2。
    """
    counted = [c for c in citations if c["status"] in ("ok", "ambiguous")]
    targets: list[tuple] = []
    for c in counted:
        key = tuple(c["page_ids"])
        if key not in targets:
            targets.append(key)
    cited = {page_id for target in targets for page_id in target}
    known = set(gold) | set(acceptable)
    gold_cited = [g for g in gold if g in cited]
    acceptable_cited = [a for a in acceptable if a in cited]
    grid = {"retrieved_cited": 0, "retrieved_not_cited": 0, "not_retrieved_cited": 0, "neither": 0}
    for g in gold:
        key = {(True, True): "retrieved_cited", (True, False): "retrieved_not_cited", (False, True): "not_retrieved_cited", (False, False): "neither"}[(g in retrieved_ids, g in cited)]
        grid[key] += 1
    return {
        "pages": len(targets), "gold": len(gold_cited), "gold_total": len(gold), "gold_cited": gold_cited,
        "gold_rate": (len(gold_cited) / len(gold)) if gold else None,
        "acceptable": len(acceptable_cited), "acceptable_cited": acceptable_cited,
        "other": sum(1 for target in targets if not set(target) & known),
        "unresolved": sum(1 for c in citations if c["status"] in ("unresolved", "ambiguous")),
        "ambiguous": sum(1 for c in citations if c["status"] == "ambiguous"),
        "invalid": sum(1 for c in citations if c["status"] == "invalid"), "wide_range": sum(1 for c in citations if c["status"] == "wide_range"),
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# 実行する条件の選び方と、順番（試行 → 質問 → 条件）
# ---------------------------------------------------------------------------


def csv_list(values) -> list[str] | None:
    """--variant a b や --variant a,b のどちらの書き方でも、リストにする。"""
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
    return tuple(level for level in LEVELS if level in items)


def select_variants(cfg: dict, args) -> list[str]:
    """--level と --variant（名前か label）で絞り込んだ条件の名前。打ち間違いは、使える名前を示して止める。順番は conditions.toml の順。"""
    levels = parse_levels(getattr(args, "level", None))
    wanted = csv_list(getattr(args, "variant", None))
    if wanted:
        known = {w.lower() for name, spec in cfg["variants"].items() for w in (name, spec["label"])}
        for item in wanted:
            if item.lower() not in known:
                choices = ", ".join(f"{name}（{spec['label']}）" for name, spec in cfg["variants"].items())
                raise ConfigError(f"--variant {item!r} は知りません（使えるもの: {choices}）")
    selected = []
    for name, spec in cfg["variants"].items():
        if spec["level"] not in levels:
            continue
        if wanted and not any(item.lower() in (name.lower(), spec["label"].lower()) for item in wanted):
            continue
        selected.append(name)
    return selected


def check_questions(cfg: dict, args) -> list[str] | None:
    wanted = csv_list(getattr(args, "question", None))
    for qid in wanted or []:
        if qid not in cfg["questions"]:
            raise ConfigError(f"--question {qid!r} は知りません（使えるもの: {', '.join(cfg['questions'])}）")
    return wanted


def order_shift_count(cfg: dict, k: int) -> int:
    """試行 k で、order を巡回させる個数（order_shift × (k − 1)。order の長さを超えたら、割った余り）。"""
    return cfg["defaults"]["order_shift"] * (k - 1) % len(cfg["defaults"]["order"])


def trial_order(cfg: dict, k: int) -> list[str]:
    """試行 k での、1 つの質問の中の条件の順。order を、試行ごとに order_shift 個ずつ巡回させる（k = 1 は order のまま）。

    全条件が、試行ごとに違う順で流れるので、時間帯や API の混み具合の影響と、条件の効果を区別しやすくなる。
    全文投入どうしが隣り合わないことは、読み込み時の環状の検証（validate_order。先頭と末尾のつながりも見る）が保証していて、巡回しても変わらない。
    """
    order = cfg["defaults"]["order"]
    shift = order_shift_count(cfg, k)
    return order[shift:] + order[:shift]


def build_plan(cfg: dict, names: list[str], questions: list[str] | None = None, trial: int = 1) -> list[Condition]:
    """試行 trial で実行する条件を、実行する順に返す。質問ごとに、その試行の順（trial_order。その質問を流さない条件は飛ばす）。"""
    plan = []
    for qid, item in cfg["questions"].items():
        if questions and qid not in questions:
            continue
        for name in trial_order(cfg, trial):
            spec = cfg["variants"][name]
            if name in names and qid in spec["questions"]:
                plan.append(Condition(spec["level"], name, spec["label"], resolve_spec(cfg, name)["mode"], qid, item["text"]))
    return plan


def make_plan(cfg: dict, args, trial: int = 1) -> list[Condition]:
    names = select_variants(cfg, args)
    return build_plan(cfg, names, check_questions(cfg, args), trial)


def trial_position(cfg: dict, cond: Condition, k: int) -> dict:
    """試行 k での、条件の位置。--variant などの絞り込みに関係なく、全条件を流したときの位置（どう区切って実行しても、同じ値になる）。

    in_question: その質問の中での順（1 から）。in_trial: その試行の全体での順。order_shift: その試行で巡回させた個数。
    """
    everything = build_plan(cfg, list(cfg["variants"]), None, k)
    of_question = [c for c in everything if c.qid == cond.qid]
    return {
        "in_question": next((i for i, c in enumerate(of_question, 1) if c.variant == cond.variant), None), "of_question": len(of_question),
        "in_trial": next((i for i, c in enumerate(everything, 1) if c.key == cond.key), None), "of_trial": len(everything),
        "order_shift": order_shift_count(cfg, k),
    }


def is_inside(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def resolve_out(args, dry_run: bool) -> Path:
    out = getattr(args, "out", None)
    if out:
        path = Path(out)
        if dry_run and is_inside(path, DEFAULT_OUT):
            raise StopRun(1, f"--dry-run の結果は {RESULTS_REL} に書けません（本物の結果と混ざるため）。--out を省くと一時フォルダに書きます。")
        return path
    if dry_run:
        return Path(tempfile.mkdtemp(prefix="lec05-dry-run-"))
    return DEFAULT_OUT


def make_context(args, cfg: dict, *, dry_run: bool = False, with_git: bool = True) -> Context:
    defaults = cfg["defaults"]
    interval = getattr(args, "interval", None)
    tpm_budget = getattr(args, "tpm_budget", None)
    return Context(
        cfg=cfg, out=resolve_out(args, dry_run), model=app_config().MODEL, dry_run=dry_run,
        interval=0.0 if dry_run else float(defaults["interval"] if interval is None else interval),
        tpm_budget=0 if dry_run else int(defaults["tpm_budget"] if tpm_budget is None else tpm_budget),
        trials=getattr(args, "trials", None) or defaults["trials"],
        argv=list(getattr(args, "_argv", None) or []), git=git_state(REPO) if with_git else {},
    )


# ---------------------------------------------------------------------------
# run: 実行の前の確認と、試行番号を外側に回した実行
# ---------------------------------------------------------------------------


def dirty_message(dirty_files: list[str]) -> str:
    files = "\n".join(f"  {line}" for line in dirty_files)
    untracked = "\n?? は git が追跡していないファイルです。中身を確かめて、commit するか、不要なら消してください。" if any(line.startswith("??") for line in dirty_files) else ""
    return (
        "app/・data/rag_data/・compare.py・conditions.toml・pyproject.toml・uv.lock に、未コミットの変更があります。\n"
        f"{files}{untracked}\n結果がどの commit のものかを確かにするため、commit してから実行してください（確認を省くときは --allow-dirty）。"
    )


def check_index() -> None:
    """検索 DB が開けて、chunks が対象資料の全ページ（214 行・3 資料）あるか。"""
    retrieval = app_retrieval()
    index = page_index()
    expected_rows = sum(len(index.pages.get(d, {})) for d in retrieval.RAG_DOCUMENT_IDS)
    expected_documents = len(retrieval.RAG_DOCUMENT_IDS)
    try:
        with retrieval.open_index(retrieval.RAG_DB) as db:
            rows, documents = db.execute("SELECT COUNT(*), COUNT(DISTINCT document_id) FROM chunks").fetchone()
    except (FileNotFoundError, sqlite3.Error) as e:
        raise StopRun(
            1, f"検索 DB を開けません（{retrieval.RAG_DB}）: {e}\ndata/rag_data/index.sqlite3 があるか確かめてください（第5回の更新を取り込む）。"
        ) from e
    if (rows, documents) != (expected_rows, expected_documents):
        raise StopRun(
            1,
            f"検索 DB の中身が、同梱のものと違います（chunks {rows} 行・{documents} 資料。あるべき値は {expected_rows} 行・{expected_documents} 資料）。"
            f"\n{retrieval.RAG_DB} を確かめてください。",
        )


def preflight(ctx: Context, args) -> list[Condition]:
    """API を呼ぶ前に、止めるべきことを、上の docstring の順に確かめる。実行する条件を返す。"""
    if not ctx.dry_run and not getattr(args, "allow_dirty", False) and ctx.git.get("dirty"):  # 1.
        raise StopRun(1, dirty_message(ctx.git["dirty_files"]))
    plan = make_plan(ctx.cfg, args)  # 2. 綴り違いは、使える名前を示して ConfigError
    if not plan:
        raise StopRun(1, "条件に一致する実行がありません（--level / --variant / --question を確かめてください）")
    for name in dict.fromkeys(c.variant for c in plan):
        ctx.effective(name)  # 条件の当て方が誤っていれば、ここで ConfigError になる
    for cond in plan:  # 3.
        sha, question_sha = ctx.effective(cond.variant).sha256, sha256_text(cond.text)
        for path in existing_trials(ctx, cond):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("最上位が object ではありません")
            except (OSError, ValueError) as e:
                raise StopRun(1, f"{path} を読めません: {e}\n壊れた記録に、続きの試行を混ぜないため止めます。確かめてください。") from e
            if record.get("variant_sha256") != sha:
                raise StopRun(
                    1,
                    f"{path} の条件の sha256 が、いまのコードと違います（{cond.label} {cond.variant} / {cond.qid}）。\n"
                    "途中で app/・data/rag_data/・conditions.toml・compare.py のどれかが変わった可能性があります。\n"
                    "条件を変えたあとは、別の --out に書いてください（同じ条件の結果に、違う条件の試行を混ぜないため）。",
                )
            recorded = record.get("question_sha256") or (sha256_text(record["question"]) if isinstance(record.get("question"), str) else None)
            if recorded is not None and recorded != question_sha:
                raise StopRun(
                    1,
                    f"{path} の質問の文が、いまの conditions.toml の {cond.qid} と違います（{cond.label} {cond.variant} / {cond.qid}）。\n"
                    "途中で質問の text が変わった可能性があります。\n"
                    "質問を変えたあとは、別の --out に書いてください（同じ条件の結果に、違う質問の試行を混ぜないため）。",
                )
    if not ctx.dry_run and not os.environ.get("GEMINI_API_KEY"):  # 4.
        raise StopRun(1, "環境変数 GEMINI_API_KEY が見つかりません（Codespaces secret に登録して、Codespaces を再起動してください）。")
    if os.environ.get("AIE_RAG_DB"):  # 5.
        raise StopRun(1, f"環境変数 AIE_RAG_DB が設定されています（{os.environ['AIE_RAG_DB']}）。比較は同梱の検索 DB で行うので、unset AIE_RAG_DB してから実行してください。")
    if any(c.mode in ("vector", "graph") for c in plan):  # 6.
        check_index()
    return plan


def result_line(position: str, cond: Condition, k: int, record: dict) -> str:
    usage = record.get("usage") or {}
    return (
        f"{position} {cond.label} {cond.variant}/{cond.qid} t{k}: {record['status']} {record['duration_s']:.1f}s "
        f"tool_calls={len(record.get('tool_calls', []))} requests={usage.get('requests')} input={usage.get('input_tokens')}"
    )


def cmd_run(args) -> int:
    cfg = load_config(args.conditions)
    ctx = make_context(args, cfg, dry_run=args.dry_run)
    plan = preflight(ctx, args)
    if args.allow_dirty and not ctx.dry_run and ctx.git.get("dirty"):
        print("--allow-dirty: 未コミットの変更の確認は飛ばします（結果の git.dirty に、そのまま記録されます）", flush=True)
    plans = {k: make_plan(cfg, args, trial=k) for k in range(1, ctx.trials + 1)}  # 試行ごとに、条件の順を巡回させる
    todo = [(k, c) for k in range(1, ctx.trials + 1) for c in plans[k] if not trial_done(ctx, c, k)]
    total = len(plan) * ctx.trials
    if ctx.dry_run:
        print(f"予行（--dry-run）: API は呼びません（偽のモデルと偽の埋め込み）。結果は {ctx.out} に書きます", flush=True)
    else:
        print(f"[model] {ctx.model}  interval={ctx.interval:g} 秒  tpm_budget={ctx.tpm_budget:,}", flush=True)
    mode = "予行" if ctx.dry_run else "本番"
    print(f"{mode}: 条件 {len(plan)} 件（条件 × 質問）× 試行 {ctx.trials} 回 = {total} 回のうち、済み {total - len(todo)} 回、これから {len(todo)} 回", flush=True)
    print(f"条件の順（質問ごと。試行ごとに {cfg['defaults']['order_shift']} 個ずつ巡回）:", flush=True)
    for k in plans:
        print(f"  t{k}: " + " ".join(c.label for c in plans[k] if c.qid == plan[0].qid), flush=True)
    executed = 0
    for k, cond in todo:
        if args.max_runs is not None and executed >= args.max_runs:
            print(f"--max-runs {args.max_runs} に達したので止めます。残り {len(todo) - executed} 回は、同じコマンドで続きから実行できます。")
            break
        if executed and ctx.interval > 0:
            time.sleep(ctx.interval)  # 実行のあとは interval 秒待つ（最後のあとは待たない）
        if not ctx.dry_run:
            print(f"  → {cond.label} {cond.variant}/{cond.qid} t{k} を実行します", flush=True)
        output = run_trial(ctx, cond, k)
        executed += 1
        position = f"[{total - len(todo) + executed}/{total}]"
        if not save_trial(trial_stem(ctx, cond, k), output):  # 置く瞬間にもうあった（同時に別の実行が書いた）ので、置き換えない
            print(f"{position} 警告: {cond.variant}/{cond.qid} t{k} の結果がもうあるので、置き換えずに飛ばします（同時に別の実行が書いた可能性があります）", file=sys.stderr, flush=True)
            continue
        print(result_line(position, cond, k, output.record), flush=True)
    print(f"\n実行 {executed} 回。保存先: {ctx.out}")
    if ctx.dry_run:
        print(f"（予行の結果です。集計の動作確認: uv run python lectures/05/compare.py summarize --out {ctx.out}）")
    elif executed:
        print("集計するには: uv run python lectures/05/compare.py summarize")
    return 0


# ---------------------------------------------------------------------------
# 集計 1: 試行ごとの値（tool_calls と answer から、解析し直す）
# ---------------------------------------------------------------------------


def med_range(values, digits: int = 0) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(statistics.median(values))} ({fmt.format(min(values))}–{fmt.format(max(values))})"


def fenced(text: str, lang: str = "") -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def cell_text(value) -> str:
    return str(value).replace("|", "｜").replace("\n", " ")


def display_width(text: str) -> int:
    """端末での表示幅（全角の文字は 2）。表の桁をそろえるため。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def print_table(rows: list[tuple]) -> None:
    widths = [max(display_width(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(str(cell) + " " * (width - display_width(str(cell))) for cell, width in zip(r, widths, strict=True)).rstrip())


def trial_view(cfg: dict, rec: dict) -> dict:
    """1 回の試行の記録から、表に出す値を計算する（取得・引用・チェックは、いまの解析とチェックで計算し直す）。"""
    question = cfg["questions"].get(rec.get("qid"), {})
    gold, acceptable = question.get("gold", []), question.get("acceptable", [])
    calls = rec.get("tool_calls") or []
    mode = rec.get("mode") or (resolve_spec(cfg, rec["variant"])["mode"] if rec.get("variant") in cfg["variants"] else None)
    rec = {**rec, "mode": mode}
    retrieved = analyze_retrieval(calls)
    scope = (rec.get("materials") or {}).get("documents") or scope_documents(None, mode)
    citations = parse_citations(rec.get("answer") or "", scope)
    retrieval_gold = gold_retrieval(rec, retrieved, gold)
    retrieved_ids = pages_in_scope(scope) if mode == "full" else set(full_text_ids(retrieved))
    usage = rec.get("usage") or {}
    return {
        "rec": rec, "trial": rec.get("trial"), "status": rec.get("status"), "retrieved": retrieved, "citations": citations,
        "question_changed": isinstance(rec.get("question"), str) and "text" in question and rec["question"] != question["text"],
        "gold": retrieval_gold, "metrics": citation_metrics(citations, gold, acceptable, retrieved_ids),
        "score": score_trial(checks_for(cfg, rec["qid"]), rec) if rec.get("qid") in cfg["questions"] else None,
        "tool_calls": len(calls), "requests": usage.get("requests"), "input": usage.get("input_tokens"), "output": usage.get("output_tokens"),
        "thoughts": rec.get("thoughts_tokens"), "seconds": rec.get("duration_s"), "cost": rec.get("cost_usd"),
    }


def load_model_records(model_dir: Path) -> list[dict]:
    records = []
    for path in sorted(model_dir.glob("*/*/t*.json")):
        if re.fullmatch(r"t\d+\.json", path.name):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def discover_models(out: Path) -> list[Path]:
    """<out>/<model>/<variant>/<qid>/t<k>.json のある <model> のフォルダ。"""
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if p.is_dir() and p.name not in ("variants", "cli") and next(p.glob("*/*/t*.json"), None) is not None)


def per_trial(views: list[dict], fn) -> str:
    """試行ごとの値を、t1,t2,t3 の順にカンマでつなぐ。"""
    return ",".join(str(fn(v)) for v in views)


def fraction(done: int, total: int) -> str:
    return f"{done}/{total}"


def gold_cell(v: dict) -> str:
    g = v["gold"]
    return "—" if not g["total"] else fraction(len(g["hit"]), g["total"])


def cited_gold_cell(v: dict) -> str:
    m = v["metrics"]
    return "—" if not m["gold_total"] else fraction(m["gold"], m["gold_total"])


def args_text(view: dict) -> str:
    """t1 のツールの引数（検索語・目次のノード）。"""
    parts = []
    for call in view["rec"].get("tool_calls") or []:
        args = call.get("args")
        value = (args.get("query") or args.get("node_id") or "（引数なし）") if isinstance(args, dict) else str(args)
        parts.append(f"{call.get('name')}({value})")
    return " ／ ".join(parts) or "ツールなし"


def opened_count(view: dict) -> int:
    return len({r["page_id"] for r in view["retrieved"]["full_text"] if r["via"] == "tree_open"})


def full_count(view: dict) -> int:
    rec = view["rec"]
    if rec.get("mode") == "full":
        return (rec.get("materials") or {}).get("pages") or 0
    return len(full_text_ids(view["retrieved"]))


def graph_counts(view: dict) -> str:
    rows = view["retrieved"]["full_text"]
    seeds = len({r["page_id"] for r in rows if r["via"] == "seed"})
    expanded = len({r["page_id"] for r in rows if r["via"] == "expanded"})
    return f"{seeds}/{expanded}"


def role_counts(view: dict) -> str:
    rows = view["retrieved"]["full_text"]
    hits = len({r["page_id"] for r in rows if r["via"] == "hit"})
    neighbors = len({r["page_id"] for r in rows if r["via"] == "neighbor"})
    return f"{hits}/{neighbors}"


def chars_value(view: dict) -> int:
    """取得した本文の字数（全文投入は、渡した資料の字数）。"""
    rec = view["rec"]
    if rec.get("mode") == "full":
        return (rec.get("materials") or {}).get("materials_chars") or 0
    return view["retrieved"]["chars_full_text"]


def rank_cell(view: dict) -> str:
    """gold の順位（検索の順位。取れなければ ×）。例: [#2,×]。全文投入と、gold の無い質問は空。"""
    ranks = view["gold"]["ranks"]
    if view["rec"].get("mode") == "full" or not ranks:
        return ""
    return "[" + ",".join(f"#{r}" if r is not None else "×" for r in ranks.values()) + "]"


def cost_cell(costs: list) -> str:
    """推定費用（USD）。中央値（最小–最大）。全部 None なら「算定不可」、一部だけなら注記する。0 にしない。"""
    known = [c for c in costs if c is not None]
    if not costs or not known:
        return "算定不可"
    text = med_range(known, 6)
    return text if len(known) == len(costs) else f"{text}（算定不可 {len(costs) - len(known)}/{len(costs)}）"


def expected_note(count: int, expected: int) -> str:
    return f" [{count}/{expected}回]" if count < expected else ""


@dataclass
class Row:
    variant: str
    label: str
    reference: bool = False


def level_rows(cfg: dict, level: str) -> list[Row]:
    """レベルの表の行。reference（比べる相手）を先頭に置き、そのあとにこのレベルの条件。"""
    rows = [Row(name, cfg["variants"][name]["label"], True) for name in cfg.get("levels", {}).get(level, {}).get("reference", [])]
    rows += [Row(name, spec["label"]) for name, spec in cfg["variants"].items() if spec["level"] == level]
    return rows


def row_title(row: Row) -> str:
    return f"{row.label} {row.variant}" + ("（参考）" if row.reference else "")


def collect_warnings(cfg: dict, views: list[dict]) -> list[str]:
    """集計の冒頭に出す警告。"""
    def name(v):
        return f"{v['rec'].get('label', '')} {v['rec'].get('variant')}/{v['rec'].get('qid')} t{v['trial']}"

    groups = {
        "status が ok ではない試行": [f"{name(v)}（{v['status']}）" for v in views if v["status"] != "ok"],
        "再試行した試行（API の失敗を待ってやり直した）": [
            f"{name(v)}（{', '.join(f'{a.get('source')}:{a.get('status')}' for a in v['rec'].get('attempts', []) if a.get('action') == 'retry')}）"
            for v in views if any(a.get("action") == "retry" for a in v["rec"].get("attempts", []))
        ],
        "モデルに見えた指示が、設定した条件と違う試行（instructions_match が false）": [name(v) for v in views if v["rec"].get("instructions_match") is False],
        "モデルが引数を直せば変わる、ツールの error を返した試行": [
            f"{name(v)}（{'; '.join(e['error'][:40] for e in v['rec'].get('tool_errors', []))}）" for v in views if v["rec"].get("tool_errors")
        ],
        "資料名・資料 ID・回の手がかりで 1 ページに決まらない引用（ambiguous）がある試行": [
            f"{name(v)}（{v['metrics']['ambiguous']} 件）" for v in views if v["metrics"]["ambiguous"]
        ],
        "要確認（required がすべて一致し、flag にも一致。回答を読んで決める。正答にも誤答にも数えない）の試行": [
            name(v) for v in views if v["score"] and v["score"]["verdict"] == "review"
        ],
        "記録の質問の文が、いまの conditions.toml と違う試行（質問を変えたあとは、別の --out に書く）": [name(v) for v in views if v["question_changed"]],
    }
    lines = []
    for title, items in groups.items():
        lines.append(f"- {title}: " + (f"{len(items)} 件 — " + "、".join(items[:12]) + (" …" if len(items) > 12 else "") if items else "なし"))
    return lines


# ---------------------------------------------------------------------------
# 集計 2: レベルごとの表 A〜D と、試行ごとの詳細
# ---------------------------------------------------------------------------


def summarize_level(cfg: dict, level: str, index: dict, expected: int, qids_filter, names_filter) -> tuple[list[str], dict] | None:
    rows = [r for r in level_rows(cfg, level) if not names_filter or r.variant in names_filter or r.reference]
    cells = []  # (row, qid, views)
    for row in rows:
        spec = cfg["variants"][row.variant]
        for qid in cfg["questions"]:
            if qid in spec["questions"] and (not qids_filter or qid in qids_filter):
                cells.append((row, qid, index.get((row.variant, qid), [])))
    if not any(views for _, _, views in cells):
        return None
    own = {r.variant for r in rows if not r.reference}
    if not any(views for row, _, views in cells if row.variant in own):
        return None  # このレベルの条件の結果がまだ無い（参考の行だけある）
    qids = [q for q in cfg["questions"] if any(qid == q for _, qid, _ in cells)]
    md = [f"### レベル: {level}", "", "質問:", ""]
    md += [f"- **{q}** {cfg['questions'][q]['text']}（gold: {', '.join(cfg['questions'][q].get('gold', [])) or 'なし'}）" for q in qids]
    md.append("")
    data: dict = {"rows": [dataclasses.asdict(r) for r in rows], "questions": qids}

    def table(title: str, header: list[str], body: list[list[str]]) -> list[str]:
        out = [f"#### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(cell_text(c) for c in line) + " |" for line in body]
        return out + [""]

    # 表 A: ツールの呼ばれ方
    body_a, data_a = [], {}
    for row, qid, views in cells:
        if not views:
            body_a.append([row_title(row), qid] + ["未実行"] * 4)
            continue
        modes = {v["rec"].get("mode") for v in views}
        data_a.setdefault(row_title(row), {})[qid] = {
            "trials": len(views), "tool_calls": [v["tool_calls"] for v in views], "requests": [v["requests"] for v in views],
            "args_t1": args_text(views[0]), "opened": [opened_count(v) for v in views] if modes == {"tree"} else None,
        }
        body_a.append([
            row_title(row), qid, per_trial(views, lambda v: v["tool_calls"]) + expected_note(len(views), expected),
            per_trial(views, lambda v: v["requests"]), args_text(views[0]),
            per_trial(views, opened_count) if modes == {"tree"} else "—",
        ])
    md += table("表A: ツールの呼ばれ方（t1,t2,t3 の順）", ["条件", "質問", "tool_calls", "requests", "引数（t1。検索語・目次のノード）", "開いたページ数（Tree）"], body_a)

    # 表 B: 根拠の取得
    body_b, data_b = [], {}
    for row, qid, views in cells:
        if not views:
            body_b.append([row_title(row), qid] + ["未実行"] * 7)
            continue
        modes = {v["rec"].get("mode") for v in views}
        data_b.setdefault(row_title(row), {})[qid] = {
            "gold": [{"hit": v["gold"]["hit"], "total": v["gold"]["total"]} for v in views],
            "full_pages": [full_count(v) for v in views], "chars": [chars_value(v) for v in views],
            "ranks": [v["gold"]["ranks"] for v in views],
        }
        body_b.append([
            row_title(row), qid, per_trial(views, gold_cell), per_trial(views, full_count),
            per_trial(views, lambda v: len(preview_only_ids(v["retrieved"]))) if modes == {"tree"} else "—",
            per_trial(views, graph_counts) if modes == {"graph"} else "—",
            per_trial(views, role_counts) if any(r["role"] for v in views for r in v["retrieved"]["full_text"]) else "—",
            per_trial(views, chars_value), " ".join(f"t{v['trial']}{rank_cell(v)}" for v in views if rank_cell(v)) or "—",
        ])
    md += table(
        "表B: 根拠の取得（gold の取得率は、全文で取得したページ ∩ gold ÷ gold。全文投入は、gold が対象資料に入っているか）",
        ["条件", "質問", "gold 取得", "全文で取得したページ数", "先頭文だけ（Tree）", "seeds／expanded（Graph）", "hit／neighbor", "取得した本文の字数", "gold の順位"], body_b)

    # 表 C1: 引用
    body_c, data_c = [], {}
    for row, qid, views in cells:
        if not views:
            body_c.append([row_title(row), qid] + ["未実行"] * 6)
            continue
        grid = Counter()
        for v in views:
            grid.update(v["metrics"]["grid"])
        data_c.setdefault(row_title(row), {})[qid] = {
            "pages": [v["metrics"]["pages"] for v in views], "gold": [{"cited": v["metrics"]["gold"], "total": v["metrics"]["gold_total"]} for v in views],
            "acceptable": [v["metrics"]["acceptable"] for v in views], "other": [v["metrics"]["other"] for v in views],
            "unresolved": [v["metrics"]["unresolved"] for v in views], "grid": dict(grid),
        }
        body_c.append([
            row_title(row), qid, per_trial(views, lambda v: v["metrics"]["pages"]), per_trial(views, cited_gold_cell),
            per_trial(views, lambda v: v["metrics"]["acceptable"]), per_trial(views, lambda v: v["metrics"]["other"]),
            per_trial(views, lambda v: v["metrics"]["unresolved"]),
            f"{grid['retrieved_cited']}/{grid['retrieved_not_cited']}/{grid['not_retrieved_cited']}/{grid['neither']}",
        ])
    md += table(
        "表C1: 引用（gold 引用 = 引用したページ ∩ gold ÷ gold。2×2 は gold ページごとの合計: 取れて引用した／取れたが引用しなかった／取らずに引用した／取れず引用もしない）",
        ["条件", "質問", "引用ページ数", "gold 引用", "acceptable", "それ以外", "未解決・曖昧", "取得×引用の 2×2"], body_c)

    # 表 C2: 自動チェック。行は（条件, 質問）。自動の判定は 3 つ（自動の正答・要確認・自動の誤答）
    body_c2, data_c2 = [], {}
    for row, qid, views in cells:
        scored = [v for v in views if v["score"] is not None]
        if not views:
            body_c2.append([row_title(row), qid] + ["未実行"] * 4)
            continue
        checks = checks_for(cfg, qid)
        check_cells = {check_label(c): fraction(sum(1 for v in scored if v["score"]["results"].get(check_label(c))), len(scored)) for c in checks}
        counts = Counter(v["score"]["verdict"] for v in scored)
        judged = any(check_role(c) == "required" for c in checks)
        verdicts = {name: (fraction(counts[name], len(scored)) if judged else "—") for name in VERDICTS}
        data_c2.setdefault(row_title(row), {})[qid] = {
            "checks": check_cells, **verdicts, "counts": {**{name: counts[name] for name in VERDICTS}, "trials": len(scored)},
        }
        body_c2.append([
            row_title(row), qid, "<br>".join(f"{label}（{check_role(c)}）{check_cells[label]}" for c in checks for label in [check_label(c)]) or "—",
            *(verdicts[name] for name in VERDICTS),
        ])
    md += table(
        "表C2: 自動チェック（チェックは、通った試行/試行数。自動の正答 = required がすべて一致し、flag が一致しない。"
        "要確認 = required がすべて一致し、flag も一致（回答を読んで決める。正答にも誤答にも数えない）。自動の誤答 = required のどれかが一致しない（flag は問わない））",
        ["条件", "質問", "チェック", *(VERDICT_NAMES[name] for name in VERDICTS)], body_c2)

    # 表 D: 時間・トークン・費用
    body_d, data_d = [], {}
    for row, qid, views in cells:
        if not views:
            body_d.append([row_title(row), qid, "0"] + ["未実行"] * 6)
            continue
        values = {
            "seconds": med_range([v["seconds"] for v in views], 1), "input": med_range([v["input"] for v in views]),
            "output": med_range([v["output"] for v in views]), "thoughts": med_range([v["thoughts"] for v in views]),
            "requests": med_range([v["requests"] for v in views]), "cost_usd": cost_cell([v["cost"] for v in views]),
        }
        data_d.setdefault(row_title(row), {})[qid] = {"trials": len(views), **values}
        body_d.append([row_title(row), qid, f"{len(views)}", *values.values()])
    md += table(
        "表D: 時間・トークン・費用（中央値 (最小–最大)。費用は生成 API の推定で、埋め込みを含まない。算定できなければ「算定不可」で、0 にしない）",
        ["条件", "質問", "試行数", "秒", "入力", "出力", "thoughts", "requests", "推定費用 USD"], body_d)

    # 試行ごとの詳細
    md += ["#### 試行ごとの詳細（回答の全文・ツールの引数・引用の raw）", ""]
    for row, qid, views in cells:
        if not views:
            continue
        md += ["<details>", f"<summary>{cell_text(row_title(row))} / {qid}（{len(views)} 回）</summary>", ""]
        for v in views:
            marks = " ".join(
                f"{'○' if ok else '×'}{label}" for label, ok in (v["score"]["results"].items() if v["score"] else [])
            )
            verdict = v["score"]["verdict"] if v["score"] else None
            flag = "（flag に一致）" if v["score"] and v["score"]["flagged"] else ""
            judgement = f"｜**要確認{flag}。目視で決める**" if verdict == "review" else (f"｜{VERDICT_NAMES[verdict]}{flag}" if verdict else "")
            md += [
                f"**t{v['trial']}**｜{v['status']}｜{v['seconds']} 秒｜tool_calls {v['tool_calls']}｜requests {v['requests']}｜"
                f"gold 取得 {gold_cell(v)}｜gold 引用 {cited_gold_cell(v)}｜{marks}{judgement}", "",
                fenced(v["rec"].get("answer") or "(回答なし)"), "",
            ]
            for call in v["rec"].get("tool_calls") or []:
                md.append(f"- `{call.get('name')}({args_json(call.get('args'))})` → {call.get('outcome')}")
            raws = [f"{c['raw']}（{c['status']}" + (f"。候補 {', '.join(c['page_ids'])}" if c["page_ids"] else "") + "）" for c in v["citations"]]
            md.append("- 引用の raw: " + ("、".join(raws) if raws else "なし"))
            md.append("")
        md += ["</details>", ""]
    data.update(table_a=data_a, table_b=data_b, table_c={"c1": data_c, "c2": data_c2}, table_d=data_d)
    return md, data


def cmd_summarize(args) -> int:
    cfg = load_config(args.conditions)
    out = Path(args.out) if args.out else DEFAULT_OUT
    levels = parse_levels(args.level)
    names = select_variants(cfg, args)
    qids = check_questions(cfg, args)
    if not names:
        print("条件に一致する条件がありません（--level / --variant を確かめてください）。")
        return 1
    trials = args.trials or cfg["defaults"]["trials"]
    models = discover_models(out)
    if not models:
        print(f"{out} に、集計できる結果がありません（<model>/<variant>/<qid>/t<k>.json）。")
        return 1
    visible = set(names) | {ref for level in levels for ref in cfg.get("levels", {}).get(level, {}).get("reference", [])}  # 参考の行も出す
    index_pages = page_index()
    total_pages = sum(len(p) for p in index_pages.pages.values())
    search_pages = sum(len(index_pages.pages.get(d, {})) for d in app_retrieval().RAG_DOCUMENT_IDS)
    md = [
        "# 比較の集計（compare.py summarize が作った。手で直さない）", "",
        f"結果の置き場所: `{out}`　試行の予定: {trials} 回", "",
        "## 前提", "",
        f"- 対象範囲: 全文投入 F は、教材 JSON の全 9 資料 {total_pages} ページ。F214 と、ベクトル検索・Tree・Graph・各改良は、第1〜3回の本講義 {search_pages} ページ。"
        "F と他の方式は対象範囲が違うので、方式の優劣は、同じ範囲の F214 と比べて言う。",
        f"- 試行数: 1 条件（条件 × 質問）につき {trials} 回。順番は、試行番号（外）→ 質問 → 条件。条件の順は、conditions.toml の order を、試行ごとに "
        f"{cfg['defaults']['order_shift']} 個ずつ巡回させている（時間帯や API の混み具合の影響と、条件の効果を区別しやすくするため。全文投入どうしが隣り合わないことは、"
        "読み込み時に確かめてある）。各試行の記録の position に、その試行での位置がある。",
        "- 全文投入（F・F214）の gold 取得は、「gold が対象資料に入っているか」で数えるので、取得の優劣を見る指標ではない。",
        "- 読み方: 3 回の値の範囲が基準（V0 など）と重ならなければ「変わった」、2 回分の差なら「傾向」、1 回分以下なら「揺れ」と書く。"
        "3/3 と 0/3 の差でも、Fisher の正確検定では p=0.1 なので、「有意」とは書かない。",
        "- gold 取得は「根拠のページを取れたか」、gold 引用は「回答が、そのページを引用したか」。取得×引用の 2×2 で、取れなかったのか、取れたのに使わなかったのかを分ける。",
        "- 自動の判定は 3 つ。自動の正答 = required がすべて一致し、flag が一致しない。要確認 = required がすべて一致し、flag も一致"
        "（回答を読んで決める。正答にも誤答にも数えない）。自動の誤答 = required のどれかが一致しない（flag は問わない）。bonus と reference は判定に入れない。"
        "どれも目安なので、回答は目視でも確かめる。引用の raw は、各条件の詳細に残してある。", "",
    ]
    data: dict = {"out": str(out), "expected_trials": trials, "models": {}}
    body, warnings = [], {}
    for model_dir in models:
        records = [r for r in load_model_records(model_dir)
                   if r.get("variant") in visible and (not qids or r.get("qid") in qids)]
        views = [trial_view(cfg, r) for r in records]
        warnings[model_dir.name] = collect_warnings(cfg, views)
        index: dict = defaultdict(list)
        for v in views:
            index[(v["rec"]["variant"], v["rec"]["qid"])].append(v)
        for items in index.values():
            items.sort(key=lambda v: v["trial"])
        body += [f"## モデル: {model_dir.name}", ""]
        data["models"][model_dir.name] = {}
        for level in LEVELS:
            if level not in levels:
                continue
            result = summarize_level(cfg, level, index, trials, qids, names)
            if result is None:
                continue
            body += result[0]
            data["models"][model_dir.name][level] = result[1]
    md += ["## 警告", ""]
    for model_name, lines in warnings.items():
        md += [f"モデル {model_name}:", *lines, ""]
    data["warnings"] = warnings
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "summary.md", "\n".join(md + body) + "\n")
    write_json_atomic(out / "summary.json", data)
    print(f"集計しました: {out / 'summary.md'}（と summary.json）")
    return 0


# ---------------------------------------------------------------------------
# status / variants
# ---------------------------------------------------------------------------


def last_status(ctx: Context, cond: Condition) -> str:
    paths = existing_trials(ctx, cond)
    if not paths:
        return "-"
    newest = max(paths, key=lambda p: int(re.search(r"\d+", p.name).group(0)))
    try:
        return json.loads(newest.read_text(encoding="utf-8")).get("status", "?")
    except (OSError, ValueError):
        return "読めません"


def cmd_status(args) -> int:
    cfg = load_config(args.conditions)
    ctx = make_context(args, cfg, with_git=False)
    plan = make_plan(cfg, args)  # 試行 1 の、実行する順（質問 → 条件）。試行ごとの順は、「順」の列に出す
    rows = [("質問", "条件", "レベル", "順（試行ごと）", "済み/予定", "最後の status", "残り", "残りの推定リクエスト")]
    total_planned = total_done = 0
    total_requests = 0.0
    full_left = 0
    for cond in plan:
        done = sum(trial_done(ctx, cond, k) for k in range(1, ctx.trials + 1))
        left = ctx.trials - done
        requests = left * ESTIMATED_REQUESTS[cond.mode]
        positions = ",".join(str(trial_position(cfg, cond, k)["in_question"]) for k in range(1, ctx.trials + 1))
        rows.append((cond.qid, f"{cond.label} {cond.variant}", cond.level, positions, f"{done}/{ctx.trials}", last_status(ctx, cond), str(left), f"{requests:.1f}"))
        total_planned, total_done, total_requests = total_planned + ctx.trials, total_done + done, total_requests + requests
        full_left += left if cond.mode == "full" else 0
    rows.append(("合計", "", "", "", f"{total_done}/{total_planned}", "", str(total_planned - total_done), f"{total_requests:.1f}"))
    print(f"結果の置き場所: {ctx.out}　モデル: {ctx.model}　試行: {ctx.trials} 回")
    print("実行する順に並べてある（試行番号 → 質問 → 条件）。「順」は、その質問の中での位置を、試行ごとに示す（試行ごとに条件の順を "
          f"{cfg['defaults']['order_shift']} 個ずつ巡回させるので、試行ごとに違う。表の並びは試行 1 の順）")
    print("残りの推定リクエスト = 残りの実行回数 × 目安（生成のみ。full 1、vector・graph 2.5、tree 6。質問の埋め込みは別）。無料枠の 1 日の上限は AI Studio の画面で確かめる")
    print_table(rows)
    print(f"\n残りの試行 {total_planned - total_done} 回のうち、全文投入（60 秒の入力トークンの予算で待つ）は {full_left} 回")
    return 0


def cmd_variants(args) -> int:
    cfg = load_config(args.conditions)
    out = Path(args.out) if args.out else DEFAULT_OUT
    names = select_variants(cfg, args)
    check_questions(cfg, args)  # 質問の綴り違いも、ここで報告する（variants では質問を使わない）
    diffs = variant_diffs(cfg)
    warnings = 0
    for name in names:
        info, spec = diffs[name], cfg["variants"][name]
        eff = info["effective"]
        base = out / "variants" / name
        write_json_atomic(base.with_name(name + ".json"), {
            "variant": name, "label": eff.label, "level": eff.level, "mode": eff.mode, "base": eff.base, "diff_baseline": info["baseline"],
            "sha256": eff.sha256, "hunks": info["hunks"], "changes": eff.changes, "settings": eff.settings, "settings_effective": eff.settings_effective,
            "documents": eff.documents, "request_limit": eff.request_limit, "instructions": eff.instructions, "tools": eff.tool_defs,
            "retrieval": {tool: v["function"] for tool, v in eff.retrieval.items()}, "code_sha256": eff.code_files,
        })
        write_text_atomic(base.with_name(name + ".txt"), info["text"])
        if info["baseline"] is None:
            print(f"=== {eff.label} {name}: {spec.get('description', '')}（基準なし。sha256 {eff.sha256[:12]}）===\n")
            continue
        if info["baseline"] == name:
            print(f"=== {eff.label} {name}（基準。sha256 {eff.sha256[:12]}）===\n")
            continue
        print(f"=== {eff.label} {name}: {info['baseline']} との差分 ===")
        print("\n".join(info["diff"]) if info["diff"] else "(差分なし)")
        if info["hunks"] != 1:
            warnings += 1
            print(f"⚠ 差分が {info['hunks']} か所です（条件は、基準と 1 か所だけ違うはず）")
        print()
    print(f"保存先: {out / 'variants'}（条件 {len(names)} 件）")
    return 1 if warnings else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help=f"結果の置き場所（既定: {RESULTS_REL}。--dry-run は一時フォルダ）")
    common.add_argument("--conditions", help="条件のファイル（既定: lectures/05/conditions.toml）")
    common.add_argument("--level", nargs="+", metavar="LEVEL", help="exercise basic impl adv all（, 区切りで複数可。既定: all）")
    common.add_argument("--variant", nargs="+", metavar="VARIANT", help="条件の名前か label（F・V0 など。, 区切りで複数可。絞り込み）")
    common.add_argument("--question", nargs="+", metavar="ID", help="質問の ID（絞り込み）。例: q1 qa2")
    common.add_argument("--trials", type=int, help="試行の回数（既定: conditions.toml の trials = 3）")

    parser = argparse.ArgumentParser(description="検索の方式と取得処理を変えて、講義資料に答える RAG エージェントの動きを比べる（実行・集計）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status", parents=[common], help="条件×質問ごとの済み・予定・残りと、LLM リクエストの目安（API なし）")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("variants", parents=[common], help="条件ごとに、モデルに見える指示・ツール・取得処理・設定を保存し、基準との差分を表示（API なし）")
    p.set_defaults(func=cmd_variants)
    p = sub.add_parser("run", parents=[common], help="実行する（既にある結果は飛ばす）")
    p.add_argument("--interval", type=float, help="実行と実行の間の秒数（既定: conditions.toml の interval = 8。予行では待たない）")
    p.add_argument("--max-runs", type=int, help="この実行で行う回数の上限")
    p.add_argument("--dry-run", action="store_true", help="API を呼ばずに予行する（偽のモデル・偽の埋め込み。結果は一時フォルダ）")
    p.add_argument("--allow-dirty", action="store_true", help="未コミットの変更があっても実行する（開発用。手順書では使わない）")
    p.add_argument("--tpm-budget", type=int, help="全文投入の前に、直近 60 秒の入力トークンをこの値以下にするよう待つ（既定: conditions.toml の tpm_budget。0 で無効）")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("summarize", parents=[common], help="summary.md と summary.json を作る")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("trials", "max_runs"):
        if getattr(args, name, None) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} は 1 以上にしてください")
    if getattr(args, "tpm_budget", None) is not None and args.tpm_budget < 0:
        parser.error("--tpm-budget は 0 以上にしてください")
    if getattr(args, "interval", None) is not None and args.interval < 0:
        parser.error("--interval は 0 以上にしてください")
    args._argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        return args.func(args)
    except StopRun as e:
        print(mask(e.message), file=sys.stderr)  # API のエラーの本文を表示することがあるので、キーは置き換える
        return e.code
    except ConfigError as e:
        print(mask(f"設定エラー: {e}"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断しました。同じコマンドで続きから実行できます。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
