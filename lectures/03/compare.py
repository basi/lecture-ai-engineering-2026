#!/usr/bin/env python3
"""第3回: 条件を変えて、エージェントの動きを比べるための実行・集計スクリプト。

何を比べるか（質問・変種・シナリオ・自動チェック）は conditions.toml に書いてある。
このスクリプトは、それを読んで実行し、結果を results/ に保存し、集計する。

使い方（リポジトリの直下で）:
    uv run python lectures/03/compare.py status                        # 予定・済み・残りの件数と、推定リクエスト数（API なし）
    uv run python lectures/03/compare.py variants                      # 変種ごとに、モデルに見える指示とツールを保存し、baseline との差分を表示
    uv run python lectures/03/compare.py run --level all --dry-run     # TestModel と偽の client で予行（API なし、出力は一時フォルダ）
    uv run python lectures/03/compare.py run --level basic --trials 1  # 段階ごとに 1 回目
    uv run python lectures/03/compare.py run --level all --trials 3    # 2・3 回目を足す（既にある結果は飛ばす）
    uv run python lectures/03/compare.py summarize                     # results/summary.md と summary.json を作る

レベル: exercise（演習）・basic（基礎 V0〜V5）・impl（実装 s1〜s6）・adv-a（ADK）・adv-b（Pi）・all。
エンジン: pydantic（app/agent.py のエージェント）・ask（第1回の素の LLM）・adk（ADK 版）・pi（Pi）。
試行番号を外側に回す（全条件の t1 → t2 → t3）。止まっても、同じコマンドで続きから実行できる。

保存先: <out>/<model>/<engine>/<variant>/<qid>/t<k>.json（ほかに t<k>.trace.txt。adk と pi は
        t<k>.raw.jsonl と t<k>.stderr.txt も）。変種の見え方は <out>/variants/pydantic/<variant>.json と .txt。
        t<k>.json があれば「済み」。成功したときだけ、最後に置く（一時ファイルに書いてから os.replace）。
        予行（--dry-run）は <model>-dry-run のフォルダに書くので、本物の結果とは混ざらない。

終了コード: 0 = 正常 / 1 = 止まった（設定の誤り・API エラー・未コミットの変更など）/ 2 = 1 日の上限（PerDay）。

秘密: 書き出す前に、GEMINI_API_KEY・GOOGLE_API_KEY の値と AIza… の形の文字列を *** に置き換える。
      環境変数の一覧は記録しない。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import hashlib
import io
import itertools
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # 「python lectures/03/compare.py」で動かすと、app/ が見つからないため

CONDITIONS_FILE = HERE / "conditions.toml"
DEFAULT_OUT = HERE / "results"

LEVELS = ("exercise", "basic", "impl", "adv-a", "adv-b")
ENGINES = ("pydantic", "ask", "adk", "pi")
ADK_REQUIREMENT = "google-adk==2.13.0"
REQUEST_LIMIT = 10  # 1 回の run の LLM 呼び出しの上限（ループの暴走を防ぐ）。承認の 2 回目の run と合計
SUBPROCESS_TIMEOUT = 300  # adk・pi の 1 回の実行の上限（秒）
RATE_LIMIT_RETRIES = 5  # 429 の再試行の回数
RATE_LIMIT_DEFAULT_WAIT = 60  # 429 で待ち時間が分からないときの秒数
SERVER_ERROR_WAITS = (30, 60, 120)  # 503・500・タイムアウトの再試行の前に待つ秒数
GUARDED_PATHS = ["app", "adk_agent", "lectures/03/compare.py", "lectures/03/conditions.toml"]
# 残りの LLM リクエスト数の目安（1 回の実行あたり）
ESTIMATED_REQUESTS = {"pydantic": 2.5, "ask": 1, "adk": 2.5, "pi": 6, "impl": 3.5}
DAILY_LIMIT_MESSAGE = (
    "1 日の上限（PerDay）に達しました。明日（日本時間 17 時以降）、同じコマンドで続きから実行してください。"
    "（済みの結果は飛ばされます）"
)


class ConfigError(Exception):
    """conditions.toml の誤り。API を呼ぶ前に止める。"""


class StopRun(Exception):
    """実行を止める。code は終了コード。"""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# 条件（conditions.toml）の読み込みと、実行する条件の組み立て
# ---------------------------------------------------------------------------


def load_config(path: Path | str | None = None) -> dict:
    """conditions.toml を読み、形を確かめる。"""
    path = Path(path) if path else CONDITIONS_FILE
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ConfigError(f"{path} を読めません: {e}") from e
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    questions, scenarios, variants = cfg.get("questions", {}), cfg.get("scenarios", {}), cfg.get("variants", {})
    for name, item in {**questions, **scenarios}.items():
        if not item.get("text"):
            raise ConfigError(f"{name}: text がありません")
        labels = []
        for check in item.get("checks", []):
            if check.get("kind") not in CHECKS:
                raise ConfigError(f"{name}: 知らないチェックの種類です: {check.get('kind')!r}（種類は {sorted(CHECKS)}）")
            labels.append(check_label(check))
        if len(labels) != len(set(labels)):
            raise ConfigError(f"{name}: チェックの label が重なっています（表で区別できません）")
    for name, scenario in scenarios.items():
        for decision in scenario.get("approvals", []):
            if decision not in ("approve", "deny"):
                raise ConfigError(f"{name}: approvals は approve か deny です: {decision!r}")
    if "baseline" not in variants:
        raise ConfigError("variants.baseline がありません（すべての比較の基準）")
    for name, variant in variants.items():
        for qid in variant.get("questions", []):
            if qid not in questions:
                raise ConfigError(f"variants.{name}: 質問 {qid} が questions にありません")
        for sid in variant.get("scenarios", []):
            if sid not in scenarios:
                raise ConfigError(f"variants.{name}: シナリオ {sid} が scenarios にありません")
        for change in variant.get("changes", []):
            if change.get("op") not in ("delete_line", "replace_line", "replace_text"):
                raise ConfigError(f"variants.{name}: op は delete_line・replace_line・replace_text のどれかです: {change.get('op')!r}")
            if not change.get("old") or "target" not in change:
                raise ConfigError(f"variants.{name}: changes には target と old が必要です")
            if change["op"] != "delete_line" and "new" not in change:
                raise ConfigError(f"variants.{name}: {change['op']} には new が必要です")
    for level, spec in cfg.get("levels", {}).items():
        for run in spec.get("runs", []):
            if run.get("engine") not in ENGINES:
                raise ConfigError(f"levels.{level}: engine が不明です: {run.get('engine')!r}")
            for qid in run.get("questions", []):
                if qid not in questions:
                    raise ConfigError(f"levels.{level}: 質問 {qid} が questions にありません")


@dataclass(frozen=True)
class Condition:
    """実行する 1 つの条件（エンジン × 変種 × 質問）。"""

    level: str
    engine: str
    variant: str
    qid: str
    kind: str  # "question" か "scenario"
    text: str
    approvals: tuple = ()
    optional: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.engine, self.variant, self.qid)


def level_runs(cfg: dict, level: str, include_optional: bool = True) -> list[tuple[str, str, list[str], bool]]:
    """レベルに含まれる (エンジン, 変種, 質問かシナリオの ID, 任意か) を、実行する順に返す。"""
    runs = []
    if level in ("basic", "impl"):
        for name, variant in cfg["variants"].items():
            if variant.get("level") != level:
                continue
            optional = bool(variant.get("optional"))
            if optional and not include_optional:
                continue
            ids = variant.get("scenarios" if level == "impl" else "questions", [])
            runs.append(("pydantic", name, list(ids), optional))
    else:
        for run in cfg.get("levels", {}).get(level, {}).get("runs", []):
            runs.append((run["engine"], run["variant"], list(run["questions"]), False))
    return runs


def build_plan(
    cfg: dict,
    *,
    levels=LEVELS,
    engines=None,
    variants=None,
    questions=None,
    include_optional: bool = False,
) -> list[Condition]:
    """実行する条件を、実行する順に返す。同じ条件（エンジン・変種・質問）は 1 回だけ（演習と基礎の baseline など）。"""
    plan, seen = [], set()
    for level in LEVELS:
        if level not in levels:
            continue
        for engine, variant, ids, optional in level_runs(cfg, level, include_optional):
            if (engines and engine not in engines) or (variants and variant not in variants):
                continue
            for qid in ids:
                if questions and qid not in questions:
                    continue
                if (engine, variant, qid) in seen:
                    continue
                seen.add((engine, variant, qid))
                scenario = cfg.get("scenarios", {}).get(qid)
                item = scenario if scenario is not None else cfg["questions"][qid]
                plan.append(
                    Condition(
                        level, engine, variant, qid,
                        "scenario" if scenario is not None else "question",
                        item["text"], tuple(item.get("approvals", ())), optional,
                    )
                )
    return plan


def checks_for(cfg: dict, qid: str) -> list[dict]:
    item = cfg.get("scenarios", {}).get(qid) or cfg["questions"][qid]
    return list(item.get("checks", []))


# ---------------------------------------------------------------------------
# 変種: モデルに見える指示とツールの説明文を、1 か所だけ変える
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


@dataclass
class Effective:
    """変種を当てた結果。モデルに見える指示と、ツール（複製した Tool）。"""

    instructions: str
    tools: list
    tool_defs: list
    sha256: str


def sha256_of(data) -> str:
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def registered_tools() -> dict:
    """agent に登録済みの Tool を、名前で引ける dict にする（agent.toolsets は public）。"""
    from pydantic_ai.toolsets import FunctionToolset

    from app.agent import agent

    return {t.name: t for ts in agent.toolsets if isinstance(ts, FunctionToolset) for t in ts.tools.values()}


def build_effective(cfg: dict, variant: str) -> Effective:
    """変種を、app/agent.py の INSTRUCTIONS と登録済みツールに当てる。app/ のファイルは書き換えない。"""
    from app.agent import INSTRUCTIONS

    spec = cfg["variants"][variant]
    names = spec.get("tools") or cfg["defaults"]["tools"]
    changes = spec.get("changes", [])
    allowed = {"instructions", *(f"tool:{n}" for n in names)}
    for change in changes:
        if change["target"] not in allowed:
            raise ConfigError(f"variants.{variant}: target {change['target']!r} は {sorted(allowed)} のどれかにしてください")
    available = registered_tools()
    instructions = apply_changes(INSTRUCTIONS, changes, "instructions", f"variants.{variant}")
    tools = []
    for name in names:
        if name not in available:
            raise ConfigError(f"variants.{variant}: ツール {name} は agent に登録されていません")
        base = available[name]
        description = apply_changes(base.description or "", changes, f"tool:{name}", f"variants.{variant}")
        # 素の関数ではなく Tool を複製して渡す（requires_approval などの設定を保つため）
        tools.append(dataclasses.replace(base, description=description))
    tool_defs = []
    for tool in tools:
        d = tool.tool_def
        tool_defs.append(
            {"name": d.name, "description": d.description, "parameters_json_schema": d.parameters_json_schema, "kind": d.kind}
        )
    return Effective(instructions, tools, tool_defs, sha256_of({"instructions": instructions, "tools": tool_defs}))


def render_effective(eff: Effective) -> str:
    """モデルに見える指示とツール定義を、差分を取りやすい文字列にする。"""
    lines = ["[instructions]", *eff.instructions.rstrip("\n").split("\n"), "", "[tools]"]
    for d in eff.tool_defs:
        lines.append(f"- {d['name']} ({d['kind']})")
        lines += [("    " + line).rstrip() for line in (d["description"] or "").split("\n")]
        props = d["parameters_json_schema"].get("properties", {})
        lines.append(f"    引数: {json.dumps(props, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def variant_diffs(cfg: dict) -> dict[str, dict]:
    """変種ごとに baseline との unified diff と、差分の数（hunk）を返す。"""
    base = render_effective(build_effective(cfg, "baseline"))
    result = {}
    for name in cfg["variants"]:
        eff = build_effective(cfg, name)
        text = render_effective(eff)
        diff = list(difflib.unified_diff(base.splitlines(), text.splitlines(), "baseline", name, lineterm=""))
        result[name] = {
            "effective": eff,
            "text": text,
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


def write_text_atomic(path: Path, text: str) -> None:
    """一時ファイルに書いてから os.replace で置く（途中で止まっても、半端なファイルが残らない）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(mask(text), encoding="utf-8")
    os.replace(tmp, path)


def write_json_atomic(path: Path, data) -> None:
    write_text_atomic(path, json.dumps(mask(data), ensure_ascii=False, indent=2, default=str) + "\n")


def now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# git: 結果がどの commit のものかを確かにする
# ---------------------------------------------------------------------------


def git_state(repo: Path = REPO, paths=GUARDED_PATHS) -> dict:
    """commit と、実行に効くファイル（app・adk_agent・このスクリプトと conditions.toml）の未コミットの変更。"""

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", *paths)
    dirty_files = [line for line in status.stdout.splitlines() if line.strip()] if status.returncode == 0 else []
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


# ---------------------------------------------------------------------------
# API エラーの分類と、再試行
# ---------------------------------------------------------------------------


class ApiFailure(Exception):
    """各エンジンの API の失敗を、共通の形にしたもの。"""

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


def judge(failure: ApiFailure, counters: dict) -> tuple[str, float | None]:
    """('retry', 待つ秒数)・('daily_limit', None)・('stop', None) のどれかを返す。

    429: 本文に PerDay があれば日の上限。なければ、待って最大 5 回まで再試行する。
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
        wait = failure.retry_after or retry_delay_from_text(text) or RATE_LIMIT_DEFAULT_WAIT
        return "retry", float(wait)
    server_side = (failure.status is not None and failure.status >= 500) or (failure.status is None and failure.transport)
    if server_side:
        counters["server"] = counters.get("server", 0) + 1
        if counters["server"] > len(SERVER_ERROR_WAITS):
            return "stop", None
        return "retry", float(SERVER_ERROR_WAITS[counters["server"] - 1])
    return "stop", None


STATUS_WORDS = {
    "RESOURCE_EXHAUSTED": 429, "UNAVAILABLE": 503, "INTERNAL": 500, "DEADLINE_EXCEEDED": 504,
    "INVALID_ARGUMENT": 400, "UNAUTHENTICATED": 401, "PERMISSION_DENIED": 403, "NOT_FOUND": 404,
}
TRANSPORT_WORDS = re.compile(r"timed out|timeout|ConnectError|ConnectionError|Connection reset|Temporary failure|ECONNRESET|ETIMEDOUT", re.I)


def failure_from_text(text: str) -> ApiFailure:
    """adk・pi の stderr や JSONL のエラー文から、status を推測して ApiFailure にする。"""
    status = None
    match = re.search(r"\b(429|500|502|503|504|400|401|403|404)\b", text)
    if match:
        status = int(match.group(1))
    else:
        status = next((code for word, code in STATUS_WORDS.items() if word in text), None)
    if status is None and "PerDay" in text:
        status = 429
    return ApiFailure(status, text, transport=status is None and bool(TRANSPORT_WORDS.search(text)))


def call_with_retries(ctx: Context, fn, attempts: list):
    """fn() を実行する。ApiFailure が出たら、judge に従って待って再試行する。記録は attempts に足す。"""
    counters: dict = {}
    while True:
        try:
            value = fn()
        except ApiFailure as failure:
            action, wait = judge(failure, counters)
            attempts.append({"attempt": len(attempts) + 1, "status": failure.status, "error": failure.text[:500], "action": action, "wait_s": wait})
            if action == "daily_limit":
                raise StopRun(2, DAILY_LIMIT_MESSAGE) from failure
            if action == "stop":
                hint = ""
                if failure.status is None or failure.status >= 500:
                    hint = "\n503 などが続くときは、AIE_MODEL=gemini-3.1-flash-lite に切り替えて、その巡の全条件を同じモデルでやり直してください（表は分かれます）。"
                raise StopRun(1, f"API エラーで止めます（{failure.status}）: {failure.text[:300]}{hint}\n同じコマンドで続きから実行できます。") from failure
            ctx.wait(wait, f"API エラー（{failure.status}）")
            continue
        attempts.append({"attempt": len(attempts) + 1, "ok": True})
        return value


# ---------------------------------------------------------------------------
# 実行の状態（Context）と、1 回の実行の結果（TrialOutput）
# ---------------------------------------------------------------------------


@dataclass
class Context:
    cfg: dict
    out: Path
    model: str
    pi_model: str
    dry_run: bool = False
    interval: float = 0.0
    git: dict = field(default_factory=dict)
    argv: list = field(default_factory=list)
    pi_version: str | None = None
    _effective: dict = field(default_factory=dict)

    def effective(self, variant: str) -> Effective:
        if variant not in self._effective:
            self._effective[variant] = build_effective(self.cfg, variant)
        return self._effective[variant]

    def model_for(self, engine: str) -> str:
        return self.pi_model if engine == "pi" else self.model

    def model_dir(self, engine: str) -> str:
        """予行の結果は <model>-dry-run のフォルダに書く（本物の結果と混ざらないように）。"""
        name = self.model_for(engine)
        return f"{name}-dry-run" if self.dry_run else name

    def wait(self, seconds: float, why: str) -> None:
        print(f"  … {why}。{seconds:.0f} 秒待って、再試行します", flush=True)
        time.sleep(seconds)

    def sha_for(self, cond: Condition) -> str:
        """条件の中身の sha256。同じ条件の既存の結果と食い違えば、途中でコードが変わったと分かる。"""
        if cond.engine == "pydantic":
            return self.effective(cond.variant).sha256
        if cond.engine == "ask":
            return sha256_of({"engine": "ask", "model": self.model})
        if cond.engine == "adk":
            agent_file = REPO / "adk_agent" / "glossary_agent" / "agent.py"
            return sha256_of({
                "engine": "adk", "adk": ADK_REQUIREMENT, "model": self.model,
                "baseline": self.effective("baseline").sha256,
                "agent_py": hashlib.sha256(agent_file.read_bytes()).hexdigest() if agent_file.exists() else None,
            })
        return sha256_of({"engine": "pi", "command": pi_command("<質問>", self.pi_model)})


@dataclass
class TrialOutput:
    record: dict
    trace: str
    raw: str | None = None  # adk・pi の標準出力（JSONL）
    stderr: str | None = None


def trial_stem(ctx: Context, cond: Condition, k: int) -> Path:
    return ctx.out / ctx.model_dir(cond.engine) / cond.engine / cond.variant / cond.qid / f"t{k}"


def sibling(stem: Path, suffix: str) -> Path:
    return stem.with_name(stem.name + suffix)


def trial_done(ctx: Context, cond: Condition, k: int) -> bool:
    return sibling(trial_stem(ctx, cond, k), ".json").exists()


def save_trial(stem: Path, output: TrialOutput) -> None:
    """trace・raw・stderr を先に置き、最後に t<k>.json を置く（json があれば「済み」）。"""
    write_text_atomic(sibling(stem, ".trace.txt"), output.trace)
    if output.raw is not None:
        write_text_atomic(sibling(stem, ".raw.jsonl"), output.raw)
    if output.stderr is not None:
        write_text_atomic(sibling(stem, ".stderr.txt"), output.stderr)
    write_json_atomic(sibling(stem, ".json"), output.record)


def base_record(ctx: Context, cond: Condition, k: int, *, command, seed=None) -> dict:
    return {
        "schema": 1,
        "engine": cond.engine,
        "variant": cond.variant,
        "variant_sha256": ctx.sha_for(cond),
        "qid": cond.qid,
        "kind": cond.kind,
        "question": cond.text,
        "trial": k,
        "model": ctx.model_for(cond.engine),
        "seed": seed,
        "command": command,
        "compare_args": ctx.argv,
        "dry_run": ctx.dry_run,
        "git": {"commit": ctx.git.get("commit"), "dirty": ctx.git.get("dirty")},
        "started_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# エンジン 1: pydantic（app/agent.py のエージェント）
# ---------------------------------------------------------------------------


def make_pydantic_model(cond: Condition, ctx: Context):
    """予行は TestModel（API を呼ばない）。本番は app.agent.build_model()。"""
    if ctx.dry_run:
        from pydantic_ai.models.test import TestModel

        return TestModel(call_tools=["save_learning_log"]) if cond.variant == "impl" else TestModel()
    from app.agent import build_model

    return build_model()


def make_decider(plan: tuple | list, log: list):
    """承認待ちの呼び出しごとに、シナリオの答え（足りない分は deny）を返す。記録は log に足す。"""
    state = {"n": 0}

    def decide(call) -> bool:
        decision = plan[state["n"]] if state["n"] < len(plan) else "deny"
        state["n"] += 1
        log.append({
            "index": state["n"], "tool_call_id": call.tool_call_id, "tool_name": call.tool_name,
            "args": args_dict(call), "decision": decision,
        })
        return decision == "approve"

    return decide


def args_dict(part) -> dict | str:
    try:
        return part.args_as_dict()
    except Exception:  # noqa: BLE001  引数が JSON として読めないとき
        return part.args if isinstance(part.args, str) else str(part.args)


def as_text(content) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)


def collect_tool_calls(messages) -> list[dict]:
    """会話から、ツール呼び出しを順に取り出す（何回目の応答か、名前、引数、戻り値の全文、outcome）。"""
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart

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


def run_pydantic_trial(ctx: Context, cond: Condition, k: int) -> TrialOutput:
    from pydantic_ai import capture_run_messages
    from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest
    from pydantic_ai.usage import UsageLimits

    from app import tools
    from app.agent import agent, print_trace, run_with_approval

    eff = ctx.effective(cond.variant)
    seed = f"{cond.qid}:t{k}"
    attempts: list = []

    def attempt() -> dict:
        random.seed(seed)  # make_quiz の出題を、変種どうしでそろえる
        with tempfile.TemporaryDirectory(prefix="compare-log-") as root:
            old_log_dir = tools.LOG_DIR
            tools.LOG_DIR = Path(root) / "learning_log"  # 本物の learning_log/ を汚さない
            approvals: list = []
            try:
                with capture_run_messages() as captured, agent.override(instructions=eff.instructions, tools=eff.tools):
                    started = time.perf_counter()
                    result, status, error = None, "ok", None
                    try:
                        result = run_with_approval(
                            agent, cond.text, decide=make_decider(cond.approvals, approvals),
                            model=make_pydantic_model(cond, ctx), usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
                        )
                    except UsageLimitExceeded as e:
                        status, error = "usage_limit_exceeded", {"type": type(e).__name__, "message": str(e)}
                    except UnexpectedModelBehavior as e:
                        status, error = "unexpected_model_behavior", {"type": type(e).__name__, "message": str(e)}
                    except ModelAPIError as e:
                        raise ApiFailure(
                            getattr(e, "status_code", None), str(e), getattr(e, "retry_after", None),
                            transport=not isinstance(e, ModelHTTPError),
                        ) from e
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

    started_at = time.perf_counter()
    got = call_with_retries(ctx, attempt, attempts)
    elapsed = time.perf_counter() - started_at
    result, status = got["result"], got["status"]

    record = base_record(ctx, cond, k, command=pydantic_command(cond), seed=seed)
    answer, messages, usage = "", got["partial"], {}
    messages_json = ModelMessagesTypeAdapter.dump_json(messages)  # 失敗したときは、途中までの会話
    if result is not None:
        messages = result.all_messages()
        messages_json = result.all_messages_json()
        usage = dataclasses.asdict(result.usage)
        if isinstance(result.output, str):
            answer = result.output
        else:
            status = "pending_approvals"  # max_rounds までに承認待ちが終わらなかった
    first_request = next((m for m in messages if isinstance(m, ModelRequest)), None)
    seen = first_request.instructions if first_request is not None and first_request.instructions else eff.instructions
    tool_calls = collect_tool_calls(messages)
    record.update({
        "finished_at": now_iso(), "duration_s": round(got["duration"], 3), "elapsed_s": round(elapsed, 3),
        "status": status, "error": got["error"], "attempts": attempts, "answer": answer,
        # モデルに実際に見えた指示（最初の request の instructions）。設定した変種と食い違えば、instructions_match が false
        "instructions": seen, "instructions_sha256": hashlib.sha256(seen.encode("utf-8")).hexdigest(),
        "instructions_match": seen.strip() == eff.instructions.strip(),  # pydantic-ai は末尾の改行を落とす
        "tools": eff.tool_defs, "tools_sha256": sha256_of(eff.tool_defs),
        "tool_calls": tool_calls, "approvals": got["approvals"], "saved_files": got["saved"], "outside_files": got["outside"],
        "usage": usage, "messages": json.loads(messages_json),
    })

    buffer = io.StringIO()
    if result is not None:
        with contextlib.redirect_stdout(buffer):
            print_trace(result)  # CLI の --trace と同じ表示
    lines = [f"[model] {ctx.model_for('pydantic')}", f"[質問] {cond.text}"]
    lines += [f"[approval] {a['tool_name']}({json.dumps(a['args'], ensure_ascii=False)}) -> {a['decision']}" for a in got["approvals"]]
    lines += [answer if result is not None else f"[{status}] {(got['error'] or {}).get('message', '')}", buffer.getvalue().rstrip("\n")]
    return TrialOutput(record, "\n".join(lines).rstrip("\n") + "\n")


def pydantic_command(cond: Condition) -> str:
    return (
        f"app.agent.run_with_approval(agent, <質問>, model=build_model(), usage_limits=UsageLimits(request_limit={REQUEST_LIMIT}))"
        f" を agent.override(instructions=<変種>, tools=<変種>) の中で実行（variant={cond.variant}）"
    )


# ---------------------------------------------------------------------------
# エンジン 2: ask（第1回の素の LLM）
# ---------------------------------------------------------------------------


class RecordingClient:
    """genai.Client の包み。app.ask.ask() が呼ぶ models.generate_content を本物に渡し、応答を記録する。"""

    def __init__(self, client):
        self._client = client
        self.responses: list = []

    @property
    def models(self):
        return self

    def generate_content(self, *, model, contents):
        response = self._client.models.generate_content(model=model, contents=contents)
        self.responses.append(response)
        return response


class FakeGenAIClient:
    """予行用の偽の client（API を呼ばない）。"""

    def __init__(self):
        self.models = self

    def generate_content(self, *, model, contents):
        usage = SimpleNamespace(prompt_token_count=11, candidates_token_count=22, thoughts_token_count=0, total_token_count=33)
        return SimpleNamespace(text="（予行）素の LLM の回答です。", usage_metadata=usage)


def is_transport_error(e: Exception) -> bool:
    module = type(e).__module__.split(".")[0]
    return module in ("httpx", "httpx2", "httpcore") or isinstance(e, (TimeoutError, ConnectionError))


def run_ask_trial(ctx: Context, cond: Condition, k: int) -> TrialOutput:
    from google.genai import errors as genai_errors

    from app import ask as ask_module

    attempts: list = []

    def attempt() -> dict:
        buffer = io.StringIO()
        started = time.perf_counter()
        with contextlib.ExitStack() as stack:
            if ctx.dry_run:
                real = FakeGenAIClient()
            else:
                from google import genai

                from app.config import get_api_key

                real = stack.enter_context(genai.Client(api_key=get_api_key()))
            client = RecordingClient(real)
            try:
                with contextlib.redirect_stdout(buffer):  # app.ask の [metrics] の行を取る
                    answer = ask_module.ask(cond.text, [], client=client)
            except genai_errors.APIError as e:
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                try:
                    retry_after = float(headers.get("retry-after")) if headers.get("retry-after") else None
                except (TypeError, ValueError):
                    retry_after = None
                raise ApiFailure(e.code, str(e), retry_after) from e
            except RuntimeError as e:  # モデルからテキスト回答が無い（ask が出す）
                return {"answer": "", "status": "no_answer", "error": {"type": "RuntimeError", "message": str(e)},
                        "duration": time.perf_counter() - started, "metrics": buffer.getvalue(),
                        "response": client.responses[-1] if client.responses else None}
            except Exception as e:
                if is_transport_error(e):
                    raise ApiFailure(None, repr(e), transport=True) from e
                raise
        return {"answer": answer, "status": "ok", "error": None, "duration": time.perf_counter() - started,
                "metrics": buffer.getvalue(), "response": client.responses[-1]}

    started_at = time.perf_counter()
    got = call_with_retries(ctx, attempt, attempts)
    meta = getattr(got["response"], "usage_metadata", None) if got["response"] is not None else None

    def tokens(name: str) -> int:
        return int(getattr(meta, name, 0) or 0) if meta is not None else 0

    candidates, thoughts = tokens("candidates_token_count"), tokens("thoughts_token_count")
    record = base_record(ctx, cond, k, command="app.ask.ask(<質問>, [], client=<記録用の包み>)")
    record.update({
        "finished_at": now_iso(), "duration_s": round(got["duration"], 3), "elapsed_s": round(time.perf_counter() - started_at, 3),
        "status": got["status"], "error": got["error"], "attempts": attempts, "answer": got["answer"],
        "tool_calls": [], "approvals": [], "saved_files": [], "outside_files": [],
        "usage": {
            "requests": 1, "tool_calls": 0, "input_tokens": tokens("prompt_token_count"),
            # pydantic-ai の Google 版と同じ数え方（回答 + 思考）にそろえる
            "output_tokens": candidates + thoughts, "candidates_tokens": candidates, "thoughts_tokens": thoughts,
            "total_tokens": tokens("total_token_count"),
        },
        "metrics_output": got["metrics"],
    })
    trace = f"[model] {ctx.model_for('ask')}\n[質問] {cond.text}\n{got['metrics']}{got['answer']}\n"
    return TrialOutput(record, trace)


# ---------------------------------------------------------------------------
# エンジン 3・4: adk と pi（外部コマンドを起動して、JSONL を解析する）
# ---------------------------------------------------------------------------


def json_lines(text: str) -> tuple[list[dict], list[str]]:
    """JSONL を読む。JSON でない行（ADK の "Log setup complete: …" など）は飛ばす。"""
    events, skipped = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                events.append(obj)
                continue
        skipped.append(line)
    return events, skipped


def pick(data, *keys, default=None):
    """camelCase と snake_case が混ざったキーから、最初に見つかった値を返す。"""
    if isinstance(data, dict):
        for key in keys:
            if data.get(key) is not None:
                return data[key]
    return default


def response_text(response) -> str:
    """functionResponse.response を、戻り値の文字列にする（{"result": "…"} なら中身）。"""
    if isinstance(response, dict) and set(response) == {"result"}:
        return as_text(response["result"])
    return as_text(response)


def parse_adk_jsonl(text: str) -> dict:
    """adk run --jsonl の標準出力を解析する。

    各行は public_event(event).model_dump(mode='json', by_alias=True, exclude_none=True) に
    session_id・node_path を足したもの（キーは camelCase が基本で、snake_case が混ざる）。
    """
    events, skipped = json_lines(text)
    tool_calls, by_id, errors = [], {}, []
    answer, requests = "", 0
    tokens = Counter()
    for event in events:
        code, message = pick(event, "errorCode", "error_code"), pick(event, "errorMessage", "error_message")
        if code or message:
            errors.append({"code": code, "message": message})
        usage = pick(event, "usageMetadata", "usage_metadata")
        if isinstance(usage, dict):
            requests += 1
            tokens["input"] += pick(usage, "promptTokenCount", "prompt_token_count", default=0)
            tokens["candidates"] += pick(usage, "candidatesTokenCount", "candidates_token_count", default=0)
            tokens["thoughts"] += pick(usage, "thoughtsTokenCount", "thoughts_token_count", default=0)
            tokens["total"] += pick(usage, "totalTokenCount", "total_token_count", default=0)
        content = event.get("content") if isinstance(event.get("content"), dict) else {}
        texts = []
        for part in content.get("parts") or []:
            call = pick(part, "functionCall", "function_call")
            reply = pick(part, "functionResponse", "function_response")
            if isinstance(call, dict):
                entry = {
                    "order": len(tool_calls) + 1, "response_index": requests, "tool_call_id": call.get("id"),
                    "name": call.get("name"), "args": call.get("args") or {}, "result": None, "outcome": "pending",
                }
                tool_calls.append(entry)
                if entry["tool_call_id"]:
                    by_id[entry["tool_call_id"]] = entry
            elif isinstance(reply, dict):
                entry = by_id.get(reply.get("id")) or next((c for c in tool_calls if c["name"] == reply.get("name") and c["result"] is None), None)
                if entry is not None:
                    entry["result"], entry["outcome"] = response_text(reply.get("response")), "success"
            elif isinstance(part.get("text"), str) and not part.get("thought"):
                texts.append(part["text"])
        if texts and content.get("role") == "model":
            answer = "".join(texts)
    usage = {
        "requests": requests, "tool_calls": len(tool_calls), "input_tokens": tokens["input"],
        # pydantic-ai の Google 版と同じ数え方（回答 + 思考）にそろえる
        "output_tokens": tokens["candidates"] + tokens["thoughts"], "candidates_tokens": tokens["candidates"],
        "thoughts_tokens": tokens["thoughts"], "total_tokens": tokens["total"],
    }
    return {"answer": answer, "tool_calls": tool_calls, "usage": usage, "errors": errors, "events": len(events), "skipped_lines": skipped}


def blocks_text(content) -> str:
    """Pi のメッセージの content（文字列か、{"type":"text","text":…} のブロックの配列）から本文を取り出す。"""
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content or [] if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))


def parse_pi_jsonl(text: str) -> dict:
    """pi --mode json の標準出力を解析する（Pi v1.5.0 の docs/json.md）。"""
    events, skipped = json_lines(text)
    header, turns, answer = None, 0, ""
    tool_calls, by_id, retries, errors = [], {}, [], []
    tokens = Counter()
    for event in events:
        kind = event.get("type")
        if kind == "session" and header is None:
            header = event
        elif kind == "turn_start":
            turns += 1
        elif kind == "tool_execution_start":
            entry = {
                "order": len(tool_calls) + 1, "response_index": turns, "tool_call_id": event.get("toolCallId"),
                "name": event.get("toolName"), "args": event.get("args") or {}, "result": None, "outcome": "pending",
            }
            tool_calls.append(entry)
            by_id[entry["tool_call_id"]] = entry
        elif kind == "tool_execution_end":
            entry = by_id.get(event.get("toolCallId"))
            if entry is not None:
                result = event.get("result")
                entry["result"] = blocks_text(result.get("content")) if isinstance(result, dict) else as_text(result)
                entry["outcome"] = "failed" if event.get("isError") else "success"
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                answer = blocks_text(message.get("content"))
                usage = message.get("usage") or {}
                tokens["input"] += pick(usage, "input", default=0)
                tokens["output"] += pick(usage, "output", default=0)
                tokens["cache_read"] += pick(usage, "cacheRead", default=0)
                tokens["total"] += pick(usage, "totalTokens", default=0)
                if message.get("stopReason") in ("error", "aborted") or message.get("errorMessage"):
                    errors.append(str(message.get("errorMessage") or message.get("stopReason")))
        elif kind == "auto_retry_start":
            retries.append({k: event.get(k) for k in ("attempt", "maxAttempts", "delayMs", "errorMessage")})
        elif kind == "auto_retry_end" and event.get("success") is False:
            errors.append(str(event.get("finalError") or "auto retry failed"))
    usage = {
        "requests": turns, "tool_calls": len(tool_calls), "input_tokens": tokens["input"], "output_tokens": tokens["output"],
        "cache_read_tokens": tokens["cache_read"], "total_tokens": tokens["total"],
    }
    return {"answer": answer, "tool_calls": tool_calls, "usage": usage, "errors": errors, "auto_retries": retries,
            "session": header, "events": len(events), "skipped_lines": skipped}


def adk_command(question: str) -> list[str]:
    return ["uv", "run", "--with", ADK_REQUIREMENT, "adk", "run", "--in_memory", "--jsonl", "--timeout", "180s",
            "adk_agent/glossary_agent", question]


def pi_command(question: str, pi_model: str) -> list[str]:
    return ["pi", "--mode", "json", "--no-session", "--no-context-files", "--tools", "read,grep,find,ls",
            "--provider", "google", "--model", pi_model, question]


def build_adk_sample() -> str:
    """ADK 2.13.0 の _print_event が出すのと同じ形の見本（予行用）。先頭の 2 行は JSON ではないログ。"""
    sid = "00000000-0000-0000-0000-000000000000"
    lines = [
        {"author": "glossary_agent", "session_id": sid, "id": "e1", "content": {"parts": [
            {"functionCall": {"id": "adk-1", "args": {"query": "LoRA"}, "name": "search_glossary"}},
            {"functionCall": {"id": "adk-2", "args": {"query": "QLoRA"}, "name": "search_glossary"}}], "role": "model"},
         "usageMetadata": {"candidatesTokenCount": 30, "promptTokenCount": 412, "thoughtsTokenCount": 25, "totalTokenCount": 467},
         "invocationId": "e-1", "nodeInfo": {"path": ""}, "timestamp": 1.0},
        {"author": "glossary_agent", "session_id": sid, "id": "e2", "content": {"parts": [
            {"functionResponse": {"id": "adk-1", "name": "search_glossary", "response": {"result": "[{\"term\": \"LoRA\", \"lecture\": 2}]"}}},
            {"functionResponse": {"id": "adk-2", "name": "search_glossary", "response": {"result": "[{\"term\": \"QLoRA\", \"lecture\": 2}]"}}}], "role": "user"},
         "invocationId": "e-1", "nodeInfo": {"path": ""}, "timestamp": 2.0},
        {"author": "glossary_agent", "session_id": sid, "id": "e3", "content": {"parts": [
            {"text": "考えています", "thought": True},
            {"text": "（予行）LoRAは低ランク行列だけを学習する手法で、QLoRAは4ビットに量子化してLoRAで学習する手法です。\n\n第2回"}], "role": "model"},
         "finishReason": "STOP",
         "usageMetadata": {"candidatesTokenCount": 60, "promptTokenCount": 980, "thoughtsTokenCount": 40, "totalTokenCount": 1080},
         "invocationId": "e-1", "nodeInfo": {"path": ""}, "timestamp": 3.0},
    ]
    header = "Log setup complete: /tmp/agents_log/agent.sample.log\nTo access latest log: tail -F /tmp/agents_log/agent.latest.log\n"
    return header + "\n".join(json.dumps(e, ensure_ascii=False) for e in lines) + "\n"


def build_pi_sample() -> str:
    """Pi v1.5.0 の docs/json.md に沿った見本（予行用）。"""
    usage1 = {"input": 1200, "output": 40, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 1240}
    usage2 = {"input": 1500, "output": 80, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 1580}
    call = {"pattern": "LoRA", "path": "data"}
    events = [
        {"type": "session", "version": 3, "id": "dry-run", "timestamp": "2026-11-03T00:00:00.000Z", "cwd": "/tmp/pi-work"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": "質問", "timestamp": 1}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "toolCall", "id": "call_1", "name": "grep", "arguments": call}],
                                              "usage": usage1, "stopReason": "toolUse"}},
        {"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "grep", "args": call},
        {"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "grep",
         "result": {"content": [{"type": "text", "text": "data/glossary.json: \"term\": \"LoRA\""}], "details": {}}, "isError": False},
        {"type": "turn_end", "message": {"role": "assistant"}, "toolResults": []},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "（予行）Pi の回答です。"}],
                                              "usage": usage2, "stopReason": "stop"}},
        {"type": "turn_end", "message": {"role": "assistant"}, "toolResults": []},
        {"type": "agent_end", "messages": [], "willRetry": False},
        {"type": "agent_settled", "aborted": False},
    ]
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n"


ADK_SAMPLE_JSONL = build_adk_sample()
PI_SAMPLE_JSONL = build_pi_sample()


def run_subprocess(cmd: list[str], *, cwd: Path, env: dict) -> tuple[str, str, int]:
    """外部コマンドを起動する。タイムアウトは、再試行できる失敗（ApiFailure）にする。"""
    try:
        # stdin は空にする（端末を受け継ぐと、入力を待って止まったり、Pi が入力を質問に足したりするため）
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=SUBPROCESS_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise ApiFailure(None, f"timed out after {SUBPROCESS_TIMEOUT}s: {cmd[0]}", transport=True) from e
    return proc.stdout, proc.stderr, proc.returncode


def run_external_trial(ctx: Context, cond: Condition, k: int, *, engine: str) -> TrialOutput:
    """adk と pi に共通の流れ: 起動 → 解析 → 失敗なら分類して再試行 → 記録。"""
    if engine == "adk":
        cmd, parse, sample = adk_command(cond.text), parse_adk_jsonl, ADK_SAMPLE_JSONL
        notes = ["make_quiz の乱数は別プロセスなので、random.seed では揃わない"]
    else:
        cmd, parse, sample = pi_command(cond.text, ctx.pi_model), parse_pi_jsonl, PI_SAMPLE_JSONL
        notes = ["Pi は読み取り専用のツール（read,grep,find,ls）だけを持つ。cwd は data/ だけを複製した一時フォルダ"]
    attempts: list = []

    def attempt() -> dict:
        started = time.perf_counter()
        if ctx.dry_run:
            stdout, stderr, code = sample, "", 0
        elif engine == "adk":
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            stdout, stderr, code = run_subprocess(cmd, cwd=REPO, env=env)
        else:
            with tempfile.TemporaryDirectory(prefix="compare-pi-") as work:
                shutil.copytree(REPO / "data", Path(work) / "data")  # 専用のエージェントと同じ知識だけを見せる
                stdout, stderr, code = run_subprocess(cmd, cwd=Path(work), env=dict(os.environ))
        parsed = parse(stdout)
        errors = [f"{e.get('code')}: {e.get('message')}" if isinstance(e, dict) else str(e) for e in parsed["errors"]]
        if (code not in (0, 2) or errors) and not parsed["answer"]:  # 回答が無く、失敗している
            raise failure_from_text("\n".join(filter(None, [stderr.strip(), *errors])) or f"exit code {code}")
        return {"stdout": stdout, "stderr": stderr, "code": code, "parsed": parsed, "duration": time.perf_counter() - started}

    started_at = time.perf_counter()
    got = call_with_retries(ctx, attempt, attempts)
    parsed = got["parsed"]
    status = "paused" if got["code"] == 2 else ("ok" if parsed["answer"] else "no_answer")
    stem_name = f"t{k}"
    record = base_record(ctx, cond, k, command=cmd)
    record.update({
        "finished_at": now_iso(), "duration_s": round(got["duration"], 3), "elapsed_s": round(time.perf_counter() - started_at, 3),
        "status": status, "error": None, "attempts": attempts, "exit_code": got["code"], "answer": parsed["answer"],
        "tool_calls": parsed["tool_calls"], "approvals": [], "saved_files": [], "outside_files": [], "usage": parsed["usage"],
        "events": parsed["events"], "skipped_lines": parsed["skipped_lines"], "notes": notes,
        "raw_file": f"{stem_name}.raw.jsonl", "stderr_file": f"{stem_name}.stderr.txt",
    })
    if engine == "pi":
        record.update({"pi_version": ctx.pi_version, "auto_retries": parsed["auto_retries"], "session": parsed["session"]})
    lines = [f"[model] {ctx.model_for(engine)}", f"[質問] {cond.text}", f"[command] {' '.join(cmd[:-1])} <質問>", parsed["answer"], "", "--- trace ---"]
    for c in parsed["tool_calls"]:
        lines.append(f"[tool call]   {c['name']}({json.dumps(c['args'], ensure_ascii=False)})")
        lines.append(f"[tool return] {c['name']} -> {str(c['result'])[:120]}…")
    lines.append(f"[usage] {parsed['usage']}")
    return TrialOutput(record, "\n".join(lines) + "\n", raw=got["stdout"], stderr=got["stderr"])


def run_adk_trial(ctx: Context, cond: Condition, k: int) -> TrialOutput:
    return run_external_trial(ctx, cond, k, engine="adk")


def run_pi_trial(ctx: Context, cond: Condition, k: int) -> TrialOutput:
    return run_external_trial(ctx, cond, k, engine="pi")


RUNNERS = {"pydantic": run_pydantic_trial, "ask": run_ask_trial, "adk": run_adk_trial, "pi": run_pi_trial}


# ---------------------------------------------------------------------------
# run: 試行番号を外側に回して、全条件を実行する
# ---------------------------------------------------------------------------


def csv_list(values) -> list[str] | None:
    """--engine a b や --engine a,b のどちらの書き方でも、リストにする。"""
    if not values:
        return None
    return [item for value in values for item in value.split(",") if item]


def check_filters(cfg: dict, args) -> None:
    """--engine・--variant・--question の値が、知っているものか確かめる（打ち間違いに気づけるように）。"""
    known = {
        "engine": ENGINES,
        "variant": tuple(cfg["variants"]),
        "question": (*cfg["questions"], *cfg.get("scenarios", {})),
    }
    for option, choices in known.items():
        for value in csv_list(getattr(args, option)) or []:
            if value not in choices:
                raise ConfigError(f"--{option} {value!r} は知りません（使えるもの: {', '.join(choices)}）")


def make_plan(cfg: dict, args) -> list[Condition]:
    check_filters(cfg, args)
    levels = LEVELS if args.level == "all" else (args.level,)
    return build_plan(
        cfg, levels=levels, engines=csv_list(args.engine), variants=csv_list(args.variant),
        questions=csv_list(args.question), include_optional=args.include_optional,
    )


def resolve_out(args, dry_run: bool) -> Path:
    if args.out:
        return Path(args.out)
    if dry_run:
        return Path(tempfile.mkdtemp(prefix="compare-dry-run-"))
    return DEFAULT_OUT


def default_model() -> str:
    from app.config import MODEL

    return MODEL


def make_context(args, cfg: dict, *, dry_run: bool = False, out: Path | None = None, argv=None) -> Context:
    model = default_model()
    return Context(
        cfg=cfg, out=out if out is not None else resolve_out(args, dry_run), model=model,
        pi_model=getattr(args, "pi_model", None) or model, dry_run=dry_run,
        interval=0.0 if dry_run else float(getattr(args, "interval", 0) or 0),
        argv=list(argv) if argv is not None else [], git=git_state(),
    )


def existing_trials(ctx: Context, cond: Condition) -> list[Path]:
    """その条件の、済みの試行の t<k>.json。"""
    folder = trial_stem(ctx, cond, 1).parent
    return sorted(p for p in folder.glob("t*.json") if re.fullmatch(r"t\d+\.json", p.name)) if folder.is_dir() else []


def preflight(ctx: Context, plan: list[Condition], args) -> None:
    """API を呼ぶ前に、止めるべきことを確かめる。"""
    if not ctx.dry_run and not args.allow_dirty and ctx.git["dirty"]:
        files = "\n".join(f"  {line}" for line in ctx.git["dirty_files"])
        untracked = ""
        if any(line.startswith("??") for line in ctx.git["dirty_files"]):
            untracked = "\n?? は git が追跡していないファイル（実行で作られたものなど）。中身を確かめて、不要なら消してください。"
        raise StopRun(
            1,
            "app/・adk_agent/・compare.py・conditions.toml に、未コミットの変更があります。\n"
            f"{files}{untracked}\n結果がどの commit のものかを確かにするため、commit してから実行してください"
            "（確認を省くときは --allow-dirty）。",
        )
    for variant in sorted({c.variant for c in plan if c.engine == "pydantic"}):
        ctx.effective(variant)  # 変種の当て方が誤っていれば、ここで ConfigError になる
    for cond in plan:
        sha = ctx.sha_for(cond)
        for path in existing_trials(ctx, cond):
            old = json.loads(path.read_text(encoding="utf-8")).get("variant_sha256")
            if old != sha:
                raise StopRun(
                    1,
                    f"{path} の変種の sha256 が、いまのコードと違います（{cond.engine}/{cond.variant}/{cond.qid}）。\n"
                    "途中で app/・conditions.toml・adk_agent/ が変わった可能性があります。\n"
                    "取り直すなら、その条件の結果を手で退避してから、もう一度実行してください。",
                )
    if ctx.dry_run:
        return
    engines = {c.engine for c in plan}
    if not os.environ.get("GEMINI_API_KEY"):
        raise StopRun(1, "環境変数 GEMINI_API_KEY が見つかりません（Codespaces secret に登録して、Codespaces を再起動してください）。")
    if "adk" in engines and shutil.which("uv") is None:
        raise StopRun(1, "uv が見つかりません（adk は uv run --with google-adk で動かします）。")
    if "pi" in engines:
        if shutil.which("pi") is None:
            raise StopRun(
                1,
                "pi が見つかりません。Codespaces のターミナルで、次を実行してください:\n"
                "  npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.5.0",
            )
        ctx.pi_version = pi_version()


def pi_version() -> str | None:
    """pi --version の出力（記録用。取れなくても止めない）。"""
    try:
        proc = subprocess.run(["pi", "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or proc.stderr).strip()[:100] or None


def cmd_run(args) -> int:
    cfg = load_config(args.conditions)
    plan = make_plan(cfg, args)
    if not plan:
        print("条件に一致する実行がありません（--level / --engine / --variant / --question を確かめてください）")
        return 1
    ctx = make_context(args, cfg, dry_run=args.dry_run, argv=args._argv)
    preflight(ctx, plan, args)

    trials = args.trials
    total = len(plan) * trials
    todo = [(k, c) for k in range(1, trials + 1) for c in plan if not trial_done(ctx, c, k)]
    mode = "予行（--dry-run）" if ctx.dry_run else "本番"
    print(f"{mode}: 条件 {len(plan)} 件 × 試行 {trials} 回 = {total} 回のうち、済み {total - len(todo)} 回、これから {len(todo)} 回")
    executed = 0
    for k, cond in todo:
        if args.max_runs is not None and executed >= args.max_runs:
            print(f"--max-runs {args.max_runs} に達したので止めます。残り {len(todo) - executed} 回は、同じコマンドで続きから実行できます。")
            break
        if executed and ctx.interval > 0:
            time.sleep(ctx.interval)
        if not ctx.dry_run:
            print(f"  → {cond.engine}/{cond.variant}/{cond.qid} t{k} を実行します", flush=True)
        output = RUNNERS[cond.engine](ctx, cond, k)
        save_trial(trial_stem(ctx, cond, k), output)
        executed += 1
        r = output.record
        usage = r.get("usage") or {}
        print(
            f"[{total - len(todo) + executed}/{total}] {cond.engine}/{cond.variant}/{cond.qid} t{k}: {r['status']} "
            f"{r['duration_s']:.1f}s tool_calls={usage.get('tool_calls')} requests={usage.get('requests')}",
            flush=True,
        )
    print(f"\n実行 {executed} 回。保存先: {ctx.out}")
    if ctx.dry_run:
        print("（予行の結果です。続けて集計するには）\n  uv run python lectures/03/compare.py summarize --out " + str(ctx.out))
    return 0


# ---------------------------------------------------------------------------
# チェック: 集計のときに判定する（種類は CHECKS に足せる）
# ---------------------------------------------------------------------------


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


def check_ending_lecture(check: dict, rec: dict) -> bool:
    lines = [line for line in rec.get("answer", "").splitlines() if line.strip()]
    tail = "\n".join(lines[-2:])
    numbers = [int(n) for n in re.findall(r"第\s*(\d+)\s*回", tail)]
    found = bool(numbers) if "lecture" not in check else check["lecture"] in numbers
    return found == check.get("expect", True)


CHECKS = {
    "answer_contains_all": lambda c, r: all(v in r["answer"] for v in c["values"]),
    "answer_contains_any": lambda c, r: any(v in r["answer"] for v in c["values"]),
    "answer_not_contains": lambda c, r: not any(v in r["answer"] for v in c["values"]),
    "answer_regex": lambda c, r: re.search(c["pattern"], r["answer"], re.I) is not None,
    "answer_not_regex": lambda c, r: re.search(c["pattern"], r["answer"], re.I) is None,
    "ending_lecture": check_ending_lecture,
    "tool_called": check_tool_called,
    "tool_not_called": lambda c, r: not calls_named(r, c["name"]),
    "tool_count": lambda c, r: in_range(len(calls_named(r, c["name"])), c),
    "tool_result_contains": lambda c, r: all(v in "\n".join(str(x.get("result") or "") for x in calls_named(r, c["name"])) for v in c["values"]),
    "requests_at_most": lambda c, r: (r.get("usage") or {}).get("requests", 10**9) <= c["value"],
    "approvals_count": lambda c, r: in_range(len(r.get("approvals", [])), c),
    "files_saved": lambda c, r: in_range(len(r.get("saved_files", [])), c),
    "files_outside_none": lambda c, r: not r.get("outside_files"),
}


def check_label(check: dict) -> str:
    if check.get("label"):
        return check["label"]
    params = ", ".join(f"{k}={v}" for k, v in check.items() if k not in ("kind", "engines", "label"))
    return f"{check.get('kind')}({params})"


def evaluate_check(check: dict, rec: dict) -> bool | None:
    """チェックを判定する。True/False。この engine には当てはまらないチェックは None。"""
    engines = check.get("engines")
    if engines and rec.get("engine") not in engines:
        return None
    rec = {**rec, "answer": rec.get("answer") or ""}
    return bool(CHECKS[check["kind"]](check, rec))


# ---------------------------------------------------------------------------
# summarize: 表 A（ツールの呼び方）・表 B（チェック）・表 C（トークンと秒）
# ---------------------------------------------------------------------------


def tool_signature(rec: dict) -> str:
    """ツールを呼んだ順を、連続する同じ名前をまとめて書く。例: search_glossary×2→make_quiz"""
    names = [c.get("name") for c in rec.get("tool_calls", [])]
    if not names:
        return "ツールなし"
    parts = []
    for name, group in itertools.groupby(names):
        count = len(list(group))
        parts.append(f"{name}×{count}" if count > 1 else name)
    return "→".join(parts)


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


def cell_text(value: str) -> str:
    return value.replace("|", "｜").replace("\n", " ")


def reparse(rec: dict, json_file: Path) -> dict:
    """adk と pi は、raw.jsonl から解析し直す（解析を直したら、summarize だけで反映される）。"""
    raw = sibling(json_file.with_suffix(""), ".raw.jsonl")
    if rec.get("engine") in ("adk", "pi") and raw.exists():
        parsed = (parse_adk_jsonl if rec["engine"] == "adk" else parse_pi_jsonl)(raw.read_text(encoding="utf-8"))
        status = "paused" if rec.get("exit_code") == 2 else ("ok" if parsed["answer"] else "no_answer")
        rec = {**rec, "answer": parsed["answer"], "tool_calls": parsed["tool_calls"], "usage": parsed["usage"], "status": status}
    return rec


def load_model_records(model_dir: Path) -> list[dict]:
    records = []
    for path in sorted(model_dir.glob("*/*/*/t*.json")):
        if not re.fullmatch(r"t\d+\.json", path.name):
            continue
        records.append(reparse(json.loads(path.read_text(encoding="utf-8")), path))
    return records


def discover_models(out: Path) -> list[Path]:
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if p.is_dir() and p.name not in ("variants", "cli") and any((p / e).is_dir() for e in ENGINES))


@dataclass
class Row:
    engine: str
    variant: str
    label: str
    qids: list
    reference: bool = False


def level_rows(cfg: dict, level: str, have: set) -> list[Row]:
    """表の行。任意の変種（V6・V7）は、結果があるときだけ出す。発展には、比べやすいよう V0 の行を足す。"""
    rows = []
    for engine, variant, ids, optional in level_runs(cfg, level, include_optional=True):
        if optional and not any((engine, variant, q) in have for q in ids):
            continue
        if level == "basic" or level == "impl":
            label = f"{cfg['variants'][variant].get('label', '')} {variant}".strip()
        else:
            label = f"{engine}/{variant}"
        rows.append(Row(engine, variant, label, ids))
    if level in ("adv-a", "adv-b") and rows:
        ids = [q for q in rows[0].qids if ("pydantic", "baseline", q) in have]
        if ids:
            rows.insert(0, Row("pydantic", "baseline", "(参考) V0 baseline", ids, reference=True))
    return rows


def trial_metrics(rec: dict) -> dict:
    usage = rec.get("usage") or {}
    return {
        "tc": usage.get("tool_calls"), "req": usage.get("requests"), "in": usage.get("input_tokens"),
        "out": usage.get("output_tokens"), "sec": rec.get("duration_s"),
    }


def summarize_model(model: str, records: list[dict], cfg: dict, expected: int, levels) -> tuple[list[str], dict]:
    index: dict = defaultdict(list)
    for rec in records:
        index[(rec["engine"], rec["variant"], rec["qid"])].append(rec)
    for trials in index.values():
        trials.sort(key=lambda r: r["trial"])
    have = set(index)
    md: list[str] = [f"## モデル: {model}", ""]
    data: dict = {}
    for level in LEVELS:
        if level not in levels:
            continue
        rows = level_rows(cfg, level, have)
        if not any((r.engine, r.variant, q) in have for r in rows for q in r.qids):
            continue
        qids = sorted({q for r in rows for q in r.qids})
        md += [f"### レベル: {level}", "", "質問: " + " / ".join(f"**{q}** {(cfg.get('scenarios', {}).get(q) or cfg['questions'][q])['text']}" for q in qids), ""]

        def cell(row: Row, qid: str):
            return index.get((row.engine, row.variant, qid), []) if qid in row.qids else None

        # 表 A
        md += ["#### 表A: ツールの呼び方（呼んだ順と回数 (その回数だった試行/試行数)｜tc = tool_calls の各回｜req = requests の各回）", "",
               "| 行 | " + " | ".join(qids) + " |", "|---|" + "---|" * len(qids)]
        table_a: dict = {}
        for row in rows:
            cells = []
            for qid in qids:
                trials = cell(row, qid)
                if trials is None:
                    cells.append("—")
                    continue
                if not trials:
                    cells.append("未実行")
                    continue
                groups = Counter(tool_signature(t) for t in trials).most_common()
                text = " / ".join(f"{sig} ({n}/{len(trials)})" for sig, n in groups)
                metrics = [trial_metrics(t) for t in trials]
                text += f"｜tc {','.join(str(m['tc']) for m in metrics)}｜req {','.join(str(m['req']) for m in metrics)}"
                if level == "impl":
                    text += f"｜承認 {','.join(str(len(t.get('approvals', []))) for t in trials)}｜保存 {','.join(str(len(t.get('saved_files', []))) for t in trials)}"
                bad = Counter(t["status"] for t in trials if t.get("status") != "ok")
                if bad:
                    text += " ⚠" + ",".join(f"{s}×{n}" for s, n in bad.items())
                if len(trials) < expected:
                    text += f" [{len(trials)}/{expected}回]"
                cells.append(cell_text(text))
                table_a.setdefault(row.label, {})[qid] = {
                    "trials": len(trials), "signatures": dict(groups), "tool_calls": [m["tc"] for m in metrics],
                    "requests": [m["req"] for m in metrics],
                }
            md.append(f"| {cell_text(row.label)} | " + " | ".join(cells) + " |")
        md.append("")

        # 表 B
        labels: dict = {}
        for qid in qids:
            for check in checks_for(cfg, qid):
                labels[(qid, check_label(check))] = check
        md += ["#### 表B: 自動チェックの合格数（合格した試行/判定した試行）", "", "| 質問 | チェック | " + " | ".join(cell_text(r.label) for r in rows) + " |",
               "|---|---|" + "---|" * len(rows)]
        table_b: dict = {}
        for (qid, label), check in labels.items():
            cells = []
            for row in rows:
                trials = cell(row, qid)
                if trials is None:
                    cells.append("—")
                    continue
                results = [r for r in (evaluate_check(check, t) for t in trials) if r is not None]
                if not trials:
                    cells.append("未実行")
                elif not results:
                    cells.append("対象外")
                else:
                    cells.append(f"{sum(results)}/{len(results)}")
                    table_b.setdefault(row.label, {}).setdefault(qid, {})[label] = {"passed": sum(results), "total": len(results)}
            md.append(f"| {qid} | {cell_text(label)} | " + " | ".join(cells) + " |")
        md.append("")

        # 表 C
        md += ["#### 表C: トークンと秒（中央値 (最小–最大)。入力・出力のトークンは、その質問の全試行）", "",
               "| 行 | 質問 | 試行数 | 入力トークン | 出力トークン | 秒 | requests |", "|---|---|---|---|---|---|---|"]
        table_c: dict = {}
        for row in rows:
            for qid in qids:
                trials = cell(row, qid)
                if not trials:
                    continue
                metrics = [trial_metrics(t) for t in trials]
                values = {k: med_range([m[k] for m in metrics], 1 if k == "sec" else 0) for k in ("in", "out", "sec", "req")}
                md.append(f"| {cell_text(row.label)} | {qid} | {len(trials)} | {values['in']} | {values['out']} | {values['sec']} | {values['req']} |")
                table_c.setdefault(row.label, {})[qid] = values
        md.append("")

        # 回答の全文
        md += ["#### 回答の全文とツールの引数", ""]
        for row in rows:
            for qid in qids:
                trials = cell(row, qid)
                if not trials:
                    continue
                md += ["<details>", f"<summary>{cell_text(row.label)} / {qid}（{len(trials)} 回）</summary>", ""]
                for t in trials:
                    m = trial_metrics(t)
                    results = [(check_label(c), evaluate_check(c, t)) for c in checks_for(cfg, qid)]
                    marks = " ".join(f"{'○' if ok else '×'}{label}" for label, ok in results if ok is not None)
                    md += [f"**t{t['trial']}**｜{t.get('status')}｜{m['sec']} 秒｜tool_calls {m['tc']}｜requests {m['req']}｜{marks}", ""]
                    md += [fenced(t.get("answer") or "(回答なし)"), ""]
                    for c in t.get("tool_calls", []):
                        md.append(f"- `{c['name']}({args_json(c.get('args'))})` → {c.get('outcome')}")
                    for a in t.get("approvals", []):
                        md.append(f"- 承認待ち `{a['tool_name']}` → **{a['decision']}**")
                    for f in t.get("saved_files", []):
                        md.append(f"- 保存されたファイル `{f['name']}`")
                    md.append("")
                md += ["</details>", ""]
        data[level] = {"rows": [dataclasses.asdict(r) for r in rows], "questions": qids, "table_a": table_a, "table_b": table_b, "table_c": table_c}
    return md, data


def cmd_summarize(args) -> int:
    cfg = load_config(args.conditions)
    out = Path(args.out) if args.out else DEFAULT_OUT
    levels = LEVELS if args.level == "all" else (args.level,)
    models = discover_models(out)
    if not models:
        print(f"{out} に、集計できる結果がありません（<model>/<engine>/<variant>/<qid>/t<k>.json）。")
        return 1
    md = [
        "# 比較の集計（compare.py summarize が作った。手で直さない）", "",
        f"結果の置き場所: `{out}`　試行の予定: {args.trials} 回", "",
        "読み方: 3 回の値の範囲が V0 と重ならなければ「変わった」、2 回分の差なら「傾向」、1 回分以下なら「揺れ」と書く。"
        "3/3 と 0/3 の差でも、Fisher の正確検定では p=0.1 なので、「有意」とは書かない。", "",
    ]
    data: dict = {"out": str(out), "expected_trials": args.trials, "models": {}}
    for model_dir in models:
        lines, model_data = summarize_model(model_dir.name, load_model_records(model_dir), cfg, args.trials, levels)
        md += lines
        data["models"][model_dir.name] = model_data
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "summary.md", "\n".join(md) + "\n")
    write_json_atomic(out / "summary.json", data)
    print(f"集計しました: {out / 'summary.md'}（と summary.json）")
    return 0


# ---------------------------------------------------------------------------
# status / variants
# ---------------------------------------------------------------------------


def cmd_status(args) -> int:
    cfg = load_config(args.conditions)
    ctx = make_context(args, cfg, out=Path(args.out) if args.out else DEFAULT_OUT)
    levels = LEVELS if args.level == "all" else (args.level,)
    rows = [("レベル", "エンジン/変種", "条件", "予定", "済み", "残り", "残りの推定リクエスト")]
    for level in levels:
        groups: dict = defaultdict(list)
        for cond in make_plan_for_level(cfg, args, level):
            groups[(cond.engine, cond.variant)].append(cond)
        for (engine, variant), conds in groups.items():
            planned = len(conds) * args.trials
            done = sum(trial_done(ctx, c, k) for c in conds for k in range(1, args.trials + 1))
            per = ESTIMATED_REQUESTS["impl" if variant == "impl" else engine]
            rows.append((level, f"{engine}/{variant}", str(len(conds)), str(planned), str(done), str(planned - done), f"{(planned - done) * per:.0f}"))
    unique = make_plan(cfg, args)
    planned = len(unique) * args.trials
    done = sum(trial_done(ctx, c, k) for c in unique for k in range(1, args.trials + 1))
    estimate = sum(ESTIMATED_REQUESTS["impl" if c.variant == "impl" else c.engine] for c in unique for k in range(1, args.trials + 1) if not trial_done(ctx, c, k))
    rows.append(("合計（重複を除く）", "", str(len(unique)), str(planned), str(done), str(planned - done), f"{estimate:.0f}"))
    print(f"結果の置き場所: {ctx.out}　モデル: {ctx.model}（pi: {ctx.pi_model}）　試行: {args.trials} 回")
    print("残りの推定リクエスト = 残りの実行回数 × 目安（pydantic・adk 2.5、ask 1、pi 6、impl 3.5）。無料枠の 1 日の上限は AI Studio の画面で確かめる")
    widths = [max(display_width(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(cell + " " * (width - display_width(cell)) for cell, width in zip(r, widths, strict=True)).rstrip())
    return 0


def display_width(text: str) -> int:
    """端末での表示幅（全角の文字は 2）。表の桁をそろえるため。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def make_plan_for_level(cfg: dict, args, level: str) -> list[Condition]:
    check_filters(cfg, args)
    return build_plan(
        cfg, levels=(level,), engines=csv_list(args.engine), variants=csv_list(args.variant),
        questions=csv_list(args.question), include_optional=args.include_optional,
    )


def cmd_variants(args) -> int:
    cfg = load_config(args.conditions)
    out = Path(args.out) if args.out else DEFAULT_OUT
    diffs = variant_diffs(cfg)
    warnings = 0
    for name, info in diffs.items():
        eff = info["effective"]
        base = out / "variants" / "pydantic" / name
        write_json_atomic(sibling(base, ".json"), {
            "variant": name, "label": cfg["variants"][name].get("label"), "sha256": eff.sha256,
            "changes": cfg["variants"][name].get("changes", []), "instructions": eff.instructions, "tools": eff.tool_defs,
        })
        write_text_atomic(sibling(base, ".txt"), info["text"])
        label = cfg["variants"][name].get("label", "")
        if name == "baseline":
            print(f"=== {label} {name}（基準）sha256 {eff.sha256[:12]} ===\n")
            continue
        print(f"=== {label} {name}: baseline との差分 ===")
        print("\n".join(info["diff"]) if info["diff"] else "(差分なし)")
        if (info["hunks"] != 1 or info["changes"] > 1) and name != "impl":
            warnings += 1
            print(f"⚠ 差分が {info['hunks']} か所、changes が {info['changes']} 件です（変種は 1 か所だけ変えるはず）")
        print()
    print(f"保存先: {out / 'variants' / 'pydantic'}（変種 {len(diffs)} 件）")
    return 1 if warnings else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", help="結果の置き場所（既定: lectures/03/results。--dry-run は一時フォルダ）")
    common.add_argument("--conditions", help="条件のファイル（既定: lectures/03/conditions.toml）")
    common.add_argument("--level", default="all", choices=("all", *LEVELS), help="exercise|basic|impl|adv-a|adv-b|all")
    common.add_argument("--engine", nargs="+", metavar="ENGINE", help="pydantic ask adk pi（絞り込み）")
    common.add_argument("--variant", nargs="+", metavar="VARIANT", help="変種（絞り込み）")
    common.add_argument("--question", nargs="+", metavar="ID", help="質問・シナリオの ID（絞り込み）。例: q01 s1")
    common.add_argument("--trials", type=int, default=3, help="試行の回数（既定 3）")
    common.add_argument("--include-optional", action="store_true", help="任意の変種（V6・V7）も含める")
    common.add_argument("--pi-model", help="Pi のモデル ID（既定: app.config.MODEL）")

    parser = argparse.ArgumentParser(description="条件を変えてエージェントの動きを比べる（実行・集計）")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status", parents=[common], help="予定・済み・残りの件数と、推定リクエスト数（API なし）")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("variants", parents=[common], help="変種ごとに、モデルに見える指示とツールを保存し、baseline との差分を表示")
    p.set_defaults(func=cmd_variants)
    p = sub.add_parser("run", parents=[common], help="実行する（既にある結果は飛ばす）")
    p.add_argument("--interval", type=float, default=8, help="実行と実行の間の秒数（既定 8）")
    p.add_argument("--max-runs", type=int, help="この実行で行う回数の上限")
    p.add_argument("--dry-run", action="store_true", help="API を呼ばずに予行する（TestModel・偽の client・見本の JSONL）")
    p.add_argument("--allow-dirty", action="store_true", help="未コミットの変更があっても実行する")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("summarize", parents=[common], help="summary.md と summary.json を作る")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("--trials は 1 以上にしてください")
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
