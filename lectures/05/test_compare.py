"""LLM も API キーもネットワークも使わない。実行：`uv run pytest -q lectures/05`（講座の `uv run pytest -q` とは別に実行する）

第5回の比較用スクリプト compare.py のテスト。条件ファイルの検証、条件の当て方（app.agent.main() の各モードとの一致）、
引用・ツールの戻り値の解析、再試行（429・503・400・埋め込み・ツールの戻り値の error）、全文投入の再現、
予行（--dry-run）と再開、実行前の確認、集計の表と自動チェックを確かめる。
モデルは FunctionModel、埋め込みは monkeypatch で偽物にする。compare.py の time.sleep は待たずに、秒数だけ記録する。
ネットワークと、この作業ツリーの git には触れない（conftest.py の no_network・hermetic_git。外に出ようとすると、待たずに例外になる）。
"""

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from pydantic_ai import Tool
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

import compare
from app import retrieval, tools
from app import retrieval_lec05 as lec05
from app.agent import agent
from app.config import MODEL, api_error_message

HERE = Path(__file__).resolve().parent
KEY = "test-key-not-a-real-key"
QA1 = "回答を決まったJSON形式で返させたいとき、第2回で学んだファインチューニングと第3回の構造化出力は、どう使い分ければよいですか"
SCOPE_214 = ["lec01-main", "lec02-main", "lec03-main"]
SCOPE_ALL = list(compare.page_index().titles)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """利用者の環境変数（検索 DB の差し替え）に左右されない。"""
    monkeypatch.delenv("AIE_RAG_DB", raising=False)


@pytest.fixture(autouse=True)
def empty_trace():
    """取得の内訳（TRACE）は、呼び出しごとに足されるので、テストごとに空にする。"""
    lec05.TRACE.clear()
    yield
    lec05.TRACE.clear()


def run(*argv):
    return compare.main([str(a) for a in argv])


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conditions_text() -> str:
    return compare.CONDITIONS_FILE.read_text(encoding="utf-8")


def write_conditions(tmp_path, text, name="conditions.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def toml_cfg() -> dict:
    import tomllib

    return tomllib.loads(conditions_text())


def real_run(out, *extra, variant="vector", question="q1", trials=1, level="all"):
    """API を呼ぶ側の経路（--dry-run なし）で、1 条件・1 質問を実行する。モデルは各テストで差し替える。"""
    return run("run", "--level", level, "--variant", variant, "--question", question, "--trials", trials,
               "--allow-dirty", "--interval", 0, "--out", out, *extra)


def result_path(out, variant="vector", qid="q1", trial=1, model=MODEL, suffix=".json"):
    return Path(out) / model / variant / qid / f"t{trial}{suffix}"


def use_model(monkeypatch, model):
    monkeypatch.setattr(compare, "make_model", lambda ctx, eff: model)


def text_model(text="LoRAは元の重みを固定し、低ランクの差分だけを学習します。\n\n根拠: 第2回・PDF p.52"):
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


def returned_tools(messages) -> list:
    return [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


def tool_then_answer(name="search_materials", args=None, answer="LoRAは重みを固定します。\n\n第2回・PDF p.52"):
    """1 回目にツールを呼び、戻り値を受け取ったら答えるモデル。呼ばれた回数は calls に数える。"""
    calls = []

    def fn(messages, info):
        calls.append(1)
        if not returned_tools(messages):
            return ModelResponse(parts=[ToolCallPart(name, args if args is not None else {"query": "LoRA QLoRA"})])
        return ModelResponse(parts=[TextPart(answer)])

    return FunctionModel(fn), calls


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


def stored_vector(page_id: str) -> list[float]:
    """検索 DB に保存された 1 ページの埋め込み。これを質問の埋め込みにすると、そのページが距離 0 で 1 位になる。"""
    with retrieval.open_index(retrieval.RAG_DB) as db:
        blob = db.execute("SELECT embedding FROM chunks WHERE page_id = ?", (page_id,)).fetchone()["embedding"]
    return memoryview(blob).cast("f").tolist()


@pytest.fixture
def embed_stored_page(monkeypatch):
    """質問の埋め込みを、lec02-main-p052 の保存済みの埋め込みにする（API を呼ばない）。呼び出しを記録して返す。"""
    vector = stored_vector("lec02-main-p052")
    calls = []

    def fake_embed(text, task_type):
        calls.append((text, task_type))
        return vector

    monkeypatch.setattr("app.retrieval.embed_text", fake_embed)
    return calls


def git(repo, *args):
    """一時リポジトリの git（cwd は一時リポジトリ。hook・署名は使わず、入力を待たず、止まったままにならない）。"""
    assert Path(repo).resolve() != compare.REPO.resolve()  # この作業ツリーでは動かさない
    assert args[0] == "init" or (Path(repo) / ".git").is_dir()  # git init した、そのフォルダでだけ動かす
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=60,
    )


# --- 条件ファイル（conditions.toml）の検証 ---


def test_the_conditions_file_in_the_repo_is_valid():
    cfg = compare.load_config()

    assert list(cfg["questions"]) == ["q1", "q2", "q3", "qa1", "qa2"]
    assert list(cfg["variants"]) == ["full", "full_aligned", "vector", "tree", "graph", "doc_hint", "neighbors", "topk15", "term_graph", "term_graph_cap3"]
    assert {name: spec["label"] for name, spec in cfg["variants"].items()} == {
        "full": "F", "full_aligned": "F214", "vector": "V0", "tree": "T", "graph": "G", "doc_hint": "B1", "neighbors": "I1",
        "topk15": "I2", "term_graph": "A1", "term_graph_cap3": "A2"}
    assert cfg["defaults"] == {"trials": 3, "interval": 8, "request_limit": 20, "tpm_budget": 200000, "order": cfg["defaults"]["order"], "order_shift": 3}
    assert {level: spec.get("reference", []) for level, spec in cfg["levels"].items()} == {
        "exercise": [], "basic": ["vector"], "impl": ["vector"], "adv": ["vector", "graph"]}


def test_q1_q2_q3_are_exactly_the_questions_in_problem_md():
    problem = (HERE / "problem.md").read_text(encoding="utf-8")
    cfg = compare.load_config()

    for qid in ("q1", "q2", "q3"):
        assert problem.count(cfg["questions"][qid]["text"]) == 1, qid
    assert cfg["questions"]["qa1"]["text"].startswith(QA1)


def test_gold_and_acceptable_pages_exist_in_lectures_json_and_do_not_overlap():
    cfg = compare.load_config()
    index = compare.page_index()

    assert len(index.page_ids) == 438 and len(compare.pages_in_scope(SCOPE_214)) == 214
    for qid, item in cfg["questions"].items():
        assert set(item["gold"]) <= index.page_ids and set(item["acceptable"]) <= index.page_ids, qid
        assert not set(item["gold"]) & set(item["acceptable"]), qid
    assert cfg["questions"]["q3"]["gold"] == [] and cfg["questions"]["q3"]["acceptable"] == []
    assert cfg["questions"]["q1"]["gold"] == ["lec02-main-p052"]
    assert cfg["questions"]["qa1"]["acceptable"] == ["lec02-main-p057", "lec03-main-p034", "lec03-main-p062"]


def replaced(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


BROKEN_CONDITIONS = [
    ("知らないキー（最上位）", lambda t: t + "\n[extra]\nx = 1\n", "知らないキー"),
    ("知らないキー（条件）", lambda t: replaced(t, 'label = "I2"', 'label = "I2"\nbogus = 1'), "知らないキー"),
    ("知らないキー（質問）", lambda t: replaced(t, 'gold = ["lec02-main-p052"]', 'gold = ["lec02-main-p052"]\nbogus = 1'), "知らないキー"),
    ("label の重なり", lambda t: replaced(t, 'label = "I2"', 'label = "I1"'), "重なっています"),
    ("label が別の条件の名前と同じ", lambda t: replaced(t, 'label = "I2"', 'label = "vector"'), "名前と同じ"),
    ("gold が lectures.json にない", lambda t: replaced(t, 'gold = ["lec02-main-p052"]', 'gold = ["lec02-main-p999"]'), "lectures.json にない"),
    ("acceptable が lectures.json にない", lambda t: replaced(t, '"lec02-exercise-p021"]', '"lec09-main-p001"]'), "lectures.json にない"),
    ("gold と acceptable の重なり", lambda t: replaced(t, 'acceptable = ["lec02-main-p051", "lec02-exercise-p021"]', 'acceptable = ["lec02-main-p052"]'), "同じページ"),
    ("order に条件が足りない", lambda t: replaced(t, ', "term_graph_cap3"]\n', ']\n'), "足りません"),
    ("order に知らない名前", lambda t: replaced(t, '"term_graph_cap3"]\n', '"term_graph_cap3", "nothing"]\n'), "にない名前"),
    ("full の隣り合わせ", lambda t: replaced(t, 'order = ["full", "vector", "tree", "full_aligned",', 'order = ["full", "full_aligned", "vector", "tree",'), "隣り合って"),
    ("先頭と末尾の full の隣り合わせ", lambda t: replaced(
        t, 'order = ["full", "vector", "tree", "full_aligned", "graph", "doc_hint", "neighbors", "topk15", "term_graph", "term_graph_cap3"]',
        'order = ["vector", "tree", "graph", "doc_hint", "neighbors", "topk15", "term_graph", "term_graph_cap3", "full", "full_aligned"]'), "隣り合って"),
    ("base の循環", lambda t: replaced(t, '[variants.vector]\nlabel = "V0"\nlevel = "exercise"\nmode = "vector"', '[variants.vector]\nlabel = "V0"\nlevel = "exercise"\nbase = "term_graph"'), "循環"),
    ("base が無い条件", lambda t: replaced(t, 'base = "term_graph"', 'base = "nothing"'), "variants にありません"),
    ("settings の名前", lambda t: replaced(t, "settings = { RAG_TOP_K = 15 }", "settings = { TOP_K = 15 }"), "書き換えられる設定ではありません"),
    ("settings の値", lambda t: replaced(t, "settings = { RAG_TOP_K = 15 }", 'settings = { RAG_TOP_K = "15" }'), "整数"),
    ("retrieval の形", lambda t: replaced(t, "app.retrieval_lec05:search_materials_with_neighbors", "search_materials_with_neighbors"), "module:attr"),
    ("retrieval の module が無い", lambda t: replaced(t, "app.retrieval_lec05:search_materials_with_neighbors", "app.no_such_module:f"), "読み込めません"),
    ("retrieval の関数が無い", lambda t: replaced(t, "app.retrieval_lec05:search_materials_with_neighbors", "app.retrieval_lec05:no_such_function"), "関数 no_such_function がありません"),
    ("retrieval のツールが無い", lambda t: replaced(t, "retrieval = { search_materials = \"app.retrieval_lec05:search_materials_with_neighbors\" }", "retrieval = { browse_tree = \"app.retrieval_lec05:search_materials_with_neighbors\" }"), "ツールに browse_tree がありません"),
    ("full に retrieval", lambda t: replaced(t, 'documents = "all"', 'documents = "all"\nretrieval = { search_materials = "app.retrieval_lec05:search_materials_term_graph" }'), "差し替えるツールがありません"),
    ("vector に documents", lambda t: replaced(t, 'description = "基本形。', 'documents = "all"\ndescription = "基本形。'), "mode = 'full'"),
    ("mode の綴り", lambda t: replaced(t, 'mode = "tree"', 'mode = "forest"'), "どれかです"),
    ("level の綴り", lambda t: replaced(t, 'level = "impl"\nbase = "vector"\ndescription = "ベクトル検索で返す件数', 'level = "implement"\nbase = "vector"\ndescription = "ベクトル検索で返す件数'), "どれかです"),
    ("質問が無い", lambda t: replaced(t, 'questions = ["q1", "q2", "q3"]\n\n[variants.full_aligned]', 'questions = ["q1", "q9"]\n\n[variants.full_aligned]'), "q9"),
    ("チェックの種類", lambda t: replaced(t, 'kind = "answer_not_regex"', 'kind = "answer_never"'), "知らないチェックの種類"),
    ("チェックの role", lambda t: replaced(t, 'role = "flag"', 'role = "warn"'), "role"),
    ("チェックの正規表現", lambda t: replaced(t, "pattern = '4-?(bit|ビット)|量子化(?!<<NEG_SURU>>)|quantiz'", "pattern = '4-?(bit'"), "正規表現"),
    ("order_shift が負", lambda t: replaced(t, "order_shift = 3", "order_shift = -1"), "defaults.order_shift"),
    ("order_shift が無い", lambda t: replaced(t, "order_shift = 3\n", ""), "defaults.order_shift がありません"),
    ("知らない部品", lambda t: replaced(t, "量子化(?!<<NEG_SURU>>)", "量子化(?!<<NEG_NOTHING>>)"), "知らない部品 <<NEG_NOTHING>>"),
    ("部品の循環", lambda t: replaced(t, "NA = 'な(い|かっ|く)'", "NA = '<<TE>>'"), "循環"),
    ("部品の名前", lambda t: replaced(t, "[fragments]\n", "[fragments]\nlowercase = 'x'\n"), "大文字"),
    ("部品が正規表現として読めない", lambda t: replaced(t, "NA = 'な(い|かっ|く)'", "NA = 'な(い|かっ'"), "正規表現として読めません"),
    ("部品が文字列でない", lambda t: replaced(t, "[fragments]\n", "[fragments]\nEMPTY = 3\n"), "空でない文字列"),
    ("チェックの label の重なり", lambda t: replaced(t, 'label = "q1_train"', 'label = "q1_fix"'), "label が重なっています"),
    ("trials", lambda t: replaced(t, "trials = 3", "trials = 0"), "defaults.trials"),
    ("reference が同じレベル", lambda t: replaced(t, 'reference = ["vector"]\n\n[levels.impl]', 'reference = ["doc_hint"]\n\n[levels.impl]'), "このレベルの条件"),
]


@pytest.mark.parametrize(("name", "edit", "message"), BROKEN_CONDITIONS, ids=[b[0] for b in BROKEN_CONDITIONS])
def test_a_wrong_conditions_file_is_rejected_before_any_api_call(tmp_path, name, edit, message):
    path = write_conditions(tmp_path, edit(conditions_text()))

    with pytest.raises(compare.ConfigError, match=message):
        compare.load_config(path)


def test_a_wrong_conditions_file_stops_the_command_with_exit_code_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "make_model", lambda ctx, eff: pytest.fail("API を呼ぶ前に止まるはず"))
    path = write_conditions(tmp_path, replaced(conditions_text(), 'old = "queryに調べたい内容を指定する。"', 'old = "存在しない文"'))

    assert run("run", "--conditions", path, "--allow-dirty", "--out", tmp_path / "out") == 1

    assert "ちょうど 1 か所" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_replacement_with_another_argument_schema_is_rejected(tmp_path, monkeypatch):
    import types

    fake = types.ModuleType("fake_retrieval")

    def two_arguments(query: str, limit: int) -> str:
        return "{}"

    fake.two_arguments = two_arguments
    monkeypatch.setitem(sys.modules, "fake_retrieval", fake)
    path = write_conditions(tmp_path, replaced(conditions_text(), "app.retrieval_lec05:search_materials_with_neighbors", "fake_retrieval:two_arguments"))

    with pytest.raises(compare.ConfigError, match="引数のスキーマが、元のツールと違います"):
        compare.load_config(path)


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


def test_the_doc_hint_old_text_is_in_the_description_exactly_once():
    base = compare.registered_tools()["search_materials"].description

    assert base.count("queryに調べたい内容を指定する。") == 1
    new = compare.build_effective(compare.load_config(), "doc_hint").tool_defs[0]["description"]
    assert "最大5ページだけ" in new and "（例: \"MCP 権限\"）" in new and "queryに調べたい内容を指定する。" not in new


# --- チェックの正規表現の部品（[fragments]）---


def test_fragments_are_expanded_into_the_patterns_when_the_file_is_loaded():
    cfg = compare.load_config()
    patterns = [check["pattern"] for item in cfg["questions"].values() for check in item["checks"] if "pattern" in check]

    assert patterns and all("<<" not in pattern and ">>" not in pattern for pattern in patterns)  # 読み込んだあとは、部品が展開されている
    assert "<<NEG_SURU>>" in conditions_text() and "(?!(?:" in next(c["pattern"] for c in cfg["questions"]["q1"]["checks"] if c["label"] == "q1_fix")
    assert set(cfg["fragments"]) >= {"NA", "TE", "NEG_SURU", "NEG_E", "TRAIN_OBJ", "LEARNING"}


def test_expand_fragments_wraps_each_part_in_a_group_and_expands_nested_parts():
    parts = {"A": "x|y", "B": "<<A>>z"}

    assert compare.expand_fragments("a<<A>>b", parts, "テスト") == "a(?:x|y)b"
    assert compare.expand_fragments("<<B>>", parts, "テスト") == "(?:(?:x|y)z)"
    assert re.fullmatch(compare.expand_fragments("a<<A>>b", parts, "テスト"), "ayb") and not re.fullmatch(compare.expand_fragments("a<<A>>b", parts, "テスト"), "ab")
    with pytest.raises(compare.ConfigError, match="知らない部品 <<C>>"):
        compare.expand_fragments("<<C>>", parts, "テスト")
    with pytest.raises(compare.ConfigError, match="循環"):
        compare.expand_fragments("<<A>>", {"A": "<<B>>", "B": "<<A>>"}, "テスト", ("A",))


# --- 全条件の上限と、コードの sha256 ---


def test_every_condition_has_the_same_request_limit_of_20_and_none_overrides_it():
    cfg = compare.load_config()

    assert cfg["defaults"]["request_limit"] == 20
    assert all("request_limit" not in spec for spec in cfg["variants"].values())  # 条件ごとの上書きはしない（打ち切りの基準を方式でそろえる）
    assert {name: compare.build_effective(cfg, name).request_limit for name in cfg["variants"]} == {name: 20 for name in cfg["variants"]}
    assert "上限を 20" not in cfg["variants"]["tree"]["description"]


def code_root_copy(tmp_path):
    """sha256 に入れるコードのファイル（app/agent.py など 5 つ）を、一時フォルダに写す。compare.CODE_ROOT をここに向けて、中身を書き換えて確かめる。"""
    root = tmp_path / "code"
    for rel in (*compare.CODE_FILES, compare.LEC05_FILE):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compare.REPO / rel, root / rel)
    return root


def all_shas(cfg):
    return {name: compare.build_effective(cfg, name).sha256 for name in cfg["variants"]}


@pytest.mark.parametrize("rel", ["app/agent.py", "app/tools.py", "app/retrieval.py", "app/config.py"])
def test_a_change_in_agent_tools_retrieval_or_config_changes_the_sha256_of_every_condition(tmp_path, monkeypatch, rel):
    cfg = compare.load_config()
    root = code_root_copy(tmp_path)
    monkeypatch.setattr(compare, "CODE_ROOT", root)
    before = all_shas(cfg)
    original = (root / rel).read_text(encoding="utf-8")

    (root / rel).write_text(original + f"\n# {rel} を変えた（コメントだけ。関数の外なので、referenced_code では見えない）\n", encoding="utf-8")
    after = all_shas(cfg)

    assert all(after[name] != before[name] for name in before)  # 全条件の sha256 が変わる
    (root / rel).write_text(original, encoding="utf-8")
    assert all_shas(cfg) == before  # 戻せば、元の値


def test_a_change_in_retrieval_lec05_only_changes_the_conditions_that_use_it(tmp_path, monkeypatch):
    cfg = compare.load_config()
    root = code_root_copy(tmp_path)
    monkeypatch.setattr(compare, "CODE_ROOT", root)
    before = all_shas(cfg)

    (root / compare.LEC05_FILE).write_text((root / compare.LEC05_FILE).read_text(encoding="utf-8") + "\n# 変えた\n", encoding="utf-8")
    after = all_shas(cfg)

    changed = {name for name in before if after[name] != before[name]}
    assert changed == {"neighbors", "term_graph", "term_graph_cap3"}  # I1・A1・A2。取得処理か設定で、このモジュールを使う条件だけ
    assert after["vector"] == before["vector"] and after["topk15"] == before["topk15"] and after["tree"] == before["tree"]  # V0 などは変わらない


def test_the_code_files_in_the_sha256_are_the_same_for_every_condition_except_the_lec05_module():
    cfg = compare.load_config()
    effective = {name: compare.build_effective(cfg, name) for name in cfg["variants"]}

    assert {name: eff.uses_lec05 for name, eff in effective.items()} == {
        name: name in ("neighbors", "term_graph", "term_graph_cap3") for name in effective}
    for name, eff in effective.items():
        expected = [*compare.CODE_FILES, *([compare.LEC05_FILE] if eff.uses_lec05 else [])]
        assert list(eff.code_files) == expected and all(len(sha) == 64 for sha in eff.code_files.values()), name
    assert effective["vector"].code_files["app/retrieval.py"] == compare.file_sha256(compare.REPO / "app/retrieval.py")


def test_the_defaults_of_the_lec05_settings_do_not_change_the_sha256_of_conditions_that_do_not_use_the_module(monkeypatch):
    cfg = compare.load_config()
    before = all_shas(cfg)

    monkeypatch.setattr(lec05, "TERM_GRAPH_SECTION_CAP", 7)  # このモジュールの設定の既定値

    after = all_shas(cfg)
    assert after["vector"] == before["vector"] and after["topk15"] == before["topk15"]
    assert after["term_graph"] != before["term_graph"] and after["term_graph_cap3"] == before["term_graph_cap3"]  # cap3 は 3 に書き換えるので、既定値に左右されない


# --- 条件の解決（base → mode → retrieval → changes → settings → documents → request_limit） ---


def test_base_chains_are_resolved_through_to_the_mode():
    cfg = compare.load_config()

    cap3 = compare.resolve_spec(cfg, "term_graph_cap3")
    assert cap3["mode"] == "vector"
    assert cap3["retrieval"] == {"search_materials": "app.retrieval_lec05:search_materials_term_graph"}  # 土台の term_graph から
    assert cap3["settings"] == {"TERM_GRAPH_SECTION_CAP": 3}
    assert compare.resolve_spec(cfg, "tree")["request_limit"] == compare.resolve_spec(cfg, "vector")["request_limit"] == 20
    assert compare.resolve_spec(cfg, "doc_hint")["request_limit"] == 20


def test_the_effective_conditions_of_every_mode():
    cfg = compare.load_config()
    from app import agent as agent_module

    full = compare.build_effective(cfg, "full")
    assert (full.instructions, full.tools, full.documents, full.prompt_template) == (
        agent_module.FULL_CONTEXT_INSTRUCTIONS, [], "all", agent_module.FULL_CONTEXT_PROMPT)
    assert compare.build_effective(cfg, "full_aligned").documents == SCOPE_214
    vector = compare.build_effective(cfg, "vector")
    assert vector.instructions == agent_module.INSTRUCTIONS and [t.name for t in vector.tools] == ["search_materials"]
    assert compare.build_effective(cfg, "tree").instructions == agent_module.TREE_INSTRUCTIONS
    assert [t.name for t in compare.build_effective(cfg, "tree").tools] == ["browse_tree"]
    assert compare.build_effective(cfg, "graph").instructions == agent_module.GRAPH_INSTRUCTIONS
    assert [t.name for t in compare.build_effective(cfg, "graph").tools] == ["search_graph"]
    assert compare.build_effective(cfg, "tree").request_limit == 20


def test_every_condition_has_its_own_sha256_and_it_does_not_change_between_builds():
    cfg = compare.load_config()
    first = {name: compare.build_effective(cfg, name).sha256 for name in cfg["variants"]}

    assert len(set(first.values())) == len(first)
    assert first == {name: compare.build_effective(cfg, name).sha256 for name in cfg["variants"]}


def test_the_sha256_follows_the_code_the_settings_the_data_and_the_limits(monkeypatch):
    cfg = compare.load_config()

    def sha(name, config=cfg):
        return compare.build_effective(config, name).sha256

    before = {name: sha(name) for name in ("neighbors", "term_graph", "vector")}

    def changed_neighbors(query: str) -> str:
        return "{}"

    changed_neighbors.__doc__ = tools.search_materials.__doc__
    with monkeypatch.context() as m:
        m.setattr(lec05, "search_materials_with_neighbors", changed_neighbors)
        assert sha("neighbors") != before["neighbors"]  # 取得処理の関数が変わった
        assert sha("term_graph") == before["term_graph"] and sha("vector") == before["vector"]  # 別の取得処理の条件は、変わらない
    assert sha("neighbors") == before["neighbors"]

    def below(*args, **kwargs):
        return []

    below.__module__ = "app.retrieval_lec05"
    with monkeypatch.context() as m:
        m.setattr(lec05, "add_neighbors", below)  # ツール関数の下で使う関数が変わった
        assert sha("neighbors") != before["neighbors"] and sha("term_graph") == before["term_graph"]
    other = dict(cfg, defaults={**cfg["defaults"], "request_limit": 13})
    assert sha("vector", other) != before["vector"]  # 上限
    with monkeypatch.context() as m:
        m.setattr(retrieval, "RAG_TOP_K", 6)  # 設定の既定値
        assert sha("vector") != before["vector"]
    with monkeypatch.context() as m:
        m.setattr(compare, "data_hashes", lambda: {"lectures.json": "x"})  # データ
        assert sha("vector") != before["vector"]
    with monkeypatch.context() as m:
        m.setattr(retrieval, "get_all_materials", lambda: "")  # 全条件で使う app.retrieval の関数
        assert sha("vector") != before["vector"]


def test_the_code_fingerprint_follows_the_functions_below_the_tool_function_in_the_same_module():
    names = compare.referenced_code(lec05.search_materials_with_neighbors)

    for expected in ("search_materials_with_neighbors", "_tool_json", "search_with_neighbors", "add_neighbors", "_pages"):
        assert f"app.retrieval_lec05:{expected}" in names, expected
    assert "app.retrieval_lec05:search_term_graph" not in names  # 用語から始める Graph 検索のコードは、前後ページの条件に関係しない
    graph = compare.referenced_code(lec05.search_materials_term_graph)
    assert "app.retrieval_lec05:select_pages" in graph and "app.retrieval_lec05:GraphData" in graph
    assert "app.retrieval_lec05:_LECTURE_PATTERN" in graph  # 関数が使う定数も入る
    assert not any(key.endswith(":TRACE") or key.endswith(":NEIGHBOR_WINDOW") for key in names | graph.keys())  # 実行ごとに変わるもの・設定は入れない


# --- 条件の再現: app.agent.main() の各モードと、同じ指示・ツール・prompt になる ---


def captured_tools(info_tools):
    return [(t.name, t.description, t.parameters_json_schema) for t in info_tools]


def capture_main(monkeypatch, capsys, *flags, question="LoRAとは？"):
    """app.agent.main() を、フラグつきで動かして、モデルが受け取ったものを返す（モデルは FunctionModel で捕まえる）。"""
    from app import agent as agent_module

    seen = {}

    def fn(messages, info):
        seen.update(instructions=info.instructions, tools=captured_tools(info.function_tools), prompt=compare.first_user_text(messages))
        return ModelResponse(parts=[TextPart("答えです。")])

    monkeypatch.setattr(agent_module, "build_model", lambda: FunctionModel(fn))
    monkeypatch.setattr(sys, "argv", ["agent", *flags, question])
    agent_module.main()
    capsys.readouterr()
    return seen


def capture_runner(monkeypatch, variant, question="LoRAとは？"):
    """compare.py の run_attempt が、モデルへ渡すものを返す。"""
    seen = {}

    def fn(messages, info):
        seen.update(instructions=info.instructions, tools=captured_tools(info.function_tools), prompt=compare.first_user_text(messages))
        return ModelResponse(parts=[TextPart("答えです。")])

    cfg = compare.load_config()
    ctx = compare.Context(cfg=cfg, out=Path("unused"), model=MODEL)
    cond = compare.Condition("exercise", variant, cfg["variants"][variant]["label"], ctx.effective(variant).mode, "q1", question)
    monkeypatch.setattr(compare, "make_model", lambda ctx, eff: FunctionModel(fn))
    compare.run_attempt(ctx, cond, ctx.effective(variant))
    return seen


@pytest.mark.parametrize(("flag", "variant"), [("--tree", "tree"), ("--graph", "graph"), ("--full-context", "full")])
def test_the_runner_gives_the_model_the_same_instructions_tools_and_prompt_as_the_cli(monkeypatch, capsys, flag, variant):
    cli = capture_main(monkeypatch, capsys, flag)
    runner = capture_runner(monkeypatch, variant)

    assert runner["instructions"] == cli["instructions"]
    assert runner["tools"] == cli["tools"]
    assert runner["prompt"] == cli["prompt"]
    if variant == "full":
        assert runner["tools"] == [] and runner["prompt"].startswith("<資料>\n資料: orientation /")
    else:
        assert len(runner["tools"]) == 1 and runner["prompt"] == "LoRAとは？"


def test_vector_has_the_same_instructions_and_one_tool_while_the_cli_has_three(monkeypatch, capsys):
    """差を明記するテスト。基本形は search_materials だけ。CLI（fork の agent）は make_quiz と承認付き save_learning_log も持つ。"""
    cli = capture_main(monkeypatch, capsys)
    runner = capture_runner(monkeypatch, "vector")

    assert runner["instructions"] == cli["instructions"]
    assert [t[0] for t in runner["tools"]] == ["search_materials"]
    assert sorted(t[0] for t in cli["tools"]) == ["make_quiz", "save_learning_log", "search_materials"]
    assert [t for t in cli["tools"] if t[0] == "search_materials"] == runner["tools"]  # search_materials の定義は同じ


def test_full_context_matches_run_full_context_and_the_tools_are_empty(monkeypatch):
    from app import agent as agent_module

    question = "LoRAとは？"
    seen = {}

    def fn(messages, info):
        seen.setdefault("calls", []).append((info.instructions, captured_tools(info.function_tools), compare.first_user_text(messages)))
        return ModelResponse(parts=[TextPart("答えです。")])

    retrieval.run_full_context(agent, question, FunctionModel(fn), instructions=agent_module.FULL_CONTEXT_INSTRUCTIONS,
                               prompt_template=agent_module.FULL_CONTEXT_PROMPT)
    runner = capture_runner(monkeypatch, "full", question)

    assert seen["calls"] == [(runner["instructions"], runner["tools"], runner["prompt"])]
    assert runner["tools"] == []


def test_full_aligned_prompt_has_214_page_headings_3_documents_and_each_block_is_in_the_full_prompt(monkeypatch):
    def materials_of(prompt):
        return re.fullmatch(r"<資料>\n(.*)\n</資料>\n\n質問: .*", prompt, re.S).group(1)

    full = materials_of(capture_runner(monkeypatch, "full")["prompt"])
    aligned = materials_of(capture_runner(monkeypatch, "full_aligned")["prompt"])

    assert full.count("ページ: PDF p.") == 438 and aligned.count("ページ: PDF p.") == 214
    assert re.findall(r"^資料: (\S+) /", full, re.M).count("orientation") == 40
    assert set(re.findall(r"^資料: (\S+) /", aligned, re.M)) == set(SCOPE_214)
    blocks = re.split(r"\n\n(?=資料: )", aligned)
    assert len(blocks) == 214
    assert all(block in full for block in blocks)  # 書式を写し取らず、同じ関数（get_all_materials）で作るので、同じ書式になる


def test_full_materials_are_get_all_materials_and_the_documents_option_does_not_change_the_global(monkeypatch):
    cfg = compare.load_config()
    original = retrieval.RAG_SOURCE

    assert compare.build_materials(compare.build_effective(cfg, "full")) == retrieval.get_all_materials()
    subset = compare.build_materials(compare.build_effective(cfg, "full_aligned"))
    assert retrieval.RAG_SOURCE is original  # 一時的な差し替えは、終われば戻る
    assert subset != retrieval.get_all_materials() and subset.count("ページ: PDF p.") == 214
    assert compare.scope_documents(None, "vector") == list(retrieval.RAG_DOCUMENT_IDS)
    assert compare.scope_documents(compare.build_effective(cfg, "full_aligned").documents, "full") == list(retrieval.RAG_DOCUMENT_IDS)


def test_the_source_stand_in_is_restored_even_when_the_call_fails(monkeypatch):
    cfg = compare.load_config()
    original = retrieval.RAG_SOURCE

    def boom():
        raise RuntimeError("失敗")

    monkeypatch.setattr(retrieval, "get_all_materials", boom)
    with pytest.raises(RuntimeError):
        compare.build_materials(compare.build_effective(cfg, "full_aligned"))

    assert retrieval.RAG_SOURCE is original


# --- 引用の解析 ---


def cited(text, scope=SCOPE_214):
    return compare.parse_citations(text, scope)


def pages_of(citations):
    return [c["page_ids"] for c in citations]


@pytest.mark.parametrize("text", ["根拠: 第2回・PDF p.52", "（第2回 p.52）", "第２回・ＰＤＦ ｐ．５２", "第2回、PDF　p. 52 より", "第2回 P.52"])
def test_the_basic_citation_forms_resolve_to_one_page(text):
    result = cited(text)

    assert [(c["page"], c["status"], c["page_ids"]) for c in result] == [(52, "ok", ["lec02-main-p052"])]
    assert result[0]["raw"]  # raw の文字列は必ず残す


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("第2回 p.51–52", ["lec02-main-p051", "lec02-main-p052"]),
        ("第3回 p.13、39、40", ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"]),
        ("第3回・PDF p.13・39・40", ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"]),
        ("第3回 p.13, 39, 40", ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"]),
        ("第2回 pp.51〜52", ["lec02-main-p051", "lec02-main-p052"]),
        ("第2回 p.51-52", ["lec02-main-p051", "lec02-main-p052"]),
        ("第2回の52ページ", ["lec02-main-p052"]),
        ("第3回の13、39、40ページ", ["lec03-main-p013", "lec03-main-p039", "lec03-main-p040"]),
        ("第3回・PDF p.13、第3回・PDF p.39", ["lec03-main-p013", "lec03-main-p039"]),
        ("第3回 p.13 と p.39", ["lec03-main-p013", "lec03-main-p039"]),
        ("第3回 p.13–p.15", ["lec03-main-p013", "lec03-main-p014", "lec03-main-p015"]),
        ("第3回 p.13～15", ["lec03-main-p013", "lec03-main-p014", "lec03-main-p015"]),  # 全角の波線は、NFKC で ~ になる
    ],
)
def test_ranges_and_lists_expand_to_one_citation_per_page(text, expected):
    result = cited(text)

    assert [c["page_ids"][0] for c in result] == expected and all(c["status"] == "ok" for c in result)
    assert all(c["raw"] for c in result)


def test_a_citation_without_a_lecture_on_its_own_line_looks_back_over_the_previous_lines_up_to_80_characters():
    result = cited("第3回の本講義では\n- PDF p.13 を参照。")  # 回は前の行にある

    assert [(c["page_ids"], c["cue"], c["cue_window"], c["status"]) for c in result] == [(["lec03-main-p013"], "lecture", "lines", "ok")]
    lecture_2 = cited("第2回・PDF p.52\n- 関連する PDF p.51")  # 手がかりの無い行の p.N は、前の行の「第2回」を使う
    assert [(c["page_ids"][0], c["cue_window"]) for c in lecture_2] == [("lec02-main-p052", "line"), ("lec02-main-p051", "lines")]
    switched = cited("第2回 p.52\n第3回 p.35\nPDF p.34")
    assert [c["page_ids"][0] for c in switched] == ["lec02-main-p052", "lec03-main-p035", "lec03-main-p034"]  # 範囲に 2 つの回があれば、直前のほう
    assert [c["cue_window"] for c in cited("第3回・PDF p.39 のほか、PDF p.62 も")] == ["line", "line"]  # 同じ行に手がかりがあれば、先にそれを使う


def test_a_citation_with_no_cue_in_the_80_characters_inherits_the_lecture_of_the_previous_citation():
    text = "第3回・PDF p.13\n" + "あ" * 100 + "\n- 補足は PDF p.62 です。\n- PDF p.52 も同じ回です。"  # 62 と 52 の前 80 字に、回が無い

    result = cited(text)

    assert [(c["page"], c["cue"], c["cue_window"], c["page_ids"]) for c in result] == [
        (13, "lecture", "line", ["lec03-main-p013"]), (62, "carry", "carry", ["lec03-main-p062"]), (52, "carry", "carry", ["lec03-main-p052"])]


def test_without_any_lecture_the_page_is_decided_only_when_one_document_has_it():
    assert [(c["status"], c["page_ids"]) for c in cited("PDF p.73")] == [("ok", ["lec01-main-p073"])]  # 73 ページあるのは第1回だけ
    many = cited("PDF p.52")[0]
    assert many["status"] == "unresolved" and many["page_ids"] == ["lec01-main-p052", "lec02-main-p052", "lec03-main-p052"]
    assert cited("PDF p.52。第2回の話です。")[0]["status"] == "unresolved"  # 手がかりは、引用の前にある 40 字だけ


def test_the_cue_window_is_the_same_line_first_40_characters_then_the_previous_80_characters():
    assert cited("第2回 PDF p.52")[0]["cue_window"] == "line"
    previous_line = cited("第2回の説明です。\nPDF p.52")[0]
    assert (previous_line["status"], previous_line["page_ids"], previous_line["cue_window"]) == ("ok", ["lec02-main-p052"], "lines")  # 前の行の「第2回」（80 字以内）
    farther_in_the_line = cited("第2回" + "あ" * 45 + "PDF p.52")[0]  # 同じ行でも 40 字を超えるが、80 字以内
    assert (farther_in_the_line["status"], farther_in_the_line["cue_window"]) == ("ok", "lines")
    too_far = cited("第2回" + "あ" * 85 + "PDF p.52")[0]  # 80 字を超えれば、手がかりとして使わない
    assert (too_far["status"], too_far["cue"], too_far["cue_window"]) == ("unresolved", None, None)


def test_in_the_full_context_a_lecture_number_names_main_and_exercise_so_cues_decide():
    both = cited("第3回・PDF p.13", SCOPE_ALL)  # 本講義にも演習にも 13 ページがある
    assert [(c["status"], c["page_ids"]) for c in both] == [("ambiguous", ["lec03-main-p013", "lec03-exercise-p013"])]
    assert cited("第3回の演習 PDF p.13", SCOPE_ALL)[0]["page_ids"] == ["lec03-exercise-p013"]
    assert cited("第3回 本講義 PDF p.13", SCOPE_ALL)[0]["page_ids"] == ["lec03-main-p013"]
    assert cited("資料: lec03-main / AIエージェント, PDF p.13", SCOPE_ALL)[0]["page_ids"] == ["lec03-main-p013"]  # 資料 ID
    assert cited("資料名「基盤モデルの仕組み」の PDF p.52", SCOPE_ALL)[0]["page_ids"] == ["lec02-main-p052"]  # lectures.json の資料名
    assert cited("LoRAで回答形式を学習し、学習前後を比較する PDF p.21", SCOPE_ALL)[0]["page_ids"] == ["lec02-exercise-p021"]
    assert cited("初回冒頭案内 p.3", SCOPE_ALL)[0]["page_ids"] == ["orientation-p003"]


def test_a_lecture_with_only_one_document_is_decided_and_the_page_that_exists_in_one_is_decided():
    assert cited("第4回 p.10", SCOPE_ALL)[0]["page_ids"] == ["lec04-exercise-p010"]  # 第4回は演習しかない
    assert cited("第5回 p.10", SCOPE_ALL)[0]["page_ids"] == ["lec05-main-p010"]
    assert cited("第2回・PDF p.52", SCOPE_ALL)[0]["page_ids"] == ["lec02-main-p052"]  # 演習は 31 ページまでなので、52 ページは本講義だけ
    assert cited("第4回 p.10", SCOPE_214)[0]["status"] == "invalid"  # 214 ページの条件には、第4回の資料が無い


def test_the_ambiguous_citation_counts_for_gold_when_a_candidate_is_gold():
    result = cited("第3回・PDF p.13", SCOPE_ALL)
    metrics = compare.citation_metrics(result, ["lec03-main-p013"], ["lec03-exercise-p013"], {"lec03-main-p013"})

    assert metrics["gold"] == 1 and metrics["ambiguous"] == 1 and metrics["unresolved"] == 1  # 曖昧の数は別に数える
    assert metrics["pages"] == 1 and metrics["other"] == 0 and metrics["acceptable"] == 1


@pytest.mark.parametrize(
    "text",
    ["4bitに量子化します。", "2026年の資料です。", "p値は関係ありません。", "p<0.05 で有意です。", "GPT-4 と MP4 と P2P と app 3", "3つの方法があります。",
     "第3回の 2 つの例", "約1/3 を学習します。", "p.2026 は、ページではありません。", "p.52、2026年", "1.5倍に増えます。"],
)
def test_things_that_are_not_citations_are_not_picked_up(text):
    result = cited(text)

    assert [c["raw"] for c in result if c["page"] != 52] == []
    assert all(c["page"] in (None, 52) for c in result)


def test_a_number_after_a_page_list_is_not_part_of_the_list_when_it_is_a_year_or_a_unit():
    assert [c["page"] for c in cited("第3回 p.13、4bit、2026年、39ページ")] == [13, 39]
    assert [c["page"] for c in cited("第3回 p.13、39、2026年、4bit")] == [13, 39]


def test_missing_pages_are_invalid_and_wide_ranges_are_not_counted():
    result = cited("第2回 p.999 と、第2回 p.1-50 と、第2回 p.52", SCOPE_214)

    assert [(c["status"], c["page"]) for c in result] == [("invalid", 999), ("wide_range", None), ("ok", 52)]
    assert result[1]["raw"] == "p.1-50"
    metrics = compare.citation_metrics(result, ["lec02-main-p052"], [], set())
    assert (metrics["pages"], metrics["invalid"], metrics["wide_range"], metrics["gold"]) == (1, 1, 1, 1)  # 数えるのは 52 ページだけ
    assert [c["status"] for c in cited("第2回 p.52-51")] == ["invalid"]  # 逆順の範囲


def test_citation_metrics_count_gold_acceptable_others_and_the_two_by_two():
    cfg = compare.load_config()
    gold, acceptable = cfg["questions"]["q2"]["gold"], cfg["questions"]["q2"]["acceptable"]
    answer = "第3回・PDF p.13、39 と、第3回・PDF p.65、第1回・PDF p.10。\nPDF p.52"
    metrics = compare.citation_metrics(cited(answer), gold, acceptable, {"lec03-main-p013", "lec03-main-p040"})

    assert (metrics["gold"], metrics["gold_total"], metrics["gold_rate"]) == (2, 3, 2 / 3)
    assert metrics["gold_cited"] == ["lec03-main-p013", "lec03-main-p039"]
    assert (metrics["acceptable"], metrics["other"], metrics["unresolved"], metrics["pages"]) == (1, 2, 0, 5)  # p.65 が acceptable。p.10 と p.52（第1回を引き継ぐ）が、それ以外
    assert compare.citation_metrics(cited("PDF p.52"), gold, acceptable, set())["unresolved"] == 1  # 手がかりも引き継ぎもない引用は、未解決として数える
    # 取得 × 引用（gold ページごと）: p.13 は取れて引用／p.39 は取らずに引用／p.40 は取れたが引用せず
    assert metrics["grid"] == {"retrieved_cited": 1, "retrieved_not_cited": 1, "not_retrieved_cited": 1, "neither": 0}


def test_citation_metrics_for_a_question_without_gold_has_no_rate():
    metrics = compare.citation_metrics(cited("第2回 p.52"), [], [], set())

    assert metrics["gold_rate"] is None and metrics["gold_total"] == 0 and metrics["other"] == 1
    assert compare.cited_gold_cell({"metrics": metrics}) == "—"  # q3 のような、gold の無い質問は「—」
    assert compare.cited_gold_cell({"metrics": compare.citation_metrics(cited("第2回 p.52"), ["lec02-main-p052"], [], set())}) == "1/1"
    assert metrics["grid"] == {"retrieved_cited": 0, "retrieved_not_cited": 0, "not_retrieved_cited": 0, "neither": 0}


def test_the_neither_cell_counts_gold_pages_that_were_neither_retrieved_nor_cited():
    metrics = compare.citation_metrics([], ["lec02-main-p052", "lec02-main-p053"], [], set())

    assert metrics["grid"]["neither"] == 2 and metrics["gold"] == 0 and metrics["pages"] == 0


# --- ツールの戻り値の解析 ---


def page_row(page_id, **extra):
    document_id, number = page_id.rsplit("-p", 1)
    return {"page_id": page_id, "document_id": document_id, "lecture": int(document_id[3:5]), "document_title": "資料", "section": "章",
            "title": f"{page_id} のタイトル", "page": int(number), "content": f"{page_id} の本文です。", **extra}


def test_a_search_result_gives_full_text_pages_with_rank_and_distance():
    content = json.dumps({"results": [page_row("lec02-main-p052", distance=0.2), page_row("lec02-main-p051", distance=0.3)]}, ensure_ascii=False)

    parsed = compare.result_pages(content)
    retrieved = compare.analyze_retrieval([{"order": 1, "name": "search_materials", "args": {"query": "q"}, "result": content}])

    assert parsed["kind"] == "results" and [(e["via"], e["rank"], e["distance"]) for e in parsed["full"]] == [("search", 1, 0.2), ("search", 2, 0.3)]
    assert compare.full_text_ids(retrieved) == ["lec02-main-p052", "lec02-main-p051"] and retrieved["preview"] == []
    assert retrieved["chars_full_text"] == retrieved["chars_unique"] == sum(r["chars"] for r in retrieved["full_text"]) and retrieved["tool_return_chars"] == len(content)


def test_neighbors_roles_are_split_into_hit_and_neighbor_and_hits_are_ranked_by_distance():
    rows = [page_row("lec02-main-p051", distance=None, role="neighbor", neighbor_of=["lec02-main-p052"]),
            page_row("lec02-main-p052", distance=0.3, role="hit", neighbor_of=[]),
            page_row("lec02-main-p060", distance=0.1, role="hit", neighbor_of=[])]
    retrieved = compare.analyze_retrieval([{"order": 1, "name": "search_materials", "args": {}, "result": json.dumps({"results": rows})}])

    assert [(r["page_id"], r["via"], r["rank"], r["role"]) for r in retrieved["full_text"]] == [
        ("lec02-main-p051", "neighbor", None, "neighbor"), ("lec02-main-p052", "hit", 2, "hit"), ("lec02-main-p060", "hit", 1, "hit")]
    view = {"retrieved": retrieved}
    assert compare.role_counts(view) == "2/1"


def test_browse_tree_separates_the_opened_page_from_the_pages_seen_only_by_their_first_sentence():
    calls = []
    for order, node_id in enumerate([None, "lec02-main", "lec02-main-p052"], 1):
        calls.append({"order": order, "name": "browse_tree", "args": {} if node_id is None else {"node_id": node_id}, "result": tools.browse_tree(node_id)})

    retrieved = compare.analyze_retrieval(calls)

    assert compare.full_text_ids(retrieved) == ["lec02-main-p052"] and retrieved["full_text"][0]["via"] == "tree_open"
    previews = compare.preview_only_ids(retrieved)
    assert "lec02-main-p001" in previews and "lec02-main-p052" not in previews and len(previews) >= 6  # 講義の直下のページは、先頭文だけ
    assert [(p["node_id"], p["kind"]) for p in retrieved["tree_path"]] == [
        ("aie2026-materials", "collection"), ("lec02-main", "document"), ("lec02-main-p052", "page")]  # たどった目次の経路
    assert retrieved["full_text"][0]["chars"] == len(retrieval.read_page("lec02-main-p052")["content"])
    view = {"retrieved": retrieved, "rec": {"mode": "tree"}}
    assert compare.opened_count(view) == 1


def test_search_graph_gives_seeds_and_expanded_pages_with_hops_and_paths(embed_stored_page):
    content = tools.search_graph("LoRAとQLoRA")  # 本物の DB と graph.json。質問の埋め込みは lec02-main-p052 の保存済みのもの

    retrieved = compare.analyze_retrieval([{"order": 1, "name": "search_graph", "args": {"query": "LoRAとQLoRA"}, "result": content}])
    seeds = [r for r in retrieved["full_text"] if r["via"] == "seed"]
    expanded = [r for r in retrieved["full_text"] if r["via"] == "expanded"]

    assert seeds[0]["page_id"] == "lec02-main-p052" and seeds[0]["rank"] == 1 and seeds[0]["hops"] == 0 and seeds[0]["distance"] < 1e-6
    assert len(seeds) == 3 and expanded and all(r["hops"] in (1, 2) and r["path"] for r in expanded) and len(seeds) + len(expanded) <= 10
    assert compare.graph_counts({"retrieved": retrieved}) == f"{len(seeds)}/{len(expanded)}"


def test_other_json_shapes_fall_back_to_pages_found_anywhere_inside():
    nested = json.dumps({"data": {"pages": [page_row("lec03-main-p013")]}}, ensure_ascii=False)

    parsed = compare.result_pages(nested)

    assert parsed["kind"] == "other" and [e["page"]["page_id"] for e in parsed["full"]] == ["lec03-main-p013"]


@pytest.mark.parametrize("content", [None, "", "これは JSON ではありません", "[1, 2, 3]", "42", '{"x": 1}', "Invalid JSON: 検索を直してください", {"results": "not a list"}, 123])
def test_returns_that_are_not_page_data_never_raise(content):
    parsed = compare.result_pages(content)

    assert parsed["full"] == [] and parsed["preview"] == []
    retrieved = compare.analyze_retrieval([{"order": 1, "name": "search_materials", "args": "not a dict", "result": content}])
    assert retrieved["full_text"] == [] and retrieved["chars_full_text"] == 0


def test_an_error_return_is_reported_and_has_no_pages():
    content = json.dumps({"error": "検索語が空です。", "results": []}, ensure_ascii=False)

    retrieved = compare.analyze_retrieval([{"order": 1, "name": "search_materials", "args": {"query": ""}, "result": content}])

    assert retrieved["full_text"] == [] and retrieved["errors"] == [{"call": 1, "name": "search_materials", "error": "検索語が空です。"}]


def test_gold_retrieval_depends_on_the_mode_and_the_documents_that_were_given():
    retrieved = {"full_text": [{"page_id": "lec02-main-p052", "via": "search", "rank": 1}, {"page_id": "lec02-main-p060", "via": "search", "rank": 2}], "preview": []}
    gold = ["lec02-main-p052", "lec02-main-p053"]

    vector = compare.gold_retrieval({"mode": "vector"}, retrieved, gold)
    assert (vector["hit"], vector["rate"], vector["ranks"]) == (["lec02-main-p052"], 0.5, {"lec02-main-p052": 1, "lec02-main-p053": None})
    full = compare.gold_retrieval({"mode": "full", "materials": {"documents": SCOPE_ALL}}, {"full_text": [], "preview": []}, gold)
    assert full["rate"] == 1.0  # 全文投入は、gold が対象資料に入っていれば取れたことになる
    aligned = compare.gold_retrieval({"mode": "full", "materials": {"documents": SCOPE_214}}, {"full_text": [], "preview": []}, ["lec02-exercise-p021"])
    assert aligned["rate"] == 0.0  # F214 では、演習の gold は範囲外
    assert compare.gold_retrieval({"mode": "full", "materials": {"documents": SCOPE_ALL}}, {"full_text": [], "preview": []}, ["lec02-exercise-p021"])["rate"] == 1.0
    assert compare.gold_retrieval({"mode": "vector"}, retrieved, [])["rate"] is None and compare.gold_cell({"gold": compare.gold_retrieval({"mode": "vector"}, retrieved, [])}) == "—"
    tree_preview_only = {"full_text": [], "preview": [{"page_id": "lec02-main-p052", "via": "tree_preview"}]}
    assert compare.gold_retrieval({"mode": "tree"}, tree_preview_only, ["lec02-main-p052"])["rate"] == 0.0  # 先頭文だけでは、取れたとは数えない


# --- 自動チェック ---


def test_every_check_in_conditions_toml_can_be_evaluated():
    cfg = compare.load_config()
    for qid in cfg["questions"]:
        for check in compare.checks_for(cfg, qid):
            assert compare.evaluate_check(check, {"answer": ""}) in (True, False), (qid, check)
            assert compare.evaluate_check(check, {}) in (True, False), (qid, check)
            assert check["role"] in compare.ROLES


def test_the_answer_is_normalized_before_the_pattern_is_searched():
    assert compare.normalize_answer(" Ｌｏ Ｒ Ａ\n　と　QLoRA ") == "loraとqlora"
    check = {"label": "x", "kind": "answer_regex", "pattern": "4-?(bit|ビット)"}
    assert compare.evaluate_check(check, {"answer": "４ＢＩＴ に量子化"}) is True  # 全角・大文字・空白
    assert compare.evaluate_check({**check, "kind": "answer_not_regex"}, {"answer": "4 bit"}) is False


QUESTION_SENTENCES = {
    "q1_fix": (["LoRAは元の重みを固定します。", "ベースモデルのパラメータは凍結したまま学習します。", "本体の重みは更新せず、追加した部分だけを学びます。"],
               ["LoRAは元の重みをすべて更新します。", "LoRAはモデル全体を再学習する手法です。", "QLoRAは量子化を加えます。"]),
    "q1_train": (["小さな行列AとBだけを学習します。", "低ランクの差分を学習します。", "アダプタだけを学習します。"],
                 ["全てのパラメータを学習します。", "元の重みの全体が対象です。"]),
    "q1_qlora": (["QLoRAは4bitに圧縮します。", "元の重みを量子化します。", "４ビットで保持します。"],
                 ["QLoRAはLoRAを速くする手法です。", "元のモデルを使います。"]),
    "q2_summary": (["コンパクションは古い履歴を要約に置き換えます。", "履歴を要約して詰めるのがコンパクションです。", "過去の会話を要約します。"],
                   ["コンパクションは履歴を削除します。", "会話をそのまま残します。"]),
    "q2_persist": (["メモリは外部の保存先に書き出して、次回のコンテキストに読み戻します。", "セッションをまたいで、新しい会話でも使えます。", "/memories のファイルを使います。"],
                   ["メモリは、その場の履歴に残ります。", "どちらも履歴の話です。"]),
    "q2_daily": (["日々のメモはすべて長期メモリに入るわけではありません。", "基準を超えたものだけが移ります。", "一度きりの調べものは残しません。"],
                 ["日々のメモはすべて長期メモリに入ります。", "毎晩の見直しで保存されます。"]),
    "q3_refuse": (["講義資料からは確認できませんでした。", "実測値の記載はありません。", "資料には、そのような数字は含まれていません。"],
                  ["平均応答時間は約3秒、正答率は80%です。", "とても速く、よく当たります。"]),
    "qa1_first": (["実務ではまず構造化出力を試します。", "構造化出力を先に使い、足りないときだけFTを使います。", "最初にAPIの形式指定を試します。"],
                  ["ファインチューニングで形式を学習させます。", "どちらも同じように使えます。"]),
    "qa1_decode": (["制約付きデコーディングで形式を守らせます。", "スキーマに合わない候補を外して選びます。", "候補のトークンを除外します。"],
                   ["プロンプトでお願いして守らせます。", "モデルが気をつけて書きます。"]),
    "qa2_persist_or_cannot": (["悪意ある入力が、後の会話でも指示として残ります。", "モデルは指示とデータを区別できません。", "見分けられないので、同じ文字列として扱います。"],
                              ["メモリは便利な機能です。", "容量に上限があります。"]),
    "qa2_review": (["メモリの中身を、ときどき見直します。", "長期メモリに何が入ったかを確かめます。", "中身を点検してください。"],
                   ["指示ファイルを読んでから入れます。", "外部の入力を信用しないようにします。"]),
}


def all_checks():
    cfg = compare.load_config()
    return {check["label"]: check for qid in cfg["questions"] for check in compare.checks_for(cfg, qid)}


@pytest.mark.parametrize("label", sorted(QUESTION_SENTENCES))
def test_each_required_pattern_matches_a_correct_paraphrase_and_not_a_wrong_sentence(label):
    check = all_checks()[label]
    good, bad = QUESTION_SENTENCES[label]

    assert check["role"] == "required" and check["kind"] == "answer_regex"
    for sentence in good:
        assert compare.evaluate_check(check, {"answer": sentence}) is True, sentence
    for sentence in bad:
        assert compare.evaluate_check(check, {"answer": sentence}) is False, sentence


# 回答は丁寧形（です・ます調）で書かれることが多い。「更新しません」「更新されません」「見当たりません」などを取りこぼさない。
# 正答の例は活用と丁寧形だけを変えたもの。誤答・無関係の例は、一致してはいけない（意味が逆のものを含む）
POLITE_SENTENCES = {
    "q1_fix": (
        ["LoRAでは、元の重みは更新されません。", "ベースモデルの重みは学習しません。", "元のパラメータは変更されません。", "本体の重みは変えません。",
         "元の重みは変更しません。", "元の重みは学習されません。", "元の重みは固定したまま、追加した部分だけを学習します。", "重みは凍結されます。"],
        ["重みを更新して学習します。", "重みをすべて学習します。", "LoRAは全パラメータを更新します。", "重みを変更して学習します。", "QLoRAは重みを4bitにします。"],
    ),
    "q1_train": (
        ["追加した小さな行列（アダプター）だけを学習します。", "学習するのは、低ランクの行列AとBです。", "元の重みとの差分だけを学習します。", "追加した行列A・Bのみを学習対象とします。",
         "追加したパラメータだけを学習します。", "追加した小さな行列のみを学習する。", "追加された部分のみが学習されます。"],
        ["全てのパラメータを学習します。", "モデル全体を学習します。", "データを追加します。", "追加の学習はしません。", "追加データを用意して、全体を学習します。",
         "学習するのは追加だけです。"],
    ),
    "q1_qlora": (
        ["QLoRAは、元の重みを4bitに量子化します。", "4ビットに圧縮した重みを使います。", "さらに量子化を加えます。"],
        ["QLoRAはLoRAを高速にします。", "16ビットで保持します。", "8bitに変換します。"],
    ),
    "q2_summary": (
        ["コンパクションは、古い履歴を要約に置き換えます。", "履歴を要約して詰め直します。", "会話の履歴を要約し、圧縮します。", "過去のやり取りを要約します。"],
        ["コンパクションは履歴を削除します。", "会話をそのまま保存します。", "メモリはファイルに保存されます。"],
    ),
    "q2_persist": (
        ["メモリは、外部の保存先に書き出して、次回のコンテキストに読み戻します。", "メモリはファイルに保存され、次回の会話でも使われます。",
         "メモリツールは、新しいセッションでも参照できるようにします。"],
        ["コンパクションは履歴を短くします。", "どちらも履歴の話です。", "メモリはセッションをまたいで使えます。", "セッションをまたぐメモリは、別のものです。"],
    ),
    "q2_daily": (
        ["日々のメモは、すべて長期メモリに入るわけではありません。", "すべての日々のメモが長期メモリに移されるわけではありません。", "日々のメモはすべては長期メモリに移されません。",
         "すべては長期メモリに残されません。", "毎晩の見直しで、基準を超えたものだけが長期メモリに移されます。"],
        ["日々のメモは、すべて長期メモリに入ります。", "日々のメモは毎晩長期メモリに移されます。", "毎晩すべてのメモを読み直して、長期メモリに移します。"],
    ),
    "q3_refuse": (
        ["講義資料からは、実測平均応答時間と正答率を確認することはできませんでした。", "資料には、実測値に関する情報は見当たりませんでした。",
         "実測平均応答時間と正答率は、講義資料に記載されていません。", "申し訳ありませんが、お答えできません。", "講義資料には記載がございません。", "実測値は不明です。",
         "講義資料には、今回作ったRAGの実測平均応答時間や正答率のデータは含まれていません。", "資料からは見つかりませんでした。"],
        ["平均応答時間は約3秒、正答率は80%でした。", "実測値を確認できます。", "資料には、RAGの説明があります。", "講義資料には実測平均応答時間が 3.2 秒と書かれています",
         "資料から計算すると、平均応答時間は3.2秒、正答率は85%です。"],
    ),
    "qa1_first": (
        ["実務では、まず構造化出力を試します。", "最初にAPIの構造化出力を使うのがよいでしょう。", "形式を守らせたいだけなら、構造化出力を優先してください。"],
        ["ファインチューニングで形式を学習させます。", "ファインチューニングを最初に検討します。"],
    ),
    "qa1_decode": (
        ["不適切な候補を外します。", "ルールに違反するトークンを取り除きます。", "許されないトークンは選ばれません。", "不正なトークンは生成されません。",
         "合わない候補は選びません。", "形式に合わない文字は出させません。", "スキーマに合わない出力は生成されません。"],
        ["候補を増やします。", "トークンを追加します。", "確率の高い候補を選びます。", "出力が形式に合わない場合は、検証してやり直します。",
         "形式に合わない場合は、プロンプトでもう一度お願いします。"],
    ),
    "qa2_persist_or_cannot": (
        ["モデルは、指示とデータを区別できません。", "指示とデータの区別がつきません。", "どちらも同じ文字列に見えるため、見分けがつきません。",
         "指示と入力を見分けることができません。"],
        ["モデルは指示とデータを区別できます。", "指示とデータの区別がつきます。", "メモリは便利な機能です。"],
    ),
    "qa2_review": (
        ["メモリの中身を、ときどき見直します。", "長期メモリに何が入ったかを、人が確かめます。", "メモリを定期的に確認してください。"],
        ["指示ファイルは読んでから入れます。", "メモリの容量には上限があります。"],
    ),
}


@pytest.mark.parametrize("label", sorted(POLITE_SENTENCES))
def test_each_required_pattern_matches_polite_correct_answers_and_not_wrong_or_unrelated_ones(label):
    check = all_checks()[label]
    good, bad = POLITE_SENTENCES[label]

    assert len(good) >= 3 and len(bad) >= 2
    for sentence in good:
        assert compare.evaluate_check(check, {"answer": sentence}) is True, sentence
    for sentence in bad:
        assert compare.evaluate_check(check, {"answer": sentence}) is False, sentence


@pytest.mark.parametrize("verb", ["更新", "学習", "変更"])
@pytest.mark.parametrize("ending", ["しない", "せず", "されない", "されず", "しません", "されません"])
def test_q1_fix_matches_every_plain_and_polite_negative_of_update_learn_and_change(verb, ending):
    check = all_checks()["q1_fix"]

    assert compare.evaluate_check(check, {"answer": f"元の重みは{verb}{ending}。"}) is True
    assert compare.evaluate_check(check, {"answer": f"元の重みを{verb}します。"}) is False  # 肯定（固定していない）は一致しない
    assert compare.evaluate_check(check, {"answer": f"元の重みを{verb}して、全体を学習します。"}) is False


@pytest.mark.parametrize("sentence", ["元の重みは変えない。", "元の重みは変えず、追加した部分だけを学びます。", "元の重みは変えません。"])
def test_q1_fix_matches_the_negatives_of_change_and_not_the_positive(sentence):
    check = all_checks()["q1_fix"]

    assert compare.evaluate_check(check, {"answer": sentence}) is True
    assert compare.evaluate_check(check, {"answer": "元の重みを変えます。"}) is False


def test_q2_daily_looks_25_characters_after_all_and_accepts_the_passive_negatives():
    check = all_checks()["q2_daily"]

    for gap, expected in ((0, True), (12, True), (25, True), (26, False)):  # 12 字を超える間も許す（25 字まで）
        assert compare.evaluate_check(check, {"answer": "すべて" + "あ" * gap + "わけではありません"}) is expected, gap
    for sentence in ("すべての日々のメモが長期メモリに移されるわけではありません。", "日々のメモはすべては長期メモリに移されません。",
                     "すべてが長期メモリに入るわけではないです。", "全部が長期メモリに残されないわけではなく、選ばれたものだけです。", "すべては長期メモリに入りません。"):
        assert compare.evaluate_check(check, {"answer": sentence}) is True, sentence
    for sentence in ("日々のメモはすべて長期メモリに移されます。", "すべての日々のメモが長期メモリに入ります。", "すべてのメモを長期メモリに移します。"):
        assert compare.evaluate_check(check, {"answer": sentence}) is False, sentence
    assert compare.evaluate_check(check, {"answer": "すべて長期メモリに入ります。入らないものはありません"}) is False  # 。をまたいだ一致は数えない


@pytest.mark.parametrize(
    "sentence",
    ["不適切な候補を外します。", "不適切な候補を外して、残りから選びます。", "ルールに違反するトークンを取り除きます。", "ルールに違反する候補を排除します。",
     "ルールに違反する候補を除外します。", "ルールに違反するトークンをマスクします。", "許されないトークンは選ばれません。", "不正なトークンは生成されません。",
     "不正な候補は出させません。", "ルール違反の候補は選びません。", "不適切な候補が外されます。", "不適切な候補は外す。"],
)
def test_qa1_decode_matches_the_verbs_for_dropping_candidates_in_plain_and_polite_forms(sentence):
    check = all_checks()["qa1_decode"]

    assert compare.evaluate_check(check, {"answer": sentence}) is True


@pytest.mark.parametrize("sentence", ["候補を増やします。", "トークンを追加します。", "候補を選びます。", "トークンを選びます。", "候補を並べて、確率の高いものを選びます。"])
def test_qa1_decode_does_not_match_sentences_that_keep_or_add_candidates(sentence):
    assert compare.evaluate_check(all_checks()["qa1_decode"], {"answer": sentence}) is False


@pytest.mark.parametrize(
    "sentence",
    ["モデルは指示とデータを区別できません。", "モデルは指示とデータを区別できず、そのまま従います。", "指示とデータの区別がつかない。", "指示と入力の区別がつきません。",
     "指示とデータを見分けられません。", "指示とデータの見分けがつきません。", "指示と入力を見分けることができません。", "指示とデータを区別することはできません。"],
)
def test_qa2_can_not_distinguish_needs_the_negative_form(sentence):
    assert compare.evaluate_check(all_checks()["qa2_persist_or_cannot"], {"answer": sentence}) is True


@pytest.mark.parametrize(
    "sentence",
    ["モデルは指示とデータを区別できます。", "モデルは指示を見分けられます。", "指示とデータの区別がつきます。", "指示と入力は見分けがつきやすいです。"],
)
def test_qa2_can_distinguish_is_not_the_right_answer(sentence):
    assert compare.evaluate_check(all_checks()["qa2_persist_or_cannot"], {"answer": sentence}) is False


@pytest.mark.parametrize(
    "sentence",
    ["講義資料からは、実測平均応答時間と正答率を確認することはできませんでした。", "実測値を確認することができません。", "実測値は確認できませんでした。",
     "資料には、そのような情報は見当たりませんでした。", "資料には見当たりません。", "実測値は講義資料に記載されておらず、記載がありません。", "講義資料には記載がございません。",
     "申し訳ありませんが、お答えできません。", "その質問には答えられません。", "実測値は不明です。", "実測値は不明でした。"],
)
def test_q3_refuse_matches_the_polite_ways_to_say_it_cannot_be_confirmed(sentence):
    assert compare.evaluate_check(all_checks()["q3_refuse"], {"answer": sentence}) is True


@pytest.mark.parametrize("particle", ["に", "の中に", "には", "から", "からは"])
@pytest.mark.parametrize("material", ["資料", "講義資料"])
def test_q3_refuse_matches_the_material_with_any_of_the_particles_and_a_long_gap(material, particle):
    check = all_checks()["q3_refuse"]

    assert compare.evaluate_check(check, {"answer": f"{material}{particle}、今回作ったRAGの実測平均応答時間や正答率のデータは含まれていません。"}) is True
    assert compare.evaluate_check(check, {"answer": f"{material}{particle}、実測平均応答時間や正答率は書かれていますが、ほかのことは分かっています。"}) is False


def test_q3_refuse_looks_40_characters_after_the_material_and_not_further():
    check = all_checks()["q3_refuse"]

    for gap, expected in ((0, True), (15, True), (40, True), (41, False)):  # 15 字を超える間も許す（40 字まで）
        assert compare.evaluate_check(check, {"answer": "講義資料には" + "x" * gap + "ありません"}) is expected, gap
    assert compare.evaluate_check(check, {"answer": "講義資料には実測値が書かれています。ありません"}) is False  # 。をまたいだ一致は数えない


@pytest.mark.parametrize(
    "sentence",
    ["資料からは見つかりませんでした。", "講義資料からは、該当する記述は見つかりませんでした。", "検索結果には該当の情報が見つからない。", "検索しても、実測値は見つからなかった。",
     "実測値の記録は見つかりません。"],
)
def test_q3_refuse_matches_not_found_on_its_own(sentence):
    assert compare.evaluate_check(all_checks()["q3_refuse"], {"answer": sentence}) is True


@pytest.mark.parametrize(
    "sentence",
    ["講義資料には実測平均応答時間が 3.2 秒と書かれています", "講義資料から、平均応答時間は約3秒と読み取れます。", "講義資料には、LoRAとQLoRAの説明があります。",
     "資料からは次のことが分かります。", "実測値が見つかりました。"],
)
def test_q3_refuse_does_not_match_an_answer_that_states_the_numbers_or_found_the_data(sentence):
    assert compare.evaluate_check(all_checks()["q3_refuse"], {"answer": sentence}) is False


@pytest.mark.parametrize(
    "sentence",
    ["追加したパラメータだけを学習します。", "追加した小さな行列のみを学習する。", "追加された部分のみが学習されます。", "足したアダプターだけを学習します。",
     "追加した2つの行列のみを学習します。", "追加したアダプタのみが学習されます。"],
)
def test_q1_train_matches_only_the_added_part_is_learned(sentence):
    assert compare.evaluate_check(all_checks()["q1_train"], {"answer": sentence}) is True


@pytest.mark.parametrize(
    "sentence",
    ["追加の学習はしません。", "追加データを用意して、全体を学習します。", "追加で学習します。", "学習するのは追加だけです。", "元の重みをすべて学習します。", "データを追加します。"],
)
def test_q1_train_does_not_match_added_without_only_or_learning(sentence):
    assert compare.evaluate_check(all_checks()["q1_train"], {"answer": sentence}) is False


def test_q1_train_looks_15_characters_after_added_and_10_after_only():
    check = all_checks()["q1_train"]

    def matches(first, second):
        return compare.evaluate_check(check, {"answer": "追加" + "x" * first + "だけ" + "x" * second + "学習"})

    assert matches(0, 0) is True and matches(15, 10) is True
    assert matches(16, 10) is False and matches(15, 11) is False


@pytest.mark.parametrize(
    "sentence",
    ["形式に合わない文字は出させません。", "スキーマに合わない出力は生成されません。", "スキーマに合わない文字を出さないようにします。", "形式に合わない出力を除外します。",
     "形式に合わない候補を外します。", "スキーマに合わないトークンは選びません。"],
)
def test_qa1_decode_matches_characters_and_outputs_that_do_not_fit_the_format(sentence):
    assert compare.evaluate_check(all_checks()["qa1_decode"], {"answer": sentence}) is True


@pytest.mark.parametrize(
    "sentence",
    ["出力が形式に合わない場合は、検証してやり直します。", "形式に合わない場合は、プロンプトでもう一度お願いします。", "出力の形式をプロンプトで指定します。",
     "スキーマに合う出力を返します。", "形式に合っている文字を選びます。"],
)
def test_qa1_decode_does_not_match_retrying_or_asking_for_the_format_in_the_prompt(sentence):
    assert compare.evaluate_check(all_checks()["qa1_decode"], {"answer": sentence}) is False


@pytest.mark.parametrize(
    "sentence",
    ["メモリはセッションをまたいで使えます。", "セッションをまたいで使えます。", "セッションをまたぐメモリは、別のものです。", "コンパクションとメモリは違います。"],
)
def test_q2_persist_is_not_satisfied_by_repeating_the_question_about_crossing_sessions(sentence):
    """質問文をなぞるだけの言い回しは、要点（書き出し・読み戻し・メモリツール）を確かめる規則にならないので、足していない。"""
    assert compare.evaluate_check(all_checks()["q2_persist"], {"answer": sentence}) is False


def test_score_trial_has_three_verdicts_correct_review_and_wrong():
    cfg = compare.load_config()
    q3 = compare.checks_for(cfg, "q3")
    refusal = "講義資料からは確認できませんでした。"

    correct = compare.score_trial(q3, {"answer": refusal})
    review = compare.score_trial(q3, {"answer": refusal + "一般には3秒前後とされます。"})
    wrong_with_flag = compare.score_trial(q3, {"answer": "平均応答時間は2.5秒です。"})  # required が一致せず、flag が一致
    wrong_without_flag = compare.score_trial(q3, {"answer": "お答えします。実測値があります。"})  # required が一致せず、flag も一致しない

    assert [s["verdict"] for s in (correct, review, wrong_with_flag, wrong_without_flag)] == ["correct", "review", "wrong", "wrong"]
    assert [s["correct"] for s in (correct, review, wrong_with_flag, wrong_without_flag)] == [True, False, False, False]  # 要確認は、正答に数えない
    assert [s["flagged"] for s in (correct, review, wrong_with_flag, wrong_without_flag)] == [False, True, True, False]
    assert compare.VERDICTS == ("correct", "review", "wrong") and [compare.VERDICT_NAMES[v] for v in compare.VERDICTS] == ["自動の正答", "要確認", "自動の誤答"]
    q1 = compare.score_trial(compare.checks_for(cfg, "q1"), {"answer": "元の重みは固定し、アダプタだけを学習します。QLoRAは4bitに量子化します。"})
    assert q1["verdict"] == "correct" and q1["flagged"] is None  # flag が無い質問は、正答か誤答か
    assert compare.score_trial([], {"answer": "x"})["verdict"] is None  # required が無ければ、判定しない
    bonus_only = compare.score_trial([{"label": "b", "kind": "answer_regex", "role": "bonus", "pattern": "x"}], {"answer": "y"})
    assert bonus_only["verdict"] is None and bonus_only["correct"] is None


# 肯定の語の直後に否定が付くと、意味が逆になる。(肯定の文, 否定の文): 肯定は一致し、否定は一致しない
NEGATION_PAIRS = {
    "q1_fix": [
        ("元の重みは固定します。", "元の重みは固定しません。"), ("元の重みは固定されます。", "元の重みは固定されません。"),
        ("元の重みは固定されています。", "元の重みは固定されていません。"), ("元の重みは凍結します。", "元の重みは凍結しません。"),
        ("ベースモデルの重みは固定したまま学習します。", "ベースモデルの重みは固定せずに学習します。"), ("元の重みは固定です。", "元の重みは固定ではありません。"),
        ("元の重みは固定しています。", "元の重みは固定していません。"), ("元の重みは固定できます。", "元の重みは固定できません。"),
    ],
    "q1_train": [
        ("アダプタだけを学習します。", "アダプタは学習しません。"), ("低ランクの差分を学習します。", "低ランクの差分は学習されません。"),
        ("小さな行列AとBを学習します。", "小さな行列AとBは学習できません。"), ("追加した部分を学習します。", "追加した部分は学習対象外です。"),
        ("アダプタを学習しています。", "アダプタは学習していません。"), ("学習するのはアダプタです。", "学習しないのはアダプタです。"),
        ("学習の対象は小さな行列です。", "学習の対象外は小さな行列です。"), ("学習の対象は小さな行列です。", "学習の対象ではないのは小さな行列です。"),
        ("アダプタを学習します。", "アダプタは学習せず、固定します。"),
    ],
    "q1_qlora": [
        ("元の重みを量子化します。", "元の重みは量子化しません。"), ("重みが量子化されます。", "重みは量子化されません。"),
        ("重みは量子化されています。", "重みは量子化されていません。"),
    ],
    "q2_summary": [
        ("履歴を要約に置き換えます。", "履歴は要約に置き換えません。"), ("古い履歴を要約に置き換えます。", "古い履歴は要約に置き換えられません。"),
        ("古い履歴を要約します。", "古い履歴は要約しません。"), ("履歴を要約して詰めます。", "履歴を要約して詰めません。"),
        ("履歴を要約して圧縮します。", "履歴を要約して圧縮しません。"), ("履歴を要約に差し替えます。", "履歴を要約に差し替えません。"),
        ("履歴を要約して置換します。", "履歴を要約して置換しません。"),
    ],
    "q2_persist": [
        ("メモリはモデルが書き出します。", "メモリはモデルが書き出しません。"), ("メモリは読み戻します。", "メモリは読み戻しません。"),
        ("メモリはファイルに保存します。", "メモリはファイルに保存しません。"), ("メモリはファイルに書きます。", "メモリはファイルに書きません。"),
        ("メモリは書き出されます。", "メモリは書き出されません。"),
    ],
    "qa1_first": [("構造化出力を優先します。", "構造化出力を優先しません。")],
    "qa1_decode": [
        ("不適切な候補を外します。", "不適切な候補は外しません。"), ("不適切な候補が外されます。", "不適切な候補は外されません。"),
        ("不正なトークンを取り除きます。", "不正なトークンは取り除きません。"), ("不正なトークンは取り除かれます。", "不正なトークンは取り除かれません。"),
        ("不正な候補を除外します。", "不正な候補は除外しません。"), ("不正な候補を排除します。", "不正な候補は排除しません。"),
        ("不正なトークンをマスクします。", "不正なトークンはマスクしません。"),
    ],
    "qa2_persist_or_cannot": [
        ("悪意ある入力が指示として残ります。", "悪意ある入力は指示として残りません。"), ("悪意ある入力が指示として働きます。", "悪意ある入力は指示として働きません。"),
        ("悪意ある入力が指示として扱われます。", "悪意ある入力は指示として扱われません。"), ("後の会話でも指示として残ります。", "後の会話では指示として残りません。"),
        ("悪意ある入力が指示として効いてしまいます。", "悪意ある入力は指示として効きません。"), ("悪意ある入力が指示として使われます。", "悪意ある入力は指示として使われません。"),
    ],
    "qa2_review": [
        ("メモリの中身を見直します。", "メモリの中身は見直しません。"), ("メモリを確認します。", "メモリは確認しません。"),
        ("メモリの中身を確認できます。", "メモリの中身は確認できません。"), ("メモリを点検します。", "メモリは点検しません。"),
        ("中身を確かめます。", "中身は確かめません。"), ("メモリをチェックします。", "メモリはチェックしません。"), ("メモリをレビューします。", "メモリはレビューしません。"),
    ],
}


@pytest.mark.parametrize(
    ("label", "positive", "negated"),
    [(label, positive, negated) for label, pairs in NEGATION_PAIRS.items() for positive, negated in pairs],
)
def test_a_negation_right_after_the_positive_word_is_not_a_match(label, positive, negated):
    check = all_checks()[label]

    assert compare.evaluate_check(check, {"answer": positive}) is True, positive
    assert compare.evaluate_check(check, {"answer": negated}) is False, negated  # 「固定します」と「固定しません」は、意味が逆


def test_the_negation_pairs_cover_the_required_rules_that_have_a_positive_word_to_negate():
    uncovered = {check["label"] for item in compare.load_config()["questions"].values() for check in item["checks"] if check["role"] == "required"} - set(NEGATION_PAIRS)

    assert uncovered == {"q2_daily", "q3_refuse"}  # この 2 つは、規則の候補がすべて否定の言い回し（「〜わけではありません」「確認できません」）で、肯定の語が無い


def test_the_answer_that_says_the_adapter_is_not_learned_is_not_an_automatic_correct_answer():
    cfg = compare.load_config()
    wrong = "LoRAは本体の重みを固定し、アダプタは学習しません。QLoRAは4bit量子化を加えます。"
    right = "LoRAは本体の重みを固定し、追加したアダプタだけを学習します。QLoRAは元の重みを4bit量子化します。"

    scored_wrong = compare.score_trial(compare.checks_for(cfg, "q1"), {"answer": wrong})
    scored_right = compare.score_trial(compare.checks_for(cfg, "q1"), {"answer": right})

    assert scored_wrong["results"] == {"q1_fix": True, "q1_train": False, "q1_qlora": True} and scored_wrong["verdict"] == "wrong"  # 「アダプタ」の語だけでは通らない
    assert scored_right["results"] == {"q1_fix": True, "q1_train": True, "q1_qlora": True} and scored_right["verdict"] == "correct"


@pytest.mark.parametrize(
    ("label", "sentence"),
    [("q1_fix", "重みは固定しません。"), ("q2_summary", "要約には置き換えません。"), ("q1_train", "LoRAは低ランクの差分を使います。"), ("q1_train", "アダプタを追加します。"),
     ("q1_train", "小さな行列は、元の重みに足されます。"), ("q1_train", "学習データを追加します。"), ("q2_summary", "古い履歴は要約に置き換えられません。"),
     ("qa2_persist_or_cannot", "悪意ある入力は、後の会話では指示として残りません。")],
)
def test_the_negated_examples_from_the_review_do_not_match(label, sentence):
    assert compare.evaluate_check(all_checks()[label], {"answer": sentence}) is False


@pytest.mark.parametrize(
    "sentence",
    ["学習するのは、低ランクの行列AとBです。", "学習の対象は小さな行列です。", "学習対象は追加した部分だけです。", "アダプタを学習します。", "小さな行列AとBだけを学習します。",
     "追加した部分のみが学習されます。", "追加したパラメータだけを学習します。", "学習できるのは、アダプタだけです。"],
)
def test_q1_train_needs_the_learned_part_and_the_word_learn_together_in_either_order(sentence):
    assert compare.evaluate_check(all_checks()["q1_train"], {"answer": sentence}) is True


def test_every_required_check_has_examples_and_the_other_roles_are_what_the_design_says():
    checks = all_checks()

    assert {label for label, check in checks.items() if check["role"] == "required"} == set(QUESTION_SENTENCES) == set(POLITE_SENTENCES)
    assert {label: check["role"] for label, check in checks.items() if check["role"] != "required"} == {
        "q2_scope": "reference", "q3_numflag": "flag", "qa1_ftuse": "bonus"}
    assert checks["q3_numflag"]["kind"] == "answer_not_regex"


def test_the_flag_check_marks_a_numeric_answer_for_review_without_scoring_it_zero():
    cfg = compare.load_config()
    refusal = "講義資料からは確認できませんでした。"

    clean = compare.score_trial(compare.checks_for(cfg, "q3"), {"answer": refusal})
    assert clean["correct"] is True and clean["flagged"] is False and clean["verdict"] == "correct"
    with_numbers = compare.score_trial(compare.checks_for(cfg, "q3"), {"answer": refusal + " 一般には3秒前後、正答率は80%程度とされます。"})
    assert with_numbers["flagged"] is True and with_numbers["correct"] is False and with_numbers["results"]["q3_refuse"] is True
    assert with_numbers["verdict"] == "review"  # required がすべて一致し、flag も一致: 要確認（0 点にも、正答にもしない）
    guess = compare.score_trial(compare.checks_for(cfg, "q3"), {"answer": "平均応答時間は2.5秒、正答率は92割です。"})
    assert guess["flagged"] is True and guess["results"]["q3_refuse"] is False and guess["verdict"] == "wrong"
    for text in ("0.5秒", "120ms", "80%", "9割", "２秒"):
        assert compare.score_trial(compare.checks_for(cfg, "q3"), {"answer": text})["flagged"] is True, text


def test_the_automatic_answer_needs_every_required_check_and_ignores_bonus_and_reference():
    cfg = compare.load_config()
    checks = compare.checks_for(cfg, "q1")
    full = "元の重みは固定し、低ランクの差分だけを学習します。QLoRAは4bitに量子化します。"

    assert compare.score_trial(checks, {"answer": full})["correct"] is True
    assert compare.score_trial(checks, {"answer": full.replace("QLoRAは4bitに量子化します。", "")})["correct"] is False
    q2 = compare.score_trial(compare.checks_for(cfg, "q2"), {"answer": "古い履歴を要約に置き換え、書き出して次回に読み戻します。基準を超えたものだけ移ります。"})
    assert q2["correct"] is True and q2["results"]["q2_scope"] is False and q2["flagged"] is None  # reference は正答に入らない
    qa1 = compare.score_trial(compare.checks_for(cfg, "qa1"), {"answer": "まず構造化出力を使います。制約付きデコーディングです。"})
    assert qa1["correct"] is True and qa1["results"]["qa1_ftuse"] is False  # bonus は正答に入らない


def test_generic_checks_from_lecture_3_still_work():
    calls = [{"name": "search_materials", "args": {"query": "LoRA"}, "response_index": 1, "result": '{"results": []}', "outcome": "success"}]
    rec = {"answer": "答え", "tool_calls": calls, "usage": {"requests": 3}}

    def passes(**check):
        return compare.evaluate_check({"label": "x", **check}, rec)

    assert passes(kind="tool_called", name="search_materials", args_contains='"query":"LoRA"') is True
    assert passes(kind="tool_called", name="search_graph") is False
    assert passes(kind="tool_not_called", name="search_graph") is True
    assert passes(kind="tool_count", name="search_materials", min=1, max=1) is True and passes(kind="tool_count", name="search_materials", value=2) is False
    assert passes(kind="tool_result_contains", name="search_materials", values=['"results"']) is True
    assert passes(kind="requests_at_most", value=3) is True and passes(kind="requests_at_most", value=2) is False
    assert passes(kind="answer_contains_all", values=["答"]) is True and passes(kind="answer_contains_any", values=["違う", "答え"]) is True
    assert passes(kind="answer_not_contains", values=["違う"]) is True


# --- ツールの戻り値の error の分類 ---


@pytest.mark.parametrize(
    ("message", "kind", "status"),
    [
        (api_error_message(429), "api", 429),
        (api_error_message(503), "api", 503),
        (api_error_message(400, "API key not valid"), "api", 400),
        (api_error_message(401, "x"), "api", 401),
        (api_error_message(403, ""), "api", 403),
        (api_error_message(404, "models/x is not found"), "api", 404),
        (api_error_message(500, "internal"), "api", 500),
        (api_error_message(502, ""), "api", 502),
        ("検索語が空です。", "model", None),
        ("対象の目次ノードが見つかりません。", "model", None),
        ("埋め込み入力が上限を超えるか計測不能です: 2500 tokens。", "model", None),
        ("埋め込みAPIが本文を切り捨てました。", "model", None),
        ("検索DBがありません。", "setup", None),
        ("no such table: chunks", "setup", None),
        ("unable to open database file", "setup", None),
        ("[Errno 2] No such file or directory: 'data/rag_data/tree.json'", "setup", None),
    ],
)
def test_tool_errors_are_classified_as_api_setup_or_model(message, kind, status):
    assert compare.classify_error_message(message) == (kind, status)


def test_a_returned_error_is_read_from_the_json_of_the_tool_result():
    assert compare.tool_error_message(json.dumps({"error": "検索語が空です。", "results": []}, ensure_ascii=False)) == "検索語が空です。"
    assert compare.tool_error_message(json.dumps({"results": []})) is None
    assert compare.tool_error_message("再試行してください") is None and compare.tool_error_message(None) is None


# --- 再試行: 埋め込みの失敗（ツールの例外処理を素通りして、run を打ち切る） ---


def flaky_embedding(monkeypatch, errors):
    """埋め込みが、errors を 1 つずつ出し、そのあとは lec02-main-p052 の保存済みの埋め込みを返す。"""
    vector = stored_vector("lec02-main-p052")
    state = {"calls": 0}

    def embed(text, task_type):
        state["calls"] += 1
        if state["calls"] <= len(errors):
            raise errors[state["calls"] - 1]
        return vector

    monkeypatch.setattr("app.retrieval.embed_text", embed)
    return state


def test_an_embedding_rate_limit_is_retried_after_the_advised_delay(tmp_path, monkeypatch, sleeps):
    state = flaky_embedding(monkeypatch, [genai_errors.ClientError(429, RATE_PER_MINUTE)])
    model, calls = tool_then_answer()
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 0

    assert sleeps == [7.0] and state["calls"] == 2
    record = load(result_path(tmp_path))
    assert record["status"] == "ok" and record["tool_errors"] == [] and record["embedding"]["calls"] == 1  # 成功した試みの分だけ
    assert [(a.get("source"), a.get("status"), a.get("action", "ok"), a.get("wait_s")) for a in record["attempts"]] == [("embedding", 429, "retry", 7.0), (None, None, "ok", None)]
    assert "retryDelay" in record["attempts"][0]["error"]  # 元のエラーの本文を残す


def test_an_embedding_daily_limit_stops_with_exit_code_2_and_saves_nothing(tmp_path, monkeypatch, sleeps, capsys):
    flaky_embedding(monkeypatch, [genai_errors.ClientError(429, RATE_PER_DAY)])
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path) == 2

    assert sleeps == [] and not result_path(tmp_path).exists() and "明日" in capsys.readouterr().err


@pytest.mark.parametrize("error", [httpx.ConnectError("接続できません"), httpx.ReadTimeout("読み取りが遅すぎます"), TimeoutError("timed out"), ConnectionError("reset")])
def test_an_embedding_connection_problem_waits_30_seconds_and_tries_again(tmp_path, monkeypatch, sleeps, error):
    state = flaky_embedding(monkeypatch, [error])
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path) == 0

    assert sleeps == [30.0] and state["calls"] == 2
    assert load(result_path(tmp_path))["attempts"][0]["source"] == "embedding" and load(result_path(tmp_path))["attempts"][0]["status"] is None


def test_an_embedding_server_error_follows_the_503_waits_and_a_bad_request_stops(tmp_path, monkeypatch, sleeps, capsys):
    server = genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}})
    flaky_embedding(monkeypatch, [server, server])
    use_model(monkeypatch, tool_then_answer()[0])
    assert real_run(tmp_path / "ok") == 0 and sleeps == [30.0, 60.0]

    flaky_embedding(monkeypatch, [genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid"}})])
    assert real_run(tmp_path / "bad") == 1 and sleeps == [30.0, 60.0]  # 400 は待たずに止まる
    assert "embedding" in capsys.readouterr().err and not result_path(tmp_path / "bad").exists()


def test_an_unexpected_exception_in_a_tool_is_not_hidden(tmp_path, monkeypatch):
    flaky_embedding(monkeypatch, [RuntimeError("予想外の不具合")])
    use_model(monkeypatch, tool_then_answer()[0])

    with pytest.raises(RuntimeError, match="予想外の不具合"):
        real_run(tmp_path)  # APIError でも接続の失敗でもないものは、握りつぶさずに、そのまま出す

    assert not result_path(tmp_path).exists()


def test_the_embedding_failure_is_not_one_of_the_types_that_the_tools_catch():
    for caught in (ValueError, FileNotFoundError, sqlite3.Error, genai_errors.APIError):
        assert not issubclass(compare.EmbeddingApiFailure, caught)
    failure = compare.EmbeddingApiFailure(429, "retryDelay: '7s' PerDay")
    assert (failure.status, failure.text, failure.transport) == (429, "retryDelay: '7s' PerDay", False)


def test_the_embedding_is_counted_timed_and_put_back(tmp_path, monkeypatch):
    flaky_embedding(monkeypatch, [])
    patched = retrieval.embed_text
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path) == 0

    record = load(result_path(tmp_path))
    assert record["embedding"]["calls"] == 1 and record["embedding"]["seconds"] >= 0
    assert retrieval.embed_text is patched  # 試行の間だけ包む。終われば、元の関数に戻る
    assert record["retrieved"]["full_text"] and record["retrieved"]["full_text"][0]["page_id"] == "lec02-main-p052"


# --- 再試行: ツールの戻り値の error（埋め込みの包みを通らない関数が返す） ---


def search_failing_then_ok(monkeypatch, errors):
    rows = [{**page_row("lec02-main-p052"), "distance": 0.1}]
    state = {"calls": 0}

    def search(query):
        state["calls"] += 1
        if state["calls"] <= len(errors):
            raise errors[state["calls"] - 1]
        return rows

    monkeypatch.setattr("app.retrieval.search", search)
    return state


def test_an_error_sentence_returned_by_the_tool_waits_30_seconds_and_redoes_the_trial(tmp_path, monkeypatch, sleeps):
    state = search_failing_then_ok(monkeypatch, [genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})])
    model, calls = tool_then_answer()
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 0

    assert sleeps == [30.0] and state["calls"] == 2 and len(calls) == 4  # 試行ごとやり直す（モデルを 2 回ずつ呼ぶ）
    record = load(result_path(tmp_path))
    assert [(a.get("source"), a.get("status"), a.get("action", "ok")) for a in record["attempts"]] == [("tool_return", 503, "retry"), (None, None, "ok")]
    assert record["tool_errors"] == [] and record["retrieved"]["full_text"][0]["page_id"] == "lec02-main-p052"


def test_a_returned_429_waits_the_default_60_seconds_and_a_returned_400_stops(tmp_path, monkeypatch, sleeps, capsys):
    search_failing_then_ok(monkeypatch, [genai_errors.ClientError(429, RATE_PER_MINUTE)])
    use_model(monkeypatch, tool_then_answer()[0])
    assert real_run(tmp_path / "rate") == 0 and sleeps == [60.0]  # 戻り値の文には retryDelay が無い

    search_failing_then_ok(monkeypatch, [genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid"}})])
    assert real_run(tmp_path / "bad") == 1 and sleeps == [60.0]
    assert "tool_return" in capsys.readouterr().err and not result_path(tmp_path / "bad").exists()


def test_a_tool_error_that_the_model_can_fix_is_recorded_and_not_retried(tmp_path, monkeypatch, sleeps):
    model, calls = tool_then_answer(args={"query": ""})  # 空の検索語
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 0

    assert sleeps == [] and len(calls) == 2
    record = load(result_path(tmp_path))
    assert record["status"] == "ok" and [(e["kind"], e["name"], e["error"]) for e in record["tool_errors"]] == [("model", "search_materials", "検索語が空です。")]
    assert [a.get("action", "ok") for a in record["attempts"]] == ["ok"] and record["retrieved"]["errors"][0]["error"] == "検索語が空です。"


def test_a_missing_tree_node_is_a_model_side_error_too(tmp_path, monkeypatch, sleeps):
    model, calls = tool_then_answer(name="browse_tree", args={"node_id": "no-such-node"})
    use_model(monkeypatch, model)

    assert real_run(tmp_path, variant="tree") == 0

    record = load(result_path(tmp_path, "tree"))
    assert sleeps == [] and [e["error"] for e in record["tool_errors"]] == ["対象の目次ノードが見つかりません。"]


@pytest.mark.parametrize("error", [sqlite3.OperationalError("no such table: chunks"), FileNotFoundError("検索DBがありません。")])
def test_a_search_database_problem_stops_the_whole_run(tmp_path, monkeypatch, sleeps, capsys, error):
    search_failing_then_ok(monkeypatch, [error])
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path) == 1

    assert sleeps == [] and "検索の準備ができていません" in capsys.readouterr().err and not result_path(tmp_path).exists()


# --- 再試行: モデルの API（第3・4回と同じ規則） ---


def test_model_rate_limit_is_retried_after_the_advised_delay(tmp_path, monkeypatch, sleeps):
    model, calls = failing_then([http_error(429, RATE_PER_MINUTE)])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 0

    assert sleeps == [7.0] and len(calls) == 2
    record = load(result_path(tmp_path))
    assert record["status"] == "ok" and record["answer"].startswith("答えです")
    assert [(a.get("source"), a.get("action", "ok")) for a in record["attempts"]] == [("model", "retry"), (None, "ok")]
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

    assert sleeps == [60.0] * 5 and not result_path(tmp_path).exists() and "続きから" in capsys.readouterr().err


def test_a_daily_limit_stops_with_exit_code_2_and_saves_nothing(tmp_path, monkeypatch, sleeps, capsys):
    model, calls = failing_then([http_error(429, RATE_PER_DAY)])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 2

    assert sleeps == [] and len(calls) == 1 and not result_path(tmp_path).exists() and "明日" in capsys.readouterr().err


def test_503_is_retried_after_30_60_120_seconds_and_a_500_is_the_same(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, failing_then([http_error(503), http_error(500)])[0])
    assert real_run(tmp_path / "ok") == 0
    assert sleeps == [30.0, 60.0] and len(load(result_path(tmp_path / "ok"))["attempts"]) == 3


def test_503_that_keeps_failing_stops_and_the_next_run_continues(tmp_path, monkeypatch, sleeps, capsys):
    use_model(monkeypatch, failing_then([http_error(503)] * 10)[0])

    assert real_run(tmp_path) == 1

    assert sleeps == [30.0, 60.0, 120.0] and not result_path(tmp_path).exists()
    assert "続きから" in capsys.readouterr().err
    use_model(monkeypatch, text_model())
    assert real_run(tmp_path) == 0 and result_path(tmp_path).exists()


def test_a_timeout_without_a_status_is_retried_like_503(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, failing_then([ModelAPIError("gemini-test", "ReadTimeout")])[0])
    assert real_run(tmp_path) == 0
    assert sleeps == [30.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_400_to_404_stop_without_retrying(tmp_path, monkeypatch, sleeps, capsys, status):
    model, calls = failing_then([http_error(status, {"error": {"message": "API key not valid"}})])
    use_model(monkeypatch, model)

    assert real_run(tmp_path) == 1

    assert sleeps == [] and len(calls) == 1 and not result_path(tmp_path).exists() and "API エラー" in capsys.readouterr().err


def test_usage_limit_and_unexpected_model_behavior_are_saved_as_results(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr("app.retrieval.search", lambda query: [])

    def loop(messages, info):  # 検索を呼び続けて、終わらないモデル
        return ModelResponse(parts=[ToolCallPart("search_materials", {"query": "LoRA"})])

    use_model(monkeypatch, FunctionModel(loop))
    assert real_run(tmp_path / "limit") == 0
    record = load(result_path(tmp_path / "limit"))
    assert record["status"] == "usage_limit" and record["error"]["type"] == "UsageLimitExceeded"
    assert record["answer"] == "" and len(record["tool_calls"]) >= 1 and record["usage"]["requests"] == record["request_limit"] == 20  # 途中までの会話は残る（上限は全条件で 20）
    assert "[usage_limit]" in result_path(tmp_path / "limit", suffix=".trace.txt").read_text(encoding="utf-8")

    def broken(messages, info):
        raise UnexpectedModelBehavior("想定外の応答")

    use_model(monkeypatch, FunctionModel(broken))
    assert real_run(tmp_path / "broken") == 0
    record = load(result_path(tmp_path / "broken"))
    assert record["status"] == "unexpected" and "想定外" in record["error"]["message"] and sleeps == []


# --- 全文投入 ---


def full_model(input_tokens=None, text="講義資料からは確認できませんでした。"):
    usage = RequestUsage(input_tokens=input_tokens, output_tokens=5) if input_tokens else RequestUsage()
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)], usage=usage))


def test_an_input_limit_error_is_saved_as_a_result_and_not_retried(tmp_path, monkeypatch, sleeps):
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "The input token count (1234567) exceeds the maximum number of tokens allowed (1048576)."}}
    model, calls = failing_then([http_error(400, body)])
    use_model(monkeypatch, model)

    assert real_run(tmp_path, variant="full", question="q1") == 0

    record = load(result_path(tmp_path, "full"))
    assert sleeps == [] and len(calls) == 1
    assert record["status"] == "input_limit" and "exceeds the maximum number of tokens" in record["error"]["message"] and record["answer"] == ""
    assert record["materials"]["pages"] == 438 and len(json.dumps(record["messages"])) < 5000  # 資料の本文は入れない
    assert "[input_limit]" in result_path(tmp_path, "full", suffix=".trace.txt").read_text(encoding="utf-8")


def test_another_400_in_full_context_stops_like_any_other_bad_request(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, failing_then([http_error(400, {"error": {"message": "API key not valid"}})])[0])

    assert real_run(tmp_path, variant="full") == 1

    assert not result_path(tmp_path, "full").exists()


def test_the_materials_in_the_saved_messages_are_replaced_by_their_length_and_hash(tmp_path, monkeypatch):
    use_model(monkeypatch, full_model())

    assert real_run(tmp_path, variant="full") == 0

    record = load(result_path(tmp_path, "full"))
    materials = retrieval.get_all_materials()
    prompt = json.dumps(record["messages"], ensure_ascii=False)
    assert f"<資料>…({len(materials)} 字, sha256={compare.sha256_text(materials)})…</資料>" in prompt
    assert len(prompt) < 5000 and "ページ: PDF p." not in prompt
    assert record["materials"]["materials_sha256"] == compare.sha256_text(materials) and record["materials"]["documents"] == SCOPE_ALL
    assert record["materials"]["prompt_chars"] == len(f"<資料>\n{materials}\n</資料>\n\n質問: {record['question']}")
    trace = result_path(tmp_path, "full", suffix=".trace.txt").read_text(encoding="utf-8")
    assert f"[full-context] 入力{record['materials']['prompt_chars']:,}文字" in trace and "[variant] F full" in trace and "[質問] q1" in trace


def test_full_context_runs_through_run_sync_without_the_cli_helper_that_exits(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "run_full_context", lambda *a, **k: pytest.fail("run_full_context は SystemExit するので呼ばない"))
    use_model(monkeypatch, full_model())

    assert real_run(tmp_path, variant="full_aligned") == 0

    record = load(result_path(tmp_path, "full_aligned"))
    assert record["tool_calls"] == [] and record["materials"]["pages"] == 214 and record["materials"]["documents"] == SCOPE_214
    assert record["mode"] == "full" and record["label"] == "F214" and record["instructions"].startswith("あなたはAIエンジニアリング講座の学習支援エージェントです")


# --- TPM ガード ---


def test_the_guard_waits_until_the_budget_is_not_exceeded_with_a_fake_clock(sleeps):
    cfg = compare.load_config()
    ctx = compare.Context(cfg=cfg, out=Path("unused"), model=MODEL, tpm_budget=200000)
    start = sleeps.now
    ctx.recent_inputs = [(start - 50, 60000), (start - 10, 90000)]  # 直近 60 秒の入力は 15 万

    waited = compare.tpm_wait(ctx, 100000)

    assert sleeps == [10.0] and waited == 10.0  # 古いほうが 60 秒を過ぎれば、9 万 + 10 万 ≤ 20 万
    assert ctx.recent_inputs == [(start - 10, 90000)]
    assert compare.tpm_wait(ctx, 100000) == 0.0 and sleeps == [10.0]  # もう待たない


def test_the_guard_waits_in_as_few_steps_as_needed_and_ignores_trials_older_than_60_seconds(sleeps):
    ctx = compare.Context(cfg=compare.load_config(), out=Path("unused"), model=MODEL, tpm_budget=200000)
    ctx.recent_inputs = [(sleeps.now - 70, 190000)]  # 60 秒より前のものは数えない

    assert compare.tpm_wait(ctx, 150000) == 0.0 and sleeps == []
    ctx.recent_inputs = [(sleeps.now - 30, 100000), (sleeps.now - 20, 100000)]
    assert compare.tpm_wait(ctx, 150000) == 40.0 and sleeps == [40.0]  # 2 つとも落ちるまで（古いほうだけでは 10 万 + 15 万 > 20 万）


def test_the_guard_does_not_wait_when_the_estimate_alone_exceeds_the_budget_and_can_be_turned_off(sleeps, capsys):
    ctx = compare.Context(cfg=compare.load_config(), out=Path("unused"), model=MODEL, tpm_budget=100000)
    ctx.recent_inputs = [(sleeps.now, 50000)]

    assert compare.tpm_wait(ctx, 150000) == 0.0 and sleeps == []  # 待っても減らない
    printed = capsys.readouterr().out
    assert "警告" in printed and "150,000" in printed and "100,000" in printed
    off = compare.Context(cfg=compare.load_config(), out=Path("unused"), model=MODEL, tpm_budget=0)
    off.recent_inputs = [(sleeps.now, 10**9)]
    assert compare.tpm_wait(off, 10**9) == 0.0 and sleeps == []


def test_two_full_trials_in_a_row_wait_for_the_60_second_window(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, full_model(input_tokens=150000))

    assert run("run", "--level", "all", "--variant", "full", "--question", "q1", "q2", "--trials", 1, "--allow-dirty", "--interval", 0,
               "--tpm-budget", 200000, "--out", tmp_path) == 0

    first, second = load(result_path(tmp_path, "full", "q1")), load(result_path(tmp_path, "full", "q2"))
    assert sleeps == [60.0]  # 1 回目の 15 万 + 2 回目の見積もり（約 13.7 万）が 20 万を超える
    assert first["usage"]["input_tokens"] == 150000
    assert first["tpm"] == {"budget": 200000, "estimate": first["tpm"]["estimate"], "estimate_source": "chars", "waited_s": 0.0}
    assert abs(first["tpm"]["estimate"] - int(first["materials"]["prompt_chars"] * 0.6)) <= 1 and second["tpm"]["waited_s"] == 60.0


def test_a_search_trial_is_never_held_back_by_the_guard(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, text_model())
    ctx_calls = []
    monkeypatch.setattr(compare, "tpm_wait", lambda ctx, estimate: ctx_calls.append(estimate) or 0.0)

    assert real_run(tmp_path) == 0 and ctx_calls == [] and load(result_path(tmp_path))["tpm"] is None  # full 以外は、ガードを通らない


def test_the_estimate_is_the_median_of_the_recorded_input_tokens_or_the_prompt_length_times_0_6(tmp_path):
    cfg = compare.load_config()
    ctx = compare.Context(cfg=cfg, out=tmp_path, model=MODEL)
    cond = compare.Condition("exercise", "full", "F", "full", "q1", cfg["questions"]["q1"]["text"])

    assert compare.tpm_estimate(ctx, cond, 1000) == (600, "chars")
    for trial, tokens in enumerate([90000, 110000, 100000], 1):
        write_record(tmp_path, "full", "q1", trial, usage={"requests": 1, "input_tokens": tokens})
    assert compare.tpm_estimate(ctx, cond, 1000) == (100000, "records")
    other = compare.Condition("exercise", "full", "F", "full", "q2", "x")
    assert compare.tpm_estimate(ctx, other, 1000) == (600, "chars")  # 条件（質問ごと）の記録だけを見る


def test_the_order_of_conditions_is_rotated_by_order_shift_for_each_trial():
    cfg = compare.load_config()
    order = cfg["defaults"]["order"]

    assert cfg["defaults"]["order_shift"] == 3
    assert compare.trial_order(cfg, 1) == order  # t1 は order のまま
    assert compare.trial_order(cfg, 2) == order[3:] + order[:3] and compare.trial_order(cfg, 3) == order[6:] + order[:6]
    assert [compare.order_shift_count(cfg, k) for k in (1, 2, 3, 4, 5)] == [0, 3, 6, 9, 2]  # 長さ（10）を超えたら、割った余り
    assert all(sorted(compare.trial_order(cfg, k)) == sorted(order) for k in range(1, 8))  # 巡回なので、全部の条件が 1 回ずつ
    assert compare.trial_order(cfg, 2)[:3] == ["full_aligned", "graph", "doc_hint"] and compare.trial_order(cfg, 3)[:3] == ["neighbors", "topk15", "term_graph"]
    plans = [compare.build_plan(cfg, list(cfg["variants"]), None, k) for k in (1, 2, 3)]
    assert [[c.variant for c in plan[:10]] for plan in plans] == [compare.trial_order(cfg, k) for k in (1, 2, 3)]  # q1（全条件を流す質問）の中の順
    assert [c.variant for c in plans[1] if c.qid == "qa1"] == ["graph", "doc_hint", "neighbors", "topk15", "term_graph", "term_graph_cap3", "vector"]  # 流さない条件は飛ばす


def test_full_conditions_are_never_next_to_each_other_in_any_trial_or_at_a_boundary():
    cfg = compare.load_config()
    names = list(cfg["variants"])

    assert [(c.qid, c.variant) for c in compare.build_plan(cfg, names)[:10]] == [("q1", n) for n in cfg["defaults"]["order"]]
    sequence = [c for k in range(1, 8) for c in compare.build_plan(cfg, names, None, k)]  # 7 試行を続けて流したとき（質問の境目・試行の境目も）
    modes = [c.mode for c in sequence]
    assert len(sequence) == 7 * 44 and all(not (a == "full" and b == "full") for a, b in zip(modes, modes[1:]))
    for k in range(1, 8):  # 1 つの試行の中では、先頭と末尾のつながりも
        trial_modes = [c.mode for c in compare.build_plan(cfg, names, None, k)]
        assert all(not (a == "full" and b == "full") for a, b in zip(trial_modes, trial_modes[1:] + trial_modes[:1])), k


# --- 実行の順番・再開・予行 ---


def dry_run(out, *extra, level="all", trials=1):
    return run("run", "--level", level, "--dry-run", "--trials", trials, "--out", out, *extra)


def printed_runs(text):
    return re.findall(r"^\[\d+/\d+\] (\S+) (\S+)/(\S+) t(\d)", text, re.M)


def test_the_order_is_trial_then_question_then_condition_and_the_conditions_rotate_per_trial(tmp_path, capsys):
    assert dry_run(tmp_path, trials=3) == 0

    printed = capsys.readouterr().out
    lines = printed_runs(printed)
    cfg = compare.load_config()
    names = list(cfg["variants"])
    assert len(lines) == 132 and [int(k) for *_, k in lines] == [1] * 44 + [2] * 44 + [3] * 44  # 試行番号が外側
    for k in (1, 2, 3):
        plan = compare.build_plan(cfg, names, None, k)
        assert [(variant, qid) for _, variant, qid, trial in lines if int(trial) == k] == [(c.variant, c.qid) for c in plan]  # 試行 k の順
    assert [qid for _, _, qid, _ in lines[:44]] == ["q1"] * 10 + ["q2"] * 10 + ["q3"] * 10 + ["qa1"] * 7 + ["qa2"] * 7  # 質問の順
    assert [label for label, _, qid, k in lines if qid == "q1"][:10] == ["F", "V0", "T", "F214", "G", "B1", "I1", "I2", "A1", "A2"]  # t1
    assert [label for label, _, qid, k in lines if qid == "q1" and k == "2"] == ["F214", "G", "B1", "I1", "I2", "A1", "A2", "F", "V0", "T"]  # t2: 3 つ巡回
    assert [label for label, _, qid, k in lines if qid == "q1" and k == "3"] == ["I1", "I2", "A1", "A2", "F", "V0", "T", "F214", "G", "B1"]  # t3: 6 つ巡回
    assert "条件の順（質問ごと。試行ごとに 3 個ずつ巡回）:" in printed and "  t2: F214 G B1 I1 I2 A1 A2 F V0 T" in printed


def test_resuming_does_not_change_the_order(tmp_path, capsys):
    assert dry_run(tmp_path / "whole", trials=3) == 0
    whole = printed_runs(capsys.readouterr().out)

    assert dry_run(tmp_path / "split", "--max-runs", 17, trials=3) == 0
    first = printed_runs(capsys.readouterr().out)
    assert dry_run(tmp_path / "split", "--max-runs", 50, trials=3) == 0  # t1 の途中から、t2 の途中まで
    second = printed_runs(capsys.readouterr().out)
    assert dry_run(tmp_path / "split", trials=3) == 0
    rest = printed_runs(capsys.readouterr().out)

    assert [len(first), len(second), len(rest)] == [17, 50, 65]
    assert [(v, q, k) for _, v, q, k in first + second + rest] == [(v, q, k) for _, v, q, k in whole]  # 区切って流しても、順は同じ


def test_each_record_has_its_position_in_that_trial_and_it_does_not_depend_on_the_filters(tmp_path):
    assert dry_run(tmp_path / "all", trials=3) == 0
    model_dir = tmp_path / "all" / f"{MODEL}-dry-run"

    def position(variant, qid, k):
        return load(model_dir / variant / qid / f"t{k}.json")["position"]

    assert position("full", "q1", 1) == {"in_question": 1, "of_question": 10, "in_trial": 1, "of_trial": 44, "order_shift": 0}
    assert position("full", "q1", 2) == {"in_question": 8, "of_question": 10, "in_trial": 8, "of_trial": 44, "order_shift": 3}
    assert position("vector", "q1", 3) == {"in_question": 6, "of_question": 10, "in_trial": 6, "of_trial": 44, "order_shift": 6}
    assert position("vector", "qa1", 2) == {"in_question": 7, "of_question": 7, "in_trial": 37, "of_trial": 44, "order_shift": 3}  # q1〜q3 の 30 件のあと
    assert dry_run(tmp_path / "some", "--variant", "vector", "--question", "q1", "qa1", trials=3) == 0  # 絞り込んで流しても、同じ位置
    some = tmp_path / "some" / f"{MODEL}-dry-run"
    for qid in ("q1", "qa1"):
        for k in (1, 2, 3):
            assert load(some / "vector" / qid / f"t{k}.json")["position"] == position("vector", qid, k)


def test_a_dry_run_runs_every_condition_and_a_second_run_rewrites_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert dry_run(out) == 0

    model_dir = out / f"{MODEL}-dry-run"  # 予行は別のフォルダに書く（本物の結果と混ざらない）
    cfg = compare.load_config()
    plan = compare.build_plan(cfg, list(cfg["variants"]))
    found = {tuple(p.relative_to(model_dir).parts[:2]) for p in model_dir.glob("*/*/t1.json")}
    assert found == {c.key for c in plan} and len(found) == 44 and not (out / MODEL).exists()
    for cond in plan:
        base = model_dir / cond.variant / cond.qid
        assert (base / "t1.trace.txt").exists() and not list(base.glob(".tmp-*"))
    record = load(model_dir / "vector/q1/t1.json")
    for key in ("schema", "lecture", "variant", "label", "level", "mode", "variant_sha256", "qid", "question", "trial", "model", "model_name", "dry_run",
                "command", "compare_args", "git", "versions", "data_sha256", "code_sha256", "question_sha256", "position", "started_at", "finished_at", "duration_s", "elapsed_s", "attempts",
                "status", "error", "answer", "instructions", "instructions_sha256", "tools", "retrieval", "settings", "request_limit",
                "tool_calls", "tool_errors", "approvals", "embedding", "retrieval_trace", "retrieved", "citations", "usage", "cost_usd",
                "thoughts_tokens", "responses", "messages"):
        assert key in record, key
    assert record["dry_run"] is True and record["lecture"] == 5 and record["mode"] == "vector" and "materials" not in record
    assert set(record["versions"]) == {"pydantic-ai-slim", "google-genai", "sqlite-vec"} and set(record["data_sha256"]) == {"lectures.json", "tree.json", "graph.json", "index.sqlite3"}
    assert "materials" in load(model_dir / "full/q1/t1.json") and "[model]" in (model_dir / "vector/q1/t1.trace.txt").read_text(encoding="utf-8")

    snapshot = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert dry_run(out) == 0 and "これから 0 回" in capsys.readouterr().out
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == snapshot  # 2 回目は、何も書き直さない

    assert dry_run(out, trials=2) == 0  # 試行を増やすと、足りない分（t2）だけ増える
    assert len(list(model_dir.glob("*/*/t2.json"))) == 44 and len(list(model_dir.glob("*/*/t1.json"))) == 44
    assert all(p.stat().st_mtime_ns == snapshot[p] for p in snapshot if p.name.startswith("t1"))


def test_the_dry_run_scripts_follow_each_mode(tmp_path):
    assert dry_run(tmp_path) == 0
    model_dir = tmp_path / f"{MODEL}-dry-run"

    def names(variant, qid="q1"):
        return [(c["name"], c["args"]) for c in load(model_dir / variant / qid / "t1.json")["tool_calls"]]

    assert names("full") == [] and names("full_aligned") == []
    assert [n for n, _ in names("vector")] == ["search_materials"] and [n for n, _ in names("graph")] == ["search_graph"]
    tree = names("tree")
    assert [n for n, _ in tree] == ["browse_tree"] * 4 and tree[0][1] == {} and tree[1][1] == {"node_id": "lec02-main"}  # root → 講義 → 章 → ページ
    assert tree[2][1]["node_id"].startswith("lec02-main-s") and tree[3][1]["node_id"].startswith("lec02-main-p")
    vector = load(model_dir / "vector/q1/t1.json")
    last_page = vector["retrieved"]["full_text"][-1]["page_id"]
    assert [c["page_ids"] for c in vector["citations"]] == [[last_page]]  # 取得した最後のページを「第N回・PDF p.N」で引用する
    assert "第" in vector["answer"] and "PDF p." in vector["answer"] and vector["usage"]["requests"] == 2
    assert load(model_dir / "full/q1/t1.json")["usage"]["requests"] == 1 and len(load(model_dir / "tree/q1/t1.json")["retrieved"]["tree_path"]) == 4


def test_the_fake_embedding_is_fixed_by_the_text_and_normalized():
    a = compare.fake_embed_text("LoRA", "RETRIEVAL_QUERY")

    assert a == compare.fake_embed_text("LoRA", "RETRIEVAL_QUERY") and len(a) == 768
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert a != compare.fake_embed_text("QLoRA", "RETRIEVAL_QUERY") and a != compare.fake_embed_text("LoRA", "RETRIEVAL_DOCUMENT")


def test_a_dry_run_never_calls_the_embedding_api_or_the_real_model(tmp_path, monkeypatch):
    monkeypatch.setattr("app.retrieval.embed_text", lambda *a: pytest.fail("予行は本物の埋め込みを呼ばない"))
    monkeypatch.setattr(compare.Context, "google_model", lambda self: pytest.fail("予行は本物のモデルを作らない"))
    monkeypatch.delenv("GEMINI_API_KEY")  # 予行は API キーも要らない

    assert dry_run(tmp_path, "--variant", "V0,G,I1,A1") == 0


def test_a_dry_run_without_out_writes_to_a_temp_folder_and_refuses_the_results_folder(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "auto"
    folder.mkdir()
    monkeypatch.setattr(compare.tempfile, "mkdtemp", lambda prefix="": str(folder))
    assert run("run", "--level", "exercise", "--variant", "vector", "--question", "q1", "--dry-run", "--trials", 1) == 0
    assert f"保存先: {folder}" in capsys.readouterr().out and (folder / f"{MODEL}-dry-run/vector/q1/t1.json").exists()

    assert run("run", "--dry-run", "--out", compare.DEFAULT_OUT / "dry") == 1
    assert "本物の結果と混ざる" in capsys.readouterr().err and not (compare.DEFAULT_OUT / "dry").exists()


def test_max_runs_stops_and_the_next_run_continues(tmp_path):
    out = tmp_path / "out"
    assert dry_run(out, "--level", "exercise", "--variant", "vector", "--max-runs", 2) == 0
    assert len(list(out.glob("*/*/*/t1.json"))) == 2
    assert dry_run(out, "--level", "exercise", "--variant", "vector") == 0
    assert len(list(out.glob("*/*/*/t1.json"))) == 5


def test_interval_is_waited_between_runs_but_not_in_a_dry_run_or_after_the_last_one(tmp_path, monkeypatch, sleeps):
    use_model(monkeypatch, text_model())
    assert run("run", "--level", "all", "--variant", "vector", "--question", "q1", "q2", "q3", "--trials", 1, "--allow-dirty", "--interval", 2, "--out", tmp_path) == 0
    assert sleeps == [2.0, 2.0]
    assert dry_run(tmp_path / "dry", "--interval", 5, "--variant", "vector") == 0 and sleeps == [2.0, 2.0]


def test_ctrl_c_exits_with_130_and_the_next_run_continues(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch, text_model())

    def interrupted(seconds):  # 1 件目を保存したあと、待ち時間の間に Ctrl+C を押した、という状況
        raise KeyboardInterrupt

    monkeypatch.setattr(compare.time, "sleep", interrupted)
    args = ("run", "--level", "all", "--variant", "vector", "--question", "q1", "q2", "--trials", 1, "--allow-dirty", "--out", tmp_path / "out")
    assert run(*args, "--interval", 2) == 130

    assert "続きから" in capsys.readouterr().err
    assert result_path(tmp_path / "out").exists() and not result_path(tmp_path / "out", qid="q2").exists()
    assert not list((tmp_path / "out").rglob(".tmp-*"))  # 半端なファイルは残らない
    monkeypatch.setattr(compare.time, "sleep", lambda seconds: None)
    assert run(*args, "--interval", 2) == 0 and result_path(tmp_path / "out", qid="q2").exists()


# --- 条件の再現: 設定・取得処理・指示 ---


def test_topk15_changes_rag_top_k_only_during_the_trial_and_puts_it_back(tmp_path, monkeypatch):
    seen = []

    def search(query):
        seen.append(retrieval.RAG_TOP_K)
        return []

    monkeypatch.setattr("app.retrieval.search", search)
    use_model(monkeypatch, tool_then_answer()[0])
    assert retrieval.RAG_TOP_K == 5

    assert real_run(tmp_path, variant="topk15") == 0

    assert seen == [15] and retrieval.RAG_TOP_K == 5
    record = load(result_path(tmp_path, "topk15"))
    assert record["settings"] == {"RAG_TOP_K": 15} and record["settings_effective"]["RAG_TOP_K"] == 15
    assert compare.build_effective(compare.load_config(), "vector").settings_effective["RAG_TOP_K"] == 5


def test_a_setting_is_put_back_even_when_the_trial_fails(tmp_path, monkeypatch):
    def search(query):
        raise RuntimeError("検索の中の不具合")

    monkeypatch.setattr("app.retrieval.search", search)
    use_model(monkeypatch, tool_then_answer()[0])

    with pytest.raises(RuntimeError):
        real_run(tmp_path, variant="topk15")

    assert retrieval.RAG_TOP_K == 5
    with pytest.raises(ValueError):
        with compare.patched_settings({"TERM_GRAPH_SECTION_CAP": 3}):
            assert lec05.TERM_GRAPH_SECTION_CAP == 3
            raise ValueError("途中で失敗")
    assert lec05.TERM_GRAPH_SECTION_CAP == 2


@pytest.mark.parametrize("variant", ["neighbors", "term_graph"])
def test_replaced_tools_look_the_same_to_the_model_and_only_the_called_function_differs(variant):
    cfg = compare.load_config()
    vector = compare.build_effective(cfg, "vector")
    eff = compare.build_effective(cfg, variant)

    assert eff.tool_defs == vector.tool_defs  # 名前・説明文・引数のスキーマ・種類
    assert eff.tools[0].requires_approval == vector.tools[0].requires_approval is False
    assert eff.tools[0].function is not vector.tools[0].function
    assert eff.retrieval["search_materials"]["function"] != vector.retrieval["search_materials"]["function"]
    assert eff.instructions == vector.instructions and eff.settings == {} and eff.sha256 != vector.sha256


def test_the_replaced_function_is_the_one_that_gets_called(tmp_path, monkeypatch):
    """Tool の複製の落とし穴の回帰テスト: 差し替えた関数が、モデルのツール呼び出しで本当に呼ばれる。"""
    monkeypatch.setattr("app.retrieval.search", lambda query: pytest.fail("元の検索（search_materials の関数）が呼ばれてはいけない"))
    seen = {}
    monkeypatch.setattr(lec05, "search_with_neighbors", lambda query, window=None: seen.setdefault("neighbors", query) and [])
    monkeypatch.setattr(lec05, "search_term_graph", lambda query, **kw: seen.setdefault("term_graph", query) and [])
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path / "n", variant="neighbors") == 0 and real_run(tmp_path / "t", variant="term_graph") == 0

    assert seen == {"neighbors": "LoRA QLoRA", "term_graph": "LoRA QLoRA"}
    assert load(result_path(tmp_path / "n", "neighbors"))["tool_calls"][0]["name"] == "search_materials"  # モデルから見える名前は、そのまま


def test_replacing_only_the_function_of_a_tool_keeps_calling_the_original_one(monkeypatch):
    """pydantic-ai 2.57.0 の落とし穴: Tool の function だけを dataclasses.replace で替えても、function_schema が元の関数を持つので、元の関数が呼ばれ続ける。"""
    base = compare.registered_tools()["search_materials"]
    called = {"original": 0, "other": 0}

    def original(query: str) -> str:
        called["original"] += 1
        return json.dumps({"results": []})

    def other(query: str) -> str:
        called["other"] += 1
        return json.dumps({"results": []})

    def run_with(tool):
        model, _ = tool_then_answer()
        with agent.override(tools=[tool]):
            agent.run_sync("q", model=model)

    run_with(dataclasses.replace(Tool(original, name=base.name, description=base.description), function=other))
    assert called == {"original": 1, "other": 0}  # 落とし穴
    run_with(Tool(other, name=base.name, description=base.description))  # 作り直せば、替えた関数が呼ばれる（compare.py はこちら）
    assert called == {"original": 1, "other": 1}


def test_the_retrieval_trace_of_the_replaced_tools_is_copied_into_the_record_and_cleared_for_each_trial(tmp_path, monkeypatch, embed_stored_page):
    use_model(monkeypatch, tool_then_answer()[0])
    lec05.TRACE.append({"tool": "left_over"})  # 前の試行の残り

    assert real_run(tmp_path, variant="neighbors", trials=2) == 0

    for k in (1, 2):
        record = load(result_path(tmp_path, "neighbors", trial=k))
        assert [e["tool"] for e in record["retrieval_trace"]] == ["neighbors"]  # 試行ごとに空にしてから実行する
        assert record["retrieval_trace"][0]["hits"][0] == "lec02-main-p052" and record["retrieval_trace"][0]["window"] == 1
        roles = {r["via"] for r in record["retrieved"]["full_text"]}
        assert roles == {"hit", "neighbor"} and len(record["retrieved"]["full_text"]) == record["retrieval_trace"][0]["pages"]
    trace = result_path(tmp_path, "neighbors", suffix=".trace.txt").read_text(encoding="utf-8")
    assert "--- 取得の内訳 ---" in trace and "[neighbors] query='LoRA QLoRA' window=1" in trace
    assert real_run(tmp_path / "t", variant="term_graph", question="q1") == 0
    entry = load(result_path(tmp_path / "t", "term_graph"))["retrieval_trace"][0]
    assert entry["tool"] == "term_graph" and entry["params"] == {"top_k": 5, "cap": 2, "vector_pool": 5, "hub_terms": 9}
    assert "[term_graph]" in result_path(tmp_path / "t", "term_graph", suffix=".trace.txt").read_text(encoding="utf-8")


def test_the_cap_setting_reaches_the_term_graph_trace_and_goes_back(tmp_path, monkeypatch, embed_stored_page):
    use_model(monkeypatch, tool_then_answer()[0])

    assert real_run(tmp_path, variant="term_graph_cap3") == 0

    entry = load(result_path(tmp_path, "term_graph_cap3"))["retrieval_trace"][0]
    assert entry["params"]["cap"] == 3 and lec05.TERM_GRAPH_SECTION_CAP == 2
    assert load(result_path(tmp_path, "term_graph_cap3"))["settings"] == {"TERM_GRAPH_SECTION_CAP": 3}


def changed_sections(info, baseline_text):
    """基準との差分がある節（[instructions]・[tools]・[取得処理]・[設定]・[対象資料]・[上限]）。"""
    def split(text):
        sections, name = {}, None
        for line in text.splitlines():
            match = re.match(r"\[(instructions|tools|取得処理|設定|対象資料|上限)\]", line)
            if match:
                name = match.group(1)
            sections.setdefault(name, []).append(line)
        return sections

    new, old = split(info["text"]), split(baseline_text)
    return {name for name in new if new[name] != old.get(name)} | {name for name in old if name not in new}


def test_each_variant_differs_from_its_baseline_in_exactly_one_section_and_one_hunk():
    cfg = compare.load_config()
    diffs = compare.variant_diffs(cfg)

    expected = {"doc_hint": {"tools"}, "neighbors": {"取得処理"}, "term_graph": {"取得処理"}, "topk15": {"設定"}, "term_graph_cap3": {"設定"}, "full_aligned": {"対象資料"}}
    for name, sections in expected.items():
        info = diffs[name]
        assert info["hunks"] == 1, name
        assert changed_sections(info, diffs[info["baseline"]]["text"]) == sections, name
    assert [diffs[n]["baseline"] for n in expected] == ["vector", "vector", "vector", "vector", "term_graph", "full"]
    assert diffs["vector"]["diff"] == [] and diffs["full"]["diff"] == [] and diffs["tree"]["baseline"] is None and diffs["graph"]["baseline"] is None
    assert len({info["effective"].sha256 for info in diffs.values()}) == len(diffs)


def test_the_variants_command_saves_the_files_and_prints_the_differences(tmp_path, capsys):
    out = tmp_path / "out"

    assert run("variants", "--out", out) == 0

    printed = capsys.readouterr().out
    assert "I2 topk15: vector との差分" in printed and "-RAG_TOP_K = 5" in printed and "+RAG_TOP_K = 15" in printed
    assert "F214 full_aligned: full との差分" in printed and "基準なし" in printed
    assert (out / "variants" / "neighbors.txt").exists()
    saved = load(out / "variants" / "term_graph_cap3.json")
    assert saved["sha256"] == compare.build_effective(compare.load_config(), "term_graph_cap3").sha256 and saved["base"] == "term_graph"
    assert saved["settings"] == {"TERM_GRAPH_SECTION_CAP": 3} and saved["retrieval"] == {"search_materials": "app.retrieval_lec05:search_materials_term_graph"}
    assert saved["hunks"] == 1 and saved["tools"][0]["name"] == "search_materials" and saved["instructions"]
    text = (out / "variants" / "vector.txt").read_text(encoding="utf-8")
    for section in ("[instructions]", "[tools]", "[取得処理]", "[設定]", "[対象資料]", "[上限]"):
        assert section in text
    assert "- search_materials: app.tools:search_materials" in text and "request_limit = 20" in text and "合計 214 ページ" in text


def test_the_variants_command_warns_when_a_variant_differs_in_two_places(tmp_path, capsys):
    two = replaced(conditions_text(), 'questions = ["q1", "q2", "q3", "qa1", "qa2"]\n\n[variants.neighbors]',
                   'settings = { RAG_TOP_K = 7 }\nquestions = ["q1", "q2", "q3", "qa1", "qa2"]\n\n[variants.neighbors]')  # 説明文と設定の 2 か所
    path = write_conditions(tmp_path, two)

    assert run("variants", "--conditions", path, "--variant", "doc_hint", "--out", tmp_path / "out") == 1

    assert "差分が 2 か所" in capsys.readouterr().out


def test_the_instructions_the_model_saw_are_recorded_and_a_mismatch_is_flagged(tmp_path, monkeypatch):
    use_model(monkeypatch, text_model())
    assert real_run(tmp_path / "a", variant="tree") == 0
    seen = load(result_path(tmp_path / "a", "tree"))
    from app.agent import TREE_INSTRUCTIONS

    assert seen["instructions"].strip() == TREE_INSTRUCTIONS.strip() and seen["instructions_match"] is True
    assert seen["instructions_sha256"] == compare.sha256_text(seen["instructions"]) and seen["tools"][0]["name"] == "browse_tree"

    real_override = agent.override
    monkeypatch.setattr(agent, "override", lambda **kw: real_override(**{k: v for k, v in kw.items() if k != "instructions"}))  # 指示の上書きだけが効かなかった
    assert real_run(tmp_path / "b", variant="tree") == 0
    record = load(result_path(tmp_path / "b", "tree"))
    assert record["instructions_match"] is False
    assert "search_materials" in record["instructions"] and "browse_tree" not in record["instructions"]  # 実際に見えた指示（基本形）を記録している


# --- 実行の前の確認（API の前に、この順で） ---

_CFG = {}


def cfg_of():
    if not _CFG:
        _CFG.update(compare.load_config())
    return _CFG


def write_record(out, variant, qid, trial, model=MODEL, **fields):
    """集計・status・TPM 見積もり用の、記録（t<k>.json）を 1 つ書く。足りないキーには、既定値を入れる。"""
    cfg = cfg_of()
    spec = cfg["variants"][variant]
    record = {
        "schema": 1, "variant": variant, "label": spec["label"], "level": spec["level"], "mode": compare.resolve_spec(cfg, variant)["mode"],
        "qid": qid, "trial": trial, "status": "ok", "duration_s": 1.0, "answer": "", "tool_calls": [], "tool_errors": [],
        "attempts": [{"attempt": 1, "ok": True}], "usage": {"requests": 1, "input_tokens": 100, "output_tokens": 10, "tool_calls": 0},
        "cost_usd": None, "thoughts_tokens": 0, "instructions_match": True, **fields,
    }
    path = result_path(out, variant, qid, trial, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def test_git_state_looks_only_at_the_guarded_paths(tmp_path):
    repo = tmp_path / "repo"
    for name in ("app/agent.py", "data/rag_data/lectures.json", "lectures/05/compare.py", "lectures/05/conditions.toml", "pyproject.toml", "uv.lock",
                 "README.md", "lectures/05/answer.md", "lectures/03/compare.py"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("元\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")

    clean = compare.git_state(repo)
    assert clean["dirty"] is False and len(clean["commit"]) == 40 and clean["dirty_files"] == []

    (repo / "README.md").write_text("関係ない変更\n", encoding="utf-8")
    (repo / "lectures/05/answer.md").write_text("関係ない変更\n", encoding="utf-8")
    (repo / "lectures/03/compare.py").write_text("第3回のファイル\n", encoding="utf-8")
    (repo / "lectures/05/results").mkdir()
    (repo / "lectures/05/results/summary.md").write_text("結果\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is False  # 守るパス以外は見ない（結果と answer.md を書いても、止まらない）

    (repo / "app/agent.py").write_text("変更\n", encoding="utf-8")
    state = compare.git_state(repo)
    assert state["dirty"] is True and any("app/agent.py" in line for line in state["dirty_files"])
    git(repo, "checkout", "--", "app/agent.py")
    (repo / "data/rag_data/new.json").write_text("新規\n", encoding="utf-8")
    assert compare.git_state(repo)["dirty"] is True  # 追加しただけの新規ファイルも、未コミットの変更
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "more")
    assert compare.git_state(repo)["dirty"] is False


@pytest.mark.parametrize("name", ["app/agent.py", "data/rag_data/lectures.json", "lectures/05/compare.py", "lectures/05/conditions.toml", "pyproject.toml", "uv.lock"])
def test_a_change_in_any_guarded_path_is_an_uncommitted_change(tmp_path, name):
    assert name in compare.GUARDED_PATHS or name.split("/")[0] in compare.GUARDED_PATHS or name.rsplit("/", 1)[0] in compare.GUARDED_PATHS
    repo = tmp_path / "repo"
    for path in ("app/agent.py", "data/rag_data/lectures.json", "lectures/05/compare.py", "lectures/05/conditions.toml", "pyproject.toml", "uv.lock"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text("元\n", encoding="utf-8")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    (repo / name).write_text("変更\n", encoding="utf-8")

    assert compare.git_state(repo)["dirty"] is True


def test_a_failed_git_status_is_not_called_clean_and_git_never_waits_for_input(monkeypatch):
    def fake(repo, *args):
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "a" * 40 + "\n", "")
        return subprocess.CompletedProcess(args, 124, "", "git が 60 秒で終わりませんでした")

    with monkeypatch.context() as m:
        m.setattr(compare, "run_git", fake)
        state = compare.git_state(Path("."))
    assert state["dirty"] is True and state["commit"] == "a" * 40 and "失敗" in state["dirty_files"][0]

    seen = {}

    def hang(cmd, **kwargs):
        seen.update(kwargs, cmd=cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(compare.subprocess, "run", hang)
    result = compare.run_git(Path("/nonexistent"), "status")
    assert result.returncode == 124 and "終わりませんでした" in result.stderr
    assert seen["stdin"] == subprocess.DEVNULL and seen["timeout"] == compare.GIT_TIMEOUT
    assert seen["env"]["GIT_OPTIONAL_LOCKS"] == "0" and seen["env"]["GIT_TERMINAL_PROMPT"] == "0"  # index を書き換えず、入力を待たない


def dirty_git(monkeypatch, files=(" M app/agent.py",)):
    monkeypatch.setattr(compare, "git_state", lambda *a, **k: {"commit": "abc", "dirty": True, "dirty_files": list(files)})


def test_uncommitted_changes_stop_a_real_run_unless_allowed_or_a_dry_run(tmp_path, monkeypatch, capsys):
    dirty_git(monkeypatch)
    use_model(monkeypatch, text_model())
    base = ("run", "--level", "all", "--variant", "vector", "--question", "q1", "--trials", 1, "--interval", 0)

    assert run(*base, "--out", tmp_path / "a") == 1
    err = capsys.readouterr().err
    assert "app/agent.py" in err and "--allow-dirty" in err and "commit してから" in err and not (tmp_path / "a").exists()
    assert run(*base, "--allow-dirty", "--out", tmp_path / "b") == 0
    assert load(result_path(tmp_path / "b"))["git"] == {"commit": "abc", "dirty": True}  # 記録には、そのまま残る
    assert run(*base, "--dry-run", "--out", tmp_path / "c") == 0  # 予行は、確認を飛ばす


def test_the_dirty_message_explains_untracked_files_only_when_there_are_some(monkeypatch, tmp_path, capsys):
    note = "?? は git が追跡していないファイルです"
    args = ("run", "--level", "all", "--variant", "vector", "--question", "q1", "--trials", 1, "--out", tmp_path / "x")

    dirty_git(monkeypatch, ("?? app/retrieval_lec05.py",))
    assert run(*args) == 1  # 追跡していないファイルがあっても、確認は厳しいまま（止まる）
    err = capsys.readouterr().err
    assert note in err and "?? app/retrieval_lec05.py" in err
    dirty_git(monkeypatch, (" M app/agent.py", " M uv.lock"))
    assert run(*args) == 1
    err = capsys.readouterr().err
    assert note not in err and "uv.lock" in err and not (tmp_path / "x").exists()


def test_aie_rag_db_stops_the_run_even_in_a_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIE_RAG_DB", str(tmp_path / "other.sqlite3"))
    use_model(monkeypatch, text_model())

    assert real_run(tmp_path / "a") == 1
    assert "AIE_RAG_DB" in capsys.readouterr().err
    assert run("run", "--dry-run", "--out", tmp_path / "b") == 1 and not (tmp_path / "b").exists()


def make_db(path, rows, documents=3):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE chunks (page_id TEXT PRIMARY KEY, document_id TEXT, lecture INTEGER, document_title TEXT, section TEXT, title TEXT, page INTEGER, content TEXT, embedding BLOB)")
    db.executemany("INSERT INTO chunks VALUES (?, ?, 1, 't', 's', 't', ?, 'c', NULL)", [(f"p{i}", f"lec0{i % documents + 1}-main", i) for i in range(rows)])
    db.commit()
    db.close()
    return path


def test_the_search_database_must_open_and_have_214_rows_and_3_documents(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch, text_model())
    monkeypatch.setattr(retrieval, "RAG_DB", tmp_path / "none.sqlite3")
    assert real_run(tmp_path / "missing") == 1
    assert "検索 DB を開けません" in capsys.readouterr().err and not (tmp_path / "missing").exists()

    monkeypatch.setattr(retrieval, "RAG_DB", make_db(tmp_path / "short.sqlite3", 10))
    assert real_run(tmp_path / "short") == 1
    err = capsys.readouterr().err
    assert "10 行" in err and "214 行" in err and "3 資料" in err

    monkeypatch.setattr(retrieval, "RAG_DB", make_db(tmp_path / "two.sqlite3", 214, documents=2))
    assert real_run(tmp_path / "two") == 1 and "2 資料" in capsys.readouterr().err

    monkeypatch.setattr(retrieval, "RAG_DB", tmp_path / "none.sqlite3")  # 検索を使わない条件（全文投入・Tree）は、DB が無くても動く
    assert real_run(tmp_path / "tree", variant="tree") == 0 and real_run(tmp_path / "full", variant="full") == 0
    assert real_run(tmp_path / "graph", variant="graph") == 1  # Graph は検索を使う

    shutil_copy = tmp_path / "copy.sqlite3"
    import shutil

    shutil.copyfile(compare.REPO / "data/rag_data/index.sqlite3", shutil_copy)
    monkeypatch.setattr(retrieval, "RAG_DB", shutil_copy)
    compare.check_index()  # 214 行・3 資料なら通る


def test_a_changed_condition_stops_the_run_instead_of_mixing_trials(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    args = ["run", "--level", "all", "--variant", "vector", "--question", "q1", "--out", out]
    assert run(*args, "--dry-run", "--trials", 1) == 0
    path = result_path(out, model=f"{MODEL}-dry-run")

    with monkeypatch.context() as m:
        m.setattr(retrieval, "RAG_TOP_K", 6)  # 途中で app/ の設定が変わった、という状況
        assert run(*args, "--dry-run", "--trials", 2) == 1
    err = capsys.readouterr().err
    assert "sha256" in err and "別の --out" in err and str(path) in err
    assert not result_path(out, trial=2, model=f"{MODEL}-dry-run").exists()
    assert run(*args, "--dry-run", "--trials", 2) == 0  # 戻せば、続きから動く

    record = load(path)
    record["variant_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    assert run(*args, "--dry-run", "--trials", 3) == 1 and "sha256" in capsys.readouterr().err


Q1_TEXT = "第2回の説明では、LoRAは何を固定し何を学習しますか？"


def test_a_changed_question_text_stops_the_run_instead_of_mixing_trials(tmp_path, capsys):
    out = tmp_path / "out"
    base = ["run", "--level", "all", "--variant", "vector", "--question", "q1", "--out", out]
    assert run(*base, "--dry-run", "--trials", 1) == 0
    path = result_path(out, model=f"{MODEL}-dry-run")
    record = load(path)
    assert record["question_sha256"] == compare.sha256_text(record["question"]) == compare.sha256_text(compare.load_config()["questions"]["q1"]["text"])

    changed = write_conditions(tmp_path, replaced(conditions_text(), Q1_TEXT, Q1_TEXT + " 簡潔に答えてください。"), "changed.toml")  # 質問の text だけを変える
    assert compare.build_effective(compare.load_config(changed), "vector").sha256 == record["variant_sha256"]  # 条件の sha256 は同じ（これだけでは見つからない）
    assert run(*base, "--conditions", changed, "--dry-run", "--trials", 2) == 1

    err = capsys.readouterr().err
    assert "質問の文が、いまの conditions.toml の q1 と違います" in err and "別の --out" in err and str(path) in err
    assert not result_path(out, trial=2, model=f"{MODEL}-dry-run").exists()
    assert run(*base, "--dry-run", "--trials", 2) == 0  # 元の文なら、続きから流せる


def test_only_the_question_that_changed_stops_the_run_and_old_records_are_compared_by_their_question_text(tmp_path, capsys):
    out = tmp_path / "out"
    base = ["run", "--level", "all", "--variant", "vector", "--question", "q1", "--out", out]
    assert run(*base, "--dry-run", "--trials", 1) == 0
    q2_changed = write_conditions(tmp_path, replaced(conditions_text(), "第3回の例では、コンパクション", "第3回の例では、コンパクションと要約"), "q2.toml")

    assert run(*base, "--conditions", q2_changed, "--dry-run", "--trials", 2) == 0  # q2 の文を変えても、q1 の記録とは関係ない

    path = result_path(out, model=f"{MODEL}-dry-run")
    record = load(path)
    del record["question_sha256"]  # question_sha256 の無い古い記録は、question の文で比べる
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert run(*base, "--dry-run", "--trials", 3) == 0
    record["question"] = "別の質問です"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert run(*base, "--dry-run", "--trials", 3) == 1 and "質問の文" in capsys.readouterr().err


def test_the_summary_warns_about_records_whose_question_differs_from_conditions_toml(tmp_path):
    write_record(tmp_path, "vector", "q1", 1, question=compare.load_config()["questions"]["q1"]["text"])
    write_record(tmp_path, "vector", "q1", 2, question="古い質問の文です")

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    warnings = (tmp_path / "summary.md").read_text(encoding="utf-8").split("## 警告")[1].split("## モデル")[0]
    assert "記録の質問の文が、いまの conditions.toml と違う試行（質問を変えたあとは、別の --out に書く）: 1 件 — V0 vector/q1 t2" in warnings


def test_only_the_selected_conditions_are_checked_against_their_stored_records(tmp_path):
    out = tmp_path / "out"
    assert run("run", "--level", "all", "--variant", "vector", "--question", "q1", "--dry-run", "--trials", 1, "--out", out) == 0
    path = result_path(out, model=f"{MODEL}-dry-run")
    record = load(path)
    record["variant_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    assert run("run", "--level", "all", "--variant", "graph", "--question", "q1", "--dry-run", "--trials", 1, "--out", out) == 0  # 別の条件は、邪魔されない


def test_an_unreadable_stored_record_stops_the_run_with_a_message(tmp_path, capsys):
    out = tmp_path / "out"
    path = result_path(out, model=f"{MODEL}-dry-run")
    path.parent.mkdir(parents=True)
    path.write_text("{壊れた", encoding="utf-8")

    assert run("run", "--level", "all", "--variant", "vector", "--question", "q1", "--dry-run", "--trials", 1, "--out", out) == 1

    assert "読めません" in capsys.readouterr().err


def test_the_real_model_is_the_one_app_agent_builds_and_is_made_once(monkeypatch):
    from app import agent as agent_module

    ctx = compare.Context(cfg=cfg_of(), out=Path("unused"), model=MODEL)
    made = []
    with monkeypatch.context() as m:
        m.setattr(agent_module, "build_model", lambda: made.append(1) or "モデル")
        assert ctx.google_model() == "モデル" and ctx.google_model() == "モデル" and made == [1]
    real = compare.Context(cfg=cfg_of(), out=Path("unused"), model=MODEL).google_model()
    assert type(real).__name__ == "GoogleModel" and real.model_name == MODEL  # 本物のモデル。ここでは、まだ API を呼ばない


def test_an_error_message_that_contains_the_key_is_masked_before_it_is_printed(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch, failing_then([http_error(400, {"error": {"message": f"bad key {KEY}"}})])[0])

    assert real_run(tmp_path) == 1

    err = capsys.readouterr().err
    assert KEY not in err and "bad key ***" in err


def test_a_missing_api_key_stops_a_real_run_but_not_a_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY")
    use_model(monkeypatch, text_model())

    assert real_run(tmp_path / "real") == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err and not (tmp_path / "real").exists()
    assert run("run", "--level", "all", "--variant", "vector", "--question", "q1", "--dry-run", "--trials", 1, "--out", tmp_path / "dry") == 0


@pytest.mark.parametrize(("option", "value"), [("--question", "q9"), ("--variant", "vectr"), ("--variant", "V9"), ("--level", "advanced")])
@pytest.mark.parametrize("command", ["run", "status", "summarize", "variants"])
def test_a_misspelled_filter_is_reported_with_the_valid_names(tmp_path, capsys, command, option, value):
    extra = ["--dry-run"] if command == "run" else []

    assert run(command, *extra, option, value, "--out", tmp_path / "out") == 1

    err = capsys.readouterr().err
    assert value in err and ("使えるもの" in err)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_conditions_are_picked_by_name_or_label_and_levels_by_commas_or_spaces():
    cfg = cfg_of()

    def select(**kw):
        return compare.select_variants(cfg, SimpleNamespace(level=kw.get("level"), variant=kw.get("variant"), question=None))

    assert select(variant=["F214,vector"]) == ["full_aligned", "vector"]  # 名前か label。順番は conditions.toml の順
    assert select(variant=["v0", "i1"]) == ["vector", "neighbors"]  # 大文字と小文字は区別しない
    assert select(level=["basic,impl"]) == ["doc_hint", "neighbors", "topk15"] and select(level=["basic", "adv"]) == ["doc_hint", "term_graph", "term_graph_cap3"]
    assert select(level=["exercise"]) == ["full", "full_aligned", "vector", "tree", "graph"] and select(level=["all"]) == list(cfg["variants"])
    assert select(level=["adv"], variant=["V0"]) == []  # 絞り込みは重ねて効く


def test_the_script_runs_from_the_command_line():
    done = subprocess.run([sys.executable, str(HERE / "compare.py"), "status", "--level", "basic"], capture_output=True, text=True, cwd=HERE.parents[1],
                          stdin=subprocess.DEVNULL, timeout=120)

    assert done.returncode == 0 and "B1 doc_hint" in done.stdout and "残りの推定リクエスト" in done.stdout


@pytest.mark.parametrize("command", ["status", "variants", "run", "summarize"])
def test_every_subcommand_has_help(capsys, command):
    with pytest.raises(SystemExit) as stopped:
        run(command, "--help")

    assert stopped.value.code == 0 and "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("option", ["--trials", "--max-runs", "--interval", "--tpm-budget"])
def test_a_negative_or_zero_count_is_rejected(option, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        run("run", "--dry-run", option, "-1" if option in ("--interval", "--tpm-budget") else "0", "--out", tmp_path)

    assert stopped.value.code == 2


def test_the_decider_denies_every_approval_and_records_it():
    log = []
    call = SimpleNamespace(tool_call_id="c1", tool_name="save_learning_log", args='{"title": "x"}', args_as_dict=lambda: {"title": "x"})

    assert compare.make_decider(log)(call) is False
    assert log == [{"index": 1, "tool_call_id": "c1", "tool_name": "save_learning_log", "args": {"title": "x"}, "decision": "denied"}]


# --- 秘密の置き換えと、書き込み ---


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

    assert compare.save_trial(tmp_path / "t1", compare.TrialOutput({"answer": "secret-gemini-value", "x": [shaped]}, f"trace secret-gemini-value {shaped}"))

    files = sorted(tmp_path.iterdir())
    assert [p.name for p in files] == ["t1.json", "t1.trace.txt"]  # 一時ファイルが残らない
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "secret-gemini-value" not in text and shaped not in text and "***" in text


def test_a_real_run_never_writes_the_key_anywhere(tmp_path, monkeypatch):
    shaped = "AIza" + "z" * 35
    use_model(monkeypatch, text_model(f"キーは {KEY} と {shaped} です。\n第2回・PDF p.52"))

    assert real_run(tmp_path) == 0
    assert run("variants", "--out", tmp_path) == 0 and run("summarize", "--out", tmp_path) == 0

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) > 10
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert KEY not in text and shaped not in text, path
    assert "***" in result_path(tmp_path).read_text(encoding="utf-8")


def test_a_result_is_written_trace_first_and_json_last_and_never_replaced(tmp_path):
    stem = tmp_path / "vector" / "q1" / "t1"
    assert compare.save_trial(stem, compare.TrialOutput({"n": 1}, "最初の trace")) is True
    assert compare.save_trial(stem, compare.TrialOutput({"n": 2}, "後から来た trace")) is False  # 同時に動いた別の実行の結果を守る

    assert load(stem.with_name("t1.json")) == {"n": 1} and stem.with_name("t1.trace.txt").read_text(encoding="utf-8") == "最初の trace"
    assert sorted(p.name for p in stem.parent.iterdir()) == ["t1.json", "t1.trace.txt"]
    stale = tmp_path / "vector" / "q1" / "t2.trace.txt"
    stale.write_text("前の実行の書きかけ", encoding="utf-8")  # trace だけ残っていても（json は無い）、置き直せる
    assert compare.save_trial(stale.with_name("t2"), compare.TrialOutput({"n": 3}, "新しい trace")) is True
    assert stale.read_text(encoding="utf-8") == "新しい trace"


def test_a_result_that_appeared_meanwhile_is_not_replaced_and_the_run_goes_on(tmp_path, monkeypatch, capsys):
    use_model(monkeypatch, text_model())
    real_trial = compare.run_trial

    def racing(ctx, cond, k):
        output = real_trial(ctx, cond, k)
        if cond.qid == "q1":  # 実行している間に、同時に動いている別の実行が、同じ結果を先に書いた
            compare.write_json_atomic(result_path(tmp_path), {"marker": "別の実行の結果"})
        return output

    monkeypatch.setattr(compare, "run_trial", racing)

    assert run("run", "--level", "all", "--variant", "vector", "--question", "q1", "q2", "--trials", 1, "--allow-dirty", "--interval", 0, "--out", tmp_path) == 0

    assert load(result_path(tmp_path)) == {"marker": "別の実行の結果"}  # 置き換えない
    assert load(result_path(tmp_path, qid="q2"))["status"] == "ok"  # 続きの条件は、そのまま保存する
    captured = capsys.readouterr()
    assert "置き換えずに飛ばします" in captured.err and "vector/q1 t1" in captured.err and not list(tmp_path.rglob(".tmp-*"))


def test_an_exclusive_write_never_replaces_an_existing_file_even_without_hard_links(tmp_path, monkeypatch):
    path = tmp_path / "a" / "t1.json"
    assert compare.write_json_atomic(path, {"n": 1}, exclusive=True) is True
    assert compare.write_json_atomic(path, {"n": 2}, exclusive=True) is False and load(path) == {"n": 1}
    assert compare.write_json_atomic(path, {"n": 3}) is True and load(path) == {"n": 3}  # exclusive でなければ、置き換える

    def no_link(src, dst):
        raise PermissionError("このファイルシステムではハードリンクを作れません")

    monkeypatch.setattr(compare.os, "link", no_link)
    other = tmp_path / "a" / "t2.json"
    assert compare.write_json_atomic(other, {"n": 1}, exclusive=True) is True
    assert compare.write_json_atomic(other, {"n": 2}, exclusive=True) is False and load(other) == {"n": 1}
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == ["t1.json", "t2.json"]  # 一時ファイルは残らない


def test_a_failed_write_leaves_neither_a_temp_file_nor_a_half_written_result(tmp_path, monkeypatch):
    path = tmp_path / "a" / "t1.json"
    compare.write_json_atomic(path, {"n": 1})

    def boom(src, dst):
        raise OSError("置き換えに失敗")

    monkeypatch.setattr(compare.os, "replace", boom)
    with pytest.raises(OSError, match="置き換えに失敗"):
        compare.write_json_atomic(path, {"n": 2})

    assert load(path) == {"n": 1} and sorted(p.name for p in path.parent.iterdir()) == ["t1.json"]  # 元のまま。書きかけも残らない


def test_temp_files_have_unique_names_in_the_same_folder_and_a_stale_one_is_left_alone(tmp_path, monkeypatch):
    folder = tmp_path / "a"
    folder.mkdir()
    stale = folder / ".tmp-stale"
    stale.write_text("前の実行の書きかけ", encoding="utf-8")
    seen, real_replace = [], compare.os.replace

    def spy(src, dst):
        assert Path(src).parent == folder  # 同じフォルダ（同じファイルシステム）に書く
        seen.append(Path(src).name)
        real_replace(src, dst)

    monkeypatch.setattr(compare.os, "replace", spy)
    compare.write_json_atomic(folder / "t1.json", {"n": 1})
    compare.write_json_atomic(folder / "t2.json", {"n": 2})

    assert len(seen) == 2 and seen[0] != seen[1] and all(name.startswith(".tmp-") and name != ".tmp-stale" for name in seen)
    assert stale.read_text(encoding="utf-8") == "前の実行の書きかけ" and sorted(p.name for p in folder.iterdir()) == [".tmp-stale", "t1.json", "t2.json"]


def test_temp_files_are_not_counted_as_results(tmp_path, monkeypatch):
    use_model(monkeypatch, text_model())
    assert real_run(tmp_path) == 0
    folder = result_path(tmp_path).parent
    (folder / ".tmp-abc123").write_text("{書きかけ", encoding="utf-8")
    (folder / ".tmp-t2.json").write_text("{書きかけ", encoding="utf-8")  # t<k>.json に見えても、.tmp- で始まるものは数えない
    cfg = cfg_of()
    ctx = compare.Context(cfg=cfg, out=tmp_path, model=MODEL)
    cond = compare.Condition("exercise", "vector", "V0", "vector", "q1", "x")

    assert [p.name for p in compare.existing_trials(ctx, cond)] == ["t1.json"] and not compare.trial_done(ctx, cond, 2)
    assert real_run(tmp_path, trials=2) == 0 and result_path(tmp_path, trial=2).exists()  # 書きかけに邪魔されず、続きの試行を保存できる


# --- 集計 ---


def retrieval_call(order, rows, query="コンパクションとメモリ"):
    return {"order": order, "response_index": order, "tool_call_id": f"c{order}", "name": "search_materials", "args": {"query": query},
            "result": json.dumps({"results": rows}, ensure_ascii=False), "outcome": "success"}


GOOD_Q2 = ("コンパクションは古い履歴を要約に置き換えます。メモリは外部の保存先に書き出して、次回読み戻します。"
           "日々のメモは基準を超えたものだけが長期メモリに移ります。\n\n根拠: 第3回・PDF p.13、39、40")


def write_q2_trials(out, model=MODEL):
    """vector の q2 を 3 回分（t1: 全部取れて全部引用して正答／t2: 全部取れたが 1 つだけ引用／t3: 2 つしか取れず、gold でないページを引用）。"""
    pages = lambda numbers: [page_row(f"lec03-main-p{n:03d}", distance=0.1 * i) for i, n in enumerate(numbers, 1)]  # noqa: E731
    write_record(out, "vector", "q2", 1, model, answer=GOOD_Q2, tool_calls=[retrieval_call(1, pages([13, 39, 40, 10, 11]))], duration_s=2.0,
                 usage={"requests": 2, "input_tokens": 1000, "output_tokens": 100, "tool_calls": 1}, thoughts_tokens=10, cost_usd=0.0001)
    write_record(out, "vector", "q2", 2, model, answer="古い履歴を要約に置き換えます。\n\n根拠: 第3回・PDF p.13", tool_calls=[retrieval_call(1, pages([13, 39, 40, 10, 11]))],
                 duration_s=4.0, usage={"requests": 2, "input_tokens": 2000, "output_tokens": 200, "tool_calls": 1}, thoughts_tokens=20, cost_usd=None)
    write_record(out, "vector", "q2", 3, model, answer="分かりません。\n\n根拠: 第3回・PDF p.10", tool_calls=[retrieval_call(1, pages([13, 39, 10, 11, 12]))],
                 duration_s=6.0, usage={"requests": 3, "input_tokens": 3000, "output_tokens": 300, "tool_calls": 1}, thoughts_tokens=30, cost_usd=0.0003)


def table_rows(summary, title, level="exercise"):
    """レベル level の、title で始まる見出しの表を、見出しの列名をキーにした dict の行のリストにする。"""
    part = summary.split(f"### レベル: {level}")[1]
    lines = part.split(f"#### {title}")[1].splitlines()[2:]
    cells = [[c.strip() for c in line.strip("|").split("|")] for line in lines[:]]
    header, rows = cells[0], []
    for line in cells[2:]:
        if len(line) != len(header):
            break
        rows.append(dict(zip(header, line, strict=True)))
    return rows


def row_of(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def test_the_summary_tables_show_calls_retrieval_citations_checks_and_costs(tmp_path):
    write_q2_trials(tmp_path)

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    a = row_of(table_rows(summary, "表A"), 条件="V0 vector", 質問="q2")
    assert (a["tool_calls"], a["requests"], a["引数（t1。検索語・目次のノード）"], a["開いたページ数（Tree）"]) == ("1,1,1", "2,2,3", "search_materials(コンパクションとメモリ)", "—")
    b = row_of(table_rows(summary, "表B"), 条件="V0 vector", 質問="q2")
    assert b["gold 取得"] == "3/3,3/3,2/3" and b["全文で取得したページ数"] == "5,5,5"  # t3 は p.40 が取れていない
    assert (b["先頭文だけ（Tree）"], b["seeds／expanded（Graph）"], b["hit／neighbor"]) == ("—", "—", "—")
    assert b["gold の順位"] == "t1[#1,#2,#3] t2[#1,#2,#3] t3[#1,#2,×]"
    c1 = row_of(table_rows(summary, "表C1"), 条件="V0 vector", 質問="q2")
    assert (c1["引用ページ数"], c1["gold 引用"], c1["acceptable"], c1["それ以外"], c1["未解決・曖昧"]) == ("3,1,1", "3/3,1/3,0/3", "0,0,0", "0,0,1", "0,0,0")
    assert c1["取得×引用の 2×2"] == "4/4/0/1"  # 3 回分の gold ページごとの合計（取れて引用 3+1+0／取れたが引用せず 0+2+2／取らずに引用 0／取れず引用もせず 0+0+1）
    c2 = row_of(table_rows(summary, "表C2"), 条件="V0 vector", 質問="q2")
    assert c2["チェック"] == "q2_summary（required）2/3<br>q2_persist（required）1/3<br>q2_daily（required）1/3<br>q2_scope（reference）0/3"
    assert (c2["自動の正答"], c2["要確認"], c2["自動の誤答"]) == ("1/3", "0/3", "2/3")  # t1 は全部一致。t2・t3 は required が足りない
    d = row_of(table_rows(summary, "表D"), 条件="V0 vector", 質問="q2")
    assert [d[k] for k in ("試行数", "秒", "入力", "出力", "thoughts", "requests", "推定費用 USD")] == [
        "3", "4.0 (2.0–6.0)", "2000 (1000–3000)", "200 (100–300)", "20 (10–30)", "2 (2–3)", "0.000200 (0.000100–0.000300)（算定不可 1/3）"]


def test_the_two_by_two_counts_every_gold_page_of_every_trial(tmp_path):
    write_q2_trials(tmp_path)
    cfg = cfg_of()
    views = [compare.trial_view(cfg, load(result_path(tmp_path, "vector", "q2", k))) for k in (1, 2, 3)]

    grids = [v["metrics"]["grid"] for v in views]
    assert grids[0] == {"retrieved_cited": 3, "retrieved_not_cited": 0, "not_retrieved_cited": 0, "neither": 0}
    assert grids[1] == {"retrieved_cited": 1, "retrieved_not_cited": 2, "not_retrieved_cited": 0, "neither": 0}
    assert grids[2] == {"retrieved_cited": 0, "retrieved_not_cited": 2, "not_retrieved_cited": 0, "neither": 1}
    assert [v["metrics"]["other"] for v in views] == [0, 0, 1] and [v["gold"]["hit"] for v in views][2] == ["lec03-main-p013", "lec03-main-p039"]


def write_q3_trials(out):
    """q3（資料にない質問）の 3 回分: t1 は断って数字なし（自動の正答）／t2 は断ったが数字も書いた（要確認）／t3 は断らずに数字を書いた（自動の誤答）。"""
    write_record(out, "vector", "q3", 1, answer="講義資料からは確認できませんでした。")
    write_record(out, "vector", "q3", 2, answer="講義資料からは確認できませんでした。一般には3秒前後とされます。")
    write_record(out, "vector", "q3", 3, answer="平均応答時間は2.5秒、正答率は85%です。")


def test_the_summary_c2_has_three_columns_for_the_automatic_verdicts(tmp_path):
    write_q3_trials(tmp_path)

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    row = row_of(table_rows(summary, "表C2"), 条件="V0 vector", 質問="q3")
    assert list(row)[:6] == ["条件", "質問", "チェック", "自動の正答", "要確認", "自動の誤答"]
    assert row["チェック"] == "q3_refuse（required）2/3<br>q3_numflag（flag）1/3"  # flag のチェックは「通れば問題なし」。数字を書いた t2・t3 は通らない
    assert (row["自動の正答"], row["要確認"], row["自動の誤答"]) == ("1/3", "1/3", "1/3")  # 要確認は、正答にも誤答にも数えない
    data = load(tmp_path / "summary.json")["models"][MODEL]["exercise"]["table_c"]["c2"]["V0 vector"]["q3"]
    assert data == {"checks": {"q3_refuse": "2/3", "q3_numflag": "1/3"}, "correct": "1/3", "review": "1/3", "wrong": "1/3",
                    "counts": {"correct": 1, "review": 1, "wrong": 1, "trials": 3}}
    assert "自動の判定は 3 つ。自動の正答 = required がすべて一致し、flag が一致しない。要確認 = required がすべて一致し、flag も一致" in summary
    assert "自動の誤答 = required のどれかが一致しない（flag は問わない）" in summary
    details = summary.split("<summary>V0 vector / q3（3 回）</summary>")[1].split("</details>")[0]
    assert "○q3_refuse ○q3_numflag｜自動の正答" in details  # t1
    assert "○q3_refuse ×q3_numflag｜**要確認（flag に一致）。目視で決める**" in details  # t2: flag に一致したのは、×q3_numflag
    assert "×q3_refuse ×q3_numflag｜自動の誤答（flag に一致）" in details  # t3: required に一致せず、flag にも一致


def test_the_summary_json_has_the_four_tables_per_level_for_the_answer_md(tmp_path):
    write_q2_trials(tmp_path)

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    data = load(tmp_path / "summary.json")
    level = data["models"][MODEL]["exercise"]
    assert set(level) >= {"rows", "questions", "table_a", "table_b", "table_c", "table_d"} and set(level["table_c"]) == {"c1", "c2"}
    assert level["table_a"]["V0 vector"]["q2"]["tool_calls"] == [1, 1, 1] and level["table_b"]["V0 vector"]["q2"]["gold"][2] == {"hit": ["lec03-main-p013", "lec03-main-p039"], "total": 3}
    assert level["table_d"]["V0 vector"]["q2"]["cost_usd"].endswith("（算定不可 1/3）")
    assert level["table_c"]["c2"]["V0 vector"]["q2"] == {
        "checks": {"q2_summary": "2/3", "q2_persist": "1/3", "q2_daily": "1/3", "q2_scope": "0/3"},
        "correct": "1/3", "review": "0/3", "wrong": "2/3", "counts": {"correct": 1, "review": 0, "wrong": 2, "trials": 3}}
    assert data["expected_trials"] == 3 and MODEL in data["warnings"]


def test_a_cost_that_cannot_be_calculated_is_never_shown_as_zero():
    assert compare.cost_cell([None, None, None]) == "算定不可" and compare.cost_cell([]) == "算定不可"
    assert compare.cost_cell([0.0001, 0.0002, 0.0003]) == "0.000200 (0.000100–0.000300)"
    assert compare.cost_cell([0.0001, None, 0.0003]).endswith("（算定不可 1/3）") and "0.000000" not in compare.cost_cell([None, 0.0002])
    assert compare.cost_cell([0.0, 0.0]) == "0.000000 (0.000000–0.000000)"  # 本当の 0 は、そのまま（None と区別する）


def test_a_trial_whose_cost_is_none_is_saved_as_null_and_summarized_as_not_available(tmp_path, monkeypatch):
    use_model(monkeypatch, text_model())  # FunctionModel は、費用を算定できない

    assert real_run(tmp_path) == 0 and run("summarize", "--out", tmp_path) == 0

    record = load(result_path(tmp_path))
    assert record["cost_usd"] is None and record["usage"]["cost"] is None
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "算定不可" in summary and "0.000000" not in summary


def two_responses(first_cost, second_cost):
    """1 回目にツールを呼び、2 回目に答えるモデル。応答ごとの費用（None なら、分からない）を指定する。"""
    from decimal import Decimal

    calls = []

    def fn(messages, info):
        calls.append(1)
        cost = first_cost if len(calls) == 1 else second_cost
        usage = RequestUsage(input_tokens=100 * len(calls), output_tokens=5, cost=Decimal(cost) if cost is not None else None)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("search_materials", {"query": "LoRA"})], usage=usage)
        return ModelResponse(parts=[TextPart("答えです。")], usage=usage)

    return FunctionModel(fn)


def test_the_trial_cost_is_unknown_when_any_response_cost_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda query: [])
    use_model(monkeypatch, two_responses("0.001", None))  # 2 つの応答のうち、1 つだけ費用が分かる

    assert real_run(tmp_path) == 0

    record = load(result_path(tmp_path))
    assert record["cost_usd"] is None and record["usage"]["cost"] is None  # 分かった分だけの合計（0.001）にしない
    assert [r["cost_usd"] for r in record["responses"]] == [pytest.approx(0.001), None]  # 応答ごとの費用は残る
    assert record["usage"]["requests"] == 2 and record["usage"]["input_tokens"] == 300
    assert "算定不可" in result_path(tmp_path, suffix=".trace.txt").read_text(encoding="utf-8") and "$0.001" not in result_path(tmp_path, suffix=".trace.txt").read_text(encoding="utf-8")
    assert run("summarize", "--out", tmp_path) == 0 and "算定不可" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_the_trial_cost_is_the_sum_when_every_response_cost_is_known(tmp_path, monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda query: [])
    use_model(monkeypatch, two_responses("0.001", "0.0025"))

    assert real_run(tmp_path) == 0

    record = load(result_path(tmp_path))
    assert record["cost_usd"] == pytest.approx(0.0035) and record["usage"]["cost"] == "0.0035"
    assert [r["cost_usd"] for r in record["responses"]] == [pytest.approx(0.001), pytest.approx(0.0025)]
    assert "$0.003500" in result_path(tmp_path, suffix=".trace.txt").read_text(encoding="utf-8")


def test_usage_of_never_uses_a_partial_cost_and_does_not_change_the_run_result():
    from decimal import Decimal

    from pydantic_ai.usage import RunUsage

    def response(cost):
        return ModelResponse(parts=[TextPart("x")], usage=RequestUsage(input_tokens=10, output_tokens=1, cost=Decimal(cost) if cost is not None else None))

    run_usage = RunUsage(requests=2, input_tokens=20, output_tokens=2, cost=Decimal("0.001"))  # pydantic-ai は None を 0 として足す（一部だけの合計）
    partial = compare.usage_of(SimpleNamespace(usage=run_usage), [response("0.001"), response(None)])
    complete = compare.usage_of(SimpleNamespace(usage=RunUsage(requests=2, cost=Decimal("0.003"))), [response("0.001"), response("0.002")])
    failed = compare.usage_of(None, [response("0.001"), response(None)])

    assert partial.cost is None and run_usage.cost == Decimal("0.001")  # 元の usage は書き換えない
    assert complete.cost == Decimal("0.003") and failed.cost is None and failed.requests == 2 and failed.input_tokens == 20
    assert compare.response_costs([response("0.001"), response(None)]) == [Decimal("0.001"), None]


def test_the_cost_and_thought_tokens_of_a_real_response_are_recorded(tmp_path, monkeypatch):
    from decimal import Decimal

    usage = RequestUsage(input_tokens=1000, output_tokens=50, details={"thoughts_tokens": 30}, cost=Decimal("0.000123"))
    use_model(monkeypatch, FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("答え")], usage=usage)))

    assert real_run(tmp_path) == 0

    record = load(result_path(tmp_path))
    assert record["cost_usd"] == pytest.approx(0.000123) and record["thoughts_tokens"] == 30 and record["usage"]["cost"] == "0.000123"
    assert record["usage"]["input_tokens"] == 1000 and record["usage"]["details"] == {"thoughts_tokens": 30}
    assert record["responses"] == [{"index": 1, "input_tokens": 1000, "output_tokens": 50, "thoughts_tokens": 30, "finish_reason": None, "cost_usd": pytest.approx(0.000123)}]


def test_the_summary_computes_retrieval_and_citations_again_from_tool_calls_and_answer(tmp_path):
    pages = [page_row("lec03-main-p013", distance=0.1), page_row("lec03-main-p039", distance=0.2), page_row("lec03-main-p040", distance=0.3)]
    write_record(tmp_path, "vector", "q2", 1, answer=GOOD_Q2, tool_calls=[retrieval_call(1, pages)],
                 retrieved={"full_text": [], "preview": []}, citations=[])  # 古い解析の結果（空）が書いてあっても

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| V0 vector | q2 | 3/3 |" in summary and "| V0 vector | q2 | 3 | 3/3 |" in summary  # 取得も引用も、いまの解析で計算し直す


def test_the_summary_warns_about_failures_retries_mismatches_tool_errors_ambiguity_and_flags(tmp_path):
    write_record(tmp_path, "vector", "q1", 1, status="usage_limit")
    write_record(tmp_path, "vector", "q1", 2, attempts=[{"attempt": 1, "source": "embedding", "status": 429, "action": "retry", "wait_s": 7.0}, {"attempt": 2, "ok": True}])
    write_record(tmp_path, "vector", "q1", 3, instructions_match=False)
    write_record(tmp_path, "vector", "q2", 1, tool_errors=[{"order": 1, "name": "search_materials", "kind": "model", "error": "検索語が空です。"}])
    write_record(tmp_path, "full", "q2", 1, answer="第3回・PDF p.13", materials={"documents": SCOPE_ALL, "pages": 438})
    write_record(tmp_path, "vector", "q3", 1, answer="確認できません。一般には3秒前後です。")

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    warnings = (tmp_path / "summary.md").read_text(encoding="utf-8").split("## 警告")[1].split("## モデル")[0]
    assert "status が ok ではない試行: 1 件" in warnings and "V0 vector/q1 t1（usage_limit）" in warnings
    assert "再試行した試行" in warnings and "V0 vector/q1 t2（embedding:429）" in warnings
    assert "instructions_match が false）: 1 件" in warnings and "V0 vector/q1 t3" in warnings
    assert "ツールの error を返した試行: 1 件" in warnings and "検索語が空です" in warnings
    assert "ambiguous）がある試行: 1 件" in warnings and "F full/q2 t1（1 件）" in warnings
    assert "要確認（required がすべて一致し、flag にも一致。回答を読んで決める。正答にも誤答にも数えない）の試行: 1 件" in warnings and "V0 vector/q3 t1" in warnings
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**要確認（flag に一致）。目視で決める**" in summary


def test_the_details_keep_the_answer_the_tool_arguments_and_the_raw_citations(tmp_path):
    write_q2_trials(tmp_path)

    assert run("summarize", "--out", tmp_path, "--level", "exercise") == 0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "<details>" in summary and "<summary>V0 vector / q2（3 回）</summary>" in summary
    assert "**t1**｜ok｜2.0 秒｜tool_calls 1｜requests 2｜gold 取得 3/3｜gold 引用 3/3｜○q2_summary ○q2_persist ○q2_daily ×q2_scope" in summary
    assert "```\n" + GOOD_Q2 + "\n```" in summary and '- `search_materials({"query":"コンパクションとメモリ"})` → success' in summary
    assert "- 引用の raw: PDF p.13、39、40（ok。候補 lec03-main-p013）、PDF p.13、39、40（ok。候補 lec03-main-p039）" in summary


def test_the_summary_after_a_dry_run_has_tables_a_to_d_for_every_level_with_references_first(tmp_path):
    out = tmp_path / "out"
    assert dry_run(out, trials=2) == 0

    assert run("summarize", "--out", out, "--trials", 2) == 0

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.splitlines()[0].startswith("# 比較の集計") and "手で直さない" in summary
    for needle in ("## 前提", "## 警告", f"## モデル: {MODEL}-dry-run", "### レベル: exercise", "### レベル: basic", "### レベル: impl", "### レベル: adv",
                   "表A", "表B", "表C1", "表C2", "表D", "<details>", "F214 full_aligned", "T tree", "A2 term_graph_cap3", "I2 topk15", "438 ページ", "214 ページ"):
        assert needle in summary, needle
    assert "未実行" not in summary and "回]" not in summary  # 全部そろっているので、足りない試行の注記も無い
    for level, firsts in (("exercise", ["F full"]), ("basic", ["V0 vector（参考）", "B1 doc_hint"]), ("impl", ["V0 vector（参考）", "I1 neighbors", "I2 topk15"]),
                          ("adv", ["V0 vector（参考）", "G graph（参考）", "A1 term_graph", "A2 term_graph_cap3"])):
        titles = list(dict.fromkeys(r["条件"] for r in table_rows(summary, "表A", level)))
        assert titles[0] == firsts[0] and [t for t in titles if t in firsts] == firsts, level  # 参考の行が先頭
    assert [r["条件"] for r in table_rows(summary, "表A", "adv")][:5] == ["V0 vector（参考）"] * 5  # 参考は、全 5 問の行をすべて先頭に置く
    data = load(out / "summary.json")["models"][f"{MODEL}-dry-run"]
    assert list(data) == ["exercise", "basic", "impl", "adv"] and [r["variant"] for r in data["adv"]["rows"]] == ["vector", "graph", "term_graph", "term_graph_cap3"]
    assert [r["reference"] for r in data["adv"]["rows"]] == [True, True, False, False]
    assert all(set(level) >= {"table_a", "table_b", "table_c", "table_d"} for level in data.values())
    c2 = table_rows(summary, "表C2")
    assert {r["質問"] for r in c2} == {"q1", "q2", "q3", "qa1", "qa2"} and len(c2) == 3 + 3 + 5 + 3 + 5  # exercise: F・F214・T（q1〜q3）、V0・G（5 問）
    assert all((r["自動の正答"], r["要確認"], r["自動の誤答"]) == ("0/2", "0/2", "2/2") for r in c2)  # 予行の台本の回答は、required に一致しない


def test_summarize_filters_by_level_variant_and_question_and_keeps_the_reference_rows(tmp_path):
    out = tmp_path / "out"
    assert dry_run(out) == 0

    assert run("summarize", "--out", out, "--level", "impl", "--variant", "I1", "--question", "q1", "q2") == 0

    data = load(out / "summary.json")["models"][f"{MODEL}-dry-run"]
    assert list(data) == ["impl"] and data["impl"]["questions"] == ["q1", "q2"]
    assert list(data["impl"]["table_a"]) == ["V0 vector（参考）", "I1 neighbors"]  # I2 は選んでいない。参考の V0 は残る


def test_summarize_without_results_says_so(tmp_path, capsys):
    assert run("summarize", "--out", tmp_path / "nothing") == 1
    assert "集計できる結果がありません" in capsys.readouterr().out
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "step1.txt").write_text("CLI の出力", encoding="utf-8")  # STEP 1〜4 の CLI の出力は、集計の対象ではない
    assert run("summarize", "--out", tmp_path) == 1


def test_a_condition_with_fewer_trials_than_expected_says_so_in_the_summary(tmp_path):
    write_record(tmp_path, "vector", "q1", 1)
    write_record(tmp_path, "vector", "q1", 2)

    assert run("summarize", "--out", tmp_path, "--level", "exercise", "--trials", 3) == 0
    assert "| V0 vector | q1 | 0,0 [2/3回] |" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert run("summarize", "--out", tmp_path, "--level", "exercise", "--trials", 2) == 0
    assert "回]" not in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_status_lists_the_conditions_in_run_order_with_their_position_in_each_trial(tmp_path, capsys):
    assert run("status", "--out", tmp_path, "--trials", 3) == 0

    printed = capsys.readouterr().out
    rows = [line.split() for line in printed.splitlines() if re.match(r"q(1|2|3|a1|a2)\b|qa", line)]
    assert [row[0] for row in rows] == ["q1"] * 10 + ["q2"] * 10 + ["q3"] * 10 + ["qa1"] * 7 + ["qa2"] * 7  # 質問ごと（試行 → 質問 → 条件）
    assert [f"{row[1]} {row[2]}" for row in rows[:10]] == ["F full", "V0 vector", "T tree", "F214 full_aligned", "G graph", "B1 doc_hint",
                                                           "I1 neighbors", "I2 topk15", "A1 term_graph", "A2 term_graph_cap3"]  # 試行 1 の順
    positions = {(row[0], row[2]): row[4] for row in rows}
    assert positions[("q1", "full")] == "1,8,5" and positions[("q1", "vector")] == "2,9,6" and positions[("q1", "neighbors")] == "7,4,1"
    assert positions[("qa1", "vector")] == "1,7,5"  # qa1 は全文投入と Tree を流さないので、位置は 7 つの中（t3 は neighbors・topk15・term_graph・term_graph_cap3 のあと）
    assert "試行ごとに条件の順を 3 個ずつ巡回" in printed and "順（試行ごと）" in printed


def test_the_status_command_counts_planned_done_remaining_and_the_last_status(tmp_path, capsys):
    write_record(tmp_path, "vector", "q1", 1)
    write_record(tmp_path, "vector", "q1", 2, status="usage_limit")
    write_record(tmp_path, "full", "q1", 1)

    assert run("status", "--out", tmp_path, "--trials", 3) == 0

    printed = capsys.readouterr().out
    lines = printed.splitlines()
    row = next(line for line in lines if line.startswith("q1 ") and "V0 vector" in line)
    assert row.split()[4:] == ["2,9,6", "2/3", "usage_limit", "1", "2.5"]  # 順（試行ごと）・済み/予定・最後の status・残り・残りの推定リクエスト（1 × 2.5）
    total = next(line for line in lines if line.startswith("合計"))
    assert "3/132" in total and "129" in total
    assert "全文投入" in printed and "17 回" in printed and "残りの推定リクエスト" in printed  # 全文投入 18 回のうち 1 回が済み
    assert run("status", "--out", tmp_path, "--level", "basic", "--variant", "B1", "--question", "q1") == 0
    assert capsys.readouterr().out.count("doc_hint") == 1
