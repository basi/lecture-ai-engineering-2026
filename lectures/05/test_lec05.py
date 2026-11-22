"""LLM も API キーもネットワークも使わない。実行：`uv run pytest -q lectures/05`

第5回の宿題（実装：前後ページ、発展：用語から始める Graph 検索）のテスト。
質問の埋め込みは偽物に差し替える。本物の検索DBを使うテストでは、DBに保存された1ページの埋め込みを
質問の埋め込みとして返す（そのページが距離 0 で1位になる）。選び方のテストは、偽の距離を select_pages に渡す。
"""

import json
import os
import sqlite3
from collections import Counter

import pytest
from google.genai import errors as genai_errors
from pydantic_ai import Tool
from pydantic_ai.toolsets import FunctionToolset

from app import retrieval, tools
from app import retrieval_lec05 as lec05
from app.agent import agent
from app.config import api_error_message

STORED_PAGE = "lec03-main-p050"  # 偽の埋め込みに使うページ。DBに保存された埋め込みを、そのまま質問の埋め込みにする
PAGES = retrieval.load_pages(retrieval.RAG_SOURCE)  # 対象の214ページ
PAGE_BY_ID = {page["page_id"]: page for page in PAGES}
DATA = lec05.graph_data()
TOC_PAGES = {"lec01-main-p058", "lec03-main-p004"}  # 用語が並ぶ目次ページ（カリキュラム、アジェンダ）
FINE_TUNING_CHAPTER = "lec02-main-s045"  # 第2回「選んだモデルを、いつ・どう直すか」。ファインチューニングの言及が最も多い章
QA1 = "回答を決まったJSON形式で返させたいとき、第2回で学んだファインチューニングと第3回の構造化出力は、どう使い分ければよいですか"


@pytest.fixture(autouse=True)
def empty_trace():
    """TRACE は呼び出しごとに足されるので、テストごとに空にする。"""
    lec05.TRACE.clear()
    yield
    lec05.TRACE.clear()


@pytest.fixture
def fresh_caches():
    """ページ一覧と用語グラフのキャッシュを空にする。upstream の関数を差し替えるテストで、前の結果を残さない。"""
    lec05._load_pages_cached.cache_clear()
    lec05._build_graph_data.cache_clear()
    yield
    lec05._load_pages_cached.cache_clear()
    lec05._build_graph_data.cache_clear()


@pytest.fixture
def embed_stored_page(monkeypatch):
    """質問の埋め込みを、DBに保存された STORED_PAGE の埋め込みにする。呼び出し（質問文, 用途）を記録して返す。"""
    with retrieval.open_index(retrieval.RAG_DB) as db:
        blob = db.execute("SELECT embedding FROM chunks WHERE page_id = ?", (STORED_PAGE,)).fetchone()["embedding"]
    vector = memoryview(blob).cast("f").tolist()  # float32 の BLOB を float のリストに戻す。struct も numpy も使わない
    calls = []

    def fake_embed(text, task_type):
        calls.append((text, task_type))
        return vector

    monkeypatch.setattr("app.retrieval.embed_text", fake_embed)
    return calls


def forbidden(*args, **kwargs):
    raise AssertionError("このテストでは呼ばれないはず")


def hit(document_id: str, number: int, distance: float = 0.1) -> dict:
    """検索結果の1行（retrieval.search() と同じキー）を、本物のページから作る。"""
    return {**PAGE_BY_ID[f"{document_id}-p{number:03d}"], "distance": distance}


def lec02_hits(*numbers: int) -> list[dict]:
    """第2回のページを、検索結果の順位（先頭が1位）で並べた hit にする。"""
    return [hit("lec02-main", number, 0.1 * (rank + 1)) for rank, number in enumerate(numbers)]


def labels(rows: list[dict]) -> list[str]:
    """[p50n, p51h, ...] の形にする。n は neighbor、h は hit。"""
    return [f"p{row['page']}{row['role'][0]}" for row in rows]


# --- 定数 ---


def test_constants_have_the_documented_defaults():
    assert lec05.NEIGHBOR_WINDOW == 1
    assert lec05.TERM_GRAPH_TOP_K == retrieval.RAG_TOP_K == 5  # 基本RAGの返す件数と同じ枠
    assert lec05.TERM_GRAPH_SECTION_CAP == 2
    assert lec05.TERM_GRAPH_VECTOR_POOL == 5
    assert lec05.TERM_GRAPH_HUB_TERMS == 9


# --- 前後ページ（実装） ---


def test_neighbors_surround_the_hits_in_page_order():
    rows = lec05.add_neighbors(lec02_hits(52, 51), window=1)

    assert labels(rows) == ["p50n", "p51h", "p52h", "p53n"]
    assert [row["distance"] for row in rows] == [None, 0.2, 0.1, None]  # hit は元の距離、neighbor は None
    assert {row["document_id"] for row in rows} == {"lec02-main"}


def test_each_row_has_the_eight_page_fields_then_distance_role_and_neighbor_of():
    rows = lec05.add_neighbors(lec02_hits(52), window=1)

    fields = ["page_id", "document_id", "lecture", "document_title", "section", "title", "page", "content"]
    for row in rows:
        assert list(row) == fields + ["distance", "role", "neighbor_of"]  # hit も neighbor も同じキーと順序
    neighbor = next(row for row in rows if row["role"] == "neighbor")
    assert neighbor["content"] == PAGE_BY_ID[neighbor["page_id"]]["content"]  # 本文と引用情報が付く


def test_neighbor_of_names_the_hits_the_page_sits_next_to():
    rows = {row["page"]: row for row in lec05.add_neighbors(lec02_hits(52, 51), window=1)}

    assert rows[50]["neighbor_of"] == ["lec02-main-p051"]
    assert rows[53]["neighbor_of"] == ["lec02-main-p052"]
    assert rows[51]["neighbor_of"] == rows[52]["neighbor_of"] == []  # hit は空リスト


def test_the_example_with_five_hits_gives_three_chunks_in_hit_order():
    rows = lec05.add_neighbors(lec02_hits(52, 51, 60, 61, 46), window=1)

    assert labels(rows) == [
        "p50n", "p51h", "p52h", "p53n",  # 1位と2位の塊
        "p59n", "p60h", "p61h", "p62n",  # 3位と4位の塊
        "p45n", "p46h", "p47n",  # 5位の塊
    ]


def test_neighbors_stop_at_the_ends_of_a_document():
    first = lec05.add_neighbors([hit("lec02-main", 1)], window=1)
    last = lec05.add_neighbors([hit("lec01-main", 73)], window=1)

    assert labels(first) == ["p1h", "p2n"]  # lec02-main-p001 の前は無い。lec01 の最後には行かない
    assert labels(last) == ["p72n", "p73h"]  # lec01-main-p073 の次は無い。lec02 の p001 には行かない
    assert {row["document_id"] for row in first} == {"lec02-main"}
    assert {row["document_id"] for row in last} == {"lec01-main"}


def test_neighbors_never_cross_into_another_document():
    exercise = {"page_id": "lec01-exercise-p002", "document_id": "lec01-exercise", "page": 2, "distance": 0.1}

    rows = lec05.add_neighbors([exercise, hit("lec01-main", 73)], window=2)

    assert [row["page_id"] for row in rows] == [
        "lec01-exercise-p002",  # 対象外の資料のページは、隣を持たない
        "lec01-main-p071", "lec01-main-p072", "lec01-main-p073",
    ]
    assert all(row["document_id"] in retrieval.RAG_DOCUMENT_IDS for row in rows if row["role"] == "neighbor")
    # 対象外の資料（exercise など）のページは、隣としても現れない
    for number in (1, 20, 73):
        rows = lec05.add_neighbors([hit("lec01-main", number)], window=3)
        assert {row["document_id"] for row in rows} == {"lec01-main"}


def test_a_page_between_two_hits_is_added_once_and_names_both_hits():
    rows = lec05.add_neighbors(lec02_hits(41, 39), window=1)

    assert labels(rows) == ["p38n", "p39h", "p40n", "p41h", "p42n"]
    assert [row["page"] for row in rows].count(40) == 1
    between = next(row for row in rows if row["page"] == 40)
    assert between["neighbor_of"] == ["lec02-main-p041", "lec02-main-p039"]  # hit の順位順（41 が1位）


def test_chunks_are_ordered_by_their_best_hit():
    rows = lec05.add_neighbors(lec02_hits(60, 52, 61), window=1)

    # 60 位と 61 位は同じ塊で、その最上位は1位。52 の塊は2位なので、あとに来る
    assert labels(rows) == ["p59n", "p60h", "p61h", "p62n", "p51n", "p52h", "p53n"]


def test_hits_next_to_each_other_stay_hits():
    rows = lec05.add_neighbors(lec02_hits(52, 51), window=1)

    roles = {row["page"]: row["role"] for row in rows}
    assert roles[51] == roles[52] == "hit"  # 51 は 52 の隣でも、neighbor にしない
    assert all(row["neighbor_of"] == [] for row in rows if row["role"] == "hit")


def test_the_same_page_is_never_returned_twice():
    rows = lec05.add_neighbors(lec02_hits(52, 52, 53), window=2)

    keys = [(row["document_id"], row["page"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_window_zero_returns_the_hits_as_they_are():
    hits = lec02_hits(52, 51, 10)

    rows = lec05.add_neighbors(hits, window=0)

    assert [row["page_id"] for row in rows] == [h["page_id"] for h in hits]  # 並べ替えず、足さない
    assert [row["distance"] for row in rows] == [h["distance"] for h in hits]
    assert all(row["role"] == "hit" and row["neighbor_of"] == [] for row in rows)


@pytest.mark.parametrize("broken", [
    {"page_id": "x-p005", "page": 5, "distance": 0.3},  # document_id が無い
    {"page_id": "x-p005", "document_id": "lec02-main", "distance": 0.3},  # page が無い
    {},
])
def test_a_hit_without_document_id_or_page_is_kept_without_neighbors(broken):
    rows = lec05.add_neighbors([hit("lec02-main", 52), broken], window=1)

    assert labels(rows[:3]) == ["p51n", "p52h", "p53n"]  # ほかの hit の隣は、これまでどおり付く
    assert len(rows) == 4
    assert rows[3]["role"] == "hit" and rows[3]["neighbor_of"] == []
    assert rows[3].get("page_id") == broken.get("page_id")


def test_a_gap_in_the_given_pages_splits_the_chunks_without_breaking_the_order():
    subset = [page for page in PAGES if page["document_id"] == "lec02-main" and page["page"] in (50, 52)]  # p51 が無い

    rows = lec05.add_neighbors(lec02_hits(50, 60), window=2, pages=subset)

    # p52 は p50 の2つ隣だが、p51 が無く連続しないので別の塊。同じ順位の塊は、ページ順に並ぶ。
    # p60 は subset に無くても、hit なので残る
    assert labels(rows) == ["p50h", "p52n", "p60h"]
    assert rows[1]["neighbor_of"] == ["lec02-main-p050"]


def test_negative_window_is_rejected():
    with pytest.raises(ValueError):
        lec05.add_neighbors(lec02_hits(52), window=-1)


def test_pages_not_in_the_given_list_are_not_added():
    subset = [page for page in PAGES if page["document_id"] == "lec02-main" and page["page"] in (51, 52)]

    rows = lec05.add_neighbors(lec02_hits(52), window=1, pages=subset)

    assert labels(rows) == ["p51n", "p52h"]  # 辞書に無い p53 は足さない


def test_default_pages_come_from_upstream_load_pages(fresh_caches, monkeypatch):
    seen = []

    def fake_load_pages(source=retrieval.RAG_SOURCE):
        seen.append(source)
        return [page for page in PAGES if page["document_id"] == "lec02-main" and page["page"] in (51, 52)]

    monkeypatch.setattr("app.retrieval.load_pages", fake_load_pages)

    rows = lec05.add_neighbors(lec02_hits(52), window=1)

    assert labels(rows) == ["p51n", "p52h"]
    assert seen == [retrieval.RAG_SOURCE]  # モジュール経由で呼ぶので、差し替えが効く


def test_neighbor_window_is_read_on_every_call(monkeypatch):
    assert labels(lec05.add_neighbors(lec02_hits(52))) == ["p51n", "p52h", "p53n"]  # 既定は前後1ページ

    monkeypatch.setattr("app.retrieval_lec05.NEIGHBOR_WINDOW", 2)
    assert labels(lec05.add_neighbors(lec02_hits(52))) == ["p50n", "p51n", "p52h", "p53n", "p54n"]

    monkeypatch.setattr("app.retrieval_lec05.NEIGHBOR_WINDOW", 0)
    assert labels(lec05.add_neighbors(lec02_hits(52))) == ["p52h"]
    assert labels(lec05.add_neighbors(lec02_hits(52), window=1)) == ["p51n", "p52h", "p53n"]  # 引数が優先


def test_search_with_neighbors_uses_upstream_search_and_records_the_trace(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", lambda query: lec02_hits(52, 51))

    rows = lec05.search_with_neighbors("LoRA", window=1)

    assert labels(rows) == ["p50n", "p51h", "p52h", "p53n"]
    assert lec05.TRACE == [{
        "tool": "neighbors", "query": "LoRA", "window": 1,
        "hits": ["lec02-main-p052", "lec02-main-p051"],
        "neighbors": ["lec02-main-p050", "lec02-main-p053"],
        "pages": 4,
    }]


def test_search_with_neighbors_rejects_a_negative_window_before_searching(monkeypatch):
    monkeypatch.setattr("app.retrieval.search", forbidden)

    with pytest.raises(ValueError):
        lec05.search_with_neighbors("LoRA", window=-1)


def test_search_with_neighbors_works_on_the_real_index(embed_stored_page):
    hits = retrieval.search("構造化出力とは")
    assert hits[0]["page_id"] == STORED_PAGE and hits[0]["distance"] == pytest.approx(0, abs=1e-6)  # 距離 0 で1位
    assert len(hits) == retrieval.RAG_TOP_K == 5

    rows = lec05.search_with_neighbors("構造化出力とは")

    assert [row["page_id"] for row in rows if row["role"] == "hit"] != []
    assert {row["page_id"] for row in rows if row["role"] == "hit"} == {h["page_id"] for h in hits}  # hit は5件そのまま
    neighbors = [row for row in rows if row["role"] == "neighbor"]
    assert neighbors  # hit に neighbor が付く
    hit_ids = {h["page_id"] for h in hits}
    for row in neighbors:
        assert row["distance"] is None
        assert row["neighbor_of"] and set(row["neighbor_of"]) <= hit_ids
    page_ids = [row["page_id"] for row in rows]
    assert len(page_ids) == len(set(page_ids))  # 同じページは1回だけ
    (trace,) = lec05.TRACE
    assert trace == {
        "tool": "neighbors", "query": "構造化出力とは", "window": 1,
        "hits": [h["page_id"] for h in hits],
        "neighbors": [row["page_id"] for row in neighbors],
        "pages": len(rows),
    }


# --- 用語の照合・回の指定（発展） ---


def test_match_terms_does_not_pick_lora_inside_qlora():
    assert lec05.match_terms("QLoRAとLoRAの違い") == ["term:qlora", "term:lora"]  # 出現順。QLoRA の中の LoRA は数えない
    assert lec05.match_terms("QLoRAの違い") == ["term:qlora"]


def test_match_terms_lists_terms_in_the_order_they_appear_without_repeats():
    assert lec05.match_terms("構造化出力とファインチューニング") == ["term:structured-output", "term:fine-tuning"]
    assert lec05.match_terms("LoRAとLoRA") == ["term:lora"]
    assert lec05.match_terms("今日はいい天気です") == []


def test_match_terms_drops_a_name_inside_a_longer_matched_name():
    # 「エージェント」は「コーディングエージェント」の中にもあるが、重なる一致は拾わない
    assert lec05.match_terms("コーディングエージェントとは") == ["term:coding-agent"]
    assert lec05.match_terms("エージェントとは") == ["term:ai-agent"]
    assert lec05.match_terms("コーディングエージェントとエージェントの違い") == ["term:coding-agent", "term:ai-agent"]


def test_match_terms_finds_an_alias_such_as_json_output():
    assert lec05.match_terms("JSON出力にしたい") == ["term:structured-output"]


def test_match_terms_finds_memory_md_as_the_memory_term():
    assert lec05.match_terms("MEMORY.md には何を書く？") == ["term:memory"]  # memory.md の中の memory を二重に拾わない


def test_match_terms_ignores_case():
    assert lec05.match_terms("ragとは") == ["term:rag"]


def test_match_terms_does_not_pick_pi_inside_api():
    assert lec05.match_terms("APIの使い方") == ["term:api"]
    assert lec05.match_terms("Piとは") == ["term:pi"]  # 単独の Pi は拾う


def test_match_terms_ignores_english_names_inside_other_english_words():
    # pipeline の中の pi、storage の中の rag は拾わない（前後が英数字なら、名前の一致にしない）
    assert lec05.match_terms("storageとpipelineの話") == []
    assert lec05.match_terms("Pi と storage") == ["term:pi"]


def test_match_terms_accepts_fullwidth_letters():
    assert lec05.match_terms("ＬｏＲＡとは") == ["term:lora"]


@pytest.mark.parametrize(("query", "expected"), [
    ("第2回と第3回の違い", [2, 3]),
    ("第三回のメモリ", [3]),
    ("第５回の宿題", []),  # 全角の5は第5回になるが、対象は第1〜3回だけ
    ("第2回の説明", [2]),
    ("第２回の説明", [2]),  # 全角の数字
    ("第3回と第2回と第3回", [2, 3]),  # 昇順・重複なし
    ("第一回と第二回", [1, 2]),
    ("第10回と第12回", []),
    ("2回目の説明", []),
    ("", []),
])
def test_lecture_hints_reads_lecture_numbers_one_to_three(query, expected):
    assert lec05.lecture_hints(query) == expected


def test_graph_data_covers_the_214_target_pages_only():
    assert len(DATA.lecture_of) == len(DATA.section_of) == len(DATA.page_terms) == len(PAGES) == 214
    assert set(DATA.lecture_of.values()) == {1, 2, 3}
    assert set(DATA.page_terms) == set(PAGE_BY_ID)
    assert all(page_ids <= set(PAGE_BY_ID) for page_ids in DATA.mentions.values())  # 対象外の資料のページは入らない
    assert DATA.section_of["lec02-main-p052"] == FINE_TUNING_CHAPTER
    assert DATA.section_of["lec02-main-p001"] == "lec02-main"  # 「冒頭」のページは、章を持たず資料の直下
    assert len(DATA.page_terms["lec01-main-p058"]) == 17 and len(DATA.page_terms["lec03-main-p004"]) == 9
    assert DATA.terms["term:qlora"]["aliases"] == ["QLoRA", "Quantized LoRA"]
    assert DATA.related["term:quantization"] == {"term:qlora"}
    assert DATA.mentions["term:quantization"] == set()  # 第1〜3回のページには言及が無い


def test_related_terms_are_undirected_even_when_the_edge_goes_one_way():
    graph = json.loads(retrieval.RAG_GRAPH.read_text(encoding="utf-8"))
    edges = {(edge["source"], edge["target"]) for edge in graph["edges"] if edge["relation"] == "related_term"}
    one_way = sorted((a, b) for a, b in edges if (b, a) not in edges)

    assert one_way  # 片方向の関係がある（例：term:adk → term:ai-agent）
    for a, b in one_way:
        assert b in DATA.related[a] and a in DATA.related[b]


def test_graph_data_is_cached_until_the_files_or_target_documents_change(monkeypatch, tmp_path, fresh_caches):
    first = lec05.graph_data()
    assert lec05.graph_data() is first  # パスも更新時刻も同じなら、作り直さない

    copy = tmp_path / "graph.json"
    copy.write_bytes(retrieval.RAG_GRAPH.read_bytes())
    monkeypatch.setattr("app.retrieval.RAG_GRAPH", copy)
    second = lec05.graph_data()
    assert second is not first and second.terms == first.terms  # パスが変われば、作り直す
    os.utime(copy, ns=(1_000_000_000, 1_000_000_000))
    third = lec05.graph_data()
    assert third is not second  # 更新時刻が変われば、作り直す

    monkeypatch.setattr("app.retrieval.RAG_DOCUMENT_IDS", ("lec01-main",))
    only_first = lec05.graph_data()
    assert len(only_first.lecture_of) == 73 and set(only_first.lecture_of.values()) == {1}  # 対象資料が変われば、作り直す


# --- 選び方（発展）。偽の距離を渡す ---


def fake_distances(near: dict[str, float] | None = None) -> dict[str, float]:
    """全ページを遠い距離（0.9 台）に置き、near のページだけ近づけた、偽の距離。"""
    distances = {page_id: 0.9 + i * 1e-4 for i, page_id in enumerate(sorted(PAGE_BY_ID))}
    distances.update(near or {})
    return distances


def near_pages(page_ids: list[str], start: float = 0.10, step: float = 0.01) -> dict[str, float]:
    """指定したページを、近い距離にする。先頭が最も近く、step ずつ離れる。"""
    return {page_id: start + step * rank for rank, page_id in enumerate(page_ids)}


def section_pages(section_id: str) -> list[str]:
    """章（graph.json の section ノード）に属するページを、page_id の順に返す。"""
    return sorted(page_id for page_id, section in DATA.section_of.items() if section == section_id)


def select(terms=(), lectures=(), near=None, **params) -> dict:
    """偽の距離（near のページだけ近い）で select_pages を呼ぶ。既定は top_k=5, cap=2, vector_pool=5, hub_terms=9。"""
    args = {"top_k": 5, "cap": 2, "vector_pool": 5, "hub_terms": 9, **params}
    return lec05.select_pages(fake_distances(near), list(terms), list(lectures), DATA, **args)


def lectures_of(result: dict) -> set[int]:
    return {DATA.lecture_of[item["page_id"]] for item in result["selected"]}


FINE_TUNING_PAGES_IN_CHAPTER = [
    page_id for page_id in section_pages(FINE_TUNING_CHAPTER) if page_id in DATA.mentions["term:fine-tuning"]
]


def test_qa1_picks_pages_from_both_lecture_2_and_lecture_3():
    terms, lectures = lec05.match_terms(QA1), lec05.lecture_hints(QA1)
    assert terms == ["term:fine-tuning", "term:structured-output"] and lectures == [2, 3]
    near = near_pages(section_pages(FINE_TUNING_CHAPTER)[:5])  # 距離の上位が、第2回の1つの章だけで埋まっている

    result = select(terms, lectures, near)

    chosen = result["selected"]
    assert lectures_of(result) == {2, 3}  # 第2回と第3回の両方から選ばれる
    assert len(chosen) <= 5
    reasons = {item["reason"]: item["page_id"] for item in chosen}
    assert DATA.lecture_of[reasons["確保:用語:構造化出力"]] == 3  # 構造化出力は第3回の言及が多い。そのページを確保する
    assert reasons["確保:用語:構造化出力"] in DATA.mentions["term:structured-output"]
    assert DATA.lecture_of[reasons["確保:用語:ファインチューニング"]] == 2
    # 用語も回も使わずに同じ距離で選ぶと、第3回は入らない（章の上限で外れた分は、距離の近い順に補われる）
    assert 3 not in lectures_of(select([], [], near))


def test_a_section_gets_at_most_cap_pages_and_the_rest_are_reported_as_skipped():
    near = near_pages(FINE_TUNING_PAGES_IN_CHAPTER[:6])  # 距離の近い6ページが、すべて同じ章

    for cap in (1, 2, 3):
        result = select(["term:fine-tuning"], [], near, cap=cap, vector_pool=6)

        picked = {item["page_id"] for item in result["selected"]}
        skipped = {item["page_id"] for item in result["skipped_by_cap"]}
        assert Counter(item["section"] for item in result["selected"])[FINE_TUNING_CHAPTER] == cap  # 同じ章は cap 件まで
        assert set(near) - picked <= skipped  # 近い6ページのうち選ばれなかったものは、章の上限で外れた
        assert not picked & skipped
        assert set(result["skipped_by_cap"][0]) == {"page_id", "distance", "section"}
        assert all(Counter(item["section"] for item in result["selected"])[item["section"]] >= cap
                   for item in result["skipped_by_cap"])  # 外したのは、すでに cap 件に達した章のページだけ


def test_contents_pages_are_not_picked_through_terms():
    near = near_pages(section_pages(FINE_TUNING_CHAPTER)[:5])  # 距離の上位には目次を入れない
    unlimited = select(["term:structured-output"], [], near, top_k=100, cap=100)

    # 候補（距離の上位と、用語に触れる目次以外のページ）がちょうど入る件数で選ぶ。補充は起きない
    result = select(["term:structured-output"], [], near, top_k=unlimited["pool_size"], cap=100)

    picked = {item["page_id"] for item in result["selected"]}
    assert TOC_PAGES <= DATA.mentions["term:structured-output"]  # 目次も、この用語に言及している
    assert not TOC_PAGES & picked
    assert not TOC_PAGES & {item["page_id"] for item in result["skipped_by_cap"]}
    assert {item["reason"] for item in result["selected"]} <= {"距離", "確保:用語:構造化出力"}
    assert {"lec02-main-p053", "lec03-main-p033"} <= picked  # 用語の少ないページは、用語経由で入る
    assert picked >= DATA.mentions["term:structured-output"] - TOC_PAGES
    # 候補が尽きても top_k に届かなければ、目次も補充で入る。用語経由（確保・距離）ではない
    assert {item["reason"] for item in unlimited["selected"] if item["page_id"] in TOC_PAGES} == {"補充"}


def test_contents_pages_come_in_when_they_are_among_the_nearest_pages():
    near = near_pages(sorted(TOC_PAGES))  # 目次を距離の上位（vector_pool の中）に入れる

    result = select(["term:structured-output"], [], near, top_k=100, cap=100)

    assert TOC_PAGES <= {item["page_id"] for item in result["selected"]}


def test_a_larger_hub_terms_lets_contents_pages_in_through_terms():
    near = near_pages(section_pages(FINE_TUNING_CHAPTER)[:5])

    result = select(["term:structured-output"], [], near, top_k=100, cap=100, hub_terms=18)

    reasons = {item["page_id"]: item["reason"] for item in result["selected"]}
    # 17 個と 9 個の用語は 18 未満なので、用語経由の候補に入る（補充ではなく、距離の順に選ばれる）
    assert {reasons[page_id] for page_id in TOC_PAGES} == {"距離"}


def test_quantization_is_replaced_by_its_related_term_qlora():
    result = select(["term:quantization"])

    assert result["groups"] == {"term:quantization": ["term:qlora"]}
    assert result["home"] == {"term:quantization": [2]}  # QLoRA は第2回に6ページ、第1回に1ページ
    secured = next(item for item in result["selected"] if item["reason"] == "確保:用語:量子化")
    assert secured["page_id"] in DATA.mentions["term:qlora"] and DATA.lecture_of[secured["page_id"]] == 2


def test_a_term_with_no_mentions_and_no_substitute_is_dropped_but_listed():
    result = select(["term:quantization"], [3])  # 第3回に、量子化も QLoRA の言及も無い

    assert result["groups"] == {"term:quantization": []}
    assert result["home"] == {"term:quantization": []}
    assert result["allowed_size"] == 71
    assert not any(item["reason"].startswith("確保") for item in result["selected"])  # 確保は無く、距離と補充だけ


@pytest.mark.parametrize("top_k", [0, 1, 3, 5, 10])
def test_selection_is_at_most_top_k_pages_sorted_by_distance(top_k):
    near = near_pages(section_pages(FINE_TUNING_CHAPTER)[:5])

    result = select(["term:fine-tuning", "term:structured-output"], [2, 3], near, top_k=top_k)

    chosen = result["selected"]
    assert len(chosen) == top_k  # 候補は十分にあるので、ちょうど top_k 件
    distances = [item["distance"] for item in chosen]
    assert distances == sorted(distances)  # 選んだ順ではなく、距離の昇順


def test_terms_with_fewer_mentions_are_secured_first():
    result = select(["term:fine-tuning", "term:structured-output"], [], top_k=1)

    # ファインチューニングは39ページ、構造化出力は12ページ。少ない構造化出力が先で、top_k に達して止まる
    assert [item["reason"] for item in result["selected"]] == ["確保:用語:構造化出力"]


def test_one_page_that_mentions_two_terms_secures_both_of_them():
    both = sorted(DATA.mentions["term:lora"] & DATA.mentions["term:qlora"])
    only_qlora = sorted(page_id for page_id in DATA.mentions["term:qlora"] - DATA.mentions["term:lora"]
                        if DATA.lecture_of[page_id] == 2)
    # QLoRA だけに触れるページのほうが近いが、LoRA と QLoRA の両方に触れるページが先に確保されている
    near = {only_qlora[0]: 0.10, both[0]: 0.11}

    result = select(["term:qlora", "term:lora"], [], near, top_k=5)

    reasons = {item["page_id"]: item["reason"] for item in result["selected"]}
    assert reasons[both[0]] == "確保:用語:LoRA"  # 言及の少ない LoRA から確保する。このページは QLoRA にも触れる
    assert reasons[only_qlora[0]] == "距離"  # QLoRA の確保は不要。確保は1件だけ
    assert [reason for reason in reasons.values() if reason.startswith("確保:用語:")] == ["確保:用語:LoRA"]


def test_each_requested_lecture_gets_its_nearest_page_even_when_far_away():
    near = near_pages(section_pages(FINE_TUNING_CHAPTER)[:5])

    result = select([], [2, 3], near)

    secured = [item for item in result["selected"] if item["reason"] == "確保:第3回"]
    assert len(secured) == 1
    third_lecture = [page_id for page_id in fake_distances(near) if DATA.lecture_of[page_id] == 3]
    assert secured[0]["page_id"] == min(third_lecture, key=lambda page_id: (fake_distances(near)[page_id], page_id))
    assert lectures_of(result) == {2, 3}
    assert result["allowed_size"] == 141  # 第2回の70ページ + 第3回の71ページ


def test_one_lecture_limits_the_candidates_without_securing_a_page():
    near = near_pages(section_pages("lec03-main-s037")[:5])  # 距離が近いのは第3回

    result = select([], [2], near)

    assert result["allowed_size"] == 70
    assert lectures_of(result) == {2}  # 補充も、回の指定の中から選ぶ
    assert len(result["selected"]) == 5
    assert not any(item["reason"].startswith("確保") for item in result["selected"])


def test_without_terms_or_lectures_the_nearest_pages_fill_in_with_the_section_cap():
    chapter = section_pages(FINE_TUNING_CHAPTER)
    sections = ("lec03-main-s005", "lec03-main-s018", "lec03-main-s022")  # 第3回の、別々の3つの章
    other = [section_pages(section)[0] for section in sections]
    near = {**near_pages(chapter[:5]), **near_pages(other, start=0.20)}

    result = select([], [], near, top_k=5, cap=2, vector_pool=8)

    assert [item["page_id"] for item in result["selected"]] == chapter[:2] + other  # 同じ章は2件まで
    assert all(item["reason"] == "距離" for item in result["selected"])
    assert [item["page_id"] for item in result["skipped_by_cap"]] == chapter[2:5]
    assert result["pool_size"] == 8 and result["allowed_size"] == 214
    assert result["groups"] == {} and result["home"] == {}


def test_the_nearest_pages_in_one_section_are_topped_up_with_pages_from_other_sections():
    chapter = section_pages(FINE_TUNING_CHAPTER)
    near = near_pages(chapter[:5])  # 距離の上位5件が、すべて同じ章

    result = select([], [], near, top_k=5, cap=2, vector_pool=5)

    chosen = result["selected"]
    assert len(chosen) == 5  # 章の上限で外れても、5件に届く
    assert [item["page_id"] for item in chosen if item["reason"] == "距離"] == chapter[:2]
    topped_up = [item for item in chosen if item["reason"] == "補充"]
    assert len(topped_up) == 3
    assert all(item["page_id"] not in near and item["section"] != FINE_TUNING_CHAPTER for item in topped_up)  # 別の章
    assert set(chapter[2:5]) <= {item["page_id"] for item in result["skipped_by_cap"]}  # 上限で外れた分は、記録される
    assert result["pool_size"] == 5  # 補充したページは、候補（pool）の外


def test_the_top_up_also_respects_the_section_cap_and_reports_the_pages_it_skipped():
    same_section = section_pages("lec03-main-s037")[:4]
    other_section = section_pages("lec03-main-s042")[0]
    near = {"lec02-main-p045": 0.10, **near_pages(same_section, start=0.20), other_section: 0.30}

    result = select([], [], near, top_k=5, cap=2, vector_pool=1)  # 候補は、距離の上位1件だけ

    reasons = {item["page_id"]: item["reason"] for item in result["selected"]}
    assert reasons["lec02-main-p045"] == "距離"
    assert reasons[same_section[0]] == reasons[same_section[1]] == "補充"
    assert same_section[2] not in reasons and same_section[3] not in reasons  # 同じ章の3件目と4件目は外れる
    assert reasons[other_section] == "補充"
    assert len(reasons) == 5
    # 外したページの記録は、top_k に達して止まるまでの分だけ
    assert [item["page_id"] for item in result["skipped_by_cap"]] == same_section[2:]


def test_when_every_page_is_in_one_section_the_cap_is_lifted_to_reach_top_k():
    page_ids = [f"lec01-main-p{number:03d}" for number in range(1, 11)]
    data = lec05.GraphData(  # 10ページがすべて、同じ1つの章（偽のデータ）
        terms={}, patterns=[], mentions={}, related={},
        page_terms={page_id: set() for page_id in page_ids},
        section_of={page_id: "lec01-main-s001" for page_id in page_ids},
        lecture_of={page_id: 1 for page_id in page_ids},
    )
    nearest_first = page_ids[::-1]  # 番号が大きいページほど近い。距離の順が、ページ番号の順と逆になる
    distances = {page_id: 0.1 * rank for rank, page_id in enumerate(nearest_first)}

    result = lec05.select_pages(distances, [], [], data, top_k=5, cap=2, vector_pool=3, hub_terms=9)

    assert [(item["page_id"], item["reason"]) for item in result["selected"]] == list(zip(nearest_first[:5], [
        "距離", "距離", "補充(上限なし)", "補充(上限なし)", "補充(上限なし)",
    ]))
    # 上限なしで選んだページは、結果に入ったので「外した」扱いにしない
    assert [item["page_id"] for item in result["skipped_by_cap"]] == nearest_first[5:]
    assert result["pool_size"] == 3 and result["allowed_size"] == 10


def test_pages_in_the_pool_are_chosen_before_the_top_up_even_when_they_are_farther():
    sections = ("lec01-main-s003", "lec01-main-s029", "lec01-main-s057")  # 構造化出力に触れない、第1回の3つの章
    others = [section_pages(section)[0] for section in sections]
    assert not set(others) & DATA.mentions["term:structured-output"]
    near = {"lec02-main-p045": 0.10, **near_pages(others, start=0.20)}  # 候補の外のページのほうが、用語のページより近い

    result = select(["term:structured-output"], [], near, top_k=5, vector_pool=1)

    assert not set(others) & {item["page_id"] for item in result["selected"]}  # 近くても、候補の外のページは入らない
    assert not any(item["reason"].startswith("補充") for item in result["selected"])
    assert max(item["distance"] for item in result["selected"]) > max(near[page_id] for page_id in others)  # もっと遠い候補
    # 候補が尽きたところで初めて、候補の外で最も近いページが補充で入る
    exact = select(["term:structured-output"], [], near, top_k=result["pool_size"], cap=100, vector_pool=1)
    one_more = select(["term:structured-output"], [], near, top_k=result["pool_size"] + 1, cap=100, vector_pool=1)
    assert "補充" not in {item["reason"] for item in exact["selected"]}
    assert [item["page_id"] for item in one_more["selected"] if item["reason"] == "補充"] == [others[0]]


@pytest.mark.parametrize("lectures", [[], [2], [2, 3]])
@pytest.mark.parametrize("top_k", [1, 5, 50, 300])
def test_the_result_has_min_of_top_k_and_allowed_pages(top_k, lectures):
    result = select(["term:fine-tuning"], lectures, top_k=top_k)

    picked = {item["page_id"] for item in result["selected"]}
    assert len(picked) == len(result["selected"]) == min(top_k, result["allowed_size"])  # 候補が尽きても、補充で top_k 件
    assert not picked & {item["page_id"] for item in result["skipped_by_cap"]}  # 選んだページは、外した扱いにしない
    assert not lectures or {DATA.lecture_of[page_id] for page_id in picked} <= set(lectures)  # 補充も、回の指定の中から
    distances = [item["distance"] for item in result["selected"]]
    assert distances == sorted(distances)


def test_select_pages_returns_the_documented_keys():
    result = select(["term:fine-tuning"], [2], near_pages(FINE_TUNING_PAGES_IN_CHAPTER[:2]))

    assert set(result) == {"selected", "skipped_by_cap", "groups", "home", "pool_size", "allowed_size"}
    assert set(result["selected"][0]) == {"page_id", "distance", "reason", "section"}
    assert result["selected"][0]["section"] == DATA.section_of[result["selected"][0]["page_id"]]
    assert result["groups"] == {"term:fine-tuning": ["term:fine-tuning"]}
    assert result["home"] == {"term:fine-tuning": [2]}
    json.dumps(result)  # TRACE に入れて JSON にできる


def test_pages_missing_from_the_graph_data_are_ignored():
    distances = {**fake_distances(), "lec09-main-p001": 0.0}  # 対象外のページの距離が混ざっても落ちない

    result = lec05.select_pages(distances, [], [], DATA, 5, 2, 5, 9)

    assert "lec09-main-p001" not in {item["page_id"] for item in result["selected"]}
    assert result["allowed_size"] == 214


# --- 用語から始める Graph 検索を、本物のDBと偽の埋め込みで動かす ---


def test_all_distances_embeds_once_and_returns_every_page(embed_stored_page):
    distances = lec05.all_distances("  構造化出力とは  ")

    assert embed_stored_page == [("構造化出力とは", "RETRIEVAL_QUERY")]  # 1回だけ。前後の空白は取る
    assert set(distances) == set(PAGE_BY_ID) and len(distances) == 214
    assert min(distances, key=distances.get) == STORED_PAGE
    assert distances[STORED_PAGE] == pytest.approx(0, abs=1e-6)  # 保存された埋め込みそのものなので、距離 0


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_all_distances_rejects_an_empty_query_before_embedding(monkeypatch, query):
    monkeypatch.setattr("app.retrieval.embed_text", forbidden)

    with pytest.raises(ValueError, match="検索語が空です。"):
        lec05.all_distances(query)


def test_all_distances_stops_before_embedding_when_the_index_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("app.retrieval.RAG_DB", tmp_path / "none.sqlite3")
    monkeypatch.setattr("app.retrieval.embed_text", forbidden)

    with pytest.raises(FileNotFoundError, match="検索DBがありません。"):  # upstream の search と同じ順序
        lec05.all_distances("質問")


def test_search_term_graph_returns_rows_shaped_like_search_results(embed_stored_page):
    rows = lec05.search_term_graph("第3回の構造化出力とは")

    assert len(embed_stored_page) == 1  # 埋め込みは1回だけ
    expected_keys = list(retrieval.search("構造化出力とは")[0])
    assert 0 < len(rows) <= 5
    assert all(list(row) == expected_keys for row in rows)  # キーも順序も retrieval.search() の行と同じ
    assert rows[0]["page_id"] == STORED_PAGE and rows[0]["distance"] == pytest.approx(0, abs=1e-6)
    distances = [row["distance"] for row in rows]
    assert distances == sorted(distances)
    assert all(row["lecture"] == 3 for row in rows)  # 「第3回」の指定で、第3回のページに絞る
    assert all(row["content"] == PAGE_BY_ID[row["page_id"]]["content"] for row in rows)


def test_search_term_graph_adds_one_entry_to_the_trace(embed_stored_page):
    rows = lec05.search_term_graph("第3回の構造化出力とは")

    (trace,) = lec05.TRACE
    assert set(trace) == {"tool", "query", "terms", "lectures", "groups", "home", "pool_size", "allowed_size",
                          "selected", "skipped_by_cap", "params"}
    assert trace["tool"] == "term_graph" and trace["query"] == "第3回の構造化出力とは"
    assert trace["terms"] == [{"id": "term:structured-output", "title": "構造化出力"}]
    assert trace["lectures"] == [3]
    assert trace["groups"] == {"term:structured-output": ["term:structured-output"]}
    assert trace["allowed_size"] == 71
    assert trace["params"] == {"top_k": 5, "cap": 2, "vector_pool": 5, "hub_terms": 9}
    assert [item["page_id"] for item in trace["selected"]] == [row["page_id"] for row in rows]
    assert {item["reason"] for item in trace["selected"]} <= {"距離", "確保:用語:構造化出力"}
    json.dumps(trace)  # ランナーが JSON にできる


def test_search_term_graph_still_returns_five_pages_when_no_term_is_found(embed_stored_page):
    # 用語集に無い言葉。偽の埋め込みでは、基本RAGの上位5件が同じ章（Part 9）に固まる
    baseline = retrieval.search("コンパクション セッション")
    assert len({DATA.section_of[hit["page_id"]] for hit in baseline}) == 1

    rows = lec05.search_term_graph("コンパクション セッション")

    (trace,) = lec05.TRACE
    assert trace["terms"] == [] and trace["lectures"] == []
    assert len(rows) == len(baseline) == 5  # 章の上限で外れても、基本RAGと同じ5件
    assert "補充" in {item["reason"] for item in trace["selected"]}
    assert max(Counter(DATA.section_of[row["page_id"]] for row in rows).values()) <= 2  # 補充も、章の上限を守る
    assert {row["page_id"] for row in rows} != {hit["page_id"] for hit in baseline}  # 選び方が違うので、中身は同じではない


def test_search_term_graph_rejects_an_empty_query(embed_stored_page):
    with pytest.raises(ValueError, match="検索語が空です。"):
        lec05.search_term_graph("  ")
    assert lec05.TRACE == []


@pytest.mark.parametrize("search", [lec05.search_with_neighbors, lec05.search_term_graph],
                         ids=["neighbors", "term_graph"])
def test_trace_is_one_list_that_the_runner_can_clear_between_trials(monkeypatch, search):
    runner_view = lec05.TRACE  # ランナーが持つ参照
    monkeypatch.setattr("app.retrieval.search", lambda query: lec02_hits(52))
    monkeypatch.setattr("app.retrieval_lec05.all_distances", lambda query: fake_distances())

    search("LoRA")
    assert len(runner_view) == 1
    runner_view.clear()  # ランナーは試行ごとに clear する
    search("LoRA")

    assert lec05.TRACE is runner_view and len(runner_view) == 1  # 作り直さず、同じリストに足す


@pytest.mark.parametrize(("name", "key", "default", "changed"), [
    ("TERM_GRAPH_TOP_K", "top_k", 5, 3),
    ("TERM_GRAPH_SECTION_CAP", "cap", 2, 3),
    ("TERM_GRAPH_VECTOR_POOL", "vector_pool", 5, 7),
    ("TERM_GRAPH_HUB_TERMS", "hub_terms", 9, 20),
])
def test_each_term_graph_constant_is_read_on_every_call(monkeypatch, name, key, default, changed):
    monkeypatch.setattr("app.retrieval_lec05.all_distances", lambda query: fake_distances())

    lec05.search_term_graph("ファインチューニングとは")
    assert lec05.TRACE[-1]["params"][key] == default

    monkeypatch.setattr(f"app.retrieval_lec05.{name}", changed)  # 引数なしの呼び出しに効く
    lec05.search_term_graph("ファインチューニングとは")
    assert lec05.TRACE[-1]["params"][key] == changed

    lec05.search_term_graph("ファインチューニングとは", **{key: 99})  # 引数のほうが優先
    assert lec05.TRACE[-1]["params"][key] == 99


def test_overriding_the_section_cap_changes_the_pages_returned(monkeypatch):
    near = near_pages(FINE_TUNING_PAGES_IN_CHAPTER[:6])
    monkeypatch.setattr("app.retrieval_lec05.all_distances", lambda query: fake_distances(near))

    def pages_from_the_chapter():
        rows = lec05.search_term_graph("第2回のファインチューニングとは")
        return [row["page_id"] for row in rows if DATA.section_of[row["page_id"]] == FINE_TUNING_CHAPTER]

    assert len(pages_from_the_chapter()) == 2  # 既定の上限は2
    monkeypatch.setattr("app.retrieval_lec05.TERM_GRAPH_SECTION_CAP", 3)
    assert len(pages_from_the_chapter()) == 3


# --- ツール用の関数 ---

TOOL_CASES = [
    pytest.param(lec05.search_materials_with_neighbors, "search_with_neighbors", id="with_neighbors"),
    pytest.param(lec05.search_materials_term_graph, "search_term_graph", id="term_graph"),
]


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_returns_the_results_as_json_like_search_materials(monkeypatch, tool, inner):
    rows = [{"page_id": "lec02-main-p052", "lecture": 2, "page": 52, "content": "元の重みは固定", "distance": 0.1}]
    monkeypatch.setattr(f"app.retrieval_lec05.{inner}", lambda query: rows)
    monkeypatch.setattr("app.retrieval.search", lambda query: rows)

    answer = tool("LoRAとは")

    assert isinstance(answer, str)
    assert json.loads(answer) == {"results": rows}
    assert answer == json.dumps({"results": rows}, ensure_ascii=False)  # 日本語はそのまま
    assert answer == tools.search_materials("LoRAとは")  # upstream の search_materials と同じ形


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_reports_an_empty_query_as_an_error_payload(tool, inner):
    assert json.loads(tool("  ")) == {"error": "検索語が空です。", "results": []}


@pytest.mark.parametrize("error", [
    ValueError("検索語が空です。"),
    FileNotFoundError("検索DBがありません。"),
    sqlite3.OperationalError("database is locked"),
])
@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_turns_expected_errors_into_an_error_payload(monkeypatch, tool, inner, error):
    def fail(query):
        raise error

    monkeypatch.setattr(f"app.retrieval_lec05.{inner}", fail)

    assert json.loads(tool("LoRAとは")) == {"error": str(error), "results": []}


@pytest.mark.parametrize("error", [
    genai_errors.ClientError(429, {"error": {"message": "quota"}}),
    genai_errors.ClientError(403, {"error": {"message": "denied"}}),
    genai_errors.ServerError(503, {"error": {"message": "busy"}}),
])
@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_turns_an_api_error_into_the_japanese_message(monkeypatch, tool, inner, error):
    def fail(query):
        raise error

    monkeypatch.setattr(f"app.retrieval_lec05.{inner}", fail)

    assert json.loads(tool("LoRAとは")) == {"error": api_error_message(error.code), "results": []}


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_does_not_swallow_unexpected_errors(monkeypatch, tool, inner):
    def fail(query):
        raise RuntimeError("想定外の失敗")

    monkeypatch.setattr(f"app.retrieval_lec05.{inner}", fail)

    with pytest.raises(RuntimeError, match="想定外の失敗"):
        tool("LoRAとは")


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_lets_a_missing_api_key_stop_the_run_like_search_materials(monkeypatch, tool, inner):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SystemExit):
        tools.search_materials("LoRAとは")
    with pytest.raises(SystemExit):
        tool("LoRAとは")


# --- ツール定義：モデルから見える形を search_materials と同じにする ---


def registered_search_materials() -> Tool:
    """app.agent.agent に登録済みの search_materials の Tool。"""
    toolset = next(ts for ts in agent.toolsets if isinstance(ts, FunctionToolset) and "search_materials" in ts.tools)
    return toolset.tools["search_materials"]


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_definition_matches_the_registered_search_materials(tool, inner):
    base = registered_search_materials()

    ours = Tool(tool, name="search_materials", description=base.description)

    assert ours.tool_def.name == base.tool_def.name == "search_materials"
    assert ours.tool_def.description == base.tool_def.description
    assert ours.tool_def.parameters_json_schema == base.tool_def.parameters_json_schema
    assert ours.tool_def == base.tool_def  # 戻り値の型（str）まで含めて同じ
    # 説明文を渡さなくても、__doc__ から同じ説明になる
    assert Tool(tool, name="search_materials").tool_def == base.tool_def


@pytest.mark.parametrize(("tool", "inner"), TOOL_CASES)
def test_tool_docstring_is_exactly_the_one_of_search_materials(tool, inner):
    assert tool.__doc__ == tools.search_materials.__doc__
    assert tool.__doc__ and "第1〜3回の本講義資料を検索" in tool.__doc__


# --- CLI ---


def test_cli_terms_prints_the_terms_and_lectures_without_api_key_or_database(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.retrieval.RAG_DB", tmp_path / "none.sqlite3")  # DB が無くても動く
    monkeypatch.setattr("app.retrieval.embed_text", forbidden)

    lec05.main(["--terms", "第2回のファインチューニングと第3回の構造化出力"])

    out = capsys.readouterr().out
    assert "拾った用語: ファインチューニング、構造化出力" in out
    assert "回の指定: 第2回、第3回" in out
    assert lec05.TRACE == []


def test_cli_terms_says_none_when_nothing_is_found(capsys):
    lec05.main(["--terms", "今日はいい天気です"])

    out = capsys.readouterr().out
    assert "拾った用語: なし" in out and "回の指定: なし" in out


def test_cli_without_options_prints_the_upstream_search_results(monkeypatch, capsys):
    monkeypatch.setattr("app.retrieval.search", lambda query: lec02_hits(52, 51))

    lec05.main(["LoRAとは"])

    out = capsys.readouterr().out
    assert "検索結果 1 / hit 1 / d=0.100" in out and "検索結果 2 / hit 2 / d=0.200" in out
    page = PAGE_BY_ID["lec02-main-p052"]
    for line in (f"講義: {page['document_title']}", f"章: {page['section']}", f"タイトル: {page['title']}",
                 "ページ: 52", page["content"]):
        assert line in out  # upstream の CLI と同じ項目


def test_cli_neighbors_labels_hits_and_the_pages_next_to_them(monkeypatch, capsys):
    monkeypatch.setattr("app.retrieval.search", lambda query: lec02_hits(52, 51))

    lec05.main(["--neighbors", "LoRAとは"])

    headings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("検索結果")]
    assert headings == [
        "検索結果 1 / lec02-main-p051 の隣",
        "検索結果 2 / hit 2 / d=0.200",
        "検索結果 3 / hit 1 / d=0.100",
        "検索結果 4 / lec02-main-p052 の隣",
    ]


def test_cli_brief_prints_one_line_per_page(monkeypatch, capsys):
    monkeypatch.setattr("app.retrieval.search", lambda query: lec02_hits(52, 51))

    lec05.main(["--neighbors", "--brief", "LoRAとは"])

    def title(number):
        return PAGE_BY_ID[f"lec02-main-p{number:03d}"]["title"]

    assert capsys.readouterr().out.splitlines() == [  # 順・講義・PDF p.・タイトル・distance・role
        f" 1  第2回  PDF p.50  {title(50)}  d=-  neighbor",
        f" 2  第2回  PDF p.51  {title(51)}  d=0.200  hit",
        f" 3  第2回  PDF p.52  {title(52)}  d=0.100  hit",
        f" 4  第2回  PDF p.53  {title(53)}  d=-  neighbor",
    ]


def test_cli_top_k_changes_rag_top_k_only_during_the_run(monkeypatch, capsys):
    before = retrieval.RAG_TOP_K
    seen = []

    def fake_search(query):
        seen.append(retrieval.RAG_TOP_K)
        return lec02_hits(52)

    monkeypatch.setattr("app.retrieval.search", fake_search)

    lec05.main(["--top-k", "15", "LoRAとは"])

    assert seen == [15]
    assert retrieval.RAG_TOP_K == before  # 実行が終わったら戻る


def test_cli_top_k_is_restored_even_when_the_search_fails(monkeypatch):
    before = retrieval.RAG_TOP_K

    def fail(query):
        raise RuntimeError("検索に失敗")

    monkeypatch.setattr("app.retrieval.search", fail)

    with pytest.raises(RuntimeError):
        lec05.main(["--top-k", "15", "LoRAとは"])
    assert retrieval.RAG_TOP_K == before


@pytest.mark.parametrize("flags", [
    ["--neighbors", "--term-graph"],
    ["--neighbors", "--top-k", "3"],
    ["--top-k", "3", "--term-graph"],
])
def test_cli_modes_cannot_be_combined(flags):
    with pytest.raises(SystemExit) as stopped:
        lec05.main([*flags, "LoRAとは"])

    assert stopped.value.code == 2


def test_cli_rejects_a_top_k_below_one():
    with pytest.raises(SystemExit) as stopped:
        lec05.main(["--top-k", "0", "LoRAとは"])

    assert stopped.value.code == 2


def test_cli_term_graph_prints_the_reasons_the_terms_and_the_pages_the_cap_dropped(embed_stored_page, capsys):
    lec05.main(["--term-graph", "第3回の構造化出力と第2回のファインチューニング"])

    out = capsys.readouterr().out
    headings = [line for line in out.splitlines() if line.startswith("検索結果")]
    assert 0 < len(headings) <= 5
    assert any(line.endswith("/ 確保:用語:構造化出力") for line in headings)
    assert any(line.endswith("/ 確保:用語:ファインチューニング") for line in headings)
    assert "拾った用語: 構造化出力、ファインチューニング" in out
    assert "回の指定: 第2回、第3回" in out
    assert "章の上限（2件）で外したページ:" in out
    assert len(embed_stored_page) == 1
