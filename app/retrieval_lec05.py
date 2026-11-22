"""第5回 宿題: 取得処理の改良。実装は前後ページ、発展は用語から始める Graph 検索。

upstream の retrieval.py は書き換えず、ここに足す。
ツール用の関数（search_materials_*）は upstream の search_materials と同じ JSON を返す。
比較用ランナーは、名前と説明文を search_materials と同じにして登録する。比べたいのは取得処理の違いだけだから。
Graph 検索は、候補が尽きて top_k 件に届かないとき、残りのページを距離順に補う（基本RAGと同じ件数の枠で、選び方だけを比べるため）。

使い方:
    uv run python -m app.retrieval_lec05 "検索語"                # 通常の検索（upstream の search と同じ。比較用）
    uv run python -m app.retrieval_lec05 --neighbors "検索語"    # 前後1ページを足す
    uv run python -m app.retrieval_lec05 --top-k 15 "検索語"     # 上位N件
    uv run python -m app.retrieval_lec05 --term-graph "検索語"   # 用語から始める Graph 検索
    --brief は1ページ1行、--terms は拾った用語と回の指定だけを表示する（API も DB も使わない）。
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import sqlite_vec
from google.genai.errors import APIError

from app import retrieval, tools
from app.config import api_error_message

# 比較用ランナーが、試行の間だけ書き換える。関数の既定値には束縛せず、呼ぶたびに読む。
NEIGHBOR_WINDOW = 1          # 実装：前後に足すページ数
TERM_GRAPH_TOP_K = 5         # 発展：返すページ数（基本RAGの RAG_TOP_K と同じ枠）
TERM_GRAPH_SECTION_CAP = 2   # 発展：同じ章（section）から選ぶ上限
TERM_GRAPH_VECTOR_POOL = 5   # 発展：候補に必ず入れる、距離の上位件数
TERM_GRAPH_HUB_TERMS = 9     # 発展：用語がこの数以上のページ（目次）は、用語経由の候補から外す
TRACE: list[dict] = []       # 呼び出しごとの取得の内訳。ランナーが試行ごとに clear する


@lru_cache(maxsize=8)
def _load_pages_cached(source: str, mtime_ns: int, document_ids: tuple[str, ...]) -> tuple[dict, ...]:
    """対象ページの一覧を読む。引数はキャッシュの鍵で、資料JSONか対象資料が変われば読み直す。"""
    return tuple(retrieval.load_pages(Path(source)))


def _pages() -> list[dict]:
    """対象ページの一覧を返す。資料JSONのパスと更新時刻、対象資料が同じなら、読み直さない。"""
    source = Path(retrieval.RAG_SOURCE)
    return list(_load_pages_cached(str(source), source.stat().st_mtime_ns, tuple(retrieval.RAG_DOCUMENT_IDS)))


# --- 実装：前後ページ ---


def _resolve_window(window: int | None) -> int:
    """None なら NEIGHBOR_WINDOW（呼ぶたびに読む）にする。負の値は ValueError。"""
    window = NEIGHBOR_WINDOW if window is None else window
    if window < 0:
        raise ValueError("前後に足すページ数は0以上にしてください。")
    return window


def _page_key(page: dict) -> tuple[str, int] | None:
    """(資料ID, ページ番号)を返す。どちらかが無ければ None。"""
    document_id, number = page.get("document_id"), page.get("page")
    return None if document_id is None or number is None else (document_id, number)


def add_neighbors(hits: list[dict], window: int | None = None, pages: list[dict] | None = None) -> list[dict]:
    """検索結果の各ページに、同じ資料の前後 window ページを足して、平らなリストで返す。

    pages は前後を探す対象（None なら対象資料の全ページ）。ここに無いページは足さない。
    この1つの規則で、資料の端と、資料またぎ、対象外の資料（exercise など）が除かれる。
    各要素はページの8項目と distance（neighbor は None）・role（hit か neighbor）・
    neighbor_of（neighbor を隣に持つ hit の page_id。hit の順位順）。
    並びは、hit と隣を資料ごとにページが連続する塊に分け、最も上位の hit の順位で塊を並べる（塊の中はページ順）。
    例：hit が lec02 の p.52, 51, 60, 61, 46 の順で window=1 なら、[50-53][59-62][45-47]。
    """
    window = _resolve_window(window)
    rows = [{**hit, "distance": hit.get("distance"), "role": "hit", "neighbor_of": []} for hit in hits]
    if window == 0:
        return rows

    index = {(page["document_id"], page["page"]): page for page in (_pages() if pages is None else pages)}
    members: dict[tuple[str, int], dict] = {}  # (資料ID, ページ番号) → 行。hit と neighbor を入れる
    best: dict[tuple[str, int], int] = {}  # 同じ鍵 → そのページに関わる最上位 hit の順位
    singles = []  # 資料IDかページ番号が無い hit は、隣を付けずにそのまま残す
    for rank, row in enumerate(rows):
        key = _page_key(row)
        if key is None:
            singles.append((rank, row))
        elif key not in members:  # 同じページの hit が重なったら、上位のほうだけ残す
            members[key] = row
            best[key] = rank

    hit_keys = list(members)  # hit の順位順
    for key in hit_keys:
        document_id, number = key
        for step in range(1, window + 1):
            for other in (number - step, number + step):
                neighbor = (document_id, other)
                if neighbor not in index or neighbor in hit_keys:  # hit は hit のまま。無いページは足さない
                    continue
                if neighbor not in members:
                    members[neighbor] = {**index[neighbor], "distance": None, "role": "neighbor", "neighbor_of": []}
                    best[neighbor] = best[key]
                members[neighbor]["neighbor_of"].append(members[key].get("page_id"))

    chunks = [(rank, [row]) for rank, row in singles]  # (塊の最上位 hit の順位, 塊の行)
    for document_id in sorted({key[0] for key in members}):
        run: list[tuple[str, int]] = []
        for key in sorted(key for key in members if key[0] == document_id):
            if run and key[1] != run[-1][1] + 1:
                chunks.append((min(best[k] for k in run), [members[k] for k in run]))
                run = []
            run.append(key)
        chunks.append((min(best[k] for k in run), [members[k] for k in run]))
    chunks.sort(key=lambda chunk: chunk[0])
    return [row for _, run_rows in chunks for row in run_rows]


def search_with_neighbors(query: str, window: int | None = None) -> list[dict]:
    """通常の検索結果に、前後のページを足して返す。内訳は TRACE に残す。"""
    window = _resolve_window(window)
    hits = retrieval.search(query)
    results = add_neighbors(hits, window)
    TRACE.append({
        "tool": "neighbors", "query": query, "window": window,
        "hits": [hit.get("page_id") for hit in hits],
        "neighbors": [row["page_id"] for row in results if row["role"] == "neighbor"],
        "pages": len(results),
    })
    return results


# --- 発展：用語から始める Graph 検索 ---


@dataclass(frozen=True)
class GraphData:
    """用語グラフの下ごしらえ。ページと用語の対応を、引きやすい形にしたもの（読み取り専用）。"""

    terms: dict[str, dict]  # term_id → {"title", "aliases"}
    patterns: list[tuple[str, str, re.Pattern]]  # 長い名前から順に（正規化した名前, term_id, 照合の正規表現）
    mentions: dict[str, set[str]]  # term_id → その用語に触れるページ（対象のページだけ）
    related: dict[str, set[str]]  # term_id → 関連する用語（片方向の関係も無向として扱う）
    page_terms: dict[str, set[str]]  # page_id → そのページが触れる用語
    section_of: dict[str, str]  # page_id → 親の章のID（章が無い冒頭のページは資料ID）
    lecture_of: dict[str, int]  # page_id → 回


def _normalize(text: str) -> str:
    """全角・半角と、大文字・小文字の違いをなくす。"""
    return unicodedata.normalize("NFKC", text).casefold()


def _name_pattern(name: str) -> re.Pattern:
    """用語の名前を探す正規表現。英数字だけの名前は、前後が英数字なら一致させない（api の中の pi を拾わない）。"""
    body = re.escape(name)
    return re.compile(rf"(?<![0-9a-z]){body}(?![0-9a-z])" if name.isascii() else body)


@lru_cache(maxsize=4)
def _build_graph_data(graph: str, graph_mtime_ns: int, source: str, source_mtime_ns: int,
                      document_ids: tuple[str, ...]) -> GraphData:
    """graph.json から GraphData を作る。引数はキャッシュの鍵（mtime と対象資料が変われば作り直す）。"""
    data = json.loads(Path(graph).read_text(encoding="utf-8"))
    pages = _load_pages_cached(source, source_mtime_ns, document_ids)
    kinds = {node["id"]: node["kind"] for node in data["nodes"]}
    terms = {node["id"]: {"title": node["title"], "aliases": list(node.get("aliases", []))}
             for node in data["nodes"] if node["kind"] == "term"}

    names = []
    for term_id, term in terms.items():
        for name in dict.fromkeys(_normalize(name) for name in [term["title"], *term["aliases"]]):
            if name:
                names.append((name, term_id))
    names.sort(key=lambda item: -len(item[0]))  # 長い名前から（同じ長さなら登録順）
    patterns = [(name, term_id, _name_pattern(name)) for name, term_id in names]

    mentions = {term_id: set() for term_id in terms}
    related = {term_id: set() for term_id in terms}
    page_terms = {page["page_id"]: set() for page in pages}
    section_of = {page["page_id"]: page["document_id"] for page in pages}  # 親が章なら、下で章のIDに置き換える
    for edge in data["edges"]:
        source_id, target_id, relation = edge["source"], edge["target"], edge["relation"]
        if relation == "mentions" and source_id in page_terms and target_id in terms:
            mentions[target_id].add(source_id)
            page_terms[source_id].add(target_id)
        elif relation == "related_term" and source_id in terms and target_id in terms:
            related[source_id].add(target_id)
            related[target_id].add(source_id)
        elif relation == "contains" and target_id in section_of and kinds.get(source_id) == "section":
            section_of[target_id] = source_id
    lecture_of = {page["page_id"]: int(page["document_id"][3:5]) for page in pages}
    return GraphData(terms, patterns, mentions, related, page_terms, section_of, lecture_of)


def graph_data() -> GraphData:
    """用語グラフの下ごしらえを返す。graph.json と資料JSONのパス・更新時刻、対象資料が同じなら、作り直さない。"""
    graph, source = Path(retrieval.RAG_GRAPH), Path(retrieval.RAG_SOURCE)
    return _build_graph_data(str(graph), graph.stat().st_mtime_ns, str(source), source.stat().st_mtime_ns,
                             tuple(retrieval.RAG_DOCUMENT_IDS))


def match_terms(query: str) -> list[str]:
    """質問に出てくる用語を、出現位置の順に、重複なしで返す。

    長い名前から当て、すでに採った一致と重なる一致は捨てる（QLoRA の中の LoRA は拾わない）。
    """
    text = _normalize(query)
    taken: list[tuple[int, int, str]] = []  # 採った一致（開始, 終了, term_id）
    for _, term_id, pattern in graph_data().patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if all(end <= other_start or other_end <= start for other_start, other_end, _ in taken):
                taken.append((start, end, term_id))
    return list(dict.fromkeys(term_id for _, _, term_id in sorted(taken)))


_LECTURE_PATTERN = re.compile(r"第([0-9]{1,3}|[一二三])回")
_KANJI_NUMBERS = {"一": 1, "二": 2, "三": 3}


def lecture_hints(query: str) -> list[int]:
    """質問に「第N回」と書かれた回（1〜3だけ）を、昇順・重複なしで返す。"""
    found = set()
    for match in _LECTURE_PATTERN.finditer(unicodedata.normalize("NFKC", query)):
        number = _KANJI_NUMBERS.get(match.group(1)) or int(match.group(1))
        if 1 <= number <= 3:
            found.add(number)
    return sorted(found)


def all_distances(query: str) -> dict[str, float]:
    """質問を1回だけ埋め込み、全ページ（214件）の距離を page_id → 距離の辞書で返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with retrieval.open_index(retrieval.RAG_DB) as db:
        vector = retrieval.embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute(
            "SELECT page_id, vec_distance_cosine(embedding, ?) AS distance FROM chunks",
            (sqlite_vec.serialize_float32(vector),),
        )
        return {row["page_id"]: row["distance"] for row in rows}


def select_pages(distances: dict[str, float], terms: list[str], lectures: list[int], data: GraphData,
                 top_k: int, cap: int, vector_pool: int, hub_terms: int) -> dict:
    """距離・拾った用語・回の指定から、返すページを選ぶ。API も DB も使わない。

    手順：用語（代わりの用語を含む）の言及先を、距離の上位と合わせて候補（pool）にする。
    用語ごとに、言及の多い回のページを1つ確保し、回が複数なら選ばれていない回のページも確保し、
    候補の残りを距離の順に、同じ章から cap 件までで埋める。
    それでも top_k に届かなければ、候補の外のページを同じ規則で補い（補充）、最後は章の上限を無視して補う（補充(上限なし)）。
    合計が top_k に達したら止め、最後に距離の昇順にする。結果は min(top_k, len(allowed)) 件になる。
    """
    lectures = sorted(set(lectures))
    # 回の指定が無ければ全ページ。対象外の資料のページ（用語グラフに無いもの）は、距離があっても除く
    allowed = {page_id for page_id in distances
               if page_id in data.lecture_of and (not lectures or data.lecture_of[page_id] in lectures)}

    def nearest(page_ids) -> list[str]:
        """距離が近い順（同じなら page_id の順）に並べる。"""
        return sorted(page_ids, key=lambda page_id: (distances[page_id], page_id))

    # 各用語の言及先。allowed に言及が無い用語は、関連用語で代わる（それも無ければ空）
    groups: dict[str, list[str]] = {}
    group_pages: dict[str, set[str]] = {}
    for term in terms:
        if data.mentions[term] & allowed:
            groups[term] = [term]
        else:
            groups[term] = sorted(other for other in data.related[term] if data.mentions[other] & allowed)
        group_pages[term] = {page_id for member in groups[term] for page_id in data.mentions[member] & allowed}

    # 候補：距離の上位と、用語の言及先。ただし用語の多いページ（目次）は、用語経由では入れない
    pool = set(nearest(allowed)[:max(0, vector_pool)])
    for term in groups:
        pool |= {page_id for page_id in group_pages[term] if len(data.page_terms[page_id]) < hub_terms}

    # 各用語の言及が最も多い回（同数なら全部）
    home: dict[str, list[int]] = {}
    for term in groups:
        counts = Counter(data.lecture_of[page_id] for page_id in group_pages[term])
        most = max(counts.values(), default=0)
        home[term] = sorted(lecture for lecture, count in counts.items() if count == most)

    selected: dict[str, str] = {}  # page_id → 選んだ理由（選んだ順）

    # 確保（用語）：言及ページが少ない用語から、home の回の言及ページが1つは入るようにする
    for term in sorted((term for term in groups if groups[term]), key=lambda term: len(group_pages[term])):
        if len(selected) >= top_k:
            break
        wanted = {page_id for page_id in group_pages[term] if data.lecture_of[page_id] in home[term]}
        if any(page_id in selected for page_id in wanted):
            continue
        candidates = nearest(wanted & pool)
        if candidates:
            selected[candidates[0]] = "確保:用語:" + data.terms[term]["title"]

    # 確保（回）：回の指定が複数なら、まだ1つも選ばれていない回の、最も近いページを入れる
    if len(lectures) >= 2:
        for lecture in lectures:
            if len(selected) >= top_k:
                break
            if any(data.lecture_of[page_id] == lecture for page_id in selected):
                continue
            candidates = nearest(page_id for page_id in allowed if data.lecture_of[page_id] == lecture)
            if candidates:
                selected[candidates[0]] = f"確保:第{lecture}回"

    per_section = Counter(data.section_of[page_id] for page_id in selected)
    skipped_by_cap = []

    def fill(page_ids, reason: str) -> None:
        """距離の順に、同じ章から cap 件までで選ぶ。章が cap 件に達していて外したページは、記録する。"""
        for page_id in nearest(page_ids):
            if len(selected) >= top_k:
                break
            section = data.section_of[page_id]
            if per_section[section] >= cap:
                skipped_by_cap.append({"page_id": page_id, "distance": distances[page_id], "section": section})
            else:
                selected[page_id] = reason
                per_section[section] += 1

    # 埋める：候補の残りを選ぶ
    fill([page_id for page_id in pool if page_id not in selected], "距離")
    # 補充：候補が尽きて top_k に届かなければ、候補の外のページを、同じ規則で選ぶ
    fill([page_id for page_id in allowed if page_id not in pool and page_id not in selected], "補充")
    # 補充（上限なし）：それでも届かなければ、章の上限を無視して、残りを距離の順に選ぶ
    for page_id in nearest(page_id for page_id in allowed if page_id not in selected):
        if len(selected) >= top_k:
            break
        selected[page_id] = "補充(上限なし)"

    return {
        "selected": [{"page_id": page_id, "distance": distances[page_id], "reason": selected[page_id],
                      "section": data.section_of[page_id]} for page_id in nearest(selected)],
        # 上限なしで選んだページは、結果に入ったので外した扱いにしない
        "skipped_by_cap": [item for item in skipped_by_cap if item["page_id"] not in selected],
        "groups": groups,
        "home": home,
        "pool_size": len(pool),
        "allowed_size": len(allowed),
    }


def search_term_graph(query: str, top_k: int | None = None, cap: int | None = None,
                      vector_pool: int | None = None, hub_terms: int | None = None) -> list[dict]:
    """質問の用語と「第N回」を手がかりに、近いページを選んで返す。内訳は TRACE に残す。

    None の引数は、呼ぶたびに TERM_GRAPH_* を読む。戻り値は retrieval.search() と同じキーで、距離の昇順。
    """
    top_k = TERM_GRAPH_TOP_K if top_k is None else top_k
    cap = TERM_GRAPH_SECTION_CAP if cap is None else cap
    vector_pool = TERM_GRAPH_VECTOR_POOL if vector_pool is None else vector_pool
    hub_terms = TERM_GRAPH_HUB_TERMS if hub_terms is None else hub_terms

    distances = all_distances(query)
    data = graph_data()
    terms = match_terms(query)
    lectures = lecture_hints(query)
    chosen = select_pages(distances, terms, lectures, data, top_k, cap, vector_pool, hub_terms)

    pages = {page["page_id"]: page for page in _pages()}
    results = [{**pages[item["page_id"]], "distance": item["distance"]} for item in chosen["selected"]]
    TRACE.append({
        "tool": "term_graph", "query": query,
        "terms": [{"id": term, "title": data.terms[term]["title"]} for term in terms],
        "lectures": lectures, "groups": chosen["groups"], "home": chosen["home"],
        "pool_size": chosen["pool_size"], "allowed_size": chosen["allowed_size"],
        "selected": chosen["selected"], "skipped_by_cap": chosen["skipped_by_cap"],
        "params": {"top_k": top_k, "cap": cap, "vector_pool": vector_pool, "hub_terms": hub_terms},
    })
    return results


# --- ツール用の関数（モデルに渡す） ---


def _tool_json(search, query: str) -> str:
    """検索結果を、search_materials と同じ JSON 文字列にする。例外の扱いも同じ。それ以外の例外は送出する。"""
    try:
        return json.dumps({"results": search(query)}, ensure_ascii=False)
    except (ValueError, FileNotFoundError, sqlite3.Error) as e:
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)
    except APIError as e:
        return json.dumps({"error": api_error_message(e.code), "results": []}, ensure_ascii=False)


# 名前と説明文を search_materials と同じにして登録する。モデルから見えるツールの形（名前・説明文・引数）は変えない。
def search_materials_with_neighbors(query: str) -> str:
    return _tool_json(search_with_neighbors, query)


def search_materials_term_graph(query: str) -> str:
    return _tool_json(search_term_graph, query)


search_materials_with_neighbors.__doc__ = tools.search_materials.__doc__
search_materials_term_graph.__doc__ = tools.search_materials.__doc__


# --- CLI ---


def _positive_int(text: str) -> int:
    """argparse 用。1以上の整数にする。"""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください。")
    return value


def _search_top_k(query: str, top_k: int | None) -> list[dict]:
    """通常の検索。top_k があれば、この呼び出しの間だけ RAG_TOP_K を変える。"""
    if top_k is None:
        return retrieval.search(query)
    original = retrieval.RAG_TOP_K
    retrieval.RAG_TOP_K = top_k
    try:
        return retrieval.search(query)
    finally:
        retrieval.RAG_TOP_K = original


def _print_terms(query: str) -> None:
    """拾った用語（title）と、回の指定を表示する。"""
    data = graph_data()
    print("拾った用語: " + ("、".join(data.terms[term]["title"] for term in match_terms(query)) or "なし"))
    print("回の指定: " + ("、".join(f"第{lecture}回" for lecture in lecture_hints(query)) or "なし"))


def _print_results(rows: list[dict], headings: list[str], labels: list[str], brief: bool) -> None:
    """検索結果を表示する。brief なら1ページ1行（順・講義・PDF p.・タイトル・distance・role か reason）。"""
    for i, (row, heading, label) in enumerate(zip(rows, headings, labels), 1):
        if brief:
            distance = "-" if row["distance"] is None else f"{row['distance']:.3f}"
            print(f"{i:>2}  第{row['lecture']}回  PDF p.{row['page']}  {row['title']}  d={distance}  {label}")
            continue
        print(f"\n{'=' * 60}\n検索結果 {i} / {heading}\n{'=' * 60}")
        print(f"講義: {row['document_title']}")
        print(f"章: {row['section']}")
        print(f"タイトル: {row['title']}")
        print(f"ページ: {row['page']}")
        print(f"\n{row['content']}")


def _print_term_graph_summary(trace: dict) -> None:
    """Graph 検索で拾った用語・回の指定・章の上限で外したページを表示する。"""
    titles = {page["page_id"]: page["title"] for page in _pages()}
    print(f"\n{'-' * 60}")
    print("拾った用語: " + ("、".join(term["title"] for term in trace["terms"]) or "なし"))
    print("回の指定: " + ("、".join(f"第{lecture}回" for lecture in trace["lectures"]) or "なし"))
    skipped = trace["skipped_by_cap"]
    print(f"章の上限（{trace['params']['cap']}件）で外したページ:" + ("" if skipped else " なし"))
    for item in skipped:
        print(f"  {item['page_id']} / d={item['distance']:.3f} / {titles[item['page_id']]}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する（第5回 宿題の取得処理）")
    parser.add_argument("query", help="調べたい内容")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--neighbors", action="store_true", help="検索結果に前後のページを足す（実装）")
    modes.add_argument("--top-k", type=_positive_int, metavar="N",
                       help="上位N件を返す（この実行の間だけ retrieval.RAG_TOP_K を変える）")
    modes.add_argument("--term-graph", action="store_true", help="用語から始める Graph 検索（発展）")
    parser.add_argument("--brief", action="store_true", help="1ページ1行で表示する")
    parser.add_argument("--terms", action="store_true",
                        help="API も DB も使わず、拾った用語と回の指定だけを表示して終わる")
    args = parser.parse_args(argv)

    if args.terms:
        _print_terms(args.query)
        return

    TRACE.clear()
    if args.term_graph:
        rows = search_term_graph(args.query)
        reasons = {item["page_id"]: item["reason"] for item in TRACE[-1]["selected"]}
        labels = [reasons[row["page_id"]] for row in rows]
        headings = [f"d={row['distance']:.3f} / {label}" for row, label in zip(rows, labels)]
    elif args.neighbors:
        rows = search_with_neighbors(args.query)
        ranks = {page_id: rank for rank, page_id in enumerate(TRACE[-1]["hits"], 1)}
        labels = [row["role"] for row in rows]
        headings = [f"hit {ranks[row['page_id']]} / d={row['distance']:.3f}" if row["role"] == "hit"
                    else f"{', '.join(row['neighbor_of'])} の隣" for row in rows]
    else:
        rows = _search_top_k(args.query, args.top_k)
        labels = ["hit"] * len(rows)
        headings = [f"hit {rank} / d={row['distance']:.3f}" for rank, row in enumerate(rows, 1)]
    _print_results(rows, headings, labels, args.brief)
    if args.term_graph:
        _print_term_graph_summary(TRACE[-1])


if __name__ == "__main__":
    main()
