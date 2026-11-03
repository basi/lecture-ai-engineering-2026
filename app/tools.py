"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
import random
import re
from datetime import datetime
from functools import cache
from itertools import count
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR = DATA_DIR.parent / "learning_log"  # save_learning_log の保存先（.gitignore で共有しない）
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """指定した回（1始まりの整数）の用語を用語集から1つ選び、確認問題と答えを返す。

    利用者に「第2回の確認問題を出して」のように、ある回の復習の問題を求められたときに使う。
    """
    glossary = load_glossary()
    terms = [e for e in glossary if e["lecture"] == lecture]
    if not terms:
        available = sorted({e["lecture"] for e in glossary})
        return f"第{lecture}回の用語は用語集にありません。出題できる回: {', '.join(map(str, available))}"
    entry = random.choice(terms)
    return json.dumps(
        {"question": f"「{entry['term']}」の意味は？", "answer": entry["definition"]},
        ensure_ascii=False,
    )


def _safe_filename(title: str) -> str:
    """題名を、ファイル名に使える文字（日本語を含む文字・数字・-・_）だけにする。

    / や \\ や .. や制御文字は、"-" に置き換わるので残らない。空になったら "log"、長すぎれば切る。
    """
    name = re.sub(r"[^\w-]+", "-", title)[:50].strip("-_")
    return name or "log"


def save_learning_log(title: str, content: str) -> str:
    """学習の記録を learning_log/ フォルダに Markdown ファイルとして保存する。

    利用者に学習の記録や保存を頼まれたときだけ使うこと。title は記録の題名、content は本文（Markdown）。
    保存は取り消しにくいので、このツールを呼ぶと、保存の前にプログラムが利用者に承認を求める
    （会話の中で先に確認を取らなくてよい）。承認されなかったときは保存しない。
    """
    log_dir = LOG_DIR.resolve()  # テストや比較用スクリプトが LOG_DIR を差し替えるので、呼ぶたびに参照する
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{_safe_filename(title)}"
    for n in count(1):
        name = f"{stem}.md" if n == 1 else f"{stem}-{n}.md"
        path = (log_dir / name).resolve()
        if path.parent != log_dir:  # シンボリックリンクなどで LOG_DIR の外を指すときは書かない
            return "保存先が learning_log/ の外になるため、保存しませんでした。"
        try:
            with open(path, "x", encoding="utf-8") as f:  # "x": 同じ名前があれば、上書きせずに失敗する
                f.write(f"# {title}\n\n{content}\n")
        except FileExistsError:
            continue  # 同じ名前があるときは、-2, -3 … を付けて試す
        return f"学習ログを保存しました: {name}"
