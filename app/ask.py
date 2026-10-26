"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 既定の質問のあと、続けて質問できる

ツールは使わず、同じ会話の履歴を保ちながら質問を続ける。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, history: list[types.Content] | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    conversation = history if history is not None else []
    conversation.append(
        types.Content(role="user", parts=[types.Part.from_text(text=question)])
    )
    started_at = perf_counter()
    response = client.models.generate_content(model=MODEL, contents=conversation)
    elapsed_ms = (perf_counter() - started_at) * 1000

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else "n/a"
    output_tokens = usage.candidates_token_count if usage is not None else "n/a"
    print(
        f"[metrics] latency={elapsed_ms:.0f} ms "
        f"input_tokens={input_tokens} output_tokens={output_tokens}"
    )

    if response.text is None:
        raise RuntimeError("モデルからテキスト回答がありません")

    conversation.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=response.text)],
        )
    )
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--once",
        action="store_true",
        help="質問に1回だけ回答して終了する（モデルや質問の比較向け）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    history: list[types.Content] = []
    question = args.question
    while True:
        print(f"[質問] {question}\n")
        try:
            print(ask(question, history))
        except errors.APIError as e:
            print(api_error_message(e.code, e.message))
            raise SystemExit(1) from e

        if args.once:
            return

        while True:
            try:
                question = input(
                    "\n続けて質問できます（終了するには「終了」または exit）: "
                ).strip()
            except EOFError:
                return
            if question.lower() in {"終了", "exit", "quit"}:
                return
            if question:
                break


if __name__ == "__main__":
    main()
