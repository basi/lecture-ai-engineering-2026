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
    response = client.models.generate_content(model=MODEL, contents=conversation)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] input={usage.prompt_token_count} "
            f"output={usage.candidates_token_count}"
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
