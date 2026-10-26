from types import SimpleNamespace
from unittest.mock import patch

from app import ask


def test_ask_sends_and_updates_conversation_history():
    requests = []

    class FakeModels:
        def generate_content(self, *, model, contents):
            requests.append((model, contents.copy()))
            return SimpleNamespace(
                text=f"answer {len(requests)}",
                usage_metadata=None,
            )

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            assert api_key == "test-key"

    history = []
    with (
        patch.object(ask.genai, "Client", FakeClient),
        patch.object(ask, "get_api_key", return_value="test-key"),
    ):
        assert ask.ask("first question", history) == "answer 1"
        assert ask.ask("follow-up", history) == "answer 2"

    assert len(requests) == 2
    assert [part.text for part in requests[0][1][0].parts] == ["first question"]
    assert [message.role for message in requests[1][1]] == [
        "user",
        "model",
        "user",
    ]
    assert [part.text for part in requests[1][1][0].parts] == ["first question"]
    assert [part.text for part in requests[1][1][1].parts] == ["answer 1"]
    assert [part.text for part in requests[1][1][2].parts] == ["follow-up"]


def test_main_skips_blank_questions(monkeypatch):
    questions = iter(["", "follow-up", "exit"])
    asked = []
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    monkeypatch.setattr(
        ask,
        "ask",
        lambda question, history: asked.append(question) or "answer",
    )

    ask.main()

    assert asked == ["first question", "follow-up"]
