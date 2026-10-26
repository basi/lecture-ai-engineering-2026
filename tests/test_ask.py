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
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                ),
            )

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    history = []
    clients = []

    def make_client(*, api_key):
        client = FakeClient(api_key=api_key)
        clients.append(client)
        return client

    with (
        patch.object(ask.genai, "Client", make_client),
        patch.object(ask, "get_api_key", return_value="test-key"),
    ):
        assert ask.ask("first question", history) == "answer 1"
        assert ask.ask("follow-up", history) == "answer 2"

    assert all(client.closed for client in clients)
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


def test_ask_prints_latency_and_token_counts(capsys):
    class FakeModels:
        def generate_content(self, *, model, contents):
            return SimpleNamespace(
                text="answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=7,
                ),
            )

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with (
        patch.object(ask.genai, "Client", FakeClient),
        patch.object(ask, "get_api_key", return_value="test-key"),
        patch.object(ask, "perf_counter", side_effect=[1.0, 1.125]),
    ):
        assert ask.ask("question") == "answer"

    assert capsys.readouterr().out == (
        "[metrics] latency=125 ms input_tokens=12 output_tokens=7 status=ok\n"
    )


def test_ask_prints_elapsed_time_when_api_request_fails(capsys):
    class FakeModels:
        def generate_content(self, *, model, contents):
            raise ask.errors.APIError(429, {"error": {"message": "rate limited"}})

    class FakeClient:
        models = FakeModels()

        def __init__(self, *, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    with (
        patch.object(ask.genai, "Client", FakeClient),
        patch.object(ask, "get_api_key", return_value="test-key"),
        patch.object(ask, "perf_counter", side_effect=[1.0, 1.25]),
    ):
        try:
            ask.ask("question")
        except ask.errors.APIError:
            pass
        else:
            raise AssertionError("APIError was not propagated")

    assert capsys.readouterr().out == (
        "[metrics] latency=250 ms "
        "input_tokens=n/a output_tokens=n/a status=error:429\n"
    )


def test_main_skips_blank_questions(monkeypatch):
    questions = iter(["", "follow-up", "exit"])
    asked = []
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    monkeypatch.setattr(
        ask,
        "ask",
        lambda question, history, client: asked.append(question) or "answer",
    )

    class FakeClient:
        def __init__(self, *, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    ask.main()

    assert asked == ["first question", "follow-up"]


def test_main_once_does_not_prompt(monkeypatch):
    asked = []
    monkeypatch.setattr("sys.argv", ["app.ask", "--once", "question"])
    monkeypatch.setattr(
        ask,
        "ask",
        lambda question, history, client: asked.append(question) or "answer",
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
    )

    class FakeClient:
        def __init__(self, *, api_key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    ask.main()

    assert asked == ["question"]
