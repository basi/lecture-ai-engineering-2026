"""lectures/06 のテスト全体に効く、安全のための fixture（どのテストも、自動で使う）。

- no_network: ネットワークに出ない。外に出ようとしたら、待たずにすぐ例外にする。
  対象はテストのプロセスだけで、stdio で起動するサーバー（別のプロセス）は対象外。
- fake_api_key: GEMINI_API_KEY を偽物にして、GOOGLE_API_KEY は無しにする（本物のキーを使わない・書き出さない）。

「import quiz_server」「import env_probe」は、lectures/06 を sys.path に入れて行う（lectures/03・05 と同じ方式。
__init__.py は作らない）。別の場所の同じ名前のファイルと同じ pytest で動かすと、名前が衝突するので、
lectures/06 は単独で実行する（uv run pytest -q lectures/06）。
"""

import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import env_probe  # noqa: E402
import quiz_server  # noqa: E402

for _module in (quiz_server, env_probe):
    if Path(_module.__file__).resolve().parent != HERE:
        raise RuntimeError(
            f"別の場所の {_module.__name__}.py（{_module.__file__}）が先に読み込まれています。"
            "lectures/06 は単独で実行してください: uv run pytest -q lectures/06"
        )

FAKE_KEY = "test-key-not-a-real-key"


class NetworkBlocked(RuntimeError):
    """テストが、ネットワークに出ようとした。モデルは FunctionModel で、MCP サーバーは in-memory か手元のサブプロセスで動かす。"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkBlocked("テストはネットワークに出ません（socket の接続・名前解決を止めています）。FunctionModel や in-memory の接続を使ってください")

    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):  # AF_UNIX（ローカルの通信）は、そのまま通す
            blocked()
        return real_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            blocked()
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)  # 名前解決も止める（ネットワークが無いと、解決の timeout を待つため）


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return FAKE_KEY
