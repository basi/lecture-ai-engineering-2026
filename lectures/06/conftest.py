"""lectures/06 のテスト全体に効く、安全のための fixture（どのテストも、自動で使う）。

- no_network: ネットワークに出ない。外に出ようとしたら、待たずにすぐ例外にする。
  対象はテストのプロセスだけで、stdio で起動するサーバー（別のプロセス）は対象外。
- fake_api_key: GEMINI_API_KEY を偽物にして、GOOGLE_API_KEY は無しにする（本物のキーを使わない・書き出さない）。
- isolated_results: run.py の結果の置き場所（lectures/06/results/）を、テストごとの一時ディレクトリに差し替える。
- waits: run.py の sleep を差し替えて、待たない（待とうとした秒数の一覧を、テストに渡す）。
- repo_files_are_untouched: テスト全体の前後で、app/mcp_server.py と app/mcp_servers.json が変わっていないことを確かめる。
  run.py は、実行中にこの 2 つを書き換えながら pytest を流す。そのときにこれらのテストも走るので、
  本物のファイルを書き換えるテストがあれば、その場で気づけるようにする。

「import quiz_server」「import env_probe」「import run」は、lectures/06 を sys.path に入れて行う（lectures/03・05 と同じ方式。
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
import run  # noqa: E402

for _module in (quiz_server, env_probe, run):
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


@pytest.fixture(autouse=True)
def isolated_results(monkeypatch, tmp_path):
    """run.py の結果は、一時ディレクトリに書く（本物の lectures/06/results/ は書き換えない）。"""
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")


@pytest.fixture(autouse=True)
def waits(monkeypatch):
    """run.py の sleep を差し替えて、待たない。待とうとした秒数の一覧を返す。"""
    waited = []
    monkeypatch.setattr(run, "sleep", waited.append)
    return waited


@pytest.fixture(scope="session", autouse=True)
def repo_files_are_untouched():
    """テスト全体の前後で、状態を切り替える 2 つのファイルが変わっていないことを確かめる。"""
    paths = [run.REPO / run.SERVER, run.REPO / run.CONFIG]
    before = [path.read_bytes() for path in paths]
    yield
    changed = [str(path) for path, data in zip(paths, before) if path.read_bytes() != data]
    assert not changed, f"テストが、本物のファイルを書き換えました: {changed}。状態の切り替えは、一時ディレクトリのコピーで確かめてください"
