"""lectures/05 のテスト全体に効く、安全のための fixture（どのテストも、自動で使う）。

- no_network: ネットワークに出ない。外に出ようとしたら、待たずにすぐ例外にする。
  本物の GoogleProvider や genai.Client が、既定の timeout を待ち続けて、テストが止まったように見えるのを防ぐ。
- hermetic_git: 一時リポジトリの git が、この作業ツリーの git や、利用者の git の設定（hook など）に触れない。
- sleeps: compare.time.sleep を待たずに、待つ秒数だけを記録する。待つと、同じ秒数だけ偽の時計（time.monotonic）が進む。
  TPM ガード（直近 60 秒の入力トークン）の確認で、時計を進めるのに使う。
- fake_api_key: GEMINI_API_KEY を偽物にして、GOOGLE_API_KEY は無しにする（本物のキーを使わない・書き出さない）。

「import compare」は、lectures/05 を sys.path に入れて行う（lectures/03 と同じ方式。__init__.py は作らない）。
別の回の compare.py と同じ pytest で動かすと、名前が衝突するので、lectures/05 は単独で実行する（uv run pytest -q lectures/05）。
"""

import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compare  # noqa: E402

if Path(compare.__file__).resolve().parent != HERE:
    raise RuntimeError(
        f"別の回の compare.py（{compare.__file__}）が先に読み込まれています。"
        "lectures/05 のテストは、単独で実行してください: uv run pytest -q lectures/05"
    )

FAKE_KEY = "test-key-not-a-real-key"


class NetworkBlocked(RuntimeError):
    """テストが、ネットワークに出ようとした。モデルは FunctionModel・TestModel、埋め込みは monkeypatch で偽物にする。"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkBlocked("テストはネットワークに出ません（socket の接続・名前解決を止めています）。FunctionModel や monkeypatch で偽物にしてください")

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
def hermetic_git(monkeypatch, tmp_path_factory):
    # tmp_path の中では、上のフォルダ（この作業ツリーなど）の .git を探さない。git init した一時リポジトリだけが、git として見える
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp().resolve()))
    # 利用者の git の設定（hook・署名・テンプレート）を読まない。入力待ちで止まらない
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


class Sleeps(list):
    """compare.time.sleep が待つはずだった秒数の記録。now は偽の時計（time.monotonic の代わり）で、待った分だけ進む。"""

    now = 1000.0


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    waited = Sleeps()

    def sleep(seconds):
        waited.append(seconds)
        waited.now += seconds

    # compare.py の time だけを差し替える（time モジュール全体の sleep を差し替えると、subprocess の待ちなどにも効いて、記録が混ざる）
    monkeypatch.setattr(compare, "time", SimpleNamespace(sleep=sleep, perf_counter=time.perf_counter, monotonic=lambda: waited.now))
    return waited


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return FAKE_KEY
