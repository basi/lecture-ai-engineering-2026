"""lectures/04 のテスト全体に効く、安全のための fixture（どのテストも、自動で使う）。

- no_network: ネットワークに出ない。外に出ようとしたら、待たずにすぐ例外にする。
  本物の GoogleProvider が、既定の timeout（読み取り 20 分）を待ち続けて、テストが止まったように見えるのを防ぐ。
- hermetic_git: 一時リポジトリの git が、この作業ツリーの git や、利用者の git の設定（hook など）に触れない。
"""

import socket

import pytest


class NetworkBlocked(RuntimeError):
    """テストが、ネットワークに出ようとした。HTTP は httpx2.MockTransport で、モデルは FunctionModel・TestModel で偽物にする。"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise NetworkBlocked("テストはネットワークに出ません（socket の接続・名前解決を止めています）。MockTransport か FunctionModel を使ってください")

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
