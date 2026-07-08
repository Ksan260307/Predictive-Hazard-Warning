# pytest がこのフォルダを見つけられるようにするためのファイル

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """テスト中はうっかり本物の通信をしないようにする安全弁。

    通信したいテストは、自分で roadinfo._fetch_json を差し替えれば
    こちらより後に適用されるので問題なく動く。
    """
    from app import roadinfo

    def _blocked(url, timeout):
        raise OSError("テスト中は通信しない")

    monkeypatch.setattr(roadinfo, "_fetch_json", _blocked)
