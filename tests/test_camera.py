# camera.py (カメラの抽象化) のテスト
#
# Kivy のカメラ実体は無い環境でもテストできるよう、
# KivyCameraCapture は __init__ を通さずに組み立てて、
# 「映像が来ない時に諦める」流れだけを確かめる。

import numpy as np

from app import camera as camera_mod
from app.camera import KivyCameraCapture


class FakeCoreCamera:
    """テスト用のKivyカメラ。テクスチャは永遠に来ない。"""

    texture = None

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def make_capture():
    """__init__ (kivy必須) を通さずに包みを組み立てる。"""
    cap = object.__new__(KivyCameraCapture)
    cap._camera = FakeCoreCamera()
    cap._opened = True
    cap._blank_count = 0
    cap._blank = np.zeros((4, 4, 3), dtype=np.uint8)
    return cap


def test_映像が来るまでは黒い画面でつなぐ(monkeypatch):
    monkeypatch.setattr(camera_mod, "_gave_up_at", 0.0)
    cap = make_capture()
    ok, frame = cap.read()
    assert ok is True
    assert frame.shape == (4, 4, 3)
    assert int(frame.sum()) == 0


def test_映像が来ないまま待ちすぎたら諦めて閉じる(monkeypatch):
    monkeypatch.setattr(camera_mod, "_gave_up_at", 0.0)
    cap = make_capture()
    for _ in range(camera_mod.GIVEUP_BLANK_FRAMES):
        ok, _ = cap.read()
        assert ok is True  # 限界までは黒画面でつなぐ
    ok, frame = cap.read()  # 限界を超えた
    assert ok is False
    assert frame is None
    assert cap.isOpened() is False
    assert cap._camera.stopped is True
    # 諦めた直後は開き直さない (デモ映像でつなぐ)
    assert camera_mod._recently_gave_up() is True


def test_諦めてから時間が経てばまた開き直せる(monkeypatch):
    monkeypatch.setattr(camera_mod, "_gave_up_at", 0.0)
    camera_mod._note_giveup(now=1000.0)
    assert camera_mod._recently_gave_up(now=1000.0 + 1) is True
    assert camera_mod._recently_gave_up(
        now=1000.0 + camera_mod.REOPEN_COOLDOWN + 1) is False


def test_映像が来たら数え直す(monkeypatch):
    monkeypatch.setattr(camera_mod, "_gave_up_at", 0.0)
    cap = make_capture()
    for _ in range(camera_mod.GIVEUP_BLANK_FRAMES - 1):
        cap.read()
    cap._blank_count = 0  # 映像が届いた時と同じ状態
    for _ in range(camera_mod.GIVEUP_BLANK_FRAMES):
        ok, _ = cap.read()
    assert ok is True  # 数え直したのでまだ諦めない
