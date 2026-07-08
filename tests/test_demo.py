# demo.py (デモ映像) のテスト
import numpy as np
import pytest

from app.demo import CROSS_END, CROSS_PERIOD, CROSS_START, DemoCamera
from app.watcher import DangerWatcher


# ---------- 正常系 ----------

def test_カメラと同じ形で読める():
    camera = DemoCamera()
    ok, frame = camera.read()
    assert ok is True
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8


def test_isOpenedはいつもTrue():
    camera = DemoCamera()
    assert camera.isOpened() is True
    camera.release()  # エラーにならない
    assert camera.isOpened() is True


def test_映像は時間で変化する():
    camera = DemoCamera()
    frames = [camera.read()[1] for _ in range(CROSS_PERIOD)]
    # 歩行者が出ている間は背景と違う映像になる
    changed = sum(1 for f in frames[1:] if not np.array_equal(f, frames[0]))
    assert changed > 0


def test_同じ種なら同じ映像になる():
    a = DemoCamera(seed=5)
    b = DemoCamera(seed=5)
    for _ in range(30):
        fa = a.read()[1]
        fb = b.read()[1]
        assert np.array_equal(fa, fb)


def test_歩行者は周期的に現れる():
    camera = DemoCamera()
    base = None
    appearances = 0
    for tick in range(1, CROSS_PERIOD * 2 + 1):
        frame = camera.read()[1]
        phase = tick % CROSS_PERIOD
        if phase == 0 or not (CROSS_START <= phase < CROSS_END):
            base = frame  # 歩行者がいない時の映像
        elif base is not None and not np.array_equal(frame, base):
            appearances += 1
    assert appearances > 0


def test_好きな大きさで作れる():
    camera = DemoCamera(width=64, height=48)
    ok, frame = camera.read()
    assert frame.shape == (48, 64, 3)


# ---------- 異常系・境界値 ----------

def test_小さすぎる大きさでは作れない():
    with pytest.raises(ValueError):
        DemoCamera(width=8, height=180)
    with pytest.raises(ValueError):
        DemoCamera(width=320, height=8)


def test_おかしな大きさでは作れない():
    with pytest.raises(ValueError):
        DemoCamera(width="320", height=180)
    with pytest.raises(ValueError):
        DemoCamera(width=320.5, height=180)
    with pytest.raises(ValueError):
        DemoCamera(width=True, height=180)


def test_ぎりぎりの大きさは作れる():
    camera = DemoCamera(width=32, height=32)
    ok, frame = camera.read()
    assert frame.shape == (32, 32, 3)


# ---------- 危険予測との組み合わせ ----------

def test_デモ映像を流すと危険予測が反応する():
    camera = DemoCamera()
    watcher = DangerWatcher()
    max_risk = 0.0
    for _ in range(CROSS_PERIOD + 30):
        ok, frame = camera.read()
        result = watcher.watch(frame)
        max_risk = max(max_risk, result["risk"])
        assert result["level"] in (0, 1, 2)
    # 壁の死角と横断してくる歩行者があるので、危険度は必ずどこかで上がる
    assert max_risk > 0.05
