# detector.py (動いている物を見つける) のテスト
import math
import random

import numpy as np
import pytest

from app.detector import MovingThingFinder, shrink_frame


def make_frame(box_x=None, box_y=None, size=30, width=160, height=120):
    """テスト用の画像を作る。黒い背景に白い四角を置く。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if box_x is not None and box_y is not None:
        frame[box_y:box_y + size, box_x:box_x + size] = 255
    return frame


# ---------- 正常系 ----------

def test_動く物が見つかる():
    finder = MovingThingFinder()
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(30, 40))
    assert len(things) >= 1


def test_右に動く物の速さはプラスになる():
    finder = MovingThingFinder()
    for i in range(4):  # 10pxずつ右へ動かす
        things = finder.find(make_frame(20 + i * 10, 40))
    assert any(t["vx"] > 0.03 for t in things)


def test_下に動く物の速さは下向きになる():
    finder = MovingThingFinder()
    for i in range(4):
        things = finder.find(make_frame(60, 10 + i * 10))
    assert any(t["vy"] > 0.03 for t in things)


def test_止まっている物は見つからない():
    finder = MovingThingFinder()
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(20, 40))  # 同じ場所のまま
    assert things == []


def test_白黒の画像でも使える():
    finder = MovingThingFinder()
    gray1 = make_frame(20, 40)[:, :, 0]
    gray2 = make_frame(30, 40)[:, :, 0]
    finder.find(gray1)
    things = finder.find(gray2)
    assert len(things) >= 1


def test_見つけた物の値は全部0から1あたりに収まる():
    finder = MovingThingFinder()
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(30, 40))
    for t in things:
        for key in ("x", "y", "w", "h", "cx", "cy"):
            assert 0 <= t[key] <= 1


def test_リセットすると忘れる():
    finder = MovingThingFinder()
    finder.find(make_frame(20, 40))
    finder.reset()
    things = finder.find(make_frame(30, 40))  # リセット後の1枚目は比べられない
    assert things == []


# ---------- 異常系 ----------

def test_画像がNoneなら止まる():
    finder = MovingThingFinder()
    with pytest.raises(ValueError):
        finder.find(None)


def test_画像でないものを渡すと止まる():
    finder = MovingThingFinder()
    with pytest.raises(ValueError):
        finder.find("これは画像ではない")


def test_空の画像なら止まる():
    finder = MovingThingFinder()
    with pytest.raises(ValueError):
        finder.find(np.zeros((0, 0, 3), dtype=np.uint8))


def test_おかしな形の画像なら止まる():
    finder = MovingThingFinder()
    with pytest.raises(ValueError):
        finder.find(np.zeros((2, 3, 4, 5), dtype=np.uint8))


def test_画像の大きさが途中で変わっても壊れない():
    finder = MovingThingFinder()
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(20, 40, width=320, height=240))
    assert things == []  # 比べられないので空
    finder.find(make_frame(30, 40, width=320, height=240))  # その後は普通に動く


# ---------- 境界値 ----------

def test_感度は1と10が使える():
    MovingThingFinder(sensitivity=1)
    MovingThingFinder(sensitivity=10)


def test_感度が0や11なら止まる():
    with pytest.raises(ValueError):
        MovingThingFinder(sensitivity=0)
    with pytest.raises(ValueError):
        MovingThingFinder(sensitivity=11)


def test_感度に小数や文字を渡すと止まる():
    with pytest.raises(ValueError):
        MovingThingFinder(sensitivity=5.5)
    with pytest.raises(ValueError):
        MovingThingFinder(sensitivity="5")


def test_最小の大きさが0や1なら止まる():
    with pytest.raises(ValueError):
        MovingThingFinder(min_size=0)
    with pytest.raises(ValueError):
        MovingThingFinder(min_size=1)


def test_最小の大きさより小さい物は見つからない():
    # min_size=0.1 → 160*120*0.1=1920ピクセル必要。30x30の箱では足りない
    finder = MovingThingFinder(min_size=0.1)
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(30, 40))
    assert things == []


def test_最小の大きさが小さければ同じ物が見つかる():
    finder = MovingThingFinder(min_size=0.001)
    finder.find(make_frame(20, 40))
    things = finder.find(make_frame(30, 40))
    assert len(things) >= 1


# ---------- shrink_frame (分析用の縮小) ----------

def test_大きい画像は縮む():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    small = shrink_frame(frame, max_width=320)
    assert small.shape == (240, 320, 3)


def test_小さい画像はそのまま返る():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert shrink_frame(frame, max_width=320) is frame


def test_ちょうどの大きさもそのまま返る():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    assert shrink_frame(frame, max_width=320) is frame


def test_白黒画像も縮められる():
    frame = np.zeros((480, 640), dtype=np.uint8)
    assert shrink_frame(frame, max_width=320).shape == (240, 320)


def test_縮めた画像でも物は見つかる():
    finder = MovingThingFinder()
    for i in range(3):
        big = make_frame(80 + i * 40, 160, size=120, width=640, height=480)
        things = finder.find(shrink_frame(big, max_width=320))
    assert len(things) >= 1


def test_縮小のおかしな入力は止まる():
    with pytest.raises(ValueError):
        shrink_frame(None)
    with pytest.raises(ValueError):
        shrink_frame("画像ではない")
    with pytest.raises(ValueError):
        shrink_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        shrink_frame(frame, max_width=4)
    with pytest.raises(ValueError):
        shrink_frame(frame, max_width=100.5)


# ---------- 移動しながらの使用 (カメラの動きの打ち消し) ----------

def textured_background(width=160, height=120, seed=0):
    """目印の多い背景 (実際の風景に近い、模様のある画像)。"""
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return np.stack([bg, bg, bg], axis=2)


def test_自分が動くだけなら何も見つからない():
    # カメラ全体が横に流れる映像 (歩きながらの撮影) → 動く物なし
    bg = textured_background()
    finder = MovingThingFinder(stabilize=True)
    results = []
    for i in range(4):
        frame = np.roll(bg, 6 * i, axis=1)
        results.append(finder.find(frame))
    assert results[-1] == []


def test_打ち消しを切ると自分の動きを物と間違える():
    bg = textured_background()
    finder = MovingThingFinder(stabilize=False)
    for i in range(3):
        things = finder.find(np.roll(bg, 6 * i, axis=1))
    assert len(things) >= 1  # 背景全体が「動いた」と誤検出される


def test_自分が動いていても本当に動く物は見つかる():
    bg = textured_background()
    finder = MovingThingFinder(stabilize=True)
    found = False
    for i in range(5):
        frame = np.roll(bg, 6 * i, axis=1).copy()
        # 背景の動きとは別に、逆方向へ動く箱を置く
        x = 100 - i * 8
        frame[40:75, x:x + 35] = 255
        things = finder.find(frame)
        if things:
            found = True
    assert found


def test_打ち消しても止まっている箱は見つからない():
    # カメラも箱も止まっている → 何も見つからない
    bg = textured_background()
    frame = bg.copy()
    frame[40:70, 60:90] = 255
    finder = MovingThingFinder(stabilize=True)
    finder.find(frame)
    things = finder.find(frame.copy())
    assert things == []


# ---------- ランダムテスト ----------

def test_でたらめな画像を流し続けても壊れない():
    rng = np.random.default_rng(0)
    finder = MovingThingFinder()
    for _ in range(50):
        frame = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
        things = finder.find(frame)
        assert isinstance(things, list)
        for t in things:
            # 返ってきた値が全部まともな数であること
            for key in ("x", "y", "w", "h", "cx", "cy", "vx", "vy"):
                assert math.isfinite(t[key])


def test_でたらめな大きさの画像でも壊れない():
    rng = np.random.default_rng(1)
    py_rng = random.Random(1)
    finder = MovingThingFinder()
    for _ in range(30):
        h = py_rng.randint(10, 100)
        w = py_rng.randint(10, 100)
        frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        things = finder.find(frame)
        assert isinstance(things, list)
