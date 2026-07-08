# hud.py (映像への描き込み) のテスト
import random

import numpy as np
import pytest

from app.hud import draw_hud


def base_frame(width=160, height=120):
    return np.full((height, width, 3), 128, dtype=np.uint8)


def base_result(**overrides):
    """watch() の返り値のうち HUD が使う部分。"""
    result = {
        "level": 0,
        "risk": 0.0,
        "things": [],
        "blind_spots": [],
        "danger_line": 0.7,
    }
    result.update(overrides)
    return result


# ---------- 正常系 ----------

def test_描き込んでも画像の形は変わらない():
    frame = base_frame()
    out = draw_hud(frame, base_result())
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_元の画像は書きかえられない():
    frame = base_frame()
    original = frame.copy()
    draw_hud(frame, base_result(level=2, risk=1.0))
    assert np.array_equal(frame, original)


def test_何か描き込まれている():
    frame = base_frame()
    out = draw_hud(frame, base_result())
    assert not np.array_equal(out, frame)  # 危険域とバーが描かれる


def test_物体があると枠が描かれる():
    thing = {"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2,
             "cx": 0.4, "cy": 0.4, "vx": 0.05, "vy": 0.0}
    plain = draw_hud(base_frame(), base_result())
    with_thing = draw_hud(base_frame(), base_result(things=[thing]))
    assert not np.array_equal(plain, with_thing)


def test_死角があるとマーカーが描かれる():
    spot = {"x": 0.3, "y": 0.6, "strength": 0.8}
    plain = draw_hud(base_frame(), base_result())
    with_spot = draw_hud(base_frame(), base_result(blind_spots=[spot]))
    assert not np.array_equal(plain, with_spot)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_どのレベルでも描ける(level):
    out = draw_hud(base_frame(), base_result(level=level, risk=level / 2))
    assert out.shape == base_frame().shape


# ---------- 異常系 ----------

def test_画像がNoneなら止まる():
    with pytest.raises(ValueError):
        draw_hud(None, base_result())


def test_白黒画像には描けない():
    with pytest.raises(ValueError):
        draw_hud(np.zeros((100, 100), dtype=np.uint8), base_result())


def test_空の画像なら止まる():
    with pytest.raises(ValueError):
        draw_hud(np.zeros((0, 0, 3), dtype=np.uint8), base_result())


def test_結果が辞書でなければ止まる():
    with pytest.raises(ValueError):
        draw_hud(base_frame(), "結果ではない")


def test_結果に必要な項目が欠けていると止まる():
    result = base_result()
    del result["risk"]
    with pytest.raises(ValueError):
        draw_hud(base_frame(), result)


def test_おかしな濃さでは止まる():
    with pytest.raises(ValueError):
        draw_hud(base_frame(), base_result(), alpha=-0.1)
    with pytest.raises(ValueError):
        draw_hud(base_frame(), base_result(), alpha=1.1)


# ---------- 境界値 ----------

def test_危険度0と1のどちらでも描ける():
    draw_hud(base_frame(), base_result(risk=0.0))
    draw_hud(base_frame(), base_result(risk=1.0))


def test_危険度が範囲を少し超えていても壊れない():
    # watch() の計算誤差を想定し、はみ出した値は内側に丸めて描く
    draw_hud(base_frame(), base_result(risk=1.0000001))


def test_とても小さな画像にも描ける():
    out = draw_hud(np.full((16, 16, 3), 100, dtype=np.uint8), base_result(level=2))
    assert out.shape == (16, 16, 3)


def test_画面のはしにいる物体でも壊れない():
    thing = {"x": 0.95, "y": 0.95, "w": 0.2, "h": 0.2,
             "cx": 1.0, "cy": 1.0, "vx": 0.5, "vy": 0.5}
    draw_hud(base_frame(), base_result(things=[thing]))


# ---------- ランダムテスト ----------

def test_でたらめな結果でも描き込みは壊れない():
    rng = random.Random(17)
    for _ in range(100):
        things = [{
            "x": rng.uniform(-0.5, 1.5), "y": rng.uniform(-0.5, 1.5),
            "w": rng.uniform(0, 1), "h": rng.uniform(0, 1),
            "cx": rng.uniform(-0.5, 1.5), "cy": rng.uniform(-0.5, 1.5),
            "vx": rng.uniform(-0.3, 0.3), "vy": rng.uniform(-0.3, 0.3),
        } for _ in range(rng.randint(0, 5))]
        spots = [{"x": rng.uniform(0, 1), "y": rng.uniform(0, 1),
                  "strength": rng.uniform(0, 1)} for _ in range(rng.randint(0, 3))]
        result = base_result(
            level=rng.choice([0, 1, 2]),
            risk=rng.uniform(0, 1),
            things=things,
            blind_spots=spots,
            danger_line=rng.uniform(0.3, 0.95),
        )
        out = draw_hud(base_frame(), result)
        assert out.shape == (120, 160, 3)
