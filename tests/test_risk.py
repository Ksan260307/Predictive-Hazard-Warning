# risk.py (危なさ・迷い・余裕の計算) のテスト
import random

import pytest

from app.future import make_futures
from app.risk import optionality_score, risk_score, uncertainty_score


def future_of(points, weight=1.0):
    """テスト用の未来を1つ作る。points は [(cx, cy), ...]"""
    return {"paths": [list(points)], "weight": weight}


# ---------- risk_score 正常系 ----------

def test_ぶつかる場所に入る未来があると危なさが上がる():
    futures = [future_of([(0.5, 0.8)], weight=1.0)]  # 下の真ん中に入る
    assert risk_score(futures, danger_y=0.7) == pytest.approx(1.0)


def test_ぶつかる場所に入らなければ危なさ0():
    futures = [future_of([(0.5, 0.2)], weight=1.0)]  # 上の方にいる
    assert risk_score(futures, danger_y=0.7) == 0.0


def test_横にそれた物は危なくない():
    futures = [future_of([(0.05, 0.9)], weight=1.0)]  # 下だが左のすみ
    assert risk_score(futures, danger_y=0.7) == 0.0


def test_危なさはぶつかる未来の重みの合計になる():
    futures = [
        future_of([(0.5, 0.9)], weight=0.3),  # ぶつかる
        future_of([(0.5, 0.1)], weight=0.7),  # ぶつからない
    ]
    assert risk_score(futures, danger_y=0.7) == pytest.approx(0.3)


def test_未来がなければ危なさ0():
    assert risk_score([], danger_y=0.7) == 0.0


def test_下に向かう物は本当に危ないと出る():
    # 実際の未来生成と組み合わせた確認
    thing = {"cx": 0.5, "cy": 0.5, "vx": 0.0, "vy": 0.05}
    futures = make_futures([thing], steps=15, samples=30, seed=1)
    assert risk_score(futures, danger_y=0.7) > 0.8


# ---------- risk_score 異常系・境界値 ----------

def test_危険ラインが範囲の外なら止まる():
    with pytest.raises(ValueError):
        risk_score([], danger_y=-0.1)
    with pytest.raises(ValueError):
        risk_score([], danger_y=1.1)


def test_ちょうど危険ラインの上なら危ない():
    futures = [future_of([(0.5, 0.7)])]
    assert risk_score(futures, danger_y=0.7) == pytest.approx(1.0)


def test_ちょうど横のはしっこも中に入る():
    assert risk_score([future_of([(0.25, 0.9)])], danger_y=0.7) == pytest.approx(1.0)
    assert risk_score([future_of([(0.75, 0.9)])], danger_y=0.7) == pytest.approx(1.0)
    assert risk_score([future_of([(0.24, 0.9)])], danger_y=0.7) == 0.0
    assert risk_score([future_of([(0.76, 0.9)])], danger_y=0.7) == 0.0


def test_危険ライン0なら真ん中にいるだけで危ない():
    assert risk_score([future_of([(0.5, 0.0)])], danger_y=0) == pytest.approx(1.0)


def test_重みが大きすぎても1でおさえる():
    futures = [future_of([(0.5, 0.9)], weight=5.0)]
    assert risk_score(futures, danger_y=0.7) == 1.0


# ---------- uncertainty_score ----------

def test_迷いは半々のとき最大になる():
    assert uncertainty_score(0.5) == pytest.approx(1.0)


def test_迷いははっきりしていると0になる():
    assert uncertainty_score(0.0) == 0.0
    assert uncertainty_score(1.0) == 0.0


def test_迷いは0から半々に向かって増えていく():
    assert uncertainty_score(0.1) < uncertainty_score(0.3) < uncertainty_score(0.5)


def test_迷いの入力が範囲の外なら止まる():
    with pytest.raises(ValueError):
        uncertainty_score(-0.01)
    with pytest.raises(ValueError):
        uncertainty_score(1.01)


def test_迷いに文字やTrueを渡すと止まる():
    with pytest.raises(ValueError):
        uncertainty_score("0.5")
    with pytest.raises(ValueError):
        uncertainty_score(True)


# ---------- optionality_score ----------

def test_未来がなければ余裕は最大():
    assert optionality_score([]) == 1.0


def test_未来が1つだけなら余裕は0():
    assert optionality_score([future_of([(0.5, 0.5)])]) == 0.0


def test_ばらけた未来ほど余裕が大きい():
    # 同じ場所に集まった未来
    narrow = [future_of([(0.5, 0.5)], weight=0.5) for _ in range(2)]
    # ばらばらの場所に行く未来
    wide = [
        future_of([(0.1, 0.1)], weight=0.25),
        future_of([(0.9, 0.1)], weight=0.25),
        future_of([(0.1, 0.9)], weight=0.25),
        future_of([(0.9, 0.9)], weight=0.25),
    ]
    assert optionality_score(wide) > optionality_score(narrow)


def test_重みがかたよると余裕が減る():
    even = [future_of([(0.3, 0.3)], 0.5), future_of([(0.7, 0.7)], 0.5)]
    skewed = [future_of([(0.3, 0.3)], 0.99), future_of([(0.7, 0.7)], 0.01)]
    assert optionality_score(even) > optionality_score(skewed)


def test_重みが全部0でも壊れない():
    futures = [future_of([(0.5, 0.5)], 0.0), future_of([(0.3, 0.3)], 0.0)]
    assert optionality_score(futures) == 0.0


# ---------- ランダムテスト ----------

def test_でたらめな未来でも3つの数は必ず0から1になる():
    rng = random.Random(9)
    for i in range(150):
        futures = []
        for _ in range(rng.randint(0, 20)):
            points = [(rng.uniform(-0.5, 1.5), rng.uniform(-0.5, 1.5))
                      for _ in range(rng.randint(1, 10))]
            futures.append(future_of(points, weight=rng.uniform(0, 0.2)))
        r = risk_score(futures, danger_y=rng.uniform(0.3, 0.95))
        assert 0 <= r <= 1
        assert 0 <= uncertainty_score(r) <= 1
        assert 0 <= optionality_score(futures) <= 1
