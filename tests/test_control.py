# control.py (注意の強さを決める) のテスト
import math
import random

import pytest

from app import control


# ---------- sigmoid / softmax ----------

def test_sigmoidは0で半分になる():
    assert control.sigmoid(0) == pytest.approx(0.5)


def test_sigmoidは大きいほど1に近づく():
    assert control.sigmoid(10) > 0.99
    assert control.sigmoid(-10) < 0.01


def test_sigmoidはとても大きな数でも壊れない():
    assert control.sigmoid(1000) == pytest.approx(1.0)
    assert control.sigmoid(-1000) == pytest.approx(0.0)


def test_softmaxは足すと1になる():
    chances = control.softmax([1.0, 2.0, 3.0], tau=0.5)
    assert sum(chances) == pytest.approx(1.0)


def test_softmaxは温度が低いと一番高い点に集中する():
    chances = control.softmax([0.0, 1.0], tau=0.01)
    assert chances[1] > 0.99


def test_softmaxはとても大きな点数でも壊れない():
    chances = control.softmax([-1000.0, 0.0, 1000.0], tau=1.0)
    assert sum(chances) == pytest.approx(1.0)
    assert chances[2] == pytest.approx(1.0)


def test_softmaxの温度が0以下なら止まる():
    with pytest.raises(ValueError):
        control.softmax([1.0, 2.0], tau=0)
    with pytest.raises(ValueError):
        control.softmax([1.0, 2.0], tau=-1)


def test_softmaxに空のリストを渡すと止まる():
    with pytest.raises(ValueError):
        control.softmax([], tau=1.0)


# ---------- decide_weights ----------

def test_気持ちは足すと1になる():
    d, r = control.decide_weights(0.3, 0.2, 0.6)
    assert d + r == pytest.approx(1.0)
    assert 0 <= d <= 1
    assert 0 <= r <= 1


def test_危ないほど用心が強くなる():
    _, r_low = control.decide_weights(0.1, 0.0, 0.5)
    _, r_high = control.decide_weights(0.9, 0.0, 0.5)
    assert r_high > r_low


def test_危なさが増えると用心は減らない():
    previous = 0.0
    for i in range(11):
        _, r = control.decide_weights(i / 10, 0.2, 0.5)
        assert r >= previous - 1e-9
        previous = r


def test_安全で余裕があるとすすむ気持ちが勝つ():
    d, r = control.decide_weights(0.0, 0.0, 1.0)
    assert d > 0.9


def test_とても危ないと用心が勝つ():
    d, r = control.decide_weights(1.0, 0.0, 1.0)
    assert r > 0.9


def test_迷いが大きいと用心が増える():
    _, r_calm = control.decide_weights(0.3, 0.0, 0.5)
    _, r_lost = control.decide_weights(0.3, 1.0, 0.5)
    assert r_lost > r_calm


def test_入力が範囲の外なら止まる():
    with pytest.raises(ValueError):
        control.decide_weights(-0.1, 0.5, 0.5)
    with pytest.raises(ValueError):
        control.decide_weights(1.1, 0.5, 0.5)
    with pytest.raises(ValueError):
        control.decide_weights(0.5, -1, 0.5)
    with pytest.raises(ValueError):
        control.decide_weights(0.5, 0.5, 2)


def test_入力に文字やTrueを渡すと止まる():
    with pytest.raises(ValueError):
        control.decide_weights("0.5", 0.5, 0.5)
    with pytest.raises(ValueError):
        control.decide_weights(0.5, True, 0.5)


def test_境界の0と1はそのまま使える():
    control.decide_weights(0, 0, 0)
    control.decide_weights(1, 1, 1)


# ---------- phase / temperature / mode ----------

def test_位相は0から90度の間になる():
    for risk in (0, 0.5, 1):
        for d in (0, 0.5, 1):
            phase = control.phase_of(risk, d, 1 - d)
            assert 0 <= phase <= math.pi / 2 + 1e-9


def test_温度はいつもプラスになる():
    for phase in (0, math.pi / 4, math.pi / 2):
        assert control.temperature_of(phase) > 0


def test_おかしな温度の元では止まる():
    with pytest.raises(ValueError):
        control.temperature_of(0.5, tau0=0)
    with pytest.raises(ValueError):
        control.temperature_of(0.5, tau1=-1)


def test_危ないときは安定モード():
    assert control.mode_of(0.9, 0.1, 0.9) == "安定"


def test_安全ではっきりしていると探索モード():
    assert control.mode_of(0.0, 0.95, 0.05) == "探索"


def test_中間は適応モード():
    assert control.mode_of(0.3, 0.55, 0.45) == "適応"


# ---------- pick_level ----------

def test_用心が弱いとあんしん():
    assert control.pick_level(0.0, phase=0.0) == 0


def test_用心が中くらいだとちゅうい():
    assert control.pick_level(0.5, phase=0.5) == 1


def test_用心が強いときけん():
    assert control.pick_level(1.0, phase=1.0) == 2


def test_用心がとても強いと必ず知らせる():
    # 最小行動量の保証: どんな選ばれ方でも最低ラインがある
    assert control.pick_level(0.9, phase=0.0) == 2
    assert control.pick_level(0.7, phase=0.0) >= 1


def test_レベルは一気に下がらない():
    # きけん(2) の次は、用心が0でも ちゅうい(1) までしか下がらない
    assert control.pick_level(0.0, phase=0.0, prev_level=2) == 1
    assert control.pick_level(0.0, phase=0.0, prev_level=1) == 0


def test_くじ引きでも必ず0か1か2になる():
    rng = random.Random(3)
    for _ in range(200):
        level = control.pick_level(rng.random(), phase=rng.uniform(0, math.pi / 2),
                                   rng=rng)
        assert level in (0, 1, 2)


def test_おかしな入力では止まる():
    with pytest.raises(ValueError):
        control.pick_level(-0.1, phase=0.0)
    with pytest.raises(ValueError):
        control.pick_level(1.5, phase=0.0)
    with pytest.raises(ValueError):
        control.pick_level(0.5, phase=0.0, prev_level=5)


# ---------- ランダムテスト ----------

def test_でたらめな入力でも答えはいつもまともになる():
    rng = random.Random(11)
    prev_level = 0
    for _ in range(300):
        risk = rng.random()
        uncertainty = rng.random()
        optionality = rng.random()
        d, r = control.decide_weights(risk, uncertainty, optionality,
                                      risk_line=rng.uniform(0.05, 0.95))
        assert d + r == pytest.approx(1.0)
        assert 0 <= d <= 1 and 0 <= r <= 1
        phase = control.phase_of(risk, d, r)
        assert 0 <= phase <= math.pi / 2 + 1e-9
        level = control.pick_level(r, phase, prev_level=prev_level)
        assert level in (0, 1, 2)
        assert level >= prev_level - 1  # 一気に下がっていない
        prev_level = level
        assert control.mode_of(risk, d, r) in ("探索", "適応", "安定")
