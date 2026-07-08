# modes.py (移動モード) のテスト
import random

import pytest

from app import modes


REQUIRED_FIELDS = ("name", "step_scale", "danger_shift", "phantom_strength",
                   "env_weight", "fast_speed", "base_move")


# ---------- 正常系 ----------

def test_3つのモードがそろっている():
    assert set(modes.MODE_KEYS) == {"walk", "bicycle", "car"}


@pytest.mark.parametrize("key", ["walk", "bicycle", "car"])
def test_モードには必要な項目が全部ある(key):
    profile = modes.mode_profile(key)
    for field in REQUIRED_FIELDS:
        assert field in profile, f"{field} が無い"


def test_モード名は日本語で表示できる():
    assert modes.mode_profile("walk")["name"] == "徒歩"
    assert modes.mode_profile("bicycle")["name"] == "自転車"
    assert modes.mode_profile("car")["name"] == "車"


def test_速い乗り物ほど遠くまで予測する():
    walk = modes.mode_profile("walk")
    bicycle = modes.mode_profile("bicycle")
    car = modes.mode_profile("car")
    assert walk["step_scale"] < bicycle["step_scale"] < car["step_scale"]


def test_速い乗り物ほど早めに警告する():
    # danger_shift が小さい(マイナス)ほど手前で危険域に入る = 早めの警告
    walk = modes.mode_profile("walk")
    car = modes.mode_profile("car")
    assert car["danger_shift"] < walk["danger_shift"]


def test_速い乗り物ほど基準の速度が高い():
    assert (modes.mode_profile("walk")["fast_speed"]
            < modes.mode_profile("bicycle")["fast_speed"]
            < modes.mode_profile("car")["fast_speed"])


@pytest.mark.parametrize("key", ["walk", "bicycle", "car"])
def test_割合の項目は0から1に収まる(key):
    profile = modes.mode_profile(key)
    for field in ("phantom_strength", "env_weight", "base_move"):
        assert 0 <= profile[field] <= 1, f"{field} が範囲の外"


def test_返した辞書を書きかえても元は変わらない():
    profile = modes.mode_profile("walk")
    profile["step_scale"] = 999
    assert modes.mode_profile("walk")["step_scale"] != 999


# ---------- 異常系 ----------

@pytest.mark.parametrize("bad", ["", "airplane", "WALK", None, 0, 1.5, True, [], {}])
def test_知らないモード名では止まる(bad):
    with pytest.raises(ValueError):
        modes.mode_profile(bad)


# ---------- ランダムテスト ----------

def test_でたらめな名前は全部エラーになる():
    rng = random.Random(6)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(200):
        name = "".join(rng.choice(letters) for _ in range(rng.randint(1, 10)))
        if name in modes.MODE_KEYS:
            assert modes.mode_profile(name)  # 偶然一致したら正常に返る
        else:
            with pytest.raises(ValueError):
                modes.mode_profile(name)
