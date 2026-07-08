# learning.py (報告からの学習) のテスト
import json
import random

import pytest

from app.learning import (
    BOOST_MAX,
    BOOST_MIN,
    DIRECTION_FALSE_ALARM,
    DIRECTION_MISSED,
    FEATURE_NAMES,
    WEIGHT_LIMIT,
    RiskLearner,
)


def situation(**kwargs):
    """テスト用の状況。指定した項目以外は0。"""
    features = {name: 0.0 for name in FEATURE_NAMES}
    features.update(kwargs)
    return features


# ---------- 正常系 ----------

def test_学習前は倍率1():
    learner = RiskLearner()
    assert learner.boost(situation(blind_spot=1.0)) == pytest.approx(1.0)


def test_誤報の報告で予測が弱まる():
    learner = RiskLearner()
    features = situation(blind_spot=1.0, mode_walk=1.0)
    learner.learn(features, DIRECTION_FALSE_ALARM)
    assert learner.boost(features) < 1.0


def test_見逃しの報告で予測が強まる():
    learner = RiskLearner()
    features = situation(intersection=1.0, mode_car=1.0)
    learner.learn(features, DIRECTION_MISSED)
    assert learner.boost(features) > 1.0


def test_学習は報告があった状況にだけ効く():
    learner = RiskLearner()
    walk_features = situation(blind_spot=1.0, mode_walk=1.0)
    car_features = situation(intersection=1.0, mode_car=1.0)
    learner.learn(walk_features, DIRECTION_FALSE_ALARM)
    # 徒歩+死角の状況は弱まるが、車+交差点の状況は変わらない
    assert learner.boost(walk_features) < 1.0
    assert learner.boost(car_features) == pytest.approx(1.0)


def test_報告の回数が数えられる():
    learner = RiskLearner()
    learner.learn(situation(moving=1.0), DIRECTION_MISSED)
    learner.learn(situation(moving=1.0), DIRECTION_FALSE_ALARM)
    assert learner.feedback_count == 2


def test_リセットで最初に戻る():
    learner = RiskLearner()
    learner.learn(situation(blind_spot=1.0), DIRECTION_MISSED)
    learner.reset()
    assert learner.boost(situation(blind_spot=1.0)) == pytest.approx(1.0)
    assert learner.feedback_count == 0


# ---------- 学習の安全弁 (報告が少ないうちは控えめ) ----------

def test_同じ重みでも報告が多いほど強く効く():
    few = RiskLearner()
    many = RiskLearner()
    few.weights["blind_spot"] = -1.0
    few.feedback_count = 1
    many.weights["blind_spot"] = -1.0
    many.feedback_count = 50
    features = situation(blind_spot=1.0)
    # 報告50件の方が、1件よりも強く弱められる
    assert many.boost(features) < few.boost(features) < 1.0


def test_報告0件なら重みがあっても倍率1():
    learner = RiskLearner()
    learner.weights["blind_spot"] = 1.0  # ファイル破損などで重みだけ残った場合
    assert learner.boost(situation(blind_spot=1.0)) == pytest.approx(1.0)


def test_信頼度は報告が増えるほど1に近づく():
    learner = RiskLearner()
    assert learner.confidence() == 0.0
    values = []
    for _ in range(50):
        learner.learn(situation(moving=0.5), DIRECTION_MISSED)
        values.append(learner.confidence())
    assert all(0 < v < 1 for v in values)
    assert values == sorted(values)  # 単調に増える
    assert values[-1] > 0.9


def test_数件の報告では予測は大きく振れない():
    learner = RiskLearner()
    features = situation(**{name: 1.0 for name in FEATURE_NAMES})
    learner.learn(features, DIRECTION_MISSED)
    # 1件の報告では倍率は 1.0 の近くにとどまる
    assert 1.0 < learner.boost(features) < 1.5


# ---------- 保存と読み込み ----------

def test_保存して読み込むと学習が引き継がれる(tmp_path):
    path = str(tmp_path / "learned.json")
    learner = RiskLearner(path=path)
    features = situation(blind_spot=1.0)
    learner.learn(features, DIRECTION_FALSE_ALARM)  # 自動保存される
    boost_before = learner.boost(features)

    loaded = RiskLearner(path=path)
    assert loaded.boost(features) == pytest.approx(boost_before)
    assert loaded.feedback_count == 1


def test_ファイルが無くても普通に始まる(tmp_path):
    learner = RiskLearner(path=str(tmp_path / "nai.json"))
    assert learner.boost(situation()) == pytest.approx(1.0)


def test_壊れたファイルでも普通に始まる(tmp_path):
    path = tmp_path / "kowareta.json"
    path.write_text("{壊れたJSON", encoding="utf-8")
    learner = RiskLearner(path=str(path))
    assert learner.boost(situation(blind_spot=1.0)) == pytest.approx(1.0)


def test_中身がおかしいファイルでも安全な値に直す(tmp_path):
    path = tmp_path / "hen.json"
    data = {"weights": {"blind_spot": 9999, "intersection": "文字", "nazo": 5},
            "feedback_count": -10}
    path.write_text(json.dumps(data), encoding="utf-8")
    learner = RiskLearner(path=str(path))
    assert learner.weights["blind_spot"] == WEIGHT_LIMIT  # 上限に収まる
    assert learner.weights["intersection"] == 0.0          # 文字は捨てる
    assert "nazo" not in learner.weights                   # 知らない項目は捨てる
    assert learner.feedback_count == 0                     # マイナスは受け付けない


# ---------- 異常系 ----------

def test_おかしな方向の報告は止まる():
    learner = RiskLearner()
    for bad in (0, 2, -2, "missed", None, 0.5):
        with pytest.raises(ValueError):
            learner.learn(situation(), bad)


def test_状況が辞書でなければ止まる():
    learner = RiskLearner()
    with pytest.raises(ValueError):
        learner.boost("状況ではない")
    with pytest.raises(ValueError):
        learner.learn(None, DIRECTION_MISSED)


def test_おかしな学習の速さは止まる():
    learner = RiskLearner()
    with pytest.raises(ValueError):
        learner.learn(situation(), DIRECTION_MISSED, rate=0)
    with pytest.raises(ValueError):
        learner.learn(situation(), DIRECTION_MISSED, rate=1.5)


def test_状況の変な値は無視される():
    learner = RiskLearner()
    weird = {"blind_spot": float("nan"), "intersection": "多い",
             "moving": float("inf"), "turning": True, "object": -5}
    # エラーにならず、まともな値だけ使われる
    boost = learner.boost(weird)
    assert BOOST_MIN <= boost <= BOOST_MAX


# ---------- 境界値 ----------

def test_何度報告しても倍率は上限と下限で止まる():
    learner = RiskLearner()
    features = situation(**{name: 1.0 for name in FEATURE_NAMES})
    for _ in range(100):
        learner.learn(features, DIRECTION_MISSED)
    assert learner.boost(features) == BOOST_MAX
    for name in FEATURE_NAMES:
        assert learner.weights[name] == WEIGHT_LIMIT

    for _ in range(200):
        learner.learn(features, DIRECTION_FALSE_ALARM)
    assert learner.boost(features) == BOOST_MIN
    for name in FEATURE_NAMES:
        assert learner.weights[name] == -WEIGHT_LIMIT


def test_空の状況では学習しても何も変わらない():
    learner = RiskLearner()
    learner.learn({}, DIRECTION_MISSED)
    assert all(w == 0.0 for w in learner.weights.values())
    assert learner.feedback_count == 1  # 回数だけは数える


# ---------- ランダムテスト ----------

def test_でたらめに報告し続けても倍率は必ず範囲内(tmp_path):
    rng = random.Random(13)
    learner = RiskLearner(path=str(tmp_path / "learn.json"))
    for _ in range(300):
        features = {name: rng.uniform(-2, 2) for name in FEATURE_NAMES}
        direction = rng.choice([DIRECTION_MISSED, DIRECTION_FALSE_ALARM])
        learner.learn(features, direction)
        boost = learner.boost(features)
        assert BOOST_MIN <= boost <= BOOST_MAX
        for weight in learner.weights.values():
            assert -WEIGHT_LIMIT <= weight <= WEIGHT_LIMIT
    # 保存されたファイルも読める
    reloaded = RiskLearner(path=str(tmp_path / "learn.json"))
    assert reloaded.feedback_count == 300
