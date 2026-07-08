# tracker.py (検出結果の時間方向のならし) のテスト
import pytest

from app.tracker import ThingTracker


def thing(cx=0.5, cy=0.5, w=0.1, h=0.1, **extra):
    """テスト用の検出結果を作る。"""
    data = {
        "x": cx - w / 2, "y": cy - h / 2,
        "w": w, "h": h,
        "cx": cx, "cy": cy,
        "vx": 0.0, "vy": 0.0,
    }
    data.update(extra)
    return data


# ---------- 基本の動き ----------

def test_1フレームだけの検出は通らない():
    tracker = ThingTracker(min_hits=3)
    assert tracker.update([thing()]) == []
    assert tracker.update([]) == []
    assert tracker.update([]) == []


def test_連続で見えた物だけが確定して返る():
    tracker = ThingTracker(min_hits=3)
    assert tracker.update([thing(cx=0.5)]) == []
    assert tracker.update([thing(cx=0.51)]) == []
    confirmed = tracker.update([thing(cx=0.52)])
    assert len(confirmed) == 1
    assert confirmed[0]["id"] == 1


def test_確定した物には同じidがつき続ける():
    tracker = ThingTracker(min_hits=1)
    first = tracker.update([thing(cx=0.5)])
    second = tracker.update([thing(cx=0.52)])
    assert first[0]["id"] == second[0]["id"]


def test_離れた2つの物は別のidになる():
    tracker = ThingTracker(min_hits=1)
    confirmed = tracker.update([thing(cx=0.2), thing(cx=0.8)])
    assert len(confirmed) == 2
    assert confirmed[0]["id"] != confirmed[1]["id"]


def test_速度は複数フレームからならして求められる():
    tracker = ThingTracker(min_hits=1, smooth=0.6)
    tracker.update([thing(cx=0.3)])
    tracker.update([thing(cx=0.35)])
    confirmed = tracker.update([thing(cx=0.40)])
    # 1フレームあたり +0.05 で動いている
    assert confirmed[0]["vx"] == pytest.approx(0.05, abs=0.02)
    assert confirmed[0]["vy"] == pytest.approx(0.0, abs=0.001)


def test_labelとscoreは引き継がれる():
    tracker = ThingTracker(min_hits=2)
    tracker.update([thing(label="person", score=0.9)])
    confirmed = tracker.update([thing(cx=0.51, label="person", score=0.85)])
    assert confirmed[0]["label"] == "person"
    assert confirmed[0]["score"] == 0.85


# ---------- 見失いへの強さ ----------

def test_少しの間見失っても追い続ける():
    tracker = ThingTracker(min_hits=2, max_missed=2)
    tracker.update([thing(cx=0.5)])
    tracker.update([thing(cx=0.52)])
    # 1フレーム見失う → 位置を進めて生き残る
    coasted = tracker.update([])
    assert len(coasted) == 1
    # 戻ってきたら同じ物として続く
    back = tracker.update([thing(cx=0.56)])
    assert back[0]["id"] == coasted[0]["id"]


def test_見失っている間は速度で位置が進む():
    tracker = ThingTracker(min_hits=2, max_missed=3)
    tracker.update([thing(cx=0.3)])
    tracker.update([thing(cx=0.35)])
    coasted = tracker.update([])
    assert coasted[0]["cx"] > 0.35


def test_長く見えない物は忘れられる():
    tracker = ThingTracker(min_hits=1, max_missed=1)
    tracker.update([thing(cx=0.5)])
    tracker.update([])   # 1回目の見失い (まだ生きている)
    assert tracker.update([]) == []  # 2回目で削除
    # 同じ場所に再び現れても新しいid
    confirmed = tracker.update([thing(cx=0.5)])
    assert confirmed[0]["id"] != 1


def test_リセットで全部忘れる():
    tracker = ThingTracker(min_hits=1)
    tracker.update([thing()])
    tracker.reset()
    # リセット直後は min_hits=1 なら新しい物としてすぐ確定する
    confirmed = tracker.update([thing()])
    assert confirmed[0]["id"] == 2  # idは通し番号のまま


# ---------- 対応づけ ----------

def test_近い物どうしが正しく対応づく():
    tracker = ThingTracker(min_hits=1)
    tracker.update([thing(cx=0.2), thing(cx=0.8)])
    confirmed = tracker.update([thing(cx=0.22), thing(cx=0.78)])
    by_cx = sorted(confirmed, key=lambda t: t["cx"])
    assert by_cx[0]["id"] == 1
    assert by_cx[1]["id"] == 2


def test_遠すぎる検出は別の物とみなす():
    tracker = ThingTracker(min_hits=1, match_distance=0.1, max_missed=0)
    tracker.update([thing(cx=0.1)])
    confirmed = tracker.update([thing(cx=0.9)])
    assert confirmed[0]["id"] == 2


# ---------- 異常系 ----------

def test_おかしな設定は止まる():
    with pytest.raises(ValueError):
        ThingTracker(min_hits=0)
    with pytest.raises(ValueError):
        ThingTracker(min_hits=2.5)
    with pytest.raises(ValueError):
        ThingTracker(max_missed=-1)
    with pytest.raises(ValueError):
        ThingTracker(match_distance=0)
    with pytest.raises(ValueError):
        ThingTracker(match_distance=1.5)
    with pytest.raises(ValueError):
        ThingTracker(smooth=0)
    with pytest.raises(ValueError):
        ThingTracker(smooth=2)


def test_おかしな検出結果は止まる():
    tracker = ThingTracker()
    with pytest.raises(ValueError):
        tracker.update("これはリストではない")
    with pytest.raises(ValueError):
        tracker.update([{"cx": 0.5}])  # 項目が足りない
    with pytest.raises(ValueError):
        tracker.update([123])


# ---------- ランダムテスト ----------

def test_でたらめな検出を流し続けても壊れない():
    import random
    rng = random.Random(1)
    tracker = ThingTracker()
    for _ in range(100):
        things = [thing(cx=rng.random(), cy=rng.random(),
                        w=rng.uniform(0.01, 0.5), h=rng.uniform(0.01, 0.5))
                  for _ in range(rng.randint(0, 5))]
        confirmed = tracker.update(things)
        for t in confirmed:
            assert "id" in t
            assert isinstance(t["vx"], float)
