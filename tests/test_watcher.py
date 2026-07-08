# watcher.py (全体の流れ) のテスト
#
# 実際の使われ方に近い形で、画像・位置情報を流し込んで動きを確かめる。
import random

import numpy as np
import pytest

from app import settings as settings_mod
from app import watcher as watcher_mod
from app.watcher import DangerWatcher


def make_frame(box_x=None, box_y=None, size=30, width=160, height=120):
    """テスト用の画像。黒い背景に白い四角。"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if box_x is not None and box_y is not None:
        x2 = min(box_x + size, width)
        y2 = min(box_y + size, height)
        frame[box_y:y2, box_x:x2] = 255
    return frame


def wall_frame(width=160, height=120):
    """左に壁(死角)のある風景。"""
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    frame[:, :int(width * 0.3)] = 40
    return frame


class FakeProvider:
    """テスト用の地図。決めた交差点を返す。"""

    def __init__(self, intersections=()):
        self.intersections = list(intersections)

    def intersections_near(self, lat, lon, radius_m=60.0):
        return self.intersections


RESULT_KEYS = ("level", "name", "text", "color", "sound", "reasons",
               "mode", "risk", "image_risk", "env_risk",
               "uncertainty", "optionality", "d_weight", "r_weight",
               "phase", "things", "blind_spots", "openness", "road", "boost",
               "danger_line", "detector")


# ---------- 正常系 (画像のみ) ----------

def test_何もない景色では安全のまま():
    watcher = DangerWatcher()
    for _ in range(10):
        result = watcher.watch(make_frame())
    assert result["level"] == 0
    assert result["name"] == "安全"
    assert result["risk"] == 0.0


def test_向かってくる物があると注意が出る():
    watcher = DangerWatcher()
    levels = []
    for i in range(10):
        frame = make_frame(box_x=65, box_y=8 + i * 8)
        result = watcher.watch(frame)
        levels.append(result["level"])
    assert max(levels) >= 1
    assert result["risk"] > 0.3
    assert "接近する物体があります" in result["reasons"]


def test_横切るだけの遠くの物では騒がない():
    watcher = DangerWatcher()
    for i in range(10):
        frame = make_frame(box_x=10 + i * 10, box_y=5, size=20)
        result = watcher.watch(frame)
    assert result["level"] == 0


def test_物がいなくなると注意はだんだん消える():
    watcher = DangerWatcher()
    for i in range(10):
        watcher.watch(make_frame(box_x=65, box_y=8 + i * 8))
    for _ in range(20):
        result = watcher.watch(make_frame())
    assert result["level"] == 0


def test_結果には必要な項目が全部ある():
    watcher = DangerWatcher()
    result = watcher.watch(make_frame())
    for key in RESULT_KEYS:
        assert key in result, f"{key} が無い"


def test_結果の数値はいつも0から1の間():
    watcher = DangerWatcher()
    for i in range(10):
        result = watcher.watch(make_frame(box_x=20 + i * 6, box_y=30 + i * 5))
        for key in ("risk", "image_risk", "env_risk", "uncertainty",
                    "optionality", "d_weight", "r_weight"):
            assert 0 <= result[key] <= 1, f"{key} が範囲の外: {result[key]}"
        assert result["level"] in (0, 1, 2)
        assert result["mode"] in ("探索", "適応", "安定")


def test_リセットしても学習は残る():
    watcher = DangerWatcher()
    watcher.watch(wall_frame())
    watcher.report_feedback(watcher_mod.FEEDBACK_MISSED)
    count_before = watcher.learner.feedback_count
    watcher.reset()
    assert watcher.level == 0
    assert watcher.belief.samples == []
    assert watcher.learner.feedback_count == count_before


# ---------- 検出のノイズ対策 (トラッカー) ----------

def test_1フレームだけのノイズでは騒がない():
    watcher = DangerWatcher()
    watcher.watch(make_frame())
    watcher.watch(make_frame(box_x=65, box_y=80))  # 一瞬だけ何かが映る
    for _ in range(10):
        result = watcher.watch(make_frame())
    assert result["level"] == 0
    assert result["risk"] == 0.0


def test_確定した物には番号がつく():
    watcher = DangerWatcher()
    for i in range(10):
        result = watcher.watch(make_frame(box_x=65, box_y=8 + i * 8))
    assert result["things"]
    assert all("id" in t for t in result["things"])


def test_モデルが無い環境では動き検出が使われる():
    watcher = DangerWatcher()
    result = watcher.watch(make_frame())
    assert result["detector"] == "motion"


def test_デモ映像の設定ではAI認識を使わない():
    watcher = DangerWatcher({"demo_mode": True, "use_ml_detector": True})
    assert watcher.object_finder is None


def test_AI認識をオフにすると動き検出になる():
    watcher = DangerWatcher({"use_ml_detector": False})
    assert watcher.object_finder is None
    result = watcher.watch(make_frame())
    assert result["detector"] == "motion"


class FakeObjectFinder:
    """テスト用のAI認識。決めた物をいつも返す。"""

    def __init__(self, things):
        self.things = things

    def find(self, frame):
        return [dict(t) for t in self.things]


def parked_car(cx=0.25, cy=0.6, w=0.3, h=0.25):
    return {
        "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h,
        "cx": cx, "cy": cy, "vx": 0.0, "vy": 0.0,
        "label": "car", "score": 0.9,
    }


def test_AI認識の止まった車の陰が死角として加わる():
    watcher = DangerWatcher()
    watcher.object_finder = FakeObjectFinder([parked_car()])
    quiet_frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    for _ in range(10):
        result = watcher.watch(quiet_frame)
    assert result["detector"] == "ml"
    vehicle_spots = [s for s in result["blind_spots"]
                     if s.get("source") == "vehicle"]
    assert len(vehicle_spots) == 1
    # 車の陰の分だけ、何も無い風景より危険度が高い
    plain = DangerWatcher()
    for _ in range(10):
        plain_result = plain.watch(quiet_frame)
    assert result["risk"] > plain_result["risk"]


def test_AI認識で物の名前が警告理由に出る():
    watcher = DangerWatcher()
    # 進路の中央へ向かってくる歩行者
    watcher.object_finder = FakeObjectFinder([{
        "x": 0.45, "y": 0.6, "w": 0.15, "h": 0.3,
        "cx": 0.52, "cy": 0.75, "vx": 0.0, "vy": 0.02,
        "label": "person", "score": 0.9,
    }])
    quiet_frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    for _ in range(10):
        result = watcher.watch(quiet_frame)
    assert result["risk"] > 0.15
    assert any("歩行者" in reason for reason in result["reasons"])


# ---------- 死角からの飛び出し予測 ----------

def test_死角があると何も動いていなくても警戒する():
    open_watcher = DangerWatcher()
    wall_watcher = DangerWatcher()
    for _ in range(10):
        open_result = open_watcher.watch(np.full((120, 160, 3), 180, dtype=np.uint8))
        wall_result = wall_watcher.watch(wall_frame())
    # 壁ぎわ(死角あり)の方が危険度が高い
    assert wall_result["risk"] > open_result["risk"]
    assert "見通しの悪い箇所があります" in wall_result["reasons"]
    assert len(wall_result["blind_spots"]) >= 1


def test_車モードは徒歩モードより死角を警戒する():
    walk = DangerWatcher({"mode": "walk"})
    car = DangerWatcher({"mode": "car"})
    for _ in range(10):
        walk_result = walk.watch(wall_frame())
        car_result = car.watch(wall_frame())
    assert car_result["risk"] > walk_result["risk"]


# ---------- モード切替 ----------

@pytest.mark.parametrize("mode", ["walk", "bicycle", "car"])
def test_どのモードでも普通に動く(mode):
    watcher = DangerWatcher({"mode": mode})
    for i in range(5):
        result = watcher.watch(make_frame(box_x=20 + i * 8, box_y=40))
    assert result["level"] in (0, 1, 2)


def test_設定でモードを切り替えられる():
    watcher = DangerWatcher({"mode": "walk"})
    watcher.apply_settings({"mode": "car"})
    assert watcher.settings["mode"] == "car"
    watcher.watch(make_frame())  # 切替後も動く


def test_結果の危険域はモードで変わる():
    walk = DangerWatcher({"mode": "walk"})
    car = DangerWatcher({"mode": "car"})
    walk_line = walk.watch(make_frame())["danger_line"]
    car_line = car.watch(make_frame())["danger_line"]
    assert car_line < walk_line  # 車は手前(小さい値)から危険域とみなす


# ---------- 位置情報との統合 ----------

def location_settings(**extra):
    values = {"use_location": True, **extra}
    return values


def test_交差点に近づくと危険度が上がる():
    watcher = DangerWatcher(location_settings(mode="car"))
    watcher.roads.provider = FakeProvider(intersections=[(35.0008, 139.0)])
    quiet_frame = np.full((120, 160, 3), 180, dtype=np.uint8)

    # 交差点から離れた場所を走っている
    watcher.update_location(34.995, 139.0, 0.0)
    watcher.update_location(34.9951, 139.0, 1.0)
    far_result = watcher.watch(quiet_frame)

    # 交差点の目の前まで来た (速度も出ている)
    watcher.update_location(35.0006, 139.0, 10.0)
    watcher.update_location(35.0007, 139.0, 11.0)
    near_result = watcher.watch(quiet_frame)

    assert near_result["env_risk"] > far_result["env_risk"]
    assert "この先に交差点があります" in near_result["reasons"]


def test_位置情報を使わない設定ならGPSは無視される():
    watcher = DangerWatcher({"use_location": False})
    watcher.update_location(35.0, 139.0, 0.0)
    assert watcher.roads.track.points == []
    assert watcher.roads.provider is None


def test_位置情報を使う設定なら地図はOSMになる():
    from app import roadinfo
    watcher = DangerWatcher({"use_location": True})
    assert isinstance(watcher.roads.provider, roadinfo.OsmProvider)


def test_重複したGPS通知が来ても壊れない():
    watcher = DangerWatcher(location_settings())
    watcher.roads.provider = None  # 地図は使わない
    watcher.update_location(35.0, 139.0, 0.0)
    watcher.update_location(35.0, 139.0, 0.0)  # 同じ通知がもう一度来た
    watcher.update_location(35.00001, 139.0, 1.0)
    assert len(watcher.roads.track.points) == 2


def test_おかしな位置情報は止まる():
    watcher = DangerWatcher(location_settings())
    with pytest.raises(ValueError):
        watcher.update_location(999, 139.0, 0.0)


def test_速度が出ているほど飛び出しを警戒する():
    slow = DangerWatcher(location_settings(mode="bicycle"))
    fast = DangerWatcher(location_settings(mode="bicycle"))
    # ゆっくり (約1m/s) と 速い (約8m/s)
    slow.update_location(35.0, 139.0, 0.0)
    slow.update_location(35.00001, 139.0, 1.0)
    fast.update_location(35.0, 139.0, 0.0)
    fast.update_location(35.00007, 139.0, 1.0)
    for _ in range(10):
        slow_result = slow.watch(wall_frame())
        fast_result = fast.watch(wall_frame())
    assert fast_result["risk"] > slow_result["risk"]


# ---------- 学習 (ユーザーからの報告) ----------

def test_誤報の報告で同じ状況の警戒が弱まる():
    watcher = DangerWatcher()
    for _ in range(10):
        before = watcher.watch(wall_frame())
    for _ in range(5):
        watcher.report_feedback(watcher_mod.FEEDBACK_FALSE_ALARM)
    for _ in range(10):
        after = watcher.watch(wall_frame())
    assert after["boost"] < before["boost"]
    assert after["risk"] < before["risk"]


def test_見逃しの報告で同じ状況の警戒が強まる():
    watcher = DangerWatcher()
    for _ in range(10):
        before = watcher.watch(wall_frame())
    for _ in range(5):
        watcher.report_feedback(watcher_mod.FEEDBACK_MISSED)
    for _ in range(10):
        after = watcher.watch(wall_frame())
    assert after["boost"] > before["boost"]
    assert after["risk"] >= before["risk"]


def test_学習をオフにすると報告は取り込まれない():
    watcher = DangerWatcher({"learning_on": False})
    watcher.watch(wall_frame())
    assert watcher.report_feedback(watcher_mod.FEEDBACK_MISSED) is False
    assert watcher.learner.feedback_count == 0


def test_おかしな報告は止まる():
    watcher = DangerWatcher()
    with pytest.raises(ValueError):
        watcher.report_feedback("なんとなく")


def test_学習はファイルに保存されて引き継がれる(tmp_path):
    path = str(tmp_path / "learned.json")
    watcher = DangerWatcher(learn_path=path)
    watcher.watch(wall_frame())
    watcher.report_feedback(watcher_mod.FEEDBACK_MISSED)

    next_watcher = DangerWatcher(learn_path=path)
    assert next_watcher.learner.feedback_count == 1


# ---------- 設定との組み合わせ ----------

def test_設定を渡して作れる():
    values = dict(settings_mod.DEFAULTS)
    values["sensitivity"] = 8
    watcher = DangerWatcher(values)
    assert watcher.settings["sensitivity"] == 8


def test_おかしな設定は自動で直る():
    watcher = DangerWatcher({"sensitivity": 9999, "update_rate": "壊れた値",
                             "mode": "rocket"})
    assert watcher.settings["sensitivity"] == 10
    assert watcher.settings["update_rate"] == settings_mod.DEFAULTS["update_rate"]
    assert watcher.settings["mode"] == settings_mod.DEFAULTS["mode"]
    watcher.watch(make_frame())


def test_途中で設定を変えても動き続ける():
    watcher = DangerWatcher()
    for i in range(5):
        watcher.watch(make_frame(box_x=20 + i * 8, box_y=40))
    watcher.apply_settings({"sensitivity": 10, "future_steps": 30})
    assert watcher.settings["future_steps"] == 30
    for i in range(5):
        result = watcher.watch(make_frame(box_x=60 + i * 8, box_y=40))
    assert result["level"] in (0, 1, 2)


# ---------- 異常系 ----------

def test_画像がNoneなら止まる():
    watcher = DangerWatcher()
    with pytest.raises(ValueError):
        watcher.watch(None)


def test_画像でないものを渡すと止まる():
    watcher = DangerWatcher()
    with pytest.raises(ValueError):
        watcher.watch([1, 2, 3])


def test_エラーの後もそのまま使い続けられる():
    watcher = DangerWatcher()
    with pytest.raises(ValueError):
        watcher.watch(None)
    result = watcher.watch(make_frame())
    assert result["level"] in (0, 1, 2)


# ---------- 境界値 ----------

def test_一番小さい設定でも動く():
    values = {key: low for key, (low, _) in settings_mod.RANGES.items()}
    watcher = DangerWatcher(values)
    for i in range(5):
        result = watcher.watch(make_frame(box_x=20 + i * 10, box_y=40))
    assert result["level"] in (0, 1, 2)


def test_一番大きい設定でも動く():
    values = {key: high for key, (_, high) in settings_mod.RANGES.items()}
    values["camera_no"] = 0
    watcher = DangerWatcher(values)
    for i in range(5):
        result = watcher.watch(make_frame(box_x=20 + i * 10, box_y=40))
    assert result["level"] in (0, 1, 2)


def test_とても小さな画像でも動く():
    watcher = DangerWatcher()
    for _ in range(5):
        result = watcher.watch(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result["level"] == 0


# ---------- ランダムテスト ----------

def test_でたらめな画像を流し続けても壊れず答えもまとも():
    rng = np.random.default_rng(2)
    watcher = DangerWatcher()
    for _ in range(60):
        frame = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
        result = watcher.watch(frame)
        for key in RESULT_KEYS:
            assert key in result
        for key in ("risk", "image_risk", "env_risk", "uncertainty",
                    "optionality", "d_weight", "r_weight"):
            assert 0 <= result[key] <= 1
        assert result["level"] in (0, 1, 2)


def test_でたらめな設定と画像と報告の組み合わせでも壊れない():
    np_rng = np.random.default_rng(3)
    rng = random.Random(3)
    for _ in range(10):
        values = {
            "sensitivity": rng.randint(-5, 20),
            "future_steps": rng.randint(-10, 100),
            "future_samples": rng.randint(-10, 300),
            "update_rate": rng.uniform(-1, 2),
            "risk_line": rng.uniform(-1, 2),
            "danger_line": rng.uniform(-1, 2),
            "min_size": rng.uniform(-1, 1),
            "mode": rng.choice(["walk", "bicycle", "car", "rocket", ""]),
            "stabilize": rng.choice([True, False]),
        }
        watcher = DangerWatcher(values)
        for _ in range(5):
            frame = np_rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
            result = watcher.watch(frame)
            assert result["level"] in (0, 1, 2)
            if rng.random() < 0.3:
                watcher.report_feedback(rng.choice([
                    watcher_mod.FEEDBACK_FALSE_ALARM,
                    watcher_mod.FEEDBACK_MISSED,
                ]))
