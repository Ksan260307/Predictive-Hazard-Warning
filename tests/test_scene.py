# scene.py (風景から死角・見通しを読む) のテスト
import numpy as np
import pytest

from app.scene import SceneReader


def open_road_frame(width=160, height=120, brightness=170):
    """見通しの良い一様な風景。"""
    return np.full((height, width, 3), brightness, dtype=np.uint8)


def wall_frame(boundary_x=0.3, width=160, height=120):
    """左側に壁がある風景。boundary_x の位置に強い縦の境界ができる。"""
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    frame[:, :int(width * boundary_x)] = 40  # 暗い壁
    return frame


def canyon_frame(width=160, height=120):
    """左右両方に壁がある風景 (ビルの谷間など)。"""
    frame = np.full((height, width, 3), 180, dtype=np.uint8)
    frame[:, :int(width * 0.25)] = 40
    frame[:, int(width * 0.8):] = 40
    return frame


# ---------- 正常系 ----------

def test_壁の境界が死角として見つかる():
    scene = SceneReader().read(wall_frame(boundary_x=0.3))
    assert len(scene["blind_spots"]) >= 1
    # 境界のあたり(x=0.3付近)に見つかる
    assert any(abs(s["x"] - 0.3) < 0.08 for s in scene["blind_spots"])


def test_一様な風景では死角は見つからない():
    scene = SceneReader().read(open_road_frame())
    assert scene["blind_spots"] == []
    assert scene["blind_risk"] == 0.0


def test_両側に壁があると死角が2つ見つかる():
    scene = SceneReader().read(canyon_frame())
    assert len(scene["blind_spots"]) == 2


def test_開けた道は壁ぎわの道より見通しが良い():
    open_scene = SceneReader().read(open_road_frame())
    # 中央に物が多い風景 (進行方向に縦の境界がたくさんある)
    cluttered = np.full((120, 160, 3), 180, dtype=np.uint8)
    for x in range(60, 100, 8):
        cluttered[:, x:x + 4] = 40
    cluttered_scene = SceneReader().read(cluttered)
    assert open_scene["openness"] > cluttered_scene["openness"]


def test_死角の強さと位置は0から1の範囲():
    scene = SceneReader().read(wall_frame())
    for spot in scene["blind_spots"]:
        assert 0 <= spot["x"] <= 1
        assert 0 <= spot["y"] <= 1
        assert 0 <= spot["strength"] <= 1


def test_白黒画像でも読める():
    gray = wall_frame()[:, :, 0]
    scene = SceneReader().read(gray)
    assert len(scene["blind_spots"]) >= 1


# ---------- 異常系 ----------

def test_画像がNoneなら止まる():
    with pytest.raises(ValueError):
        SceneReader().read(None)


def test_画像でないものを渡すと止まる():
    with pytest.raises(ValueError):
        SceneReader().read("画像ではない")


def test_空の画像なら止まる():
    with pytest.raises(ValueError):
        SceneReader().read(np.zeros((0, 0, 3), dtype=np.uint8))


def test_おかしな形の画像なら止まる():
    with pytest.raises(ValueError):
        SceneReader().read(np.zeros((2, 3, 4, 5), dtype=np.uint8))


# ---------- 境界値 ----------

def test_とても小さな画像でも壊れない():
    scene = SceneReader().read(np.full((8, 8, 3), 100, dtype=np.uint8))
    assert 0 <= scene["blind_risk"] <= 1
    assert 0 <= scene["openness"] <= 1


def test_真っ黒と真っ白の画像でも壊れない():
    for value in (0, 255):
        scene = SceneReader().read(np.full((120, 160, 3), value, dtype=np.uint8))
        assert scene["blind_spots"] == []


def test_画面のはしの境界は死角として拾わない():
    # x=0.05 (探す範囲の外) にある境界は無視される
    frame = np.full((120, 160, 3), 180, dtype=np.uint8)
    frame[:, :8] = 40
    scene = SceneReader().read(frame)
    assert all(s["x"] > 0.08 for s in scene["blind_spots"])


# ---------- ランダムテスト ----------

def test_でたらめな画像でも値は必ず範囲内():
    rng = np.random.default_rng(4)
    reader = SceneReader()
    for _ in range(60):
        h = int(rng.integers(8, 120))
        w = int(rng.integers(8, 160))
        frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        scene = reader.read(frame)
        assert 0 <= scene["blind_risk"] <= 1
        assert 0 <= scene["openness"] <= 1
        for spot in scene["blind_spots"]:
            assert 0 <= spot["x"] <= 1
            assert 0 <= spot["strength"] <= 1


# ---------- 止まっている車の陰 (AI認識との連携) ----------

from app.scene import merge_blind_spots, vehicle_blind_spots


def car_thing(cx=0.3, cy=0.6, w=0.3, h=0.2, label="car", vx=0.0, vy=0.0):
    """テスト用の「車」の検出結果。"""
    return {
        "x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h,
        "cx": cx, "cy": cy, "vx": vx, "vy": vy, "label": label,
    }


def test_止まっている車の陰が死角になる():
    spots = vehicle_blind_spots([car_thing(cx=0.3)])
    assert len(spots) == 1
    spot = spots[0]
    # 左側の車なら、進路の中央に近い「右端」が飛び出し口
    assert spot["x"] == pytest.approx(0.3 + 0.15)
    assert spot["source"] == "vehicle"
    assert 0 < spot["strength"] <= 1


def test_右側の車は左端が死角になる():
    spots = vehicle_blind_spots([car_thing(cx=0.7)])
    assert spots[0]["x"] == pytest.approx(0.7 - 0.15)


def test_走っている車は死角にならない():
    assert vehicle_blind_spots([car_thing(vx=0.05)]) == []


def test_小さすぎる車は死角にならない():
    assert vehicle_blind_spots([car_thing(w=0.05, h=0.05)]) == []


def test_車以外は死角にならない():
    assert vehicle_blind_spots([car_thing(label="person")]) == []
    assert vehicle_blind_spots([car_thing(label=None)]) == []
    thing = car_thing()
    del thing["label"]  # 動き検出の物 (ラベル無し)
    assert vehicle_blind_spots([thing]) == []


def test_大きい車ほど死角が強い():
    small = vehicle_blind_spots([car_thing(w=0.2, h=0.15)])[0]
    big = vehicle_blind_spots([car_thing(w=0.5, h=0.3)])[0]
    assert big["strength"] > small["strength"]


def test_近い死角候補は強い方だけが残る():
    spots = [
        {"x": 0.30, "y": 0.6, "strength": 0.5},
        {"x": 0.33, "y": 0.6, "strength": 0.9},  # 上と同じ場所を指している
        {"x": 0.80, "y": 0.6, "strength": 0.4},
    ]
    merged = merge_blind_spots(spots)
    assert len(merged) == 2
    strengths = {round(s["strength"], 2) for s in merged}
    assert strengths == {0.9, 0.4}


def test_離れた死角候補は全部残る():
    spots = [
        {"x": 0.1, "y": 0.6, "strength": 0.5},
        {"x": 0.5, "y": 0.6, "strength": 0.5},
        {"x": 0.9, "y": 0.6, "strength": 0.5},
    ]
    assert len(merge_blind_spots(spots)) == 3


def test_まとめる距離がおかしい時は止まる():
    with pytest.raises(ValueError):
        merge_blind_spots([], min_gap=0)
    with pytest.raises(ValueError):
        merge_blind_spots([], min_gap=1.5)
