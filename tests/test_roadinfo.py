# roadinfo.py (GPSと地図) のテスト
import random

import pytest

from app import roadinfo
from app.roadinfo import (
    GpsTrack,
    OsmProvider,
    RoadWatcher,
    distance_m,
    heading_deg,
    _parse_overpass,
)


class FakeProvider:
    """テスト用の地図。決めた交差点を返すだけ。失敗もさせられる。"""

    def __init__(self, intersections=(), fail=False):
        self.intersections = list(intersections)
        self.fail = fail
        self.call_count = 0

    def intersections_near(self, lat, lon, radius_m=60.0):
        self.call_count += 1
        if self.fail:
            raise OSError("圏外")
        return self.intersections


# ---------- 距離と方角 ----------

def test_同じ地点なら距離0():
    assert distance_m(35.0, 139.0, 35.0, 139.0) == 0.0


def test_緯度1度はおよそ111km():
    d = distance_m(35.0, 139.0, 36.0, 139.0)
    assert 110000 < d < 112000


def test_北へ進むと方角は0度():
    assert heading_deg(35.0, 139.0, 35.001, 139.0) == pytest.approx(0.0, abs=0.1)


def test_東へ進むと方角は90度():
    assert heading_deg(35.0, 139.0, 35.0, 139.001) == pytest.approx(90.0, abs=0.5)


# ---------- GpsTrack 正常系 ----------

def test_2点あれば速度が出る():
    track = GpsTrack()
    track.add(35.0, 139.0, 0.0)
    track.add(35.00009, 139.0, 10.0)  # 北へ約10mを10秒 → 約1m/s
    assert track.speed() == pytest.approx(1.0, abs=0.1)


def test_1点だけでは速度は不明():
    track = GpsTrack()
    assert track.speed() is None
    track.add(35.0, 139.0, 0.0)
    assert track.speed() is None


def test_まっすぐ進むと旋回は0():
    track = GpsTrack()
    for i in range(3):
        track.add(35.0 + i * 0.0001, 139.0, float(i))
    assert track.turn_rate() == pytest.approx(0.0, abs=1.0)


def test_直角に曲がると旋回が出る():
    track = GpsTrack()
    track.add(35.0, 139.0, 0.0)
    track.add(35.0001, 139.0, 1.0)      # 北へ
    track.add(35.0001, 139.0001, 2.0)   # 東へ (90度曲がった)
    assert abs(track.turn_rate()) == pytest.approx(45.0, abs=5.0)  # 90度/2秒


def test_止まっている時は旋回なしとする():
    track = GpsTrack()
    track.add(35.0, 139.0, 0.0)
    track.add(35.0, 139.0 + 1e-9, 1.0)  # ほぼ動いていない
    track.add(35.0001, 139.0, 2.0)
    assert track.turn_rate() == 0.0


def test_履歴は上限を超えない():
    track = GpsTrack(keep=5)
    for i in range(20):
        track.add(35.0 + i * 0.0001, 139.0, float(i))
    assert len(track.points) == 5


# ---------- GpsTrack 異常系・境界値 ----------

def test_緯度経度が範囲の外なら止まる():
    track = GpsTrack()
    with pytest.raises(ValueError):
        track.add(91.0, 139.0, 0.0)
    with pytest.raises(ValueError):
        track.add(35.0, 181.0, 0.0)
    with pytest.raises(ValueError):
        track.add(-91.0, 139.0, 0.0)
    with pytest.raises(ValueError):
        track.add(35.0, -181.0, 0.0)


def test_緯度経度のちょうど端は使える():
    track = GpsTrack()
    track.add(90.0, 180.0, 0.0)
    track.add(-90.0, -180.0, 1.0)


def test_時刻が進んでいない記録は無視される():
    track = GpsTrack()
    assert track.add(35.0, 139.0, 10.0) is True
    # GPSは同じ通知を重複して寄こすことがあるので、エラーにせず無視する
    assert track.add(35.1, 139.0, 5.0) is False
    assert track.add(35.1, 139.0, 10.0) is False  # 同時刻も無視
    assert len(track.points) == 1
    assert track.points[0] == (35.0, 139.0, 10.0)  # 中身は変わらない


def test_文字やTrueを渡すと止まる():
    track = GpsTrack()
    with pytest.raises(ValueError):
        track.add("35.0", 139.0, 0.0)
    with pytest.raises(ValueError):
        track.add(35.0, True, 0.0)
    with pytest.raises(ValueError):
        track.add(35.0, 139.0, None)


def test_履歴の数が少なすぎる作り方は止まる():
    with pytest.raises(ValueError):
        GpsTrack(keep=2)


# ---------- RoadWatcher ----------

def test_情報がない時は影響なしの状態を返す():
    watcher = RoadWatcher()
    state = watcher.state()
    assert state == roadinfo.NEUTRAL_STATE


def test_交差点に近づくと接近度が上がる():
    # 交差点は (35.001, 139.0)。南から北へ近づいていく
    provider = FakeProvider(intersections=[(35.001, 139.0)])
    watcher = RoadWatcher(provider=provider)
    scores = []
    for i in range(5):
        watcher.update(35.0 + i * 0.0002, 139.0, float(i))
        scores.append(watcher.state()["intersection_score"])
    assert scores[-1] > scores[0]
    assert all(0 <= x <= 1 for x in scores)


def test_交差点から遠いと接近度0():
    provider = FakeProvider(intersections=[(36.0, 140.0)])  # 100km以上先
    watcher = RoadWatcher(provider=provider)
    watcher.update(35.0, 139.0, 0.0)
    assert watcher.state()["intersection_score"] == 0.0


def test_地図が失敗してもアプリは止まらない():
    provider = FakeProvider(fail=True)
    watcher = RoadWatcher(provider=provider)
    watcher.update(35.0, 139.0, 0.0)  # エラーにならない
    state = watcher.state()
    assert state["intersection_score"] == 0.0
    assert state["speed"] is None  # 1点しかないので速度は不明


def test_地図には動いた時だけ問い合わせる():
    provider = FakeProvider(intersections=[(35.001, 139.0)])
    watcher = RoadWatcher(provider=provider, refresh_m=30.0)
    watcher.update(35.0, 139.0, 0.0)
    watcher.update(35.0000001, 139.0, 1.0)  # ほぼ動いていない
    watcher.update(35.0000002, 139.0, 2.0)
    assert provider.call_count == 1  # 最初の1回だけ


def test_地図なしでも速度と旋回は出る():
    watcher = RoadWatcher(provider=None)
    for i in range(3):
        watcher.update(35.0 + i * 0.0001, 139.0, float(i * 10))
    state = watcher.state()
    assert state["speed"] is not None
    assert state["intersection_score"] == 0.0


# ---------- OpenStreetMap の応答の読み取り ----------

def osm_response(ways, nodes):
    """Overpass API の応答を合成する。

    ways  : {way番号: [点の番号, ...]}
    nodes : {点の番号: (緯度, 経度)}
    """
    elements = []
    for way_id, node_ids in ways.items():
        elements.append({"type": "way", "id": way_id, "nodes": list(node_ids)})
    for node_id, (lat, lon) in nodes.items():
        elements.append({"type": "node", "id": node_id, "lat": lat, "lon": lon})
    return {"elements": elements}


def test_2本の道路が共有する点は交差点になる():
    data = osm_response(
        ways={1: [10, 11, 12], 2: [20, 11, 21]},  # 点11を共有している
        nodes={10: (35.0, 139.0), 11: (35.001, 139.001), 12: (35.002, 139.002),
               20: (35.0, 139.002), 21: (35.002, 139.0)},
    )
    result = _parse_overpass(data)
    assert result == [(35.001, 139.001)]


def test_道路が交わらなければ交差点なし():
    data = osm_response(
        ways={1: [10, 11], 2: [20, 21]},
        nodes={10: (35.0, 139.0), 11: (35.001, 139.0),
               20: (35.0, 139.001), 21: (35.001, 139.001)},
    )
    assert _parse_overpass(data) == []


def test_同じ道の中で同じ点が2回出ても交差点にしない():
    # 環状の道は最初と最後が同じ点になるが、1本の道なので交差点ではない
    data = osm_response(
        ways={1: [10, 11, 12, 10]},
        nodes={10: (35.0, 139.0), 11: (35.001, 139.0), 12: (35.001, 139.001)},
    )
    assert _parse_overpass(data) == []


def test_空や欠けた応答でも壊れない():
    assert _parse_overpass({}) == []
    assert _parse_overpass({"elements": []}) == []
    assert _parse_overpass("応答ではない") == []
    # 座標の無い点・nodes の無い道は無視される
    data = {"elements": [{"type": "node", "id": 1},
                         {"type": "way", "id": 2},
                         "変な要素"]}
    assert _parse_overpass(data) == []


def test_OsmProviderが応答から交差点を取り出す(monkeypatch):
    fake_response = osm_response(
        ways={1: [10, 11], 2: [20, 11]},
        nodes={10: (35.0, 139.0), 11: (35.0005, 139.0005), 20: (35.001, 139.0)},
    )
    called_urls = []

    def fake_fetch(url, timeout):
        called_urls.append(url)
        return fake_response

    monkeypatch.setattr(roadinfo, "_fetch_json", fake_fetch)
    provider = OsmProvider()
    result = provider.intersections_near(35.0, 139.0)
    assert result == [(35.0005, 139.0005)]
    assert "overpass" in called_urls[0]
    assert "35.000000" in called_urls[0]  # 現在地が問い合わせに入っている


def test_OsmProviderのおかしな設定は止まる():
    with pytest.raises(ValueError):
        OsmProvider(timeout=0)
    with pytest.raises(ValueError):
        OsmProvider(timeout=-1)
    provider = OsmProvider()
    with pytest.raises(ValueError):
        provider.intersections_near(999, 139.0)


def test_通信エラーはRoadWatcherが受け止める(monkeypatch):
    def fail_fetch(url, timeout):
        raise OSError("ネットワークなし")

    monkeypatch.setattr(roadinfo, "_fetch_json", fail_fetch)
    watcher = RoadWatcher(provider=OsmProvider())
    watcher.update(35.0, 139.0, 0.0)  # 止まらない
    assert watcher.state()["intersection_score"] == 0.0


# ---------- ランダムテスト ----------

def test_でたらめな移動でも状態は必ずまともな値():
    rng = random.Random(12)
    provider = FakeProvider(intersections=[(35.0005, 139.0005)])
    watcher = RoadWatcher(provider=provider)
    t = 0.0
    for _ in range(200):
        t += rng.uniform(0.1, 5.0)
        lat = 35.0 + rng.uniform(-0.001, 0.001)
        lon = 139.0 + rng.uniform(-0.001, 0.001)
        watcher.update(lat, lon, t)
        state = watcher.state()
        assert 0 <= state["intersection_score"] <= 1
        assert 0 <= state["turning_score"] <= 1
        assert state["speed"] is None or state["speed"] >= 0
