# 位置情報(GPS)と地図から「道路の状況」を読み取るファイル
#
# ・GpsTrack    : GPSの履歴から自分の速度・向き・旋回を計算する
# ・RoadWatcher : 地図情報と合わせて「交差点への接近」などを判定する
# ・OsmProvider : OpenStreetMap (Overpass API) で近くの交差点を探す。
#                 実際の交差点ノードを取得するのでAPIキー不要かつ正確
#
# 地図が使えない環境(圏外など)でも、GPSの履歴だけで動く。
# 何も情報が無い時は NEUTRAL_STATE(影響なし)を返す。

import json
import math
import urllib.parse
import urllib.request

EARTH_RADIUS_M = 6371000.0

# 交差点の影響が届く距離(m)。この距離で影響ゼロ、近づくほど強くなる
INTERSECTION_REACH_M = 100.0
# 「旋回中」とみなす向きの変化 (度/秒)
FULL_TURN_RATE = 30.0

# 位置情報が無い時の状態
NEUTRAL_STATE = {
    "speed": None,              # 速度 m/s (None = 不明)
    "turn_rate": 0.0,           # 向きの変化 度/秒
    "turning_score": 0.0,       # 旋回の度合い (0〜1)
    "intersection_score": 0.0,  # 交差点への接近度 (0〜1)
    "intersection_distance": None,  # いちばん近い交差点までの距離 m
}


def distance_m(lat1, lon1, lat2, lon2):
    """2地点間の距離(m)。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def heading_deg(lat1, lon1, lat2, lon2):
    """地点1から地点2へ向かう方角(北=0度、時計回り)。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(x, y)) % 360


def _check_point(lat, lon, t):
    if not isinstance(lat, (int, float)) or isinstance(lat, bool) or not -90 <= lat <= 90:
        raise ValueError("緯度(lat)は-90〜90の数にしてください")
    if not isinstance(lon, (int, float)) or isinstance(lon, bool) or not -180 <= lon <= 180:
        raise ValueError("経度(lon)は-180〜180の数にしてください")
    if not isinstance(t, (int, float)) or isinstance(t, bool):
        raise ValueError("時刻(t)は数にしてください")


class GpsTrack:
    """GPSの履歴。速度・向き・旋回を計算する。"""

    def __init__(self, keep=10):
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 3:
            raise ValueError("履歴の数(keep)は3以上の整数にしてください")
        self.keep = keep
        self.points = []  # [(lat, lon, t), ...]

    def add(self, lat, lon, t):
        """位置を1つ記録する。記録したら True を返す。

        GPSは同じ時刻の通知を重複して寄こすことがあるので、
        時刻が進んでいない記録は黙って無視する (False を返す)。
        """
        _check_point(lat, lon, t)
        if self.points and t <= self.points[-1][2]:
            return False
        self.points.append((lat, lon, t))
        self.points = self.points[-self.keep:]
        return True

    def speed(self):
        """現在の速度 (m/s)。履歴が足りなければ None。"""
        if len(self.points) < 2:
            return None
        (lat1, lon1, t1), (lat2, lon2, t2) = self.points[-2], self.points[-1]
        dt = t2 - t1
        if dt <= 0:
            return None
        return distance_m(lat1, lon1, lat2, lon2) / dt

    def turn_rate(self):
        """向きの変化の速さ (度/秒)。履歴が足りなければ 0。"""
        if len(self.points) < 3:
            return 0.0
        p1, p2, p3 = self.points[-3], self.points[-2], self.points[-1]
        # ほぼ動いていない時は向きが定まらないので旋回なしとする
        if distance_m(p1[0], p1[1], p2[0], p2[1]) < 0.5:
            return 0.0
        if distance_m(p2[0], p2[1], p3[0], p3[1]) < 0.5:
            return 0.0
        h1 = heading_deg(p1[0], p1[1], p2[0], p2[1])
        h2 = heading_deg(p2[0], p2[1], p3[0], p3[1])
        change = (h2 - h1 + 180) % 360 - 180  # -180〜180 に直す
        dt = p3[2] - p1[2]
        if dt <= 0:
            return 0.0
        return change / dt


class RoadWatcher:
    """GPSと地図を合わせて、道路の状況(交差点接近・旋回)を判定する係。

    provider には「近くの交差点を返すもの」を渡す(無ければ None)。
    provider が失敗しても全体は止まらず、地図なしの判定を続ける。
    """

    def __init__(self, provider=None, refresh_m=30.0):
        self.provider = provider
        self.refresh_m = refresh_m  # 前回の問い合わせ地点からこれだけ動いたら再問い合わせ
        self.track = GpsTrack()
        self._intersections = []
        self._last_query_point = None

    def update(self, lat, lon, t):
        """新しい位置を1つ取り込む。"""
        if not self.track.add(lat, lon, t):
            return  # 時刻が進んでいない重複の通知は無視する
        if self.provider is None:
            return
        if self._last_query_point is not None:
            moved = distance_m(self._last_query_point[0], self._last_query_point[1], lat, lon)
            if moved < self.refresh_m:
                return
        try:
            self._intersections = list(self.provider.intersections_near(lat, lon))
            self._last_query_point = (lat, lon)
        except Exception:
            # 圏外・キー切れなどで地図が読めなくても、アプリは止めない
            pass

    def state(self):
        """いまの道路状況を返す。情報が無い項目は NEUTRAL_STATE と同じ値。"""
        result = dict(NEUTRAL_STATE)
        if not self.track.points:
            return result

        result["speed"] = self.track.speed()
        rate = self.track.turn_rate()
        result["turn_rate"] = rate
        result["turning_score"] = min(1.0, abs(rate) / FULL_TURN_RATE)

        if self._intersections:
            lat, lon, _ = self.track.points[-1]
            nearest = min(
                distance_m(lat, lon, ilat, ilon)
                for ilat, ilon in self._intersections
            )
            result["intersection_distance"] = nearest
            result["intersection_score"] = max(0.0, 1.0 - nearest / INTERSECTION_REACH_M)
        return result


# ---------------------------------------------------------------
# OpenStreetMap (Overpass API) 連携
# ---------------------------------------------------------------

def _fetch_json(url, timeout):
    """URLからJSONを取ってくる(テストで差し替えられるように分けてある)。"""
    with urllib.request.urlopen(url, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _parse_overpass(data):
    """Overpass API の応答から交差点を取り出す。

    OpenStreetMapでは道路は「way (点の列)」で表され、
    2本以上の道路が同じ点を共有していれば、そこは交差点。

    注意: 1本の道が途中で区切られて2つの way になっている場所も
    共有点として数えてしまうことがある (実際より少し多めに出る)。
    多めに警戒する分には安全側なので、そのまま使う。
    """
    elements = data.get("elements", []) if isinstance(data, dict) else []
    node_pos = {}    # 点の番号 → (緯度, 経度)
    node_count = {}  # 点の番号 → その点を通る道路の数
    for e in elements:
        if not isinstance(e, dict):
            continue
        if e.get("type") == "node" and "lat" in e and "lon" in e:
            node_pos[e.get("id")] = (e["lat"], e["lon"])
        elif e.get("type") == "way":
            nodes = e.get("nodes") or []
            for nid in set(nodes):
                node_count[nid] = node_count.get(nid, 0) + 1

    return [node_pos[nid] for nid, count in node_count.items()
            if count >= 2 and nid in node_pos]


class OsmProvider:
    """OpenStreetMap (Overpass API) で近くの交差点を探す係。

    実際の道路データから「2本以上の道路が交わる点」を取得する。
    APIキーは不要。車が通る道だけを対象にする (歩道・小道は除く)。
    """

    URL = "https://overpass-api.de/api/interpreter"
    # 対象にする道路の種類 (車の通行がある道)
    HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary"
                "|unclassified|residential|living_street|service")

    def __init__(self, timeout=5.0):
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("タイムアウト(timeout)は0より大きい数にしてください")
        self.timeout = float(timeout)

    def intersections_near(self, lat, lon, radius_m=60.0):
        _check_point(lat, lon, 0)
        query = (
            '[out:json][timeout:{t}];'
            'way(around:{r:.0f},{lat:.6f},{lon:.6f})["highway"~"^({h})$"];'
            'out body;>;out skel qt;'
        ).format(t=max(1, int(self.timeout)), r=radius_m,
                 lat=lat, lon=lon, h=self.HIGHWAYS)
        url = self.URL + "?" + urllib.parse.urlencode({"data": query})
        data = _fetch_json(url, self.timeout)
        return _parse_overpass(data)
