# 検出結果を時間方向にならすファイル (トラッカー)
#
# 検出器 (detector.py / objectdetect.py) は1フレームごとに独立して物を
# 見つけるため、光のちらつき等による「1フレームだけのノイズ」も混ざる。
# ここで数フレームの連続性を確かめ、本当にいる物だけを通す。
#
#   ・数フレーム連続で見えた物だけを「確定」として返す (ノイズ除去)
#   ・物ごとに番号(id)を振って追いかけ、動きの速さをならして返す
#   ・一瞬見失っても少しの間は「いるはず」として位置を進める (ちらつき防止)

import math

# 検出結果に必ず入っていなければならない項目
REQUIRED_KEYS = ("x", "y", "w", "h", "cx", "cy")


class ThingTracker:
    """検出された物を追いかけて、確かな物だけを通す係。

    使い方:
        tracker = ThingTracker()
        things = tracker.update(raw_things)   # 毎フレーム呼ぶ

    返ってくる things は検出結果と同じ形式に "id" が加わったもの。
    vx, vy は複数フレームでならした値に置き換えられる。
    label や score など検出器が付けた項目はそのまま引き継がれる。
    """

    def __init__(self, min_hits=3, max_missed=3, match_distance=0.15, smooth=0.6):
        """
        min_hits       : 何フレーム連続で見えたら「確定」とするか
        max_missed     : 何フレームまで見失っても追い続けるか
        match_distance : 前の物と「同じ物」とみなす距離 (画面の割合)
        smooth         : 速度のならし方 (1に近いほど最新の動きを重視)
        """
        if not isinstance(min_hits, int) or isinstance(min_hits, bool) or min_hits < 1:
            raise ValueError("確定までのフレーム数(min_hits)は1以上の整数にしてください")
        if not isinstance(max_missed, int) or isinstance(max_missed, bool) or max_missed < 0:
            raise ValueError("見失いの許容数(max_missed)は0以上の整数にしてください")
        if not 0 < match_distance <= 1:
            raise ValueError("対応づけの距離(match_distance)は0より大きく1以下にしてください")
        if not 0 < smooth <= 1:
            raise ValueError("ならしの強さ(smooth)は0より大きく1以下にしてください")

        self.min_hits = min_hits
        self.max_missed = max_missed
        self.match_distance = match_distance
        self.smooth = smooth
        self._tracks = []
        self._next_id = 1

    def update(self, things):
        """検出結果を1フレーム分取り込み、確定した物のリストを返す。"""
        if not isinstance(things, (list, tuple)):
            raise ValueError("検出結果(things)はリストで渡してください")
        for t in things:
            if not isinstance(t, dict):
                raise ValueError("検出結果の中身は辞書にしてください")
            for key in REQUIRED_KEYS:
                if key not in t:
                    raise ValueError("検出結果に " + key + " がありません")

        # 検出と既存トラックを、近い順に1対1で結びつける (貪欲法)
        known_count = len(self._tracks)
        pairs = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(things):
                dist = math.hypot(det["cx"] - track["cx"], det["cy"] - track["cy"])
                if dist < self.match_distance:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._absorb(self._tracks[ti], things[di])

        # 結びつかなかった検出 → 新しいトラックとして覚える
        for di, det in enumerate(things):
            if di not in matched_dets:
                self._tracks.append(self._new_track(det))

        # 結びつかなかった既存トラック → 見失いを数え、今の速さで位置を進める
        survivors = []
        for ti, track in enumerate(self._tracks):
            if ti in matched_tracks or ti >= known_count:
                survivors.append(track)  # 今回も見えた物 / 今回初めて見えた物
                continue
            track["missed"] += 1
            if track["missed"] > self.max_missed:
                continue  # 長く見えない物は忘れる
            for key, v_key in (("x", "vx"), ("cx", "vx"), ("y", "vy"), ("cy", "vy")):
                track[key] += track[v_key]
            survivors.append(track)
        self._tracks = survivors

        # 確定した (連続して見えている) 物だけを返す
        confirmed = []
        for track in self._tracks:
            if track["hits"] >= self.min_hits:
                confirmed.append(self._to_thing(track))
        return confirmed

    def reset(self):
        """追いかけていた物を全部忘れて、最初からやり直す。"""
        self._tracks = []

    # ---------------- 内部処理 ----------------

    def _new_track(self, det):
        track = {key: float(det[key]) for key in REQUIRED_KEYS}
        track["vx"] = float(det.get("vx", 0.0))
        track["vy"] = float(det.get("vy", 0.0))
        track["id"] = self._next_id
        track["hits"] = 1
        track["missed"] = 0
        self._copy_extras(det, track)
        self._next_id += 1
        return track

    def _absorb(self, track, det):
        """トラックに新しい検出を取り込む。速度は急に変えず、ならして更新する。"""
        inst_vx = det["cx"] - track["cx"]
        inst_vy = det["cy"] - track["cy"]
        # 見失っていた間の移動は複数フレーム分なので、1フレームあたりに直す
        frames = track["missed"] + 1
        inst_vx /= frames
        inst_vy /= frames

        track["vx"] = self.smooth * inst_vx + (1 - self.smooth) * track["vx"]
        track["vy"] = self.smooth * inst_vy + (1 - self.smooth) * track["vy"]
        for key in REQUIRED_KEYS:
            track[key] = float(det[key])
        track["hits"] += 1
        track["missed"] = 0
        self._copy_extras(det, track)

    @staticmethod
    def _copy_extras(det, track):
        """検出器が付けた追加の項目 (label, score など) を引き継ぐ。"""
        for key, value in det.items():
            if key not in REQUIRED_KEYS and key not in ("vx", "vy"):
                track[key] = value

    @staticmethod
    def _to_thing(track):
        thing = {key: track[key] for key in REQUIRED_KEYS}
        thing["vx"] = track["vx"]
        thing["vy"] = track["vy"]
        thing["id"] = track["id"]
        for key, value in track.items():
            if key not in thing and key not in ("hits", "missed"):
                thing[key] = value
        return thing
