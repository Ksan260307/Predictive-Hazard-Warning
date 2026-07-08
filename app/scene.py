# 画像から「道路の見通し」を読み取るファイル
#
# 動いている物だけでなく、風景そのものから危険の芽を探す。
#   ・死角の候補: 建物の角・塀・駐車車両などの「縦の遮蔽境界」。
#     その陰から人や車が飛び出してくる可能性がある。
#   ・見通し (openness): 進行方向がどれだけ開けているか。
#
# AI物体認識が使える時は、「止まっている車」の陰も死角として拾える
# (vehicle_blind_spots)。こちらは実際に車だと分かっている分、確かな根拠になる。
#
# ここで見つけた死角は、watcher で「見えない飛び出しの未来」として
# 未来の分布に混ぜ込まれる。

import math

import numpy as np

from app import imgproc

# 死角とみなす縦エッジの強さ(正規化後)の下限
SPOT_THRESHOLD = 0.35
# 死角を探す範囲: 進行方向の左右(画面の割合)。中央の通り道と画面端は除く
SEARCH_BANDS = ((0.10, 0.45), (0.55, 0.90))
# 死角のおおよその路面位置(画面の縦方向)。手前の風景として固定値を使う
SPOT_ROAD_Y = 0.6

# 陰に人が隠れうる「遮蔽物」のクラス (AI物体認識のラベル)
OCCLUDER_LABELS = ("car", "truck", "bus")
# これより遅い物を「止まっている」とみなす (1フレームあたりの移動の割合)
STILL_SPEED = 0.005
# 遮蔽物として扱う最小の大きさ (画面全体を1とした面積)。遠く小さい車は無視
MIN_OCCLUDER_AREA = 0.02
# 近すぎる死角候補どうしをまとめる距離 (画面の割合)
MERGE_GAP = 0.08


def vehicle_blind_spots(things):
    """検出された「止まっている車」の陰を、死角の候補として返す。

    駐車車両の陰は人が飛び出しやすい場所。風景の縦境界と違い、
    実際に車だと分かっている物から作るので、模様や看板を死角と間違えない。

    things : tracker を通った物のリスト。label の無い物 (動き検出) は無視する。
    返り値: SceneReader.read の blind_spots と同じ形式のリスト。
    """
    spots = []
    for t in things:
        if t.get("label") not in OCCLUDER_LABELS:
            continue
        if math.hypot(t.get("vx", 0.0), t.get("vy", 0.0)) > STILL_SPEED:
            continue  # 走っている車の陰は「死角」ではなく車そのものを警戒する
        area = t["w"] * t["h"]
        if area < MIN_OCCLUDER_AREA:
            continue
        # 進路の中央に近い側の端が「飛び出し口」になる
        if t["cx"] < 0.5:
            x = t["x"] + t["w"]
        else:
            x = t["x"]
        spots.append({
            "x": min(1.0, max(0.0, x)),
            "y": min(0.95, t["y"] + t["h"]),  # 車の足元 (路面との接点)
            # 大きい(近い)車ほど陰も大きく、危険も強い
            "strength": min(1.0, 0.4 + 2.0 * area),
            "source": "vehicle",
        })
    return spots


def merge_blind_spots(spots, min_gap=MERGE_GAP):
    """近すぎる死角候補をまとめて、強い方を残す。

    風景の縦境界と車の端が同じ場所を指している時に、
    同じ死角を二重に数えないようにする。
    """
    if not 0 < min_gap <= 1:
        raise ValueError("まとめる距離(min_gap)は0より大きく1以下にしてください")
    merged = []
    for spot in sorted(spots, key=lambda s: -s["strength"]):
        if all(abs(spot["x"] - kept["x"]) >= min_gap for kept in merged):
            merged.append(spot)
    return merged


class SceneReader:
    """風景から死角候補と見通しを読み取る係。

    使い方:
        reader = SceneReader()
        scene = reader.read(frame)
        scene["blind_spots"] : [{"x", "y", "strength"}, ...] 死角の候補
        scene["blind_risk"]  : いちばん強い死角の強さ (0〜1)
        scene["openness"]    : 進行方向の見通しの良さ (0〜1)
    """

    def read(self, frame):
        gray = self._to_gray(frame)
        h, w = gray.shape

        # 画面の下側(手前の風景)だけを見る。上側は空や遠景なので使わない
        lower = gray[int(h * 0.35):, :]
        if lower.size == 0:
            lower = gray

        # 縦方向の境界(壁の角・塀の端など)の強さを列ごとに集計する
        sobel = imgproc.sobel_x(lower)
        column_edge = np.abs(sobel).mean(axis=0) / 255.0
        if len(column_edge) >= 5:
            kernel = np.ones(5) / 5.0
            column_edge = np.convolve(column_edge, kernel, mode="same")

        # 左右の帯それぞれで、いちばん強い縦境界を死角候補として拾う
        blind_spots = []
        for band_lo, band_hi in SEARCH_BANDS:
            lo, hi = int(w * band_lo), int(w * band_hi)
            segment = column_edge[lo:hi]
            if segment.size == 0:
                continue
            peak_index = int(np.argmax(segment))
            strength = min(1.0, float(segment[peak_index]) / 2.0)
            if strength >= SPOT_THRESHOLD:
                blind_spots.append({
                    "x": (lo + peak_index) / w,
                    "y": SPOT_ROAD_Y,
                    "strength": strength,
                })

        blind_risk = max((s["strength"] for s in blind_spots), default=0.0)

        # 見通し: 進行方向の中央にごちゃごちゃした境界が少ないほど開けている
        center = column_edge[int(w * 0.35):int(w * 0.65)]
        if center.size > 0:
            clutter = min(1.0, float(center.mean()) / 1.5)
        else:
            clutter = 0.0
        openness = 1.0 - clutter

        return {
            "blind_spots": blind_spots,
            "blind_risk": blind_risk,
            "openness": openness,
        }

    @staticmethod
    def _to_gray(frame):
        """画像を確認して白黒に直す。不正な画像はエラーにする。"""
        if frame is None:
            raise ValueError("画像がありません")
        if not isinstance(frame, np.ndarray):
            raise ValueError("画像の形式が正しくありません")
        if frame.size == 0:
            raise ValueError("画像が空です")
        if frame.ndim == 3 and frame.shape[2] == 3:
            return imgproc.to_gray(frame)
        if frame.ndim == 2:
            return frame
        raise ValueError("画像はカラー(縦x横x3)か白黒(縦x横)にしてください")
