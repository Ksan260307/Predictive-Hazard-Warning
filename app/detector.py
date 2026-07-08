# カメラの画像から「動いている物」を見つけるファイル
#
# 前の画像と今の画像を比べて、変わった場所 = 動いた物 として拾います。
#
# 移動しながら使う想定のため、まず「自分の移動によるカメラ全体の動き」を
# 推定して打ち消し、それでも残った動き = 本当に動いている物 として扱います。
#
# 見つけた物は、画面の大きさに関係なく使えるように
# 0〜1 の割合(画面の左上が0、右下が1)で返します。

import math

import cv2
import numpy as np

# 前の画像の物と「同じ物」とみなす距離 (画面の15%まで)
SAME_THING_DISTANCE = 0.15

def shrink_frame(frame, max_width=320):
    """分析用に画像を小さくして返す。表示用の元画像はそのまま使える。

    画像が大きいままだと分析が重くなるため、横幅を max_width までに抑える。
    横幅がすでに max_width 以下ならそのまま返す。
    """
    if frame is None:
        raise ValueError("画像がありません")
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
        raise ValueError("画像の形式が正しくありません")
    if not isinstance(max_width, int) or isinstance(max_width, bool) or max_width < 8:
        raise ValueError("縮小後の横幅(max_width)は8以上の整数にしてください")

    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = max(1, round(h * max_width / w))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


# カメラの動きの推定に最低限必要な目印の数。
# 目印が少なすぎる(画面がのっぺりしている)時は推定をあきらめる
MIN_ALIGN_POINTS = 40

# カメラの動きを打ち消した時に無視する画面のふちの幅 (割合)。
# ふちは合わせきれずブレが残りやすいため
EDGE_MARGIN = 0.05


class MovingThingFinder:
    """動いている物を見つける係。

    使い方:
        finder = MovingThingFinder(sensitivity=5, min_size=0.005)
        things = finder.find(frame)   # 毎フレーム呼ぶ

    返ってくる things は、物ごとの辞書のリスト:
        x, y   : 左上の位置 (0〜1)
        w, h   : 大きさ (0〜1)
        cx, cy : 真ん中の位置 (0〜1)
        vx, vy : 1フレームあたりの動き (0〜1の割合)
    """

    def __init__(self, sensitivity=5, min_size=0.005, stabilize=True):
        if not isinstance(sensitivity, int) or isinstance(sensitivity, bool):
            raise ValueError("感度(sensitivity)は1〜10の整数にしてください")
        if not 1 <= sensitivity <= 10:
            raise ValueError("感度(sensitivity)は1〜10の間にしてください")
        if not 0 < min_size < 1:
            raise ValueError("最小の大きさ(min_size)は0より大きく1より小さくしてください")

        # 感度が高いほど、小さな変化でも拾う(しきい値を下げる)
        self.threshold = 60 - sensitivity * 5
        self.min_size = float(min_size)
        self.stabilize = bool(stabilize)  # 自分の移動によるカメラの動きを打ち消すか
        self._prev_gray = None    # 前の画像(白黒)
        self._prev_things = []    # 前に見つけた物

    def find(self, frame):
        """画像1枚を受け取り、動いている物のリストを返す。"""
        gray = self._to_gray(frame)

        # 最初の1枚、または画像の大きさが変わった時は比べられないので空を返す
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._prev_things = []
            return []

        # 移動しながら使う場合、前の画像を「今のカメラ位置」に合わせてから比べる。
        # これで背景の見かけの動きが消え、本当に動いている物だけが残る
        prev = self._prev_gray
        if self.stabilize:
            prev = self._align_to(prev, gray)

        # 前の画像との差を取り、変わった場所を白にする
        diff = cv2.absdiff(gray, prev)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # 打ち消しをした場合、合わせきれない画面のふちは見ない
        if self.stabilize:
            mh = max(1, int(mask.shape[0] * EDGE_MARGIN))
            mw = max(1, int(mask.shape[1] * EDGE_MARGIN))
            mask[:mh, :] = 0
            mask[-mh:, :] = 0
            mask[:, :mw] = 0
            mask[:, -mw:] = 0

        mask = cv2.dilate(mask, None, iterations=2)  # 白い場所を少しふくらませてつなげる

        # 白いかたまりを物として拾う
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w = gray.shape
        min_area = self.min_size * img_w * img_h

        things = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue  # 小さすぎる物はノイズとして捨てる
            things.append({
                "x": x / img_w, "y": y / img_h,
                "w": w / img_w, "h": h / img_h,
                "cx": (x + w / 2) / img_w, "cy": (y + h / 2) / img_h,
                "vx": 0.0, "vy": 0.0,
            })

        # 前の画像の物と近い物を「同じ物」とみなして、動きの速さを計算する
        for t in things:
            nearest = None
            nearest_dist = SAME_THING_DISTANCE
            for p in self._prev_things:
                dist = math.hypot(t["cx"] - p["cx"], t["cy"] - p["cy"])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = p
            if nearest is not None:
                t["vx"] = t["cx"] - nearest["cx"]
                t["vy"] = t["cy"] - nearest["cy"]

        self._prev_gray = gray
        self._prev_things = things
        return things

    def reset(self):
        """覚えている画像を忘れて、最初からやり直す。"""
        self._prev_gray = None
        self._prev_things = []

    @staticmethod
    def _align_to(prev, gray):
        """前の画像を、今の画像のカメラ位置に合わせて動かして返す。

        1. 前の画像から追いやすい目印(角など)を拾う
        2. その目印が今の画像でどこへ動いたかを追う
        3. 画面全体としての動き(=自分の移動)を計算して打ち消す

        目印が足りない・計算できない時は、前の画像をそのまま返す。
        """
        prev_points = cv2.goodFeaturesToTrack(
            prev, maxCorners=200, qualityLevel=0.01, minDistance=8,
        )
        if prev_points is None or len(prev_points) < MIN_ALIGN_POINTS:
            return prev

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, prev_points, None)
        if next_points is None or status is None:
            return prev
        good = status.flatten() == 1
        if good.sum() < MIN_ALIGN_POINTS:
            return prev

        # 画面全体の動きを1つの平行移動+回転+拡大として推定する
        # (はぐれた目印は自動で無視される)
        matrix, _ = cv2.estimateAffinePartial2D(prev_points[good], next_points[good])
        if matrix is None:
            return prev
        h, w = prev.shape
        return cv2.warpAffine(prev, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _to_gray(frame):
        """画像を調べて、白黒画像にして返す。おかしな画像なら止める。"""
        if frame is None:
            raise ValueError("カメラの画像がありません")
        if not isinstance(frame, np.ndarray):
            raise ValueError("画像の形式が正しくありません")
        if frame.size == 0:
            raise ValueError("画像が空です")
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("画像はカラー(縦x横x3)か白黒(縦x横)にしてください")
        # ざらつきをならして、細かいノイズを拾いにくくする
        return cv2.GaussianBlur(gray, (5, 5), 0)
