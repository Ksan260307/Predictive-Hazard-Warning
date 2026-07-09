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

import numpy as np

from app import imgproc

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
    return imgproc.resize(frame, max_width, new_h)


# カメラの動きの推定に最低限必要な「目印の多さ」(画素の割合)。
# 目印が少なすぎる(画面がのっぺりしている)時は推定をあきらめる
MIN_TEXTURE_DENSITY = 0.05

# ズレの推定を信じる「確からしさ」の下限。
# これより低い時 (映像がでたらめに変わった時など) は打ち消さない
MIN_SHIFT_STRENGTH = 0.2

# カメラの動きを打ち消した時に無視する画面のふちの幅 (割合)。
# ふちは合わせきれずブレが残りやすいため
EDGE_MARGIN = 0.05

# 画面のこれ以上の割合がいっぺんに動いていたら、「自分が動いた
# (カメラ全体のブレ)」の可能性を疑う。
# 室内で歩く・急に向きを変えるなど、手ブレ打ち消しが利かない場面での
# 過剰な反応を防ぐ (打ち消しを使う時だけ効く)
GLOBAL_MOTION_FRAC = 0.35

# 疑わしい時でも、変化のほとんど (この割合以上) が1つの塊に集まっていて、
# かつその塊が画面を覆いつくしていなければ、「目の前の大きな物」として残す。
# バスや直前の歩行者など、一番危険な物を取りこぼさないため
DOMINANT_BLOB_FRAC = 0.6
# 「目の前の物」とみなせる塊の大きさの上限 (画面全体を1として)。
# これを超えて画面ほぼ全部が変わっているのは、物ではなく自分の動き
DOMINANT_AREA_LIMIT = 0.8


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
        diff = np.abs(gray.astype(np.int16) - prev.astype(np.int16))
        mask = diff > self.threshold

        # 打ち消しをした場合、合わせきれない画面のふちは見ない
        if self.stabilize:
            mh = max(1, int(mask.shape[0] * EDGE_MARGIN))
            mw = max(1, int(mask.shape[1] * EDGE_MARGIN))
            mask[:mh, :] = False
            mask[-mh:, :] = False
            mask[:, :mw] = False
            mask[:, -mw:] = False

        mask = imgproc.dilate(mask, iterations=2)  # 白い場所を少しふくらませてつなげる

        boxes = imgproc.find_boxes(mask)

        # 画面の広い範囲がいっぺんに動いた時は「自分が動いた(カメラのブレ)」を疑う。
        # ただし、変化が1つの塊に集まっている時は「目の前の大きな物」なので残す。
        # 歩きながらの撮影で画面全体が流れるような時だけ、誤検出を捨てる
        if self.stabilize and float(mask.mean()) > GLOBAL_MOTION_FRAC:
            dominant = self._dominant_box(mask, boxes)
            if dominant is None:
                self._prev_gray = gray
                self._prev_things = []
                return []
            boxes = [dominant]

        # 白いかたまりを物として拾う
        img_h, img_w = gray.shape
        min_area = self.min_size * img_w * img_h

        things = []
        for x, y, w, h in boxes:
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
    def _dominant_box(mask, boxes):
        """変化のほとんどを占める「1つの大きな塊」があればその箱を返す。

        ・変化した画素の DOMINANT_BLOB_FRAC 以上が1つの塊に集まっていて、
        ・その塊が画面を覆いつくしていない (DOMINANT_AREA_LIMIT 以下)
        なら「目の前の大きな物」とみなす。どちらも満たさなければ None
        (= 画面のあちこちが変わった。自分が動いただけ) を返す。
        """
        total = float(mask.sum())
        if total <= 0 or not boxes:
            return None
        best = None
        best_pixels = 0.0
        for box in boxes:
            x, y, w, h = box
            pixels = float(mask[y:y + h, x:x + w].sum())
            if pixels > best_pixels:
                best_pixels = pixels
                best = box
        img_h, img_w = mask.shape
        x, y, w, h = best
        if best_pixels / total < DOMINANT_BLOB_FRAC:
            return None  # 変化が複数の場所に散らばっている
        if (w * h) / float(img_w * img_h) > DOMINANT_AREA_LIMIT:
            return None  # 画面ほぼ全部が1つの塊 = 物ではなく自分の動き
        return best

    @staticmethod
    def _align_to(prev, gray):
        """前の画像を、今の画像のカメラ位置に合わせて動かして返す。

        1. 画像に目印 (模様や境界) が十分あるかを確かめる
        2. 画面全体が何画素ズレたかを推定する (位相相関)
        3. そのぶんだけ前の画像を動かして、自分の移動を打ち消す

        目印が足りない・推定が確かでない時は、前の画像をそのまま返す。
        """
        if imgproc.texture_density(prev) < MIN_TEXTURE_DENSITY:
            return prev

        dx, dy, strength = imgproc.estimate_shift(prev, gray)
        if strength < MIN_SHIFT_STRENGTH:
            return prev
        if dx == 0 and dy == 0:
            return prev
        h, w = prev.shape
        # 画面の2割を超えるズレはカメラの動きとしては大きすぎるので信じない
        if abs(dx) > w * 0.2 or abs(dy) > h * 0.2:
            return prev
        return imgproc.shift_image(prev, dx, dy)

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
            gray = imgproc.to_gray(frame)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("画像はカラー(縦x横x3)か白黒(縦x横)にしてください")
        # ざらつきをならして、細かいノイズを拾いにくくする
        return imgproc.blur(gray)
