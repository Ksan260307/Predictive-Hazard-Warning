# numpy だけでできた画像処理の道具箱
#
# 以前は OpenCV (cv2) を使っていたが、Android 用のビルドで OpenCV を
# 入れるのがとても大変なため、必要な処理だけを numpy で作り直した。
#
# 置き場所ごとの対応:
#   ・白黒変換 / ぼかし / 縮小       → detector.py, scene.py が使う
#   ・差分 / 膨張 / かたまり探し     → detector.py が使う
#   ・縦の境界 (Sobel)               → scene.py が使う
#   ・画面全体のズレの推定 (位置合わせ) → detector.py の手ブレ打ち消し
#   ・線・枠・三角の描画             → hud.py が使う
#   ・PNG の書き出し / 読み戻し      → snapshot.py が使う
#
# 画像は cv2 と同じ「上から下へ・色は B,G,R の順」の numpy 配列で扱う。

import math
import struct
import zlib

import numpy as np

# ガウスぼかしの重み (5画素分)。足すと1になる
_BLUR_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


# ---------------------------------------------------------------
# 変換 (白黒・ぼかし・縮小)
# ---------------------------------------------------------------

def to_gray(frame):
    """カラー画像 (B,G,R) を白黒にして返す。白黒画像はそのまま返す。"""
    if frame.ndim == 2:
        return frame
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def blur(gray):
    """5x5のぼかしをかけて、ざらつき (ノイズ) をならす。"""
    a = gray.astype(np.float32)
    # 縦方向 → 横方向の順に、重みをつけて混ぜる (分離フィルタ)
    p = np.pad(a, ((2, 2), (0, 0)), mode="edge")
    a = sum(_BLUR_KERNEL[i] * p[i:i + gray.shape[0], :] for i in range(5))
    p = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    a = sum(_BLUR_KERNEL[i] * p[:, i:i + gray.shape[1]] for i in range(5))
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def resize(frame, new_w, new_h):
    """画像を指定の大きさにする。縮める時も伸ばす時も使える。"""
    h, w = frame.shape[:2]
    if (new_w, new_h) == (w, h):
        return frame

    # ちょうど割り切れる縮小なら、ブロックごとの平均 (きれいで速い)
    if new_w <= w and new_h <= h and w % new_w == 0 and h % new_h == 0:
        fh, fw = h // new_h, w // new_w
        if frame.ndim == 3:
            out = frame.reshape(new_h, fh, new_w, fw, frame.shape[2]).mean(axis=(1, 3))
        else:
            out = frame.reshape(new_h, fh, new_w, fw).mean(axis=(1, 3))
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    # それ以外は近い4画素を混ぜる (バイリニア補間)
    ys = np.clip((np.arange(new_h) + 0.5) * h / new_h - 0.5, 0, h - 1)
    xs = np.clip((np.arange(new_w) + 0.5) * w / new_w - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)

    a = frame.astype(np.float32)
    if frame.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    out = top * (1 - wy) + bottom * wy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------
# 動き検出のための処理 (差分のかたまり探し)
# ---------------------------------------------------------------

def dilate(mask, iterations=1):
    """白い場所を1画素ずつふくらませる (となり8方向)。"""
    m = mask.astype(bool)
    for _ in range(iterations):
        p = np.pad(m, 1, mode="constant")
        m = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
             | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
             | p[1:-1, 1:-1])
    return m


def find_boxes(mask):
    """白いかたまりごとに、外接する四角 (x, y, w, h) を返す。

    行ごとに白の区間 (run) を拾い、上の行の区間とつながっていれば
    同じかたまりとしてまとめる (ななめのつながりも同じとみなす)。
    """
    h, w = mask.shape
    parent = []  # かたまり番号の合流表 (union-find)

    def find_root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    all_runs = []   # (行, 開始, 終了, かたまり番号)
    prev_runs = []  # 1つ上の行の区間

    for y in range(h):
        row = mask[y]
        d = np.diff(row.astype(np.int8))
        starts = list(np.flatnonzero(d == 1) + 1)
        ends = list(np.flatnonzero(d == -1) + 1)
        if row[0]:
            starts.insert(0, 0)
        if row[-1]:
            ends.append(w)

        runs = []
        for s, e in zip(starts, ends):
            label = None
            for ps, pe, pl in prev_runs:
                # ななめも含めて触れていれば同じかたまり
                if ps <= e and pe >= s:
                    root = find_root(pl)
                    if label is None:
                        label = root
                    elif root != label:
                        parent[root] = label  # 2つのかたまりを合流させる
            if label is None:
                label = len(parent)
                parent.append(label)
            runs.append((s, e, label))
        all_runs.extend((y, s, e, label) for s, e, label in runs)
        prev_runs = runs

    # かたまりごとに、いちばん外側の位置を集める
    boxes = {}
    for y, s, e, label in all_runs:
        root = find_root(label)
        if root not in boxes:
            boxes[root] = [s, y, e - 1, y]
        else:
            box = boxes[root]
            box[0] = min(box[0], s)
            box[2] = max(box[2], e - 1)
            box[3] = y  # 行は上から見ているので、最後に見た行が下端
    return [(x1, y1, x2 - x1 + 1, y2 - y1 + 1)
            for x1, y1, x2, y2 in boxes.values()]


# ---------------------------------------------------------------
# 風景の分析 (縦の境界)
# ---------------------------------------------------------------

def sobel_x(gray):
    """横方向の明るさの変わり目 (縦の境界) の強さを返す。"""
    a = gray.astype(np.float32)
    p = np.pad(a, 1, mode="edge")
    # 縦方向をならしてから、横方向の差を取る (Sobelフィルタと同じ)
    smooth = p[:-2, :] + 2.0 * p[1:-1, :] + p[2:, :]
    return smooth[:, 2:] - smooth[:, :-2]


# ---------------------------------------------------------------
# カメラの動きの打ち消し (画面全体のズレの推定)
# ---------------------------------------------------------------

def texture_density(gray, threshold=20):
    """画像の「目印の多さ」を0〜1で返す。

    となりとの明るさの差が大きい画素の割合。のっぺりした画像 (壁や
    真っ黒な背景) では低くなり、風景の写った画像では高くなる。
    """
    a = gray.astype(np.int16)
    gx = np.abs(a[:, 1:] - a[:, :-1]) > threshold
    gy = np.abs(a[1:, :] - a[:-1, :]) > threshold
    return float(max(gx.mean(), gy.mean()))


def estimate_shift(prev, gray):
    """2枚の画像の間で、画面全体が何画素ズレたかを推定する。

    それぞれの画像の波の成分を比べる方法 (位相相関) を使う。
    返り値: (右へのズレ, 下へのズレ, 確からしさ0〜1)
    """
    h, w = prev.shape
    f1 = np.fft.rfft2(prev.astype(np.float32))
    f2 = np.fft.rfft2(gray.astype(np.float32))
    cross = f2 * np.conj(f1)
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, 1e-9)
    response = np.fft.irfft2(cross, s=(h, w))

    peak = np.unravel_index(int(np.argmax(response)), response.shape)
    strength = float(response[peak])
    dy, dx = int(peak[0]), int(peak[1])
    # 後ろ半分は「逆向きのズレ」を表す
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return dx, dy, min(1.0, max(0.0, strength))


def shift_image(img, dx, dy):
    """画像を右へdx・下へdy画素だけ動かして返す。はみ出た所は端の色で埋める。"""
    h, w = img.shape[:2]
    ys = np.clip(np.arange(h) - dy, 0, h - 1)
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    return img[ys][:, xs]


# ---------------------------------------------------------------
# 描画 (HUD用)
# ---------------------------------------------------------------

def _clip_box(img, x1, y1, x2, y2):
    """四角の座標を画像の中に収める。何も残らなければ None。"""
    h, w = img.shape[:2]
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    if x1 > x2 or y1 > y2:
        return None
    return x1, y1, x2, y2


def fill_rect(img, x1, y1, x2, y2, color):
    """塗りつぶした四角を描く。"""
    box = _clip_box(img, x1, y1, x2, y2)
    if box is None:
        return
    x1, y1, x2, y2 = box
    img[y1:y2 + 1, x1:x2 + 1] = color


def draw_rect(img, x1, y1, x2, y2, color, thickness=1):
    """枠だけの四角を描く。"""
    t = max(1, int(thickness))
    fill_rect(img, x1, y1, x2, y1 + t - 1, color)          # 上
    fill_rect(img, x1, y2 - t + 1, x2, y2, color)          # 下
    fill_rect(img, x1, y1, x1 + t - 1, y2, color)          # 左
    fill_rect(img, x2 - t + 1, y1, x2, y2, color)          # 右


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    """半透明の四角を重ねる (網かけ)。"""
    box = _clip_box(img, x1, y1, x2, y2)
    if box is None:
        return
    x1, y1, x2, y2 = box
    region = img[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    mixed = region * (1.0 - alpha) + tint * alpha
    img[y1:y2 + 1, x1:x2 + 1] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def draw_line(img, p1, p2, color, thickness=1):
    """2点を結ぶ線を描く。画面の外にはみ出た部分は描かない。"""
    h, w = img.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.rint(np.linspace(x1, x2, steps)).astype(int)
    ys = np.rint(np.linspace(y1, y2, steps)).astype(int)
    t = max(1, int(thickness))
    offsets = range(-(t // 2), t - t // 2)
    for oy in offsets:
        for ox in offsets:
            xk = xs + ox
            yk = ys + oy
            ok = (xk >= 0) & (xk < w) & (yk >= 0) & (yk < h)
            img[yk[ok], xk[ok]] = color


def draw_arrow(img, p1, p2, color, thickness=1, tip_length=0.35):
    """矢印を描く (線 + 先端のかえし2本)。"""
    draw_line(img, p1, p2, color, thickness)
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    tip = tip_length * length
    angle = math.atan2(dy, dx)
    for turn in (math.pi / 6, -math.pi / 6):
        end = (p2[0] + tip * math.cos(angle + turn),
               p2[1] + tip * math.sin(angle + turn))
        draw_line(img, p2, end, color, thickness)


def fill_triangle(img, pts, color):
    """3点を結ぶ三角を塗りつぶす。"""
    h, w = img.shape[:2]
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x_lo, x_hi = max(0, int(min(xs))), min(w - 1, int(math.ceil(max(xs))))
    y_lo, y_hi = max(0, int(min(ys))), min(h - 1, int(math.ceil(max(ys))))
    if x_lo > x_hi or y_lo > y_hi:
        return
    yy, xx = np.mgrid[y_lo:y_hi + 1, x_lo:x_hi + 1]

    def side(ax, ay, bx, by):
        return (bx - ax) * (yy - ay) - (by - ay) * (xx - ax)

    d1 = side(xs[0], ys[0], xs[1], ys[1])
    d2 = side(xs[1], ys[1], xs[2], ys[2])
    d3 = side(xs[2], ys[2], xs[0], ys[0])
    inside = (((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
              | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
    region = img[y_lo:y_hi + 1, x_lo:x_hi + 1]
    region[inside] = color


# ---------------------------------------------------------------
# PNG の書き出しと読み戻し
# ---------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data)))


def encode_png(frame):
    """画像をPNG形式のバイト列にする。カラー (B,G,R) と白黒に対応。"""
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("画像の形式が正しくありません")
    if frame.ndim == 3 and frame.shape[2] == 3:
        pixels = frame[:, :, ::-1]  # PNGは R,G,B の順なので入れ替える
        color_type = 2
    elif frame.ndim == 2:
        pixels = frame
        color_type = 0
    else:
        raise ValueError("画像はカラー(縦x横x3)か白黒(縦x横)にしてください")
    if frame.size == 0:
        raise ValueError("画像が空です")

    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    h, w = pixels.shape[:2]
    # 各行の先頭に「フィルタなし(0)」の1バイトをつける
    rows = pixels.reshape(h, -1)
    raw = np.zeros((h, rows.shape[1] + 1), dtype=np.uint8)
    raw[:, 1:] = rows

    header = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (_PNG_SIGNATURE
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + _png_chunk(b"IEND", b""))


def decode_png(data):
    """encode_png で作ったPNGを画像 (numpy配列) に読み戻す。

    自前で書き出した形式 (8ビット・フィルタなし) だけに対応する。
    """
    data = bytes(data)
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("PNGではありません")

    pos = len(_PNG_SIGNATURE)
    width = height = color_type = None
    body = b""
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", chunk[:10])
            if depth != 8 or color_type not in (0, 2):
                raise ValueError("対応していないPNGです")
        elif tag == b"IDAT":
            body += chunk
        elif tag == b"IEND":
            break
    if width is None:
        raise ValueError("PNGが壊れています")

    channels = 3 if color_type == 2 else 1
    raw = np.frombuffer(zlib.decompress(body), dtype=np.uint8)
    raw = raw.reshape(height, width * channels + 1)
    if np.any(raw[:, 0] != 0):
        raise ValueError("対応していないPNGです (フィルタつき)")
    pixels = raw[:, 1:]
    if channels == 3:
        return pixels.reshape(height, width, 3)[:, :, ::-1].copy()  # R,G,B → B,G,R
    return pixels.reshape(height, width).copy()
