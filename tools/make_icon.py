# アプリのアイコンとスプラッシュ画像を作るスクリプト
# 実行: python tools/make_icon.py
# data/icon.png (512x512) と data/presplash.png (1280x720) を作る。

import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BG = (28, 20, 15)          # 濃紺 (B, G, R)
ROAD = (70, 56, 44)
LANE = (150, 140, 128)
AMBER = (25, 170, 235)     # 警告の黄
WHITE = (245, 245, 245)
GREEN = (90, 165, 45)


def rounded_mask(size, radius):
    """角丸四角の型を作る。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (size - radius, size), 255, -1)
    cv2.rectangle(mask, (0, radius), (size, size - radius), 255, -1)
    for cx in (radius, size - radius):
        for cy in (radius, size - radius):
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def draw_motif(canvas, cx, cy, scale):
    """道路と警告マークの絵柄を描く。cx, cy は中心、scale は大きさ。"""
    s = scale
    # 遠くへ伸びる道路 (台形)
    road = np.array([
        (cx - int(1.7 * s), cy + int(1.9 * s)),
        (cx + int(1.7 * s), cy + int(1.9 * s)),
        (cx + int(0.45 * s), cy - int(0.7 * s)),
        (cx - int(0.45 * s), cy - int(0.7 * s)),
    ])
    cv2.fillPoly(canvas, [road], ROAD)
    # 中央線
    for i in range(4):
        t0 = i / 4.0
        t1 = t0 + 0.13
        y0 = int(cy + 1.9 * s - 2.6 * s * t0)
        y1 = int(cy + 1.9 * s - 2.6 * s * t1)
        w0 = max(2, int(0.06 * s * (1 - 0.75 * t0)))
        cv2.rectangle(canvas, (cx - w0, y1), (cx + w0, y0), LANE, -1)
    # 先読みの波 (未来の分布のイメージ)
    for i, r in enumerate((1.05, 1.35, 1.65)):
        cv2.ellipse(canvas, (cx, cy - int(0.7 * s)), (int(r * s * 0.9), int(r * s * 0.45)),
                    0, 200, 340, GREEN if i == 0 else LANE, max(2, int(0.05 * s)))
    # 警告の三角
    tri_cy = cy + int(0.55 * s)
    t = int(1.05 * s)
    tri = np.array([(cx, tri_cy - t), (cx - t, tri_cy + t), (cx + t, tri_cy + t)])
    cv2.fillPoly(canvas, [tri], AMBER)
    cv2.polylines(canvas, [tri], True, WHITE, max(3, int(0.09 * s)))
    # ビックリマーク
    bw = max(4, int(0.16 * s))
    cv2.rectangle(canvas, (cx - bw // 2, tri_cy - int(0.45 * t)),
                  (cx + bw // 2, tri_cy + int(0.25 * t)), (40, 40, 40), -1)
    cv2.circle(canvas, (cx, tri_cy + int(0.6 * t)), bw // 2 + 1, (40, 40, 40), -1)


def make_icon(size=512):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = BG
    draw_motif(canvas, size // 2, size // 2 - size // 16, size // 5)
    # 角丸 + 透明背景
    mask = rounded_mask(size, size // 6)
    icon = np.dstack([canvas, mask])
    return icon


def make_presplash(width=1280, height=720):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BG
    draw_motif(canvas, width // 2, height // 2, height // 5)
    return canvas


def save_png(path, image):
    """日本語のフォルダ名でも保存できるPNG書き込み。"""
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNGへの変換に失敗しました: " + path)
    with open(path, "wb") as f:
        f.write(data.tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    save_png(os.path.join(OUT_DIR, "icon.png"), make_icon())
    save_png(os.path.join(OUT_DIR, "presplash.png"), make_presplash())
    print("OK:", OUT_DIR)


if __name__ == "__main__":
    main()
