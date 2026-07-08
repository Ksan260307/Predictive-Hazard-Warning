# 分析結果をカメラ映像に描き込むファイル (HUD = 画面上の計器表示)
#
# 映像の上に以下を重ねて描く:
#   ・危険域の網かけ (画面下の衝突判定エリア。警告レベルで色が変わる)
#   ・検出した物体の枠と進行方向の矢印
#   ・死角候補のマーカー (縦線と三角)
#   ・画面上部の危険度バー
#
# 文字は描かない (映像上の文字は運転中に読めないため、言葉はUI側に出す)。

import cv2
import numpy as np

from app.risk import CENTER_LEFT, CENTER_RIGHT

# レベルごとの表示色 (OpenCVなので B, G, R の順)
LEVEL_COLORS = {
    0: (90, 165, 45),    # 緑
    1: (25, 175, 235),   # 黄
    2: (45, 45, 220),    # 赤
}
BOX_COLOR = (255, 165, 0)      # 検出物体: 明るい青
SPOT_COLOR = (60, 220, 235)    # 死角: 黄


def draw_hud(frame, result, alpha=0.25):
    """分析結果を描き込んだ新しい画像を返す。渡した画像そのものは変えない。

    frame  : カラー画像 (縦x横x3)
    result : DangerWatcher.watch() の返り値
             (level, risk, things, blind_spots, danger_line を使う)
    alpha  : 危険域の網かけの濃さ (0〜1)
    """
    if frame is None:
        raise ValueError("画像がありません")
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("HUDはカラー画像(縦x横x3)にだけ描けます")
    if frame.size == 0:
        raise ValueError("画像が空です")
    if not isinstance(result, dict):
        raise ValueError("結果(result)は watch() の返り値を渡してください")
    for key in ("level", "risk", "things", "blind_spots", "danger_line"):
        if key not in result:
            raise ValueError("結果に " + key + " がありません")
    if not 0 <= alpha <= 1:
        raise ValueError("網かけの濃さ(alpha)は0〜1にしてください")

    out = frame.copy()
    h, w = out.shape[:2]
    level = result["level"]
    color = LEVEL_COLORS.get(level, LEVEL_COLORS[0])

    # --- 危険域の網かけ (画面下・進路の中央) ---
    top = int(h * min(max(result["danger_line"], 0.0), 1.0))
    left = int(w * CENTER_LEFT)
    right = int(w * CENTER_RIGHT)
    if top < h - 1 and left < right:
        shade = out.copy()
        cv2.rectangle(shade, (left, top), (right, h - 1), color, -1)
        cv2.addWeighted(shade, alpha, out, 1 - alpha, 0, dst=out)
        cv2.rectangle(out, (left, top), (right, h - 1), color, 1)

    # --- 検出した物体: 枠 + 進行方向の矢印 ---
    for t in result["things"]:
        x1 = int(t["x"] * w)
        y1 = int(t["y"] * h)
        x2 = int((t["x"] + t["w"]) * w)
        y2 = int((t["y"] + t["h"]) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        # 矢印: 今の速さで8フレーム先までの移動を示す
        cx, cy = int(t["cx"] * w), int(t["cy"] * h)
        ex = int((t["cx"] + t["vx"] * 8) * w)
        ey = int((t["cy"] + t["vy"] * 8) * h)
        if (ex, ey) != (cx, cy):
            cv2.arrowedLine(out, (cx, cy), (ex, ey), BOX_COLOR, 2, tipLength=0.35)

    # --- 死角候補: 縦線 + 根元の三角マーカー ---
    for spot in result["blind_spots"]:
        x = int(spot["x"] * w)
        y_top = int(h * 0.35)
        cv2.line(out, (x, y_top), (x, h - 1), SPOT_COLOR, 2)
        size = max(4, int(h * 0.04))
        base_y = int(h * spot["y"])
        pts = np.array([
            (x, base_y - size),
            (x - size, base_y + size),
            (x + size, base_y + size),
        ])
        cv2.fillPoly(out, [pts], SPOT_COLOR)

    # --- 画面上部の危険度バー ---
    risk = min(1.0, max(0.0, float(result["risk"])))
    bar_w = int(w * 0.35)
    bar_h = max(4, int(h * 0.03))
    x0, y0 = int(w * 0.02), int(h * 0.03)
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (70, 70, 70), -1)
    fill = int(bar_w * risk)
    if fill > 0:
        cv2.rectangle(out, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)

    return out
