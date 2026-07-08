# デモ映像を作るファイル
#
# カメラが無い環境(会議室のPCなど)でもアプリの動きを確認できるように、
# 「壁ぎわの道を進み、右から歩行者が横断してくる」風景を合成する。
#
# cv2.VideoCapture と同じ使い方 (isOpened / read / release) ができるので、
# main.py はカメラとデモを同じコードで扱える。

import numpy as np

# 歩行者が現れる周期 (フレーム数)
CROSS_PERIOD = 120
# 周期の中で歩行者が歩いている区間
CROSS_START = 10
CROSS_END = 90


class DemoCamera:
    """デモ映像を1フレームずつ作って返す係。"""

    def __init__(self, width=320, height=180, seed=0):
        if not isinstance(width, int) or isinstance(width, bool) or width < 32:
            raise ValueError("横幅(width)は32以上の整数にしてください")
        if not isinstance(height, int) or isinstance(height, bool) or height < 32:
            raise ValueError("高さ(height)は32以上の整数にしてください")
        self.width = width
        self.height = height
        self._tick = 0
        self._base = self._make_base(width, height, seed)

    @staticmethod
    def _make_base(width, height, seed):
        """動かない背景を作る: 空 + 路面 + 左の壁 + 細かい模様。"""
        rng = np.random.default_rng(seed)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 空 (上40%) と路面 (下60%)。路面は手前ほど明るいグレー
        sky_h = int(height * 0.4)
        frame[:sky_h] = (140, 120, 100)  # 青みがかった空
        for y in range(sky_h, height):
            shade = 120 + int(60 * (y - sky_h) / max(1, height - sky_h))
            frame[y] = (shade, shade, shade)

        # 左側の壁 (暗い建物)。この端が「死角」になる
        frame[:, : int(width * 0.28)] = (40, 42, 48)

        # 手ブレ補正が目印にできるよう、全体に細かい模様を足す
        noise = rng.integers(-12, 13, size=(height, width, 1), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame

    def isOpened(self):
        return True

    def read(self):
        """(True, 次のフレーム) を返す。cv2.VideoCapture.read() と同じ形。"""
        self._tick += 1
        frame = self._base.copy()
        h, w = self.height, self.width

        # 歩行者: 周期的に右から現れて、進路の中央へ横断してくる
        phase = self._tick % CROSS_PERIOD
        if CROSS_START <= phase < CROSS_END:
            progress = (phase - CROSS_START) / (CROSS_END - CROSS_START)
            cx = 0.9 - 0.5 * progress          # 右(0.9)から中央(0.4)へ
            cy = 0.45 + 0.35 * progress        # だんだん手前(下)へ
            size = int(h * (0.10 + 0.08 * progress))  # 近づくほど大きく
            x1 = max(0, int(cx * w - size // 2))
            y1 = max(0, int(cy * h - size))
            x2 = min(w - 1, x1 + size)
            y2 = min(h - 1, int(cy * h))
            frame[y1:y2, x1:x2] = (230, 228, 225)  # 明るい服の歩行者

        return True, frame

    def release(self):
        pass
