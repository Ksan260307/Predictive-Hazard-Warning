# カメラを開くファイル
#
# 環境ごとに使える方法が違うので、ここでまとめて面倒を見る:
#   ・Android : Kivy のカメラ機能 (プレビューの絵をそのまま取り出す)
#   ・PC      : OpenCV (cv2) が入っていれば cv2.VideoCapture (開発用)
#
# どちらで開いても cv2.VideoCapture と同じ使い方
# (isOpened / read / release) ができるようにそろえる。
# read() が返す画像は「上から下へ・色は B,G,R の順」の numpy 配列。

import sys

import numpy as np


def open_camera(camera_no):
    """カメラを開いて返す。開けない時は None を返す。"""
    from kivy.utils import platform

    if platform == "android":
        try:
            return KivyCameraCapture(index=camera_no)
        except Exception:
            return None  # 権限がまだ無い時など。あとで開き直せる

    # PCでは OpenCV があればそれを使う (無ければカメラなしで動く)
    try:
        import cv2
    except ImportError:
        return None
    if sys.platform == "win32":
        capture = cv2.VideoCapture(camera_no, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(camera_no)
    if capture.isOpened():
        return capture
    capture.release()
    return None


class KivyCameraCapture:
    """Kivyのカメラ機能を cv2.VideoCapture と同じ形で使えるようにする包み。"""

    def __init__(self, index=0, resolution=(640, 480)):
        from kivy.core.camera import Camera as CoreCamera
        self._camera = CoreCamera(index=index, resolution=resolution,
                                  stopped=True)
        self._camera.start()
        self._opened = True
        # 映像が届くまでの間に見せる黒い画面
        self._blank = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    def isOpened(self):
        return self._opened

    def read(self):
        """(True, 画像) を返す。映像がまだ来ていない間は黒い画面を返す。"""
        if not self._opened:
            return False, None
        texture = getattr(self._camera, "texture", None)
        if texture is None:
            return True, self._blank.copy()
        try:
            pixels = texture.pixels  # R,G,B,A の順・下から上へ
            w, h = texture.size
        except Exception:
            return True, self._blank.copy()
        arr = np.frombuffer(pixels, dtype=np.uint8)
        if arr.size != w * h * 4:
            return True, self._blank.copy()
        arr = arr.reshape(h, w, 4)
        # 上下をひっくり返し、色を B,G,R の順に入れ替える
        return True, arr[::-1, :, [2, 1, 0]]

    def release(self):
        if self._opened:
            try:
                self._camera.stop()
            except Exception:
                pass
        self._opened = False
