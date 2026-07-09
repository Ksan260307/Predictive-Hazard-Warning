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
import time

import numpy as np

# 映像が届かないまま何フレーム待ったら諦めるか (15fpsで約3秒)
GIVEUP_BLANK_FRAMES = 45
# 諦めた直後は開き直してもまた映らないことが多いので、
# この秒数はデモ映像でつなぐ (呼び出し側が定期的に開き直してくれる)
REOPEN_COOLDOWN = 15.0

# 最後に「映像が来ない」と諦めた時刻 (このモジュール内で覚えておく)
_gave_up_at = 0.0


def _recently_gave_up(now=None):
    """少し前に映像が来ず諦めたばかりかどうか。"""
    now = time.time() if now is None else now
    return (now - _gave_up_at) < REOPEN_COOLDOWN


def _note_giveup(now=None):
    """「映像が来ないまま諦めた」ことを覚える。"""
    global _gave_up_at
    _gave_up_at = time.time() if now is None else now


def open_camera(camera_no):
    """カメラを開いて返す。開けない時は None を返す。"""
    from kivy.utils import platform

    if platform == "android":
        # 直前に「開けたのに映像が来ない」で諦めたばかりなら、開き直しても
        # 同じ結果になりやすい。None を返してデモ映像でつないでもらう
        if _recently_gave_up():
            return None
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
        self._blank_count = 0  # 映像が来ないまま何フレーム経ったか
        # 映像が届くまでの間に見せる黒い画面
        self._blank = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    def isOpened(self):
        return self._opened

    def read(self):
        """(True, 画像) を返す。映像がまだ来ていない間は黒い画面を返す。

        映像が来ないまま長く経った (権限が拒否された・他のアプリが
        カメラを使っている等) 時は (False, None) を返して閉じる。
        呼び出し側はそれを合図にデモ映像へ切り替えられる。
        """
        if not self._opened:
            return False, None
        texture = getattr(self._camera, "texture", None)
        if texture is None:
            return self._blank_frame()
        try:
            pixels = texture.pixels  # R,G,B,A の順・下から上へ
            w, h = texture.size
        except Exception:
            return self._blank_frame()
        arr = np.frombuffer(pixels, dtype=np.uint8)
        if arr.size != w * h * 4:
            return self._blank_frame()
        arr = arr.reshape(h, w, 4)
        self._blank_count = 0  # ちゃんと映った
        # 上下をひっくり返し、色を B,G,R の順に入れ替える
        return True, arr[::-1, :, [2, 1, 0]]

    def _blank_frame(self):
        """映像が来ていない間の1フレーム。待ちすぎたら諦める。"""
        self._blank_count += 1
        if self._blank_count > GIVEUP_BLANK_FRAMES:
            _note_giveup()
            self.release()
            return False, None
        return True, self._blank.copy()

    def release(self):
        if self._opened:
            try:
                self._camera.stop()
            except Exception:
                pass
        self._opened = False
