# 報告があった場面の映像を残すファイル
#
# 直近の数秒分のフレームを覚えておき、ユーザーが「誤報を報告」
# 「見逃しを報告」した瞬間に、その場面を画像(PNG)で保存する。
#
# 走行ログ (triplog.py) と合わせて見返すことで、
# 「誤報が出た時、実際には何が映っていたのか」を確かめられる。
#
# 画像は自前のPNG書き出し (imgproc.encode_png) で作るので、
# 追加のライブラリが無くても・保存先が日本語名でも動く。

import collections
import os
import time

import numpy as np

from app import imgproc

# 覚えておくフレーム数 (15fpsで約3秒)
KEEP_FRAMES = 45
# 1回の報告で保存する枚数 (覚えている中から等間隔に選ぶ)
SAVE_COUNT = 3
# フォルダに残す画像の上限。超えたら古い物から消す
MAX_FILES = 60


class SnapshotKeeper:
    """直近の映像を覚えておき、報告の瞬間の場面を画像で残す係。

    使い方:
        keeper = SnapshotKeeper("trip_snapshots")
        keeper.add(frame)              # 毎フレーム呼ぶ
        paths = keeper.save("missed")  # 報告があった時。保存した場所を返す
    """

    def __init__(self, folder, keep=KEEP_FRAMES, save_count=SAVE_COUNT,
                 max_files=MAX_FILES):
        if not isinstance(folder, str) or not folder.strip():
            raise ValueError("保存先のフォルダ(folder)を指定してください")
        if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
            raise ValueError("覚えるフレーム数(keep)は1以上の整数にしてください")
        if not isinstance(save_count, int) or isinstance(save_count, bool) or save_count < 1:
            raise ValueError("保存する枚数(save_count)は1以上の整数にしてください")
        if (not isinstance(max_files, int) or isinstance(max_files, bool)
                or max_files < save_count):
            raise ValueError("画像の上限(max_files)は保存枚数以上の整数にしてください")
        self.folder = folder
        self.save_count = save_count
        self.max_files = max_files
        self._frames = collections.deque(maxlen=keep)

    def add(self, frame):
        """フレームを1枚覚える。古い物から順に忘れていく。"""
        if frame is None:
            raise ValueError("画像がありません")
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or frame.size == 0:
            raise ValueError("画像の形式が正しくありません")
        self._frames.append(frame)

    def save(self, tag, t=None):
        """覚えているフレームから数枚を選んでPNGで保存する。

        tag : ファイル名の頭につける言葉 (報告の種類など)
        t   : 時刻 (テスト用。省略すると現在時刻)

        返り値: 保存したファイルのパスのリスト。
                何も覚えていない・保存に失敗した分は含まれない。
        """
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("名前(tag)を文字で渡してください")
        frames = list(self._frames)
        if not frames:
            return []

        now = time.time() if t is None else t
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        stamp += "_{:03d}".format(int(round(now * 1000)) % 1000)

        # 覚えている中から等間隔に選ぶ (一番古い物〜一番新しい物)
        count = min(self.save_count, len(frames))
        if count == 1:
            picks = [frames[-1]]
        else:
            step = (len(frames) - 1) / (count - 1)
            picks = [frames[int(round(i * step))] for i in range(count)]

        saved = []
        try:
            os.makedirs(self.folder, exist_ok=True)
        except OSError:
            return []
        for i, frame in enumerate(picks):
            name = "{}_{}_{}.png".format(tag.strip(), stamp, i)
            path = os.path.join(self.folder, name)
            try:
                encoded = imgproc.encode_png(frame)
                with open(path, "wb") as f:
                    f.write(encoded)
                saved.append(path)
            except (OSError, ValueError):
                continue  # 1枚の失敗で全体は止めない
        self._prune()
        return saved

    def clear(self):
        """覚えているフレームを全部忘れる (保存済みの画像は残る)。"""
        self._frames.clear()

    def _prune(self):
        """フォルダの画像が上限を超えたら、古い物から消す。"""
        try:
            files = [os.path.join(self.folder, name)
                     for name in os.listdir(self.folder)
                     if name.endswith(".png")]
            if len(files) <= self.max_files:
                return
            files.sort(key=os.path.getmtime)
            for path in files[:len(files) - self.max_files]:
                os.remove(path)
        except OSError:
            pass  # 掃除に失敗しても保存自体は続ける
