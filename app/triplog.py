# 走行ログを書き残すファイル
#
# 監視中の状態 (危険度・レベル・理由など) を1秒ごとに1行のJSONで記録する。
# ユーザーが「誤報を報告」「見逃しを報告」した瞬間も記録する。
#
# あとからログを見返すことで、
#   ・誤報が出た場面はどんな状況だったか
#   ・しきい値や感度をどう調整すべきか
# を実データにもとづいて確かめられる。
#
# ファイルは1行1レコードのJSON (JSONL形式)。大きくなりすぎたら
# 古い分を .old に退避して書き続ける (合わせて約10MBまで)。

import json
import os
import time

# 状態を記録する間隔 (秒)。毎フレーム書くと多すぎるので間引く
LOG_INTERVAL = 1.0
# ログファイルの上限 (バイト)。超えたら .old に退避して新しく始める
MAX_BYTES = 5 * 1024 * 1024

# 記録する項目 (watch() の結果から抜き出す)
RESULT_KEYS = ("level", "risk", "image_risk", "env_risk", "detector", "reasons")


class TripLogger:
    """走行ログを書き残す係。

    使い方:
        logger = TripLogger("triplog.jsonl")
        logger.log(result)                 # 毎フレーム呼んでよい (自動で間引く)
        logger.log_feedback("false_alarm") # 報告があった時
    """

    def __init__(self, path, interval=LOG_INTERVAL, max_bytes=MAX_BYTES):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("ログの保存先(path)を指定してください")
        if not isinstance(interval, (int, float)) or isinstance(interval, bool) or interval < 0:
            raise ValueError("記録の間隔(interval)は0以上の数にしてください")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1024:
            raise ValueError("ログの上限(max_bytes)は1024以上の整数にしてください")
        self.path = path
        self.interval = float(interval)
        self.max_bytes = max_bytes
        self._last_time = None

    def log(self, result, t=None):
        """監視の状態を1つ記録する。前の記録から間隔が空くまでは何もしない。

        result : DangerWatcher.watch() の返り値
        t      : 時刻 (テスト用。省略すると現在時刻)
        """
        if not isinstance(result, dict):
            raise ValueError("結果(result)は watch() の返り値を渡してください")
        for key in ("level", "risk"):
            if key not in result:
                raise ValueError("結果に " + key + " がありません")

        now = time.time() if t is None else t
        if self._last_time is not None and now - self._last_time < self.interval:
            return False
        self._last_time = now

        record = {"t": round(now, 3), "kind": "state"}
        for key in RESULT_KEYS:
            if key in result:
                record[key] = result[key]
        road = result.get("road")
        if isinstance(road, dict):
            record["speed"] = road.get("speed")
        record["things"] = len(result.get("things", []))
        self._write(record)
        return True

    def log_feedback(self, kind, t=None, snapshots=None):
        """ユーザーの報告を記録する。こちらは間引かず必ず書く。

        snapshots : その場面の画像のパスのリスト (snapshot.py が保存した物)。
                    渡すとログにも書かれ、あとで場面を見返せる。
        """
        if not isinstance(kind, str) or not kind:
            raise ValueError("報告の種類(kind)を文字で渡してください")
        now = time.time() if t is None else t
        record = {"t": round(now, 3), "kind": "feedback", "feedback": kind}
        if snapshots:
            record["snapshots"] = [str(p) for p in snapshots]
        self._write(record)

    # ---------------- 内部処理 ----------------

    def _write(self, record):
        self._rotate_if_big()
        line = json.dumps(record, ensure_ascii=False)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass  # 書き込みに失敗しても監視は止めない

    def _rotate_if_big(self):
        """ログが大きくなりすぎたら .old に退避する (前の .old は消える)。"""
        try:
            if os.path.getsize(self.path) < self.max_bytes:
                return
            backup = self.path + ".old"
            if os.path.exists(backup):
                os.remove(backup)
            os.replace(self.path, backup)
        except OSError:
            pass  # 退避に失敗しても書き込み自体は続ける
