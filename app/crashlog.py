# 起動時・処理中のエラーを取りこぼさず残すためのファイル
#
# Android 実機では画面にトレースバックが出ないまま落ちるので、原因が
# 分かりにくい。そこで、例外が起きたら:
#   ・全文を組み立てて (format_report)
#   ・端末内の読み取れる場所とログ(logcat)に書き出す (save_report)
#   ・画面にも出す (main.py 側)
# ようにして、adb が無くても原因を確かめられるようにする。

import os
import platform as _platform
import sys
import time
import traceback


def format_report(exc=None):
    """エラーの全文 (時刻・環境・トレースバック) を文字列にして返す。"""
    lines = []
    lines.append("=== 先読み危険予測 クラッシュ記録 ===")
    lines.append("時刻: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("Python: " + sys.version.replace("\n", " "))
    lines.append("OS: " + _platform.platform())
    try:
        from kivy.utils import platform as kivy_platform
        lines.append("Kivy platform: " + str(kivy_platform))
    except Exception:
        pass
    lines.append("")
    if exc is None:
        exc = sys.exc_info()[1]
    if exc is not None:
        lines.append("".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)).rstrip())
    else:
        lines.append("(例外情報がありません)")
    return "\n".join(lines)


def _candidate_dirs(user_data_dir=None):
    """クラッシュ記録を置きたい場所の候補を、優先度の高い順に返す。"""
    dirs = []
    if user_data_dir:
        dirs.append(user_data_dir)  # Kivyがくれる書き込み可能な場所
    # Androidで、ファイルマネージャから見える場所
    for env_key in ("EXTERNAL_STORAGE",):
        base = os.environ.get(env_key)
        if base:
            dirs.append(os.path.join(base, "Download"))
            dirs.append(base)
    # 最後の手段: このファイルと同じ場所 (PCや開発時)
    dirs.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return dirs


def save_report(text, user_data_dir=None, filename="sakiyomiguard_crash.txt"):
    """クラッシュ記録を書き出す。書けた場所のパスのリストを返す。

    ログ(logcat)には必ず出し、加えて端末内のファイルにも残す。
    どこにも書けなくても例外は投げない (記録処理でさらに落ちないため)。
    """
    # まず標準出力へ (Androidでは logcat に "python" タグで出る)
    try:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
    except Exception:
        pass

    saved = []
    for folder in _candidate_dirs(user_data_dir):
        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            saved.append(path)
            break  # 1か所書ければ十分
        except Exception:
            continue
    return saved
