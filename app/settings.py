# 設定を管理するファイル
#
# アプリの動きを変える数値を、ここでまとめて管理します。
# ・おかしな値が来ても、必ず「使える値」に直して返します
# ・ファイルが壊れていても、初期値でちゃんと動きます

import json
import os

# 設定ファイルの置き場所(このファイルと同じフォルダの1つ上)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settei.json")

# 設定の初期値
DEFAULTS = {
    "camera_no": 0,        # 使うカメラの番号
    "sensitivity": 5,      # 動きを見つける感度 (1=にぶい 〜 10=するどい)
    "min_size": 0.005,     # 見つける物の最小の大きさ (画面全体を1とした割合)
    "future_steps": 15,    # 何歩先の未来まで予測するか
    "future_samples": 30,  # 未来を何通り考えるか
    "update_rate": 0.5,    # 未来予測の入れ替わりの速さ
    "risk_line": 0.5,      # このくらい危なくなったら注意を強める
    "danger_line": 0.7,    # 画面のどこから下を「ぶつかる場所」とみなすか
    "sound_on": True,      # 音で知らせる
    "voice_on": False,     # 声で知らせる (警告の内容を読み上げる)
    "show_boxes": True,    # 見つけた物に枠を表示する
    "mode": "walk",        # 移動モード (walk=徒歩 / bicycle=自転車 / car=車)
    "stabilize": True,     # 自分の移動によるカメラの揺れを打ち消す
    "use_ml_detector": True,  # AIモデルで物体を認識する (モデルが無い時は動き検出)
    "power_save": True,    # 静かな時は分析を間引いて電池を節約する
    "use_location": False, # 位置情報(GPS)を使う (地図はOpenStreetMap。キー不要)
    "learning_on": True,   # ユーザーの報告から学習する
    "demo_mode": False,    # カメラの代わりにデモ映像を使う
    "save_log": False,     # 走行ログを保存する (誤報の見直しや調整に使える)
}

# それぞれの設定で許される範囲 (最小, 最大)
RANGES = {
    "camera_no": (0, 10),
    "sensitivity": (1, 10),
    "min_size": (0.001, 0.2),
    "future_steps": (1, 60),
    "future_samples": (1, 200),
    "update_rate": (0.01, 1.0),
    "risk_line": (0.05, 0.95),
    "danger_line": (0.3, 0.95),
}

# 整数で持つ設定
INT_KEYS = ("camera_no", "sensitivity", "future_steps", "future_samples")
# はい/いいえ で持つ設定
BOOL_KEYS = ("sound_on", "voice_on", "show_boxes", "stabilize", "use_location",
             "learning_on", "demo_mode", "use_ml_detector", "power_save",
             "save_log")
# 決まった候補から選ぶ設定
CHOICE_KEYS = {"mode": ("walk", "bicycle", "car")}
# 自由な文字で持つ設定
TEXT_KEYS = ()


def check_settings(values):
    """設定の中身を調べて、必ず使える形に直して返す。

    ・知らない項目は捨てる
    ・型がおかしい値は初期値に戻す
    ・範囲から外れた値は範囲の中に収める
    """
    fixed = dict(DEFAULTS)
    if not isinstance(values, dict):
        return fixed

    for key in DEFAULTS:
        if key not in values:
            continue
        value = values[key]

        if key in BOOL_KEYS:
            # はい/いいえ の項目。True/False か 0/1 だけ受け付ける
            if isinstance(value, bool):
                fixed[key] = value
            elif value in (0, 1):
                fixed[key] = bool(value)
            continue

        if key in CHOICE_KEYS:
            # 決まった候補の中にある値だけ受け付ける
            if value in CHOICE_KEYS[key]:
                fixed[key] = value
            continue

        if key in TEXT_KEYS:
            # 文字の項目。文字以外は捨てる
            if isinstance(value, str):
                fixed[key] = value.strip()
            continue

        # 数の項目
        if isinstance(value, bool):
            continue  # True/False を数として使うのは間違いなので捨てる
        try:
            if key in INT_KEYS:
                number = int(value)
            else:
                number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue  # 数に直せない値は捨てて初期値のまま

        low, high = RANGES[key]
        # 範囲の外なら、いちばん近い端に収める
        number = max(low, min(high, number))
        fixed[key] = number

    return fixed


def load_settings(path=SETTINGS_FILE):
    """設定ファイルを読み込む。読めない時は初期値を返す。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        # ファイルが無い・壊れている時は初期値で動く
        return dict(DEFAULTS)
    return check_settings(data)


def save_settings(values, path=SETTINGS_FILE):
    """設定をファイルに保存する。保存した内容を返す。"""
    checked = check_settings(values)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checked, f, ensure_ascii=False, indent=2)
    return checked
