# 移動モード(徒歩・自転車・車)の定義
#
# モードによって「どれだけ先を読むか」「どれだけ早めに警告するか」が変わる。
# 速い乗り物ほど、遠くまで・早めに警告する必要がある。

# 各モードのパラメータ:
#   name             : 画面に表示する名前
#   step_scale       : 予測する先読みの長さの倍率 (速いほど遠くまで読む)
#   danger_shift     : 危険ラインの補正。マイナスほど手前(早め)に警告する
#   phantom_strength : 死角からの「飛び出し予測」をどれだけ重く見るか (0〜1)
#   env_weight       : 地図情報(交差点など)による危険をどれだけ重く見るか (0〜1)
#   fast_speed       : このモードで「速い」とみなす速度 (m/s)
#   base_move        : GPSが無い時に仮定する移動度合い (0〜1)
MODES = {
    "walk": {
        "name": "徒歩",
        "step_scale": 0.8,
        "danger_shift": 0.05,
        "phantom_strength": 0.35,
        "env_weight": 0.5,
        "fast_speed": 2.5,    # 早歩き程度
        "base_move": 0.3,
    },
    "bicycle": {
        "name": "自転車",
        "step_scale": 1.3,
        "danger_shift": 0.0,
        "phantom_strength": 0.5,
        "env_weight": 0.75,
        "fast_speed": 8.0,    # 約30km/h
        "base_move": 0.5,
    },
    "car": {
        "name": "車",
        "step_scale": 2.0,
        "danger_shift": -0.10,
        "phantom_strength": 0.6,
        "env_weight": 1.0,
        "fast_speed": 16.7,   # 約60km/h
        "base_move": 0.7,
    },
}

MODE_KEYS = tuple(MODES)  # ("walk", "bicycle", "car")


def mode_profile(key):
    """モード名からパラメータ一式を取り出す。知らない名前ならエラー。"""
    if not isinstance(key, str) or key not in MODES:
        raise ValueError("モードは walk / bicycle / car のいずれかを指定してください")
    return dict(MODES[key])
