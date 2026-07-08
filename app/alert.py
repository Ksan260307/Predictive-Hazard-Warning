# ユーザーへの知らせ方を決めるファイル
#
# 注意レベル(0/1/2)を、画面に出す言葉・色・音に変えます。

# レベルごとの知らせ方
#   name  : 大きく出す言葉
#   text  : 説明の言葉
#   color : 画面の色 (赤, 緑, 青, 濃さ) それぞれ0〜1
#   sound : 音を鳴らすかどうか
_MESSAGES = {
    0: {
        "name": "安全",
        "text": "周囲に差し迫った危険はありません",
        "color": (0.15, 0.6, 0.25, 1),   # 緑
        "sound": False,
    },
    1: {
        "name": "注意",
        "text": "注意が必要な状況です。周囲を確認してください",
        "color": (0.85, 0.65, 0.1, 1),   # 黄色
        "sound": False,
    },
    2: {
        "name": "危険",
        "text": "衝突の恐れがあります。直ちに減速・停止してください",
        "color": (0.8, 0.1, 0.1, 1),     # 赤
        "sound": True,
    },
}


def message_for(level):
    """注意レベルに合った知らせ方を返す。

    返り値: {"level", "name", "text", "color", "sound"} の辞書
    """
    if not isinstance(level, int) or isinstance(level, bool):
        raise ValueError("レベルは0・1・2のどれかの整数にしてください")
    if level not in _MESSAGES:
        raise ValueError("レベルは0・1・2のどれかにしてください")
    info = dict(_MESSAGES[level])
    info["level"] = level
    return info
