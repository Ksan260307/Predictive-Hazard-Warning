# まわりの危険を予測して知らせるアプリの中身

APP_NAME = "先読み危険予測"
__version__ = "2.0.0"

#
# 「いま見えている物」から「この先ありえる未来」を予測して判断する作りです。
#
#   カメラの画像        → detector.py / objectdetect.py  (いま見えている物)
#   未来の予測          → future.py    (未来の分布)
#   危なさなどの計算    → risk.py      (危険度・迷い・余裕)
#   注意の強さを決める  → control.py
#   知らせ方            → alert.py
#   全部をつなぐ        → watcher.py
#   設定                → settings.py
