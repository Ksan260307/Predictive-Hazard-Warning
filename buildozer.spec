# Android用にアプリを固めるための設定ファイル (buildozer 用)
#
# 使い方 (Linux か WSL の上で):
#   pip install buildozer
#   buildozer android debug
# できた .apk をスマホに入れると動きます。詳しくは README.md を見てください。

[app]
title = 先読み危険予測
package.name = sakiyomiguard
package.domain = org.example
source.dir = .
source.include_exts = py,json,md,png,onnx
version = 2.0.0

# アプリのアイコンと起動画面
icon.filename = %(source.dir)s/data/icon.png
presplash.filename = %(source.dir)s/data/presplash.png

# アプリが使うライブラリ
requirements = python3,kivy,plyer

# 横画面で使う
orientation = landscape
fullscreen = 0

# カメラと位置情報を使う許可
android.permissions = CAMERA,ACCESS_FINE_LOCATION,ACCESS_COARSE_LOCATION,INTERNET,VIBRATE

android.api = 33
android.accept_sdk_license = True
android.minapi = 24

android.sdk = 33
android.ndk = 25b

[buildozer]
log_level = 2
warn_on_root = 1
