# AI物体認識のモデル (YOLOv8n) を用意するスクリプト
#
# 使い方:
#   pip install ultralytics
#   python tools/download_model.py
#
# ultralytics が YOLOv8n をダウンロードして ONNX 形式に変換し、
# このスクリプトが data/yolov8n.onnx に配置する。
# 配置後、アプリは自動でAI物体認識を使うようになる
# (設定「AIで物体を認識する」がオンの場合)。
#
# モデルが無くてもアプリは従来の動き検出で動くので、このスクリプトは必須ではない。

import os
import shutil
import sys

TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "yolov8n.onnx",
)


def main():
    if os.path.isfile(TARGET):
        print("すでにモデルがあります:", TARGET)
        print("入れ直す場合は、このファイルを削除してから再実行してください。")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics がインストールされていません。先に以下を実行してください:")
        print()
        print("    pip install ultralytics")
        print()
        return 1

    print("YOLOv8n をダウンロードして ONNX 形式に変換します (数分かかります)...")
    model = YOLO("yolov8n.pt")
    exported = model.export(format="onnx", imgsz=640, opset=12)

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    shutil.move(str(exported), TARGET)

    # OpenCV で読めることを確かめる
    import cv2
    cv2.dnn.readNetFromONNX(TARGET)
    print("完了:", TARGET)
    print("アプリを起動すると、AI物体認識が自動で有効になります。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
