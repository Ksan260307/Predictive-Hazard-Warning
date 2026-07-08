# 学習済みAIモデルで「人・車・自転車など」を見つけるファイル
#
# detector.py の「動いた物」検出との違い:
#   ・止まっている人や車も見つけられる (動きに頼らない)
#   ・「何であるか」(歩行者・車・自転車など) が分かる
#   ・光のちらつき・木の揺れなどの誤検出に強い
#
# YOLOv8 の ONNX モデルを onnxruntime で動かす。
# モデルファイル (data/yolov8n.onnx) と onnxruntime の両方が
# そろっている時だけ使える。モデルの用意は tools/download_model.py を参照。
# どちらかが無い環境では watcher.py が従来の動き検出に自動で切り替える。
#
# 見つけた物は detector.py と同じ形式 (0〜1の割合) で返すので、
# 後段 (tracker / future / hud) はどちらの検出器でも同じように動く。

import os

import numpy as np

from app import imgproc

# モデルファイルの置き場所 (プロジェクトの data フォルダ)
MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "yolov8n.onnx",
)

# モデルに入れる画像の一辺の長さ (YOLOv8の標準)
INPUT_SIZE = 640

# COCOデータセットのクラス番号 → 名前。交通の危険に関わる物だけを使う
TARGET_CLASSES = {
    0: "person",      # 歩行者
    1: "bicycle",     # 自転車
    2: "car",         # 車
    3: "motorcycle",  # バイク
    5: "bus",         # バス
    7: "truck",       # トラック
    15: "cat",        # 動物 (飛び出しの危険)
    16: "dog",
}

# 画面表示・警告文で使う日本語名
LABELS_JA = {
    "person": "歩行者",
    "bicycle": "自転車",
    "car": "車",
    "motorcycle": "バイク",
    "bus": "バス",
    "truck": "トラック",
    "cat": "動物",
    "dog": "動物",
}


def model_available(path=MODEL_FILE):
    """モデルファイルが置いてあるかどうかを返す。"""
    return isinstance(path, str) and os.path.isfile(path)


def runtime_available():
    """AIモデルを動かすライブラリ (onnxruntime) が入っているかを返す。"""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _nms(boxes, scores, threshold):
    """同じ物に重なった枠をまとめ、残す枠の番号を返す (NMS)。

    boxes  : [x, y, 幅, 高さ] のリスト
    scores : 各枠の確信度
    threshold : これ以上重なっている枠は同じ物とみなす (0〜1)
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    order = np.argsort(scores)[::-1]  # 確信度の高い順
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        # 残りの枠との重なり具合 (IoU) を計算する
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = (np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1))
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= threshold]
    return keep


def parse_detections(output, conf=0.35, nms=0.45, input_size=INPUT_SIZE):
    """YOLOv8のONNX出力から、物のリストを取り出す。

    output     : モデルの出力。形は (1, 4+クラス数, 候補数) か、その転置。
                 各候補は [中心x, 中心y, 幅, 高さ, クラス0の点数, クラス1の点数, ...]
                 (座標は入力画像のピクセル単位)
    conf       : この確信度に満たない候補は捨てる (0〜1)
    nms        : 同じ物への重複した枠をまとめる強さ (0〜1)
    input_size : モデルに入れた画像の一辺の長さ

    返り値: detector.py と同じ形式の物のリスト (x, y, w, h, cx, cy, vx, vy)
            に label (クラス名) と score (確信度) が加わったもの。
    """
    if not 0 < conf < 1:
        raise ValueError("確信度のしきい値(conf)は0より大きく1より小さくしてください")
    if not 0 <= nms <= 1:
        raise ValueError("重複除去の強さ(nms)は0〜1にしてください")

    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("モデルの出力の形が正しくありません")
    # (4+クラス数, 候補数) で来たら (候補数, 4+クラス数) に揃える
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 5:
        raise ValueError("モデルの出力の形が正しくありません")

    # 候補ごとに、いちばん点数の高いクラスを選ぶ
    class_scores = arr[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    # 確信度が低い候補と、交通に関係ないクラスの候補を捨てる
    boxes = []
    kept_scores = []
    kept_labels = []
    for i in range(len(arr)):
        if scores[i] < conf:
            continue
        label = TARGET_CLASSES.get(int(class_ids[i]))
        if label is None:
            continue
        cx, cy, w, h = arr[i, :4]
        boxes.append([float(cx - w / 2), float(cy - h / 2), float(w), float(h)])
        kept_scores.append(float(scores[i]))
        kept_labels.append(label)

    if not boxes:
        return []

    # 同じ物に重なった枠をまとめる (NMS)
    keep = _nms(boxes, kept_scores, nms)

    things = []
    for i in keep:
        x, y, w, h = boxes[i]
        # 0〜1の割合に直し、画面からはみ出た分は切りそろえる
        x1 = max(0.0, min(1.0, x / input_size))
        y1 = max(0.0, min(1.0, y / input_size))
        x2 = max(0.0, min(1.0, (x + w) / input_size))
        y2 = max(0.0, min(1.0, (y + h) / input_size))
        if x2 <= x1 or y2 <= y1:
            continue
        things.append({
            "x": x1, "y": y1,
            "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "vx": 0.0, "vy": 0.0,  # 速度はトラッカーが複数フレームから求める
            "label": kept_labels[i],
            "score": kept_scores[i],
        })
    return things


class ObjectFinder:
    """AIモデルで人・車などを見つける係。detector.MovingThingFinder と同じ使い方。

    使い方:
        if objectdetect.model_available() and objectdetect.runtime_available():
            finder = ObjectFinder()
            things = finder.find(frame)   # 毎フレーム呼ぶ
    """

    def __init__(self, path=MODEL_FILE, conf=0.35, nms=0.45):
        if not 0 < conf < 1:
            raise ValueError("確信度のしきい値(conf)は0より大きく1より小さくしてください")
        if not 0 <= nms <= 1:
            raise ValueError("重複除去の強さ(nms)は0〜1にしてください")
        if not model_available(path):
            raise ValueError("モデルファイルがありません: " + str(path))
        try:
            import onnxruntime
        except ImportError:
            raise ValueError("onnxruntime が入っていないためAI認識は使えません")
        try:
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise ValueError("モデルファイルを読み込めません: " + str(path)) from exc
        self._input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.nms = nms

    def find(self, frame):
        """画像1枚から、人・車などのリストを返す。"""
        if frame is None:
            raise ValueError("画像がありません")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("画像の形式が正しくありません")
        if frame.ndim == 2:
            frame = np.stack([frame, frame, frame], axis=2)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("画像はカラー(縦x横x3)か白黒(縦x横)にしてください")

        # モデルの入力に合わせる: 640x640 に伸縮 → R,G,B の順 → 0〜1 に直す
        resized = imgproc.resize(frame, INPUT_SIZE, INPUT_SIZE)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[np.newaxis])

        output = self.session.run(None, {self._input_name: blob})[0]
        return parse_detections(output, conf=self.conf, nms=self.nms)
