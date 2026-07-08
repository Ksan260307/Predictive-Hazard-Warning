# objectdetect.py (AIモデルによる物体認識) のテスト
#
# モデルファイルが無くてもテストできるように、
# 出力の解析 (parse_detections) は合成データで確かめる。
import os

import numpy as np
import pytest

from app import objectdetect
from app.objectdetect import ObjectFinder, model_available, parse_detections


def fake_output(detections, n_candidates=100, n_classes=80, input_size=640):
    """YOLOv8のONNX出力 (1, 4+クラス数, 候補数) を合成する。

    detections: [(cx, cy, w, h, class_id, score), ...] (座標はピクセル)
    """
    arr = np.zeros((1, 4 + n_classes, n_candidates), dtype=np.float32)
    for i, (cx, cy, w, h, class_id, score) in enumerate(detections):
        arr[0, 0, i] = cx
        arr[0, 1, i] = cy
        arr[0, 2, i] = w
        arr[0, 3, i] = h
        arr[0, 4 + class_id, i] = score
    return arr


# ---------- 出力の解析 ----------

def test_歩行者を1人見つけられる():
    output = fake_output([(320, 320, 64, 128, 0, 0.9)])  # class 0 = person
    things = parse_detections(output)
    assert len(things) == 1
    t = things[0]
    assert t["label"] == "person"
    assert t["score"] == pytest.approx(0.9)
    assert t["cx"] == pytest.approx(0.5)
    assert t["cy"] == pytest.approx(0.5)
    assert t["w"] == pytest.approx(64 / 640)
    assert t["h"] == pytest.approx(128 / 640)
    assert t["vx"] == 0.0 and t["vy"] == 0.0


def test_検出結果の形式はdetectorと同じ():
    output = fake_output([(320, 320, 64, 128, 2, 0.8)])
    things = parse_detections(output)
    for key in ("x", "y", "w", "h", "cx", "cy", "vx", "vy"):
        assert key in things[0]


def test_確信度の低い候補は捨てられる():
    output = fake_output([(320, 320, 64, 128, 0, 0.2)])
    assert parse_detections(output, conf=0.35) == []


def test_交通に関係ないクラスは捨てられる():
    # class 10 = fire hydrant (消火栓): 対象外
    output = fake_output([(320, 320, 64, 128, 10, 0.9)])
    assert parse_detections(output) == []


def test_同じ物への重複した枠はまとめられる():
    output = fake_output([
        (320, 320, 64, 128, 0, 0.9),
        (322, 321, 64, 128, 0, 0.8),  # ほぼ同じ場所
    ])
    things = parse_detections(output)
    assert len(things) == 1
    assert things[0]["score"] == pytest.approx(0.9)  # 高い方が残る


def test_離れた2つの物は両方残る():
    output = fake_output([
        (100, 320, 64, 128, 0, 0.9),
        (500, 320, 64, 128, 2, 0.8),
    ])
    things = parse_detections(output)
    assert len(things) == 2
    labels = {t["label"] for t in things}
    assert labels == {"person", "car"}


def test_画面からはみ出た枠は切りそろえられる():
    output = fake_output([(10, 10, 100, 100, 0, 0.9)])  # 左上にはみ出す
    things = parse_detections(output)
    assert len(things) == 1
    assert things[0]["x"] >= 0.0
    assert things[0]["y"] >= 0.0


def test_転置された出力でも解析できる():
    output = fake_output([(320, 320, 64, 128, 0, 0.9)])
    transposed = output[0].T[np.newaxis, :, :]  # (1, 候補数, 4+クラス数)
    things = parse_detections(transposed)
    assert len(things) == 1
    assert things[0]["label"] == "person"


def test_何も見つからなければ空のリスト():
    output = fake_output([])
    assert parse_detections(output) == []


def test_おかしな出力は止まる():
    with pytest.raises(ValueError):
        parse_detections(np.zeros((3,)))  # 次元が足りない
    with pytest.raises(ValueError):
        parse_detections(np.zeros((1, 2, 10)))  # 4+クラス数に満たない
    with pytest.raises(ValueError):
        parse_detections(fake_output([]), conf=0)
    with pytest.raises(ValueError):
        parse_detections(fake_output([]), conf=1)
    with pytest.raises(ValueError):
        parse_detections(fake_output([]), nms=-0.1)


# ---------- モデルファイルの扱い ----------

def test_モデルが無い時はmodel_availableがFalse():
    assert model_available("存在しないファイル.onnx") is False
    assert model_available(None) is False


def test_モデルが無い時にObjectFinderを作ると止まる():
    with pytest.raises(ValueError):
        ObjectFinder(path="存在しないファイル.onnx")


def test_壊れたモデルファイルは読み込みで止まる(tmp_path):
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"this is not a model")
    with pytest.raises(ValueError):
        ObjectFinder(path=str(broken))


def test_おかしな設定は止まる(tmp_path):
    with pytest.raises(ValueError):
        ObjectFinder(path="x.onnx", conf=0)
    with pytest.raises(ValueError):
        ObjectFinder(path="x.onnx", nms=1.5)


# ---------- 実モデルでの動作 (モデルがある環境のみ) ----------

needs_model = pytest.mark.skipif(
    not (model_available() and objectdetect.runtime_available()),
    reason="data/yolov8n.onnx か onnxruntime が無い環境ではスキップ")


@needs_model
def test_実モデルで画像1枚を処理できる():
    finder = ObjectFinder()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    things = finder.find(frame)  # 真っ黒な画像なので何も見つからないはず
    assert isinstance(things, list)


@needs_model
def test_実モデルは白黒画像も受け付ける():
    finder = ObjectFinder()
    frame = np.zeros((480, 640), dtype=np.uint8)
    assert isinstance(finder.find(frame), list)


@needs_model
def test_実モデルでもおかしな画像は止まる():
    finder = ObjectFinder()
    with pytest.raises(ValueError):
        finder.find(None)
    with pytest.raises(ValueError):
        finder.find(np.zeros((2, 2, 5), dtype=np.uint8))
