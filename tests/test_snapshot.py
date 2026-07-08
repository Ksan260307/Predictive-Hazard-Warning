# snapshot.py (報告場面の画像の保存) のテスト
import os

import numpy as np
import pytest

from app import imgproc
from app.snapshot import SnapshotKeeper


def frame(shade=128, width=64, height=48):
    return np.full((height, width, 3), shade, dtype=np.uint8)


# ---------- 保存の基本 ----------

def test_報告するとPNGが保存される(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    for i in range(10):
        keeper.add(frame(shade=i * 20))
    paths = keeper.save("missed", t=100.0)
    assert len(paths) == 3
    for path in paths:
        assert os.path.isfile(path)
        assert os.path.basename(path).startswith("missed_")
        # 保存した画像はPNGとして読み戻せる
        with open(path, "rb") as f:
            image = imgproc.decode_png(f.read())
        assert image is not None
        assert image.shape == (48, 64, 3)


def test_古い場面から新しい場面まで等間隔に選ばれる(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    for i in range(45):
        keeper.add(frame(shade=min(255, i * 5)))
    paths = keeper.save("false_alarm", t=100.0)
    shades = []
    for path in sorted(paths):
        with open(path, "rb") as f:
            image = imgproc.decode_png(f.read())
        shades.append(int(image.mean()))
    # 一番古い場面(暗い)と一番新しい場面(明るい)の両方が入っている
    assert shades[0] < 30
    assert shades[-1] > 190


def test_何も覚えていなければ何も保存しない(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    assert keeper.save("missed") == []
    assert not (tmp_path / "snaps").exists()


def test_1枚しか覚えていなくても保存できる(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    keeper.add(frame())
    assert len(keeper.save("missed", t=100.0)) == 1


def test_覚えるフレームは上限を超えない(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"), keep=5)
    for i in range(100):
        keeper.add(frame())
    assert len(keeper._frames) == 5


def test_白黒画像も保存できる(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    keeper.add(np.full((48, 64), 100, dtype=np.uint8))
    assert len(keeper.save("missed", t=100.0)) == 1


def test_クリアで忘れる(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    keeper.add(frame())
    keeper.clear()
    assert keeper.save("missed") == []


# ---------- ファイルの管理 ----------

def test_画像が増えすぎたら古い物から消される(tmp_path):
    folder = tmp_path / "snaps"
    keeper = SnapshotKeeper(str(folder), max_files=6)
    keeper.add(frame())
    for i in range(5):
        keeper.save("missed", t=100.0 + i)
    files = [name for name in os.listdir(folder) if name.endswith(".png")]
    assert len(files) <= 6


def test_2回続けて報告しても別の名前で保存される(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    keeper.add(frame())
    first = keeper.save("missed", t=100.0)
    second = keeper.save("missed", t=100.5)
    assert first != second
    assert all(os.path.isfile(p) for p in first + second)


# ---------- 異常系 ----------

def test_おかしな設定は止まる(tmp_path):
    with pytest.raises(ValueError):
        SnapshotKeeper("")
    with pytest.raises(ValueError):
        SnapshotKeeper(None)
    with pytest.raises(ValueError):
        SnapshotKeeper(str(tmp_path), keep=0)
    with pytest.raises(ValueError):
        SnapshotKeeper(str(tmp_path), save_count=0)
    with pytest.raises(ValueError):
        SnapshotKeeper(str(tmp_path), save_count=5, max_files=3)


def test_おかしな画像や名前は止まる(tmp_path):
    keeper = SnapshotKeeper(str(tmp_path / "snaps"))
    with pytest.raises(ValueError):
        keeper.add(None)
    with pytest.raises(ValueError):
        keeper.add("画像ではない")
    with pytest.raises(ValueError):
        keeper.add(np.zeros((0, 0, 3), dtype=np.uint8))
    keeper.add(frame())
    with pytest.raises(ValueError):
        keeper.save("")
    with pytest.raises(ValueError):
        keeper.save(123)
