# triplog.py (走行ログ) のテスト
import json

import pytest

from app.triplog import TripLogger


def result(level=0, risk=0.0, **extra):
    """テスト用の watch() の結果。"""
    data = {
        "level": level, "risk": risk,
        "image_risk": risk, "env_risk": 0.0,
        "detector": "motion", "reasons": [],
        "things": [], "road": {"speed": None},
    }
    data.update(extra)
    return data


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- 状態の記録 ----------

def test_状態が1行のJSONで記録される(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path)
    assert logger.log(result(level=1, risk=0.5), t=100.0) is True
    lines = read_lines(path)
    assert len(lines) == 1
    record = lines[0]
    assert record["kind"] == "state"
    assert record["t"] == 100.0
    assert record["level"] == 1
    assert record["risk"] == 0.5
    assert record["detector"] == "motion"
    assert record["things"] == 0


def test_間隔が空くまでは記録しない(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path, interval=1.0)
    assert logger.log(result(), t=100.0) is True
    assert logger.log(result(), t=100.5) is False  # まだ1秒経っていない
    assert logger.log(result(), t=101.0) is True
    assert len(read_lines(path)) == 2


def test_速度も記録される(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path)
    logger.log(result(road={"speed": 5.0}), t=100.0)
    assert read_lines(path)[0]["speed"] == 5.0


# ---------- 報告の記録 ----------

def test_報告は間引かずに必ず記録される(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path)
    logger.log_feedback("false_alarm", t=100.0)
    logger.log_feedback("missed", t=100.1)  # 直後でも記録される
    lines = read_lines(path)
    assert len(lines) == 2
    assert lines[0]["kind"] == "feedback"
    assert lines[0]["feedback"] == "false_alarm"
    assert lines[1]["feedback"] == "missed"


def test_報告に場面の画像を添えられる(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path)
    logger.log_feedback("missed", t=100.0,
                        snapshots=["snaps/a.jpg", "snaps/b.jpg"])
    record = read_lines(path)[0]
    assert record["snapshots"] == ["snaps/a.jpg", "snaps/b.jpg"]


def test_画像が無い報告にはsnapshots欄が付かない(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path)
    logger.log_feedback("missed", t=100.0, snapshots=[])
    assert "snapshots" not in read_lines(path)[0]


# ---------- ファイルの管理 ----------

def test_大きくなりすぎたログは退避される(tmp_path):
    path = str(tmp_path / "log.jsonl")
    logger = TripLogger(path, interval=0.0, max_bytes=1024)
    for i in range(200):
        logger.log(result(), t=float(i))
    # 上限を超えた分は .old に移り、本体は上限より小さい
    assert (tmp_path / "log.jsonl.old").exists()
    lines = read_lines(path)
    assert len(lines) >= 1


def test_書き込みに失敗しても止まらない(tmp_path):
    logger = TripLogger(str(tmp_path / "無い場所" / "log.jsonl"))
    logger.log(result(), t=100.0)  # フォルダが無くてもエラーにならない
    logger.log_feedback("missed")


# ---------- 異常系 ----------

def test_おかしな設定は止まる(tmp_path):
    with pytest.raises(ValueError):
        TripLogger("")
    with pytest.raises(ValueError):
        TripLogger(None)
    with pytest.raises(ValueError):
        TripLogger(str(tmp_path / "l.jsonl"), interval=-1)
    with pytest.raises(ValueError):
        TripLogger(str(tmp_path / "l.jsonl"), max_bytes=10)


def test_おかしな結果は止まる(tmp_path):
    logger = TripLogger(str(tmp_path / "l.jsonl"))
    with pytest.raises(ValueError):
        logger.log("結果ではない")
    with pytest.raises(ValueError):
        logger.log({"level": 0})  # risk が無い
    with pytest.raises(ValueError):
        logger.log_feedback("")
    with pytest.raises(ValueError):
        logger.log_feedback(123)
