# power.py (省電力のための分析の間引き) のテスト
import pytest

from app.power import FrameThrottle, CALM_AFTER, SKIP_STRIDE


def calm_result():
    """「静か」な分析結果。"""
    return {"level": 0, "risk": 0.0, "things": []}


def busy_result():
    """何かが起きている分析結果。"""
    return {"level": 1, "risk": 0.5, "things": [{"cx": 0.5}]}


def run_frames(throttle, result, frames):
    """フレームを流して、分析した回数を返す。"""
    analyzed = 0
    for _ in range(frames):
        if throttle.should_analyze():
            throttle.note(result)
            analyzed += 1
    return analyzed


# ---------- 基本の動き ----------

def test_最初は毎フレーム分析する():
    throttle = FrameThrottle()
    assert run_frames(throttle, calm_result(), 10) == 10
    assert throttle.saving is False


def test_静かな時間が続くと間引きが始まる():
    throttle = FrameThrottle(calm_after=5, stride=3)
    run_frames(throttle, calm_result(), 5)
    assert throttle.saving is True
    # 間引き中は3フレームに1回だけ分析する
    analyzed = run_frames(throttle, calm_result(), 30)
    assert analyzed == pytest.approx(10, abs=1)


def test_何かが起きたらすぐ毎フレーム分析に戻る():
    throttle = FrameThrottle(calm_after=5, stride=3)
    run_frames(throttle, calm_result(), 10)
    assert throttle.saving is True
    # 物が見えた → 間引き解除
    throttle.note(busy_result())
    assert throttle.saving is False
    assert run_frames(throttle, busy_result(), 10) == 10


def test_危険が出ている間は間引かない():
    throttle = FrameThrottle(calm_after=3, stride=3)
    danger = {"level": 2, "risk": 0.9, "things": []}
    assert run_frames(throttle, danger, 20) == 20


def test_危険度が少しでも高ければ静かとみなさない():
    throttle = FrameThrottle(calm_after=3, stride=3)
    uneasy = {"level": 0, "risk": 0.3, "things": []}  # レベル0でも危険度あり
    assert run_frames(throttle, uneasy, 20) == 20


def test_オフにすると常に毎フレーム分析する():
    throttle = FrameThrottle(enabled=False, calm_after=3, stride=3)
    assert run_frames(throttle, calm_result(), 30) == 30
    assert throttle.saving is False


def test_リセットで間引きが解除される():
    throttle = FrameThrottle(calm_after=3, stride=3)
    run_frames(throttle, calm_result(), 10)
    assert throttle.saving is True
    throttle.reset()
    assert throttle.saving is False
    assert throttle.should_analyze() is True


def test_既定値でもいずれ間引きが始まる():
    throttle = FrameThrottle()
    run_frames(throttle, calm_result(), CALM_AFTER)
    assert throttle.saving is True
    analyzed = run_frames(throttle, calm_result(), SKIP_STRIDE * 10)
    assert analyzed == 10


# ---------- 異常系 ----------

def test_おかしな設定は止まる():
    with pytest.raises(ValueError):
        FrameThrottle(calm_after=0)
    with pytest.raises(ValueError):
        FrameThrottle(calm_after=1.5)
    with pytest.raises(ValueError):
        FrameThrottle(stride=0)
    with pytest.raises(ValueError):
        FrameThrottle(stride=True)


def test_おかしな結果は止まる():
    throttle = FrameThrottle()
    with pytest.raises(ValueError):
        throttle.note("結果ではない")
    with pytest.raises(ValueError):
        throttle.note({"level": 0})  # 項目が足りない
