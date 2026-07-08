# alert.py (知らせ方) のテスト
import random

import pytest

from app.alert import message_for


# ---------- 正常系 ----------

def test_レベル0は安全で音なし():
    info = message_for(0)
    assert info["name"] == "安全"
    assert info["sound"] is False
    assert info["level"] == 0


def test_レベル1は注意で音なし():
    info = message_for(1)
    assert info["name"] == "注意"
    assert info["sound"] is False


def test_レベル2は危険で音あり():
    info = message_for(2)
    assert info["name"] == "危険"
    assert info["sound"] is True


@pytest.mark.parametrize("level", [0, 1, 2])
def test_必要な項目が全部そろっている(level):
    info = message_for(level)
    for key in ("level", "name", "text", "color", "sound"):
        assert key in info
    # 色は (赤, 緑, 青, 濃さ) の4つで、どれも0〜1
    assert len(info["color"]) == 4
    for c in info["color"]:
        assert 0 <= c <= 1


def test_返した辞書を書きかえても元は変わらない():
    info = message_for(0)
    info["name"] = "書きかえた"
    assert message_for(0)["name"] == "安全"


# ---------- 異常系・境界値 ----------

@pytest.mark.parametrize("bad", [-1, 3, 100, 2.0, "2", None, True, False])
def test_おかしなレベルでは止まる(bad):
    with pytest.raises(ValueError):
        message_for(bad)


# ---------- ランダムテスト ----------

def test_でたらめな整数を入れても正しく動くか止まるかのどちらか():
    rng = random.Random(5)
    for _ in range(200):
        level = rng.randint(-10, 10)
        if level in (0, 1, 2):
            assert message_for(level)["level"] == level
        else:
            with pytest.raises(ValueError):
                message_for(level)
