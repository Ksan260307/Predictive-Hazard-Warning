# settings.py (設定) のテスト
import json
import random

import pytest

from app import settings as s


# ---------- 正常系 ----------

def test_初期値がそのまま使える():
    fixed = s.check_settings(dict(s.DEFAULTS))
    assert fixed == s.DEFAULTS


def test_正しい値はそのまま残る():
    fixed = s.check_settings({"sensitivity": 8, "update_rate": 0.3, "sound_on": False})
    assert fixed["sensitivity"] == 8
    assert fixed["update_rate"] == 0.3
    assert fixed["sound_on"] is False


def test_保存して読み込むと同じ値になる(tmp_path):
    path = tmp_path / "settei.json"
    values = dict(s.DEFAULTS)
    values["sensitivity"] = 9
    values["sound_on"] = False
    s.save_settings(values, path)
    loaded = s.load_settings(path)
    assert loaded["sensitivity"] == 9
    assert loaded["sound_on"] is False


def test_保存したファイルは日本語でも読めるJSONになっている(tmp_path):
    path = tmp_path / "settei.json"
    s.save_settings(s.DEFAULTS, path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


# ---------- 異常系 ----------

def test_ファイルが無い時は初期値になる(tmp_path):
    loaded = s.load_settings(tmp_path / "nai.json")
    assert loaded == s.DEFAULTS


def test_壊れたファイルでも初期値で動く(tmp_path):
    path = tmp_path / "kowareta.json"
    path.write_text("{これはJSONではない", encoding="utf-8")
    assert s.load_settings(path) == s.DEFAULTS


def test_中身がリストのファイルでも初期値で動く(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert s.load_settings(path) == s.DEFAULTS


def test_辞書でないものを渡しても初期値になる():
    assert s.check_settings(None) == s.DEFAULTS
    assert s.check_settings("もじ") == s.DEFAULTS
    assert s.check_settings(123) == s.DEFAULTS


def test_数に直せない値は初期値に戻る():
    fixed = s.check_settings({"sensitivity": "つよい", "update_rate": None})
    assert fixed["sensitivity"] == s.DEFAULTS["sensitivity"]
    assert fixed["update_rate"] == s.DEFAULTS["update_rate"]


def test_TrueFalseを数の項目に入れても捨てられる():
    fixed = s.check_settings({"sensitivity": True})
    assert fixed["sensitivity"] == s.DEFAULTS["sensitivity"]


def test_知らない項目は捨てられる():
    fixed = s.check_settings({"nazo_no_koumoku": 999})
    assert "nazo_no_koumoku" not in fixed


def test_はいいいえの項目に文字を入れても初期値のまま():
    fixed = s.check_settings({"sound_on": "はい"})
    assert fixed["sound_on"] == s.DEFAULTS["sound_on"]


# ---------- 境界値 ----------

@pytest.mark.parametrize("key", list(s.RANGES))
def test_範囲のいちばん端の値は使える(key):
    low, high = s.RANGES[key]
    assert s.check_settings({key: low})[key] == low
    assert s.check_settings({key: high})[key] == high


@pytest.mark.parametrize("key", list(s.RANGES))
def test_範囲の外の値は端に収められる(key):
    low, high = s.RANGES[key]
    assert s.check_settings({key: low - 1})[key] == low
    assert s.check_settings({key: high + 1})[key] == high


def test_更新率0はいちばん小さい値に収められる():
    # β=0 だと未来が全く入れ替わらないので、0は許さない設計
    assert s.check_settings({"update_rate": 0})["update_rate"] == 0.01


def test_更新率1はそのまま使える():
    assert s.check_settings({"update_rate": 1.0})["update_rate"] == 1.0


# ---------- モード・文字の設定 ----------

@pytest.mark.parametrize("mode", ["walk", "bicycle", "car"])
def test_正しいモードはそのまま使える(mode):
    assert s.check_settings({"mode": mode})["mode"] == mode


@pytest.mark.parametrize("bad", ["airplane", "", None, 5, True, [], "WALK"])
def test_おかしなモードは初期値に戻る(bad):
    assert s.check_settings({"mode": bad})["mode"] == s.DEFAULTS["mode"]


def test_昔のAPIキー設定が残っていても捨てられる():
    # 以前のバージョンの設定ファイルに api_key が残っていても壊れない
    fixed = s.check_settings({"api_key": "old_key", "sensitivity": 7})
    assert "api_key" not in fixed
    assert fixed["sensitivity"] == 7


def test_新しいスイッチ項目もTrueFalseで扱える():
    fixed = s.check_settings({"stabilize": False, "use_location": True,
                              "learning_on": False, "voice_on": True,
                              "save_log": True, "power_save": False,
                              "use_ml_detector": False})
    assert fixed["stabilize"] is False
    assert fixed["use_location"] is True
    assert fixed["learning_on"] is False
    assert fixed["voice_on"] is True
    assert fixed["save_log"] is True
    assert fixed["power_save"] is False
    assert fixed["use_ml_detector"] is False


def test_追加された設定の初期値():
    d = s.DEFAULTS
    assert d["use_ml_detector"] is True   # モデルが無ければ自動で動き検出
    assert d["power_save"] is True        # 省電力は既定でオン
    assert d["save_log"] is False         # ログは明示的にオンにする
    assert d["voice_on"] is False         # 読み上げも明示的にオンにする


# ---------- ランダムテスト ----------

def test_でたらめな値を入れても必ず使える設定になる():
    rng = random.Random(42)
    garbage_makers = [
        lambda: rng.uniform(-1e6, 1e6),
        lambda: rng.randint(-1000, 1000),
        lambda: rng.choice(["a", "", None, [], {}, float("nan"), float("inf")]),
        lambda: rng.choice([True, False]),
    ]
    for _ in range(300):
        values = {key: rng.choice(garbage_makers)() for key in s.DEFAULTS}
        fixed = s.check_settings(values)
        # 必ず全項目そろっている
        assert set(fixed) == set(s.DEFAULTS)
        # 数の項目は必ず範囲の中
        for key, (low, high) in s.RANGES.items():
            assert low <= fixed[key] <= high, f"{key} が範囲の外: {fixed[key]}"
        # はい/いいえ の項目は必ず True/False
        for key in s.BOOL_KEYS:
            assert isinstance(fixed[key], bool)
        # 選択の項目は必ず候補の中
        for key, choices in s.CHOICE_KEYS.items():
            assert fixed[key] in choices
        # 文字の項目は必ず文字
        for key in s.TEXT_KEYS:
            assert isinstance(fixed[key], str)
