# future.py (未来の予測) のテスト
import random

import pytest

from app.future import (
    CLIP_HIGH,
    CLIP_LOW,
    FutureBelief,
    make_futures,
    make_phantom_futures,
)


def one_thing(cx=0.5, cy=0.5, vx=0.0, vy=0.0):
    return {"cx": cx, "cy": cy, "vx": vx, "vy": vy}


# ---------- make_futures 正常系 ----------

def test_たのんだ数だけ未来ができる():
    futures = make_futures([one_thing()], steps=10, samples=20, seed=1)
    assert len(futures) == 20


def test_重みは全部足すと1になる():
    futures = make_futures([one_thing()], steps=10, samples=20, seed=1)
    assert sum(f["weight"] for f in futures) == pytest.approx(1.0)


def test_道のりの長さはsteps通りになる():
    futures = make_futures([one_thing()], steps=7, samples=3, seed=1)
    for f in futures:
        for path in f["paths"]:
            assert len(path) == 7


def test_物が2つなら道のりも2本できる():
    things = [one_thing(0.2, 0.2), one_thing(0.8, 0.8)]
    futures = make_futures(things, steps=5, samples=4, seed=1)
    for f in futures:
        assert len(f["paths"]) == 2


def test_ゆらぎ0ならまっすぐ進む():
    futures = make_futures([one_thing(0.5, 0.5, vx=0.1, vy=0.0)],
                           steps=3, samples=2, wobble=0.0)
    for f in futures:
        path = f["paths"][0]
        assert path[0] == pytest.approx((0.6, 0.5))
        assert path[1] == pytest.approx((0.7, 0.5))
        assert path[2] == pytest.approx((0.8, 0.5))


def test_同じ種なら同じ未来になる():
    a = make_futures([one_thing()], steps=5, samples=5, seed=42)
    b = make_futures([one_thing()], steps=5, samples=5, seed=42)
    assert a == b


def test_物がなければ未来もない():
    assert make_futures([], steps=10, samples=10) == []


# ---------- make_futures 異常系 ----------

def test_stepsが0以下なら止まる():
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps=0, samples=10)
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps=-1, samples=10)


def test_samplesが0以下なら止まる():
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps=10, samples=0)


def test_stepsに小数や文字を渡すと止まる():
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps=1.5, samples=10)
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps="10", samples=10)


def test_ゆらぎがマイナスなら止まる():
    with pytest.raises(ValueError):
        make_futures([one_thing()], steps=10, samples=10, wobble=-0.1)


def test_物のデータに足りない項目があれば止まる():
    with pytest.raises(ValueError):
        make_futures([{"cx": 0.5, "cy": 0.5}], steps=10, samples=10)


# ---------- make_futures 境界値 ----------

def test_steps1とsamples1でも動く():
    futures = make_futures([one_thing()], steps=1, samples=1, seed=1)
    assert len(futures) == 1
    assert futures[0]["weight"] == pytest.approx(1.0)


def test_速すぎる物も画面のまわりで止まる():
    # とんでもない速さでも、位置がはみ出しすぎない(クリッピング)
    futures = make_futures([one_thing(0.5, 0.5, vx=100.0, vy=-100.0)],
                           steps=10, samples=3, seed=1)
    for f in futures:
        for path in f["paths"]:
            for cx, cy in path:
                assert CLIP_LOW <= cx <= CLIP_HIGH
                assert CLIP_LOW <= cy <= CLIP_HIGH


# ---------- make_phantom_futures (死角からの飛び出し予測) ----------

def one_spot(x=0.3, y=0.6, strength=0.8):
    return {"x": x, "y": y, "strength": strength}


def test_死角から飛び出しの未来ができる():
    phantoms = make_phantom_futures([one_spot()], steps=10, samples=5,
                                    strength=0.4, seed=1)
    assert len(phantoms) == 5


def test_飛び出しの重みの合計はstrengthになる():
    phantoms = make_phantom_futures([one_spot()], steps=10, samples=8,
                                    strength=0.4, seed=1)
    assert sum(f["weight"] for f in phantoms) == pytest.approx(0.4)


def test_左の死角からは右へ飛び出してくる():
    phantoms = make_phantom_futures([one_spot(x=0.2)], steps=5, samples=5,
                                    strength=0.5, seed=1)
    for f in phantoms:
        path = f["paths"][0]
        assert path[-1][0] > 0.2  # 進路の中央(右)へ向かう


def test_右の死角からは左へ飛び出してくる():
    phantoms = make_phantom_futures([one_spot(x=0.8)], steps=5, samples=5,
                                    strength=0.5, seed=1)
    for f in phantoms:
        path = f["paths"][0]
        assert path[-1][0] < 0.8


def test_死角がなければ飛び出しもない():
    assert make_phantom_futures([], steps=10, samples=5, strength=0.5) == []


def test_strength0なら飛び出しもない():
    assert make_phantom_futures([one_spot()], steps=10, samples=5, strength=0) == []


def test_強い死角ほど重い飛び出しになる():
    spots = [one_spot(x=0.2, strength=1.0), one_spot(x=0.8, strength=0.25)]
    phantoms = make_phantom_futures(spots, steps=5, samples=3,
                                    strength=0.4, seed=1)
    strong = sum(f["weight"] for f in phantoms[:3])   # 最初の死角の分
    weak = sum(f["weight"] for f in phantoms[3:])     # 2つ目の死角の分
    assert strong > weak


def test_飛び出しのおかしな引数は止まる():
    with pytest.raises(ValueError):
        make_phantom_futures([one_spot()], steps=0, samples=5, strength=0.5)
    with pytest.raises(ValueError):
        make_phantom_futures([one_spot()], steps=5, samples=0, strength=0.5)
    with pytest.raises(ValueError):
        make_phantom_futures([one_spot()], steps=5, samples=5, strength=1.5)
    with pytest.raises(ValueError):
        make_phantom_futures([one_spot()], steps=5, samples=5, strength=-0.1)
    with pytest.raises(ValueError):
        make_phantom_futures([{"x": 0.3}], steps=5, samples=5, strength=0.5)


def test_飛び出しの位置もはみ出さない():
    phantoms = make_phantom_futures([one_spot()], steps=50, samples=10,
                                    strength=1.0, seed=2)
    for f in phantoms:
        for path in f["paths"]:
            for cx, cy in path:
                assert CLIP_LOW <= cx <= CLIP_HIGH
                assert CLIP_LOW <= cy <= CLIP_HIGH


def test_でたらめな死角でも飛び出しはまともな形になる():
    rng = random.Random(21)
    for i in range(100):
        spots = [one_spot(rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.01, 1))
                 for _ in range(rng.randint(1, 4))]
        strength = rng.uniform(0, 1)
        phantoms = make_phantom_futures(spots, steps=rng.randint(1, 20),
                                        samples=rng.randint(1, 10),
                                        strength=strength, seed=i)
        total = sum(f["weight"] for f in phantoms)
        if strength > 0:
            assert total == pytest.approx(strength)
        for f in phantoms:
            assert f["weight"] >= 0


# ---------- FutureBelief ----------

def test_更新すると未来がたまる():
    belief = FutureBelief()
    futures = make_futures([one_thing()], steps=5, samples=10, seed=1)
    belief.update(futures, rate=0.5)
    assert len(belief.samples) > 0
    assert belief.total_weight == pytest.approx(0.5)


def test_更新率1なら古い未来は全部消える():
    belief = FutureBelief()
    old = make_futures([one_thing(0.1, 0.1)], steps=5, samples=5, seed=1)
    new = make_futures([one_thing(0.9, 0.9)], steps=5, samples=5, seed=2)
    belief.update(old, rate=1.0)
    belief.update(new, rate=1.0)
    # 古い未来(0.1あたりから始まる道のり)は残っていない
    for f in belief.samples:
        assert f["paths"][0][0][0] > 0.5


def test_何も見えないと予測はだんだん薄れる():
    belief = FutureBelief()
    futures = make_futures([one_thing()], steps=5, samples=10, seed=1)
    belief.update(futures, rate=1.0)
    weight_before = belief.total_weight
    belief.update([], rate=0.5)
    belief.update([], rate=0.5)
    assert belief.total_weight < weight_before


def test_薄れきった予測はやがて消える():
    belief = FutureBelief()
    futures = make_futures([one_thing()], steps=5, samples=10, seed=1)
    belief.update(futures, rate=1.0)
    for _ in range(50):
        belief.update([], rate=0.5)
    assert belief.samples == []


def test_ため込む数に上限がある():
    belief = FutureBelief(keep_max=10)
    for i in range(10):
        futures = make_futures([one_thing()], steps=5, samples=30, seed=i)
        belief.update(futures, rate=0.3)
    assert len(belief.samples) <= 10


def test_clearで全部忘れる():
    belief = FutureBelief()
    belief.update(make_futures([one_thing()], steps=5, samples=5, seed=1), rate=0.5)
    belief.clear()
    assert belief.samples == []
    assert belief.total_weight == 0


# ---------- FutureBelief 異常系・境界値 ----------

def test_更新率0や0より小さい値では止まる():
    belief = FutureBelief()
    with pytest.raises(ValueError):
        belief.update([], rate=0)
    with pytest.raises(ValueError):
        belief.update([], rate=-0.5)


def test_更新率が1より大きいと止まる():
    belief = FutureBelief()
    with pytest.raises(ValueError):
        belief.update([], rate=1.5)


def test_更新率に文字やTrueを渡すと止まる():
    belief = FutureBelief()
    with pytest.raises(ValueError):
        belief.update([], rate="0.5")
    with pytest.raises(ValueError):
        belief.update([], rate=True)


def test_更新率1はぎりぎり使える():
    belief = FutureBelief()
    belief.update(make_futures([one_thing()], steps=2, samples=2, seed=1), rate=1)
    assert belief.total_weight == pytest.approx(1.0)


def test_おかしな作り方は止まる():
    with pytest.raises(ValueError):
        FutureBelief(keep_max=0)
    with pytest.raises(ValueError):
        FutureBelief(eps=-1)


# ---------- ランダムテスト ----------

def test_でたらめな物でも未来はちゃんとした形になる():
    rng = random.Random(7)
    for i in range(100):
        things = [
            one_thing(rng.uniform(0, 1), rng.uniform(0, 1),
                      rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
            for _ in range(rng.randint(1, 5))
        ]
        steps = rng.randint(1, 30)
        samples = rng.randint(1, 50)
        futures = make_futures(things, steps=steps, samples=samples, seed=i)
        assert len(futures) == samples
        assert sum(f["weight"] for f in futures) == pytest.approx(1.0)
        for f in futures:
            assert f["weight"] >= 0
            assert len(f["paths"]) == len(things)
            for path in f["paths"]:
                assert len(path) == steps
                for cx, cy in path:
                    assert CLIP_LOW <= cx <= CLIP_HIGH
                    assert CLIP_LOW <= cy <= CLIP_HIGH


def test_でたらめに更新し続けても重みは1を超えない():
    rng = random.Random(8)
    belief = FutureBelief()
    for i in range(100):
        if rng.random() < 0.7:
            futures = make_futures([one_thing()], steps=3,
                                   samples=rng.randint(1, 20), seed=i)
        else:
            futures = []
        belief.update(futures, rate=rng.uniform(0.01, 1.0))
        assert belief.total_weight <= 1.0 + 1e-9
        assert len(belief.samples) <= belief.keep_max
