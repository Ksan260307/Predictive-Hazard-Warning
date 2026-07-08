# 未来を予測するファイル
#
# 「未来の分布」を作って持ち続ける部分です。
#
# ・make_futures : いま見えている物から「ありえる未来」をたくさん作る
# ・FutureBelief : 作った未来をため込み、少しずつ入れ替える
#                  (新しい分布 = (1-β)×古い分布 + β×新しい予測)

import math
import random

# 未来1通りごとの位置のはみ出しを抑える範囲
CLIP_LOW = -0.5
CLIP_HIGH = 1.5

# 物の動きに加えるゆらぎの標準的な大きさ (1歩あたり)
DEFAULT_WOBBLE = 0.015


def make_futures(things, steps, samples, wobble=DEFAULT_WOBBLE, seed=None):
    """いま見えている物から、ありえる未来をたくさん作る。

    things  : detector が見つけた物のリスト (cx, cy, vx, vy が必要)
    steps   : 何歩先まで予測するか
    samples : 未来を何通り作るか
    wobble  : 動きに加えるゆらぎの大きさ
    seed    : 乱数の種 (テストで結果を固定したい時に使う)

    返り値: 未来のリスト。未来1つは
        {"paths": [物ごとの位置の列], "weight": 起こりやすさ}
    weight は全部足すと1になるように調整して返す。
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("何歩先まで見るか(steps)は1以上の整数にしてください")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("未来の数(samples)は1以上の整数にしてください")
    if wobble < 0:
        raise ValueError("ゆらぎ(wobble)は0以上にしてください")
    for t in things:
        for key in ("cx", "cy", "vx", "vy"):
            if key not in t:
                raise ValueError("物のデータに " + key + " がありません")

    if not things:
        return []  # 何も見えていなければ未来も作らない

    rng = random.Random(seed)
    futures = []
    for _ in range(samples):
        paths = []
        noise_total = 0.0
        for t in things:
            # 動きに少しゆらぎを加える(未来は1通りに決まらないため)
            nx = rng.gauss(0, wobble)
            ny = rng.gauss(0, wobble)
            noise_total += nx * nx + ny * ny

            cx, cy = t["cx"], t["cy"]
            vx, vy = t["vx"] + nx, t["vy"] + ny
            path = []
            for _ in range(steps):
                cx = max(CLIP_LOW, min(CLIP_HIGH, cx + vx))
                cy = max(CLIP_LOW, min(CLIP_HIGH, cy + vy))
                path.append((cx, cy))
            paths.append(path)

        # ゆらぎが小さい未来ほど「起こりやすい」とみなす
        if wobble > 0:
            weight = math.exp(-noise_total / (2 * wobble * wobble))
        else:
            weight = 1.0
        futures.append({"paths": paths, "weight": weight})

    # 全部足して1になるようにする (確率としてあつかうため)
    total = sum(f["weight"] for f in futures)
    if total <= 0:
        for f in futures:
            f["weight"] = 1.0 / len(futures)
    else:
        for f in futures:
            f["weight"] /= total
    return futures


def make_phantom_futures(blind_spots, steps, samples, strength, seed=None):
    """死角から人や車が「飛び出してくるかもしれない未来」を作る。

    まだ見えていないが起こりうる未来を、分布に混ぜるための関数。
    観測する前から「起こりうること」を警戒に使えるようにする。

    blind_spots : scene.py が見つけた死角のリスト ({"x","y","strength"})
    steps       : 何歩先まで予測するか
    samples     : 死角1つあたり何通りの飛び出しを考えるか
    strength    : この未来たちの重みの合計 (0〜1)。
                  実際に見えている物より軽く扱うための上限。

    返り値: 未来のリスト。重みの合計は strength になる。
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("何歩先まで見るか(steps)は1以上の整数にしてください")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("未来の数(samples)は1以上の整数にしてください")
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        raise ValueError("重み(strength)は数にしてください")
    if not 0 <= strength <= 1:
        raise ValueError("重み(strength)は0〜1にしてください")
    for spot in blind_spots:
        for key in ("x", "y", "strength"):
            if key not in spot:
                raise ValueError("死角のデータに " + key + " がありません")

    if not blind_spots or strength <= 0:
        return []

    rng = random.Random(seed)
    futures = []
    for spot in blind_spots:
        # 死角の位置から、進路の中央へ向かって出てくると考える
        direction = 1.0 if spot["x"] < 0.5 else -1.0
        for _ in range(samples):
            cx, cy = spot["x"], spot["y"]
            vx = direction * rng.uniform(0.015, 0.06)  # 横へ飛び出す速さ
            vy = rng.uniform(0.0, 0.03)                # こちらへ近づく速さ
            path = []
            for _ in range(steps):
                cx = max(CLIP_LOW, min(CLIP_HIGH, cx + vx))
                cy = max(CLIP_LOW, min(CLIP_HIGH, cy + vy))
                path.append((cx, cy))
            futures.append({"paths": [path], "weight": spot["strength"]})

    # 重みの合計がちょうど strength になるようにそろえる
    total = sum(f["weight"] for f in futures)
    if total <= 0:
        return []
    for f in futures:
        f["weight"] = f["weight"] / total * strength
    return futures


class FutureBelief:
    """未来の予測をため込んでおく入れ物。

    毎フレーム update() を呼ぶと、
    古い予測を薄めて、新しい予測を混ぜる。
        新しい分布 = (1 - rate) × 古い分布 + rate × 新しい予測

    新しい予測が空(何も見えない)の時は、古い予測がだんだん薄れて消える。
    これが「リスクの連続的な減衰」になる。
    """

    def __init__(self, keep_max=300, eps=1e-4):
        if not isinstance(keep_max, int) or isinstance(keep_max, bool) or keep_max < 1:
            raise ValueError("ため込む数(keep_max)は1以上の整数にしてください")
        if eps < 0:
            raise ValueError("切り捨てライン(eps)は0以上にしてください")
        self.keep_max = keep_max  # ため込む未来の上限 (計算量を抑えるため)
        self.eps = eps            # これより薄い未来は捨てる
        self.samples = []

    @property
    def total_weight(self):
        """残っている未来の重みの合計 (0〜1)。0に近いほど「何も見えていない」。"""
        return sum(f["weight"] for f in self.samples)

    def update(self, new_futures, rate):
        """古い予測を薄めて、新しい予測を混ぜる。

        rate : 入れ替わりの速さ。0より大きく1以下。
        """
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError("更新率(rate)は数にしてください")
        if not 0 < rate <= 1:
            raise ValueError("更新率(rate)は0より大きく1以下にしてください")

        merged = [
            {"paths": f["paths"], "weight": f["weight"] * (1 - rate)}
            for f in self.samples
        ]
        merged += [
            {"paths": f["paths"], "weight": f["weight"] * rate}
            for f in new_futures
        ]

        # 薄すぎる未来は捨てる
        merged = [f for f in merged if f["weight"] >= self.eps]
        # 濃い順に並べて、上限までしか持たない (計算量を抑えるため)
        merged.sort(key=lambda f: f["weight"], reverse=True)
        self.samples = merged[: self.keep_max]

    def clear(self):
        """ため込んだ予測を全部忘れる。"""
        self.samples = []
