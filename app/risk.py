# 危なさなどの数値を計算するファイル
#
# 未来の分布から
#   ・risk_score        : どのくらい危ないか
#   ・uncertainty_score : どのくらい迷っているか
#   ・optionality_score : どのくらい余裕があるか
# の3つを計算します。どれも 0〜1 の数で返します。

import math

# 「ぶつかる場所」の横方向の範囲 (画面の真ん中あたり)
CENTER_LEFT = 0.25
CENTER_RIGHT = 0.75


def risk_score(futures, danger_y=0.7, center_left=CENTER_LEFT, center_right=CENTER_RIGHT):
    """危なさを計算する。

    「ぶつかる場所」(画面の下の真ん中あたり) に入ってくる未来の重みを足す。
    つまり「ぶつかる未来がどのくらい起こりやすいか」を返す。

    futures  : FutureBelief.samples (未来のリスト)
    danger_y : 画面のどこから下を危ないとみなすか (0〜1)
    """
    if not 0 <= danger_y <= 1:
        raise ValueError("危険ライン(danger_y)は0〜1にしてください")

    total = 0.0
    for f in futures:
        hit = False
        for path in f["paths"]:
            for cx, cy in path:
                if cy >= danger_y and center_left <= cx <= center_right:
                    hit = True
                    break
            if hit:
                break
        if hit:
            total += f["weight"]
    # 万一の計算誤差でも0〜1からはみ出さないようにする
    return max(0.0, min(1.0, total))


def uncertainty_score(hit_chance):
    """迷いを計算する。

    「ぶつかるかどうか」の見通しがはっきりしているほど0に、
    半々で読めないほど1に近づく。
    (「ぶつかる/ぶつからない」の2択のエントロピーを、最大値で割ったもの)
    """
    if not isinstance(hit_chance, (int, float)) or isinstance(hit_chance, bool):
        raise ValueError("危なさ(hit_chance)は数にしてください")
    if not 0 <= hit_chance <= 1:
        raise ValueError("危なさ(hit_chance)は0〜1にしてください")

    p = float(hit_chance)
    if p <= 0 or p >= 1:
        return 0.0
    entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return entropy / math.log(2)  # いちばん迷っている時に1になるように割る


def optionality_score(futures, w_ess=0.5, w_div=0.3, w_val=0.2):
    """余裕を計算する。

    「この先の選びみちがどれだけ広く残っているか」。
    3つの値の足し合わせで作る:
      ・ESS       : 生きている未来の数の多さ
      ・Diversity : 未来どうしの行き先のばらけ具合
      ・ValueSpread: 未来ごとの危なさのばらつき

    未来が1つも無い時は 1.0 (= 何もない、自由) を返す。
    """
    n = len(futures)
    if n == 0:
        return 1.0
    if n == 1:
        return 0.0

    weights = [f["weight"] for f in futures]
    total = sum(weights)
    if total <= 0:
        return 0.0
    weights = [w / total for w in weights]

    # (1) 生きている未来の数 ESS = 1 / Σp²  → 0〜1に直す
    ess = 1.0 / sum(w * w for w in weights)
    ess_part = (ess - 1) / (n - 1)

    # (2) 行き先のばらけ具合: 未来ごとの最後の位置がどれだけ散っているか
    end_xs, end_ys = [], []
    for f in futures:
        xs = [path[-1][0] for path in f["paths"] if path]
        ys = [path[-1][1] for path in f["paths"] if path]
        if xs:
            end_xs.append(sum(xs) / len(xs))
            end_ys.append(sum(ys) / len(ys))
    div_part = 0.0
    if len(end_xs) > 1:
        spread = math.sqrt(_variance(end_xs) + _variance(end_ys))
        div_part = min(1.0, spread / 0.2)

    # (3) 危なさのばらつき: 未来ごとに「どこまで下まで行ったか」を比べる
    depths = []
    for f in futures:
        deepest = max((cy for path in f["paths"] for _, cy in path), default=0.0)
        depths.append(deepest)
    val_part = 0.0
    if len(depths) > 1:
        val_part = min(1.0, math.sqrt(_variance(depths)) / 0.15)

    score = w_ess * ess_part + w_div * div_part + w_val * val_part
    return max(0.0, min(1.0, score))


def _variance(numbers):
    """数のばらつき(分散)を計算する。"""
    mean = sum(numbers) / len(numbers)
    return sum((x - mean) ** 2 for x in numbers) / len(numbers)
