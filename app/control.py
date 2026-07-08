# 「どのくらい注意するか」を決めるファイル
#
#   risk / uncertainty / optionality の3つの数から
#   ・すすむ気持ち D_weight と 用心する気持ち R_weight を決めて
#   ・位相(いまの調子)とモードを求めて
#   ・最終的な注意レベル(0=あんしん 1=ちゅうい 2=きけん)を選ぶ

import math

# 注意レベル (行動の選択肢)
LEVEL_SAFE = 0     # あんしん
LEVEL_CAREFUL = 1  # ちゅうい
LEVEL_DANGER = 2   # きけん
ALL_LEVELS = (LEVEL_SAFE, LEVEL_CAREFUL, LEVEL_DANGER)


def sigmoid(x):
    """0〜1になめらかに変わる曲線。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _check_zero_to_one(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(name + " は数にしてください")
    if not 0 <= value <= 1:
        raise ValueError(name + " は0〜1にしてください")


def decide_weights(risk, uncertainty, optionality, k_r=8.0, risk_line=0.5, k_u=0.7):
    """すすむ気持ち(D_weight)と用心する気持ち(R_weight)を決める。

    計算のしかた:
        R = sigmoid(k_r * (risk - risk_line))
        D = (1 - risk) * optionality
        Speed = exp(-k_u * uncertainty)   … 迷うほどゆっくりに
        D = D * Speed,  R = R + (1 - Speed)
        最後に D + R = 1 になるように直す

    返り値: (d_weight, r_weight)  どちらも0〜1で、足すと1。
    """
    _check_zero_to_one("危なさ(risk)", risk)
    _check_zero_to_one("迷い(uncertainty)", uncertainty)
    _check_zero_to_one("余裕(optionality)", optionality)
    if not 0 <= risk_line <= 1:
        raise ValueError("注意ライン(risk_line)は0〜1にしてください")

    r = sigmoid(k_r * (risk - risk_line))
    d = (1 - risk) * optionality
    speed = math.exp(-k_u * uncertainty)  # 迷いが大きいほど小さくなる

    d = d * speed
    r = r + (1 - speed)

    total = d + r
    if total <= 1e-9:
        return 0.5, 0.5  # どちらも0なら真ん中にしておく
    return d / total, r / total


def phase_of(risk, d_weight, r_weight):
    """位相(いまの調子を表す角度)を求める。

    activity = |D - R| (気持ちの偏り = 動きの強さ)
    phase = atan2(risk, activity)   → 0〜π/2 の角度になる
    """
    _check_zero_to_one("危なさ(risk)", risk)
    _check_zero_to_one("D_weight", d_weight)
    _check_zero_to_one("R_weight", r_weight)
    activity = abs(d_weight - r_weight)
    return math.atan2(risk, activity)


def temperature_of(phase, tau0=0.3, tau1=0.2):
    """選び方のゆるさ(温度)を位相から決める。

    τ = τ0 + τ1 * sin(phase)
    """
    if tau0 <= 0 or tau1 < 0:
        raise ValueError("温度の元になる数(tau0, tau1)が正しくありません")
    return tau0 + tau1 * math.sin(phase)


def softmax(scores, tau):
    """点数のリストを「選ばれやすさ」(合計1)に直す。

    tau (温度) が小さいほど、いちばん点数が高いものに集中する。
    """
    if tau <= 0:
        raise ValueError("温度(tau)は0より大きくしてください")
    if not scores:
        raise ValueError("点数(scores)が空です")
    biggest = max(scores)
    # いちばん大きい点数を引いてから計算する(数が大きくなりすぎるのを防ぐ)
    exps = [math.exp((s - biggest) / tau) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def pick_level(r_weight, phase, prev_level=LEVEL_SAFE, rng=None):
    """注意レベル(0/1/2)を選ぶ。

    ・各レベルに「いまの用心の強さに合っているか」で点数をつけ、
      softmax で選ぶ (rng を渡すとくじ引き、渡さなければ一番点が高いもの)
    ・用心が強い時は最低でも注意を出す (最小行動量の保証)
    ・レベルを下げる時は1段ずつ (急にあんしんに戻らない = なめらかさの制約)
    """
    _check_zero_to_one("R_weight", r_weight)
    if prev_level not in ALL_LEVELS:
        raise ValueError("前のレベル(prev_level)は0・1・2のどれかにしてください")

    # レベルごとの点数: そのレベルの強さ(0, 0.5, 1)が用心の強さに近いほど高い
    scores = [-(level / 2 - r_weight) ** 2 for level in ALL_LEVELS]
    tau = temperature_of(phase)
    chances = softmax(scores, tau)

    if rng is None:
        # ふだんは一番点数の高いレベルを選ぶ(表示がふらつかないように)
        level = max(ALL_LEVELS, key=lambda lv: chances[lv])
    else:
        # くじ引きで選ぶ(確率的な選び方)
        level = rng.choices(ALL_LEVELS, weights=chances)[0]

    # 最小行動量の保証: 用心がとても強い時は必ず知らせる
    if r_weight >= 0.85:
        level = max(level, LEVEL_DANGER)
    elif r_weight >= 0.6:
        level = max(level, LEVEL_CAREFUL)

    # レベルを下げる時は1段ずつ(急に「あんしん」に戻ってちらつくのを防ぐ)
    level = max(level, prev_level - 1)
    return level


def mode_of(risk, d_weight, r_weight):
    """いまの内部状態の名前を返す。

    ・探索: 危険が少なく、判断がはっきりしている (低リスク・高活動)
    ・安定: 危険が高いため慎重に動く             (高リスク・低活動)
    ・適応: その中間
    """
    _check_zero_to_one("危なさ(risk)", risk)
    activity = abs(d_weight - r_weight)
    if risk >= 0.5:
        return "安定"
    if activity >= 0.4:
        return "探索"
    return "適応"
