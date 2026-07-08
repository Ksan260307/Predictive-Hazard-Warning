# 危険予測を「学習」で調整するファイル
#
# ユーザーからの報告を取り込んで、予測の強さを状況ごとに調整する。
#   ・「誤報だった」        → 同じ状況では予測を弱める
#   ・「危険を見逃した」    → 同じ状況では予測を強める
#
# 状況(features)は「モード・死角・交差点・移動速度…」などの0〜1の値。
# 学習結果はファイルに保存され、次回起動時も引き継がれる。
#
# 学習が調整するのは「推測にもとづく予測」(飛び出し・交差点など)だけ。
# 実際に見えている衝突の危険そのものは、学習で弱められない設計にする。

import json
import math

# 学習が扱う状況の項目
FEATURE_NAMES = (
    "mode_walk",     # 徒歩モードか
    "mode_bicycle",  # 自転車モードか
    "mode_car",      # 車モードか
    "blind_spot",    # 死角の強さ
    "intersection",  # 交差点への接近度
    "turning",       # 旋回の度合い
    "moving",        # 移動速度の度合い
    "object",        # 見えている動く物の多さ
)

# 予測の強さの倍率の範囲 (これ以上は強くも弱くもしない)
BOOST_MIN = 0.25
BOOST_MAX = 2.5
# 1項目あたりの学習の重みの上限
WEIGHT_LIMIT = 1.2
# 学習の効果が半分になる報告の件数。
# 報告がまだ少ないうちは偶然の偏りかもしれないので、控えめに効かせる
# (件数が増えるほど信頼して、フルに効かせる)
CONFIDENCE_HALF = 5

# 報告の種類 → 学習の方向
DIRECTION_MISSED = 1       # 危険を見逃した → 予測を強める
DIRECTION_FALSE_ALARM = -1  # 誤報だった → 予測を弱める


class RiskLearner:
    """報告を積み重ねて、予測の強さを状況ごとに調整する係。

    使い方:
        learner = RiskLearner(path="learned.json")
        boost = learner.boost(features)     # いまの状況での倍率 (0.25〜2.5)
        learner.learn(features, DIRECTION_FALSE_ALARM)  # 報告を取り込む
    """

    def __init__(self, path=None):
        self.path = path
        self.weights = {name: 0.0 for name in FEATURE_NAMES}
        self.feedback_count = 0
        if path is not None:
            self._load()

    def boost(self, features):
        """いまの状況に対する予測の倍率を返す (0.25〜2.5)。

        学習していない状態では 1.0 (調整なし)。
        報告がまだ少ないうちは効果を控えめにし、
        数件の報告だけで予測が大きく振れないようにする。
        """
        z = 0.0
        for name, value in self._clean(features).items():
            z += self.weights[name] * value
        z *= self.confidence()
        return max(BOOST_MIN, min(BOOST_MAX, math.exp(z)))

    def confidence(self):
        """学習をどれだけ信じるか (0〜1)。報告が多いほど1に近づく。"""
        return self.feedback_count / (self.feedback_count + CONFIDENCE_HALF)

    def learn(self, features, direction, rate=0.25):
        """報告を1件取り込む。

        direction: DIRECTION_MISSED (1) か DIRECTION_FALSE_ALARM (-1)
        rate     : 1回の報告でどれだけ動かすか
        """
        if direction not in (DIRECTION_MISSED, DIRECTION_FALSE_ALARM):
            raise ValueError("direction は 1(見逃し) か -1(誤報) にしてください")
        if not 0 < rate <= 1:
            raise ValueError("学習の速さ(rate)は0より大きく1以下にしてください")

        for name, value in self._clean(features).items():
            new_weight = self.weights[name] + rate * direction * value
            self.weights[name] = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, new_weight))
        self.feedback_count += 1
        if self.path is not None:
            self.save()

    def reset(self):
        """学習した内容を全部消して、最初の状態に戻す。"""
        self.weights = {name: 0.0 for name in FEATURE_NAMES}
        self.feedback_count = 0
        if self.path is not None:
            self.save()

    def save(self, path=None):
        """学習結果をファイルに保存する。"""
        target = path or self.path
        if target is None:
            return
        data = {"weights": self.weights, "feedback_count": self.feedback_count}
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """保存された学習結果を読む。壊れていたら最初の状態で始める。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved = data.get("weights", {})
            count = data.get("feedback_count", 0)
        except (OSError, ValueError, AttributeError):
            return
        if not isinstance(saved, dict):
            return
        for name in FEATURE_NAMES:
            value = saved.get(name, 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                self.weights[name] = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, float(value)))
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self.feedback_count = count

    @staticmethod
    def _clean(features):
        """状況の辞書を安全な形に直す。知らない項目・変な値は捨てる。"""
        if not isinstance(features, dict):
            raise ValueError("状況(features)は辞書で渡してください")
        cleaned = {}
        for name in FEATURE_NAMES:
            value = features.get(name, 0.0)
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            cleaned[name] = max(0.0, min(1.0, float(value)))
        return cleaned
