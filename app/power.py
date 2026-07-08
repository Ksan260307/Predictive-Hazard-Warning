# 省電力のために分析の回数を減らすファイル
#
# カメラ映像の分析はスマートフォンでは電池と発熱の主な原因になる。
# そこで「何も起きていない時間」が続いたら、分析を数フレームに1回に間引く。
#   ・映像の表示は間引かない (分析だけを減らす)
#   ・何かが見えた・危険度が上がった瞬間に、すぐ毎フレーム分析へ戻る
#   ・危険が出ている間は絶対に間引かない (安全側に倒す)

# この危険度以下で、物も見えていない状態を「静か」とみなす
CALM_RISK = 0.15
# 静かな分析結果がこれだけ続いたら間引きを始める (15fpsで約3秒)
CALM_AFTER = 45
# 間引き中は何フレームに1回分析するか
SKIP_STRIDE = 3


class FrameThrottle:
    """分析を間引くかどうかを決める係。

    使い方 (毎フレーム):
        if throttle.should_analyze():
            result = watcher.watch(frame)
            throttle.note(result)
        else:
            前回の結果で表示だけ更新する
    """

    def __init__(self, enabled=True, calm_after=CALM_AFTER, stride=SKIP_STRIDE):
        """
        enabled    : 間引きをするかどうか (False なら常に毎フレーム分析)
        calm_after : 静かな結果が何回続いたら間引きを始めるか
        stride     : 間引き中は何フレームに1回分析するか
        """
        if not isinstance(calm_after, int) or isinstance(calm_after, bool) or calm_after < 1:
            raise ValueError("間引きを始めるまでの回数(calm_after)は1以上の整数にしてください")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("間引きの間隔(stride)は1以上の整数にしてください")
        self.enabled = bool(enabled)
        self.calm_after = calm_after
        self.stride = stride
        self._calm = 0
        self._count = 0

    @property
    def saving(self):
        """いま間引き中かどうか。"""
        return self.enabled and self._calm >= self.calm_after

    def should_analyze(self):
        """このフレームで分析すべきかどうか。毎フレーム1回呼ぶ。"""
        self._count += 1
        if not self.saving:
            return True
        return self._count % self.stride == 0

    def note(self, result):
        """分析結果を1つ取り込み、「静かさ」を数える。

        result : DangerWatcher.watch() の返り値 (level, risk, things を使う)
        """
        if not isinstance(result, dict):
            raise ValueError("結果(result)は watch() の返り値を渡してください")
        for key in ("level", "risk", "things"):
            if key not in result:
                raise ValueError("結果に " + key + " がありません")

        calm = (result["level"] == 0
                and result["risk"] <= CALM_RISK
                and not result["things"])
        if calm:
            self._calm += 1
        else:
            self._calm = 0  # 何かが起きたら、すぐ毎フレーム分析に戻る

    def reset(self):
        """数え直す (カメラの切替時など)。"""
        self._calm = 0
        self._count = 0
