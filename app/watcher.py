# 全部の部品をつないで、画像1枚から「いまの注意」を作るファイル
#
# 処理の流れ:
#
#   観測(画像 + GPS) → 未来生成(見える物 + 死角からの飛び出し)
#     → 分布の更新 → 指標の計算 → 制御 → 行動(注意レベル)の選択
#
# 画面(main.py)からは、このファイルの DangerWatcher だけを使えばよい。

from app import alert
from app import control
from app import learning
from app import modes
from app import objectdetect
from app import risk as risk_mod
from app import roadinfo
from app import settings as settings_mod
from app import scene as scene_mod
from app.detector import MovingThingFinder
from app.future import FutureBelief, make_futures, make_phantom_futures
from app.scene import SceneReader
from app.tracker import ThingTracker

# 死角からの飛び出し予測が持てる重みの上限。
# 実際に見えている物の危険を、推測が上回りすぎないようにする
PHANTOM_CAP = 0.6

# 注意レベルを上げるのに必要な「連続して危険が出た回数」。
# 一瞬の揺れやノイズで警告が跳ね上がらないよう、数フレーム続けて
# 上がるべきと判断された時だけ1段上げる (下げる時はすぐ効かせる)
ESCALATE_CONFIRM = 3

# 用心 (r_weight) がこの値以上の時は、確認を待たずにすぐレベルを上げる。
# 差し迫った危険の警告を遅らせないため (control.pick_level の
# 「必ず知らせる」保証と同じしきい値)
URGENT_R_WEIGHT = 0.85

# ユーザーからの報告の種類
FEEDBACK_FALSE_ALARM = "false_alarm"   # 誤報だった
FEEDBACK_MISSED = "missed"             # 危険を見逃した


class DangerWatcher:
    """カメラ画像・位置情報を渡すと、いまの危険度と注意レベルを返す係。

    使い方:
        watcher = DangerWatcher(settings)
        watcher.update_location(lat, lon, t)   # GPSがあれば随時
        result = watcher.watch(frame)          # 毎フレーム
        watcher.report_feedback("false_alarm") # ユーザーからの報告
    """

    def __init__(self, settings=None, learn_path=None):
        self.settings = settings_mod.check_settings(settings)
        self.learner = learning.RiskLearner(path=learn_path)
        self._build_parts()

    def _build_parts(self):
        s = self.settings
        self.finder = MovingThingFinder(
            sensitivity=s["sensitivity"],
            min_size=s["min_size"],
            stabilize=s["stabilize"],
        )
        self.object_finder = self._make_object_finder()
        self.tracker = ThingTracker()
        self.scene_reader = SceneReader()
        self.belief = FutureBelief()   # 未来の分布 Y
        self.roads = roadinfo.RoadWatcher(provider=self._make_provider())
        self.level = control.LEVEL_SAFE
        self._rise = 0                 # レベルを上げてよいと連続で判断した回数
        self.last_features = {}
        self._tick = 0

    def _make_provider(self):
        """設定に応じて地図の問い合わせ先を作る。位置情報を使わないなら None。

        地図は OpenStreetMap を使う (キー不要)。圏外でも止まらない。
        """
        if not self.settings["use_location"]:
            return None
        return roadinfo.OsmProvider()

    def _make_object_finder(self):
        """AI物体認識を使える時だけ作る。使えない時は None (動き検出で動く)。

        デモ映像は合成画像なのでAIでは認識できず、常に動き検出を使う。
        """
        s = self.settings
        if not s["use_ml_detector"] or s["demo_mode"]:
            return None
        if not objectdetect.model_available():
            return None
        try:
            return objectdetect.ObjectFinder()
        except ValueError:
            return None  # モデルファイルが壊れている時は動き検出で動く

    def apply_settings(self, new_settings):
        """設定を入れ替える。おかしな値は自動で直される。

        物の見つけ方と地図の使い方が変わるので作り直す。
        ため込んだ未来・GPS履歴・学習・今のレベルは引き継ぐ。
        """
        self.settings = settings_mod.check_settings(new_settings)
        s = self.settings
        self.finder = MovingThingFinder(
            sensitivity=s["sensitivity"],
            min_size=s["min_size"],
            stabilize=s["stabilize"],
        )
        self.object_finder = self._make_object_finder()
        self.tracker = ThingTracker()  # 検出器が変わるので追跡もやり直す
        self.roads.provider = self._make_provider()

    def update_location(self, lat, lon, t):
        """GPSの位置を1つ取り込む。位置情報を使わない設定なら何もしない。"""
        if not self.settings["use_location"]:
            return
        self.roads.update(lat, lon, t)

    def watch(self, frame):
        """画像1枚を調べて、いまの状態をまとめた辞書を返す。

        返り値の主な中身:
            level        : 注意レベル (0=安全 / 1=注意 / 2=危険)
            name, text   : 画面に表示する言葉
            reasons      : なぜ注意が必要か (文のリスト)
            risk         : 総合の危険度 (0〜1)
            image_risk   : 画像(見える物+死角)からの危険度
            env_risk     : 地図情報(交差点など)からの危険度
            things       : 見つけた物 (id つき。AI認識時は label, score も)
            blind_spots  : 死角の候補
            detector     : 使った検出器 ("ml"=AI認識 / "motion"=動き検出)
            そのほか mode, uncertainty, optionality, d_weight,
            r_weight, phase, openness, road, boost
        """
        s = self.settings
        profile = modes.mode_profile(s["mode"])
        self._tick += 1

        # (1) 観測: 動く物・風景・道路状況
        #     AIモデルがあれば人・車などを直接認識し、無ければ動きで見つける
        if self.object_finder is not None:
            raw_things = self.object_finder.find(frame)
            detector_kind = "ml"
        else:
            raw_things = self.finder.find(frame)
            detector_kind = "motion"
        # 数フレーム連続で見えた物だけを通す (1フレームだけのノイズを捨てる)
        things = self.tracker.update(raw_things)
        scene = self.scene_reader.read(frame)
        road = self.roads.state()

        # AI認識で「止まっている車」が分かる時は、その陰も死角として加える。
        # 風景の縦境界と同じ場所を指す候補は、強い方だけを残す
        vehicle_spots = scene_mod.vehicle_blind_spots(things)
        if vehicle_spots:
            scene = dict(scene)
            scene["blind_spots"] = scene_mod.merge_blind_spots(
                scene["blind_spots"] + vehicle_spots)
            scene["blind_risk"] = max(
                (s["strength"] for s in scene["blind_spots"]), default=0.0)

        # 移動の度合い: GPSがあれば実測、なければモードの想定値
        if road["speed"] is None:
            move_factor = profile["base_move"]
        else:
            move_factor = min(1.0, road["speed"] / profile["fast_speed"])

        # 学習: いまの状況で予測をどれだけ強める/弱めるか
        features = self._make_features(s["mode"], scene, road, things, move_factor)
        self.last_features = features
        boost = self.learner.boost(features) if s["learning_on"] else 1.0

        # モードによる調整: 速い乗り物ほど遠くまで読み、早めに警告する
        steps = max(1, round(s["future_steps"] * profile["step_scale"]))
        danger_y = max(0.05, min(0.95, s["danger_line"] + profile["danger_shift"]))

        # (2) 未来生成: 見えている物の未来
        futures = make_futures(
            things, steps=steps, samples=s["future_samples"], seed=self._tick,
        )

        # (2') 死角からの飛び出しを「見えない未来」として混ぜる。
        #      移動が速いほど・死角が強いほど重くなる。学習でも増減する
        phantom_total = min(
            PHANTOM_CAP,
            profile["phantom_strength"] * scene["blind_risk"]
            * (0.4 + 0.6 * move_factor) * boost,
        )
        phantoms = make_phantom_futures(
            scene["blind_spots"], steps=steps,
            samples=max(1, s["future_samples"] // 3),
            strength=phantom_total, seed=self._tick,
        )
        if futures and phantoms:
            # 全体の重みが1を超えないよう、見えている物の分を少しゆずる
            for f in futures:
                f["weight"] *= (1 - phantom_total)

        # (3) 分布の更新
        self.belief.update(futures + phantoms, s["update_rate"])

        # (4) 指標の計算
        image_risk = risk_mod.risk_score(self.belief.samples, danger_y=danger_y)
        env_risk = self._env_risk(profile, road, move_factor, boost)
        # 総合の危険度: どちらか一方でも高ければ高くなる合成
        risk = 1.0 - (1.0 - image_risk) * (1.0 - env_risk)
        uncertainty = risk_mod.uncertainty_score(risk)
        optionality = risk_mod.optionality_score(self.belief.samples)

        # (5) 制御
        d_weight, r_weight = control.decide_weights(
            risk, uncertainty, optionality, risk_line=s["risk_line"],
        )
        phase = control.phase_of(risk, d_weight, r_weight)
        inner_mode = control.mode_of(risk, d_weight, r_weight)

        # (6) 行動選択 (上げる時だけ数フレームの確認を挟んで、跳ねを抑える)
        target = control.pick_level(r_weight, phase, prev_level=self.level)
        self.level = self._smooth_level(target, r_weight)

        # (7) 知らせ方
        result = alert.message_for(self.level)
        result.update({
            "reasons": self._reasons(things, image_risk, scene, road),
            "mode": inner_mode,
            "risk": risk,
            "image_risk": image_risk,
            "env_risk": env_risk,
            "uncertainty": uncertainty,
            "optionality": optionality,
            "d_weight": d_weight,
            "r_weight": r_weight,
            "phase": phase,
            "things": things,
            "blind_spots": scene["blind_spots"],
            "openness": scene["openness"],
            "road": road,
            "boost": boost,
            "danger_line": danger_y,
            "detector": detector_kind,
        })
        return result

    def report_feedback(self, kind):
        """ユーザーからの報告を学習に取り込む。

        kind: "false_alarm" (誤報だった) / "missed" (危険を見逃した)
        学習をオフにしている時は取り込まず False を返す。
        """
        if kind == FEEDBACK_FALSE_ALARM:
            direction = learning.DIRECTION_FALSE_ALARM
        elif kind == FEEDBACK_MISSED:
            direction = learning.DIRECTION_MISSED
        else:
            raise ValueError('報告は "false_alarm" か "missed" にしてください')
        if not self.settings["learning_on"]:
            return False
        self.learner.learn(self.last_features, direction)
        return True

    def _smooth_level(self, target, r_weight=0.0):
        """レベルの変化をなめらかにする。上げる時だけ確認回数を要求する。

        ・用心がとても強い (URGENT_R_WEIGHT 以上): 確認を待たずにすぐ従う。
          差し迫った危険の警告は1フレームも遅らせない
        ・target が今より高い: ESCALATE_CONFIRM 回連続で上と判断されたら1段上げる
        ・target が今と同じか低い: すぐ従う (下げ幅は control 側で1段ずつに制限済み)
        """
        if r_weight >= URGENT_R_WEIGHT:
            self._rise = 0
            return target
        if target > self.level:
            self._rise += 1
            if self._rise >= ESCALATE_CONFIRM:
                self._rise = 0
                return self.level + 1
            return self.level
        self._rise = 0
        return target

    def reset(self):
        """覚えていることを全部忘れて、最初の状態に戻る(学習は残す)。"""
        self._build_parts()

    # ---------------- 内部の計算 ----------------

    @staticmethod
    def _make_features(mode_key, scene, road, things, move_factor):
        """学習に渡す「いまの状況」を作る。"""
        return {
            "mode_walk": 1.0 if mode_key == "walk" else 0.0,
            "mode_bicycle": 1.0 if mode_key == "bicycle" else 0.0,
            "mode_car": 1.0 if mode_key == "car" else 0.0,
            "blind_spot": scene["blind_risk"],
            "intersection": road["intersection_score"],
            "turning": road["turning_score"],
            "moving": move_factor,
            "object": min(1.0, len(things) / 3.0),
        }

    @staticmethod
    def _env_risk(profile, road, move_factor, boost):
        """地図情報からの危険度。交差点への接近と旋回(見通しの変化)を見る。"""
        near = road["intersection_score"] * (0.3 + 0.7 * move_factor)
        turn = road["turning_score"] * 0.5 * move_factor
        value = profile["env_weight"] * max(near, turn) * boost
        return max(0.0, min(1.0, value))

    @staticmethod
    def _reasons(things, image_risk, scene, road):
        """「なぜ注意が必要か」を短い文のリストで返す。"""
        reasons = []
        if things and image_risk > 0.15:
            # AI認識で「何であるか」が分かっている時は名前で知らせる
            names = sorted({objectdetect.LABELS_JA[t["label"]] for t in things
                            if t.get("label") in objectdetect.LABELS_JA})
            if names:
                reasons.append("・".join(names) + "が接近しています")
            else:
                reasons.append("接近する物体があります")
        if scene["blind_risk"] > 0.35:
            reasons.append("見通しの悪い箇所があります")
        if road["intersection_score"] > 0.4:
            reasons.append("この先に交差点があります")
        if road["turning_score"] > 0.5:
            reasons.append("旋回中のため死角が変化しています")
        return reasons
