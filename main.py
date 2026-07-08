# 先読み危険予測 (画面部分)
#
# カメラ映像・位置情報から衝突などの危険を予測し、
# 「安全 / 注意 / 危険」の3段階で知らせるアプリ。
# 起動: python main.py
#
# スマートフォンの横持ちを想定した2カラム構成:
#   左  : カメラ映像 (危険域・検出物体・死角をHUDとして重ね描き)
#   右  : 状態パネル (警告・危険度ゲージ・モード・警告履歴・操作ボタン)

import collections
import os
import sys
import threading
import time

os.environ.setdefault("KIVY_NO_ARGS", "1")

import cv2

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.utils import platform

import app as app_pkg
from app import modes
from app import power as power_mod
from app import settings as settings_mod
from app import watcher as watcher_mod
from app.demo import DemoCamera
from app.detector import shrink_frame
from app.hud import draw_hud
from app.snapshot import SnapshotKeeper
from app.triplog import TripLogger
from app.watcher import DangerWatcher

# 学習結果・走行ログ・報告場面の画像の保存先 (設定ファイルと同じ場所)
LEARN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triplog.jsonl")
SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trip_snapshots")

# 配色 (ダークテーマ)
COLORS = {
    "bg": (0.06, 0.08, 0.11, 1),
    "panel": (0.11, 0.14, 0.19, 1),
    "panel_light": (0.17, 0.21, 0.28, 1),
    "panel_press": (0.24, 0.30, 0.38, 1),
    "text": (0.92, 0.94, 0.96, 1),
    "text_dim": (0.58, 0.64, 0.72, 1),
    "safe": (0.16, 0.62, 0.34, 1),
    "warn": (0.90, 0.66, 0.10, 1),
    "danger": (0.82, 0.16, 0.16, 1),
    "accent": (0.25, 0.55, 0.95, 1),
}
LEVEL_COLORS = {0: COLORS["safe"], 1: COLORS["warn"], 2: COLORS["danger"]}

# 分析に使う映像の横幅 (表示は元の解像度のまま)。
# AI物体認識はモデルの入力が大きいので、縮めすぎると精度が落ちる
ANALYSIS_WIDTH = 320
ML_ANALYSIS_WIDTH = 640


def setup_japanese_font():
    """日本語が表示できるフォントを探して登録する。"""
    candidates = [
        r"C:\Windows\Fonts\meiryo.ttc",                              # Windows
        r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
        "/system/fonts/NotoSansCJK-Regular.ttc",                     # Android
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",    # Linux
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",            # Mac
    ]
    for path in candidates:
        if os.path.exists(path):
            LabelBase.register(DEFAULT_FONT, path)
            return True
    return False


def play_beep(freq=880, ms=400):
    """通知音を鳴らす(鳴らせない環境では何もしない)。"""
    def _beep():
        try:
            import winsound
            winsound.Beep(freq, ms)
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()
    threading.Thread(target=_beep, daemon=True).start()


def vibrate(seconds=0.5):
    """振動させる (Androidのみ。使えない環境では何もしない)。"""
    try:
        from plyer import vibrator
        vibrator.vibrate(seconds)
    except Exception:
        pass


def speak(text):
    """警告の内容を読み上げる (使えない環境では何もしない)。"""
    def _speak():
        try:
            from plyer import tts
            tts.speak(text)
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


# ---------------------------------------------------------------
# 見た目の部品
# ---------------------------------------------------------------

class Card(BoxLayout):
    """角の丸い背景つきの入れ物。"""

    def __init__(self, bg=COLORS["panel"], radius=12, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            self._bg_color = Color(*bg)
            self._bg_rect = RoundedRectangle(radius=[radius])
        self.bind(pos=self._sync, size=self._sync)

    def _sync(self, *_):
        self._bg_rect.pos = self.pos
        self._bg_rect.size = self.size

    def set_bg(self, rgba):
        self._bg_color.rgba = rgba


class RiskGauge(Widget):
    """危険度を示す横棒ゲージ。値によって緑→黄→赤に変わる。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._value = 0.0
        with self.canvas:
            Color(*COLORS["panel_light"])
            self._track = RoundedRectangle(radius=[6])
            self._fill_color = Color(*COLORS["safe"])
            self._fill = RoundedRectangle(radius=[6])
        self.bind(pos=self._redraw, size=self._redraw)

    def set_value(self, value):
        self._value = min(1.0, max(0.0, float(value)))
        self._redraw()

    def _redraw(self, *_):
        self._track.pos = self.pos
        self._track.size = self.size
        if self._value < 0.35:
            self._fill_color.rgba = COLORS["safe"]
        elif self._value < 0.65:
            self._fill_color.rgba = COLORS["warn"]
        else:
            self._fill_color.rgba = COLORS["danger"]
        self._fill.pos = self.pos
        self._fill.size = (self.width * self._value, self.height)


def flat_button(text, on_release=None, bg=None, **kwargs):
    """フラットな見た目のボタンを作る。"""
    button = Button(
        text=text,
        background_normal="", background_down="",
        background_color=bg or COLORS["panel_light"],
        color=COLORS["text"],
        **kwargs,
    )
    base = button.background_color[:]

    def press(*_):
        button.background_color = COLORS["panel_press"]

    def release(*_):
        button.background_color = base
        if on_release:
            on_release()
    button.bind(on_press=press, on_release=release)
    return button


def dim_label(text, size="12sp", **kwargs):
    return Label(text=text, font_size=size, color=COLORS["text_dim"], **kwargs)


# ---------------------------------------------------------------
# 監視画面
# ---------------------------------------------------------------

class WatchScreen(Screen):
    """監視画面: 左にカメラ映像、右に状態パネル。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="horizontal", padding="10dp", spacing="10dp")

        # ---- 左: カメラ映像 ----
        camera_card = Card(padding="6dp")
        camera_area = FloatLayout()
        self.camera_view = Image(fit_mode="contain", size_hint=(1, 1))
        camera_area.add_widget(self.camera_view)
        self.badge_label = Label(
            text="", font_size="12sp", bold=True, color=COLORS["text"],
            size_hint=(None, None), size=("170dp", "26dp"),
            pos_hint={"x": 0.01, "top": 0.99},
        )
        camera_area.add_widget(self.badge_label)
        camera_card.add_widget(camera_area)
        root.add_widget(camera_card)

        # ---- 右: 状態パネル ----
        panel = BoxLayout(orientation="vertical", spacing="8dp",
                          size_hint_x=None, width="330dp")

        # 警告カード
        self.status_card = Card(orientation="vertical", padding="10dp",
                                size_hint_y=None, height="96dp")
        self.status_label = Label(text="起動中", font_size="30sp", bold=True,
                                  color=COLORS["text"], size_hint_y=0.6)
        self.reason_label = Label(text="カメラを準備しています", font_size="12sp",
                                  color=COLORS["text"], size_hint_y=0.4)
        self.reason_label.bind(size=lambda lb, _: setattr(lb, "text_size", (lb.width, None)))
        self.status_card.add_widget(self.status_label)
        self.status_card.add_widget(self.reason_label)
        panel.add_widget(self.status_card)

        # 危険度ゲージ
        gauge_card = Card(orientation="vertical", padding="10dp", spacing="6dp",
                          size_hint_y=None, height="84dp")
        gauge_top = BoxLayout(size_hint_y=None, height="20dp")
        gauge_top.add_widget(dim_label("危険度", halign="left"))
        self.risk_label = Label(text="0%", font_size="15sp", bold=True,
                                color=COLORS["text"])
        gauge_top.add_widget(self.risk_label)
        self.gauge = RiskGauge(size_hint_y=None, height="14dp")
        self.state_label = dim_label("状態 --   速度 --", size="11.5sp",
                                     size_hint_y=None, height="18dp")
        gauge_card.add_widget(gauge_top)
        gauge_card.add_widget(self.gauge)
        gauge_card.add_widget(self.state_label)
        panel.add_widget(gauge_card)

        # モード切替
        mode_row = BoxLayout(size_hint_y=None, height="42dp", spacing="6dp")
        self._mode_buttons = {}
        for key in modes.MODE_KEYS:
            button = ToggleButton(
                text=modes.MODES[key]["name"], group="mode",
                allow_no_selection=False,
                background_normal="", background_down="",
                background_color=COLORS["panel_light"],
                color=COLORS["text"],
            )
            button.bind(state=self._tint_mode_button)
            button.bind(on_release=lambda b, k=key: App.get_running_app().change_mode(k))
            mode_row.add_widget(button)
            self._mode_buttons[key] = button
        panel.add_widget(mode_row)

        # 警告履歴
        events_card = Card(orientation="vertical", padding="10dp", spacing="2dp")
        events_card.add_widget(dim_label("警告履歴", size_hint_y=None, height="18dp"))
        self._event_labels = []
        for _ in range(4):
            label = Label(text="", font_size="12sp", color=COLORS["text"],
                          halign="left")
            label.bind(size=lambda lb, _: setattr(lb, "text_size", (lb.width, None)))
            events_card.add_widget(label)
            self._event_labels.append(label)
        panel.add_widget(events_card)

        # 報告ボタン
        feedback_row = BoxLayout(size_hint_y=None, height="42dp", spacing="6dp")
        feedback_row.add_widget(flat_button(
            "誤報を報告",
            on_release=lambda: App.get_running_app()
            .send_feedback(watcher_mod.FEEDBACK_FALSE_ALARM)))
        feedback_row.add_widget(flat_button(
            "見逃しを報告",
            on_release=lambda: App.get_running_app()
            .send_feedback(watcher_mod.FEEDBACK_MISSED)))
        panel.add_widget(feedback_row)

        # 一時停止・設定
        control_row = BoxLayout(size_hint_y=None, height="42dp", spacing="6dp")
        self.pause_button = flat_button(
            "一時停止", on_release=lambda: App.get_running_app().toggle_pause())
        control_row.add_widget(self.pause_button)
        control_row.add_widget(flat_button(
            "設定", bg=COLORS["accent"],
            on_release=lambda: setattr(self.manager, "current", "settings")))
        panel.add_widget(control_row)

        root.add_widget(panel)
        self.add_widget(root)

    @staticmethod
    def _tint_mode_button(button, state):
        if state == "down":
            button.background_color = COLORS["accent"]
        else:
            button.background_color = COLORS["panel_light"]

    # ---- 表示の更新 ----

    def select_mode(self, key):
        if key in self._mode_buttons:
            self._mode_buttons[key].state = "down"

    def show_result(self, result, pulse=False):
        self.status_label.text = result["name"]
        color = list(LEVEL_COLORS[result["level"]])
        if pulse:  # 危険時は帯を明滅させる
            color = [min(1.0, c * 1.35) for c in color[:3]] + [1]
        self.status_card.set_bg(color)
        if result["reasons"]:
            self.reason_label.text = " / ".join(result["reasons"])
        else:
            self.reason_label.text = result["text"]

        self.risk_label.text = "{:.0f}%".format(result["risk"] * 100)
        self.gauge.set_value(result["risk"])
        road = result["road"]
        speed_text = ("{:.0f}km/h".format(road["speed"] * 3.6)
                      if road["speed"] is not None else "--")
        self.state_label.text = "状態 {}   速度 {}".format(result["mode"], speed_text)

    def show_events(self, events):
        for label, event in zip(self._event_labels, list(events) + [None] * 4):
            label.text = event if event else ""

    def set_badge(self, text):
        self.badge_label.text = text

    def flash_message(self, message):
        self.state_label.text = message

    def show_error(self, message):
        self.status_label.text = "エラー"
        self.status_card.set_bg(COLORS["panel_light"])
        self.reason_label.text = message

    def show_frame(self, frame):
        from kivy.graphics.texture import Texture
        flipped = cv2.flip(frame, 0)  # Kivyは上下が逆なので反転する
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(flipped.tobytes(), colorfmt="bgr", bufferfmt="ubyte")
        self.camera_view.texture = texture


# ---------------------------------------------------------------
# 設定画面
# ---------------------------------------------------------------

class SettingsScreen(Screen):
    """設定画面: 横持ちに合わせた2列のカード配置。"""

    SLIDER_ROWS = [
        ("camera_no", "カメラ番号", 0, 10, True),
        ("sensitivity", "検出感度", 1, 10, True),
        ("min_size", "検出する物の最小の大きさ", 0.001, 0.2, False),
        ("future_steps", "予測の先読みの長さ", 1, 60, True),
        ("future_samples", "予測の本数", 1, 200, True),
        ("update_rate", "予測の更新の速さ", 0.01, 1.0, False),
        ("risk_line", "警戒を強める危険度", 0.05, 0.95, False),
        ("danger_line", "危険域とみなす画面位置", 0.3, 0.95, False),
    ]
    SWITCH_ROWS = [
        ("use_ml_detector", "AIで物体を認識 (モデル導入時)"),
        ("power_save", "静かな時は分析を減らす (省電力)"),
        ("stabilize", "移動による映像の揺れを補正"),
        ("use_location", "位置情報(GPS)を使う"),
        ("learning_on", "報告から学習する"),
        ("sound_on", "危険時に音で知らせる"),
        ("voice_on", "警告の内容を声で読み上げる"),
        ("show_boxes", "映像に分析結果を重ねる"),
        ("save_log", "走行ログを保存する"),
        ("demo_mode", "デモ映像を使う"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sliders = {}
        self._switches = {}

        root = BoxLayout(orientation="vertical", padding="10dp", spacing="8dp")

        # 上部バー
        header = BoxLayout(size_hint_y=None, height="46dp", spacing="8dp")
        header.add_widget(flat_button("< 戻る", size_hint_x=None, width="110dp",
                                      on_release=self._back))
        header.add_widget(Label(text="設定", font_size="19sp", bold=True,
                                color=COLORS["text"]))
        header.add_widget(flat_button("初期設定に戻す", size_hint_x=None, width="150dp",
                                      on_release=lambda: self.show_values(settings_mod.DEFAULTS)))
        header.add_widget(flat_button("保存して戻る", bg=COLORS["accent"],
                                      size_hint_x=None, width="150dp",
                                      on_release=self._save_and_back))
        root.add_widget(header)

        scroll = ScrollView()
        grid = GridLayout(cols=2, spacing="8dp", padding=[0, 0, "8dp", 0],
                          size_hint_y=None)
        grid.bind(minimum_height=grid.setter("height"))

        for key, title, low, high, is_int in self.SLIDER_ROWS:
            grid.add_widget(self._make_slider_card(key, title, low, high, is_int))
        for key, title in self.SWITCH_ROWS:
            grid.add_widget(self._make_switch_card(key, title))

        # 学習リセット
        learn_card = Card(orientation="vertical", padding="10dp", spacing="6dp",
                          size_hint_y=None, height="86dp")
        learn_card.add_widget(dim_label("学習データ", size_hint_y=None, height="20dp"))
        learn_card.add_widget(flat_button(
            "学習した内容を消去", size_hint_y=None, height="38dp",
            on_release=lambda: App.get_running_app().reset_learning()))
        grid.add_widget(learn_card)

        # アプリ情報
        about_card = Card(orientation="vertical", padding="10dp",
                          size_hint_y=None, height="86dp")
        about_card.add_widget(Label(
            text="{}  v{}".format(app_pkg.APP_NAME, app_pkg.__version__),
            font_size="14sp", bold=True, color=COLORS["text"],
            size_hint_y=None, height="24dp"))
        about_card.add_widget(dim_label(
            "未来の分布に基づく先読み型の危険予測エンジンを搭載",
            size="11.5sp"))
        grid.add_widget(about_card)

        scroll.add_widget(grid)
        root.add_widget(scroll)
        self.add_widget(root)

    def _make_slider_card(self, key, title, low, high, is_int):
        card = Card(orientation="vertical", padding="10dp",
                    size_hint_y=None, height="86dp")
        top = BoxLayout(size_hint_y=None, height="20dp")
        label = dim_label(title, size="12.5sp", halign="left")
        label.bind(size=lambda lb, _: setattr(lb, "text_size", lb.size))
        top.add_widget(label)
        value_label = Label(text="", font_size="13sp", bold=True,
                            color=COLORS["text"], size_hint_x=None, width="64dp")
        top.add_widget(value_label)
        slider = Slider(min=low, max=high, step=1 if is_int else 0.01,
                        cursor_size=("18dp", "18dp"))

        def on_change(_, value):
            value_label.text = str(int(round(value)) if is_int else round(value, 2))
        slider.bind(value=on_change)

        card.add_widget(top)
        card.add_widget(slider)
        self._sliders[key] = (slider, is_int)
        return card

    def _make_switch_card(self, key, title):
        card = Card(padding="10dp", size_hint_y=None, height="56dp")
        label = dim_label(title, size="12.5sp", halign="left")
        label.bind(size=lambda lb, _: setattr(lb, "text_size", lb.size))
        card.add_widget(label)
        switch = Switch(size_hint_x=None, width="70dp")
        card.add_widget(switch)
        self._switches[key] = switch
        return card

    def show_values(self, settings):
        checked = settings_mod.check_settings(settings)
        for key, (slider, _) in self._sliders.items():
            slider.value = checked[key]
        for key, switch in self._switches.items():
            switch.active = checked[key]

    def read_values(self):
        values = {}
        for key, (slider, is_int) in self._sliders.items():
            values[key] = int(round(slider.value)) if is_int else float(slider.value)
        for key, switch in self._switches.items():
            values[key] = bool(switch.active)
        values["mode"] = App.get_running_app().settings_values["mode"]
        return values

    def _save_and_back(self):
        App.get_running_app().save_new_settings(self.read_values())
        self._back()

    def _back(self):
        self.manager.current = "watch"


# ---------------------------------------------------------------
# アプリ本体
# ---------------------------------------------------------------

class SakiyomiApp(App):
    """アプリ本体。カメラと位置情報を watcher に渡し、結果を表示する。"""

    title = "{} v{}".format(app_pkg.APP_NAME, app_pkg.__version__)

    def build(self):
        Window.clearcolor = COLORS["bg"]
        if platform not in ("android", "ios"):
            Window.size = (900, 440)  # スマートフォンの横持ちと同じ比率

        self.settings_values = settings_mod.load_settings()
        self.watcher = DangerWatcher(self.settings_values, learn_path=LEARN_FILE)
        self.throttle = power_mod.FrameThrottle(
            enabled=self.settings_values["power_save"])
        self.trip_logger = (TripLogger(LOG_FILE)
                            if self.settings_values["save_log"] else None)
        self.snapshots = (SnapshotKeeper(SNAP_DIR)
                          if self.settings_values["save_log"] else None)
        self.capture = None
        self.using_demo = False
        self.paused = False
        self.events = collections.deque(maxlen=4)
        self._last_beep = 0.0
        self._last_level = 0
        self._last_result = None
        self._pulse = False
        self._retry_wait = 0

        manager = ScreenManager()
        self.watch_screen = WatchScreen(name="watch")
        self.settings_screen = SettingsScreen(name="settings")
        self.settings_screen.show_values(self.settings_values)
        self.watch_screen.select_mode(self.settings_values["mode"])
        manager.add_widget(self.watch_screen)
        manager.add_widget(self.settings_screen)

        self._open_camera()
        self._start_gps_if_enabled()
        Clock.schedule_interval(self._tick, 1.0 / 15)
        return manager

    # ---------------- カメラ ----------------

    def _open_camera(self):
        """カメラを開く。開けない時やデモ設定時はデモ映像に切り替える。"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.throttle.reset()  # 映像が変わるので、間引かずに分析し直す

        if self.settings_values["demo_mode"]:
            self.capture = DemoCamera()
            self.using_demo = True
            self.watch_screen.set_badge("デモ映像")
            return

        camera_no = self.settings_values["camera_no"]
        if sys.platform == "win32":
            capture = cv2.VideoCapture(camera_no, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(camera_no)

        if capture.isOpened():
            self.capture = capture
            self.using_demo = False
            self.watcher.apply_settings(self.settings_values)
            self.watch_screen.set_badge("")
        else:
            capture.release()
            # カメラが見つからない間はデモ映像でつなぎ、定期的に再接続を試す。
            # デモは合成映像なのでAI認識は効かず、動き検出に切り替えてもらう
            self.capture = DemoCamera()
            self.using_demo = True
            self.watcher.apply_settings({**self.settings_values, "demo_mode": True})
            self._retry_wait = 150  # 約10秒後に再接続
            self.watch_screen.set_badge("デモ映像 (カメラ未接続)")

    def _tick(self, dt):
        """1回分の監視。カメラ→分析→HUD→画面表示。"""
        # カメラが未接続なら時々つなぎ直しを試す
        if self.using_demo and not self.settings_values["demo_mode"]:
            self._retry_wait -= 1
            if self._retry_wait <= 0:
                self._open_camera()

        if self.capture is None:
            self.watch_screen.show_error("カメラを初期化できません")
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.watch_screen.show_error("映像を取得できません")
            self._open_camera()
            return

        if self.paused:
            self.watch_screen.set_badge("監視を停止中")
            self.watch_screen.show_frame(frame)
            return

        # 静かな時間が続いたら分析を間引く (省電力)。表示は毎フレーム更新する
        if not self.throttle.should_analyze() and self._last_result is not None:
            if self.settings_values["show_boxes"]:
                frame = draw_hud(frame, self._last_result)
            self.watch_screen.show_frame(frame)
            return

        # 分析は縮小した映像で行う (速度対策)。座標は割合なので表示に流用できる。
        # AI認識の時はモデルの入力に合わせて大きめにする
        width = (ML_ANALYSIS_WIDTH if self.watcher.object_finder is not None
                 else ANALYSIS_WIDTH)
        analysis_frame = shrink_frame(frame, width)
        result = self.watcher.watch(analysis_frame)
        self.throttle.note(result)
        self._last_result = result
        if self.trip_logger is not None:
            self.trip_logger.log(result)
        if self.snapshots is not None:
            self.snapshots.add(analysis_frame)  # 報告時に場面を残せるように

        if self.settings_values["show_boxes"]:
            frame = draw_hud(frame, result)

        self._pulse = not self._pulse
        self.watch_screen.show_frame(frame)
        self.watch_screen.show_result(
            result, pulse=(result["level"] == 2 and self._pulse))

        self._notify(result)
        self._last_level = result["level"]

    def _notify(self, result):
        """レベルの変化に応じて音・声・振動・履歴を出す。"""
        level = result["level"]
        went_up = level > self._last_level

        if went_up:
            stamp = time.strftime("%H:%M:%S")
            self.events.appendleft("{}  {} ({})".format(
                stamp, result["name"],
                result["reasons"][0] if result["reasons"] else "周囲の状況"))
            self.watch_screen.show_events(self.events)
            # 声: 運転中は画面を見られないので、内容を読み上げる
            if self.settings_values["voice_on"] and level >= 1:
                spoken = (result["reasons"][0] if result["reasons"]
                          else result["text"])
                speak("{}。{}".format(result["name"], spoken))

        if not self.settings_values["sound_on"]:
            return
        now = Clock.get_time()
        if level == 2:
            # 危険: 上がった瞬間と、続いている間は2秒ごとに鳴らす
            if went_up or now - self._last_beep > 2.0:
                self._last_beep = now
                play_beep(880, 400)
                vibrate(0.6)
        elif level == 1 and went_up:
            # 注意: 上がった瞬間だけ控えめに鳴らす
            self._last_beep = now
            play_beep(600, 150)

    # ---------------- 位置情報 (GPS) ----------------

    def _start_gps_if_enabled(self):
        self._gps_running = False
        if not self.settings_values["use_location"]:
            return
        try:
            from plyer import gps
            gps.configure(on_location=self._on_gps_location)
            gps.start(minTime=1000, minDistance=1)
            self._gps_running = True
        except Exception:
            pass  # GPSが無い環境 (PCなど) では画像と地図以外の情報で動く

    def _stop_gps(self):
        if not getattr(self, "_gps_running", False):
            return
        try:
            from plyer import gps
            gps.stop()
        except Exception:
            pass
        self._gps_running = False

    def _on_gps_location(self, **kwargs):
        lat = kwargs.get("lat")
        lon = kwargs.get("lon")
        if lat is None or lon is None:
            return
        try:
            self.watcher.update_location(lat, lon, time.time())
        except ValueError:
            pass  # 乱れた位置情報は無視する

    # ---------------- 操作 ----------------

    def toggle_pause(self):
        self.paused = not self.paused
        self.watch_screen.pause_button.text = "再開" if self.paused else "一時停止"
        if not self.paused:
            self.watch_screen.set_badge("デモ映像" if self.using_demo else "")

    def change_mode(self, mode_key):
        values = dict(self.settings_values)
        values["mode"] = mode_key
        self.settings_values = settings_mod.save_settings(values)
        self.watcher.apply_settings(self.settings_values)
        self.watch_screen.flash_message(
            "モードを「{}」に切り替えました".format(modes.MODES[mode_key]["name"]))

    def send_feedback(self, kind):
        accepted = self.watcher.report_feedback(kind)
        if self.trip_logger is not None:
            # 報告の瞬間の場面も画像で残し、ログから辿れるようにする
            paths = self.snapshots.save(kind) if self.snapshots is not None else []
            self.trip_logger.log_feedback(kind, snapshots=paths)
        if accepted:
            self.watch_screen.flash_message("報告を学習に反映しました")
        else:
            self.watch_screen.flash_message("学習がオフのため反映されません")

    def reset_learning(self):
        self.watcher.learner.reset()
        self.watch_screen.flash_message("学習した内容を消去しました")

    def save_new_settings(self, values):
        old = self.settings_values
        self.settings_values = settings_mod.save_settings(values)
        self.watcher.apply_settings(self.settings_values)
        self.throttle = power_mod.FrameThrottle(
            enabled=self.settings_values["power_save"])
        self.trip_logger = (TripLogger(LOG_FILE)
                            if self.settings_values["save_log"] else None)
        self.snapshots = (SnapshotKeeper(SNAP_DIR)
                          if self.settings_values["save_log"] else None)
        self.settings_screen.show_values(self.settings_values)
        if (self.settings_values["camera_no"] != old["camera_no"]
                or self.settings_values["demo_mode"] != old["demo_mode"]):
            self._open_camera()
        if self.settings_values["use_location"] != old["use_location"]:
            self._stop_gps()
            self._start_gps_if_enabled()

    def on_stop(self):
        self._stop_gps()
        if self.capture is not None:
            self.capture.release()


if __name__ == "__main__":
    setup_japanese_font()
    SakiyomiApp().run()
