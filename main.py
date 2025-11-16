import sys
import time
import json
import os
import tempfile
import pyautogui
import numpy as np

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton, QTextEdit, QLineEdit, QSpinBox,
    QComboBox, QGroupBox, QGridLayout, QProgressBar, QMessageBox,
    QCheckBox, QTabWidget, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, QMutex, QWaitCondition, QMutexLocker,
    Qt, QCoreApplication, QTimer, QSettings, QSize
)
from PyQt5.QtGui import QFontDatabase

# ====== Global safety ======
pyautogui.FAILSAFE = True

PROFILES_FILE = "profiles.json"
PROFILE_SCHEMA_VERSION = 1

# ====== Data model ======
@dataclass
class Settings:
    # Click timing
    interval_mean: float
    interval_std: float
    duration_mean: float
    duration_std: float
    run_duration: Optional[float]  # seconds or None

    # Rest scheduling (minutes)
    first_rest_min: int
    first_rest_max: int
    subsequent_rest_min: int
    subsequent_rest_max: int
    rest_duration_min: int
    rest_duration_max: int

    # Human-like rest & micro-rests
    human_like_rests: bool
    micro_rests_enabled: bool
    micro_rest_prob: float          # probability per click to insert micro-rest
    micro_rest_min_s: float
    micro_rest_max_s: float

    # Pause on user activity
    pause_on_mouse_move: bool
    idle_after_move_s: float        # pause length after movement
    move_speed_px_per_s: float      # speed threshold to consider "active"

    # Area & jitter
    click_mode: str                 # "cursor", "jitter", "box"
    jitter_px: int                  # radius for jitter mode
    area_x: int
    area_y: int
    area_w: int
    area_h: int

# ====== Helpers ======
def fmt_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def settings_to_payload(s: Settings) -> dict:
    payload = asdict(s)
    payload["run_duration"] = int(payload["run_duration"] or 0)
    payload["_schema"] = PROFILE_SCHEMA_VERSION
    return payload

def payload_to_settings(payload: dict, default: Settings) -> Settings:
    # migrate in future if _schema changes; for now just normalize run_duration
    d = payload.copy()
    rd = d.get("run_duration", 0) or 0
    d["run_duration"] = None if int(rd) == 0 else float(rd)

    legacy_dynamic_keys = [
        "dynamic_std_enabled",
        "interval_std_gain_per_min",
        "interval_std_max",
        "duration_std_gain_per_min",
        "duration_std_max",
        "std_increase_rate_interval_std",
        "std_max_interval_std",
        "std_increase_rate_duration_std",
        "std_max_duration_std",
    ]
    for key in legacy_dynamic_keys:
        d.pop(key, None)

    # supply any missing fields from default
    for k, v in asdict(default).items():
        d.setdefault(k, v)
    return Settings(**d)

# ====== Area selection dialog ======
class AreaPickerDialog(QDialog):
    def __init__(self, parent=None, start: Tuple[int,int,int,int]=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Click Area")
        self.setModal(True)
        self.top_left = None
        self.bottom_right = None

        v = QVBoxLayout(self)
        self.info = QLabel(
            "Click “Capture Top-Left”, you’ll have 3 seconds to move the mouse.\n"
            "Then capture Bottom-Right.\n"
            "No global hotkeys required."
        )
        v.addWidget(self.info)

        grid = QGridLayout()
        v.addLayout(grid)

        self.tl_label = QLabel("Top-Left: (—, —)")
        self.br_label = QLabel("Bottom-Right: (—, —)")
        grid.addWidget(self.tl_label, 0, 0, 1, 2)
        grid.addWidget(self.br_label, 1, 0, 1, 2)

        self.btn_cap_tl = QPushButton("Capture Top-Left")
        self.btn_cap_br = QPushButton("Capture Bottom-Right")
        grid.addWidget(self.btn_cap_tl, 2, 0)
        grid.addWidget(self.btn_cap_br, 2, 1)

        self.countdown = QLabel("")
        v.addWidget(self.countdown)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        v.addWidget(self.buttons)

        self.btn_cap_tl.clicked.connect(lambda: self._capture_corner("tl"))
        self.btn_cap_br.clicked.connect(lambda: self._capture_corner("br"))
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

        if start:
            x, y, w, h = start
            self.top_left = (x, y)
            self.bottom_right = (x + w, y + h)
            self._update_labels()

    def _update_labels(self):
        tl = f"Top-Left: ({self.top_left[0]}, {self.top_left[1]})" if self.top_left else "Top-Left: (—, —)"
        br = f"Bottom-Right: ({self.bottom_right[0]}, {self.bottom_right[1]})" if self.bottom_right else "Bottom-Right: (—, —)"
        self.tl_label.setText(tl)
        self.br_label.setText(br)

    def _capture_corner(self, which: str):
        self.count = 3
        self.countdown.setText(f"Capturing in {self.count}… move your mouse")
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self._tick_capture(which))
        self.timer.start(1000)

    def _tick_capture(self, which: str):
        self.count -= 1
        if self.count <= 0:
            self.timer.stop()
            pos = pyautogui.position()
            if which == "tl":
                self.top_left = (pos.x, pos.y)
            else:
                self.bottom_right = (pos.x, pos.y)
            self._update_labels()
            self.countdown.setText("Captured.")
        else:
            self.countdown.setText(f"Capturing in {self.count}… move your mouse")

    def _accept(self):
        if not self.top_left or not self.bottom_right:
            QMessageBox.warning(self, "Area", "Capture both corners first.")
            return
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < 5 or h < 5:
            QMessageBox.warning(self, "Area", "Area too small.")
            return
        self.area = (x, y, w, h)
        self.accept()

# ====== Worker thread ======
class AutoclickerThread(QThread):
    log_signal           = pyqtSignal(str)
    status_signal        = pyqtSignal(str, str)  # main, sub
    finished_signal      = pyqtSignal()
    rest_progress_signal = pyqtSignal(int)
    next_rest_label      = pyqtSignal(str)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.mutex = QMutex()
        self.cond = QWaitCondition()

        self.running = False
        self.start_time: Optional[float] = None

        self.is_resting = False
        self.next_rest_time = 0.0
        self.last_rest_anchor = 0.0  # time when current rest window started

        self.paused_until = 0.0
        self.last_mouse_pos = pyautogui.position()
        self.last_pos_time = time.time()

        self.rng = np.random.default_rng()

        self.set_first_rest_time()

    # ----- Rest scheduling -----
    def _sample_rest_gap_minutes(self, first: bool) -> float:
        s = self.settings
        lo = s.first_rest_min if first else s.subsequent_rest_min
        hi = s.first_rest_max if first else s.subsequent_rest_max
        if not s.human_like_rests:
            return float(self.rng.integers(lo, hi + 1))
        # lognormal-ish between [lo, hi], biased toward the lower end but with long tail
        mid = (lo + hi) / 2.0
        sigma = 0.4
        mu = np.log(max(1e-6, mid))
        val = float(self.rng.lognormal(mean=mu, sigma=sigma))
        # clamp or resample if wildly outside
        if not (lo <= val <= hi):
            val = float(clamp(val, lo, hi))
        return val

    def _sample_rest_duration_minutes(self) -> float:
        s = self.settings
        lo, hi = s.rest_duration_min, s.rest_duration_max
        if not s.human_like_rests:
            return float(self.rng.integers(lo, hi + 1))
        # gamma with shape k>1 gives unimodal right-skew; scale to [lo, hi]
        k = 2.0
        theta = max(1e-6, (hi - lo) / (k * 1.6))
        val = float(self.rng.gamma(shape=k, scale=theta)) + lo
        return float(clamp(val, lo, hi))

    def set_first_rest_time(self):
        self.last_rest_anchor = 0.0
        first_gap_m = self._sample_rest_gap_minutes(first=True)
        self.next_rest_time = first_gap_m * 60.0

    def schedule_next_rest(self, elapsed_s: float):
        self.last_rest_anchor = elapsed_s
        gap_m = self._sample_rest_gap_minutes(first=False)
        self.next_rest_time = elapsed_s + gap_m * 60.0

    # ----- Settings updates & control -----
    def update_params(self, settings: Settings):
        with QMutexLocker(self.mutex):
            self.settings = settings
            # only reset schedule if not running or currently resting
            if not self.running or self.is_resting:
                self.set_first_rest_time()

    def start_clicking(self):
        with QMutexLocker(self.mutex):
            self.running = True
            self.start_time = time.time()
            self.is_resting = False
            self.paused_until = 0.0
            self.last_mouse_pos = pyautogui.position()
            self.last_pos_time = self.start_time
            self.set_first_rest_time()
            self.cond.wakeAll()

    def stop_clicking(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.start_time = None
            self.is_resting = False
            self.paused_until = 0.0
            self.cond.wakeAll()

    # ----- Status -----
    def _emit_status(self, main: str, sub: str = ""):
        self.status_signal.emit(main, sub)

    def _update_status(self):
        with QMutexLocker(self.mutex):
            running = self.running
            start_time = self.start_time
            is_resting = self.is_resting
            next_rest_time = self.next_rest_time
            last_rest_anchor = self.last_rest_anchor
            paused_until = self.paused_until

        if not running or not start_time:
            self._emit_status("Stopped", "")
            self.rest_progress_signal.emit(0)
            self.next_rest_label.emit("—")
            return

        now = time.time()
        elapsed = now - start_time

        if paused_until > now:
            self._emit_status("Paused (mouse activity)", f"Resumes in {fmt_hms(paused_until - now)}")
            self.rest_progress_signal.emit(0)
            self.next_rest_label.emit("—")
            return

        if is_resting:
            self._emit_status("Resting…", "")
            self.rest_progress_signal.emit(0)
            self.next_rest_label.emit("—")
            return

        to_rest = next_rest_time - elapsed
        if to_rest > 0:
            window = max(1e-6, next_rest_time - last_rest_anchor)
            prog = int(clamp((elapsed - last_rest_anchor) / window * 100, 0, 100))
            self._emit_status("Running…", f"Next rest in {fmt_hms(to_rest)}")
            self.rest_progress_signal.emit(prog)
            self.next_rest_label.emit(fmt_hms(to_rest))
        else:
            self._emit_status("Running…", "Rest overdue")
            self.rest_progress_signal.emit(100)
            self.next_rest_label.emit("00:00")

    # ----- Mouse activity pause -----
    def _check_mouse_activity_and_pause(self):
        if not self.settings.pause_on_mouse_move:
            return
        now = time.time()
        pos = pyautogui.position()
        dt = max(1e-6, now - self.last_pos_time)
        dx = pos.x - self.last_mouse_pos.x
        dy = pos.y - self.last_mouse_pos.y
        dist = (dx * dx + dy * dy) ** 0.5
        speed = dist / dt  # px/s
        self.last_mouse_pos = pos
        self.last_pos_time = now
        if speed >= self.settings.move_speed_px_per_s:
            with QMutexLocker(self.mutex):
                self.paused_until = max(self.paused_until, now + self.settings.idle_after_move_s)
            self.log_signal.emit(f"Mouse activity detected ({speed:.0f}px/s) → pausing for {self.settings.idle_after_move_s:.1f}s")

    # ----- Area targeting -----
    def _pick_click_point(self, rng: np.random.Generator) -> Tuple[int, int]:
        s = self.settings
        if s.click_mode == "cursor":
            pos = pyautogui.position()
            return pos.x, pos.y
        if s.click_mode == "jitter":
            cx, cy = pyautogui.position()
            r = int(s.jitter_px)
            if r <= 0:
                return cx, cy
            dx = int(rng.integers(-r, r + 1))
            dy = int(rng.integers(-r, r + 1))
            return cx + dx, cy + dy
        # "box"
        x = int(rng.integers(s.area_x, s.area_x + max(1, s.area_w)))
        y = int(rng.integers(s.area_y, s.area_y + max(1, s.area_h)))
        return x, y

    # ----- Main loop -----
    def run(self):
        next_click_at = None

        while True:
            with QMutexLocker(self.mutex):
                if not self.running:
                    # Wait indefinitely until woken by start_clicking() or app exit
                    self.cond.wait(self.mutex)

                running = self.running
                start_time = self.start_time

            if not running or not start_time:
                continue

            now = time.time()
            elapsed = now - start_time

            std_i = self.settings.interval_std
            std_d = self.settings.duration_std

            # Stop by run_duration
            if self.settings.run_duration and elapsed >= self.settings.run_duration:
                with QMutexLocker(self.mutex):
                    self.running = False
                    self.start_time = None
                    self.is_resting = False
                self._emit_status("Run completed", "")
                self.finished_signal.emit()
                continue

            # Rest time?
            if not self.is_resting and elapsed >= self.next_rest_time:
                self.is_resting = True
                rest_m = self._sample_rest_duration_minutes()
                rest_s = max(1.0, rest_m * 60.0)
                self.log_signal.emit(f"Resting for {rest_m:.1f} min…")
                # sleep in chunks so stop can interrupt
                t_end = time.time() + rest_s
                while True:
                    with QMutexLocker(self.mutex):
                        if not self.running:
                            break
                        remaining_ms = int(max(0.0, t_end - time.time()) * 1000)
                        if remaining_ms <= 0:
                            break
                        self.cond.wait(self.mutex, min(remaining_ms, 500))
                    self._emit_status("Resting…", f"{fmt_hms(t_end - time.time())} left")
                self.is_resting = False
                self.log_signal.emit("Rest complete.")
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    self.schedule_next_rest(elapsed)
                next_click_at = None
                self._update_status()
                continue

            # Schedule next click
            if next_click_at is None:
                wait = max(0.05, float(self.rng.normal(self.settings.interval_mean, std_i)))
                next_click_at = time.time() + wait
                self.next_rest_label.emit(fmt_hms(max(0.0, self.next_rest_time - (time.time() - start_time))))

            # Pause for mouse activity if needed
            self._check_mouse_activity_and_pause()
            with QMutexLocker(self.mutex):
                paused_until = self.paused_until
                is_resting = self.is_resting
                running = self.running

            if not running:
                next_click_at = None
                continue

            now = time.time()
            if paused_until > now or is_resting:
                with QMutexLocker(self.mutex):
                    self.cond.wait(self.mutex, 200)
                self._update_status()
                continue

            # Wait until next_click_at (drift-resistant)
            delay_ms = int(max(0.0, next_click_at - time.time()) * 1000)
            if delay_ms > 0:
                with QMutexLocker(self.mutex):
                    self.cond.wait(self.mutex, min(delay_ms, 200))
                self._update_status()
                continue

            # Optional micro-rest before clicking
            if self.settings.micro_rests_enabled and self.rng.random() < self.settings.micro_rest_prob:
                micro = float(self.rng.uniform(self.settings.micro_rest_min_s, self.settings.micro_rest_max_s))
                self.log_signal.emit(f"Micro-rest {micro:.1f}s")
                t_end = time.time() + micro
                while True:
                    with QMutexLocker(self.mutex):
                        if not self.running:
                            break
                        remaining_ms = int(max(0.0, t_end - time.time()) * 1000)
                        if remaining_ms <= 0:
                            break
                        self.cond.wait(self.mutex, min(remaining_ms, 200))
                next_click_at = time.time()
                continue

            # Perform click at a target point
            try:
                x, y = self._pick_click_point(self.rng)
                pyautogui.moveTo(x, y, duration=0)
                dur = max(0.03, float(self.rng.normal(self.settings.duration_mean, std_d)))
                pyautogui.mouseDown()
                time.sleep(dur)
                pyautogui.mouseUp()
                self.log_signal.emit(f"Clicked {dur:.2f}s at ({x},{y})")
            except pyautogui.FailSafeException:
                self.log_signal.emit("Failsafe triggered (top-left). Stopping.")
                self.stop_clicking()
                continue
            except Exception as e:
                self.log_signal.emit(f"Click error: {e!r}")
                self.stop_clicking()
                continue

            # Schedule the next target time (stochastic intervals, drift-resistant)
            wait = max(0.05, float(self.rng.normal(self.settings.interval_mean, std_i)))
            next_click_at = time.time() + wait

            self._update_status()

# ====== GUI ======
class AutoclickerGUI(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings_obj = settings
        self.profiles = {}
        self.setWindowTitle("ML-Inspired Autoclicker")

        # QSettings must be created before methods that use it
        self._settings = QSettings("autoclicker.ml", "app")

        self.tabs = QTabWidget()
        main = QWidget()
        adv = QWidget()
        prof = QWidget()

        self._build_tab_main(main)
        self._build_tab_advanced(adv)
        self._build_tab_profiles(prof)

        self.tabs.addTab(main, "Main")
        self.tabs.addTab(adv, "Advanced")
        self.tabs.addTab(prof, "Profiles")

        root = QVBoxLayout(self)
        root.addWidget(self.tabs)

        self._apply_styles()

        # safe now that _settings exists
        self._load_profiles()

        # worker thread
        self.thread = AutoclickerThread(self._read_settings_from_ui())
        self.thread.log_signal.connect(self._append_log)
        self.thread.status_signal.connect(self._set_status)
        self.thread.finished_signal.connect(lambda: self._set_status("Run done", ""))
        self.thread.rest_progress_signal.connect(self.main_rest_progress.setValue)
        self.thread.next_rest_label.connect(self.main_next_rest.setText)
        self.thread.start()

        # heartbeat timer
        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(300)
        self.ui_timer.timeout.connect(self.thread._update_status)
        self.ui_timer.start()

        # window restore
        self._restore_window()

        QCoreApplication.instance().aboutToQuit.connect(self._on_quit)

    # ---- UI construction ----
    def _build_tab_main(self, w: QWidget):
        v = QVBoxLayout(w)

        # Essential controls
        ctrl = QHBoxLayout()
        self.main_start = QPushButton("Start")
        self.main_stop = QPushButton("Stop")
        ctrl.addWidget(self.main_start)
        ctrl.addWidget(self.main_stop)
        v.addLayout(ctrl)

        status_line = QHBoxLayout()
        self.main_status = QLabel("Status: Idle")
        self.main_sub = QLabel("")
        self.main_sub.setStyleSheet("color:#9E9E9E;")
        status_line.addWidget(self.main_status, 2)
        status_line.addWidget(self.main_sub, 1, alignment=Qt.AlignRight)
        v.addLayout(status_line)

        prog_line = QHBoxLayout()
        prog_line.addWidget(QLabel("Next Rest Progress:"))
        self.main_rest_progress = QProgressBar()
        self.main_rest_progress.setRange(0, 100)
        prog_line.addWidget(self.main_rest_progress)
        v.addLayout(prog_line)

        next_rest_line = QHBoxLayout()
        next_rest_line.addWidget(QLabel("Next rest in:"))
        self.main_next_rest = QLabel("—")
        next_rest_line.addWidget(self.main_next_rest)
        v.addLayout(next_rest_line)

        # Quick toggles
        toggles = QGroupBox("Quick Toggles")
        tg = QGridLayout(toggles)
        self.cb_pause_on_move = QCheckBox("Pause when mouse moves fast")
        self.cb_pause_on_move.setChecked(self.settings_obj.pause_on_mouse_move)
        self.spin_idle_after_move = QDoubleSpinBox(); self.spin_idle_after_move.setRange(0.5, 30.0); self.spin_idle_after_move.setDecimals(1)
        self.spin_idle_after_move.setValue(self.settings_obj.idle_after_move_s)
        self.spin_move_speed = QDoubleSpinBox(); self.spin_move_speed.setRange(10, 5000); self.spin_move_speed.setDecimals(0)
        self.spin_move_speed.setValue(self.settings_obj.move_speed_px_per_s)
        tg.addWidget(self.cb_pause_on_move, 0, 0, 1, 2)
        tg.addWidget(QLabel("Idle after move (s):"), 1, 0); tg.addWidget(self.spin_idle_after_move, 1, 1)
        tg.addWidget(QLabel("Speed threshold (px/s):"), 2, 0); tg.addWidget(self.spin_move_speed, 2, 1)
        v.addWidget(toggles)

        # Area/jitter quick controls
        area = QGroupBox("Targeting")
        ag = QGridLayout(area)
        self.combo_click_mode = QComboBox()
        self.combo_click_mode.addItems(["cursor", "jitter", "box"])
        self.combo_click_mode.setCurrentText(self.settings_obj.click_mode)
        self.spin_jitter = QSpinBox(); self.spin_jitter.setRange(0, 500); self.spin_jitter.setValue(self.settings_obj.jitter_px)
        self.btn_pick_area = QPushButton("Pick Area")
        self.lbl_area = QLabel(self._area_text())
        ag.addWidget(QLabel("Mode:"), 0, 0); ag.addWidget(self.combo_click_mode, 0, 1)
        ag.addWidget(QLabel("Jitter radius (px):"), 1, 0); ag.addWidget(self.spin_jitter, 1, 1)
        ag.addWidget(self.btn_pick_area, 2, 0); ag.addWidget(self.lbl_area, 2, 1)
        v.addWidget(area)

        # Log
        v.addWidget(QLabel("Log:"))
        self.main_log = QTextEdit(); self.main_log.setReadOnly(True); v.addWidget(self.main_log)

        # wire up
        self.main_start.clicked.connect(self._sync_and_start)
        self.main_stop.clicked.connect(self._stop_clicked)
        self.btn_pick_area.clicked.connect(self._pick_area)
        self.cb_pause_on_move.toggled.connect(self._apply_quick_toggles)
        self.spin_idle_after_move.valueChanged.connect(self._apply_quick_toggles)
        self.spin_move_speed.valueChanged.connect(self._apply_quick_toggles)
        self.combo_click_mode.currentTextChanged.connect(self._apply_quick_toggles)
        self.spin_jitter.valueChanged.connect(self._apply_quick_toggles)

    def _build_tab_advanced(self, w: QWidget):
        v = QVBoxLayout(w)

        # Click timing
        ag = QGroupBox("Autoclicker Timing")
        gl = QGridLayout(ag)
        self.spin_interval_mean = QDoubleSpinBox(); self.spin_interval_mean.setDecimals(4); self.spin_interval_mean.setRange(0.0, 10.0)
        self.spin_interval_std  = QDoubleSpinBox(); self.spin_interval_std.setDecimals(4);  self.spin_interval_std.setRange(0.0, 10.0)
        self.spin_duration_mean = QDoubleSpinBox(); self.spin_duration_mean.setDecimals(4); self.spin_duration_mean.setRange(0.0, 1.0)
        self.spin_duration_std  = QDoubleSpinBox(); self.spin_duration_std.setDecimals(4);  self.spin_duration_std.setRange(0.0, 1.0)
        self.spin_run_duration  = QSpinBox(); self.spin_run_duration.setRange(0, 999999)

        s = self.settings_obj
        self.spin_interval_mean.setValue(s.interval_mean)
        self.spin_interval_std.setValue(s.interval_std)
        self.spin_duration_mean.setValue(s.duration_mean)
        self.spin_duration_std.setValue(s.duration_std)
        self.spin_run_duration.setValue(int(s.run_duration or 0))

        gl.addWidget(QLabel("Interval Mean (s):"), 0, 0); gl.addWidget(self.spin_interval_mean, 0, 1)
        gl.addWidget(QLabel("Interval Std (s):"), 0, 2);  gl.addWidget(self.spin_interval_std, 0, 3)
        gl.addWidget(QLabel("Duration Mean (s):"), 1, 0); gl.addWidget(self.spin_duration_mean, 1, 1)
        gl.addWidget(QLabel("Duration Std (s):"), 1, 2);  gl.addWidget(self.spin_duration_std, 1, 3)
        gl.addWidget(QLabel("Run Duration (s, 0 = ∞):"), 2, 0); gl.addWidget(self.spin_run_duration, 2, 1, 1, 3)
        v.addWidget(ag)

        # Rests
        rg = QGroupBox("Rest Settings")
        rl = QGridLayout(rg)
        self.spin_first_min = QSpinBox(); self.spin_first_min.setRange(1, 240); self.spin_first_min.setValue(s.first_rest_min)
        self.spin_first_max = QSpinBox(); self.spin_first_max.setRange(1, 480); self.spin_first_max.setValue(s.first_rest_max)
        self.spin_sub_min   = QSpinBox(); self.spin_sub_min.setRange(1, 240);   self.spin_sub_min.setValue(s.subsequent_rest_min)
        self.spin_sub_max   = QSpinBox(); self.spin_sub_max.setRange(1, 480);   self.spin_sub_max.setValue(s.subsequent_rest_max)
        self.spin_rest_min  = QSpinBox(); self.spin_rest_min.setRange(1, 180);  self.spin_rest_min.setValue(s.rest_duration_min)
        self.spin_rest_max  = QSpinBox(); self.spin_rest_max.setRange(1, 480);  self.spin_rest_max.setValue(s.rest_duration_max)
        self.cb_human_like  = QCheckBox("Use human-like distributions (log-normal / gamma)")
        self.cb_human_like.setChecked(s.human_like_rests)

        rl.addWidget(QLabel("First rest min (m):"), 0, 0); rl.addWidget(self.spin_first_min, 0, 1)
        rl.addWidget(QLabel("First rest max (m):"), 0, 2); rl.addWidget(self.spin_first_max, 0, 3)
        rl.addWidget(QLabel("Subseq rest min (m):"),1, 0); rl.addWidget(self.spin_sub_min,   1, 1)
        rl.addWidget(QLabel("Subseq rest max (m):"),1, 2); rl.addWidget(self.spin_sub_max,   1, 3)
        rl.addWidget(QLabel("Rest duration min (m):"),2, 0); rl.addWidget(self.spin_rest_min, 2, 1)
        rl.addWidget(QLabel("Rest duration max (m):"),2, 2); rl.addWidget(self.spin_rest_max, 2, 3)
        rl.addWidget(self.cb_human_like, 3, 0, 1, 4)
        v.addWidget(rg)

        # Micro-rests
        mg = QGroupBox("Micro-Rests")
        ml = QGridLayout(mg)
        self.cb_micro = QCheckBox("Enable micro-rests")
        self.cb_micro.setChecked(s.micro_rests_enabled)
        self.spin_micro_prob = QDoubleSpinBox(); self.spin_micro_prob.setRange(0.0, 1.0); self.spin_micro_prob.setDecimals(3); self.spin_micro_prob.setValue(s.micro_rest_prob)
        self.spin_micro_min  = QDoubleSpinBox(); self.spin_micro_min.setRange(0.5, 10.0); self.spin_micro_min.setDecimals(1); self.spin_micro_min.setValue(s.micro_rest_min_s)
        self.spin_micro_max  = QDoubleSpinBox(); self.spin_micro_max.setRange(0.5, 30.0); self.spin_micro_max.setDecimals(1); self.spin_micro_max.setValue(s.micro_rest_max_s)
        ml.addWidget(self.cb_micro, 0, 0, 1, 4)
        ml.addWidget(QLabel("Probability per click:"), 1, 0); ml.addWidget(self.spin_micro_prob, 1, 1)
        ml.addWidget(QLabel("Duration min (s):"),      1, 2); ml.addWidget(self.spin_micro_min,  1, 3)
        ml.addWidget(QLabel("Duration max (s):"),      2, 2); ml.addWidget(self.spin_micro_max,  2, 3)
        v.addWidget(mg)

        # Apply while stopped
        apply_line = QHBoxLayout()
        self.btn_apply_adv = QPushButton("Apply (when stopped)")
        apply_line.addWidget(self.btn_apply_adv)
        v.addLayout(apply_line)
        self.btn_apply_adv.clicked.connect(self._apply_advanced)

    def _build_tab_profiles(self, w: QWidget):
        v = QVBoxLayout(w)
        pg = QGroupBox("Profile Management")
        pl = QGridLayout(pg)
        self.profile_combo = QComboBox()
        self.profile_name_edit = QLineEdit(); self.profile_name_edit.setPlaceholderText("Enter profile name…")
        self.save_profile_btn = QPushButton("Save Profile")
        self.load_profile_btn = QPushButton("Load Profile")
        pl.addWidget(QLabel("Profile:"), 0, 0); pl.addWidget(self.profile_combo, 0, 1)
        pl.addWidget(self.profile_name_edit, 1, 0, 1, 2)
        pl.addWidget(self.save_profile_btn, 2, 0); pl.addWidget(self.load_profile_btn, 2, 1)
        v.addWidget(pg)

        # wire
        self.save_profile_btn.clicked.connect(self.save_profiles)
        self.load_profile_btn.clicked.connect(self.load_profile)

        v.addStretch(1)

    # ---- Styles, status, log ----
    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #E0E0E0; }
            QGroupBox { border:1px solid #333; border-radius:8px; margin-top:10px; }
            QGroupBox::title { color:#BB86FC; subcontrol-origin: margin; left:15px; }
            QLabel, QCheckBox { color:#BB86FC; }
            QPushButton { background-color:#BB86FC; color:#121212; padding:8px; border:none; border-radius:5px; }
            QPushButton:hover { background-color:#3700B3; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background-color:#1E1E1E; border:1px solid #BB86FC; border-radius:5px; color:#FFFFFF;
            }
            QProgressBar { border:1px solid #BB86FC; border-radius:5px; text-align:center; }
            QProgressBar::chunk { background-color:#03DAC6; }
            QTabBar::tab { padding:8px; }
        """)

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.main_log.append(f"[{ts}] {msg}")

    def _set_status(self, main: str, sub: str):
        self.main_status.setText(f"Status: {main}")
        self.main_sub.setText(sub)

    # ---- Settings sync/validate ----
    def _read_settings_from_ui(self) -> Settings:
        rd = self.spin_run_duration.value() or 0
        rd = None if rd == 0 else float(rd)

        mode = self.combo_click_mode.currentText()
        x, y, w, h = self.settings_obj.area_x, self.settings_obj.area_y, self.settings_obj.area_w, self.settings_obj.area_h

        return Settings(
            interval_mean=self.spin_interval_mean.value(),
            interval_std=self.spin_interval_std.value(),
            duration_mean=self.spin_duration_mean.value(),
            duration_std=self.spin_duration_std.value(),
            run_duration=rd,
            first_rest_min=self.spin_first_min.value(),
            first_rest_max=self.spin_first_max.value(),
            subsequent_rest_min=self.spin_sub_min.value(),
            subsequent_rest_max=self.spin_sub_max.value(),
            rest_duration_min=self.spin_rest_min.value(),
            rest_duration_max=self.spin_rest_max.value(),
            human_like_rests=self.cb_human_like.isChecked(),
            micro_rests_enabled=self.cb_micro.isChecked(),
            micro_rest_prob=self.spin_micro_prob.value(),
            micro_rest_min_s=self.spin_micro_min.value(),
            micro_rest_max_s=self.spin_micro_max.value(),
            pause_on_mouse_move=self.cb_pause_on_move.isChecked(),
            idle_after_move_s=self.spin_idle_after_move.value(),
            move_speed_px_per_s=self.spin_move_speed.value(),
            click_mode=mode,
            jitter_px=self.spin_jitter.value(),
            area_x=x, area_y=y, area_w=w, area_h=h
        )

    def _validate(self, s: Settings) -> bool:
        if s.first_rest_min > s.first_rest_max or s.subsequent_rest_min > s.subsequent_rest_max or s.rest_duration_min > s.rest_duration_max:
            QMessageBox.warning(self, "Validation", "Rest ranges invalid.")
            return False
        if s.micro_rests_enabled and s.micro_rest_min_s > s.micro_rest_max_s:
            QMessageBox.warning(self, "Validation", "Micro-rest min/max invalid.")
            return False
        if s.click_mode == "box" and (s.area_w < 5 or s.area_h < 5):
            QMessageBox.warning(self, "Validation", "Pick a valid area first.")
            return False
        if s.interval_mean <= 0.0:
            QMessageBox.warning(self, "Validation", "Interval mean must be > 0.")
            return False
        return True

    def _apply_advanced(self):
        if self.thread.running:
            QMessageBox.information(self, "Apply", "Stop the clicker before applying advanced settings.")
            return
        s = self._read_settings_from_ui()
        if not self._validate(s): return
        self.settings_obj = s
        self.thread.update_params(s)
        self._append_log("Advanced settings applied.")

    def _apply_quick_toggles(self):
        s = self.settings_obj
        s.pause_on_mouse_move = self.cb_pause_on_move.isChecked()
        s.idle_after_move_s = self.spin_idle_after_move.value()
        s.move_speed_px_per_s = self.spin_move_speed.value()
        s.click_mode = self.combo_click_mode.currentText()
        s.jitter_px = self.spin_jitter.value()
        self.thread.update_params(s)
        self.lbl_area.setText(self._area_text())

    def _sync_and_start(self):
        if self.thread.running:
            return
        s = self._read_settings_from_ui()
        if not self._validate(s): return
        self.settings_obj = s
        self.thread.update_params(s)
        self._set_controls_enabled(False)
        self.thread.start_clicking()
        self._set_status("Running", "")
        self._append_log("Started.")

    def _stop_clicked(self):
        self.thread.stop_clicking()
        self._set_status("Stopped", "")
        self._append_log("Stopped.")
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        self.main_start.setEnabled(enabled)
        self.btn_apply_adv.setEnabled(enabled)
        for w in [
            self.spin_interval_mean, self.spin_interval_std, self.spin_duration_mean, self.spin_duration_std,
            self.spin_run_duration,
            self.spin_first_min, self.spin_first_max, self.spin_sub_min, self.spin_sub_max, self.spin_rest_min, self.spin_rest_max,
            self.cb_human_like, self.cb_micro, self.spin_micro_prob, self.spin_micro_min, self.spin_micro_max
        ]:
            w.setEnabled(enabled)

    # ---- Area management ----
    def _pick_area(self):
        start = (self.settings_obj.area_x, self.settings_obj.area_y, self.settings_obj.area_w, self.settings_obj.area_h)
        dlg = AreaPickerDialog(self, start=start)
        if dlg.exec_() == QDialog.Accepted:
            x, y, w, h = dlg.area
            self.settings_obj.area_x, self.settings_obj.area_y, self.settings_obj.area_w, self.settings_obj.area_h = x, y, w, h
            self.thread.update_params(self.settings_obj)
            self.lbl_area.setText(self._area_text())
            self._append_log(f"Area set to x={x}, y={y}, w={w}, h={h}")

    def _area_text(self) -> str:
        s = self.settings_obj
        return f"({s.area_x},{s.area_y}) {s.area_w}×{s.area_h}"

    # ---- Profiles ----
    def save_profiles(self):
        name = self.profile_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Profile", "Enter a profile name first.")
            return
        s = self._read_settings_from_ui()
        if not self._validate(s): return
        self.profiles[name] = settings_to_payload(s)
        fd, tmp = tempfile.mkstemp(prefix="profiles_", suffix=".json", dir=".")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, PROFILES_FILE)
        if name not in [self.profile_combo.itemText(i) for i in range(self.profile_combo.count())]:
            self.profile_combo.addItem(name)
        self._append_log(f"Profile '{name}' saved.")

    def _load_profiles(self):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
        except Exception:
            self.profiles = {}
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(self.profiles.keys()))
        last = ""
        if hasattr(self, "_settings"):
            val = self._settings.value("last_profile", "")
            last = val if isinstance(val, str) else ""
        if last:
            idx = self.profile_combo.findText(last)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

    def load_profile(self):
        name = self.profile_combo.currentText()
        p = self.profiles.get(name)
        if not p:
            QMessageBox.warning(self, "Profile", f"No profile named '{name}'.")
            return
        s = payload_to_settings(p, self.settings_obj)
        self._apply_settings_to_ui(s)
        self.settings_obj = s
        self.thread.update_params(s)
        if hasattr(self, "_settings"):
            self._settings.setValue("last_profile", name)
        self._append_log(f"Profile '{name}' loaded.")

    def _apply_settings_to_ui(self, s: Settings):
        self.spin_interval_mean.setValue(s.interval_mean)
        self.spin_interval_std.setValue(s.interval_std)
        self.spin_duration_mean.setValue(s.duration_mean)
        self.spin_duration_std.setValue(s.duration_std)
        self.spin_run_duration.setValue(int(s.run_duration or 0))

        self.spin_first_min.setValue(s.first_rest_min)
        self.spin_first_max.setValue(s.first_rest_max)
        self.spin_sub_min.setValue(s.subsequent_rest_min)
        self.spin_sub_max.setValue(s.subsequent_rest_max)
        self.spin_rest_min.setValue(s.rest_duration_min)
        self.spin_rest_max.setValue(s.rest_duration_max)
        self.cb_human_like.setChecked(s.human_like_rests)

        self.cb_pause_on_move.setChecked(s.pause_on_mouse_move)
        self.spin_idle_after_move.setValue(s.idle_after_move_s)
        self.spin_move_speed.setValue(s.move_speed_px_per_s)

        self.combo_click_mode.setCurrentText(s.click_mode)
        self.spin_jitter.setValue(s.jitter_px)
        self.lbl_area.setText(self._area_text())

        self.cb_micro.setChecked(s.micro_rests_enabled)
        self.spin_micro_prob.setValue(s.micro_rest_prob)
        self.spin_micro_min.setValue(s.micro_rest_min_s)
        self.spin_micro_max.setValue(s.micro_rest_max_s)

    # ---- lifecycle ----
    def _on_quit(self):
        try:
            self.thread.stop_clicking()
            with QMutexLocker(self.thread.mutex):
                self.thread.cond.wakeAll()
            self.thread.wait(2000)
        except Exception:
            pass
        self._save_window()

    def closeEvent(self, e):
        self._on_quit()
        super().closeEvent(e)

    def _restore_window(self):
        geo = self._settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)
        size = self._settings.value("size")
        if size:
            try:
                w, h = map(int, str(size).split(","))
                self.resize(QSize(w, h))
            except Exception:
                self.resize(900, 800)
        else:
            self.resize(900, 800)

    def _save_window(self):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("size", f"{self.width()},{self.height()}")

# ====== Entrypoint ======
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Use system default font on macOS to avoid Roboto warning
    sys_font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    app.setFont(sys_font)

    default_settings = Settings(
        interval_mean=1.0490, interval_std=0.9313,
        duration_mean=0.1613, duration_std=0.0257,
        run_duration=None,

        first_rest_min=30, first_rest_max=45,
        subsequent_rest_min=20, subsequent_rest_max=30,
        rest_duration_min=1,  rest_duration_max=3,

        human_like_rests=True,
        micro_rests_enabled=True,
        micro_rest_prob=0.10,
        micro_rest_min_s=2.0,
        micro_rest_max_s=5.0,

        pause_on_mouse_move=True,
        idle_after_move_s=3.0,
        move_speed_px_per_s=250.0,

        click_mode="cursor",
        jitter_px=6,
        area_x=100, area_y=100, area_w=400, area_h=300
    )

    window = AutoclickerGUI(default_settings)
    window.show()
    sys.exit(app.exec_())
