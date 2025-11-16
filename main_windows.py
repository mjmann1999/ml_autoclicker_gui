"""Windows-specific launcher for the ML Autoclicker GUI."""

import sys
import ctypes
from dataclasses import replace

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from main import Settings, build_default_settings, launch_gui


def _configure_windows_runtime():
    """Enable high-DPI support and best-effort process awareness on Windows."""

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        # Older versions of Windows or wine environments may not expose shcore.
        pass

    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def windows_default_settings() -> Settings:
    """Return defaults with tweaks that feel more natural on Windows displays."""

    base = build_default_settings()
    return replace(
        base,
        move_speed_px_per_s=325.0,
        idle_after_move_s=2.5,
        jitter_px=8,
    )


if __name__ == "__main__":
    _configure_windows_runtime()
    sys.exit(launch_gui(windows_default_settings(), force_system_font=True))
