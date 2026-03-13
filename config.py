"""
Global configuration for AirVision.

All tunable parameters live here so the rest of the code stays free of
magic numbers.  Import with:

    from config import cfg
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    device_index: int = 0          # Webcam index (0 = default)
    width: int = 1280              # Capture width in pixels
    height: int = 720              # Capture height in pixels
    fps: int = 30                  # Requested capture FPS


@dataclass
class DetectionConfig:
    # MediaPipe Hands settings
    max_num_hands: int = 1
    min_detection_confidence: float = 0.75
    min_tracking_confidence: float = 0.65

    # How many frames a gesture must be stable before it fires
    gesture_hold_frames: int = 3


@dataclass
class CursorConfig:
    # Smoothing: higher alpha = more responsive but jitterier (0–1)
    smoothing_alpha: float = 0.25

    # Fraction of the camera frame used as the "active zone"
    # (keeps the useful movement area away from the edges)
    active_zone_margin: float = 0.15

    # PyAutoGUI safety: set False to allow cursor near screen edges
    failsafe: bool = True

    # Extra multiplier so small hand movements map to larger screen moves
    speed_multiplier: float = 1.0


@dataclass
class GestureConfig:
    # ── Pinch thresholds ────────────────────────────────────────────────
    # Normalised distance (relative to hand size) below which a pinch fires
    click_pinch_threshold: float = 0.045
    right_click_pinch_threshold: float = 0.045

    # ── Double-click ────────────────────────────────────────────────────
    double_click_interval_sec: float = 0.40   # max gap between two clicks

    # ── Scroll ──────────────────────────────────────────────────────────
    scroll_speed: int = 3                     # PyAutoGUI scroll units
    scroll_cooldown_sec: float = 0.12         # min time between scroll ticks

    # ── Click cooldowns ─────────────────────────────────────────────────
    click_cooldown_sec: float = 0.35

    # ── Pause gesture ───────────────────────────────────────────────────
    # Number of fingers that must be extended to trigger pause
    pause_finger_count: int = 5


@dataclass
class DisplayConfig:
    show_landmarks: bool = True
    show_fps: bool = True
    show_gesture_label: bool = True
    show_active_zone: bool = True
    window_name: str = "AirVision — Hand Gesture Control"

    # Colours (BGR)
    colour_landmark: Tuple[int, int, int] = (0, 255, 180)
    colour_connection: Tuple[int, int, int] = (200, 200, 200)
    colour_cursor_point: Tuple[int, int, int] = (0, 255, 0)
    colour_pinch_active: Tuple[int, int, int] = (0, 80, 255)
    colour_text: Tuple[int, int, int] = (255, 255, 255)
    colour_active_zone: Tuple[int, int, int] = (60, 60, 60)


@dataclass
class AirVisionConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)


# Singleton used across the project
cfg = AirVisionConfig()
