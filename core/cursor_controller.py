"""
CursorController — translates normalised hand positions to screen coordinates
and drives the OS mouse via PyAutoGUI.

Features
--------
* Maps from a configurable "active zone" of the camera frame to the full
  screen, so users don't have to move their hand to the extreme edges.
* Applies exponential moving-average smoothing to eliminate jitter.
* Exposes left_click(), right_click(), double_click(), and scroll().
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

try:
    import pyautogui
    pyautogui.PAUSE = 0          # disable built-in pause for real-time use
    pyautogui.FAILSAFE = True    # overridden in __init__ after config loads
    _PYAUTOGUI_AVAILABLE = True
except (KeyError, Exception):
    # Headless / no-display environment (e.g. CI, server).
    # The rest of the module still loads; mouse calls become no-ops.
    pyautogui = None              # type: ignore[assignment]
    _PYAUTOGUI_AVAILABLE = False

from config import cfg
from utils.smoother import ExponentialSmoother

logger = logging.getLogger(__name__)


class CursorController:
    """
    Owns the mapping from camera-space to screen-space and the mouse actions.
    """

    def __init__(self) -> None:
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = cfg.cursor.failsafe
            self._screen_w, self._screen_h = pyautogui.size()
        else:
            logger.warning("PyAutoGUI unavailable (no display). Mouse actions will be no-ops.")
            self._screen_w, self._screen_h = 1920, 1080  # safe fallback
        self._smoother = ExponentialSmoother(alpha=cfg.cursor.smoothing_alpha)

        # Active zone margins (fraction of frame to ignore at each edge)
        m = cfg.cursor.active_zone_margin
        self._zone_x_min = m
        self._zone_x_max = 1.0 - m
        self._zone_y_min = m
        self._zone_y_max = 1.0 - m

        self._last_click_time: float = 0.0
        self._last_scroll_time: float = 0.0

        logger.info(
            "CursorController ready. Screen: %dx%d",
            self._screen_w, self._screen_h,
        )

    # ------------------------------------------------------------------ #
    #  Cursor movement
    # ------------------------------------------------------------------ #

    def move(self, norm_x: float, norm_y: float) -> None:
        """
        Move the cursor to the position indicated by the normalised (0-1)
        camera coordinates, mapping through the active zone.
        """
        sx, sy = self._to_screen(norm_x, norm_y)
        sx_smooth, sy_smooth = self._smoother.smooth(sx, sy)
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.moveTo(sx_smooth, sy_smooth)

    def _to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """Map normalised coordinates inside the active zone to screen coords."""
        # Clamp to active zone
        zx = max(self._zone_x_min, min(self._zone_x_max, norm_x))
        zy = max(self._zone_y_min, min(self._zone_y_max, norm_y))

        # Re-scale from zone space to [0, 1]
        rel_x = (zx - self._zone_x_min) / (self._zone_x_max - self._zone_x_min)
        rel_y = (zy - self._zone_y_min) / (self._zone_y_max - self._zone_y_min)

        # Apply speed multiplier (clamped so cursor stays on screen)
        sx = rel_x * self._screen_w * cfg.cursor.speed_multiplier
        sy = rel_y * self._screen_h * cfg.cursor.speed_multiplier
        sx = max(0, min(self._screen_w - 1, sx))
        sy = max(0, min(self._screen_h - 1, sy))
        return sx, sy

    # ------------------------------------------------------------------ #
    #  Click actions
    # ------------------------------------------------------------------ #

    def left_click(self) -> bool:
        """
        Fire a left click if the cooldown has elapsed.
        Returns True if the click was sent.
        """
        now = time.time()
        if now - self._last_click_time < cfg.gesture.click_cooldown_sec:
            return False
        self._last_click_time = now
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.click()
        logger.debug("Left click")
        return True

    def right_click(self) -> bool:
        """Fire a right click if the cooldown has elapsed."""
        now = time.time()
        if now - self._last_click_time < cfg.gesture.click_cooldown_sec:
            return False
        self._last_click_time = now
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.rightClick()
        logger.debug("Right click")
        return True

    def double_click(self) -> bool:
        """Fire a double click if the cooldown has elapsed."""
        now = time.time()
        if now - self._last_click_time < cfg.gesture.click_cooldown_sec:
            return False
        self._last_click_time = now
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.doubleClick()
        logger.debug("Double click")
        return True

    # ------------------------------------------------------------------ #
    #  Scroll
    # ------------------------------------------------------------------ #

    def scroll(self, direction: int) -> bool:
        """
        Scroll the active window.

        Parameters
        ----------
        direction: +1 = scroll up, -1 = scroll down
        """
        now = time.time()
        if now - self._last_scroll_time < cfg.gesture.scroll_cooldown_sec:
            return False
        self._last_scroll_time = now
        units = cfg.gesture.scroll_speed * direction
        if _PYAUTOGUI_AVAILABLE:
            pyautogui.scroll(units)
        logger.debug("Scroll %s", "up" if direction > 0 else "down")
        return True

    # ------------------------------------------------------------------ #
    #  Active zone bounds (for overlay drawing)
    # ------------------------------------------------------------------ #

    @property
    def active_zone(self) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) in normalised [0,1] space."""
        return (
            self._zone_x_min,
            self._zone_y_min,
            self._zone_x_max,
            self._zone_y_max,
        )

    def reset_smoother(self) -> None:
        """Call when cursor was paused so the smoother doesn't lerp from stale data."""
        self._smoother.reset()
