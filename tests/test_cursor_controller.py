"""
Tests for CursorController — screen mapping and cooldown logic.
No display required; mouse calls are no-ops in headless mode.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import pytest

from core.cursor_controller import CursorController
from config import cfg


@pytest.fixture()
def controller():
    return CursorController()


class TestActiveZoneMapping:
    def test_active_zone_bounds_within_unit(self, controller):
        xmin, ymin, xmax, ymax = controller.active_zone
        assert 0 <= xmin < xmax <= 1
        assert 0 <= ymin < ymax <= 1

    def test_active_zone_matches_config_margin(self, controller):
        m = cfg.cursor.active_zone_margin
        xmin, ymin, xmax, ymax = controller.active_zone
        assert xmin == pytest.approx(m)
        assert xmax == pytest.approx(1.0 - m)

    def test_to_screen_center_maps_to_screen_center(self, controller):
        """Normalised (0.5, 0.5) should map to approximately screen centre."""
        sx, sy = controller._to_screen(0.5, 0.5)
        assert abs(sx - controller._screen_w / 2) < controller._screen_w * 0.05
        assert abs(sy - controller._screen_h / 2) < controller._screen_h * 0.05

    def test_to_screen_clamps_at_edges(self, controller):
        sx, sy = controller._to_screen(0.0, 0.0)
        assert sx >= 0 and sy >= 0
        sx2, sy2 = controller._to_screen(1.0, 1.0)
        assert sx2 <= controller._screen_w
        assert sy2 <= controller._screen_h


class TestClickCooldowns:
    def test_rapid_clicks_blocked_by_cooldown(self, controller):
        first  = controller.left_click()
        second = controller.left_click()   # within cooldown
        assert first is True
        assert second is False

    def test_click_allowed_after_cooldown(self, controller):
        controller.left_click()
        time.sleep(cfg.gesture.click_cooldown_sec + 0.01)
        result = controller.left_click()
        assert result is True

    def test_right_click_cooldown(self, controller):
        assert controller.right_click() is True
        assert controller.right_click() is False

    def test_double_click_cooldown(self, controller):
        assert controller.double_click() is True
        assert controller.double_click() is False


class TestScrollCooldowns:
    def test_rapid_scroll_blocked(self, controller):
        first  = controller.scroll(1)
        second = controller.scroll(1)
        assert first is True
        assert second is False

    def test_scroll_allowed_after_cooldown(self, controller):
        controller.scroll(1)
        time.sleep(cfg.gesture.scroll_cooldown_sec + 0.01)
        assert controller.scroll(-1) is True


class TestSmootherReset:
    def test_reset_smoother_no_error(self, controller):
        controller.move(0.5, 0.5)
        controller.reset_smoother()
        controller.move(0.5, 0.5)   # should not raise
