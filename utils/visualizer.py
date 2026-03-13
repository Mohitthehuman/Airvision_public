"""
Visualizer — draws debug overlays on the camera frame.

Draws
-----
* MediaPipe hand landmarks and connections
* Active zone rectangle
* Current gesture label
* FPS counter
* Cursor control point indicator
* Pinch indicator when a click gesture is active
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import cfg
from core.hand_detector import HandData, INDEX_TIP, THUMB_TIP, MIDDLE_TIP
from gestures.base_gesture import GestureResult


_mp_drawing = mp.solutions.drawing_utils
_mp_hands   = mp.solutions.hands

# Reusable drawing specs created once
_LANDMARK_SPEC   = _mp_drawing.DrawingSpec(color=cfg.display.colour_landmark,   thickness=2, circle_radius=3)
_CONNECTION_SPEC = _mp_drawing.DrawingSpec(color=cfg.display.colour_connection, thickness=1)


class Visualizer:
    """Stateless helper that annotates OpenCV frames in-place."""

    def draw(
        self,
        frame: np.ndarray,
        hand: Optional[HandData],
        gesture: Optional[GestureResult],
        fps: float,
        active_zone: Tuple[float, float, float, float],
        paused: bool,
    ) -> np.ndarray:
        """
        Annotate *frame* (modified in place) and return it.

        Parameters
        ----------
        frame       : BGR frame from the webcam
        hand        : HandData or None
        gesture     : recognised GestureResult or None
        fps         : current frames per second
        active_zone : (x_min, y_min, x_max, y_max) normalised
        paused      : True if cursor control is paused
        """
        h, w = frame.shape[:2]

        # ── Active zone ──────────────────────────────────────────────────
        if cfg.display.show_active_zone:
            xm, ym, xx, yx = active_zone
            pt1 = (int(xm * w), int(ym * h))
            pt2 = (int(xx * w), int(yx * h))
            cv2.rectangle(frame, pt1, pt2, cfg.display.colour_active_zone, 1)
            cv2.putText(
                frame, "active zone", (pt1[0] + 4, pt1[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, cfg.display.colour_active_zone, 1,
            )

        if hand is None:
            self._draw_no_hand(frame, w, h, fps)
            return frame

        # ── Landmarks ────────────────────────────────────────────────────
        if cfg.display.show_landmarks:
            self._draw_landmarks(frame, hand)

        # ── Cursor control point (index fingertip) ───────────────────────
        tip = hand.landmarks[INDEX_TIP]
        color = cfg.display.colour_pinch_active if paused else cfg.display.colour_cursor_point
        cv2.circle(frame, (tip.px, tip.py), 10, color, cv2.FILLED)
        cv2.circle(frame, (tip.px, tip.py), 12, (255, 255, 255), 1)

        # ── Pinch visualisation ──────────────────────────────────────────
        if gesture and gesture.name in ("left_click", "double_click"):
            t = hand.landmarks[THUMB_TIP]
            i = hand.landmarks[INDEX_TIP]
            cv2.line(frame, (t.px, t.py), (i.px, i.py), cfg.display.colour_pinch_active, 2)
            cv2.circle(frame, (t.px, t.py), 8, cfg.display.colour_pinch_active, cv2.FILLED)

        if gesture and gesture.name == "right_click":
            t = hand.landmarks[THUMB_TIP]
            m = hand.landmarks[MIDDLE_TIP]
            cv2.line(frame, (t.px, t.py), (m.px, m.py), (0, 128, 255), 2)
            cv2.circle(frame, (t.px, t.py), 8, (0, 128, 255), cv2.FILLED)

        # ── Gesture label ────────────────────────────────────────────────
        if cfg.display.show_gesture_label:
            label = gesture.name.replace("_", " ").upper() if gesture else ""
            if paused:
                label = "PAUSED"
            if label:
                cv2.putText(
                    frame, label,
                    (w // 2 - len(label) * 7, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if paused else cfg.display.colour_text,
                    2,
                )

        # ── Finger count HUD ─────────────────────────────────────────────
        ext = hand.extended_fingers
        icons = ["T", "I", "M", "R", "P"]
        for i, (ext_flag, icon) in enumerate(zip(ext, icons)):
            col = (0, 220, 0) if ext_flag else (80, 80, 80)
            cv2.putText(
                frame, icon, (10 + i * 22, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2,
            )

        # ── FPS ──────────────────────────────────────────────────────────
        if cfg.display.show_fps:
            self._draw_fps(frame, fps)

        return frame

    # ── Private helpers ───────────────────────────────────────────────────

    def _draw_landmarks(self, frame: np.ndarray, hand: HandData) -> None:
        """Manually draw each landmark and connection for full colour control."""
        h, w = frame.shape[:2]
        lms = hand.landmarks

        # Connections (subset of MediaPipe's standard graph)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (9, 10), (10, 11), (11, 12),
            # Ring
            (13, 14), (14, 15), (15, 16),
            # Pinky
            (17, 18), (18, 19), (19, 20),
            # Palm
            (0, 17), (5, 9), (9, 13), (13, 17),
        ]
        for a, b in connections:
            cv2.line(
                frame,
                (lms[a].px, lms[a].py),
                (lms[b].px, lms[b].py),
                cfg.display.colour_connection, 1,
            )

        for lm in lms:
            cv2.circle(frame, (lm.px, lm.py), 4, cfg.display.colour_landmark, cv2.FILLED)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        text = f"FPS: {fps:.0f}"
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    cfg.display.colour_text, 2)

    def _draw_no_hand(self, frame: np.ndarray, w: int, h: int, fps: float) -> None:
        msg = "No hand detected — show your hand to the camera"
        cv2.putText(
            frame, msg, (w // 2 - 280, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 255), 2,
        )
        if cfg.display.show_fps:
            self._draw_fps(frame, fps)
