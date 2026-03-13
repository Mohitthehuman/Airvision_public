"""
HandDetector — thin wrapper around MediaPipe Hands.

Responsibilities
----------------
* Run MediaPipe detection on each BGR frame.
* Return a structured HandData object with normalised landmark positions,
  pixel positions, and convenience helpers (finger tip coords, hand width).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import cfg

# ── MediaPipe landmark indices ─────────────────────────────────────────────
WRIST           = 0
THUMB_CMC       = 1
THUMB_MCP       = 2
THUMB_IP        = 3
THUMB_TIP       = 4
INDEX_MCP       = 5
INDEX_PIP       = 6
INDEX_DIP       = 7
INDEX_TIP       = 8
MIDDLE_MCP      = 9
MIDDLE_PIP      = 10
MIDDLE_DIP      = 11
MIDDLE_TIP      = 12
RING_MCP        = 13
RING_PIP        = 14
RING_DIP        = 15
RING_TIP        = 16
PINKY_MCP       = 17
PINKY_PIP       = 18
PINKY_DIP       = 19
PINKY_TIP       = 20


@dataclass
class LandmarkPoint:
    """One MediaPipe landmark in both normalised (0-1) and pixel space."""
    x: float          # normalised x (0–1, already mirror-corrected)
    y: float          # normalised y (0–1)
    z: float          # relative depth
    px: int           # pixel x
    py: int           # pixel y


@dataclass
class HandData:
    """All information extracted from one detected hand."""
    landmarks: List[LandmarkPoint] = field(default_factory=list)
    handedness: str = "Unknown"   # "Left" or "Right" (in mirror view)

    # ── Convenience accessors ──────────────────────────────────────────

    def tip(self, idx: int) -> LandmarkPoint:
        return self.landmarks[idx]

    @property
    def index_tip(self) -> LandmarkPoint:
        return self.landmarks[INDEX_TIP]

    @property
    def middle_tip(self) -> LandmarkPoint:
        return self.landmarks[MIDDLE_TIP]

    @property
    def ring_tip(self) -> LandmarkPoint:
        return self.landmarks[RING_TIP]

    @property
    def pinky_tip(self) -> LandmarkPoint:
        return self.landmarks[PINKY_TIP]

    @property
    def thumb_tip(self) -> LandmarkPoint:
        return self.landmarks[THUMB_TIP]

    @property
    def wrist(self) -> LandmarkPoint:
        return self.landmarks[WRIST]

    # ── Geometry helpers ───────────────────────────────────────────────

    def dist(self, a: int, b: int) -> float:
        """Euclidean distance (normalised) between two landmark indices."""
        la, lb = self.landmarks[a], self.landmarks[b]
        return math.hypot(la.x - lb.x, la.y - lb.y)

    @property
    def hand_size(self) -> float:
        """Approximate hand size = wrist-to-middle-MCP distance (normalised)."""
        return max(self.dist(WRIST, MIDDLE_MCP), 1e-6)

    def normalised_dist(self, a: int, b: int) -> float:
        """Distance between two landmarks normalised by hand size."""
        return self.dist(a, b) / self.hand_size

    def is_finger_extended(self, finger_tip: int, finger_pip: int) -> bool:
        """
        Simple heuristic: finger is extended when tip is above (lower y value)
        its PIP joint, with a small margin.
        """
        tip = self.landmarks[finger_tip]
        pip = self.landmarks[finger_pip]
        return tip.y < pip.y - 0.02   # tip must be clearly above pip

    def is_thumb_extended(self) -> bool:
        """Thumb extension check uses x-axis because thumb moves laterally."""
        tip = self.landmarks[THUMB_TIP]
        ip  = self.landmarks[THUMB_IP]
        mcp = self.landmarks[THUMB_MCP]
        # Extended when tip is to the left of IP (for right hand, mirrored)
        return abs(tip.x - mcp.x) > 0.05

    @property
    def extended_fingers(self) -> List[bool]:
        """
        Returns [thumb, index, middle, ring, pinky] extension booleans.
        """
        return [
            self.is_thumb_extended(),
            self.is_finger_extended(INDEX_TIP,  INDEX_PIP),
            self.is_finger_extended(MIDDLE_TIP, MIDDLE_PIP),
            self.is_finger_extended(RING_TIP,   RING_PIP),
            self.is_finger_extended(PINKY_TIP,  PINKY_PIP),
        ]

    @property
    def finger_count(self) -> int:
        return sum(self.extended_fingers)


class HandDetector:
    """
    Wraps MediaPipe Hands for single-hand detection.

    Usage::

        detector = HandDetector()
        hand_data = detector.detect(bgr_frame)   # returns HandData or None
    """

    def __init__(self) -> None:
        self._mp_hands = mp.solutions.hands
        d = cfg.detection
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=d.max_num_hands,
            min_detection_confidence=d.min_detection_confidence,
            min_tracking_confidence=d.min_tracking_confidence,
        )

    def detect(self, bgr_frame: np.ndarray) -> Optional[HandData]:
        """
        Run detection on a BGR frame (already flipped / mirrored).
        Returns a HandData if a hand is found, else None.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mp_hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Take the first hand only
        raw_landmarks  = results.multi_hand_landmarks[0]
        raw_handedness = results.multi_handedness[0]

        landmarks: List[LandmarkPoint] = []
        for lm in raw_landmarks.landmark:
            landmarks.append(LandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z,
                px=int(lm.x * w), py=int(lm.y * h),
            ))

        handedness = raw_handedness.classification[0].label
        return HandData(landmarks=landmarks, handedness=handedness)

    def close(self) -> None:
        self._hands.close()

    # Support both explicit close() and context-manager usage
    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
