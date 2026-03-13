"""
AirVision — Main entry point.

Run with:
    python main.py

Optional flags (edit config.py for persistent changes):
    --camera   INT   webcam device index (default 0)
    --alpha    FLOAT smoothing alpha, 0-1 (default 0.25)
    --debug          enable verbose logging
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from config import cfg
from core.camera_manager import CameraManager
from core.cursor_controller import CursorController
from core.gesture_recognizer import GestureRecognizer
from core.hand_detector import HandDetector
from gestures.click_gestures import LeftClickGesture, RightClickGesture
from gestures.cursor_gesture import CursorGesture
from gestures.pause_gesture import PauseGesture
from gestures.scroll_gesture import ScrollGesture
from utils.fps_counter import FPSCounter
from utils.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AirVision hand gesture controller")
    p.add_argument("--camera", type=int, default=None, help="Webcam device index")
    p.add_argument("--alpha",  type=float, default=None, help="EMA smoothing alpha (0-1)")
    p.add_argument("--debug",  action="store_true", help="Verbose logging")
    return p.parse_args()


def configure(args: argparse.Namespace) -> None:
    """Apply CLI overrides to the global config singleton."""
    if args.camera is not None:
        cfg.camera.device_index = args.camera
    if args.alpha is not None:
        cfg.cursor.smoothing_alpha = float(args.alpha)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def build_recognizer() -> GestureRecognizer:
    """
    Create and return the gesture recognizer with all gestures registered
    in priority order (highest priority first).

    Priority order rationale
    ------------------------
    1. Pause   — must win so user can always stop interaction
    2. Scroll  — two-finger pose is unambiguous; check before clicks
    3. Left click / double click
    4. Right click
    5. Cursor move — fallback for the single pointing finger
    """
    gestures = [
        PauseGesture(),
        ScrollGesture(),
        LeftClickGesture(),
        RightClickGesture(),
        CursorGesture(),
    ]
    return GestureRecognizer(gestures)


def run_loop(
    camera: CameraManager,
    detector: HandDetector,
    recognizer: GestureRecognizer,
    controller: CursorController,
    visualizer: Visualizer,
    fps_counter: FPSCounter,
) -> None:
    """Main processing loop. Exits on 'q' keypress or window close."""
    logger = logging.getLogger("main")
    paused = False

    logger.info("AirVision started.  Press 'q' to quit.")

    while True:
        ok, frame = camera.read_frame()
        if not ok or frame is None:
            logger.warning("Failed to read frame — retrying…")
            continue

        fps = fps_counter.tick()

        # ── Hand detection ──────────────────────────────────────────────
        hand = detector.detect(frame)

        # ── Gesture recognition ─────────────────────────────────────────
        gesture = recognizer.recognize(hand) if hand else None

        # ── Action dispatch ─────────────────────────────────────────────
        if gesture is not None:
            name = gesture.name

            if name == "pause":
                if not paused:
                    paused = True
                    controller.reset_smoother()
                    logger.debug("Cursor PAUSED")

            else:
                if paused:
                    paused = False
                    logger.debug("Cursor RESUMED")

                if name == "cursor_move":
                    controller.move(gesture.data["x"], gesture.data["y"])

                elif name == "left_click":
                    controller.move(gesture.data.get("x", 0), gesture.data.get("y", 0)) if "x" in gesture.data else None
                    controller.left_click()

                elif name == "double_click":
                    controller.double_click()

                elif name == "right_click":
                    controller.right_click()

                elif name == "scroll":
                    controller.scroll(gesture.data["direction"])

        elif paused is True and hand is None:
            # Hand left frame; resume automatically
            paused = False
            controller.reset_smoother()

        # ── Visualisation ───────────────────────────────────────────────
        frame = visualizer.draw(
            frame=frame,
            hand=hand,
            gesture=gesture,
            fps=fps,
            active_zone=controller.active_zone,
            paused=paused,
        )

        cv2.imshow(cfg.display.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(cfg.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Quit requested.")
            break


def main() -> None:
    args = parse_args()
    configure(args)
    logger = logging.getLogger("main")

    camera     = CameraManager()
    detector   = HandDetector()
    recognizer = build_recognizer()
    controller = CursorController()
    visualizer = Visualizer()
    fps_counter = FPSCounter(window=30)

    try:
        camera.open()
        run_loop(camera, detector, recognizer, controller, visualizer, fps_counter)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("AirVision shut down cleanly.")


if __name__ == "__main__":
    main()
