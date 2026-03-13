"""
CameraManager — wraps OpenCV VideoCapture.

Responsibilities
----------------
* Open / close the webcam.
* Expose a generator that yields BGR frames.
* Apply any camera-level settings (resolution, FPS).
"""

import cv2
import logging

from config import cfg

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Open the webcam and apply settings from config."""
        idx = cfg.camera.device_index
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {idx}.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera opened: %dx%d @ %.1f fps", actual_w, actual_h, actual_fps)

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released.")

    def read_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Return (success, bgr_frame). Flips horizontally for mirror view."""
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.flip(frame, 1)   # mirror so movements feel natural
        return ret, frame

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the actual capture stream."""
        if self._cap is None:
            return (cfg.camera.width, cfg.camera.height)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    # ------------------------------------------------------------------ #
    #  Context manager support
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()
