# AirVision — Hand Gesture Computer Control

Control your computer with hand gestures detected from your laptop's webcam.
No special hardware required — just Python, a camera, and your hand.

---

## Features

| Gesture | Action |
|---|---|
| ☝️ Index finger pointing | Move the cursor |
| 👌 Thumb + Index pinch | Left click |
| 🤌 Thumb + Middle pinch | Right click |
| 👌👌 Two rapid pinches | Double click |
| ✌️ Two fingers up, move vertically | Scroll up / down |
| 🖐️ Open palm (all 5 fingers) | Pause — freeze cursor |

---

## Architecture

```
airvision/
├── main.py                    # Entry point & main loop
├── config.py                  # All tunable parameters (single source of truth)
├── requirements.txt
│
├── core/                      # Framework layer
│   ├── camera_manager.py      # Webcam open/read/release
│   ├── hand_detector.py       # MediaPipe wrapper → HandData
│   ├── gesture_recognizer.py  # Priority queue + hold-frame stability filter
│   └── cursor_controller.py   # Screen mapping, EMA smoothing, PyAutoGUI calls
│
├── gestures/                  # One file per gesture family
│   ├── base_gesture.py        # Abstract BaseGesture + GestureResult dataclass
│   ├── cursor_gesture.py      # Single index finger → cursor move
│   ├── click_gestures.py      # Pinch → left / right / double click
│   ├── scroll_gesture.py      # Two-finger vertical motion → scroll
│   └── pause_gesture.py      # Open palm → pause
│
└── utils/
    ├── smoother.py            # Exponential moving-average smoother
    ├── fps_counter.py         # Rolling-window FPS estimator
    └── visualizer.py          # OpenCV debug overlay (landmarks, labels, FPS)
```

### How it works

```
Webcam frame
    │
    ▼
CameraManager.read_frame()        # BGR frame, horizontally flipped (mirror)
    │
    ▼
HandDetector.detect()             # MediaPipe Hands → HandData
    │                               (21 normalised landmarks + helpers)
    ▼
GestureRecognizer.recognize()     # Tries each BaseGesture in priority order
    │                               Stability filter: gesture must hold for N frames
    ▼
Action dispatch (main.py)
    ├── cursor_move  → CursorController.move()
    ├── left_click   → CursorController.left_click()
    ├── right_click  → CursorController.right_click()
    ├── double_click → CursorController.double_click()
    ├── scroll       → CursorController.scroll(direction)
    └── pause        → freeze cursor
    │
    ▼
Visualizer.draw()                 # Annotated frame → cv2.imshow()
```

### Gesture detection logic

**Cursor move** — index finger tip's normalised (x, y) is mapped from the
configurable *active zone* of the camera frame to the full screen resolution,
then passed through an Exponential Moving Average (EMA) smoother to eliminate
jitter before being forwarded to PyAutoGUI.

**Pinch click** — the normalised distance between thumb tip and index (or
middle) tip is compared against a threshold that is itself normalised by hand
size, making detection resolution-independent.

**Scroll** — index + middle extended, ring + pinky folded.  The vertical
midpoint between the two fingertips is tracked frame-to-frame; upward delta
→ scroll up, downward delta → scroll down.  A deadzone prevents accidental
triggers from hand tremors.

**Pause** — all five fingers extended (finger_count ≥ 5).  Cursor movement
and clicks are suspended; the EMA smoother is reset so no "snap" occurs on
resume.

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A webcam (built-in or USB)
- Linux / macOS / Windows

### Steps

```bash
# 1. Clone / enter the project directory
cd airvision

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Linux only) PyAutoGUI needs python3-tk and scrot for screenshots
#    and xdotool / python3-xlib for mouse control on X11
sudo apt-get install python3-tk python3-dev scrot
pip install python3-xlib
```

### Running

```bash
python main.py
```

#### Optional flags

| Flag | Description | Default |
|---|---|---|
| `--camera N` | Use webcam at device index N | 0 |
| `--alpha F` | EMA smoothing factor (0.1–0.5) | 0.25 |
| `--debug` | Verbose debug logging | off |

Example:
```bash
python main.py --camera 1 --alpha 0.3 --debug
```

Press **`q`** or close the window to exit.

---

## Configuration

All tuneable parameters live in `config.py`.  No code changes elsewhere are
needed:

```python
# config.py (excerpt)
cfg.cursor.smoothing_alpha   = 0.25   # higher = more responsive
cfg.cursor.active_zone_margin = 0.15  # fraction of frame to ignore at edges
cfg.gesture.click_pinch_threshold = 0.045
cfg.gesture.scroll_speed     = 3
cfg.display.show_landmarks   = True
```

---

## Extending AirVision

### Adding a new gesture

1. Create `gestures/my_gesture.py` subclassing `BaseGesture`.
2. Implement `name` and `check(hand: HandData) -> Optional[GestureResult]`.
3. Register it in `main.py → build_recognizer()` at the desired priority.

### Ideas for future gestures

- **Air drawing** — track index tip path while pinching with ring finger.
- **Slide control** — swipe left/right with two fingers → previous/next slide.
- **Zoom** — two-hand pinch spread.
- **Volume** — fist vertical position controls system volume.
- **Window snap** — drag gesture to snap windows left/right.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `RuntimeError: Cannot open camera at index 0` | Try `--camera 1` or check OS camera permissions |
| Cursor jitter | Lower `--alpha` (e.g. `0.15`) or increase `active_zone_margin` |
| Sluggish cursor | Raise `--alpha` (e.g. `0.4`) |
| Clicks misfiring | Raise `click_pinch_threshold` in `config.py` |
| `PyAutoGUI` fails (Linux Wayland) | Use X11 session or install `ydotool` and wrap PyAutoGUI |
| Low FPS | Reduce `cfg.camera.width/height`, or lower MediaPipe confidence thresholds |

---

## License

MIT — free to use, modify, and distribute.
