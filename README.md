# Hand Gesture Projects

Three webcam-based hand gesture tools powered by MediaPipe.

## Setup (first time only)

```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The hand landmark model (`hand_landmarker.task`) is downloaded automatically on first run if it's missing.

---

## Projects

### Virtual Mouse
Control your mouse with hand gestures.

```bash
python virtual_mouse.py
```

| Gesture | Action |
|---|---|
| Index finger only | Move cursor |
| Index + thumb pinch (quick) | Left click |
| Index + thumb pinch (hold) | Drag |
| Peace sign (index + middle) | Scroll — move hand up/down |
| Pinky only | Right click |

Press **Q** to quit. Move mouse to any screen corner to trigger the failsafe and abort.

---

### Gesture Whiteboard
Draw on a virtual canvas with your hands.

```bash
python whiteboard.py
```

| Gesture | Action |
|---|---|
| Index finger only | Draw |
| Peace sign (index + middle) | Cycle color |
| Open palm (all fingers) | Erase |
| Fist | Grab & drag a stroke |

Press **C** to clear, **Q** to quit.

---

### Presenter Clicker
```bash
python presenter_clicker.py
```

---

## Notes

- Requires a webcam
- Works best in good lighting with a plain background behind your hand
- Run inside the `.venv` virtual environment — `pip install` outside it won't apply
