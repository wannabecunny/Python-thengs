"""
Gesture Presenter Clicker
─────────────────────────
Control a PowerPoint or PDF presentation without a physical remote.

  Open palm  (4+ fingers up)  →  Next slide   [ → ]
  Closed fist (no fingers up) →  Prev slide   [ ← ]
  Thumbs up  (thumb only up)  →  Exit         [ Esc ]

Hold any gesture for 0.6 s to trigger.  The gesture must return to
neutral before the same action can fire again (one shot per show).

Press  Q  to quit.
"""

import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui

# ── pyautogui settings ────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.0

# ── MediaPipe model ───────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Tuning ────────────────────────────────────────────────────────────────────
HOLD_SECS = 0.6    # how long to hold a gesture before it fires

# ── Visual accent colors (BGR) ────────────────────────────────────────────────
ACCENT = {
    "OPEN_PALM":   (0,  220, 100),   # green
    "CLOSED_FIST": (0,   80, 220),   # orange-red
    "THUMBS_UP":   (0,  210, 230),   # yellow
    "NONE":        (55,  55,  75),
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class FrameGrabber:
    def __init__(self, cap):
        self._cap = cap
        self._buf = deque(maxlen=1)
        self._running = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self._buf.append(frame)

    def read(self):
        return self._buf[-1] if self._buf else None

    def stop(self):
        self._running = False
        self._t.join(timeout=1)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~3 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


def _dim(color, f):
    return tuple(int(c * f) for c in color)


class _GestureDebouncer:
    def __init__(self, window=5):
        self._buf = deque(maxlen=window)
    def update(self, g):
        self._buf.append(g)
        return max(set(self._buf), key=self._buf.count)


def _rounded_rect(img, pt1, pt2, color, radius, filled=True):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    tk = -1 if filled else 1
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, tk)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, tk)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, tk)


# ── Face blurring ─────────────────────────────────────────────────────────────

class FaceBlur:
    """Detects and pixelates faces using the built-in Haar cascade."""

    def __init__(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._clf   = cv2.CascadeClassifier(path)
        self._tick  = 0
        self._faces = []

    def apply(self, frame):
        # Run detection every 3 frames; reuse cached boxes otherwise
        self._tick = (self._tick + 1) % 3
        if self._tick == 0:
            small = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            result = self._clf.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=5, minSize=(40, 40))
            self._faces = result if len(result) else []
        faces = self._faces
        for (x, y, w, h) in faces:
            # Scale coordinates back to full resolution and add padding
            x, y, w, h = x*2, y*2, w*2, h*2
            pad = int(w * 0.20)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]
            # Pixelate: shrink then enlarge for a mosaic effect
            block = 12
            rh, rw = roi.shape[:2]
            tiny  = cv2.resize(roi, (max(1, rw//block), max(1, rh//block)),
                               interpolation=cv2.INTER_LINEAR)
            frame[y1:y2, x1:x2] = cv2.resize(tiny, (rw, rh),
                                              interpolation=cv2.INTER_NEAREST)
        return frame


# ── Hand detector ──────────────────────────────────────────────────────────────

class HandDetector:
    TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky tips
    PIPS = [6, 10, 14, 18]   # corresponding PIP joints
    MCPS = [5,  9, 13, 17]   # corresponding MCP (base knuckle) joints

    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._det = mp_vision.HandLandmarker.create_from_options(opts)
        self._t0  = time.time()
        self._res = None

    def process(self, rgb):
        ts = int((time.time() - self._t0) * 1000)
        self._res = self._det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )

    def landmarks(self, w, h):
        if not self._res or not self._res.hand_landmarks:
            return None
        hand = self._res.hand_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand]

    @staticmethod
    def _is_extended(lms, tip, pip, mcp):
        ax=lms[pip][0]-lms[mcp][0]; ay=lms[pip][1]-lms[mcp][1]
        tx=lms[tip][0]-lms[pip][0]; ty=lms[tip][1]-lms[pip][1]
        return ax*tx+ay*ty>0

    def classify(self, lms):
        fingers = [self._is_extended(lms, tip, pip, mcp)
                   for tip, pip, mcp in zip(self.TIPS, self.PIPS, self.MCPS)]
        n_up = sum(fingers)

        if n_up >= 4:
            return "OPEN_PALM"

        # Thumb-up: tip must rise > 30% of the hand's own height above the
        # middle-finger knuckle (lm 9). Scaling by hand height makes the check
        # camera-distance independent and immune to fist thumb-wrap.
        hand_h   = lms[0][1] - lms[9][1]   # wrist y − middle MCP y (always > 0)
        thumb_up = hand_h > 0 and (lms[9][1] - lms[4][1]) > hand_h * 0.30

        if n_up == 0 and not thumb_up:
            return "CLOSED_FIST"
        if thumb_up and n_up == 0:
            return "THUMBS_UP"
        return "NONE"

    def draw_skeleton(self, frame, lms, gesture):
        col  = ACCENT.get(gesture, ACCENT["NONE"])
        bone = _dim(col, 0.35)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, lms[a], lms[b], bone, 2, cv2.LINE_AA)
        tips = {4, 8, 12, 16, 20}
        for i, pt in enumerate(lms):
            if i in tips:
                cv2.circle(frame, pt, 6, _dim(col, 0.30), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, _dim(col, 0.85), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 3, _dim(col, 0.55), -1, cv2.LINE_AA)

    def close(self):
        self._det.close()


# ── Slide controller ───────────────────────────────────────────────────────────

class SlideController:
    ACTIONS = {
        "OPEN_PALM":   ("right",  "Next Slide  →"),
        "CLOSED_FIST": ("left",   "←  Prev Slide"),
        "THUMBS_UP":   ("escape", "Exit Presentation"),
    }

    def __init__(self):
        self._current  = "NONE"
        self._start    = 0.0
        self._fired    = False   # True once this gesture instance has triggered
        self.last_sent = ""

    def update(self, gesture):
        """Call every frame. Returns (fired_label, hold_fraction 0–1)."""
        now = time.time()

        # New gesture → reset timer and fired flag
        if gesture != self._current:
            self._current = gesture
            self._start   = now
            self._fired   = False

        # Nothing to do while neutral or after the gesture already fired
        if gesture == "NONE" or self._fired:
            return "", 0.0

        elapsed   = now - self._start
        hold_frac = min(1.0, elapsed / HOLD_SECS)

        if elapsed >= HOLD_SECS:
            key, label = self.ACTIONS[gesture]
            pyautogui.press(key)
            self.last_sent = label
            self._fired    = True
            return label, 1.0

        return "", hold_frac


# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(frame, gesture, hold_frac, last_sent):
    h, w = frame.shape[:2]
    col  = ACCENT.get(gesture, ACCENT["NONE"])

    # ── Top bar ───────────────────────────────────────────────────────────────
    top_panel = frame.copy()
    cv2.rectangle(top_panel, (0, 0), (w, 30), (8, 8, 18), -1)
    cv2.addWeighted(top_panel, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, 30), (w, 30), _dim(col, 0.35), 1)
    cv2.putText(frame, "Presenter Clicker",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 110), 1, cv2.LINE_AA)
    hint = "Q: Quit"
    hw, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, hint, (w - hw - 10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 110), 1, cv2.LINE_AA)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    BAR = 72
    by  = h - BAR
    panel = frame.copy()
    cv2.rectangle(panel, (0, by), (w, h), (8, 8, 18), -1)
    cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, by), (w, by), _dim(col, 0.40), 2)

    # Gesture badge
    LABELS = {
        "OPEN_PALM":   "NEXT",
        "CLOSED_FIST": "PREV",
        "THUMBS_UP":   "EXIT",
        "NONE":        "IDLE",
    }
    label = LABELS.get(gesture, "IDLE")
    font  = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(label, font, 0.58, 1)
    px, py = 10, 6
    bx1 = 12
    by1 = by + BAR//2 - th//2 - py
    bx2 = bx1 + tw + px*2
    by2 = by + BAR//2 + th//2 + py + bl
    _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(col, 0.25), 8)
    _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(col, 0.70), 8, filled=False)
    cv2.putText(frame, label, (bx1 + px, by2 - py - bl),
                font, 0.58, (230, 230, 235), 1, cv2.LINE_AA)

    # Hold progress bar (pill shape)
    bar_x = bx2 + 16
    bar_y = by + BAR // 2
    bar_w = 120
    bar_r = 4
    cv2.rectangle(frame, (bar_x + bar_r, bar_y - bar_r),
                  (bar_x + bar_w - bar_r, bar_y + bar_r), (30, 30, 45), -1)
    cv2.circle(frame, (bar_x + bar_r, bar_y), bar_r, (30, 30, 45), -1)
    cv2.circle(frame, (bar_x + bar_w - bar_r, bar_y), bar_r, (30, 30, 45), -1)
    filled = int(hold_frac * bar_w)
    if filled > bar_r:
        fill_col = (255, 255, 255) if hold_frac >= 1.0 else col
        cv2.rectangle(frame, (bar_x + bar_r, bar_y - bar_r),
                      (bar_x + filled - bar_r, bar_y + bar_r), fill_col, -1)
        cv2.circle(frame, (bar_x + bar_r, bar_y), bar_r, fill_col, -1)
        cv2.circle(frame, (bar_x + filled - bar_r, bar_y), bar_r, fill_col, -1)
    cv2.putText(frame, "hold", (bar_x, bar_y - bar_r - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (80, 80, 100), 1, cv2.LINE_AA)

    # Last action sent
    if last_sent:
        ls = f"sent: {last_sent}"
        lw, _ = cv2.getTextSize(ls, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(frame, ls, (w - lw - 12, by + BAR//2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 140), 1, cv2.LINE_AA)


def draw_guide(frame):
    h, w = frame.shape[:2]
    guide = [
        ("Open palm",   "→  Next slide",   "OPEN_PALM"),
        ("Closed fist", "←  Prev slide",   "CLOSED_FIST"),
        ("Thumbs up",   "Esc  Exit",        "THUMBS_UP"),
    ]
    gw, gh = 280, 30 + len(guide)*26 + 10
    gx = (w - gw) // 2
    gy = (h - 72 - gh) // 2
    bg = frame.copy()
    _rounded_rect(bg, (gx, gy), (gx+gw, gy+gh), (10, 10, 22), 10)
    cv2.addWeighted(bg, 0.80, frame, 0.20, 0, frame)
    _rounded_rect(frame, (gx, gy), (gx+gw, gy+gh), (40, 40, 65), 10, filled=False)
    cv2.putText(frame, "Show your hand to begin",
                (gx+16, gy+22),
                cv2.FONT_HERSHEY_DUPLEX, 0.48, (160, 160, 200), 1, cv2.LINE_AA)
    for k, (gesture_name, action, gkey) in enumerate(guide):
        lx = gx + 16
        ly = gy + 48 + k*26
        ac = ACCENT.get(gkey, (80, 80, 100))
        _rounded_rect(frame, (lx, ly-14), (lx+94, ly+5), _dim(ac, 0.25), 4)
        cv2.putText(frame, gesture_name, (lx+5, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _dim(ac, 0.9), 1, cv2.LINE_AA)
        cv2.putText(frame, action, (lx+106, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 165), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector   = HandDetector()
    controller = SlideController()
    debouncer  = _GestureDebouncer()
    face_blur  = FaceBlur()
    grabber    = FrameGrabber(cap)

    cv2.namedWindow("Presenter Clicker", cv2.WINDOW_NORMAL)

    while True:
        frame = grabber.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)
        face_blur.apply(frame)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector.process(rgb)
        lms = detector.landmarks(fw, fh)

        gesture = "NONE"
        if lms:
            gesture = debouncer.update(detector.classify(lms))
            detector.draw_skeleton(frame, lms, gesture)

        _, hold_frac = controller.update(gesture)
        draw_hud(frame, gesture, hold_frac, controller.last_sent)

        if not lms:
            draw_guide(frame)

        cv2.imshow("Presenter Clicker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    grabber.stop()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
