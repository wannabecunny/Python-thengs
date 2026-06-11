"""
Gesture-Controlled Virtual Mouse
─────────────────────────────────
Gestures  (single hand, right or left):

  Index only          →  MOVE       cursor follows index fingertip
  Index + thumb pinch →  DRAG       hold pinch to drag
  Peace sign (I + M)  →  SCROLL     move hand up/down to scroll
  Thumbs up           →  CLICK      single left click (debounced)
  Pinky only          →  RIGHT-CLICK (debounced)

Press  Q  to quit.
"""

import math
import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pyautogui

# ── pyautogui settings ────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # move mouse to corner to abort
pyautogui.PAUSE    = 0.0    # remove artificial inter-call delay

SCREEN_W, SCREEN_H = pyautogui.size()

# ── MediaPipe model ───────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Tuning ────────────────────────────────────────────────────────────────────
FRAME_MARGIN    = 0.18   # camera crop margin → cursor mapping zone
SMOOTH          = 0.68   # EMA weight on old position (higher = smoother / more lag)
PINCH_THRESH    = 42     # camera-pixel distance for index-thumb pinch
CLICK_COOLDOWN  = 18     # frames before another click fires (left or right)
SCROLL_SPEED    = 8      # scroll units per camera-pixel of vertical movement
PINCH_DRAG_FRAMES = 10   # hold pinch this many frames to switch from click → drag

# ── Visual accent colors (BGR) ────────────────────────────────────────────────
ACCENT = {
    "MOVE":        (0,   220, 100),
    "DRAG":        (0,   210, 230),
    "THUMBS_UP":   (0,   200, 255),   # warm yellow for left click
    "SCROLL":      (0,   180, 220),
    "RIGHT_CLICK": (100,  60, 220),
    "PINCH":       (0,   210, 230),
    "NONE":        (55,   55,  75),
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


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _OneEuro:
    def __init__(self, f_min=2.0, beta=0.15, d_cutoff=1.0):
        self.f_min, self.beta, self.d_cutoff = f_min, beta, d_cutoff
        self._x = self._dx = None; self._t = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._dx, self._t = float(x), 0.0, t; return float(x)
        dt = max(t - self._t, 1e-6); self._t = t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.f_min + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x


def _dim(color, f):
    return tuple(int(c * f) for c in color)


def _rounded_rect(img, pt1, pt2, color, radius, filled=True):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    tk = -1 if filled else 1
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, tk)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, tk)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, tk)


# ── Hand detector ──────────────────────────────────────────────────────────────

class HandDetector:
    TIPS = [8, 12, 16, 20]
    PIPS = [6, 10, 14, 18]
    MCPS = [5,  9, 13, 17]

    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.70,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._det   = mp_vision.HandLandmarker.create_from_options(opts)
        self._t0    = time.time()
        self._res   = None

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

    def _fingers_up(self, lms, w):
        thumb_up = abs(lms[4][0] - lms[2][0]) > w * 0.04
        up = [thumb_up]
        for tip, pip, mcp in zip(self.TIPS, self.PIPS, self.MCPS):
            up.append(self._is_extended(lms, tip, pip, mcp))
        return up  # [thumb, index, middle, ring, pinky]

    def classify(self, lms, w):
        thumb, index, middle, ring, pinky = self._fingers_up(lms, w)
        pinch_d = _dist(lms[4], lms[8])

        if pinch_d < PINCH_THRESH and index:
            return "PINCH"                               # click / drag
        if index and middle and not ring and not pinky:
            return "SCROLL"
        if pinky and not index and not middle and not ring:
            return "RIGHT_CLICK"
        if index and not middle and not ring and not pinky:
            return "MOVE"
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


# ── Mouse controller ───────────────────────────────────────────────────────────

class MouseController:
    def __init__(self, fw, fh):
        mx = int(fw * FRAME_MARGIN)
        my = int(fh * FRAME_MARGIN)
        self._zone = (mx, my, fw - 2*mx, fh - 2*my)   # x, y, w, h

        self._sx = SCREEN_W // 2
        self._sy = SCREEN_H // 2
        self._oe_x = _OneEuro(2.0, 0.15)
        self._oe_y = _OneEuro(2.0, 0.15)

        self._dragging     = False
        self._pinch_frames = 0    # counts consecutive frames in PINCH state
        self._scroll_prev  = None
        self._right_cd     = 0

    def _to_screen(self, cx, cy):
        zx, zy, zw, zh = self._zone
        rx = max(0.0, min(1.0, (cx - zx) / zw))
        ry = max(0.0, min(1.0, (cy - zy) / zh))
        return int(rx * SCREEN_W), int(ry * SCREEN_H)

    def _smooth_move(self, tx, ty):
        now = time.monotonic()
        self._sx = round(self._oe_x(tx, now))
        self._sy = round(self._oe_y(ty, now))
        pyautogui.moveTo(self._sx, self._sy)

    def update(self, lms, gesture):
        if self._right_cd > 0:
            self._right_cd -= 1

        if gesture == "PINCH":
            # Move cursor while pinching (needed for drag)
            tx, ty = self._to_screen(*lms[8])
            self._smooth_move(tx, ty)

            self._pinch_frames += 1
            # After holding long enough, escalate to drag
            if self._pinch_frames >= PINCH_DRAG_FRAMES and not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True

            self._scroll_prev = None

        else:
            # Pinch just ended — decide click vs drag release
            if self._pinch_frames > 0:
                if self._dragging:
                    pyautogui.mouseUp()
                    self._dragging = False
                else:
                    pyautogui.click()       # short pinch = single left click
            self._pinch_frames = 0

            if gesture == "MOVE":
                tx, ty = self._to_screen(*lms[8])
                self._smooth_move(tx, ty)
                self._scroll_prev = None

            elif gesture == "SCROLL":
                mid_y = (lms[8][1] + lms[12][1]) // 2
                if self._scroll_prev is not None:
                    delta = self._scroll_prev - mid_y   # up = positive
                    if abs(delta) > 3:
                        pyautogui.scroll(int(delta * SCROLL_SPEED / 10))
                self._scroll_prev = mid_y

            elif gesture == "RIGHT_CLICK":
                self._scroll_prev = None
                if self._right_cd == 0:
                    pyautogui.rightClick()
                    self._right_cd = CLICK_COOLDOWN * 3

            else:
                self._scroll_prev = None

    def _release_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False
        self._pinch_frames = 0

    def release_all(self):
        self._release_drag()

    def zone_rect(self):
        zx, zy, zw, zh = self._zone
        return (zx, zy), (zx + zw, zy + zh)


# ── HUD drawing ────────────────────────────────────────────────────────────────

def draw_hud(frame, gesture, dragging, pinch_d, screen_pos):
    h, w = frame.shape[:2]
    col  = ACCENT.get(gesture, ACCENT["NONE"])

    # ── Top bar ───────────────────────────────────────────────────────────────
    top_panel = frame.copy()
    cv2.rectangle(top_panel, (0, 0), (w, 30), (8, 8, 18), -1)
    cv2.addWeighted(top_panel, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (0, 30), (w, 30), _dim(col, 0.35), 1)
    cv2.putText(frame, "Virtual Mouse",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 110), 1, cv2.LINE_AA)
    quit_hint = "Q: Quit"
    qw, _ = cv2.getTextSize(quit_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, quit_hint, (w - qw - 10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 110), 1, cv2.LINE_AA)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    BAR = 64
    by  = h - BAR
    panel = frame.copy()
    cv2.rectangle(panel, (0, by), (w, h), (8, 8, 18), -1)
    cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, by), (w, by), _dim(col, 0.40), 2)

    # Mode badge
    LABELS = {
        "MOVE": "MOVE", "PINCH": "DRAG" if dragging else "CLICK",
        "SCROLL": "SCROLL", "RIGHT_CLICK": "R-CLICK", "NONE": "IDLE",
    }
    label = LABELS.get(gesture, gesture)
    font  = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(label, font, 0.58, 1)
    px, py  = 10, 6
    bx1, by1 = 12, by + BAR//2 - th//2 - py
    bx2, by2 = bx1 + tw + px*2, by + BAR//2 + th//2 + py + bl
    _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(col, 0.25), 8)
    _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(col, 0.70), 8, filled=False)
    cv2.putText(frame, label, (bx1 + px, by2 - py - bl),
                font, 0.58, (230, 230, 235), 1, cv2.LINE_AA)

    # Pinch distance bar (pill shape)
    if gesture in ("MOVE", "PINCH", "NONE"):
        bar_x  = bx2 + 16
        bar_y  = by + BAR // 2
        bar_w  = 80
        bar_r  = 4
        frac   = max(0.0, min(1.0, 1.0 - pinch_d / (PINCH_THRESH * 2.5)))
        cv2.rectangle(frame, (bar_x + bar_r, bar_y - bar_r),
                      (bar_x + bar_w - bar_r, bar_y + bar_r), (30, 30, 45), -1)
        cv2.circle(frame, (bar_x + bar_r, bar_y), bar_r, (30, 30, 45), -1)
        cv2.circle(frame, (bar_x + bar_w - bar_r, bar_y), bar_r, (30, 30, 45), -1)
        filled = int(frac * bar_w)
        if filled > bar_r:
            fill_col = (0, 120, 255) if frac > 0.7 else _dim(col, 0.8)
            cv2.rectangle(frame, (bar_x + bar_r, bar_y - bar_r),
                          (bar_x + filled - bar_r, bar_y + bar_r), fill_col, -1)
            cv2.circle(frame, (bar_x + bar_r, bar_y), bar_r, fill_col, -1)
            cv2.circle(frame, (bar_x + filled - bar_r, bar_y), bar_r, fill_col, -1)
        cv2.putText(frame, "pinch", (bar_x, bar_y - bar_r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 100), 1, cv2.LINE_AA)

    # Screen position
    if screen_pos:
        sx, sy = screen_pos
        pos_str = f"cursor  {sx} , {sy}"
        pw, _ = cv2.getTextSize(pos_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(frame, pos_str, (w - pw - 12, by + BAR//2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 140), 1, cv2.LINE_AA)


def draw_guide(frame):
    h, w = frame.shape[:2]
    guide = [
        ("Index only",         "MOVE",    "MOVE"),
        ("Index + thumb close","CLICK",   "CLICK"),
        ("Peace sign (2 up)",  "SCROLL",  "SCROLL"),
        ("Pinky only",         "R-CLICK", "RIGHT_CLICK"),
    ]
    gw, gh = 250, 30 + len(guide)*26 + 10
    gx = (w - gw) // 2
    gy = (h - 64 - gh) // 2
    bg = frame.copy()
    _rounded_rect(bg, (gx, gy), (gx+gw, gy+gh), (10, 10, 22), 10)
    cv2.addWeighted(bg, 0.80, frame, 0.20, 0, frame)
    _rounded_rect(frame, (gx, gy), (gx+gw, gy+gh), (40, 40, 65), 10, filled=False)
    cv2.putText(frame, "Show your hand",
                (gx+16, gy+22),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (160, 160, 200), 1, cv2.LINE_AA)
    for k, (action, mode_lbl, gkey) in enumerate(guide):
        lx = gx + 16
        ly = gy + 48 + k*26
        ac = ACCENT.get(gkey, (80, 80, 100))
        _rounded_rect(frame, (lx, ly-14), (lx+72, ly+5), _dim(ac, 0.25), 4)
        cv2.putText(frame, mode_lbl, (lx+5, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _dim(ac, 0.9), 1, cv2.LINE_AA)
        cv2.putText(frame, action, (lx+82, ly),
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

    detector = HandDetector()
    mouse    = MouseController(fw, fh)
    grabber  = FrameGrabber(cap)

    cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)

    while True:
        frame = grabber.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector.process(rgb)
        lms = detector.landmarks(fw, fh)

        canvas = np.zeros((fh, fw, 3), dtype=np.uint8)

        # Draw mapping zone boundary
        z1, z2 = mouse.zone_rect()
        cv2.rectangle(canvas, z1, z2, (35, 35, 60), 1)
        cv2.putText(canvas, "mapping zone",
                    (z1[0] + 4, z1[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 80), 1, cv2.LINE_AA)

        gesture  = "NONE"
        pinch_d  = 999.0
        hand_vis = lms is not None

        if lms:
            gesture = detector.classify(lms, fw)
            pinch_d = _dist(lms[4], lms[8])
            mouse.update(lms, gesture)
            detector.draw_skeleton(canvas, lms, gesture)

            # Cursor dot at index tip with glow rings
            col = ACCENT.get(gesture, ACCENT["NONE"])
            if gesture == "PINCH":
                cv2.circle(canvas, lms[8], 18, _dim(col, 0.10), -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8], 12, _dim(col, 0.30), -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8],  6, col,              -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8],  2, (255, 255, 255),  -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8], 18, _dim(col, 0.60),   1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, lms[8], 14, _dim(col, 0.12), -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8], 10, _dim(col, 0.25), -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8],  5, col,              -1, cv2.LINE_AA)
                cv2.circle(canvas, lms[8],  2, (255, 255, 255),  -1, cv2.LINE_AA)
        else:
            mouse._release_drag()

        screen_pos = (mouse._sx, mouse._sy) if hand_vis else None
        draw_hud(canvas, gesture, mouse._dragging, pinch_d, screen_pos)

        if not hand_vis:
            draw_guide(canvas)

        cv2.imshow("Virtual Mouse", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mouse.release_all()
    grabber.stop()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
