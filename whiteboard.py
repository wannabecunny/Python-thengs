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


MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

GESTURE_ACCENT = {
    "DRAW":        (0, 220, 100),
    "ERASE":       (40,  70, 230),
    "GRAB":        (0, 210, 230),
    "COLOR_CYCLE": (0, 180, 220),
    "NONE":        (55,  55,  75),
}

# Priority order for choosing the mode badge when both hands are active
GESTURE_PRIORITY = ["DRAW", "ERASE", "GRAB", "COLOR_CYCLE", "NONE"]

GRAB_THRESHOLD = 80   # max pixel distance to select a stroke
N_HANDS        = 2    # maximum simultaneous hands
SMOOTH         = 0.55 # draw smoothing factor (higher = smoother / more lag)


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


def _dim(color, factor):
    return tuple(int(c * factor) for c in color)


class _OneEuro:
    def __init__(self, f_min=1.5, beta=0.12, d_cutoff=1.0):
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


def _rounded_rect(img, pt1, pt2, color, radius, filled=True):
    x1, y1 = pt1
    x2, y2 = pt2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    tk = -1 if filled else 1
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, tk)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, tk)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, tk)


# ---------------------------------------------------------------------------
# Stroke — one independent drawn object
# ---------------------------------------------------------------------------

class Stroke:
    def __init__(self, color, thickness):
        self.color = color
        self.thickness = thickness
        self.points = []
        self._ox = 0
        self._oy = 0

    def add_point(self, pt):
        self.points.append(pt)

    def translate(self, dx, dy):
        self._ox += dx
        self._oy += dy

    def get_points(self):
        if self._ox == 0 and self._oy == 0:
            return self.points
        ox, oy = self._ox, self._oy
        return [(x + ox, y + oy) for x, y in self.points]

    def min_distance_to(self, pt):
        pts = self.get_points()
        if not pts:
            return float('inf')
        px, py = pt
        return min(math.sqrt((p[0] - px) ** 2 + (p[1] - py) ** 2) for p in pts)

    def intersects_circle(self, center, radius):
        return self.min_distance_to(center) <= radius


# ---------------------------------------------------------------------------
# GestureDetector — supports up to N_HANDS simultaneous hands
# ---------------------------------------------------------------------------

class GestureDebouncer:
    """Majority-vote over a short window to iron out single-frame flips."""
    def __init__(self, window=5):
        self._buf = deque(maxlen=window)

    def update(self, gesture):
        self._buf.append(gesture)
        return max(set(self._buf), key=self._buf.count)

    def reset(self):
        self._buf.clear()


class GestureDetector:
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    FINGER_MCPS = [5,  9, 13, 17]

    def __init__(self, max_num_hands=N_HANDS,
                 min_detection_confidence=0.7, min_tracking_confidence=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._results  = None
        self._start_time = time.time()

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        self._results = self._detector.detect_for_video(mp_image, timestamp_ms)

    def get_all_landmarks(self, frame_w, frame_h) -> list:
        """Return a list (one entry per detected hand) of 21-point landmark lists."""
        if not self._results or not self._results.hand_landmarks:
            return []
        result = []
        for hand in self._results.hand_landmarks:
            result.append([(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in hand])
        return result

    @staticmethod
    def _is_extended(lms, tip, pip, mcp):
        ax=lms[pip][0]-lms[mcp][0]; ay=lms[pip][1]-lms[mcp][1]
        tx=lms[tip][0]-lms[pip][0]; ty=lms[tip][1]-lms[pip][1]
        return ax*tx+ay*ty>0

    def _fingers_up(self, lms, frame_w):
        thumb_up = abs(lms[4][0] - lms[2][0]) > frame_w * 0.04
        fingers = [thumb_up]
        for tip, pip, mcp in zip(self.FINGER_TIPS, self.FINGER_PIPS, self.FINGER_MCPS):
            fingers.append(self._is_extended(lms, tip, pip, mcp))
        return fingers  # [thumb, index, middle, ring, pinky]

    def classify(self, lms, frame_w, frame_h) -> str:
        # Thumb is intentionally ignored — unreliable due to lateral movement
        _, index, middle, ring, pinky = self._fingers_up(lms, frame_w)
        # Open palm (all 4 fingers up) → Erase
        if index and middle and ring and pinky:
            return "ERASE"
        # Peace sign (index + middle, ring + pinky down) → Color cycle
        if index and middle and not ring and not pinky:
            return "COLOR_CYCLE"
        # Index only → Draw
        if index and not middle and not ring and not pinky:
            return "DRAW"
        # Fist → Grab: tip must be below its base knuckle (MCP).
        # MCP reference is more forgiving than PIP for partial/tight fists.
        if all(lms[t][1] > lms[m][1]
               for t, m in zip(self.FINGER_TIPS, self.FINGER_MCPS)):
            return "GRAB"
        return "NONE"

    def get_index_tip(self, lms):
        return lms[8]

    def get_palm_center(self, lms):
        xs = [lms[i][0] for i in [5, 9, 13, 17]]
        ys = [lms[i][1] for i in [5, 9, 13, 17]]
        return (sum(xs) // 4, sum(ys) // 4)

    def draw_landmarks(self, skeleton_frame, lms, gesture="NONE"):
        accent      = GESTURE_ACCENT.get(gesture, (55, 55, 75))
        bone_color  = _dim(accent, 0.35)
        tip_color   = _dim(accent, 0.85)
        joint_color = _dim(accent, 0.55)
        for a, b in HAND_CONNECTIONS:
            cv2.line(skeleton_frame, lms[a], lms[b], bone_color, 2, cv2.LINE_AA)
        tips = {4, 8, 12, 16, 20}
        for i, pt in enumerate(lms):
            if i in tips:
                cv2.circle(skeleton_frame, pt, 6, _dim(tip_color, 0.3), -1, cv2.LINE_AA)
                cv2.circle(skeleton_frame, pt, 4, tip_color,             -1, cv2.LINE_AA)
            else:
                cv2.circle(skeleton_frame, pt, 3, joint_color, -1, cv2.LINE_AA)

    def close(self):
        self._detector.close()


# ---------------------------------------------------------------------------
# GlowCanvas — stroke-based drawing, multi-hand aware
# ---------------------------------------------------------------------------

class GlowCanvas:
    COLORS = [
        (0,   0,   255),   # Red
        (0,  128,  255),   # Orange
        (0,  255,  255),   # Yellow
        (0,  255,    0),   # Green
        (255, 255,   0),   # Cyan
        (255,   0,    0),  # Blue
        (255,   0,  128),  # Purple
        (255,   0,  255),  # Pink
        (255, 255,  255),  # White
    ]
    COLOR_NAMES = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink", "White"]

    def __init__(self, width, height,
                 trail_decay=0.72, blur_kernel=21, glow_weight=0.40):
        self._w = width
        self._h = height
        self._trail_decay = trail_decay
        self._blur_kernel = blur_kernel
        self._glow_weight = glow_weight
        self.color_idx     = 0
        self.pen_thickness = 8
        self.eraser_radius = 40

        self.strokes: list[Stroke] = []
        # One active stroke per hand, keyed by hand index (0 or 1)
        self.current_strokes: dict[int, Stroke] = {}
        self.selected_stroke: Stroke | None = None

        self.draw_canvas  = np.zeros((height, width, 3), dtype=np.uint8)
        self.trail_canvas = np.zeros((height, width, 3), dtype=np.float32)
        self._tmp         = np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def current_color(self):
        return self.COLORS[self.color_idx]

    @property
    def current_color_name(self):
        return self.COLOR_NAMES[self.color_idx]

    def next_color(self):
        self.color_idx = (self.color_idx + 1) % len(self.COLORS)

    # ── Per-hand stroke drawing ─────────────────────────────────────────────

    def begin_stroke(self, hand_id: int):
        self.current_strokes[hand_id] = Stroke(self.current_color, self.pen_thickness)

    def continue_stroke(self, pt, hand_id: int):
        stroke = self.current_strokes.get(hand_id)
        if stroke is None:
            return
        if stroke.points:
            prev = stroke.points[-1]
            self._tmp[:] = 0
            cv2.line(self._tmp, prev, pt, stroke.color,
                     stroke.thickness + 4, cv2.LINE_AA)
            self.trail_canvas += self._tmp.astype(np.float32)
        stroke.add_point(pt)

    def end_stroke(self, hand_id: int):
        stroke = self.current_strokes.pop(hand_id, None)
        if stroke and stroke.points:
            self.strokes.append(stroke)
            self._render_stroke_to(self.draw_canvas, stroke)

    # ── Grab / drag ─────────────────────────────────────────────────────────

    def grab_nearest(self, point) -> bool:
        best_dist = GRAB_THRESHOLD
        self.selected_stroke = None
        for stroke in self.strokes:
            d = stroke.min_distance_to(point)
            if d < best_dist:
                best_dist = d
                self.selected_stroke = stroke
        if self.selected_stroke:
            self._rebuild_canvas()
        return self.selected_stroke is not None

    def drag_selected(self, dx, dy):
        if self.selected_stroke:
            self.selected_stroke.translate(dx, dy)

    def release_grab(self):
        self.selected_stroke = None
        self._rebuild_canvas()

    # ── Erase ───────────────────────────────────────────────────────────────

    def erase(self, center):
        before = len(self.strokes)
        if self.selected_stroke and self.selected_stroke.intersects_circle(center, self.eraser_radius):
            self.selected_stroke = None
        self.strokes = [s for s in self.strokes
                        if not s.intersects_circle(center, self.eraser_radius)]
        if len(self.strokes) != before:
            self._rebuild_canvas()

    # ── Canvas internals ────────────────────────────────────────────────────

    def _render_stroke_to(self, canvas, stroke, pts=None, outline=None, outline_w=8):
        if pts is None:
            pts = stroke.get_points()
        if not pts:
            return
        if outline:
            if len(pts) == 1:
                cv2.circle(canvas, pts[0],
                           stroke.thickness // 2 + outline_w // 2,
                           outline, -1, cv2.LINE_AA)
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], outline,
                         stroke.thickness + outline_w, cv2.LINE_AA)
        if len(pts) == 1:
            cv2.circle(canvas, pts[0], stroke.thickness // 2,
                       stroke.color, -1, cv2.LINE_AA)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], stroke.color,
                     stroke.thickness, cv2.LINE_AA)

    def _rebuild_canvas(self):
        self.draw_canvas[:] = 0
        for stroke in self.strokes:
            if stroke is not self.selected_stroke:
                self._render_stroke_to(self.draw_canvas, stroke)

    def decay_trail(self):
        np.multiply(self.trail_canvas, self._trail_decay, out=self.trail_canvas)

    def composite(self, skeleton_frame):
        trail_uint8 = np.clip(self.trail_canvas, 0, 255).astype(np.uint8)
        glow   = cv2.GaussianBlur(trail_uint8, (self._blur_kernel, self._blur_kernel), 0)
        base   = cv2.addWeighted(skeleton_frame, 1.0, self.draw_canvas, 1.0, 0)
        output = cv2.addWeighted(base, 1.0, glow, self._glow_weight, 0)

        # All active (in-progress) strokes drawn on top
        for stroke in self.current_strokes.values():
            if stroke.points:
                self._render_stroke_to(output, stroke)

        # Selected stroke with highlight ring
        if self.selected_stroke:
            pts = self.selected_stroke.get_points()
            self._render_stroke_to(output, self.selected_stroke, pts,
                                   outline=(160, 210, 255), outline_w=10)
        return output

    def clear(self):
        self.strokes.clear()
        self.current_strokes.clear()
        self.selected_stroke = None
        self.draw_canvas[:]  = 0
        self.trail_canvas[:] = 0

    # ── Cursor drawing (called per hand) ────────────────────────────────────

    def _draw_cursor(self, frame, gesture, cursor_pt, grabbed=False):
        if gesture == "DRAW":
            col = self.current_color
            cv2.circle(frame, cursor_pt, 16, _dim(col, 0.18), -1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, 10, _dim(col, 0.55), -1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt,  5, col,             -1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt,  2, (255, 255, 255), -1, cv2.LINE_AA)

        elif gesture == "ERASE":
            er = self.eraser_radius
            ea = GESTURE_ACCENT["ERASE"]
            cv2.circle(frame, cursor_pt, er,     _dim(ea, 0.18), -1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, er,     ea, 2, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, er - 5, _dim(ea, 0.4), 1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, 5,      ea, -1, cv2.LINE_AA)

        elif gesture == "GRAB":
            ga     = GESTURE_ACCENT["GRAB"]
            ga_col = ga if grabbed else _dim(ga, 0.45)
            rc     = 18
            for arm in [(-rc, 0), (rc, 0), (0, -rc), (0, rc)]:
                end = (cursor_pt[0] + arm[0], cursor_pt[1] + arm[1])
                cv2.line(frame, cursor_pt, end, _dim(ga_col, 0.5), 3, cv2.LINE_AA)
                cv2.line(frame, cursor_pt, end, ga_col,            1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, rc, _dim(ga_col, 0.35), 2, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, rc, ga_col,              1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt,  5, ga_col,             -1, cv2.LINE_AA)
            tag = "HOLD" if grabbed else "SEARCH"
            cv2.putText(frame, tag,
                        (cursor_pt[0] + rc + 4, cursor_pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ga_col, 1, cv2.LINE_AA)

        elif gesture == "COLOR_CYCLE":
            ca = GESTURE_ACCENT["COLOR_CYCLE"]
            cv2.circle(frame, cursor_pt, 14, _dim(ca, 0.25), -1, cv2.LINE_AA)
            cv2.circle(frame, cursor_pt, 14, ca,              2, cv2.LINE_AA)

    # ── Full UI overlay ──────────────────────────────────────────────────────

    def draw_ui_overlay(self, frame, hand_data: list,
                        hand_visible=False, grabbed=False):
        """
        hand_data: list of (gesture, cursor_pt) tuples, one per detected hand.
        """
        h, w = frame.shape[:2]

        # Pick highest-priority gesture across all active hands for the badge
        primary = "NONE"
        for g, _ in hand_data:
            if GESTURE_PRIORITY.index(g) < GESTURE_PRIORITY.index(primary):
                primary = g
        accent = GESTURE_ACCENT.get(primary, (55, 55, 75))

        # ── Bottom HUD panel ──────────────────────────────────────────────
        BAR_H = 76
        bar_y = h - BAR_H

        panel = frame.copy()
        cv2.rectangle(panel, (0, bar_y), (w, h), (8, 8, 18), -1)
        cv2.addWeighted(panel, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, bar_y),     (w, bar_y),     _dim(accent, 0.40), 2)
        cv2.line(frame, (0, bar_y - 1), (w, bar_y - 1), _dim(accent, 0.15), 3)

        # ── Mode badge ────────────────────────────────────────────────────
        MODE_LABELS = {
            "DRAW": "DRAW", "ERASE": "ERASE",
            "GRAB": "GRAB", "COLOR_CYCLE": "COLOR", "NONE": "IDLE",
        }
        label = MODE_LABELS.get(primary, primary)
        font  = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), base = cv2.getTextSize(label, font, 0.58, 1)
        px, py = 10, 7
        bx1 = 12
        by1 = bar_y + BAR_H // 2 - th // 2 - py
        bx2 = bx1 + tw + px * 2
        by2 = bar_y + BAR_H // 2 + th // 2 + py + base
        _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(accent, 0.25), 8)
        _rounded_rect(frame, (bx1, by1), (bx2, by2), _dim(accent, 0.70), 8, filled=False)
        cv2.putText(frame, label, (bx1 + px, by2 - py - base),
                    font, 0.58, (230, 230, 235), 1, cv2.LINE_AA)

        # Hands active indicator (dots right of badge)
        n_active = len(hand_data)
        for k in range(N_HANDS):
            dot_x = bx2 + 12 + k * 14
            dot_y = bar_y + BAR_H // 2
            dot_col = (0, 200, 120) if k < n_active else (40, 40, 55)
            cv2.circle(frame, (dot_x, dot_y), 5, dot_col, -1, cv2.LINE_AA)

        # ── Color circles ─────────────────────────────────────────────────
        R, GAP = 13, 9
        n = len(self.COLORS)
        total_row_w = n * R * 2 + (n - 1) * GAP
        cx0 = (w - total_row_w) // 2 + R
        cy  = bar_y + BAR_H // 2

        for i, col in enumerate(self.COLORS):
            cx = cx0 + i * (R * 2 + GAP)
            ct = (cx, cy)
            if i == self.color_idx:
                cv2.circle(frame, ct, R + 11, _dim(col, 0.08), -1)
                cv2.circle(frame, ct, R +  7, _dim(col, 0.20), -1)
                cv2.circle(frame, ct, R +  3, _dim(col, 0.50), -1)
                cv2.circle(frame, ct, R +  2, (240, 240, 255), 1)
            cv2.circle(frame, ct, R, col, -1)
            cv2.circle(frame, ct, R, (25, 25, 35), 1)

        # ── Color name ────────────────────────────────────────────────────
        cname = self.current_color_name
        (cnw, cnh), _ = cv2.getTextSize(cname, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, cname,
                    (w - cnw - 14, bar_y + BAR_H // 2 + cnh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 185), 1, cv2.LINE_AA)

        # ── Top bar ───────────────────────────────────────────────────────
        TOP_H = 28
        top_panel = frame.copy()
        cv2.rectangle(top_panel, (0, 0), (w, TOP_H), (8, 8, 18), -1)
        cv2.addWeighted(top_panel, 0.70, frame, 0.30, 0, frame)
        cv2.line(frame, (0, TOP_H), (w, TOP_H), (30, 30, 50), 1)
        cv2.putText(frame, "Gesture Whiteboard",
                    (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 110), 1, cv2.LINE_AA)
        hint = "C: Clear    Q: Quit"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, hint, (w - hw - 10, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 110), 1, cv2.LINE_AA)

        # ── No-hand guide ─────────────────────────────────────────────────
        if not hand_visible:
            guide = [
                ("Index finger only",  "DRAW",  "DRAW"),
                ("Peace sign (2 up)",  "COLOR", "COLOR_CYCLE"),
                ("Open palm (all up)", "ERASE", "ERASE"),
                ("Fist (all curled)",  "DRAG",  "GRAB"),
            ]
            gw = 230
            gh = 30 + len(guide) * 26 + 10
            gx = (w - gw) // 2
            gy = (h - BAR_H - gh) // 2
            bg = frame.copy()
            _rounded_rect(bg, (gx, gy), (gx + gw, gy + gh), (10, 10, 22), 10)
            cv2.addWeighted(bg, 0.80, frame, 0.20, 0, frame)
            _rounded_rect(frame, (gx, gy), (gx + gw, gy + gh), (40, 40, 65), 10, filled=False)
            cv2.putText(frame, "Show your hand",
                        (gx + 20, gy + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (160, 160, 200), 1, cv2.LINE_AA)
            for k, (action, mode_label, gesture_key) in enumerate(guide):
                lx = gx + 20
                ly = gy + 48 + k * 26
                a_col = GESTURE_ACCENT.get(gesture_key, (80, 80, 100))
                _rounded_rect(frame, (lx, ly - 14), (lx + 58, ly + 5), _dim(a_col, 0.25), 4)
                cv2.putText(frame, mode_label, (lx + 5, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, _dim(a_col, 0.9), 1, cv2.LINE_AA)
                cv2.putText(frame, action, (lx + 68, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 165), 1, cv2.LINE_AA)

        # ── Cursors (one per active hand) ─────────────────────────────────
        for gesture, cursor_pt in hand_data:
            if cursor_pt is not None:
                self._draw_cursor(frame, gesture, cursor_pt, grabbed)

        return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector    = GestureDetector()
    canvas      = GlowCanvas(frame_w, frame_h)
    grabber     = FrameGrabber(cap)
    debouncers  = [GestureDebouncer() for _ in range(N_HANDS)]

    # Per-hand state (index = hand slot 0 or 1)
    states = [
        {'drawing': False, 'grabbing': False, 'grab_prev': None, 'smooth_pt': None,
         'oe_x': _OneEuro(1.5, 0.12), 'oe_y': _OneEuro(1.5, 0.12)}
        for _ in range(N_HANDS)
    ]
    grabbing_hand  = None   # index of the hand that currently holds a stroke
    color_cooldown = 0

    cv2.namedWindow("Gesture Whiteboard", cv2.WINDOW_NORMAL)

    while True:
        frame = grabber.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector.process(rgb)
        all_lms = detector.get_all_landmarks(frame_w, frame_h)
        n_detected = len(all_lms)

        # Clean up state for any hand that is no longer visible
        for i in range(N_HANDS):
            s = states[i]
            if i >= n_detected:
                debouncers[i].reset()
                if s['drawing']:
                    canvas.end_stroke(i)
                    s['drawing']   = False
                    s['smooth_pt'] = None
                if s['grabbing']:
                    canvas.release_grab()
                    s['grabbing']  = False
                    s['grab_prev'] = None
                    if grabbing_hand == i:
                        grabbing_hand = None

        hand_data      = []   # (gesture, cursor_pt) per detected hand
        skeleton_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        for i, lms in enumerate(all_lms):
            s       = states[i]
            gesture = debouncers[i].update(detector.classify(lms, frame_w, frame_h))
            tip     = detector.get_index_tip(lms)
            palm    = detector.get_palm_center(lms)
            cursor_pt = None

            # Finalize stroke when leaving DRAW
            if s['drawing'] and gesture != "DRAW":
                canvas.end_stroke(i)
                s['drawing']   = False
                s['smooth_pt'] = None

            # Release grab when leaving GRAB
            if s['grabbing'] and gesture != "GRAB":
                canvas.release_grab()
                s['grabbing']  = False
                s['grab_prev'] = None
                if grabbing_hand == i:
                    grabbing_hand = None

            if gesture == "DRAW":
                now = time.monotonic()
                s['smooth_pt'] = (
                    round(s['oe_x'](tip[0], now)),
                    round(s['oe_y'](tip[1], now)),
                )
                cursor_pt = s['smooth_pt']
                if not s['drawing']:
                    canvas.begin_stroke(i)
                    s['drawing'] = True
                canvas.continue_stroke(s['smooth_pt'], i)

            elif gesture == "GRAB":
                cursor_pt = palm
                if not s['grabbing'] and grabbing_hand is None:
                    canvas.grab_nearest(palm)
                    s['grabbing']  = True
                    s['grab_prev'] = palm
                    grabbing_hand  = i
                elif s['grabbing']:
                    if s['grab_prev'] is not None:
                        canvas.drag_selected(palm[0] - s['grab_prev'][0],
                                             palm[1] - s['grab_prev'][1])
                    s['grab_prev'] = palm

            elif gesture == "ERASE":
                cursor_pt = palm
                canvas.erase(palm)
                s['grab_prev'] = None

            elif gesture == "COLOR_CYCLE":
                cursor_pt = tip
                if color_cooldown == 0:
                    canvas.next_color()
                    color_cooldown = 20
                s['grab_prev'] = None

            else:
                s['grab_prev'] = None

            hand_data.append((gesture, cursor_pt))
            detector.draw_landmarks(skeleton_frame, lms, gesture)

        if color_cooldown > 0:
            color_cooldown -= 1

        canvas.decay_trail()

        output = canvas.composite(skeleton_frame)
        output = canvas.draw_ui_overlay(
            output, hand_data,
            hand_visible=(n_detected > 0),
            grabbed=(grabbing_hand is not None and canvas.selected_stroke is not None),
        )

        cv2.imshow("Gesture Whiteboard", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()

    grabber.stop()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
