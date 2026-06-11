"""
Fruit Ninja – Webcam Gesture Edition
─────────────────────────────────────
Hover your hand over "1 PLAYER" or "2 PLAYERS" on the menu to select.
Swipe your fingers through fruits to slice them. Avoid slicing bombs!

Press Q to quit.
"""

import math
import os
import random
import threading
import time
import urllib.request
from collections import deque

import io
import wave

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
try:
    import winsound as _ws
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False

_SR = 22050  # audio sample rate


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ── Constants ─────────────────────────────────────────────────────────────────
GRAVITY         = 0.78
SWIPE_THRESHOLD = 14
TRAIL_LEN       = 24
SPAWN_LO, SPAWN_HI = 0.85, 1.55
WAVE_MIN, WAVE_MAX = 2, 4
BOMB_CHANCE     = 0.15
BOMB_PENALTY    = 20
GAME_DURATION   = 90
LIVES_START     = 3
COMBO_WINDOW    = 1.5
HOVER_TIME      = 0.9
MISS_PENALTY_Y  = 80
COUNTDOWN_SECS  = 3.0

MODE_1P = "1P"
MODE_2P = "2P"

# BGR player colours
P1_COL = (255, 200,  30)   # cyan-gold
P2_COL = (  0, 140, 255)   # orange
GOLD   = ( 20, 195, 245)   # bright gold

COMBO_SHOUTS = {3: "NICE!", 5: "GREAT!", 8: "AWESOME!", 12: "INSANE!"}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Fruit palette — colors boosted so Phong shading yields vivid, saturated hues (BGR)
FRUIT_KINDS = {
    "watermelon": {
        "outer": ( 28, 195,  55), "inner": ( 40,  52, 240),
        "juice": ( 70,  88, 245), "r": (40, 52),
    },
    "apple": {
        "outer": ( 18,  22, 248), "inner": ( 65,  90, 252),
        "juice": ( 48,  60, 245), "r": (28, 38),
    },
    "orange": {
        "outer": (  8, 128, 255), "inner": ( 20, 162, 255),
        "juice": ( 10, 135, 255), "r": (28, 38),
    },
    "peach": {
        "outer": ( 78, 148, 255), "inner": (120, 178, 255),
        "juice": ( 95, 155, 255), "r": (26, 36),
    },
    "strawberry": {
        "outer": ( 18,  18, 235), "inner": ( 50,  60, 248),
        "juice": ( 38,  42, 238), "r": (22, 32),
    },
}
FRUIT_NAMES = list(FRUIT_KINDS.keys())

# Dojo palette (BGR)
BANNER_COL  = ( 38,  48, 188)   # deep red banner → RGB(188,48,38)
BANNER_GOLD = ( 22, 172, 208)   # gold trim → RGB(208,172,22)
BAMBOO_MID  = ( 58, 148,  72)   # bamboo green mid
BAMBOO_DARK = ( 40, 110,  52)   # bamboo node
BAMBOO_LITE = ( 80, 168,  92)   # bamboo highlight


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _dim(c, f):
    return tuple(int(x*f) for x in c)

def _seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    if dx == dy == 0:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy)))
    return _dist(p, (ax+t*dx, ay+t*dy))

def _blend(dst, src, alpha):
    cv2.addWeighted(src, alpha, dst, 1-alpha, 0, dst)

def _text(frame, txt, pos, font, scale, color, thickness=2, shadow=True, depth=0):
    """Draw text with a clean shadow + dark stroke outline + bright face."""
    x, y = pos
    sh = max(3, depth + 2)
    # Soft drop shadow
    if shadow or depth > 0:
        cv2.putText(frame, txt, (x+sh, y+sh), font, scale,
                    (0, 0, 0), thickness + 5, cv2.LINE_AA)
    # Dark stroke border (gives clean edge without muddy layers)
    cv2.putText(frame, txt, (x, y), font, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    # Bright face
    cv2.putText(frame, txt, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def _progress_bar(frame, x, y, w, h, progress, color):
    """3-D inset progress trough with gradient fill."""
    cv2.rectangle(frame, (x-2,y-2), (x+w+2,y+h+2), _dim(BANNER_GOLD,0.55), -1)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (25,20,14), -1)
    cv2.line(frame, (x,y),   (x+w,y),   (0,0,0), 1)
    cv2.line(frame, (x,y),   (x,y+h),   (0,0,0), 1)
    fill = int(w * min(1.0, max(0.0, progress)))
    if fill > 0:
        mid = y + h//2
        bright = tuple(min(255, int(c*1.4)) for c in color)
        cv2.rectangle(frame, (x,y),   (x+fill,mid),  bright, -1)
        cv2.rectangle(frame, (x,mid), (x+fill,y+h),  color,  -1)
        cv2.line(frame, (x,y+1), (x+fill,y+1),
                 tuple(min(255,int(c*1.9)) for c in color), 1)
    cv2.rectangle(frame, (x,y), (x+w,y+h), BANNER_GOLD, 1)

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


def _rounded_box(frame, x, y, w, h, color, filled=True, r=12, depth=0):
    """Rounded rectangle with optional 3-D extrusion (filled only)."""
    r  = min(r, w//2, h//2)
    tk = -1 if filled else 2

    def _rr(ix, iy, ic, itk):
        cv2.rectangle(frame, (ix+r,iy),   (ix+w-r,iy+h),  ic, itk)
        cv2.rectangle(frame, (ix,iy+r),   (ix+w,iy+h-r),  ic, itk)
        for cx,cy in [(ix+r,iy+r),(ix+w-r,iy+r),(ix+r,iy+h-r),(ix+w-r,iy+h-r)]:
            cv2.circle(frame, (cx,cy), r, ic, itk, cv2.LINE_AA)

    if filled and depth > 0:
        for i in range(depth, 0, -1):
            t = (depth - i + 1) / depth
            _rr(x+i, y+i, _dim(color, 0.18 + 0.16*t), -1)
    _rr(x, y, color, tk)
    if filled and depth > 0:
        light  = tuple(min(255, int(c*1.8 + 18)) for c in color)
        shadow = _dim(color, 0.22)
        cv2.line(frame, (x+r,  y+2),   (x+w-r, y+2),   light,  2, cv2.LINE_AA)
        cv2.line(frame, (x+2,  y+r),   (x+2,   y+h-r), light,  2, cv2.LINE_AA)
        cv2.line(frame, (x+r,  y+h-2), (x+w-r, y+h-2), shadow, 2, cv2.LINE_AA)
        cv2.line(frame, (x+w-2,y+r),   (x+w-2, y+h-r), shadow, 2, cv2.LINE_AA)
        _rr(x, y, _dim(color, 0.50), 2)


# ── FrameGrabber ──────────────────────────────────────────────────────────────

class FrameGrabber:
    def __init__(self, cap):
        self._cap=cap; self._buf=deque(maxlen=1); self._running=True
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()

    def _run(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret: self._buf.append(frame)

    def read(self): return self._buf[-1] if self._buf else None

    def stop(self): self._running=False; self._t.join(timeout=1)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model ready.")


# ── Hand detector ─────────────────────────────────────────────────────────────

class HandDetector:
    TIPS=[8,12,16,20]; PIPS=[6,10,14,18]; MCPS=[5,9,13,17]; SMOOTH_A=0.45

    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self._det=mp_vision.HandLandmarker.create_from_options(opts)
        self._t0=time.time(); self._res=None; self._smooth={}

    def process(self, rgb):
        ts=int((time.time()-self._t0)*1000)
        self._res=self._det.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    def all_hands(self, w, h):
        if not self._res or not self._res.hand_landmarks:
            self._smooth.clear(); return []
        raw=[[(int(lm.x*w),int(lm.y*h)) for lm in hand]
             for hand in self._res.hand_landmarks]
        raw.sort(key=lambda lms: lms[0][0])
        result, seen = [], set()
        for i, lms in enumerate(raw[:2]):
            pid=f"P{i+1}"; seen.add(pid)
            sp=self._smooth_tip(lms, pid)
            result.append((pid, lms, (round(sp[0]), round(sp[1]))))
        for k in list(self._smooth.keys()):
            if k not in seen: del self._smooth[k]
        return result

    @staticmethod
    def _is_extended(lms, tip, pip, mcp):
        ax=lms[pip][0]-lms[mcp][0]; ay=lms[pip][1]-lms[mcp][1]
        tx=lms[tip][0]-lms[pip][0]; ty=lms[tip][1]-lms[pip][1]
        return ax*tx+ay*ty>0

    def _smooth_tip(self, lms, pid):
        ext=[lms[t] for t,p,m in zip(self.TIPS,self.PIPS,self.MCPS)
             if self._is_extended(lms,t,p,m)]
        if not ext: ext=[lms[8]]
        rx=sum(p[0] for p in ext)/len(ext)
        ry=sum(p[1] for p in ext)/len(ext)
        now=time.monotonic()
        if pid not in self._smooth:
            self._smooth[pid]=(_OneEuro(1.5,0.12),_OneEuro(1.5,0.12))
        fx,fy=self._smooth[pid]
        return (fx(rx,now), fy(ry,now))

    def is_open_palm(self, lms, w):
        return (abs(lms[4][0]-lms[2][0])>w*0.04 and
                all(self._is_extended(lms,t,p,m)
                    for t,p,m in zip(self.TIPS,self.PIPS,self.MCPS)))

    def close(self): self._det.close()


# ── Dojo Background ───────────────────────────────────────────────────────────

class Background:
    BANNER_H = 68

    def __init__(self, fw, fh):
        self.fw, self.fh = fw, fh
        self._base = self._build()
        self._current = self._base.copy()
        self._vignette = self._build_vignette()

    def _build_vignette(self):
        fw, fh = self.fw, self.fh
        y_a, x_a = np.ogrid[:fh, :fw]
        cx, cy = fw / 2.0, fh / 2.0
        # Elliptical distance (widescreen-aware), clipped so centre stays bright
        dx = (x_a - cx) / cx
        dy = (y_a - cy) / (cy * 0.80)
        v  = np.clip(1.0 - np.sqrt(dx**2 + dy**2) * 0.50, 0.32, 1.0).astype(np.float32)
        return v[:, :, np.newaxis]   # (fh, fw, 1) for broadcast

    def reset(self):
        np.copyto(self._current, self._base)

    def add_stain(self, cx, cy, color, r):
        """Paint a semi-transparent juice stain blob onto the background layer."""
        bh = self.BANNER_H
        rr = r * 4
        x1 = max(0, cx - rr);  y1 = max(bh + 4, cy - rr)
        x2 = min(self.fw - 1, cx + rr); y2 = min(self.fh - bh - 4, cy + rr)
        if x2 <= x1 or y2 <= y1:
            return
        roi = self._current[y1:y2, x1:x2]
        overlay = roi.copy()
        lx, ly = cx - x1, cy - y1
        c = _dim(color, 0.88)

        # Main blob
        cv2.circle(overlay, (lx, ly), r, c, -1, cv2.LINE_AA)
        # Satellite drops
        for _ in range(random.randint(3, 6)):
            ang = random.uniform(0, math.pi * 2)
            dist = random.uniform(r * 0.5, r * 2.0)
            dr = max(3, random.randint(r // 5, r // 2 + 2))
            sx = int(lx + dist * math.cos(ang))
            sy = int(ly + dist * math.sin(ang))
            cv2.circle(overlay, (sx, sy), dr, c, -1, cv2.LINE_AA)
        # Downward drip tail
        drip_len = random.randint(r, r * 2 + 4)
        dang = random.uniform(math.pi * 0.25, math.pi * 0.75)
        ex = int(lx + drip_len * math.cos(dang))
        ey = int(ly + drip_len * math.sin(dang))
        cv2.line(overlay, (lx, ly), (ex, ey), c, max(3, r // 3), cv2.LINE_AA)

        # Blend at ~80% opacity so background texture shows through slightly
        cv2.addWeighted(overlay, 0.80, roi, 0.20, 0, roi)

    def _build(self):
        fw, fh = self.fw, self.fh
        bh = self.BANNER_H
        rng = np.random.default_rng(7)
        img = np.zeros((fh, fw, 3), dtype=np.uint8)

        play_top = bh
        play_bot = fh - bh
        play_h   = play_bot - play_top
        vp_y     = play_top + int(play_h * 0.40)   # horizon (vanishing point y)
        vp_x     = fw // 2                           # vanishing point x

        # ── Back wall (play_top → vp_y) ──────────────────────────────────────
        wall_top = np.array([ 55, 105, 158], dtype=np.float32)  # darker amber
        wall_bot = np.array([ 68, 128, 182], dtype=np.float32)  # lighter at horizon
        for y in range(play_top, vp_y):
            t    = (y - play_top) / max(1, vp_y - play_top)
            band = math.sin(y / 34) * 5
            img[y] = np.clip(wall_top + (wall_bot - wall_top)*t + band, 0, 255).astype(np.uint8)

        # Vertical wood-panel lines on wall
        panel_w = fw // 10
        wall_line = tuple(int(c * 0.78) for c in wall_bot.tolist())
        for px in range(0, fw, panel_w):
            cv2.line(img, (px, play_top), (px, vp_y), wall_line, 1)
        # Horizontal wainscoting rails
        for frac in (0.35, 0.70):
            ry = int(play_top + (vp_y - play_top) * frac)
            cv2.line(img, (0, ry),   (fw, ry),   wall_line, 2)
            cv2.line(img, (0, ry+3), (fw, ry+3),
                     tuple(min(255,int(c*1.12)) for c in wall_line), 1)

        # ── Perspective floor (vp_y → play_bot) ──────────────────────────────
        floor_near = np.array([ 88, 152, 208], dtype=np.float32)  # warm, near viewer
        floor_far  = np.array([ 68, 124, 180], dtype=np.float32)  # cooler at horizon
        for y in range(vp_y, play_bot):
            t    = (y - vp_y) / max(1, play_bot - vp_y)
            band = math.sin(y / 28) * 4
            img[y] = np.clip(floor_far + (floor_near - floor_far)*t + band, 0, 255).astype(np.uint8)

        # ── Perspective floor grid ────────────────────────────────────────────
        floor_h   = play_bot - vp_y
        grid_dark = tuple(int(c * 0.56) for c in floor_near.tolist())   # stronger contrast
        grid_mid  = tuple(int(c * 0.68) for c in floor_near.tolist())

        # Radial lines converging to vanishing point (alternating dark/mid)
        for i in range(17):
            t  = i / 16
            xb = int(t * fw)
            col = grid_dark if (i % 2 == 0) else grid_mid
            cv2.line(img, (vp_x, vp_y), (xb, play_bot), col, 1)

        # Horizontal depth lines – perspective foreshortening
        for i in range(1, 12):
            t      = i / 12
            y_line = int(vp_y + floor_h * t**1.6)
            thick  = max(1, int(2.5 * t))
            col    = grid_dark if t > 0.5 else grid_mid
            cv2.line(img, (0, y_line), (fw, y_line), col, thick)

        # ── Horizon shadow ────────────────────────────────────────────────────
        cv2.line(img, (0, vp_y-1), (fw, vp_y-1),
                 tuple(int(c*0.52) for c in wall_bot.tolist()), 3)
        cv2.line(img, (0, vp_y+1), (fw, vp_y+1), (6,6,6), 1)

        # ── Texture noise ─────────────────────────────────────────────────────
        noise = (rng.random((fh, fw))*22 - 11).astype(np.int16)
        for c in range(3):
            img[:,:,c] = np.clip(img[:,:,c].astype(np.int16)+noise, 0, 255)

        # ── Bamboo poles on sides ─────────────────────────────────────────────
        pole_xs = [0, 26, 52, fw-72, fw-46, fw-20]
        pw = 20
        for bx in pole_xs:
            for dx in range(pw):
                shade = int(math.sin(dx/pw*math.pi)*38)
                mid   = np.array(BAMBOO_MID, dtype=np.int16)
                col   = np.clip(mid+shade, 0, 255).astype(np.uint8)
                img[play_top:play_bot, bx+dx] = col
            for ny in range(play_top+20, play_bot, 65+rng.integers(0,25)):
                cv2.rectangle(img, (bx-1,ny), (bx+pw,ny+6), BAMBOO_DARK, -1)
                cv2.line(img, (bx, ny), (bx+pw-1, ny), BAMBOO_LITE, 1)

        # ── Top and bottom banners ────────────────────────────────────────────
        img[:bh]      = BANNER_COL
        img[fh-bh:]   = BANNER_COL
        img[bh-5:bh+1]        = BANNER_GOLD
        img[fh-bh-1:fh-bh+5]  = BANNER_GOLD

        for ex in [fw//5, fw//2, 4*fw//5]:
            cv2.circle(img, (ex, bh//2),    22, _dim(BANNER_GOLD,0.7), 2, cv2.LINE_AA)
            cv2.circle(img, (ex, fh-bh//2), 22, _dim(BANNER_GOLD,0.7), 2, cv2.LINE_AA)
            cv2.circle(img, (ex, bh//2),     8, _dim(BANNER_GOLD,0.5),-1, cv2.LINE_AA)
            cv2.circle(img, (ex, fh-bh//2),  8, _dim(BANNER_GOLD,0.5),-1, cv2.LINE_AA)

        return img

    def draw(self, frame):
        np.copyto(frame, self._current)
        # Vignette: cinematic darkening toward corners
        f32 = frame.astype(np.float32)
        f32 *= self._vignette
        np.clip(f32, 0, 255, out=f32)
        frame[:] = f32.astype(np.uint8)


# ── 3D Sphere shading ────────────────────────────────────────────────────────

_SPHERE_CACHE: dict = {}

def _sphere(r: int, color_bgr: tuple, shininess: int = 52):
    """Return (img, mask): a Phong-lit sphere, cached by (r, color, shininess)."""
    key = (r, color_bgr, shininess)
    if key in _SPHERE_CACHE:
        return _SPHERE_CACHE[key]

    d = 2 * r + 1
    # Normalised surface-point coordinates on unit sphere
    y_f, x_f = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
    xn = x_f / r; yn = y_f / r
    dist2 = xn**2 + yn**2
    mask = dist2 <= 1.0
    zn = np.sqrt(np.maximum(0.0, 1.0 - dist2))   # z of surface normal

    # Fixed key light: upper-left, slightly toward viewer
    lx, ly, lz = -0.42, -0.70, 0.58
    ln = math.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx/ln, ly/ln, lz/ln

    dot_nl   = xn*lx + yn*ly + zn*lz
    diffuse  = np.maximum(0.0, dot_nl)

    # Phong specular with view = [0, 0, 1]
    rz       = 2.0 * dot_nl * zn - lz
    specular = np.maximum(0.0, rz) ** shininess

    # Rim darkening: edges of sphere fade to shadow
    rim = zn ** 0.30

    # Soft fill light from below-right (reduces cave-shadow look)
    fill = np.maximum(0.0, -xn*0.3 + yn*0.5 + zn*0.4) * 0.22

    img = np.zeros((d, d, 3), dtype=np.float32)
    for c in range(3):
        bc = color_bgr[c] / 255.0
        val = bc * (0.16 + (diffuse * 0.78 + fill) * rim) + specular * 0.92
        img[:, :, c] = np.where(mask, np.clip(val, 0.0, 1.0) * 255.0, 0.0)

    result = (img.astype(np.uint8), mask)
    _SPHERE_CACHE[key] = result
    return result


def _blit(frame, img, mask, cx, cy):
    """Stamp a pre-computed sphere image onto frame at (cx, cy)."""
    r  = (img.shape[0] - 1) // 2
    fh, fw = frame.shape[:2]
    fy1 = max(0, cy - r); fy2 = min(fh, cy + r + 1)
    fx1 = max(0, cx - r); fx2 = min(fw, cx + r + 1)
    sy1 = fy1 - (cy - r); sy2 = sy1 + (fy2 - fy1)
    sx1 = fx1 - (cx - r); sx2 = sx1 + (fx2 - fx1)
    if fy2 <= fy1 or fx2 <= fx1:
        return
    roi   = frame[fy1:fy2, fx1:fx2]
    src   = img  [sy1:sy2, sx1:sx2]
    m     = mask [sy1:sy2, sx1:sx2]
    roi[m] = src[m]


# ── Fruit ─────────────────────────────────────────────────────────────────────

class Fruit:
    def __init__(self, x, y, vx, vy, kind, radius):
        self.x=float(x); self.y=float(y)
        self.vx=float(vx); self.vy=float(vy)
        self.kind=kind; self.radius=radius
        self.angle=random.uniform(0,360); self.spin=random.uniform(-5,5)
        self.sliced=False

    def update(self):
        self.vy+=GRAVITY; self.x+=self.vx; self.y+=self.vy
        self.angle=(self.angle+self.spin)%360

    def center(self): return (int(self.x), int(self.y))

    def draw(self, frame, t):
        cx,cy=self.center(); r=self.radius
        if self.kind=="bomb":
            _draw_bomb(frame, cx, cy, r, self.angle, t)
        else:
            _draw_fruit(frame, cx, cy, r, self.kind, self.angle, t)


def _draw_fruit(frame, cx, cy, r, kind, angle, t):
    info  = FRUIT_KINDS[kind]
    outer = info["outer"]

    # Drop shadow (elliptical for perspective feel)
    cv2.ellipse(frame, (cx+6,cy+8), (r+4, r+2), 0, 0, 360, (12,10,6), -1, cv2.LINE_AA)

    # Three-layer bloom glow (outer→inner, each slightly smaller and brighter)
    pulse = 0.5 + 0.5*math.sin(t*2.5 + cx*0.02)
    for radius_add, alpha in [(16, 0.07+0.04*pulse),
                               (10, 0.12+0.05*pulse),
                               ( 6, 0.18+0.06*pulse)]:
        cv2.circle(frame, (cx,cy), r+radius_add, _dim(outer, alpha), -1, cv2.LINE_AA)

    # ── 3-D Phong sphere ─────────────────────────────────────────────────────
    sph_img, sph_mask = _sphere(r, outer)
    _blit(frame, sph_img, sph_mask, cx, cy)

    # ── Per-fruit surface details (drawn on top of sphere) ───────────────────
    if kind == "watermelon":
        # Darker green stripes that follow the spin
        for i in range(4):
            a  = math.radians(angle + i*45)
            p1 = (int(cx + 2*(r//r)*math.cos(a)),     int(cy + 2*math.sin(a)))
            p2 = (int(cx + (r-4)*math.cos(a)),         int(cy + (r-4)*math.sin(a)))
            cv2.line(frame, p1, p2, _dim(outer, 0.42), 3, cv2.LINE_AA)

    elif kind == "apple":
        stem_tip = (cx, cy - r - r//3)
        cv2.line(frame, (cx, cy-r), stem_tip, (28, 85, 22), 2, cv2.LINE_AA)
        a  = math.radians(angle + 40)
        lx = int(cx + r//3*math.cos(a)); ly = int(cy - r + r//3*math.sin(a))
        cv2.ellipse(frame, (lx,ly), (r//4, r//6), int(angle)+40,
                    0, 360, (32, 110, 28), -1, cv2.LINE_AA)

    elif kind == "orange":
        # Navel + faint segment lines (slightly transparent feel)
        cv2.circle(frame, (cx,cy), max(2, r//6), (15, 90, 200), -1, cv2.LINE_AA)
        for i in range(8):
            a  = math.radians(angle + i*45)
            ex = int(cx + (r-5)*math.cos(a)); ey = int(cy + (r-5)*math.sin(a))
            cv2.line(frame, (cx,cy), (ex,ey), _dim(outer, 0.58), 1, cv2.LINE_AA)

    elif kind == "peach":
        a   = math.radians(angle)
        gx1 = int(cx + 3*math.cos(a - math.pi/2))
        gy1 = int(cy + 3*math.sin(a - math.pi/2))
        gx2 = int(cx + (r-4)*math.cos(a))
        gy2 = int(cy + (r-4)*math.sin(a))
        cv2.line(frame, (gx1,gy1), (gx2,gy2), _dim(outer, 0.48), 2, cv2.LINE_AA)
        lx  = int(cx + r*0.5*math.cos(a)); ly = int(cy + r*0.5*math.sin(a))
        cv2.ellipse(frame, (lx,ly), (r//4, r//6), int(angle),
                    0, 360, (28, 100, 22), -1, cv2.LINE_AA)

    elif kind == "strawberry":
        for i in range(7):
            sa = math.radians(angle + i*51)
            sx = int(cx + r*0.50*math.cos(sa))
            sy = int(cy + r*0.50*math.sin(sa))
            cv2.circle(frame, (sx,sy), 2, (205, 215, 235), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, int(cy - r*0.78)), (r//2, r//3),
                    int(angle), 0, 180, (28, 108, 25), -1, cv2.LINE_AA)

    # Thin rim — darkens the sphere edge for extra depth
    cv2.circle(frame, (cx,cy), r, _dim(outer, 0.28), 2, cv2.LINE_AA)


def _draw_bomb(frame, cx, cy, r, angle, t):
    # Pulsing red danger glow
    pulse = 0.5 + 0.5*math.sin(t*6.0)
    cv2.circle(frame, (cx,cy), r + int(10 + pulse*8),
               _dim((30,30,180), 0.12 + 0.10*pulse), -1, cv2.LINE_AA)

    # Shadow
    cv2.circle(frame, (cx+6, cy+6), r+2, (12,10,8), -1, cv2.LINE_AA)

    # ── 3-D Phong sphere (very dark, high shininess for metallic look) ────────
    bomb_col = (38, 38, 38)
    sph_img, sph_mask = _sphere(r, bomb_col, shininess=90)
    _blit(frame, sph_img, sph_mask, cx, cy)

    # Dark rim
    cv2.circle(frame, (cx,cy), r, (50,50,50), 2, cv2.LINE_AA)

    # Fuse cord
    fx  = int(cx + r*math.cos(math.radians(angle - 60)))
    fy  = int(cy + r*math.sin(math.radians(angle - 60)))
    fx2 = int(cx + (r + r//2)*math.cos(math.radians(angle - 70)))
    fy2 = int(cy + (r + r//2)*math.sin(math.radians(angle - 70)))
    cv2.line(frame, (fx,fy), (fx2,fy2), (28,70,135), 3, cv2.LINE_AA)

    # Sparking tip
    spark_col = (18, 210, 255) if pulse > 0.5 else (18, 155, 225)
    cv2.circle(frame, (fx2,fy2), 5, spark_col, -1, cv2.LINE_AA)
    cv2.circle(frame, (fx2,fy2), 9, _dim(spark_col, 0.38), 2, cv2.LINE_AA)


# ── Particles ─────────────────────────────────────────────────────────────────

class JuiceParticle:
    def __init__(self, x, y, color):
        self.x=float(x); self.y=float(y)
        self.vx=random.uniform(-11,11); self.vy=random.uniform(-16,-4)
        self.color=color
        self.life=random.randint(18,30); self.max_life=self.life
        self.r=random.randint(3,10)
        self.px=self.x; self.py=self.y   # previous position for motion trail

    def update(self):
        self.px=self.x; self.py=self.y
        self.vy+=0.72; self.x+=self.vx; self.y+=self.vy; self.life-=1

    def draw(self, frame):
        a  = self.life / self.max_life
        r  = max(1, int(self.r * a))
        cx,cy = int(self.x), int(self.y)
        # Motion streak (comet tail toward previous position)
        px,py = int(self.px), int(self.py)
        if (px,py) != (cx,cy):
            cv2.line(frame, (px,py), (cx,cy),
                     _dim(self.color, a*0.55), max(1,r-1), cv2.LINE_AA)
        # Main droplet
        cv2.circle(frame, (cx,cy), r, _dim(self.color,a), -1, cv2.LINE_AA)
        # Tiny highlight to make each droplet look glossy
        if r >= 4:
            hx=cx-max(1,r//3); hy=cy-max(1,r//3)
            cv2.circle(frame,(hx,hy),max(1,r//3),_dim((255,255,255),a*0.55),-1,cv2.LINE_AA)

    @property
    def dead(self): return self.life<=0


class HalfPiece:
    """Cut fruit half — shows outer skin + flesh colour."""
    def __init__(self, x, y, vx, vy, radius, outer, inner, start_angle, span):
        self.x=float(x); self.y=float(y)
        self.vx=float(vx); self.vy=float(vy)
        self.radius=radius; self.outer=outer; self.inner=inner
        self.start=start_angle; self.span=span
        self.rot=0.0; self.spin=random.uniform(-10,10)
        self.life=26; self.max_life=26

    def update(self):
        self.vy+=GRAVITY; self.x+=self.vx; self.y+=self.vy
        self.rot=(self.rot+self.spin)%360; self.life-=1

    def draw(self, frame):
        a   = self.life/self.max_life
        cx,cy = int(self.x),int(self.y)
        r   = self.radius
        st  = int(self.start+self.rot)

        # Shadow
        cv2.ellipse(frame,(cx+4,cy+4),(r,r),0,st,st+self.span,(12,10,6),-1,cv2.LINE_AA)

        # Outer skin half
        cv2.ellipse(frame,(cx,cy),(r,r),0,st,st+self.span,
                    _dim(self.outer,a),-1,cv2.LINE_AA)

        # Flesh on the flat cut face
        face_r = max(2, r-4)
        face_a = st + self.span//2
        cv2.ellipse(frame,(cx,cy),(face_r,max(2,r//3)),face_a,
                    -85, 85, _dim(self.inner,a*0.95),-1,cv2.LINE_AA)

        # Highlight arc on the upper curved surface
        bright = tuple(min(255,int(c*1.45)) for c in self.outer)
        cv2.ellipse(frame,(cx,cy),(r-3,r-3),0,
                    st+20, st+self.span//2, _dim(bright,a*0.75), 2, cv2.LINE_AA)

        # Dark rim
        cv2.ellipse(frame,(cx,cy),(r,r),0,st,st+self.span,
                    _dim(self.outer,a*0.30),2,cv2.LINE_AA)

        # Seeds for watermelon
        if r > 35:
            for i in range(3):
                seed_a = math.radians(face_a + i*40 - 40)
                sx=int(cx+(r//3)*math.cos(seed_a)); sy=int(cy+(r//3)*math.sin(seed_a))
                cv2.ellipse(frame,(sx,sy),(3,5),int(face_a)+i*40,0,360,
                            _dim((5,5,15),a),-1)

    @property
    def dead(self): return self.life<=0


class SliceFlash:
    """Starburst + double expanding ring at cut point."""
    def __init__(self, x, y, color):
        self.x=x; self.y=y; self.color=color
        self.life=26; self.max_life=26

    def update(self): self.life-=1

    def draw(self, frame):
        t = 1.0 - self.life/self.max_life
        a = 1.0 - t

        # Outer expanding colour ring
        rr1 = int(t * 95) + 5
        cv2.circle(frame,(self.x,self.y),rr1,_dim(self.color,a*0.80),
                   max(1,int(5*a)),cv2.LINE_AA)
        # Slightly delayed second ring
        if t > 0.15:
            rr2 = int((t-0.15)*80) + 5
            cv2.circle(frame,(self.x,self.y),rr2,_dim(self.color,a*0.50),
                       max(1,int(3*a)),cv2.LINE_AA)

        # Starburst rays (first 10 frames)
        if self.life > 16:
            ray_a = (self.life - 16) / 10.0
            n = 8
            for i in range(n):
                ang = math.radians(i * 45 + t * 15)
                r1  = 6
                r2  = int(6 + ray_a * 38)
                p1  = (int(self.x + r1*math.cos(ang)), int(self.y + r1*math.sin(ang)))
                p2  = (int(self.x + r2*math.cos(ang)), int(self.y + r2*math.sin(ang)))
                cv2.line(frame, p1, p2, (255,255,255), max(1,int(2*ray_a)), cv2.LINE_AA)
                # Coloured inner ray
                cv2.line(frame, p1,
                         (int(self.x+(r1+6)*math.cos(ang)), int(self.y+(r1+6)*math.sin(ang))),
                         _dim(self.color,ray_a*0.8), max(1,int(3*ray_a)), cv2.LINE_AA)

        # Bright white + coloured core flash
        if self.life > 17:
            cr = max(2, int((1-t)*36))
            cv2.circle(frame,(self.x,self.y),cr, _dim(self.color,0.7),-1,cv2.LINE_AA)
            cv2.circle(frame,(self.x,self.y),max(2,cr-6),(255,255,255),-1,cv2.LINE_AA)

    @property
    def dead(self): return self.life<=0


class Sparkle:
    def __init__(self, x, y, color=None):
        ang=random.uniform(0,math.pi*2); spd=random.uniform(3,10)
        self.x=float(x); self.y=float(y)
        self.vx=math.cos(ang)*spd; self.vy=math.sin(ang)*spd
        self.color=color or GOLD
        self.life=random.randint(14,24); self.max_life=self.life
        self.size=random.randint(2,6)

    def update(self):
        self.vx*=0.90; self.vy*=0.90; self.x+=self.vx; self.y+=self.vy; self.life-=1

    def draw(self, frame):
        a=self.life/self.max_life; col=_dim(self.color,a)
        x,y=int(self.x),int(self.y); s=max(1,int(self.size*a))
        cv2.line(frame,(x-s,y),(x+s,y),col,1,cv2.LINE_AA)
        cv2.line(frame,(x,y-s),(x,y+s),col,1,cv2.LINE_AA)
        cv2.line(frame,(x-s,y-s),(x+s,y+s),col,1,cv2.LINE_AA)
        cv2.line(frame,(x+s,y-s),(x-s,y+s),col,1,cv2.LINE_AA)

    @property
    def dead(self): return self.life<=0


class FloatingText:
    def __init__(self, x, y, text, color=(255,255,255), scale=0.85):
        self.x=float(x); self.y=float(y); self.text=text
        self.color=color; self.scale=scale; self.life=50; self.max_life=50

    def update(self): self.y-=1.5; self.life-=1

    def draw(self, frame):
        a=self.life/self.max_life
        _text(frame, self.text,(int(self.x),int(self.y)),
              cv2.FONT_HERSHEY_DUPLEX, self.scale, _dim(self.color,a),
              depth=max(1, int(self.scale*3)))

    @property
    def dead(self): return self.life<=0


# ── Sound ─────────────────────────────────────────────────────────────────────

def _to_wav(arr_f32):
    """Convert float32 numpy array to in-memory WAV bytes (mono 16-bit)."""
    s16 = (np.clip(arr_f32, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(_SR)
        wf.writeframes(s16.tobytes())
    return buf.getvalue()


class SoundManager:
    """Synthesizes game sounds and plays them via winsound (Windows stdlib)."""

    _FLAGS = 0  # set in __init__ if audio available

    def __init__(self):
        self._ok = False
        if not _AUDIO_OK:
            return
        try:
            self._FLAGS = _ws.SND_MEMORY | _ws.SND_ASYNC | _ws.SND_NODEFAULT
            rng = np.random.default_rng(42)
            self._slice = _to_wav(self._gen_slice(rng))
            self._bomb  = _to_wav(self._gen_bomb(rng))
            self._miss  = _to_wav(self._gen_miss())
            self._combo = _to_wav(self._gen_combo())
            self._life  = _to_wav(self._gen_life_lost(rng))
            self._ok = True
        except Exception:
            pass

    # ── synthesis ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ad(t, attack=0.005, decay=1.0):
        return np.clip(t / attack, 0, 1) * np.exp(-t * decay)

    def _gen_slice(self, rng):
        dur = 0.14
        t = np.linspace(0, dur, int(_SR * dur), False)
        freq = np.linspace(1200, 180, len(t))
        phase = np.cumsum(freq / _SR * 2 * np.pi)
        wave_ = np.sin(phase) * 0.45 + rng.standard_normal(len(t)) * 0.40
        return np.clip(wave_ * self._ad(t, 0.003, 28) * 0.75, -1, 1).astype(np.float32)

    def _gen_bomb(self, rng):
        dur = 0.45
        t = np.linspace(0, dur, int(_SR * dur), False)
        freq = np.linspace(110, 35, len(t))
        phase = np.cumsum(freq / _SR * 2 * np.pi)
        body   = np.sin(phase) * np.exp(-t * 5)
        click  = np.exp(-t * 180) * 0.7
        rumble = rng.standard_normal(len(t)) * np.exp(-t * 12) * 0.18
        return np.clip((body + click + rumble) * 0.85, -1, 1).astype(np.float32)

    def _gen_miss(self):
        dur = 0.28
        t = np.linspace(0, dur, int(_SR * dur), False)
        freq = np.linspace(380, 55, len(t))
        phase = np.cumsum(freq / _SR * 2 * np.pi)
        return np.clip(np.sin(phase) * np.exp(-t * 5) * 0.45, -1, 1).astype(np.float32)

    def _gen_combo(self):
        dur = 0.35
        t = np.linspace(0, dur, int(_SR * dur), False)
        f = 1046.5
        wave_ = (np.sin(2*np.pi*f*t)*0.55 + np.sin(2*np.pi*f*2*t)*0.25 +
                 np.sin(2*np.pi*f*3*t)*0.12 + np.sin(2*np.pi*f*4.1*t)*0.05)
        return np.clip(wave_ * self._ad(t, 0.004, 9) * 0.80, -1, 1).astype(np.float32)

    def _gen_life_lost(self, rng):
        dur = 0.55
        t = np.linspace(0, dur, int(_SR * dur), False)
        freq = np.linspace(220, 55, len(t))
        phase = np.cumsum(freq / _SR * 2 * np.pi)
        groan  = np.sin(phase) * np.exp(-t * 4) * 0.6
        crunch = rng.standard_normal(len(t)) * np.exp(-t * 40) * 0.35
        return np.clip((groan + crunch) * 0.80, -1, 1).astype(np.float32)

    # ── playback ───────────────────────────────────────────────────────────────

    def _play(self, wav_bytes):
        if not self._ok:
            return
        try:
            _ws.PlaySound(wav_bytes, self._FLAGS)
        except Exception:
            pass

    def play_slice(self):     self._play(self._slice)
    def play_bomb(self):      self._play(self._bomb)
    def play_miss(self):      self._play(self._miss)
    def play_combo(self):     self._play(self._combo)
    def play_life_lost(self): self._play(self._life)


# ── Game ──────────────────────────────────────────────────────────────────────

class Game:
    MENU="MENU"; COUNTDOWN="COUNTDOWN"; PLAYING="PLAYING"; GAME_OVER="GAME_OVER"
    PLAYER_IDS=("P1","P2")

    def __init__(self, fw, fh):
        self.fw,self.fh=fw,fh
        self.bg=Background(fw,fh)
        self.sounds=SoundManager()
        self.state=self.MENU; self._mode=MODE_1P
        self._palm_hold_start=None
        self._btn_hover={MODE_1P:None,MODE_2P:None}
        self._countdown_start=None
        self._t0=time.time()
        self._reset()

    def _reset(self):
        self.bg.reset()
        self.scores={p:0 for p in self.PLAYER_IDS}
        self.lives=LIVES_START
        self.fruits=[]; self.particles=[]; self.float_texts=[]
        self.trails={p:deque(maxlen=TRAIL_LEN) for p in self.PLAYER_IDS}
        self._next_spawn=time.time()+0.8
        self._combos={p:0 for p in self.PLAYER_IDS}
        self._combo_times={p:0.0 for p in self.PLAYER_IDS}
        self._flash={p:0 for p in self.PLAYER_IDS}
        self._score_pop={p:0 for p in self.PLAYER_IDS}
        self._sliced_ids=set(); self._game_end=None
        self._flashbang=0

    # ── Button rects ──────────────────────────────────────────────────────────

    def _btn_rect(self, mode):
        return ((self.fw//2-320, self.fh//2+30, 275, 105) if mode==MODE_1P
                else (self.fw//2+45, self.fh//2+30, 275, 105))

    # ── Palm hold (GAME_OVER → MENU) ─────────────────────────────────────────

    def update_palm_hold(self, is_palm):
        if is_palm:
            if self._palm_hold_start is None: self._palm_hold_start=time.time()
            elif time.time()-self._palm_hold_start>=1.0:
                self._palm_hold_start=None; return True
        else: self._palm_hold_start=None
        return False

    def palm_progress(self):
        if self._palm_hold_start is None: return 0.0
        return min(1.0,(time.time()-self._palm_hold_start)/1.0)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, hands):
        now=time.time(); self.bg.draw  # bg.update not needed (static base)

        if   self.state==self.MENU:      self._update_menu(hands,now)
        elif self.state==self.COUNTDOWN: self._update_countdown(now)
        elif self.state==self.PLAYING:   self._update_playing(hands,now)

    def _update_menu(self, hands, now):
        for mode in (MODE_1P,MODE_2P):
            bx,by,bw,bh=self._btn_rect(mode)
            hovering=any(bx<=sp[0]<=bx+bw and by<=sp[1]<=by+bh for _,_,sp in hands)
            if hovering:
                if self._btn_hover[mode] is None: self._btn_hover[mode]=now
                elif now-self._btn_hover[mode]>=HOVER_TIME:
                    self._btn_hover={MODE_1P:None,MODE_2P:None}
                    self._start(mode); return
            else: self._btn_hover[mode]=None

    def _start(self, mode):
        self._reset(); self._mode=mode
        self._countdown_start=time.time(); self.state=self.COUNTDOWN

    def _update_countdown(self, now):
        if now-self._countdown_start>=COUNTDOWN_SECS:
            self.state=self.PLAYING
            if self._mode==MODE_2P: self._game_end=now+GAME_DURATION
            for _ in range(24):
                self.particles.append(
                    Sparkle(self.fw//2+random.randint(-80,80),
                            self.fh//2+random.randint(-60,60)))
            self.float_texts.append(
                FloatingText(self.fw//2-55,self.fh//2+10,"GO!",(50,240,100),2.5))

    def _update_playing(self, hands, now):
        for pid,_,sp in hands: self.trails[pid].append(sp)

        if self._mode==MODE_2P and self._game_end and now>=self._game_end:
            self.state=self.GAME_OVER; return
        if self._mode==MODE_1P and self.lives<=0:
            self.state=self.GAME_OVER; return

        if now>=self._next_spawn:
            self._spawn_wave(); self._next_spawn=now+random.uniform(SPAWN_LO,SPAWN_HI)

        for pid in self.PLAYER_IDS:
            if self._combos[pid]>0 and now-self._combo_times[pid]>COMBO_WINDOW:
                self._combos[pid]=0

        new_fruits=[]
        for f in self.fruits:
            f.update()
            if f.y<=self.fh+MISS_PENALTY_Y:
                new_fruits.append(f)
            elif f.kind!="bomb":
                self.sounds.play_miss()
        self.fruits=new_fruits

        self._sliced_ids=set()
        for pid,_,_ in hands: self._check_slices(pid)

        for p in self.particles: p.update()
        self.particles=[p for p in self.particles if not p.dead]
        for t in self.float_texts: t.update()
        self.float_texts=[t for t in self.float_texts if not t.dead]
        for pid in self.PLAYER_IDS:
            if self._flash[pid]>0: self._flash[pid]-=1
            if self._score_pop[pid]>0: self._score_pop[pid]-=1

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_wave(self):
        for _ in range(random.randint(WAVE_MIN,WAVE_MAX)):
            x=random.randint(int(self.fw*0.1),int(self.fw*0.9))
            vy=random.uniform(-26,-18); vx=random.uniform(-3.5,3.5)
            if random.random()<BOMB_CHANCE:
                self.fruits.append(Fruit(x,self.fh+10,vx,vy,"bomb",random.randint(25,33)))
            else:
                kind=random.choice(FRUIT_NAMES)
                rmin,rmax=FRUIT_KINDS[kind]["r"]
                self.fruits.append(Fruit(x,self.fh+10,vx,vy,kind,random.randint(rmin,rmax)))

    # ── Collision ─────────────────────────────────────────────────────────────

    def _check_slices(self, pid):
        trail=list(self.trails[pid])
        if len(trail)<2: return
        recent=trail[-8:]
        speed=max(_dist(recent[i],recent[i+1]) for i in range(len(recent)-1))
        if speed<SWIPE_THRESHOLD: return
        to_remove=[]
        for i,f in enumerate(self.fruits):
            if i in self._sliced_ids or f.sliced: continue
            cx,cy=f.center()
            if any(_seg_dist((cx,cy),trail[j],trail[j+1])<f.radius+18
                   for j in range(len(trail)-1)):
                self._sliced_ids.add(i); f.sliced=True
                (self._hit_bomb if f.kind=="bomb" else self._hit_fruit)(f,pid)
                to_remove.append(i)
        if to_remove:
            self.fruits=[f for i,f in enumerate(self.fruits) if i not in to_remove]

    def _hit_fruit(self, f, pid):
        cx,cy=f.center()
        info=FRUIT_KINDS[f.kind]
        outer,inner,juice=info["outer"],info["inner"],info["juice"]

        self._combos[pid]+=1; self._combo_times[pid]=time.time()
        mult=max(1,self._combos[pid]); pts=10*mult
        self.scores[pid]+=pts; self._score_pop[pid]=14
        self.sounds.play_slice()

        pcol=P1_COL if pid=="P1" else P2_COL
        label=f"+{pts}" if mult==1 else f"+{pts}  x{mult}"
        self.float_texts.append(FloatingText(cx-25,cy,label,pcol,0.72+0.05*mult))

        # Juice stain on background (persists)
        self.bg.add_stain(cx, cy, juice, f.radius + random.randint(4, 12))

        # Juice particles
        for _ in range(14):
            self.particles.append(JuiceParticle(cx,cy,juice))
        # Halves
        for start,span in [(0,180),(180,180)]:
            self.particles.append(HalfPiece(
                cx,cy,random.uniform(-7,7),random.uniform(-10,-3),
                f.radius,outer,inner,start,span))
        # Slice flash
        self.particles.append(SliceFlash(cx,cy,juice))

        # Combo shout + sparkles
        if mult in COMBO_SHOUTS:
            self.sounds.play_combo()
            self.float_texts.append(FloatingText(
                self.fw//2-90,self.fh//2,COMBO_SHOUTS[mult],GOLD,1.7))
            for _ in range(30):
                self.particles.append(Sparkle(cx+random.randint(-40,40),
                                               cy+random.randint(-40,40),juice))

    def _hit_bomb(self, f, pid):
        cx,cy=f.center()
        self._combos[pid]=0; self._flash[pid]=18
        self._flashbang=80
        self.sounds.play_bomb(); self.sounds.play_life_lost()
        if self._mode==MODE_1P:
            self.lives=max(0,self.lives-1)
            self.float_texts.append(FloatingText(cx-40,cy,"BOOM!  -1 LIFE",(50,50,220),0.9))
        else:
            self.scores[pid]=max(0,self.scores[pid]-BOMB_PENALTY)
            self.float_texts.append(FloatingText(cx-50,cy,f"BOMB!  -{BOMB_PENALTY}",(50,50,220),0.9))
        for _ in range(18): self.particles.append(JuiceParticle(cx,cy,(55,55,55)))
        self.particles.append(SliceFlash(cx,cy,(40,40,180)))

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame, hands):
        t=time.time()-self._t0
        self.bg.draw(frame)

        # Flash overlays
        for pid,_,_ in hands:
            if self._flash[pid]>0:
                ov=np.zeros_like(frame); ov[:]=BANNER_COL
                _blend(frame,ov,0.22*(self._flash[pid]/18))

        for f in self.fruits:  f.draw(frame,t)
        for p in self.particles: p.draw(frame)
        for txt in self.float_texts: txt.draw(frame)

        for pid,lms,_ in hands:
            col=P1_COL if pid=="P1" else P2_COL
            self._draw_trail(frame, self.trails[pid], col)
            self._draw_skeleton(frame, lms, col)

        if   self.state==self.PLAYING:   self._draw_hud(frame,t)
        elif self.state==self.MENU:      self._draw_menu(frame,hands,t)
        elif self.state==self.COUNTDOWN: self._draw_countdown(frame,hands)
        elif self.state==self.GAME_OVER: self._draw_game_over(frame)

        self._draw_flashbang(frame)

    def _draw_flashbang(self, frame):
        if self._flashbang <= 0:
            return
        total = 80
        progress = 1.0 - (self._flashbang / total)  # 0 → 1 as flash fades
        # Hold peak white for first 18% of duration, then curve down
        if progress < 0.18:
            alpha = 0.97
        else:
            t_fade = (progress - 0.18) / 0.82
            alpha = 0.97 * (1.0 - t_fade ** 0.50)
        white = np.full_like(frame, 255)
        cv2.addWeighted(white, alpha, frame, 1.0 - alpha, 0, frame)
        self._flashbang -= 1

    def _draw_trail(self, frame, trail_dq, col):
        trail=list(trail_dq)
        if len(trail)<2: return
        recent=trail[-8:]
        speed=(max(_dist(recent[i],recent[i+1]) for i in range(len(recent)-1))
               if len(recent)>=2 else 0)

        cv2.circle(frame,trail[-1],6,_dim(col,0.5),-1,cv2.LINE_AA)
        if speed<SWIPE_THRESHOLD*0.6: return

        n=len(trail)
        # Outer glow
        for i in range(1,n):
            a=i/n
            cv2.line(frame,trail[i-1],trail[i],_dim(col,a*0.25),max(1,int(16*a)),cv2.LINE_AA)
        # Colour body
        for i in range(1,n):
            a=i/n
            cv2.line(frame,trail[i-1],trail[i],_dim(col,a*0.7),max(1,int(7*a)),cv2.LINE_AA)
        # Bright white core
        for i in range(1,n):
            a=i/n
            cv2.line(frame,trail[i-1],trail[i],(255,255,255),max(1,int(3*a)),cv2.LINE_AA)

        tip=trail[-1]
        cv2.circle(frame,tip,16,_dim(col,0.4),-1,cv2.LINE_AA)
        cv2.circle(frame,tip,11,_dim(col,0.8),-1,cv2.LINE_AA)
        cv2.circle(frame,tip, 6,(255,255,255),-1,cv2.LINE_AA)

    def _draw_skeleton(self, frame, lms, col):
        bone=_dim(col,0.25)
        for a,b in HAND_CONNECTIONS:
            cv2.line(frame,lms[a],lms[b],bone,1,cv2.LINE_AA)
        for i,pt in enumerate(lms):
            cv2.circle(frame,pt,5 if i in{4,8,12,16,20} else 3,_dim(col,0.6),-1,cv2.LINE_AA)
        cv2.circle(frame,lms[8],9,col,-1,cv2.LINE_AA)
        cv2.circle(frame,lms[8],13,(255,255,255),2,cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, t):
        if self._mode==MODE_1P: self._hud_1p(frame,t)
        else:                    self._hud_2p(frame,t)

    def _hud_1p(self, frame, t):
        fw,fh=self.fw,self.fh
        bh=Background.BANNER_H

        # Score (centre of top banner)
        s=str(self.scores["P1"])
        pop=self._score_pop["P1"]
        scale=2.0+0.4*(pop/14) if pop>0 else 2.0
        sw=cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,scale,3)[0][0]
        _text(frame,s,(fw//2-sw//2,bh-8),cv2.FONT_HERSHEY_DUPLEX,scale,BANNER_GOLD,thickness=3,depth=5)

        # Lives (left of top banner — coloured circles)
        for i in range(LIVES_START):
            cx=30+i*38; cy=bh//2
            if i<self.lives:
                cv2.circle(frame,(cx,cy),14,(50,50,218),-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(110,110,255),2,cv2.LINE_AA)
                cv2.circle(frame,(cx-4,cy-4),4,(180,180,255),-1,cv2.LINE_AA)
            else:
                cv2.circle(frame,(cx,cy),14,(40,35,30),-1,cv2.LINE_AA)
                cv2.circle(frame,(cx,cy),14,(70,60,50),2,cv2.LINE_AA)
                # X mark
                cv2.line(frame,(cx-7,cy-7),(cx+7,cy+7),(80,80,180),2,cv2.LINE_AA)
                cv2.line(frame,(cx+7,cy-7),(cx-7,cy+7),(80,80,180),2,cv2.LINE_AA)

        # Combo
        c=self._combos["P1"]
        if c>=2:
            age=time.time()-self._combo_times["P1"]
            fade=max(0.0,1.0-age/COMBO_WINDOW)
            ctxt=f"COMBO  x{c}"
            cw=cv2.getTextSize(ctxt,cv2.FONT_HERSHEY_DUPLEX,1.1,2)[0][0]
            # Dark pill backing for readability over any background
            px1,py1=fw//2-cw//2-18,fh-bh+28
            px2,py2=fw//2+cw//2+18,fh-bh+58
            roi=frame[py1:py2,px1:px2]
            bg=np.zeros_like(roi); bg[:]=(8,5,3)
            cv2.addWeighted(bg,0.68,roi,0.32,0,roi)
            _text(frame,ctxt,(fw//2-cw//2,fh-bh+50),cv2.FONT_HERSHEY_DUPLEX,1.1,_dim(GOLD,fade),depth=3)

    def _hud_2p(self, frame, t):
        fw,fh=self.fw,self.fh
        bh=Background.BANNER_H

        # Timer bar along top banner
        tl=max(0.0,(self._game_end-time.time()) if self._game_end else GAME_DURATION)
        prog=tl/GAME_DURATION
        bar_col=(50,200,50) if tl>20 else (40,40,200)
        cv2.rectangle(frame,(0,0),(int(fw*prog),8),bar_col,-1)

        # Timer digits
        ts=f"{int(tl):02d}"
        tc=(230,230,230) if tl>15 else (80,80,255)
        tw=cv2.getTextSize(ts,cv2.FONT_HERSHEY_DUPLEX,1.9,3)[0][0]
        _text(frame,ts,(fw//2-tw//2,bh-8),cv2.FONT_HERSHEY_DUPLEX,1.9,tc,thickness=3,depth=4)

        # P1 score (left banner)
        self._draw_player_score(frame,"P1",20,bh,"left")
        # P2 score (right banner)
        self._draw_player_score(frame,"P2",fw-20,bh,"right")

    def _draw_player_score(self, frame, pid, x, bh, align):
        col=P1_COL if pid=="P1" else P2_COL
        score=self.scores[pid]; pop=self._score_pop[pid]
        scale=1.6+0.3*(pop/14) if pop>0 else 1.6
        s=str(score); sw=cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,scale,2)[0][0]
        sx=x if align=="left" else x-sw
        _text(frame,s,(sx,bh-8),cv2.FONT_HERSHEY_DUPLEX,scale,col,depth=4)

        # Player label
        lbl=pid+" "; lw=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,1)[0][0]
        lx=x if align=="left" else x-lw
        _text(frame,lbl,(lx,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,_dim(col,0.8),thickness=1)

        # Combo
        c=self._combos[pid]
        if c>=2:
            age=time.time()-self._combo_times[pid]
            fade=max(0.0,1.0-age/COMBO_WINDOW)
            ct=f"x{c}"; ctw=cv2.getTextSize(ct,cv2.FONT_HERSHEY_DUPLEX,0.9,2)[0][0]
            ctx=x if align=="left" else x-ctw
            _text(frame,ct,(ctx,bh+35),cv2.FONT_HERSHEY_DUPLEX,0.9,_dim(GOLD,fade),depth=3)

    # ── Menu ──────────────────────────────────────────────────────────────────

    def _draw_menu(self, frame, hands, t):
        fw,fh=self.fw,self.fh; bh=Background.BANNER_H

        # Dim the play area
        ov=frame.copy()
        cv2.rectangle(ov,(0,bh),(fw,fh-bh),(0,0,0),-1)
        _blend(frame,ov,0.35)

        # Glowing title (within top banner area — redraw over banner)
        glow=0.5+0.5*math.sin(t*1.8)
        # Shadow glow passes
        for off in range(4,0,-1):
            cv2.putText(frame,"FRUIT NINJA",(fw//2-208,bh-10),
                        cv2.FONT_HERSHEY_DUPLEX,2.0,
                        _dim(BANNER_GOLD,0.05*off*glow),off*2,cv2.LINE_AA)
        _text(frame,"FRUIT NINJA",(fw//2-208,bh-10),
              cv2.FONT_HERSHEY_DUPLEX,2.0,BANNER_GOLD,thickness=3,depth=6)

        # Subtitle
        sub="GESTURE EDITION"
        sw=cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.85,1)[0][0]
        _text(frame,sub,(fw//2-sw//2,bh+48),cv2.FONT_HERSHEY_SIMPLEX,0.85,(210,195,165))

        # Instruction
        ins="Hover your hand over a button to select mode"
        iw=cv2.getTextSize(ins,cv2.FONT_HERSHEY_SIMPLEX,0.65,1)[0][0]
        _text(frame,ins,(fw//2-iw//2,bh+84),cv2.FONT_HERSHEY_SIMPLEX,0.65,(175,160,135),
              thickness=1,shadow=False)

        # Buttons
        hand_pts=[sp for _,_,sp in hands]
        for mode,col,sub_txt,icon in [
            (MODE_1P, P1_COL, "3 Lives  |  Endless",   "1P"),
            (MODE_2P, P2_COL, "90 Seconds  |  2 Hands","2P"),
        ]:
            bx,by,bw,bh2=self._btn_rect(mode)
            hvr=self._btn_hover[mode]
            prog=min(1.0,(time.time()-hvr)/HOVER_TIME) if hvr else 0.0
            active=any(bx<=sp[0]<=bx+bw and by<=sp[1]<=by+bh2 for sp in hand_pts)

            # Button fill (3-D extruded box)
            bg_col=_dim(col,0.10+0.14*prog)
            _rounded_box(frame,bx,by,bw,bh2,bg_col,filled=True,depth=6)
            # Progress bar at bottom of button
            if prog>0:
                fill=int(bw*prog)
                cv2.rectangle(frame,(bx+8,by+bh2-10),(bx+8+fill,by+bh2-4),col,-1)
            # Animated border
            pulse=0.6+0.4*math.sin(t*4)*int(active)
            _rounded_box(frame,bx,by,bw,bh2,_dim(col,0.5+0.4*pulse),filled=False)

            # Icon circle
            cv2.circle(frame,(bx+38,by+bh2//2),24,_dim(col,0.25),-1,cv2.LINE_AA)
            cv2.circle(frame,(bx+38,by+bh2//2),24,_dim(col,0.6),2,cv2.LINE_AA)
            iw2=cv2.getTextSize(icon,cv2.FONT_HERSHEY_DUPLEX,0.85,2)[0][0]
            _text(frame,icon,(bx+38-iw2//2,by+bh2//2+8),
                  cv2.FONT_HERSHEY_DUPLEX,0.85,col if active else _dim(col,0.75))

            # Label + sub
            lbl_col=col if active else _dim(col,0.75)
            lw=cv2.getTextSize(mode.replace("1P","1 PLAYER").replace("2P","2 PLAYERS"),
                               cv2.FONT_HERSHEY_DUPLEX,1.05,2)[0][0]
            lbl_full="1 PLAYER" if mode==MODE_1P else "2 PLAYERS"
            _text(frame,lbl_full,(bx+74,by+50),cv2.FONT_HERSHEY_DUPLEX,1.05,lbl_col,depth=4)
            sw2=cv2.getTextSize(sub_txt,cv2.FONT_HERSHEY_SIMPLEX,0.60,1)[0][0]
            _text(frame,sub_txt,(bx+74,by+78),cv2.FONT_HERSHEY_SIMPLEX,0.60,
                  _dim(col,0.55),thickness=1,shadow=False)

        # Bottom legend
        _text(frame,"P1 = left hand",(fw//2-210,fh-bh+48),
              cv2.FONT_HERSHEY_SIMPLEX,0.68,P1_COL)
        _text(frame,"P2 = right hand",(fw//2+30,fh-bh+48),
              cv2.FONT_HERSHEY_SIMPLEX,0.68,P2_COL)

    # ── Countdown ─────────────────────────────────────────────────────────────

    def _draw_countdown(self, frame, hands):
        fw,fh=self.fw,self.fh; bh=Background.BANNER_H
        elapsed=time.time()-self._countdown_start
        phase=min(2,int(elapsed)); frac=elapsed-phase

        labels=["3","2","1"]
        colors=[(50,180,255),(80,220,255),(50,255,180)]
        txt=labels[phase]; col=colors[phase]
        scale=4.5-frac*2.5
        alpha=1.0 if frac<0.6 else max(0.05,1.0-(frac-0.6)/0.4)

        # Ring pulse
        rng2=int(50+frac*100)
        cv2.circle(frame,(fw//2,fh//2),rng2,_dim(col,alpha*0.14),rng2//4,cv2.LINE_AA)

        tw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,scale,8)[0][0]
        th=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,scale,8)[0][1]
        _text(frame,txt,(fw//2-tw//2,fh//2+th//2),
              cv2.FONT_HERSHEY_DUPLEX,scale,_dim(col,max(0.05,alpha)),thickness=8,
              depth=max(1,int(scale*2.2)))

        _text(frame,"PLAYER 1",(60,fh//2),cv2.FONT_HERSHEY_DUPLEX,0.85,P1_COL)
        if self._mode==MODE_2P:
            lw=cv2.getTextSize("PLAYER 2",cv2.FONT_HERSHEY_DUPLEX,0.85,2)[0][0]
            _text(frame,"PLAYER 2",(fw-lw-60,fh//2),cv2.FONT_HERSHEY_DUPLEX,0.85,P2_COL)

        # Float texts (GO! etc)
        for t in self.float_texts: t.draw(frame)
        for p in self.particles:   p.draw(frame)

    # ── Game Over ─────────────────────────────────────────────────────────────

    def _draw_game_over(self, frame):
        fw,fh=self.fw,self.fh; bh=Background.BANNER_H
        ov=frame.copy()
        cv2.rectangle(ov,(0,bh),(fw,fh-bh),(0,0,0),-1)
        _blend(frame,ov,0.62)

        p1=self.scores["P1"]; p2=self.scores["P2"]
        if   self._mode==MODE_1P:  wt,wc="GAME OVER",P1_COL; sl=[(f"Score:  {p1}",(255,255,255))]
        elif p1>p2:                wt,wc="PLAYER 1 WINS!",P1_COL; sl=[(f"P1:  {p1}",P1_COL),(f"P2:  {p2}",_dim(P2_COL,0.6))]
        elif p2>p1:                wt,wc="PLAYER 2 WINS!",P2_COL; sl=[(f"P1:  {p1}",_dim(P1_COL,0.6)),(f"P2:  {p2}",P2_COL)]
        else:                      wt,wc="TIE GAME!",GOLD;       sl=[(f"P1:  {p1}",P1_COL),(f"P2:  {p2}",P2_COL)]

        # Glow halo
        ww=cv2.getTextSize(wt,cv2.FONT_HERSHEY_DUPLEX,2.0,4)[0][0]
        for off in range(5,0,-1):
            cv2.putText(frame,wt,(fw//2-ww//2,fh//2-88),cv2.FONT_HERSHEY_DUPLEX,2.0,
                        _dim(wc,0.04*off),off*3,cv2.LINE_AA)
        _text(frame,wt,(fw//2-ww//2,fh//2-88),cv2.FONT_HERSHEY_DUPLEX,2.0,wc,thickness=4,depth=7)

        for i,(txt,col) in enumerate(sl):
            sw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,1.3,2)[0][0]
            _text(frame,txt,(fw//2-sw//2,fh//2-18+i*50),cv2.FONT_HERSHEY_DUPLEX,1.3,col,depth=4)

        prog=self.palm_progress()
        _progress_bar(frame,fw//2-160,fh//2+82,320,22,prog,BANNER_GOLD)
        ins="Open palm to return to menu"
        iw=cv2.getTextSize(ins,cv2.FONT_HERSHEY_SIMPLEX,0.80,1)[0][0]
        _text(frame,ins,(fw//2-iw//2,fh//2+130),cv2.FONT_HERSHEY_SIMPLEX,0.80,(215,200,170))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    download_model()
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grabber=FrameGrabber(cap); detector=HandDetector(); game=Game(fw,fh)
    cv2.namedWindow("Fruit Ninja",cv2.WINDOW_NORMAL)

    while True:
        frame=grabber.read()
        if frame is None: continue

        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        detector.process(rgb)
        hands=detector.all_hands(fw,fh)
        is_palm=any(detector.is_open_palm(lms,fw) for _,lms,_ in hands)

        if game.state==Game.GAME_OVER:
            if game.update_palm_hold(is_palm):
                game.state=Game.MENU
                game._btn_hover={MODE_1P:None,MODE_2P:None}
        else:
            game._palm_hold_start=None

        canvas=np.zeros((fh,fw,3),dtype=np.uint8)
        game.update(hands)
        game.draw(canvas,hands)

        cv2.imshow("Fruit Ninja",canvas)
        if cv2.waitKey(1)&0xFF==ord("q"): break

    grabber.stop(); detector.close(); cap.release(); cv2.destroyAllWindows()


if __name__=="__main__":
    main()
