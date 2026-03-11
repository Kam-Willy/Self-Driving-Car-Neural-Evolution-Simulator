"""
Vector and geometry utilities used across the simulation.
All functions operate on plain Python floats / NumPy arrays for speed.
"""
import numpy as np


# ── 2-D vector helpers ───────────────────────────────────────────────────────

def vec2(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=float)


def length(v: np.ndarray) -> float:
    return float(np.sqrt(v @ v))


def normalize(v: np.ndarray) -> np.ndarray:
    n = length(v)
    return v / n if n > 1e-9 else v


def rotate(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def perpendicular(v: np.ndarray) -> np.ndarray:
    """90-degree counter-clockwise rotation."""
    return np.array([-v[1], v[0]])


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed angle from v1 to v2 in radians ([-π, π])."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot   = float(np.dot(v1, v2))
    return float(np.arctan2(cross, dot))


# ── Geometry helpers ─────────────────────────────────────────────────────────

def point_to_segment_dist(p: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray) -> float:
    """Closest distance from point p to segment a–b."""
    ab = b - a
    t  = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t  = float(np.clip(t, 0.0, 1.0))
    closest = a + t * ab
    return length(p - closest)


def closest_point_on_polyline(p: np.ndarray,
                               polyline: list) -> tuple:
    """
    Return (closest_point, segment_index, t) for a list of np.ndarray points.
    Useful for finding progress along the centre-line.
    """
    best_dist = float('inf')
    best_pt   = polyline[0]
    best_idx  = 0
    best_t    = 0.0

    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab   = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            continue
        t   = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        pt  = a + t * ab
        d   = length(p - pt)
        if d < best_dist:
            best_dist = d
            best_pt   = pt
            best_idx  = i
            best_t    = t

    return best_pt, best_idx, best_t


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    while angle >  np.pi: angle -= 2 * np.pi
    while angle <= -np.pi: angle += 2 * np.pi
    return angle


def car_corners(x: float, y: float,
                angle: float,
                half_len: float,
                half_wid: float) -> np.ndarray:
    """
    Return (4, 2) array of world-space corners of a rotated rectangle.
    Order: front-right, front-left, rear-left, rear-right.
    """
    local = np.array([
        [ half_len,  half_wid],
        [ half_len, -half_wid],
        [-half_len, -half_wid],
        [-half_len,  half_wid],
    ])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([x, y])


def ray_cast_distance(ox: float, oy: float,
                      direction: float,
                      step: float,
                      max_dist: float,
                      hit_fn) -> float:
    """
    March a ray from (ox, oy) in *direction* (radians).
    hit_fn(x, y) -> bool  — returns True when the ray has hit something.
    Returns normalised distance [0, 1].
    """
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    dist = step
    while dist < max_dist:
        cx = ox + dist * cos_d
        cy = oy + dist * sin_d
        if hit_fn(cx, cy):
            return dist / max_dist
        dist += step

    return 1.0


# ── Colour helpers ───────────────────────────────────────────────────────────

def fitness_to_rgb(fitness: float, max_fitness: float) -> tuple:
    """Map normalised fitness to a (r, g, b) tuple (0-255)."""
    t = float(np.clip(fitness / (max_fitness + 1e-9), 0.0, 1.0))
    # green (good) → yellow → red (bad)
    r = int(255 * min(1.0, 2.0 * (1.0 - t)))
    g = int(255 * min(1.0, 2.0 * t))
    b = 60

    return r, g, b
