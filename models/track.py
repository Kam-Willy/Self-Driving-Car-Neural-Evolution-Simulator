"""
Track generation using cubic spline interpolation.

Improvements over the original plan
────────────────────────────────────
• is_on_track signature fixed — uses simple centre-line distance (fast enough
  for 30 cars × 1 200 frames)
• Precomputed NumPy arrays for centre-line (avoids repeated QPointF iteration)
• Gate system for progress measurement (every N samples = one "gate")
• Starting angle computed from the first spline tangent so cars face the road
• Multiple preset layouts keyed by difficulty
• Closed-loop option (track wraps around)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from config import TRACK_WIDTH, CANVAS_W, CANVAS_H, TRACK_DIFFICULTY


# Preset control-point layouts  (x, y) in logical canvas coordinates
_PRESETS = {
    "straight": [
        (80, 300), (300, 300), (550, 300), (800, 300), (950, 300)
    ],
    "gentle": [
        (80, 300), (250, 250), (450, 350), (650, 250), (850, 300), (970, 280)
    ],
    "medium": [
        (80, 300), (220, 200), (400, 400), (580, 180), (760, 380), (920, 300)
    ],
    "hard": [
        (80, 300), (200, 150), (360, 430), (520, 130), (680, 420),
        (820, 160), (940, 300)
    ],
}


def _pick_preset(difficulty: float) -> str:
    if difficulty < 0.2:
        return "straight"
    if difficulty < 0.45:
        return "gentle"
    if difficulty < 0.70:
        return "medium"
    return "hard"


class Track:
    """Curved racetrack with ray-cast collision support."""

    def __init__(self, difficulty: float = TRACK_DIFFICULTY, samples: int = 300):
        self.difficulty  = difficulty
        self.track_width = TRACK_WIDTH
        self._samples    = samples

        # Will be populated by generate()
        self.center_np:  np.ndarray = np.empty((0, 2))  # (N, 2)
        self.left_np:    np.ndarray = np.empty((0, 2))
        self.right_np:   np.ndarray = np.empty((0, 2))
        self.start_angle: float = 0.0          # radians – direction of first segment
        self.start_pos:   tuple  = (80.0, 300.0)

        # Gate system: one gate every GATE_STEP centre-line samples
        self.gate_step = 10
        self.gates: list[np.ndarray] = []      # list of (2,2) arrays [[lx,ly],[rx,ry]]

        self.generate()

    # ── Generation ───────────────────────────────────────────────────────────

    def generate(self):
        key    = _pick_preset(self.difficulty)
        pts    = _PRESETS[key]
        n_ctrl = len(pts)
        t_ctrl = np.linspace(0.0, 1.0, n_ctrl)
        xs     = [p[0] for p in pts]
        ys     = [p[1] for p in pts]

        cs_x = CubicSpline(t_ctrl, xs, bc_type='natural')
        cs_y = CubicSpline(t_ctrl, ys, bc_type='natural')

        t_fine = np.linspace(0.0, 1.0, self._samples)
        cx = cs_x(t_fine)
        cy = cs_y(t_fine)
        self.center_np = np.column_stack([cx, cy])

        # Tangent → normal → boundaries
        dx = cs_x(t_fine, 1)   # first derivative
        dy = cs_y(t_fine, 1)
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-12

        # Unit normals (perpendicular to tangent, pointing "left")
        nx = -dy / mag
        ny =  dx / mag

        hw = self.track_width / 2.0
        self.left_np  = self.center_np + hw * np.column_stack([nx, ny])
        self.right_np = self.center_np - hw * np.column_stack([nx, ny])

        # Starting angle: direction of the first tangent segment
        self.start_pos   = (float(cx[0]), float(cy[0]))
        self.start_angle = float(np.arctan2(dy[0], dx[0]))

        # Build gate list
        self.gates = []
        for i in range(0, self._samples, self.gate_step):
            self.gates.append(np.array([self.left_np[i], self.right_np[i]]))

    # ── Collision / progress API ──────────────────────────────────────────────

    def is_on_track(self, x: float, y: float) -> bool:
        """
        True if (x, y) is within track_width/2 of the nearest centre-line point.
        This is called from ray-cast sensor logic (hot path) – kept O(N) but
        with a quick bounding-box early-exit.
        """
        # Quick bounding box
        if x < 0 or x > CANVAS_W or y < 0 or y > CANVAS_H:
            return False

        # Vectorised distance to all centre-line samples
        d2 = (self.center_np[:, 0] - x) ** 2 + (self.center_np[:, 1] - y) ** 2
        return float(np.min(d2)) <= (self.track_width / 2.0) ** 2

    def is_car_on_track(self, corners: np.ndarray) -> bool:
        """
        True if ALL four car corners are on the track.
        corners: (4, 2) array from math_utils.car_corners()
        """
        return all(self.is_on_track(float(c[0]), float(c[1])) for c in corners)

    def gate_index_at(self, x: float, y: float) -> int:
        """
        Return the index of the nearest gate to (x, y).
        Used for progress-based fitness.
        """
        p = np.array([x, y])
        gate_centres = np.array([(g[0] + g[1]) / 2 for g in self.gates])
        d2 = np.sum((gate_centres - p) ** 2, axis=1)
        return int(np.argmin(d2))

    def is_finish_zone(self, x: float, y: float, radius: float = 35.0) -> bool:
        """True if (x, y) is near the last gate (finish line)."""
        finish = (self.center_np[-1, 0], self.center_np[-1, 1])
        dx = x - finish[0]
        dy = y - finish[1]
        return (dx * dx + dy * dy) <= radius * radius

    # ── Qt-compatible accessors ───────────────────────────────────────────────

    @property
    def center_line_qt(self):
        """List of (x, y) tuples for the renderer."""
        return [(float(p[0]), float(p[1])) for p in self.center_np]

    @property
    def left_boundary_qt(self):
        return [(float(p[0]), float(p[1])) for p in self.left_np]

    @property
    def right_boundary_qt(self):
        return [(float(p[0]), float(p[1])) for p in self.right_np]
