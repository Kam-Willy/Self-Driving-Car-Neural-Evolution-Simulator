"""
Car physics model (simplified bicycle kinematics).

Improvements over the original plan
────────────────────────────────────
• Sensor ray-cast now calls track.is_on_track(x, y)  ← was missing 3 args bug
• Stall detection: cars travelling too slowly for too long are eliminated
• Progress tracking via gate index (not raw pixel distance)
• Speed fed back into NN inputs so the car can regulate velocity
• Angle-to-next-gate fed as input (directional hint)
• Inertia model: steering effectiveness scales with speed
• All corner collision uses math_utils.car_corners (avoids duplicated trig)
• QColor dependency is optional – falls back to tuple RGB when PyQt5 absent
"""

import numpy as np
from config import (
    CAR_LENGTH, CAR_WIDTH, CAR_MAX_VELOCITY, CAR_ACCELERATION,
    CAR_BRAKE_FACTOR, CAR_MAX_TURN_RATE,
    SENSOR_COUNT, SENSOR_ANGLES_DEG, SENSOR_MAX_DIST, SENSOR_STEP,
    MIN_SPEED_THRESHOLD, STALL_FRAMES, NN_INPUT_SIZE,
)
from utils.math_utils import car_corners, ray_cast_distance, wrap_angle

try:
    from PyQt5.QtGui import QColor, QPolygonF
    from PyQt5.QtCore import QPointF
    _QT = True
except ImportError:
    _QT = False


class Car:
    """
    A single agent in the simulation.

    Attributes
    ──────────
    x, y        : world position (pixels)
    angle       : heading in radians (0 = rightward)
    velocity    : current speed (pixels / frame)
    alive       : False when crashed or stalled
    finished    : True when the finish zone is reached
    fitness     : set by Population.evaluate_fitness() after a generation
    """

    def __init__(self, x: float = 100.0, y: float = 300.0, angle: float = 0.0):
        # ── Pose ──────────────────────────────────────────────────────────
        self.x      = float(x)
        self.y      = float(y)
        self.angle  = float(angle)

        # ── Dynamics ──────────────────────────────────────────────────────
        self.velocity = 0.0
        self._stall_counter = 0

        # ── Dimensions ────────────────────────────────────────────────────
        self.half_len = CAR_LENGTH / 2.0
        self.half_wid = CAR_WIDTH  / 2.0

        # ── Sensors ───────────────────────────────────────────────────────
        # Normalised distances [0, 1]; 1.0 = nothing within SENSOR_MAX_DIST
        self.sensor_distances: list[float] = [1.0] * SENSOR_COUNT
        # Raw pixel distances (for visualisation)
        self.sensor_raw_px:   list[float] = [float(SENSOR_MAX_DIST)] * SENSOR_COUNT

        # ── State ─────────────────────────────────────────────────────────
        self.alive           = True
        self.finished        = False
        self.distance_traveled = 0.0
        self.steps           = 0
        self.gate_progress   = 0          # highest gate index reached
        self.speed_history: list[float] = []   # for avg-speed bonus

        # ── Assigned externally by Population ─────────────────────────────
        self.brain   = None   # NeuralNetwork instance
        self.genome  = None   # Genome instance
        self.color   = (100, 180, 220)  # (r, g, b) tuple; overwritten by Population

        # ── Cached for rendering ──────────────────────────────────────────
        self._corners: np.ndarray = np.zeros((4, 2))

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, steering: float, throttle: float, track) -> None:
        """
        Advance physics by one frame.

        Parameters
        ----------
        steering : float ∈ [-1, 1]   – negative = left turn
        throttle : float ∈ [ 0, 1]   – fraction of max speed
        track    : Track
        """
        if not self.alive or self.finished:
            return

        self.steps += 1

        # ── Velocity update (simple lerp + braking) ───────────────────────
        target_v = throttle * CAR_MAX_VELOCITY
        if target_v < self.velocity:
            self.velocity = max(target_v, self.velocity * CAR_BRAKE_FACTOR)
        else:
            self.velocity += (target_v - self.velocity) * CAR_ACCELERATION

        self.velocity = max(0.0, min(CAR_MAX_VELOCITY, self.velocity))

        # ── Stall detection ───────────────────────────────────────────────
        if self.velocity < MIN_SPEED_THRESHOLD:
            self._stall_counter += 1
            if self._stall_counter > STALL_FRAMES:
                self.alive = False
                return
        else:
            self._stall_counter = 0

        # ── Steering (effectiveness scales with speed) ────────────────────
        speed_frac = self.velocity / CAR_MAX_VELOCITY
        turn = steering * CAR_MAX_TURN_RATE * (0.3 + 0.7 * speed_frac)
        self.angle = wrap_angle(self.angle + turn)

        # ── Position update ───────────────────────────────────────────────
        dx = self.velocity * np.cos(self.angle)
        dy = self.velocity * np.sin(self.angle)
        new_x = self.x + dx
        new_y = self.y + dy

        # ── Collision check ───────────────────────────────────────────────
        corners = car_corners(new_x, new_y, self.angle, self.half_len, self.half_wid)
        if not track.is_car_on_track(corners):
            self.alive = False
            return

        # ── Accept move ───────────────────────────────────────────────────
        self.x, self.y = new_x, new_y
        self._corners  = corners
        moved = np.sqrt(dx * dx + dy * dy)
        self.distance_traveled += moved
        self.speed_history.append(self.velocity)

        # ── Progress ──────────────────────────────────────────────────────
        gate_idx = track.gate_index_at(self.x, self.y)
        if gate_idx > self.gate_progress:
            self.gate_progress = gate_idx

        # ── Finish check ──────────────────────────────────────────────────
        if track.is_finish_zone(self.x, self.y):
            self.finished = True
            self.alive    = True   # keep alive so renderer shows it

        # ── Sensor update ─────────────────────────────────────────────────
        self._update_sensors(track)

    # ── Sensor ray-casting ───────────────────────────────────────────────────

    def _update_sensors(self, track) -> None:
        """Cast SENSOR_COUNT rays and store normalised distances."""
        for i, deg in enumerate(SENSOR_ANGLES_DEG):
            direction = self.angle + np.radians(deg)

            def _hit(cx, cy, _track=track):
                return not _track.is_on_track(cx, cy)

            norm_dist = ray_cast_distance(
                self.x, self.y,
                direction,
                SENSOR_STEP,
                SENSOR_MAX_DIST,
                _hit,
            )
            self.sensor_distances[i] = norm_dist
            self.sensor_raw_px[i]    = norm_dist * SENSOR_MAX_DIST

    # ── Neural network inputs ─────────────────────────────────────────────────

    def get_nn_inputs(self, track=None) -> np.ndarray:
        """
        Build the (NN_INPUT_SIZE,) input vector for the brain.

        Slots
        ─────
        0–7  : sensor distances (normalised)
        8    : normalised speed
        9    : angle difference to next gate direction (normalised to [-1, 1])
        """
        inputs = list(self.sensor_distances)           # 8 values

        # Normalised speed
        inputs.append(self.velocity / CAR_MAX_VELOCITY)

        # Angle hint toward next gate
        if track is not None and len(track.gates) > 0:
            next_idx  = min(self.gate_progress + 1, len(track.gates) - 1)
            gate_ctr  = (track.gates[next_idx][0] + track.gates[next_idx][1]) / 2.0
            to_gate   = gate_ctr - np.array([self.x, self.y])
            target_ang = np.arctan2(float(to_gate[1]), float(to_gate[0]))
            diff = wrap_angle(target_ang - self.angle)
            inputs.append(diff / np.pi)               # normalise to [-1, 1]
        else:
            inputs.append(0.0)

        arr = np.array(inputs[:NN_INPUT_SIZE], dtype=np.float64)
        # Pad if needed (should not happen with correct config)
        if len(arr) < NN_INPUT_SIZE:
            arr = np.concatenate([arr, np.zeros(NN_INPUT_SIZE - len(arr))])
        return arr

    # ── Rendering helpers ─────────────────────────────────────────────────────

    def get_corners(self) -> np.ndarray:
        """Return (4, 2) world-space corners (cached from last update)."""
        if self._corners.shape == (4, 2):
            return self._corners
        return car_corners(self.x, self.y, self.angle, self.half_len, self.half_wid)

    def get_sensor_endpoints(self) -> list[tuple[float, float]]:
        """Return list of (x, y) endpoints for each sensor ray (for visualisation)."""
        endpoints = []
        for i, deg in enumerate(SENSOR_ANGLES_DEG):
            direction = self.angle + np.radians(deg)
            dist = self.sensor_raw_px[i]
            ex = self.x + dist * np.cos(direction)
            ey = self.y + dist * np.sin(direction)
            endpoints.append((ex, ey))
        return endpoints

    if _QT:
        def get_polygon_qt(self):
            """QPolygonF for PyQt5 QPainter rendering."""
            corners = self.get_corners()
            return QPolygonF([QPointF(float(c[0]), float(c[1])) for c in corners])
