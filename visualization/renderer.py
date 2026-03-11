"""
Main PyQt5 application window with Matplotlib canvas.

New in this version
────────────────────
• cv2 VideoRecorder integration – records from gen-0 to victory automatically
• "⏺ Recording" / "⏹ Stop Rec" toggle button in sidebar
• Recording status indicator (red dot when active)
• Alive-car counter updated every physics tick (not just end-of-generation)
  shows:  "🚗 Alive: 18 / 30  (gen 3)"
• On victory: confetti rendered into video for 3 s before file is closed
• Output path shown in sidebar after recording stops
"""

import sys
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QApplication, QSizePolicy,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui  import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure  import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines   import Line2D

from config import (
    WINDOW_TITLE, WINDOW_W, WINDOW_H, CANVAS_DPI,
    CANVAS_W, CANVAS_H, FRAME_INTERVAL_MS, MAX_STEPS_PER_GEN,
    POPULATION_SIZE, SHOW_SENSORS, SHOW_FITNESS_PLOT,
    GENERATION_LIMIT, ENABLE_RECORDING, VIDEO_OUTPUT_PATH,
)
from visualization.animation      import ConfettiSystem
from visualization.video_recorder import VideoRecorder


# ── Colour helper ────────────────────────────────────────────────────────────

def _qcolor_to_rgba(qcolor, alpha=0.85):
    r, g, b, _ = qcolor.getRgb()
    return (r / 255, g / 255, b / 255, alpha)


# ── Simulation canvas ────────────────────────────────────────────────────────

class SimCanvas(FigureCanvas):
    """Matplotlib figure embedded in PyQt5 – draws track, cars, sensors."""

    def __init__(self, parent=None):
        self.fig = Figure(
            figsize=(CANVAS_W / CANVAS_DPI, CANVAS_H / CANVAS_DPI),
            dpi=CANVAS_DPI,
            facecolor='#1a1a2e',
        )
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111)
        self._setup_ax()

        self._car_patches:  list = []
        self._sensor_lines: list = []
        self._confetti = ConfettiSystem()
        self._overlay_text = []

    def _setup_ax(self):
        ax = self.ax
        ax.set_xlim(0, CANVAS_W)
        ax.set_ylim(0, CANVAS_H)
        ax.set_aspect('equal')
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#333355')

    # ── Track ─────────────────────────────────────────────────────────────────

    def draw_track(self, track):
        self.ax.clear()
        self._setup_ax()
        self._overlay_text = []

        left  = track.left_boundary_qt
        right = track.right_boundary_qt
        cline = track.center_line_qt

        n = min(len(left), len(right))
        if n < 2:
            self.draw()
            return

        lx = [p[0] for p in left[:n]]
        ly = [p[1] for p in left[:n]]
        rx = [p[0] for p in right[:n]]
        ry = [p[1] for p in right[:n]]
        cx = [p[0] for p in cline]
        cy = [p[1] for p in cline]

        from matplotlib.patches import Polygon as MplPoly
        road_poly = MplPoly(
            list(zip(lx + rx[::-1], ly + ry[::-1])),
            closed=True, facecolor='#2d2d44', edgecolor='none', zorder=1,
        )
        self.ax.add_patch(road_poly)

        self.ax.plot(lx, ly, color='#ffffff', lw=1.5, zorder=2)
        self.ax.plot(rx, ry, color='#ffffff', lw=1.5, zorder=2)
        self.ax.plot(cx, cy, color='#ffff00', lw=0.8,
                     linestyle='--', alpha=0.4, zorder=2)

        if len(left) > 1 and len(right) > 1:
            self.ax.plot(
                [left[0][0], right[0][0]], [left[0][1], right[0][1]],
                color='#ff4444', lw=3, zorder=3,
            )
            self.ax.plot(
                [left[-1][0], right[-1][0]], [left[-1][1], right[-1][1]],
                color='#44ff44', lw=3, zorder=3,
            )

        self.draw()

    # ── Cars ──────────────────────────────────────────────────────────────────

    def draw_cars(self, cars, generation=0, show_sensors=SHOW_SENSORS):
        for p in self._car_patches:
            try: p.remove()
            except: pass
        self._car_patches.clear()

        for ln in self._sensor_lines:
            try: ln.remove()
            except: pass
        self._sensor_lines.clear()

        for t in self._overlay_text:
            try: t.remove()
            except: pass
        self._overlay_text = []

        # Identify best (leading) alive car
        best_car  = None
        best_prog = -1
        alive_count = 0
        total_count = len(cars)

        for car in cars:
            if car.alive and not car.finished:
                alive_count += 1
                if car.gate_progress > best_prog:
                    best_prog = car.gate_progress
                    best_car  = car

        for car in cars:
            if not (car.alive or car.finished):
                continue

            corners  = car.get_corners()
            vertices = [(float(c[0]), float(c[1])) for c in corners]

            try:
                rgba = _qcolor_to_rgba(car.color)
            except Exception:
                r, g, b = (car.color if isinstance(car.color, tuple)
                           else (100, 180, 220))
                rgba = (r / 255, g / 255, b / 255, 0.85)

            patch = MplPolygon(
                vertices, closed=True,
                facecolor=rgba, edgecolor='white', linewidth=0.8, zorder=5,
            )
            self.ax.add_patch(patch)
            self._car_patches.append(patch)

            if show_sensors and car is best_car:
                for (ex, ey) in car.get_sensor_endpoints():
                    line = Line2D(
                        [car.x, ex], [car.y, ey],
                        color='#00ffcc', lw=0.6, alpha=0.5, zorder=4,
                    )
                    self.ax.add_line(line)
                    self._sensor_lines.append(line)

        # ── HUD burned into every frame (visible in video) ────────────────
        hud_gen = self.ax.text(
            10, CANVAS_H - 10,
            f"Gen {generation}",
            color='#ffffff', fontsize=11, fontweight='bold',
            va='top', ha='left', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#000000bb', edgecolor='none'),
        )
        alive_color = '#00ff88' if alive_count > 0 else '#ff6644'
        hud_alive = self.ax.text(
            10, CANVAS_H - 40,
            f"Cars alive: {alive_count} / {total_count}",
            color=alive_color, fontsize=10, fontweight='bold',
            va='top', ha='left', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#000000bb', edgecolor='none'),
        )
        self._overlay_text = [hud_gen, hud_alive]

        if self._confetti.active:
            self._confetti.update()
            self._confetti.draw(self.ax)

        self.draw()

    def start_confetti(self):
        self._confetti.reset()

    def stop_confetti(self):
        self._confetti.stop()


# ── Fitness chart ─────────────────────────────────────────────────────────────

class FitnessCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3, 2.5), dpi=80, facecolor='#1a1a2e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._init_ax()

    def _init_ax(self):
        self.ax.set_facecolor('#0f0f23')
        self.ax.set_title('Fitness History', color='#cccccc', fontsize=8)
        self.ax.set_xlabel('Generation', color='#aaaaaa', fontsize=7)
        self.ax.set_ylabel('Fitness',    color='#aaaaaa', fontsize=7)
        self.ax.tick_params(colors='#888888', labelsize=6)
        for spine in self.ax.spines.values():
            spine.set_color('#333355')

    def update_chart(self, best_hist, avg_hist):
        self.ax.clear()
        self._init_ax()
        if best_hist:
            gens = list(range(1, len(best_hist) + 1))
            self.ax.plot(gens, best_hist, color='#00ff88',
                         lw=1.5, label='Best', zorder=3)
            self.ax.plot(gens, avg_hist,  color='#ffaa00',
                         lw=1.0, linestyle='--', label='Avg', alpha=0.8)
            self.ax.fill_between(gens, avg_hist, best_hist,
                                  alpha=0.15, color='#00ff88')
            self.ax.legend(fontsize=6, facecolor='#1a1a2e',
                           labelcolor='#cccccc', loc='upper left')
        self.draw()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(80, 60, WINDOW_W, WINDOW_H)
        self.setStyleSheet("background-color: #12122a; color: #dddddd;")

        self._recorder = VideoRecorder()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(218)
        sidebar.setStyleSheet("background-color: #1a1a32;")
        sl = QVBoxLayout(sidebar)
        sl.setSpacing(6)

        title = QLabel("🧬 Car Evolution")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        sl.addWidget(title)

        _btn = ("QPushButton { background:#2a2a4a; border:1px solid #4444aa;"
                " border-radius:4px; padding:5px; color:#dddddd; }"
                "QPushButton:hover { background:#3a3a6a; }"
                "QPushButton:disabled { color:#555566; }")

        self.btn_start = QPushButton("▶  Start / New Track")
        self.btn_start.setStyleSheet(_btn)
        self.btn_start.clicked.connect(self.start_evolution)
        sl.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setStyleSheet(_btn)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        sl.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("⏹  Reset")
        self.btn_reset.setStyleSheet(_btn)
        self.btn_reset.clicked.connect(self.reset_simulation)
        sl.addWidget(self.btn_reset)

        # Divider
        for _ in range(1):
            sep = QLabel("─" * 26)
            sep.setStyleSheet("color:#333355; font-size:8px;")
            sep.setAlignment(Qt.AlignCenter)
            sl.addWidget(sep)

        rec_hdr = QLabel("📹 Video Recording")
        rec_hdr.setStyleSheet("color:#aaaacc; font-size:8px; font-weight:bold;")
        rec_hdr.setAlignment(Qt.AlignCenter)
        sl.addWidget(rec_hdr)

        self.btn_record = QPushButton("⏺  Start Recording")
        self.btn_record.setStyleSheet(_btn)
        self.btn_record.clicked.connect(self.toggle_recording)
        if not self._recorder.is_enabled():
            self.btn_record.setEnabled(False)
            self.btn_record.setToolTip(
                "opencv-python not installed.\nRun: pip install opencv-python")
        sl.addWidget(self.btn_record)

        self.lbl_rec_status = QLabel("● Not recording")
        self.lbl_rec_status.setStyleSheet("color:#666688; font-size:8px;")
        self.lbl_rec_status.setAlignment(Qt.AlignCenter)
        self.lbl_rec_status.setWordWrap(True)
        sl.addWidget(self.lbl_rec_status)

        sep2 = QLabel("─" * 26)
        sep2.setStyleSheet("color:#333355; font-size:8px;")
        sep2.setAlignment(Qt.AlignCenter)
        sl.addWidget(sep2)

        # Speed slider
        sl.addWidget(self._mk_label("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 8)
        self.speed_slider.setValue(1)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(self._on_speed)
        sl.addWidget(self.speed_slider)
        self.lbl_speed = QLabel("1×")
        self.lbl_speed.setAlignment(Qt.AlignCenter)
        self.lbl_speed.setStyleSheet("color:#aaaaaa; font-size:9px;")
        sl.addWidget(self.lbl_speed)

        # Stats
        self.lbl_gen       = self._mk_label("Generation: –")
        self.lbl_alive     = QLabel("🚗 Alive: –")
        self.lbl_alive.setStyleSheet(
            "color:#00ff88; font-size:10px; font-weight:bold;")
        self.lbl_alive.setWordWrap(True)
        self.lbl_best_gen  = self._mk_label("Gen best: –")
        self.lbl_best_ever = self._mk_label("All-time best: –")
        self.lbl_mut       = self._mk_label("Mut strength: –")
        self.lbl_status    = self._mk_label("Status: Ready")
        self.lbl_status.setStyleSheet("color:#ffcc44; font-size:9px;")

        for w in [self.lbl_gen, self.lbl_alive, self.lbl_best_gen,
                  self.lbl_best_ever, self.lbl_mut, self.lbl_status]:
            sl.addWidget(w)

        if SHOW_FITNESS_PLOT:
            self.fitness_canvas = FitnessCanvas(sidebar)
            sl.addWidget(self.fitness_canvas)

        sl.addStretch()
        root.addWidget(sidebar)

        self.sim_canvas = SimCanvas(central)
        root.addWidget(self.sim_canvas, stretch=1)

        # Sim state
        self.population      = None
        self.cars:     list  = []
        self.track           = None
        self._generation     = 0
        self._running        = False
        self._paused         = False
        self._step_count     = 0
        self._steps_per_tick = 1

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mk_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#aaaacc; font-size:9px;")
        lbl.setWordWrap(True)
        return lbl

    def _update_alive_label(self):
        alive    = sum(1 for c in self.cars if c.alive and not c.finished)
        total    = len(self.cars)
        finished = sum(1 for c in self.cars if c.finished)
        color    = '#00ff88' if alive > 0 else '#ff6644'
        self.lbl_alive.setStyleSheet(
            f"color:{color}; font-size:10px; font-weight:bold;")
        extra = f"   ✔ {finished} done" if finished else ""
        self.lbl_alive.setText(
            f"🚗 Alive: {alive} / {total}{extra}\n"
            f"   (generation {self._generation})")

    # ── Controls ─────────────────────────────────────────────────────────────

    def start_evolution(self):
        self.timer.stop()
        from models.track         import Track
        from evolution.population import Population

        self.track       = Track()
        self.population  = Population(size=POPULATION_SIZE, track=self.track)
        self.cars        = self.population.create_cars()
        self._generation = 0
        self._step_count = 0

        self.sim_canvas.draw_track(self.track)
        self.sim_canvas.stop_confetti()

        self._running = True
        self._paused  = False
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("⏸  Pause")

        # Auto-start recording on simulation start
        if ENABLE_RECORDING and self._recorder.is_enabled() \
                and not self._recorder.is_active():
            self._start_recording()

        self.lbl_status.setText("Status: Evolving…")
        self._update_alive_label()
        self.timer.start(FRAME_INTERVAL_MS)

    def toggle_pause(self):
        if self._paused:
            self._paused = False
            self.btn_pause.setText("⏸  Pause")
            self.timer.start(FRAME_INTERVAL_MS)
            self.lbl_status.setText("Status: Evolving…")
        else:
            self._paused = True
            self.btn_pause.setText("▶  Resume")
            self.timer.stop()
            self.lbl_status.setText("Status: Paused")

    def reset_simulation(self):
        self.timer.stop()
        if self._recorder.is_active():
            self._stop_recording()
        self._running = self._paused = False
        self.population = None
        self.cars = []
        self._generation = 0
        self.sim_canvas.stop_confetti()
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause")
        self.lbl_status.setText("Status: Ready")
        self.lbl_gen.setText("Generation: –")
        self.lbl_alive.setText("🚗 Alive: –")
        self.lbl_best_gen.setText("Gen best: –")
        self.lbl_best_ever.setText("All-time best: –")
        if SHOW_FITNESS_PLOT:
            self.fitness_canvas.update_chart([], [])

    def toggle_recording(self):
        if self._recorder.is_active():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._recorder.start()
        if self._recorder.is_active():
            self.btn_record.setText("⏹  Stop Recording")
            self.lbl_rec_status.setStyleSheet(
                "color:#ff4444; font-size:8px; font-weight:bold;")
            self.lbl_rec_status.setText(
                f"● REC → {self._recorder.output_path}")

    def _stop_recording(self):
        self._recorder.stop()
        self.btn_record.setText("⏺  Start Recording")
        self.lbl_rec_status.setStyleSheet("color:#44cc66; font-size:8px;")
        self.lbl_rec_status.setText(
            f"✔ Saved: {self._recorder.output_path}\n"
            f"  ({self._recorder.frame_count} frames)")

    def _on_speed(self, value):
        self._steps_per_tick = value
        self.lbl_speed.setText(f"{value}×")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _tick(self):
        if not self._running or self._paused:
            return
        for _ in range(self._steps_per_tick):
            self._physics_step()
            if self._check_generation_done():
                return
        self._render()

    def _physics_step(self):
        self._step_count += 1
        for car in self.cars:
            if car.alive and not car.finished:
                inputs             = car.get_nn_inputs(self.track)
                steering, throttle = car.brain.forward(inputs)
                car.update(steering, throttle, self.track)
        # Live alive counter every physics frame
        self._update_alive_label()

    def _check_generation_done(self):
        alive_count    = sum(1 for c in self.cars if c.alive and not c.finished)
        finished_count = sum(1 for c in self.cars if c.finished)
        timeout        = self._step_count >= MAX_STEPS_PER_GEN
        done           = (alive_count == 0) or timeout

        if not done:
            return False

        self.timer.stop()
        self.population.evaluate_fitness(self.cars)
        s = self.population.stats()

        if finished_count > 0:
            self.lbl_status.setText(
                f"🏆 VICTORY! Car finished gen {self._generation}!")
            self.sim_canvas.start_confetti()
            self._render()
            # Record confetti for 3 s then stop
            QTimer.singleShot(3000, self._on_victory_done)
            return True

        self._generation = self.population.next_generation()
        self.cars        = self.population.create_cars()
        self._step_count = 0

        self.lbl_gen.setText(f"Generation: {self._generation}")
        self.lbl_best_gen.setText(f"Gen best: {s['gen_best']:.0f}")
        self.lbl_best_ever.setText(f"All-time best: {s['best_fitness']:.0f}")
        self.lbl_mut.setText(f"Mut strength: {s['mut_strength']:.3f}")
        self._update_alive_label()

        if SHOW_FITNESS_PLOT:
            self.fitness_canvas.update_chart(
                self.population.best_fitness_history,
                self.population.avg_fitness_history,
            )

        if GENERATION_LIMIT > 0 and self._generation >= GENERATION_LIMIT:
            self.lbl_status.setText(
                f"Status: Limit reached ({self._generation} gens)")
            if self._recorder.is_active():
                self._stop_recording()
            return True

        self.timer.start(FRAME_INTERVAL_MS)
        return True

    def _on_victory_done(self):
        if self._recorder.is_active():
            self._stop_recording()
        self.sim_canvas.stop_confetti()
        if self.population and self._running:
            self._generation = self.population.next_generation()
            self.cars        = self.population.create_cars()
            self._step_count = 0
            self.lbl_status.setText(
                f"Status: Evolving… (gen {self._generation})")
            self.timer.start(FRAME_INTERVAL_MS)

    def _render(self):
        self.sim_canvas.draw_cars(
            self.cars,
            generation=self._generation,
            show_sensors=SHOW_SENSORS,
        )
        if self._recorder.is_active():
            self._recorder.capture_frame(self.sim_canvas)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_app():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
