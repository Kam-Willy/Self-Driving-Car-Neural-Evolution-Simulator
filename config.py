"""
Configuration parameters for the Self-Driving Car Evolution Simulator.
Centralised here so every module imports from a single source of truth.
"""

# ── Simulation ──────────────────────────────────────────────────────────────
POPULATION_SIZE      = 30          # Number of cars per generation
GENERATION_LIMIT     = 200         # Stop after this many generations (0 = unlimited)
FRAME_INTERVAL_MS    = 30          # Timer interval → ~33 fps
MAX_STEPS_PER_GEN    = 1200        # Force next generation after this many frames
                                   # (prevents immortal-but-stuck cars)
MIN_SPEED_THRESHOLD  = 0.05        # Cars slower than this for STALL_FRAMES die
STALL_FRAMES         = 120         # Frames below MIN_SPEED_THRESHOLD before culled

# ── Neural network ───────────────────────────────────────────────────────────
NN_INPUT_SIZE    = 10              # 8 distance sensors + speed + angle-to-next-gate
NN_HIDDEN_LAYERS = [24, 16]        # Two hidden layers
NN_OUTPUT_SIZE   = 2               # (steering, throttle)

# ── Genetic algorithm ────────────────────────────────────────────────────────
MUTATION_RATE      = 0.12          # Probability that any single weight is mutated
MUTATION_STRENGTH  = 0.15          # Std-dev of Gaussian noise added to a weight
ELITISM_RATIO      = 0.10          # Top fraction preserved unchanged each generation
TOURNAMENT_SIZE    = 5             # Number of competitors in tournament selection
CROSSOVER_RATE     = 0.70          # Probability of crossover vs. clone+mutate

# ── Car physics ──────────────────────────────────────────────────────────────
CAR_MAX_VELOCITY      = 6.0        # pixels per frame
CAR_ACCELERATION      = 0.12       # lerp factor toward target velocity
CAR_BRAKE_FACTOR      = 0.85       # velocity multiplier when throttle = 0
CAR_MAX_TURN_RATE     = 0.08       # radians per frame at full steering
CAR_LENGTH            = 28         # pixels
CAR_WIDTH             = 14         # pixels
SENSOR_COUNT          = 8          # number of ray-cast sensors
SENSOR_ANGLES_DEG     = [-70, -45, -20, -5, 5, 20, 45, 70]
SENSOR_MAX_DIST       = 220        # pixels
SENSOR_STEP           = 4          # ray-cast step size (pixels)

# ── Track ────────────────────────────────────────────────────────────────────
TRACK_WIDTH       = 60             # pixels (road width)
TRACK_DIFFICULTY  = 0.6            # 0.0 (easy/straight) → 1.0 (tight curves)
CANVAS_W          = 1000           # logical canvas width
CANVAS_H          = 600            # logical canvas height

# ── Fitness weights ──────────────────────────────────────────────────────────
FITNESS_DISTANCE_WEIGHT  = 1.0     # reward per pixel of progress
FITNESS_FINISH_BONUS     = 5000    # bonus for crossing the finish line
FITNESS_CRASH_PENALTY    = 0.6     # multiply fitness by this on crash
FITNESS_SPEED_BONUS      = 0.3     # fraction of avg_speed added to fitness

# ── Visualisation ────────────────────────────────────────────────────────────
WINDOW_TITLE   = "Self-Driving Car Neural Evolution"
WINDOW_W       = 1300
WINDOW_H       = 820
CANVAS_DPI     = 90
SHOW_SENSORS   = True              # draw sensor rays on the best-alive car
SHOW_FITNESS_PLOT = True           # live fitness chart in sidebar
CONFETTI_COUNT    = 80

# ── Video recording (cv2) ─────────────────────────────────────────────────────
ENABLE_RECORDING  = True           # Set False to disable video capture entirely
VIDEO_OUTPUT_PATH = "evolution_recording.mp4"  # output file name
VIDEO_FPS         = 30             # frames per second in the output video
VIDEO_WIDTH       = 1280           # output video pixel width
VIDEO_HEIGHT      = 720            # output video pixel height
# Codec: mp4v works everywhere; avc1/H264 gives smaller files if available
VIDEO_CODEC       = "mp4v"
