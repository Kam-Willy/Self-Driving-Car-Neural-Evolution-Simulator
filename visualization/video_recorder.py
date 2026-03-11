"""
VideoRecorder
=============
Captures every rendered frame of the simulation (from generation 0 through
to the first victorious car) and writes them into an MP4 file using OpenCV.

How it works
────────────
1.  MainWindow calls recorder.capture_frame(qt_widget) after every render.
2.  The widget is rendered off-screen to a QImage, converted to a NumPy
    RGB array, resized to VIDEO_WIDTH × VIDEO_HEIGHT, and written to the
    cv2.VideoWriter.
3.  When a car finishes (victory), MainWindow calls recorder.stop() which
    flushes and closes the file.

Requirements
────────────
    pip install opencv-python

The module degrades gracefully if cv2 is not installed: recording is simply
skipped and a warning is printed once.
"""

import os
import time
import numpy as np

from config import (
    ENABLE_RECORDING,
    VIDEO_OUTPUT_PATH,
    VIDEO_FPS,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_CODEC,
)

# Try to import cv2; fail softly
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[VideoRecorder] WARNING: opencv-python not installed. "
          "Recording disabled.  Run:  pip install opencv-python")


class VideoRecorder:
    """
    Thread-safe (single-threaded) video recorder for the simulation.

    Usage
    ─────
        recorder = VideoRecorder()
        recorder.start()                         # open the video file
        recorder.capture_frame(some_qt_widget)   # call each render tick
        recorder.stop()                          # flush & close
    """

    def __init__(self):
        self._writer   = None
        self._active   = False
        self._frame_count = 0
        self._start_time  = None
        self._output_path = VIDEO_OUTPUT_PATH
        self._enabled     = ENABLE_RECORDING and _CV2_AVAILABLE

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, output_path: str | None = None):
        """Open the VideoWriter and begin recording."""
        if not self._enabled:
            return

        if self._active:
            self.stop()

        if output_path:
            self._output_path = output_path

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(
            self._output_path,
            fourcc,
            float(VIDEO_FPS),
            (VIDEO_WIDTH, VIDEO_HEIGHT),
        )

        if not self._writer.isOpened():
            print(f"[VideoRecorder] ERROR: Could not open '{self._output_path}' "
                  f"for writing. Check codec ({VIDEO_CODEC}) and permissions.")
            self._writer  = None
            self._active  = False
            return

        self._active      = True
        self._frame_count = 0
        self._start_time  = time.time()
        print(f"[VideoRecorder] Recording started → {self._output_path}")

    def capture_frame(self, qt_widget):
        """
        Grab the current rendered content of *qt_widget* and write one frame.

        Parameters
        ----------
        qt_widget : any QWidget (typically the SimCanvas FigureCanvas)
        """
        if not self._active or self._writer is None:
            return

        try:
            frame = self._widget_to_frame(qt_widget)
            if frame is not None:
                self._writer.write(frame)
                self._frame_count += 1
        except Exception as e:
            print(f"[VideoRecorder] Frame capture error: {e}")

    def stop(self):
        """Flush the video writer and close the file."""
        if not self._active:
            return

        self._active = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        elapsed = time.time() - self._start_time if self._start_time else 0
        size_mb = self._file_size_mb()
        print(f"[VideoRecorder] Recording stopped. "
              f"{self._frame_count} frames, {elapsed:.1f}s wall-clock, "
              f"{size_mb:.1f} MB → {self._output_path}")

    def is_active(self) -> bool:
        return self._active

    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def output_path(self) -> str:
        return self._output_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _widget_to_frame(self, qt_widget) -> np.ndarray | None:
        """
        Render *qt_widget* to a QImage, convert to BGR NumPy array,
        and resize to (VIDEO_WIDTH, VIDEO_HEIGHT).
        """
        from PyQt5.QtGui import QImage
        from PyQt5.QtCore import QSize

        # Grab the widget's pixel buffer
        pixmap = qt_widget.grab()          # QPixmap of the widget
        image  = pixmap.toImage()          # QImage

        # Convert to ARGB32 for predictable byte layout
        image = image.convertToFormat(QImage.Format_RGB888)

        w = image.width()
        h = image.height()

        if w == 0 or h == 0:
            return None

        # Pointer to raw bytes
        ptr    = image.bits()
        ptr.setsize(h * w * 3)            # RGB888 = 3 bytes per pixel
        arr    = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))

        # RGB → BGR for OpenCV
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Resize to target video dimensions
        if (w, h) != (VIDEO_WIDTH, VIDEO_HEIGHT):
            bgr = cv2.resize(bgr, (VIDEO_WIDTH, VIDEO_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        return bgr

    def _file_size_mb(self) -> float:
        try:
            return os.path.getsize(self._output_path) / (1024 * 1024)
        except OSError:
            return 0.0
