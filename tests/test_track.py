"""Unit tests for Track (unittest compatible)."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.track import Track

class TestTrack(unittest.TestCase):

    def test_boundaries_populated(self):
        for diff in [0.1, 0.4, 0.7, 0.95]:
            t = Track(difficulty=diff)
            self.assertGreater(len(t.center_np), 0)
            self.assertGreater(len(t.left_np),   0)
            self.assertGreater(len(t.right_np),  0)

    def test_start_on_track(self):
        t = Track()
        sx, sy = t.start_pos
        self.assertTrue(t.is_on_track(sx, sy))

    def test_offscreen_not_on_track(self):
        t = Track()
        self.assertFalse(t.is_on_track(-500, -500))
        self.assertFalse(t.is_on_track(5000, 5000))

    def test_gates_exist(self):
        t = Track()
        self.assertGreater(len(t.gates), 0)

    def test_gate_index_at_start(self):
        t = Track()
        sx, sy = t.start_pos
        self.assertLess(t.gate_index_at(sx, sy), 5)

    def test_finish_zone(self):
        t = Track()
        ex, ey = float(t.center_np[-1, 0]), float(t.center_np[-1, 1])
        self.assertTrue(t.is_finish_zone(ex, ey))

if __name__ == '__main__':
    unittest.main()
