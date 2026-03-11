"""Unit tests for NeuralNetwork (unittest compatible, pytest optional)."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from models.neural_net import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):

    def _make(self):
        return NeuralNetwork(input_size=10, hidden_sizes=[24,16], output_size=2)

    def test_forward_output_types(self):
        net = self._make()
        s, t = net.forward(np.random.randn(10))
        self.assertIsInstance(s, float)
        self.assertIsInstance(t, float)

    def test_steering_range(self):
        net = self._make()
        for _ in range(50):
            s, _ = net.forward(np.random.randn(10))
            self.assertGreaterEqual(s, -1.0)
            self.assertLessEqual(s, 1.0)

    def test_throttle_range(self):
        net = self._make()
        for _ in range(50):
            _, t = net.forward(np.random.randn(10))
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0)

    def test_weights_round_trip(self):
        net = self._make()
        x   = np.random.randn(10)
        s1, t1 = net.forward(x)
        net2 = self._make()
        net2.set_weights_from_flattened(net.get_weights_flattened())
        s2, t2 = net2.forward(x)
        self.assertAlmostEqual(s1, s2, places=10)
        self.assertAlmostEqual(t1, t2, places=10)

    def test_clone_independence(self):
        net = self._make()
        x   = np.random.randn(10)
        s1, _ = net.forward(x)
        clone = net.clone()
        clone.weights[0] += 99.0
        s2, _ = net.forward(x)
        self.assertAlmostEqual(s1, s2, places=10)

    def test_total_params(self):
        net = self._make()
        expected = (10*24+24) + (24*16+16) + (16*2+2)
        self.assertEqual(net.total_params, expected)

if __name__ == '__main__':
    unittest.main()
