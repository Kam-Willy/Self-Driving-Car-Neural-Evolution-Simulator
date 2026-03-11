"""Unit tests for Genome (unittest compatible, pytest optional)."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from evolution.genome import Genome

class TestGenome(unittest.TestCase):

    def test_mutation_changes_genes(self):
        g = Genome()
        original = g.genes.copy()
        g.mutate(rate=1.0, strength=0.5)
        self.assertFalse(np.allclose(g.genes, original))

    def test_mutation_brain_consistency(self):
        g = Genome()
        g.mutate(rate=1.0, strength=0.5)
        np.testing.assert_array_almost_equal(
            g.brain.get_weights_flattened(), g.genes)

    def test_crossover_length(self):
        g1, g2 = Genome(), Genome()
        child = g1.crossover(g2)
        self.assertEqual(len(child.genes), len(g1.genes))

    def test_crossover_size_mismatch(self):
        g_big   = Genome(input_size=10)
        g_small = Genome(input_size=8)
        with self.assertRaises(ValueError):
            g_big.crossover(g_small)

    def test_clone_independence(self):
        g = Genome()
        c = g.clone()
        c.genes[:] = 999.0
        self.assertFalse(np.allclose(g.genes, c.genes))

if __name__ == '__main__':
    unittest.main()
