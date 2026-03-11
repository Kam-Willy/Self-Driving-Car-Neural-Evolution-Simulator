"""
Genome: wraps a NeuralNetwork and exposes genetic operators.

Improvements over the original plan
────────────────────────────────────
• Crossover checks both parents have the same gene count before mixing
  (avoids silent silently-wrong child genomes)
• Mutation supports three modes: Gaussian (default), reset, and scale
• Adaptive mutation: stores generation index so Population can adjust
  strength over time
• clone() delegates to NeuralNetwork.clone() for a true deep copy
"""

import numpy as np
import random
from models.neural_net import NeuralNetwork
from config import (
    MUTATION_RATE, MUTATION_STRENGTH,
    NN_INPUT_SIZE, NN_HIDDEN_LAYERS, NN_OUTPUT_SIZE,
)


class Genome:
    """One individual in the evolutionary population."""

    def __init__(self,
                 input_size:   int  = NN_INPUT_SIZE,
                 hidden_sizes       = NN_HIDDEN_LAYERS,
                 output_size:  int  = NN_OUTPUT_SIZE):

        self.brain = NeuralNetwork(input_size, hidden_sizes, output_size)
        self.genes = self.brain.get_weights_flattened()   # float64 array

        self.fitness:    float = 0.0
        self.generation: int   = 0

    # ── Mutation ─────────────────────────────────────────────────────────────

    def mutate(self,
               rate:     float = MUTATION_RATE,
               strength: float = MUTATION_STRENGTH):
        """
        Apply per-weight mutation.

        For each weight, with probability *rate* choose:
          80% → Gaussian perturbation (exploration near current value)
          15% → Full reset from N(0, 1)  (escape local minima)
           5% → Scale by random factor   (global magnitude adjustment)
        """
        genes = self.genes.copy()
        mask  = np.random.random(len(genes)) < rate

        for idx in np.where(mask)[0]:
            r = random.random()
            if r < 0.80:
                genes[idx] += np.random.normal(0.0, strength)
            elif r < 0.95:
                genes[idx]  = np.random.normal(0.0, 1.0)
            else:
                genes[idx] *= np.random.uniform(0.5, 1.5)

        self.genes = genes
        self.brain.set_weights_from_flattened(genes)

    # ── Crossover ────────────────────────────────────────────────────────────

    def crossover(self, other: "Genome") -> "Genome":
        """
        Uniform crossover between self and other.

        Raises ValueError if the two genomes have different gene counts
        (would produce a silently wrong child in the original code).
        """
        if len(self.genes) != len(other.genes):
            raise ValueError(
                f"Genome size mismatch: {len(self.genes)} vs {len(other.genes)}"
            )

        # Uniform crossover mask
        mask        = np.random.random(len(self.genes)) < 0.5
        child_genes = np.where(mask, self.genes, other.genes)

        child       = Genome.__new__(Genome)
        child.brain = NeuralNetwork.__new__(NeuralNetwork)
        child.brain.layer_sizes = self.brain.layer_sizes[:]
        child.brain.weights     = [W.copy() for W in self.brain.weights]
        child.brain.biases      = [b.copy() for b in self.brain.biases]
        child.brain.set_weights_from_flattened(child_genes)
        child.genes      = child_genes.copy()
        child.fitness    = 0.0
        child.generation = self.generation + 1
        return child

    # ── Utility ──────────────────────────────────────────────────────────────

    def clone(self) -> "Genome":
        """Deep copy — used for elitism."""
        g             = Genome.__new__(Genome)
        g.brain       = self.brain.clone()
        g.genes       = self.genes.copy()
        g.fitness     = self.fitness
        g.generation  = self.generation
        return g

    def __repr__(self) -> str:
        return (f"Genome(fitness={self.fitness:.1f}, "
                f"gen={self.generation}, genes={len(self.genes)})")
