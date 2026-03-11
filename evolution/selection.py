"""
Selection strategies for the genetic algorithm.

Provides
────────
tournament_select  – O(k) tournament selection (default)
rank_select        – Linear rank-based selection (less pressure than fitness-proportionate)
roulette_select    – Fitness-proportionate (classic, but sensitive to scale)

All functions accept a list of Genome objects and return a single parent.
"""

import random
import numpy as np
from evolution.genome import Genome
from config import TOURNAMENT_SIZE


def tournament_select(population: list[Genome],
                      k: int = TOURNAMENT_SIZE) -> Genome:
    """
    Pick *k* individuals at random; return the one with the highest fitness.
    Efficient and robust – the default selection method.
    """
    k = min(k, len(population))
    competitors = random.sample(population, k)
    return max(competitors, key=lambda g: g.fitness)


def rank_select(population: list[Genome]) -> Genome:
    """
    Linear rank-based selection.
    Genomes are sorted by fitness; selection probability ∝ rank.
    Avoids over-dominance by very-high-fitness individuals.
    """
    n      = len(population)
    sorted_pop = sorted(population, key=lambda g: g.fitness)
    # Rank 1 (worst) … rank n (best)
    ranks  = np.arange(1, n + 1, dtype=float)
    probs  = ranks / ranks.sum()
    idx    = np.random.choice(n, p=probs)
    return sorted_pop[idx]


def roulette_select(population: list[Genome]) -> Genome:
    """
    Fitness-proportionate (roulette-wheel) selection.
    Works best when all fitness values are positive and not too spread out.
    """
    fitnesses = np.array([g.fitness for g in population], dtype=float)
    min_f     = fitnesses.min()
    if min_f < 0:
        fitnesses -= min_f          # shift to non-negative
    total     = fitnesses.sum()
    if total < 1e-12:
        return random.choice(population)
    probs = fitnesses / total
    idx   = np.random.choice(len(population), p=probs)
    return population[idx]
