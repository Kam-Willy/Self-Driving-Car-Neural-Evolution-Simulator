"""
Population management for the genetic algorithm.

Improvements over the original plan
────────────────────────────────────
• Fitness function uses gate_progress (actual track progress), avg speed
  bonus, and finish bonus — not raw pixel distance
• Adaptive mutation: strength decreases as best fitness plateaus
• Diversity injection: if the population converges too much, a fraction of
  new individuals are generated randomly (prevents premature convergence)
• History tracking: best_fitness_history list for the live plot
• Car colour maps directly to normalised fitness (green = best, red = worst)
"""

import random
import numpy as np

from evolution.genome    import Genome
from evolution.selection import tournament_select
from models.car          import Car
from config import (
    POPULATION_SIZE, ELITISM_RATIO, CROSSOVER_RATE,
    MUTATION_RATE, MUTATION_STRENGTH,
    FITNESS_DISTANCE_WEIGHT, FITNESS_FINISH_BONUS,
    FITNESS_CRASH_PENALTY, FITNESS_SPEED_BONUS,
)

try:
    from PyQt5.QtGui import QColor
    _QT = True
except ImportError:
    _QT = False


class Population:
    """Manages genomes, evaluates fitness, and breeds the next generation."""

    def __init__(self, size: int = POPULATION_SIZE, track=None):
        self.size        = size
        self.track       = track
        self.generation  = 0

        self.genomes: list[Genome] = [Genome() for _ in range(size)]

        self.best_fitness         = 0.0
        self.best_genome: Genome | None = None
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history:  list[float] = []

        # Adaptive mutation
        self._plateau_counter  = 0
        self._current_strength = MUTATION_STRENGTH

    # ── Fitness evaluation ───────────────────────────────────────────────────

    def evaluate_fitness(self, cars: list[Car]) -> None:
        """
        Assign fitness scores to genomes based on their car's performance.

        Fitness formula
        ───────────────
        base   = gate_progress × DISTANCE_WEIGHT
        speed  = avg_speed × SPEED_BONUS × base
        finish = FITNESS_FINISH_BONUS  (if car.finished)
        crash  = multiply by CRASH_PENALTY  (if not alive and not finished)
        """
        for genome, car in zip(self.genomes, cars):
            base = car.gate_progress * FITNESS_DISTANCE_WEIGHT

            speed_bonus = 0.0
            if car.speed_history:
                avg_speed  = np.mean(car.speed_history)
                speed_bonus = avg_speed * FITNESS_SPEED_BONUS

            fitness = base + speed_bonus

            if car.finished:
                fitness += FITNESS_FINISH_BONUS
            elif not car.alive:
                fitness *= FITNESS_CRASH_PENALTY

            genome.fitness = max(0.0, fitness)

        # Update history
        fitnesses = [g.fitness for g in self.genomes]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(float(np.mean(fitnesses)))

        # Update best genome ever
        top = max(self.genomes, key=lambda g: g.fitness)
        if top.fitness >= self.best_fitness:
            self._plateau_counter = 0
            self.best_fitness  = top.fitness
            self.best_genome   = top.clone()
        else:
            self._plateau_counter += 1

    # ── Breeding ─────────────────────────────────────────────────────────────

    def next_generation(self) -> int:
        """
        Build the next generation.

        Strategy
        ────────
        1. Sort by fitness descending
        2. Elites carried over unchanged
        3. Remaining slots: CROSSOVER_RATE → crossover+mutate
                            else          → clone+mutate
        4. Diversity injection if plateau too long
        """
        # Adaptive mutation: increase strength when stuck
        if self._plateau_counter > 10:
            self._current_strength = min(MUTATION_STRENGTH * 3.0,
                                         self._current_strength * 1.1)
        else:
            self._current_strength = max(MUTATION_STRENGTH,
                                         self._current_strength * 0.95)

        self.genomes.sort(key=lambda g: g.fitness, reverse=True)

        num_elite = max(1, int(self.size * ELITISM_RATIO))
        new_pop   = [g.clone() for g in self.genomes[:num_elite]]

        # Diversity injection when plateau too long
        if self._plateau_counter > 20:
            inject_n  = max(2, self.size // 10)
            new_pop  += [Genome() for _ in range(inject_n)]
            self._plateau_counter = 0

        while len(new_pop) < self.size:
            parent1 = tournament_select(self.genomes)
            if random.random() < CROSSOVER_RATE:
                parent2 = tournament_select(self.genomes)
                child   = parent1.crossover(parent2)
            else:
                child = parent1.clone()
            child.mutate(rate=MUTATION_RATE, strength=self._current_strength)
            new_pop.append(child)

        self.genomes   = new_pop[:self.size]
        self.generation += 1
        return self.generation

    # ── Car factory ──────────────────────────────────────────────────────────

    def create_cars(self) -> list[Car]:
        """
        Spawn one Car per genome at the track start position.
        Car colour encodes fitness rank (green = best previous, red = worst).
        """
        cars = []
        n    = len(self.genomes)

        # Compute per-genome fitness rank for colouring
        fitnesses  = [g.fitness for g in self.genomes]
        max_fit    = max(fitnesses) if max(fitnesses) > 0 else 1.0

        if self.track:
            sx, sy   = self.track.start_pos
            sa       = self.track.start_angle
        else:
            sx, sy, sa = 100.0, 300.0, 0.0

        for i, genome in enumerate(self.genomes):
            car       = Car(sx, sy, angle=sa)
            car.brain  = genome.brain
            car.genome = genome

            # Colour: normalised rank → hue (green best, red worst)
            norm = fitnesses[i] / max_fit if max_fit > 0 else 0.5
            hue  = int(norm * 120)          # 0° = red, 120° = green
            if _QT:
                car.color = QColor.fromHsv(hue, 200, 220)
            else:
                car.color = (int(255 * (1 - norm)), int(255 * norm), 60)

            cars.append(car)

        return cars

    # ── Statistics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        fitnesses = [g.fitness for g in self.genomes]
        return {
            "generation":    self.generation,
            "best_fitness":  self.best_fitness,
            "gen_best":      max(fitnesses),
            "gen_avg":       float(np.mean(fitnesses)),
            "gen_std":       float(np.std(fitnesses)),
            "plateau":       self._plateau_counter,
            "mut_strength":  self._current_strength,
        }
