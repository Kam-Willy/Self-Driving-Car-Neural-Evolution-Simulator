"""
Celebration animation helpers.

Provides confetti particle system drawn onto a Matplotlib Axes.

Improvements over the original plan
────────────────────────────────────
• Particle class with velocity, gravity, fade-out lifetime
• update() advances all particles one frame and prunes dead ones
• draw() renders current state onto provided Axes (no full figure clear)
• ConfettiSystem.reset() re-spawns particles so it can be called each victory
"""

import random
import numpy as np
import matplotlib.patches as mpatches

from config import CONFETTI_COUNT, CANVAS_W, CANVAS_H


_COLORS = [
    '#FF4136', '#0074D9', '#2ECC40', '#FFDC00',
    '#B10DC9', '#FF69B4', '#FF6B35', '#7FDBFF',
]


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'size', 'color', 'life', 'max_life', 'shape')

    def __init__(self):
        self.reset()

    def reset(self):
        self.x       = random.uniform(0, CANVAS_W)
        self.y       = random.uniform(CANVAS_H * 0.6, CANVAS_H)
        self.vx      = random.uniform(-3, 3)
        self.vy      = random.uniform(4, 10)    # upward (canvas y flipped)
        self.size    = random.uniform(8, 20)
        self.color   = random.choice(_COLORS)
        self.max_life = random.randint(40, 100)
        self.life     = self.max_life
        self.shape    = random.choice(['s', 'o', 'D', '^'])

    def update(self):
        self.x    += self.vx
        self.y    -= self.vy                    # move upward in canvas coords
        self.vy   -= 0.25                       # gravity
        self.vx   *= 0.98                       # air resistance
        self.life -= 1

    @property
    def alpha(self):
        return max(0.1, self.life / self.max_life)

    @property
    def alive(self):
        return self.life > 0 and 0 <= self.x <= CANVAS_W


class ConfettiSystem:
    """Manages a pool of Particle objects and renders them each frame."""

    def __init__(self, count: int = CONFETTI_COUNT):
        self._count    = count
        self.particles: list[Particle] = []
        self._artists  = []        # track scatter artists for cleanup
        self.active    = False

    def reset(self):
        """Spawn a fresh batch of confetti."""
        self.particles = [Particle() for _ in range(self._count)]
        self.active    = True

    def update(self):
        """Advance one frame. Re-spawn dead particles while active."""
        for p in self.particles:
            p.update()
        if self.active:
            for p in self.particles:
                if not p.alive:
                    p.reset()

    def stop(self):
        """Stop re-spawning; particles fade out naturally."""
        self.active = False

    def draw(self, ax):
        """
        Render current particles onto *ax* (Matplotlib Axes).
        Removes previous frame's artists first.
        """
        for a in self._artists:
            try:
                a.remove()
            except ValueError:
                pass
        self._artists.clear()

        if not self.particles:
            return

        for p in self.particles:
            if not p.alive:
                continue
            sc = ax.scatter(
                p.x, p.y,
                s=p.size ** 2,
                c=p.color,
                marker=p.shape,
                alpha=p.alpha,
                zorder=10,
            )
            self._artists.append(sc)
