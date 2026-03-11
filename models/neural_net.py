"""
Feedforward neural network built purely with NumPy.

Improvements over the original plan
────────────────────────────────────
• Leaky-ReLU hidden activations (avoids dying-ReLU in sparse populations)
• Proper He-initialisation for Leaky-ReLU (√(2/(1+α²)·n))
• Explicit dtype=float64 throughout (avoids silent float32 precision loss)
• get_weights_flattened / set_weights_from_flattened are now symmetric and
  round-trip safe (tested in the unit test suite)
• clone() helper for elitism copies
• __repr__ for quick inspection
"""

import numpy as np
from config import NN_INPUT_SIZE, NN_HIDDEN_LAYERS, NN_OUTPUT_SIZE


LEAKY_ALPHA = 0.05   # slope for x < 0 in Leaky-ReLU


class NeuralNetwork:
    """
    Feedforward NN: input → [hidden …] → output

    Activations
    ───────────
    Hidden  : Leaky-ReLU(α=0.05)
    Output 0: tanh  → steering  ∈ [-1, 1]
    Output 1: sigmoid → throttle ∈ [ 0, 1]
    """

    def __init__(self,
                 input_size: int  = NN_INPUT_SIZE,
                 hidden_sizes     = NN_HIDDEN_LAYERS,
                 output_size: int = NN_OUTPUT_SIZE):

        self.layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.weights: list[np.ndarray] = []
        self.biases:  list[np.ndarray] = []
        self._build()

    # ── Construction / weight access ─────────────────────────────────────────

    def _build(self):
        """He-initialise all weight matrices and zero-initialise biases."""
        self.weights.clear()
        self.biases.clear()
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            # He init for Leaky-ReLU
            std = np.sqrt(2.0 / ((1.0 + LEAKY_ALPHA ** 2) * fan_in))
            W = np.random.randn(self.layer_sizes[i + 1], fan_in).astype(np.float64) * std
            b = np.zeros((self.layer_sizes[i + 1], 1), dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)

    # ── Activations ──────────────────────────────────────────────────────────

    @staticmethod
    def _leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, LEAKY_ALPHA * x)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> tuple[float, float]:
        """
        Run a single forward pass.

        Parameters
        ----------
        X : array-like, shape (input_size,)

        Returns
        -------
        steering : float  ∈ [-1, 1]
        throttle : float  ∈ [ 0, 1]
        """
        a = np.asarray(X, dtype=np.float64).reshape(-1, 1)

        # Hidden layers → Leaky-ReLU
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self._leaky_relu(W @ a + b)

        # Output layer
        z_out = self.weights[-1] @ a + self.biases[-1]

        steering = float(self._tanh(z_out[0, 0]))
        throttle = float(self._sigmoid(z_out[1, 0]))

        return steering, throttle

    # ── Serialisation ────────────────────────────────────────────────────────

    def get_weights_flattened(self) -> np.ndarray:
        """Return all weights and biases as a single 1-D float64 array."""
        parts = []
        for W, b in zip(self.weights, self.biases):
            parts.append(W.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_weights_from_flattened(self, flat: np.ndarray):
        """Restore weights from a flattened array produced by get_weights_flattened."""
        flat = np.asarray(flat, dtype=np.float64)
        idx = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            ws = W.size
            bs = b.size
            self.weights[i] = flat[idx: idx + ws].reshape(W.shape)
            idx += ws
            self.biases[i]  = flat[idx: idx + bs].reshape(b.shape)
            idx += bs
        if idx != flat.size:
            raise ValueError(
                f"Flat array size mismatch: expected {idx}, got {flat.size}"
            )

    def clone(self) -> "NeuralNetwork":
        """Deep copy of this network."""
        nn = NeuralNetwork.__new__(NeuralNetwork)
        nn.layer_sizes = self.layer_sizes[:]
        nn.weights = [W.copy() for W in self.weights]
        nn.biases  = [b.copy() for b in self.biases]
        return nn

    @property
    def total_params(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.weights, self.biases))

    def __repr__(self) -> str:
        return (f"NeuralNetwork(layers={self.layer_sizes}, "
                f"params={self.total_params})")
