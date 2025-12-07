from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CubicBezier:
    """
    Cubic Bézier curve in D dimensions.

    Control points:
        P0, P1, P2, P3: shape (D,)
    """
    P0: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    P3: np.ndarray

    def __post_init__(self) -> None:
        self.P0 = np.asarray(self.P0, dtype=float)
        self.P1 = np.asarray(self.P1, dtype=float)
        self.P2 = np.asarray(self.P2, dtype=float)
        self.P3 = np.asarray(self.P3, dtype=float)

        if not (self.P0.shape == self.P1.shape == self.P2.shape == self.P3.shape):
            raise ValueError("All control points must have the same shape.")

    @property
    def dim(self) -> int:
        return int(self.P0.size)

    def eval(self, t: np.ndarray | float) -> np.ndarray:
        """
        Evaluate curve at t in [0, 1].
        Returns shape (..., D)
        """
        t = np.asarray(t, dtype=float)
        t_shape = t.shape

        t = t.reshape(-1, 1)  # (N, 1)
        one = 1.0 - t

        b0 = one**3
        b1 = 3 * one**2 * t
        b2 = 3 * one * t**2
        b3 = t**3

        pts = b0 * self.P0 + b1 * self.P1 + b2 * self.P2 + b3 * self.P3
        pts = pts.reshape(*t_shape, self.dim)
        return pts

    def derivative(self, t: np.ndarray | float) -> np.ndarray:
        """
        First derivative wrt t.
        """
        t = np.asarray(t, dtype=float)
        t_shape = t.shape

        t = t.reshape(-1, 1)
        one = 1.0 - t

        # d/dt of cubic Bézier basis combination:
        # 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
        d = (
            3 * one**2 * (self.P1 - self.P0)
            + 6 * one * t * (self.P2 - self.P1)
            + 3 * t**2 * (self.P3 - self.P2)
        )

        d = d.reshape(*t_shape, self.dim)
        return d

    def sample(self, n: int = 60) -> np.ndarray:
        t = np.linspace(0.0, 1.0, n)
        return self.eval(t)
