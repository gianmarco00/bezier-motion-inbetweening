from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def make_toy_hand_trajectory(n: int = 60, dim: int = 3, seed: int = 0) -> np.ndarray:
    """
    Simple smooth toy trajectory for demos.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)

    # A smooth arc with a little structured variation
    x = t
    y = np.sin(np.pi * t) * 0.5
    z = (1 - np.cos(np.pi * t)) * 0.25

    traj = np.stack([x, y, z], axis=1)

    if dim != 3:
        # Pad or truncate for generality
        if dim > 3:
            pad = rng.normal(scale=0.01, size=(n, dim - 3))
            traj = np.concatenate([traj, pad], axis=1)
        else:
            traj = traj[:, :dim]

    return traj
