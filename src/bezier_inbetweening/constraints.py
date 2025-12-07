from __future__ import annotations

import numpy as np
from .bezier import CubicBezier


def enforce_point_constraint_weighted(
    curve: CubicBezier,
    t_c: float,
    target: np.ndarray,
    weight: float = 100.0,
) -> CubicBezier:
    """
    Adjust P1 and P2 so that B(t_c) is pulled toward `target`.
    Uses weighted least squares with fixed endpoints.

    This is a pragmatic, clean demo of "keyjoint target control".
    """
    if not (0.0 <= t_c <= 1.0):
        raise ValueError("t_c must be in [0,1].")

    target = np.asarray(target, dtype=float)
    if target.shape != curve.P0.shape:
        raise ValueError("target must have same dimension as control points.")

    P0, P3 = curve.P0, curve.P3
    d = curve.dim

    # Create a small pseudo-dataset:
    # We'll fit P1,P2 with:
    # 1) a soft prior that keeps them near original P1,P2
    # 2) a strongly weighted constraint equation at t_c

    # Unknown vector per dim: [P1, P2]
    # We'll form A x = b.

    # Prior rows: identity
    A_prior = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Weight the prior lightly (1.0)
    w_prior = 1.0

    # Constraint row at t_c:
    t = float(t_c)
    one = 1.0 - t
    c0 = one**3
    c1 = 3 * one**2 * t
    c2 = 3 * one * t**2
    c3 = t**3

    A_c = np.array([[c1, c2]])
    # Right side for each dimension:
    rhs_c = target - c0 * P0 - c3 * P3

    # Stack weighted system
    A = np.vstack([
        w_prior * A_prior,
        weight * A_c,
    ])

    new_P1 = np.zeros(d)
    new_P2 = np.zeros(d)

    for j in range(d):
        b = np.concatenate([
            w_prior * np.array([curve.P1[j], curve.P2[j]]),
            weight * np.array([rhs_c[j]]),
        ])
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        new_P1[j], new_P2[j] = x[0], x[1]

    return CubicBezier(P0=P0, P1=new_P1, P2=new_P2, P3=P3)
