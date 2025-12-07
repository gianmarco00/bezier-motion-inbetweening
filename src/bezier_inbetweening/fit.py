from __future__ import annotations

import numpy as np
from .bezier import CubicBezier


def _default_ts(n: int) -> np.ndarray:
    if n < 2:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def fit_cubic_bezier(
    points: np.ndarray,
    ts: np.ndarray | None = None,
    fix_ends: bool = True,
) -> CubicBezier:
    """
    Fit a cubic Bézier curve to a sequence of points (N, D).

    If fix_ends=True:
        P0 = points[0], P3 = points[-1]
        Solve least squares for P1, P2.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be of shape (N, D).")

    n, d = points.shape
    if n < 2:
        raise ValueError("Need at least 2 points.")

    if ts is None:
        ts = _default_ts(n)
    ts = np.asarray(ts, dtype=float)

    if ts.ndim != 1 or ts.shape[0] != n:
        raise ValueError("ts must be shape (N,).")

    if fix_ends:
        P0 = points[0]
        P3 = points[-1]

        t = ts.reshape(-1, 1)
        one = 1.0 - t

        c0 = (one**3)            # (N,1)
        c1 = 3 * one**2 * t      # (N,1)
        c2 = 3 * one * t**2      # (N,1)
        c3 = (t**3)              # (N,1)

        # Build A for unknowns P1 and P2:
        # points = c0*P0 + c1*P1 + c2*P2 + c3*P3
        # => (c1 c2) [P1; P2] = points - c0*P0 - c3*P3
        A = np.concatenate([c1, c2], axis=1)  # (N, 2)

        rhs = points - c0 * P0 - c3 * P3  

        P1 = np.zeros(d)
        P2 = np.zeros(d)
        for j in range(d):
            x, *_ = np.linalg.lstsq(A, rhs[:, j], rcond=None)
            P1[j], P2[j] = x[0], x[1]

        return CubicBezier(P0=P0, P1=P1, P2=P2, P3=P3)

    t = ts.reshape(-1, 1)
    one = 1.0 - t

    B0 = one**3
    B1 = 3 * one**2 * t
    B2 = 3 * one * t**2
    B3 = t**3
    A = np.concatenate([B0, B1, B2, B3], axis=1)  # (N, 4)

    P = np.zeros((4, d))
    for j in range(d):
        x, *_ = np.linalg.lstsq(A, points[:, j], rcond=None)
        P[:, j] = x

    return CubicBezier(P0=P[0], P1=P[1], P2=P[2], P3=P[3])


def fit_piecewise_bezier(
    points: np.ndarray,
    num_segments: int = 2,
    fix_ends: bool = True,
) -> list[CubicBezier]:
    
    points = np.asarray(points, dtype=float)
    n = points.shape[0]

    if num_segments < 1:
        raise ValueError("num_segments must be >= 1.")
    if n < 2 * num_segments:
        raise ValueError("Too few points for the requested number of segments.")

    curves: list[CubicBezier] = []
    idxs = np.linspace(0, n - 1, num_segments + 1).astype(int)

    for s in range(num_segments):
        a, b = idxs[s], idxs[s + 1]
        seg = points[a : b + 1]
        ts = np.linspace(0.0, 1.0, seg.shape[0])
        curves.append(fit_cubic_bezier(seg, ts=ts, fix_ends=fix_ends))

    return curves
