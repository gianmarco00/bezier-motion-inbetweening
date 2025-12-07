import numpy as np
from bezier_inbetweening import fit_cubic_bezier, enforce_point_constraint_weighted
from bezier_inbetweening.utils import make_toy_hand_trajectory


def test_constraint_pulls_midpoint():
    points = make_toy_hand_trajectory(n=80)
    ts = np.linspace(0, 1, points.shape[0])

    base = fit_cubic_bezier(points, ts=ts, fix_ends=True)

    t_c = 0.5
    base_mid = base.eval(t_c)
    base_mid = base_mid[0] if base_mid.ndim == 2 else base_mid

    target = base_mid + np.array([0.0, 0.2, 0.0])

    constrained = enforce_point_constraint_weighted(base, t_c=t_c, target=target, weight=200.0)
    new_mid = constrained.eval(t_c)
    new_mid = new_mid[0] if new_mid.ndim == 2 else new_mid

    # Ensure we're closer to target than the base curve
    assert np.linalg.norm(new_mid - target) < np.linalg.norm(base_mid - target)
