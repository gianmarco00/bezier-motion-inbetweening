import numpy as np
from bezier_inbetweening import fit_cubic_bezier
from bezier_inbetweening.utils import make_toy_hand_trajectory, mse


def test_fit_reasonable_error():
    points = make_toy_hand_trajectory(n=50)
    ts = np.linspace(0, 1, points.shape[0])

    curve = fit_cubic_bezier(points, ts=ts, fix_ends=True)
    recon = curve.eval(ts)

    assert mse(points, recon) < 1e-2
