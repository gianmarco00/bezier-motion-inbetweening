import numpy as np
from bezier_inbetweening import CubicBezier


def test_endpoints():
    P0 = np.array([0.0, 0.0, 0.0])
    P1 = np.array([1.0, 0.0, 0.0])
    P2 = np.array([1.0, 1.0, 0.0])
    P3 = np.array([0.0, 1.0, 0.0])

    c = CubicBezier(P0, P1, P2, P3)
    assert np.allclose(c.eval(0.0), P0)
    assert np.allclose(c.eval(1.0), P3)
