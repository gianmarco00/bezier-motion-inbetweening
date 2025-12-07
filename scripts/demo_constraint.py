import numpy as np
import matplotlib.pyplot as plt

from bezier_inbetweening import fit_cubic_bezier, enforce_point_constraint_weighted
from bezier_inbetweening.utils import make_toy_hand_trajectory


def main():
    points = make_toy_hand_trajectory(n=80)
    ts = np.linspace(0, 1, points.shape[0])

    base = fit_cubic_bezier(points, ts=ts, fix_ends=True)

    # Create a "dramatic beat" target near mid-time
    t_c = 0.5
    mid = base.eval(t_c)[0] if base.eval(t_c).ndim == 2 else base.eval(t_c)
    target = mid + np.array([0.0, 0.15, 0.0])  # exaggerate upward arc a bit

    constrained = enforce_point_constraint_weighted(base, t_c=t_c, target=target, weight=200.0)

    recon_base = base.eval(ts)
    recon_const = constrained.eval(ts)

    plt.figure()
    plt.plot(recon_base[:, 0], recon_base[:, 1], label="base")
    plt.plot(recon_const[:, 0], recon_const[:, 1], label="constrained")
    plt.scatter([target[0]], [target[1]], label="target")
    plt.legend()
    plt.title("Constraint demo: mid-trajectory exaggeration target")
    plt.show()


if __name__ == "__main__":
    main()
