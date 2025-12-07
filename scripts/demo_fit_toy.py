import numpy as np
import matplotlib.pyplot as plt

from bezier_inbetweening import fit_cubic_bezier
from bezier_inbetweening.utils import make_toy_hand_trajectory, mse


def main():
    points = make_toy_hand_trajectory(n=80)
    ts = np.linspace(0, 1, points.shape[0])

    curve = fit_cubic_bezier(points, ts=ts, fix_ends=True)
    recon = curve.eval(ts)

    print("MSE:", mse(points, recon))

    # Plot Y vs X for a quick 2D visualization
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], label="original")
    plt.plot(recon[:, 0], recon[:, 1], label="bezier fit")
    plt.legend()
    plt.title("Toy keyjoint trajectory: Bézier fit")
    plt.show()


if __name__ == "__main__":
    main()
