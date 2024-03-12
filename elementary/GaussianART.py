"""
Williamson, J. R. (1996).
Gaussian ARTMAP: A Neural Network for Fast Incremental Learning of Noisy Multidimensional Maps.
Neural Networks, 9, 881 – 897. doi:10.1016/0893-6080(95)00115-8.
"""

import numpy as np
from typing import Optional, Iterable
from matplotlib.axes import Axes
from common.BaseART import BaseART


def plot_gaussian_contours_fading(
        ax: Axes,
        mean: np.ndarray,
        std_dev: np.ndarray,
        color: np.ndarray,
        max_std: int = 2,
        sigma_steps: float = 0.25,
        linewidth: int = 1
):
    """
    Plots concentric ellipses to represent the contours of a 2D Gaussian distribution, with fading colors.

    Parameters:
        - ax: Matplotlib axis object. If None, creates a new figure and axis.
    - mean: A numpy array representing the mean (μ) of the distribution.
    - std_dev: A numpy array representing the standard deviation (σ) of the distribution.
    - color: A 4D numpy array including RGB and alpha channels to specify the color and initial opacity.
    - max_std: Max standard deviations to draw contours to. Default is 2.
    - sigma_steps: Step size in standard deviations for each contour. Default is 0.25.

    """
    from matplotlib.patches import Ellipse

    # Calculate the number of steps
    steps = int(max_std / sigma_steps)
    alphas = np.linspace(1, 0.1, steps)

    if len(color) != 4:
        color = np.concatenate([color, [1.]])

    for i, alpha in zip(range(1, steps + 1), alphas):
        # Adjust the alpha value of the color
        current_color = np.copy(color)
        current_color[-1] = alpha  # Update the alpha channel

        # Width and height of the ellipse are 2*i*sigma_steps times the std_dev values
        width, height = 2 * i * sigma_steps * std_dev[0], 2 * i * sigma_steps * std_dev[1]
        ellipse = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, edgecolor=current_color, facecolor='none', linewidth=linewidth,
                          linestyle='dashed', label=f'{i * sigma_steps}σ')
        ax.add_patch(ellipse)


class GaussianART(BaseART):
    # implementation of GaussianART
    pi2 = np.pi*2

    @staticmethod
    def validate_params(params: dict):
        assert "rho" in params
        assert "sigma_init" in params
        assert 1.0 >= params["rho"] > 0.


    def category_choice(self, i: np.ndarray, w: np.ndarray, params: dict) -> tuple[float, Optional[dict]]:
        mean = w[:self.dim_]
        sigma = w[self.dim_:-1]
        n = w[-1]
        sig = np.diag(np.multiply(sigma,sigma))
        dist = mean-i
        exp_dist_sig_dist = np.exp(-0.5*np.matmul(dist.T, np.matmul(np.linalg.inv(sig), dist)))
        cache = {
            "exp_dist_sig_dist": exp_dist_sig_dist
        }
        p_i_cj = exp_dist_sig_dist/np.sqrt((self.pi2**self.dim_)*np.linalg.det(sig))
        p_cj = n/np.sum(w_[-1] for w_ in self.W)

        activation = p_i_cj*p_cj

        return activation, cache


    def match_criterion(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> float:
        if cache is None:
            raise ValueError("No cache provided")
        exp_dist_sig_dist = cache["exp_dist_sig_dist"]
        return exp_dist_sig_dist


    def match_criterion_bin(self, i: np.ndarray, w: np.ndarray, params: dict, cache: Optional[dict] = None) -> bool:
        return self.match_criterion(i, w, params=params, cache=cache) >= params["rho"]


    def update(self, i: np.ndarray, w: np.ndarray, params, cache: Optional[dict] = None) -> np.ndarray:
        mean = w[:self.dim_]
        sigma = w[self.dim_:-1]
        n = w[-1]

        n_new = n+1
        mean_new = (1-(1/n_new))*mean + (1/n_new)*i
        sigma_new = np.sqrt((1-(1/n_new))*np.multiply(sigma, sigma) + (1/n_new)*((mean_new - i)**2))

        return np.concatenate([mean_new, sigma_new, [n_new]])


    def new_weight(self, i: np.ndarray, params: dict) -> np.ndarray:
        return np.concatenate([i, params["sigma_init"], [1.]])


    def plot_cluster_bounds(self, ax: Axes, colors: Iterable, linewidth: int = 1):
        for w, col in zip(self.W, colors):
            mean = w[:self.dim_]
            sigma = w[self.dim_:-1]
            plot_gaussian_contours_fading(ax, mean, sigma, col, linewidth=linewidth)
