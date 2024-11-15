"""Collection of visualization utilities."""
import numpy as np
from matplotlib.axes import Axes


def plot_gaussian_contours_fading(
    ax: Axes,
    mean: np.ndarray,
    std_dev: np.ndarray,
    color: np.ndarray,
    max_std: int = 2,
    sigma_steps: float = 0.25,
    linewidth: int = 1,
):
    """Plot concentric ellipses to represent the contours of a 2D Gaussian distribution
    with fading colors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis object to plot the ellipses.
    mean : np.ndarray
        A numpy array representing the mean (μ) of the distribution.
    std_dev : np.ndarray
        A numpy array representing the standard deviation (σ) of the distribution.
    color : np.ndarray
        A 4D numpy array including RGB and alpha channels to specify the color and
        initial opacity.
    max_std : int, optional
        Maximum number of standard deviations to draw contours to, by default 2.
    sigma_steps : float, optional
        Step size in standard deviations for each contour, by default 0.25.
    linewidth : int, optional
        Width of the boundary line, by default 1.

    """
    from matplotlib.patches import Ellipse

    # Calculate the number of steps
    steps = int(max_std / sigma_steps)
    alphas = np.linspace(1, 0.1, steps)

    if len(color) != 4:
        color = np.concatenate([color, [1.0]])

    for i, alpha in zip(range(1, steps + 1), alphas):
        # Adjust the alpha value of the color
        current_color = np.copy(color)
        current_color[-1] = alpha  # Update the alpha channel

        # Width and height of the ellipse are 2*i*sigma_steps times the std_dev values
        width, height = (
            2 * i * sigma_steps * std_dev[0],
            2 * i * sigma_steps * std_dev[1],
        )
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=width,
            height=height,
            edgecolor=current_color,
            facecolor="none",
            linewidth=linewidth,
            linestyle="dashed",
            label=f"{i * sigma_steps}σ",
        )
        ax.add_patch(ellipse)


def plot_gaussian_contours_covariance(
    ax: Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    color: np.ndarray,
    max_std: int = 2,
    sigma_steps: float = 0.25,
    linewidth: int = 1,
):
    """Plot concentric ellipses to represent the contours of a 2D Gaussian distribution
    with fading colors. Accepts a covariance matrix to properly represent the
    distribution's orientation and shape.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis object to plot the ellipses.
    mean : np.ndarray
        A numpy array representing the mean (μ) of the distribution.
    covariance : np.ndarray
        A 2x2 numpy array representing the covariance matrix of the distribution.
    color : np.ndarray
        A 4D numpy array including RGB and alpha channels to specify the color and
        initial opacity.
    max_std : int, optional
        Maximum number of standard deviations to draw contours to, by default 2.
    sigma_steps : float, optional
        Step size in standard deviations for each contour, by default 0.25.
    linewidth : int, optional
        Width of the boundary line, by default 1.

    """
    from matplotlib.patches import Ellipse

    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    major_axis = np.sqrt(
        eigenvalues[0]
    )  # The major axis length (sqrt of larger eigenvalue)
    minor_axis = np.sqrt(
        eigenvalues[1]
    )  # The minor axis length (sqrt of smaller eigenvalue)
    angle = np.arctan2(
        *eigenvectors[:, 0][::-1]
    )  # Angle in radians between the x-axis and the major axis of the ellipse

    # Calculate the number of steps
    steps = int(max_std / sigma_steps)
    alphas = np.linspace(1, 0.1, steps)

    for i, alpha in zip(range(1, steps + 1), alphas):
        # Adjust the alpha value of the color
        current_color = np.copy(color)
        current_color[-1] = alpha  # Update the alpha channel

        # Width and height of the ellipse based on the covariance
        width, height = (
            2 * i * sigma_steps * major_axis * 2,
            2 * i * sigma_steps * minor_axis * 2,
        )
        ellipse = Ellipse(
            xy=(mean[0], mean[1]),
            width=width,
            height=height,
            angle=float(np.degrees(angle)),
            edgecolor=current_color,
            facecolor="None",
            linewidth=linewidth,
            linestyle="dashed",
            label=f"{i * sigma_steps}σ",
        )
        ax.add_patch(ellipse)


def plot_weight_matrix_as_ellipse(
    ax: Axes,
    s: float,
    W: np.ndarray,
    mean: np.ndarray,
    color: np.ndarray,
    linewidth: int = 1,
):
    """Plot the transformation of a unit circle by the weight matrix W as an ellipse.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis object to plot the ellipse.
    s : float
        Scalar to scale the weight matrix W.
    W : np.ndarray
        2x2 weight matrix.
    mean : np.ndarray
        The center point (x, y) of the ellipse.
    color : np.ndarray
        Color of the ellipse.
    linewidth : int, optional
        Width of the boundary line, by default 1.

    """
    # Compute the transformation matrix
    transform_matrix = W[:2, :2]

    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Unit circle

    # Apply the linear transformation to the circle to get an ellipse
    ellipse = 0.25 * s * s * (transform_matrix @ circle)

    # Shift the ellipse to the specified mean
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]

    # Plotting
    ax.plot(ellipse[0, :], ellipse[1, :], color=color, linewidth=linewidth)
