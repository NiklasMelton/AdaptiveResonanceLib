import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import colorspacious as cs
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.transforms import Affine2D

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class CustomSymLogScale(mscale.ScaleBase):
    """Custom symmetric log scale that behaves logarithmically for large values but is
    linear between -linthresh and +linthresh."""

    name = "custom_symlog"

    def __init__(self, axis, *, linthresh=1e-5, base=10, subs=None, **kwargs):
        # Properly call the parent class constructor
        super().__init__(axis)
        self.linthresh = linthresh
        self.base = base
        self.subs = subs

    def get_transform(self):
        return self.CustomSymLogTransform(self.linthresh, self.base)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(plt.LogLocator(self.base, subs=self.subs))
        axis.set_minor_locator(plt.LogLocator(self.base, subs=self.subs, numticks=12))
        axis.set_major_formatter(plt.ScalarFormatter())

    class CustomSymLogTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, linthresh, base):
            mtransforms.Transform.__init__(self)
            self.linthresh = linthresh
            self.log_base = np.log(base)

        def transform_non_affine(self, values):
            """Apply custom symmetric log transformation."""
            sign = np.sign(values)
            abs_values = np.abs(values)

            with np.errstate(invalid="ignore"):
                return sign * np.where(
                    abs_values < self.linthresh,
                    abs_values,  # Linear transformation below the threshold
                    self.linthresh
                    + np.log(np.maximum(abs_values / self.linthresh, 1e-50))
                    / self.log_base,
                )

        def inverted(self):
            return CustomSymLogScale.InvertedCustomSymLogTransform(
                self.linthresh, self.log_base
            )

    class InvertedCustomSymLogTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, linthresh, log_base):
            mtransforms.Transform.__init__(self)
            self.linthresh = linthresh
            self.log_base = log_base

        def transform_non_affine(self, values):
            """Apply inverse of custom symmetric log transformation."""
            sign = np.sign(values)
            abs_values = np.abs(values)

            return sign * np.where(
                abs_values < self.linthresh,
                abs_values,
                self.linthresh * np.exp((abs_values - self.linthresh) * self.log_base),
            )

        def inverted(self):
            return CustomSymLogScale.CustomSymLogTransform(
                self.linthresh, self.log_base
            )


# Register the scale with matplotlib
mscale.register_scale(CustomSymLogScale)


def smart_title(input_string):
    # Apply title() to the string
    title_string = input_string.title()

    # Create a new string by keeping the original uppercase where applicable
    result = "".join(
        [
            orig if orig.isupper() else title
            for orig, title in zip(input_string, title_string)
        ]
    )

    return result


def generate_ciecam02_colors(n):
    # Generate n evenly spaced hues in the JCh color space (CIECAM02)
    hues = np.linspace(0, 360, n, endpoint=False)  # Generate equally spaced hues
    chroma = 35 * (np.sin(np.linspace(0, 100, n, endpoint=False)) + 1) + 30
    lightness = 25 * (np.sin(np.linspace(0, 100, n, endpoint=False) / 3) + 1) + 30
    # lightness = [50 for _ in range(n)]

    # Convert JCh to RGB using colorspacious
    colors = []
    for h, c, j in zip(hues, chroma, lightness):
        jch_color = [j, c, h]  # J, C, and H values
        rgb_color = cs.cspace_convert(jch_color, "JCh", "sRGB1")  # Convert to RGB
        rgb_color = np.clip(
            rgb_color, 0, 1
        )  # Ensure values are within the [0, 1] range
        colors.append(rgb_color)

    return colors


def wash_out_color(rgb_color, factor=0.5):
    """Make a color look washed out by blending it with white.

    Parameters:
        rgb_color (list or np.ndarray): The RGB color to wash out, with values in the range [0, 1].
        factor (float): The blending factor. 0 means no change, 1 means fully white. Default is 0.5.

    Returns:
        np.ndarray: The washed-out color.

    """
    white = np.array([1, 1, 1])
    washed_out_color = rgb_color * (1 - factor) + white * factor
    return np.clip(washed_out_color, 0, 1)  # Ensure values remain in [0, 1]


def filter_data(df):
    # First, group by ['dataset', 'match_tracking', 'model', 'epsilon'] and calculate the mean values
    df_grouped = df.groupby(
        ["dataset", "match_tracking", "model", "epsilon"], as_index=False
    ).agg(
        {
            "mean_accuracy": "mean",
            "mean_precision": "mean",
            "mean_recall": "mean",
            "mean_f1": "mean",
            "mean_time": "mean",
            "mean_clusters": "mean",
            "params": "first",  # Assuming 'params' and 'shuffle' can be kept as is within the group
            "shuffle": "first",
        }
    )
    df_grouped = df_grouped[~np.isinf(df_grouped["epsilon"])]

    # Then, within each ['dataset', 'match_tracking', 'model'] group, sort by mean_accuracy (descending)
    # and mean_time (ascending), then take the first row of each group
    df_sorted = df_grouped.sort_values(
        ["mean_accuracy", "mean_time"], ascending=[False, True]
    )

    # Drop duplicates, keeping the first row within each ['dataset', 'match_tracking', 'model'] group
    df_final = df_sorted.drop_duplicates(
        subset=["dataset", "match_tracking", "model"], keep="first"
    )
    df_final = df_final.sort_values(by=["dataset", "model", "match_tracking"])

    # Return the final dataframe
    return df_final


def squash_folds(df):
    df_final = df.groupby(
        ["dataset", "match_tracking", "model", "epsilon"], as_index=False
    ).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_clusters=("clusters", "mean"),
        mean_time=("time", "mean"),
        accuracy=("accuracy", list),
        clusters=("clusters", list),
        time=("time", list),
        params=("params", "first"),
        shuffle=("shuffle", "first"),
    )
    return df_final


def get_clipped_ellipse(x, y, width, height, min_value=1e-5, num_vertices=360):
    """Generate a clipped ellipse with a higher number of vertices for better
    resolution.

    Parameters:
    - x, y: center of the ellipse
    - width, height: dimensions of the ellipse
    - min_value: the minimum value to clip the vertices
    - num_vertices: the number of vertices to use for generating the ellipse (resolution)

    Returns:
    - A matplotlib Polygon object representing the clipped ellipse

    """

    # Generate the parametric angle for the ellipse with higher resolution
    angles = np.linspace(0, 2 * np.pi, num_vertices)

    # Parametric equation for the ellipse
    ellipse_x = (width / 2) * np.cos(angles)
    ellipse_y = (height / 2) * np.sin(angles)

    # Combine x and y coordinates into a single array of vertices
    vertices = np.vstack([ellipse_x, ellipse_y]).T

    # Apply transformation to move the ellipse to the correct center
    transform = Affine2D().translate(x, y)
    vertices = transform.transform(vertices)

    # Clip the vertices where they are below the threshold
    vertices[:, 0] = np.maximum(vertices[:, 0], min_value)
    vertices[:, 1] = np.maximum(vertices[:, 1], min_value)

    # Create a new polygon from the clipped vertices
    clipped_ellipse = plt.Polygon(
        vertices, edgecolor="gray", facecolor="none", linestyle="--", linewidth=1
    )

    return clipped_ellipse


def plot_combined(df):
    # Define the unique markers for match_tracking methods (assuming 5 unique values)
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
    ]  # Circle, Square, Diamond, Triangle Up, Triangle Down
    match_tracking_unique = df["match_tracking"].unique()

    dataset_unique = df["dataset"].unique()
    colors = generate_ciecam02_colors(len(dataset_unique))

    # Create a mapping of markers to match_tracking and colors to datasets
    marker_map = {
        match_tracking_unique[i]: markers[i] for i in range(len(match_tracking_unique))
    }
    color_map = {dataset_unique[i]: colors[i] for i in range(len(dataset_unique))}

    # Prepare subplots, with one plot per model stacked vertically
    unique_models = df["model"].unique()
    num_models = len(unique_models)
    fig, axes = plt.subplots(
        num_models, 3, figsize=(30, 6 * num_models), constrained_layout=False
    )

    # Loop over each unique model to create a subplot for each
    for idx, model in enumerate(unique_models):
        model_data = df[df["model"] == model]

        ax = axes[idx, 0] if num_models > 1 else axes  # In case of a single subplot
        for _, row in model_data.iterrows():
            ax.scatter(
                row["mean_time"],
                row["mean_accuracy"],
                color=color_map[row["dataset"]],
                marker=marker_map[row["match_tracking"]],
                label=f"Dataset: {row['dataset']}, Match Tracking: {row['match_tracking']}",
                s=30,  # Size of markers,
                edgecolors="black",
            )

        # Set titles and labels
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(
            f"Accuracy vs Time for {model.replace('ART', ' ARTMAP')}", fontsize=18
        )
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xscale("log")  # Apply log scale to the time axis
        ax.grid(True)

    ####################################################################################################################
    # plot clusters vs time relative

    # Calculate the ratios of accuracy and time relative to 'MT+'
    ratio_list = []
    for model in df["model"].unique():
        for dataset in df["dataset"].unique():
            # Get the subset of data for the model and dataset
            subset = df[(df["model"] == model) & (df["dataset"] == dataset)]
            mt_plus = subset[subset["match_tracking"] == "MT+"]

            mt_plus_accuracy = mt_plus["mean_accuracy"].values[0]
            mt_plus_time = mt_plus["mean_time"].values[0]
            mt_plus_clusters = mt_plus["mean_clusters"].values[0]

            # Calculate ratios for each other match_tracking method
            for _, row in subset.iterrows():
                if row["match_tracking"] != "MT+":
                    ratio_list.append(
                        {
                            "model": row["model"],
                            # 'dataset': row['dataset'],
                            "match_tracking": row["match_tracking"],
                            "accuracy_ratio": row["mean_accuracy"] / mt_plus_accuracy,
                            "time_ratio": row["mean_time"] / mt_plus_time,
                            "clusters_ratio": row["mean_clusters"] / mt_plus_clusters,
                        }
                    )

    # Convert the list of ratios to a DataFrame
    ratio_df = pd.DataFrame(ratio_list)

    # Calculate standard deviation and sample size for each group
    stats_df = (
        ratio_df.groupby(["model", "match_tracking"])
        .agg(
            time_ratio_std=("time_ratio", "std"),
            time_ratio_count=("time_ratio", "count"),
            time_ratio_mean=("time_ratio", "mean"),
            accuracy_ratio_std=("accuracy_ratio", "std"),
            accuracy_ratio_count=("accuracy_ratio", "count"),
            accuracy_ratio_mean=("accuracy_ratio", "mean"),
            clusters_ratio_std=("clusters_ratio", "std"),
            clusters_ratio_count=("clusters_ratio", "count"),
            clusters_ratio_mean=("clusters_ratio", "mean"),
        )
        .reset_index()
    )

    # Calculate the margin of error for a 97.5% confidence interval (z-score = 1.96)
    z_score = stats.norm.ppf(0.975)

    stats_df["time_ratio_margin"] = z_score * (
        stats_df["time_ratio_std"] / np.sqrt(stats_df["time_ratio_count"])
    )
    stats_df["accuracy_ratio_margin"] = z_score * (
        stats_df["accuracy_ratio_std"] / np.sqrt(stats_df["accuracy_ratio_count"])
    )
    stats_df["clusters_ratio_margin"] = z_score * (
        stats_df["clusters_ratio_std"] / np.sqrt(stats_df["clusters_ratio_count"])
    )

    # Calculate confidence bounds
    stats_df["time_ratio_lower"] = (
        stats_df["time_ratio_mean"] - stats_df["time_ratio_margin"]
    )
    stats_df["time_ratio_upper"] = (
        stats_df["time_ratio_mean"] + stats_df["time_ratio_margin"]
    )

    stats_df["accuracy_ratio_lower"] = (
        stats_df["accuracy_ratio_mean"] - stats_df["accuracy_ratio_margin"]
    )
    stats_df["accuracy_ratio_upper"] = (
        stats_df["accuracy_ratio_mean"] + stats_df["accuracy_ratio_margin"]
    )

    stats_df["clusters_ratio_lower"] = (
        stats_df["clusters_ratio_mean"] - stats_df["clusters_ratio_margin"]
    )
    stats_df["clusters_ratio_upper"] = (
        stats_df["clusters_ratio_mean"] + stats_df["clusters_ratio_margin"]
    )

    # Loop over each unique model to create a subplot for each
    for idx, model in enumerate(unique_models):
        model_data = stats_df[stats_df["model"] == model]

        ax = axes[idx, 1]
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1)

        for _, row in model_data.iterrows():
            ax.scatter(
                row["time_ratio_mean"],
                row["accuracy_ratio_mean"],
                color="gray",  # All methods will have the same color, as we are averaging over datasets
                marker=marker_map[row["match_tracking"]],
                label=f"Match Tracking: {row['match_tracking']}",
                s=50,  # Size of markers
                edgecolors="black",
            )

            # Plot the confidence ellipsoid using the bounds
            width = 2 * (row["time_ratio_upper"] - row["time_ratio_lower"]) / 2
            height = 2 * (row["accuracy_ratio_upper"] - row["accuracy_ratio_lower"]) / 2

            # Create an ellipse patch
            if model != "GaussianART":
                ellipsis = patches.Ellipse(
                    (row["time_ratio_mean"], row["accuracy_ratio_mean"]),
                    width=width,
                    height=height,
                    edgecolor="gray",
                    facecolor="none",
                    linestyle="--",  # Dashed line
                    linewidth=1,
                )
            else:
                ellipsis = get_clipped_ellipse(
                    row["time_ratio_mean"], row["accuracy_ratio_mean"], width, height
                )
                ax.set_xlim(8e-2, 3e1)

            # Add the ellipse to the axis
            ax.add_patch(ellipsis)
        ax.scatter([1.0], [1.0], color="red", marker=marker_map["MT+"], s=50)

        # Set titles and labels
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(
            f"Accuracy Ratio vs Time Ratio for {model.replace('ART', ' ARTMAP')}",
            fontsize=18,
        )
        ax.set_xlabel("Time Ratio (relative to MT+)", fontsize=14)
        ax.set_ylabel("Accuracy Ratio (relative to MT+)", fontsize=14)

        if model == "GaussianART":
            ax.set_xscale("log")
        ax.grid(True)

    ####################################################################################################################
    # plot clusters vs time relative

    # Loop over each unique model to create a subplot for each
    for idx, model in enumerate(unique_models):
        model_data = stats_df[stats_df["model"] == model]

        ax = axes[idx, 2]
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.axvline(1.0, color="red", linestyle="--", linewidth=1)

        for _, row in model_data.iterrows():
            ax.scatter(
                row["time_ratio_mean"],
                row["clusters_ratio_mean"],
                color="gray",  # All methods will have the same color, as we are averaging over datasets
                marker=marker_map[row["match_tracking"]],
                label=f"Match Tracking: {row['match_tracking']}",
                s=50,  # Size of markers
                edgecolors="black",
            )

            # Plot the confidence ellipsoid using the bounds
            width = row["time_ratio_upper"] - row["time_ratio_lower"]
            height = row["clusters_ratio_upper"] - row["clusters_ratio_lower"]

            # Create an ellipse patch
            if model != "GaussianART":
                ellipsis = patches.Ellipse(
                    (row["time_ratio_mean"], row["clusters_ratio_mean"]),
                    width=width,
                    height=height,
                    edgecolor="gray",
                    facecolor="none",
                    linestyle="--",  # Dashed line
                    linewidth=1,
                )
            else:
                ellipsis = get_clipped_ellipse(
                    row["time_ratio_mean"], row["clusters_ratio_mean"], width, height
                )
                # print(row)
                ax.set_xlim(8e-2, 3e1)
                # ax.set_ylim(1e-1, 2e2)

            # Add the ellipse to the axis
            ax.add_patch(ellipsis)

        ax.scatter([1.0], [1.0], color="red", marker=marker_map["MT+"], s=50)

        # Set titles and labels
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(
            f"Category Ratio vs Time Ratio for {model.replace('ART', ' ARTMAP')}",
            fontsize=18,
        )
        ax.set_xlabel("Time Ratio (relative to MT+)", fontsize=14)
        ax.set_ylabel("Category Ratio (relative to MT+)", fontsize=14)
        ax.set_yscale("log")  # Apply log scale to the time ratio axis
        if model == "GaussianART":
            ax.set_xscale("log")
        ax.grid(True)

    # Custom legends for match_tracking (markers) and dataset (colors)
    match_tracking_legend = [
        mlines.Line2D(
            [],
            [],
            marker=marker_map[mt],
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label=f"{mt}",
        )
        if mt != "MT+"
        else mlines.Line2D(
            [],
            [],
            marker=marker_map[mt],
            color="w",
            markerfacecolor="red",
            markersize=10,
            label=f"{mt}",
        )
        for mt in match_tracking_unique
    ] + [
        mlines.Line2D(
            [], [], color="gray", linestyle="--", label=f"95% Confidence\nInterval"
        )
    ]
    dataset_legend = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor=color_map[ds],
            markersize=10,
            label=f"{smart_title(ds)}",
        )
        for ds in dataset_unique
    ]
    # Place legends outside of the plot area
    fig.legend(
        handles=match_tracking_legend,
        loc="upper right",
        title="Match Tracking",
        fontsize=14,
        ncols=2,
        bbox_to_anchor=(0.91, 0.98),
        labelspacing=0.615,
        title_fontsize=14,
    )
    fig.legend(
        handles=dataset_legend,
        loc="upper center",
        title="Dataset",
        fontsize=14,
        ncols=7,
        bbox_to_anchor=(0.44, 0.98),
        title_fontsize=14,
    )
    fig.subplots_adjust(
        hspace=0.4
    )  # Adjust space on the right to make room for legends
    plt.savefig("figures/combined_plot.png")


def tick_to_string(tick):
    if abs(tick) > 1e-1:
        return "-∞" if tick < 0 else "∞"
    if abs(tick) > 1e-6:
        superscript_map = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

        exponent = int(np.log10(abs(tick)))  # Get the exponent
        base = 10 if tick > 0 else -10  # Handle negative values
        exponent_str = str(exponent).translate(
            superscript_map
        )  # Convert to superscript
        return f"{base}{exponent_str}"

    return "0" if tick == 0.0 else ""


def plot_epsilon_behavior_bounds(df):
    mt_df = df[df["match_tracking"].isin(["MT-", "MT+", "MT0"])]

    # Create a figure for this match_tracking type
    unique_models = mt_df["model"].unique()
    num_models = len(unique_models)
    fig, axs = plt.subplots(num_models, 3, figsize=(30, 4 * num_models), sharex=True)

    for idx1, model in enumerate(unique_models):
        mt_model_df = mt_df[(mt_df["model"] == model)]
        mt_model_df = mt_model_df.explode(["accuracy", "clusters", "time"]).reset_index(
            drop=True
        )
        for idx2, target in enumerate(["accuracy", "clusters", "time"]):
            target_df = mt_model_df[
                ["model", "dataset", "match_tracking", "epsilon", target]
            ]

            def normalize_group(group):
                base_target = group[group["match_tracking"] == "MT0"][target].mean()
                group[target] = group[target] / base_target
                return group

            # Apply the function to each group of the same 'dataset'
            target_df = (
                target_df.groupby("dataset")
                .apply(normalize_group)
                .reset_index(drop=True)
            )
            target_df.loc[target_df["match_tracking"] == "MT-", "epsilon"] = -target_df[
                "epsilon"
            ]
            target_df.loc[np.isinf(target_df["epsilon"]), "epsilon"] = np.sign(
                target_df["epsilon"]
            )

            z_score = stats.norm.ppf(0.975)
            epsilon_df = target_df.groupby("epsilon", as_index=False).agg(
                target_mean=(target, "mean"),
                target_std=(target, "std"),
                target_count=(target, "count"),
            )

            epsilon_df["target_margin"] = z_score * (
                epsilon_df["target_std"] / np.sqrt(epsilon_df["target_count"])
            )

            # Calculate confidence bounds
            epsilon_df["target_lower"] = (
                epsilon_df["target_mean"] - epsilon_df["target_margin"]
            )
            epsilon_df["target_upper"] = (
                epsilon_df["target_mean"] + epsilon_df["target_margin"]
            )

            epsilon_df["epsilon"] = pd.to_numeric(
                epsilon_df["epsilon"], errors="coerce"
            )
            epsilon_df["target_lower"] = pd.to_numeric(
                epsilon_df["target_lower"], errors="coerce"
            )
            epsilon_df["target_upper"] = pd.to_numeric(
                epsilon_df["target_upper"], errors="coerce"
            )

            ax = axs[idx1, idx2]
            ax.plot(
                epsilon_df["epsilon"],
                epsilon_df["target_mean"],
                label="Average",
                linestyle="--",
                color="black",
                marker="x",
            )
            ax.fill_between(
                epsilon_df["epsilon"],
                epsilon_df["target_lower"],
                epsilon_df["target_upper"],
                color="lightblue",
                alpha=0.5,
                label="95% Confidence Interval",
            )

            # Set axis to log scale
            ax.set_xscale("custom_symlog", linthresh=1e-6)
            if target in ["clusters", "time"] and model in [
                "GaussianART",
                "FuzzyART",
                "HypersphereART",
            ]:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            # Display the y-axis as percentages
            # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
            ax.set_title(f"{model.replace('ART', ' ARTMAP')}", fontsize=18)
            if target == "clusters":
                target = "category"
            ax.set_ylabel(f"{target.title()} Ratio\n(Relative to MT0)", fontsize=14)
            ax.set_xlabel("ϵ", fontsize=14)
            xticks = [
                -1.0,
                -1e-1,
                -1e-2,
                -1e-3,
                -1e-4,
                -1e-5,
                -1e-6,
                0.0,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1.0,
            ]
            ax.set_xticks(xticks)
            ax.set_xticklabels([tick_to_string(tick) for tick in xticks], fontsize=12)
            ax.tick_params(axis="y", labelsize=12)

            minor_ticks = [
                -9e-1,
                -8e-1,
                -7e-1,
                -6e-1,
                -5e-1,
                -4e-1,
                -3e-1,
                -2e-1,
                -9e-2,
                -8e-2,
                -7e-2,
                -6e-2,
                -5e-2,
                -4e-2,
                -3e-2,
                -2e-2,
                -9e-3,
                -8e-3,
                -7e-3,
                -6e-3,
                -5e-3,
                -4e-3,
                -3e-3,
                -2e-3,
                -9e-4,
                -8e-4,
                -7e-4,
                -6e-4,
                -5e-4,
                -4e-4,
                -3e-4,
                -2e-4,
                -9e-5,
                -8e-5,
                -7e-5,
                -6e-5,
                -5e-5,
                -4e-5,
                -3e-5,
                -2e-5,
                -9e-6,
                -8e-6,
                -7e-6,
                -6e-6,
                -5e-6,
                -4e-6,
                -3e-6,
                -2e-6,
                2e-6,
                3e-6,
                4e-6,
                5e-6,
                6e-6,
                7e-6,
                8e-6,
                9e-6,
                2e-5,
                3e-5,
                4e-5,
                5e-5,
                6e-5,
                7e-5,
                8e-5,
                9e-5,
                2e-4,
                3e-4,
                4e-4,
                5e-4,
                6e-4,
                7e-4,
                8e-4,
                9e-4,
                2e-3,
                3e-3,
                4e-3,
                5e-3,
                6e-3,
                7e-3,
                8e-3,
                9e-3,
                2e-2,
                3e-2,
                4e-2,
                5e-2,
                6e-2,
                7e-2,
                8e-2,
                9e-2,
                2e-1,
                3e-1,
                4e-1,
                5e-1,
                6e-1,
                7e-1,
                8e-1,
                9e-1,
            ]

            ax.set_xticks(minor_ticks, minor=True)
            ax.grid(True)

    dataset_legend = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            label="Mean",
            marker="x",
            markersize=8,
        ),
        patches.Patch(color="lightblue", label="95% Confidence Interval"),
    ]
    # Place legends outside of the plot area
    fig.legend(
        handles=dataset_legend,
        loc="upper center",
        fontsize=18,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(
        hspace=0.4
    )  # Adjust space on the right to make room for legends

    plt.savefig(f"figures/epsilon_behavior_bounds.png")


def plot_simple(df):
    # Define the unique markers for match_tracking methods (assuming 5 unique values)
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
    ]  # Circle, Square, Diamond, Triangle Up, Triangle Down
    match_tracking_unique = df["match_tracking"].unique()

    dataset_unique = df["dataset"].unique()
    colors = generate_ciecam02_colors(len(dataset_unique))

    # Create a mapping of markers to match_tracking and colors to datasets
    marker_map = {
        match_tracking_unique[i]: markers[i] for i in range(len(match_tracking_unique))
    }
    color_map = {dataset_unique[i]: colors[i] for i in range(len(dataset_unique))}

    # Prepare subplots, with one plot per model stacked vertically
    unique_models = df["model"].unique()
    num_models = len(unique_models)
    fig, axes = plt.subplots(
        1, 1, figsize=(30, 6 * num_models), constrained_layout=False
    )
    ax = axes
    # Loop over each unique model to create a subplot for each
    for idx, model in enumerate(unique_models):
        model_data = df[df["model"] == model]
        for _, row in model_data.iterrows():
            ax.scatter(
                row["mean_time"],
                row["mean_accuracy"],
                color=(
                    color_map[row["dataset"]]
                    if model == "HullART"
                    else wash_out_color(color_map[row["dataset"]])
                ),
                marker=marker_map[row["match_tracking"]],
                label=f"Dataset: {row['dataset']}, Match Tracking: {row['match_tracking']}",
                s=(100 if model == "HullART" else 20),  # Size of markers,
                edgecolors="black",
            )

    # Set titles and labels
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Accuracy vs Time", fontsize=18)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xscale("log")  # Apply log scale to the time axis
    ax.grid(True)
    plt.savefig(f"figures/simple_accuracy_vs_time.png")


def plot_best_metric(df, metric="mean_accuracy"):
    """Plots the best mean_accuracy for each model-dataset pair as a grouped bar chart.
    Adds a separate grouped bar chart for the mean accuracy of each model across all
    datasets. Dataset labels are tilted 45 degrees, and the two charts are visually
    distinct.

    Args:
    - df (pd.DataFrame): DataFrame containing columns ['model', 'dataset', 'mean_accuracy'].

    """

    # Find the best accuracy for each model-dataset pair
    best_metrics = df.groupby(["model", "dataset"])[metric].max().reset_index()

    # Pivot the data to prepare for bar plotting
    pivoted = best_metrics.pivot(index="dataset", columns="model", values=metric)

    # Compute mean accuracy for each model across all datasets
    mean_accuracies = pivoted.mean(axis=0)

    # Plotting
    datasets = pivoted.index
    models = pivoted.columns

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Grouped bar chart for best accuracies by dataset and model
    bar_width = 0.2
    x = range(len(datasets))  # X positions for datasets
    colors = generate_ciecam02_colors(len(models))

    for i, model in enumerate(models):
        accuracies = pivoted[model].values
        bar_positions = [pos + i * bar_width for pos in x]

        ax1.bar(bar_positions, accuracies, bar_width, label=model, color=colors[i])

    # Adjust x-ticks and labels
    ax1.set_xticks([pos + (len(models) * bar_width) / 2 - bar_width / 2 for pos in x])
    ax1.set_xticklabels(datasets, rotation=45, ha="right")
    ax1.set_xlabel("Dataset", fontsize=14)
    ax1.set_ylabel(f"Best {metric}", fontsize=14)
    ax1.set_title(f"Best {metric} by Model and Dataset", fontsize=16)
    ax1.legend(title="Models", fontsize=10, title_fontsize=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Mean accuracy bar chart
    ax2.bar(models, mean_accuracies, color=colors)
    ax2.set_title(f"Mean {metric} Across All Datasets", fontsize=16)
    ax2.set_ylabel(f"Mean {metric}", fontsize=14)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a dividing line
    fig.subplots_adjust(wspace=0.3)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(f"figures/best_{metric}_bar_chart.png")
    plt.show()


if __name__ == "__main__":
    results = pd.read_parquet("hull_art.parquet")
    results = squash_folds(results)
    results = results[
        ~results["dataset"].isin(["iris", "land mines", "students", "wine"])
    ]
    # plot_epsilon_behavior_bounds(results)

    filtered_results = filter_data(results)
    plot_best_metric(filtered_results, "mean_accuracy")
    plot_best_metric(filtered_results, "mean_precision")
    plot_best_metric(filtered_results, "mean_recall")
    plot_best_metric(filtered_results, "mean_f1")

    # plt.show()
