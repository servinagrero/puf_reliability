from pathlib import Path

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_unique_error_rows(frame: pl.DataFrame, output_path: Path):
    data = (
        frame.select(["family", "unique_error_rows", "n_observations"])
        .with_columns(
            (pl.col("unique_error_rows") / pl.col("n_observations")).alias(
                "unique_error_rows_norm"
            )
        )
        .sort("unique_error_rows_norm", descending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["family"], data["unique_error_rows_norm"])
    ax.set_xlabel("Unique number of row errors (%)")
    ax.set(xlim=(0.7, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_effective_independent_errors(frame: pl.DataFrame, output_path: Path):
    data = (
        frame.select(["family", "effective_independent_error_bits", "n_bits"])
        .with_columns(
            (pl.col("effective_independent_error_bits") / pl.col("n_bits")).alias(
                "effective_independent_error_bits_norm"
            )
        )
        .sort("effective_independent_error_bits_norm", descending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["family"], data["effective_independent_error_bits_norm"])
    ax.set_xlabel("Number of effective independent bits (%)")
    ax.set(xlim=(0.0, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_flip_rate(frame: pl.DataFrame, output_path: Path):
    data = frame.select(
        [
            "family",
            "flip_rate_min",
            "flip_rate_mean",
            "flip_rate_max",
            "per_device_flip_rate_min",
            "per_device_flip_rate_mean",
            "per_device_flip_rate_max",
        ]
    ).sort("flip_rate_mean", descending=True)
    families = data["family"].to_list()
    x_positions = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, family in enumerate(families):
        df_family = data.filter(pl.col("family") == family)
        ax.hlines(
            y=df_family["flip_rate_mean"],
            xmin=i - width / 2,
            xmax=i,
            color="tab:orange",
            linestyle="-",
            linewidth=1,
        )
        ax.fill_between(
            [i - width / 2, i],
            df_family["flip_rate_min"],
            df_family["flip_rate_max"],
            color="tab:orange",
            alpha=0.25,
            label="Global" if i == 0 else "",
        )

        ax.hlines(
            y=df_family["per_device_flip_rate_mean"],
            xmin=i,
            xmax=i + width / 2,
            color="tab:blue",
            linestyle="-",
            linewidth=1,
        )
        ax.fill_between(
            [i, i + width / 2],
            df_family["per_device_flip_rate_min"],
            df_family["per_device_flip_rate_max"],
            color="tab:blue",
            alpha=0.25,
            label="Per device" if i == 0 else "",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(families, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean flip error rate")
    ax.set(ylim=(-0.01, 1.01))
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_block_failure(frame: pl.DataFrame, output_path: Path):
    empirical = frame["empirical_block_failure"].to_numpy()
    independent = frame["independent_block_failure"].to_numpy()
    order = np.argsort(np.abs(empirical - independent))
    families = np.asarray(frame["family"].to_list())[order]
    y = np.arange(len(families))
    bar_width = 0.3

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        y - bar_width / 2,
        empirical[order],
        height=bar_width,
        label="Empirical correlated",
    )
    ax.barh(
        y + bar_width / 2,
        independent[order],
        height=bar_width,
        label="Independent baseline",
    )
    ax.set_yticks(y, families)
    ax.set_xlabel("Block failure probability")
    ax.set(xlim=(0, 1.01))
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_block_failure_curves(frame: pl.DataFrame, output_path: Path):
    families = frame["family"].unique().sort().to_list()
    if not families:
        return
    correction_capacity = frame["correction_capacity"].unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("tab10_r", len(families))
    family_lines = []
    style_lines = None

    for i, family in enumerate(families):
        data = frame.filter(pl.col("family") == family).sort("correction_capacity")
        color = cmap(i)

        (line_empirical,) = ax.plot(
            correction_capacity,
            data["empirical_block_failure"].to_numpy(),
            linestyle="-",
            alpha=0.8,
            color=color,
        )

        (line_independent,) = ax.plot(
            correction_capacity,
            data["independent_block_failure"].to_numpy(),
            linestyle="--",
            alpha=0.8,
            color=color,
        )

        if style_lines is None:
            style_lines = [line_empirical, line_independent]

        family_lines.append(
            plt.Line2D([0], [0], color=color, linewidth=3, label=family)
        )

    ax.set_xlabel("Correction capacity")
    ax.set_ylabel("P(block failure)")
    ax.set(xlim=(-1, correction_capacity[-1] + 1), ylim=(-0.02, 1.02))

    style_legend = ax.legend(
        handles=style_lines,
        labels=["Empirical", "Independent"],
        loc="lower left",
        framealpha=0.8,
        fontsize=10,
        bbox_to_anchor=(0.01, 0.01),
    )
    ax.add_artist(style_legend)

    fig.legend(
        handles=family_lines,
        labels=families,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=10,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mi_heatmaps(qc_dir: Path, figures_dir: Path):
    for path in sorted(qc_dir.glob("*.pairwise_mi.npy")):
        family = path.name.removesuffix(".pairwise_mi.npy")
        mi = np.load(path)
        if mi.size == 0:
            print(f"Mutual Information matrix {path} is empty")
            continue
        limit = np.nanmax(mi)
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mi, cmap="magma", vmin=0.0, vmax=limit, aspect="auto")
        fig.colorbar(im, ax=ax, label="Pairwise error MI")
        ax.set_title(family)
        ax.set_xlabel("Bit index")
        ax.set_ylabel("Bit index")
        ax.xaxis.set_major_locator(MultipleLocator(32))
        ax.yaxis.set_major_locator(MultipleLocator(32))
        fig.tight_layout()
        fig.savefig(figures_dir / f"{family}_pairwise_mi.png", dpi=300)
        plt.close(fig)


def plot_validation_delta(frame: pl.DataFrame, output_path: Path):
    pivot = frame.pivot(
        values="log_loss",
        index="family",
        on="model",
        aggregate_function="first",
    )
    if not {"independent", "error_graph"}.issubset(set(pivot.columns)):
        print(
            f"Columns ['independent', 'error_graph'] not found in data frame. Found {set(pivot.columns)}"
        )
        return
    pivot = pivot.with_columns(
        (pl.col("independent") - pl.col("error_graph")).alias("log_loss_gain")
    ).sort("log_loss_gain")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = np.where(pivot["log_loss_gain"] > 0, "tab:blue", "tab:red")
    ax.barh(pivot["family"], pivot["log_loss_gain"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Held-out log-loss gain over independent baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
