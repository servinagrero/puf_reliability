from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pufre.graph import ROLE_VALUES
from pufre.plotting.graph import ROLE_COLORS

ROLES = ROLE_VALUES


def plot_diff_distribution(
    bit_table: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    values = _finite(baseline["diff_mean"].to_numpy())
    selected = baseline.filter(pl.col("selection").is_not_null())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(40, max(5, int(np.sqrt(values.size)))), alpha=0.75)
    for selection in selected["selection"].unique().sort().to_list():
        subset = selected.filter(pl.col("selection") == selection)
        ax.scatter(
            subset["diff_mean"],
            np.zeros(subset.height),
            s=14,
            alpha=0.75,
            label=selection,
        )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Mean physical difference $f_1 - f_2$")
    ax.set_ylabel("Count")
    if selected.height:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_selection_diff_distributions(
    bit_table: pl.DataFrame,
    selection_table: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
    selection_size: int,
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    selected = selection_table.filter(pl.col("selection_size") == selection_size)
    values = _finite(baseline["diff_mean"].to_numpy())
    bins = np.histogram_bin_edges(
        values, bins=min(40, max(5, int(np.sqrt(values.size))))
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color="0.85", edgecolor="0.5", label="all bits")
    for selection in selected["selection"].unique().sort().to_list():
        bit_ids = selected.filter(pl.col("selection") == selection)["bit_id"].to_list()
        subset = baseline.filter(pl.col("bit_id").is_in(bit_ids))
        ax.hist(
            _finite(subset["diff_mean"].to_numpy()),
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=selection,
        )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Mean physical difference $f_1 - f_2$")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_space_by_margin(
    bit_table: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
    margin_column: str = "normalized_abs_diff_mean",
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        baseline["stability"],
        baseline["upc"],
        c=baseline[margin_column],
        s=18,
        alpha=0.85,
        cmap="viridis",
    )
    ax.set_xlabel("Stability")
    ax.set_ylabel("UPC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(_column_label(margin_column))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_phase_space_by_role(
    bit_table: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    fig, ax = plt.subplots(figsize=(6, 5))
    for role in ROLES:
        subset = baseline.filter(pl.col("graph_role") == role)
        if subset.is_empty():
            continue
        ax.scatter(
            subset["stability"],
            subset["upc"],
            s=18,
            alpha=0.85,
            color=ROLE_COLORS[role],
            label=role,
        )
    ax.set_xlabel("Stability")
    ax.set_ylabel("UPC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_by_role(
    bit_table: pl.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    values = [
        _finite(baseline.filter(pl.col("graph_role") == role)[column].to_numpy())
        for role in ROLES
    ]
    labels = [role for role, value in zip(ROLES, values) if value.size]
    values = [value for value in values if value.size]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Baseline graph role")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hub_authority_by_metric(
    bit_table: pl.DataFrame,
    column: str,
    color_label: str,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        baseline["out_strength"],
        baseline["in_strength"],
        c=baseline[column],
        s=18,
        alpha=0.85,
        cmap="viridis",
    )
    ax.set_xlabel("Hub / source strength")
    ax.set_ylabel("Authority / sink strength")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_temperature_degradation_by_margin(
    bit_table: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
    margin_column: str = "baseline_normalized_abs_diff_mean",
    transform: str = "robust_z",
):
    data = bit_table.filter(pl.col("temperature") != baseline_temperature)
    if data.is_empty():
        return
    x_values = np.asarray(data[margin_column], dtype=float)
    x_plot, x_label = _transform_values(
        x_values, _column_label(margin_column), transform
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        x_plot,
        data["delta_flip_rate"],
        c=data["temperature"],
        s=14,
        alpha=0.75,
        cmap="plasma",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Delta flip rate from room temperature")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Temperature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_selection_summary(
    summary: pl.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
):
    value_column = f"{column}_mean"
    fig, ax = plt.subplots(figsize=(8, 4))
    for selection in summary["selection"].unique().sort().to_list():
        data = summary.filter(pl.col("selection") == selection).sort("temperature")
        ax.plot(
            data["temperature"],
            data[value_column],
            marker="o",
            label=selection,
        )
    ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_selection_summary_by_size(
    summary: pl.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
):
    value_column = f"{column}_mean"
    if "selection_size" not in summary.columns:
        summary = add_selection_parts(summary)
    sizes = summary["selection_size"].unique().sort().to_list()
    fig, axes = plt.subplots(
        len(sizes),
        1,
        figsize=(8, max(3, 2.7 * len(sizes))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        size_frame = summary.filter(pl.col("selection_size") == size)
        for method in size_frame["selection_method"].unique().sort().to_list():
            data = size_frame.filter(pl.col("selection_method") == method).sort(
                "temperature"
            )
            ax.plot(data["temperature"], data[value_column], marker="o", label=method)
        ax.set_title(f"k = {size}")
        ax.set_ylabel(ylabel)
        # ax.set(ylim=(None, 1.02))
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Temperature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_selection_block_failure(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    data = frame.filter(pl.col("temperature") == baseline_temperature)
    if data.is_empty():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for selection in data["selection"].unique().sort().to_list():
        selected = data.filter(pl.col("selection") == selection).sort(
            "correction_capacity"
        )
        ax.plot(
            selected["correction_capacity"],
            selected["empirical_failure_probability"],
            marker="o",
            label=f"{selection} empirical",
        )
        ax.plot(
            selected["correction_capacity"],
            selected["independent_failure_probability"],
            linestyle="--",
            label=f"{selection} independent",
        )
    ax.set_xlabel("Correction capacity")
    ax.set_ylabel("Block failure probability")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_selection_block_failure_by_size(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    baseline_temperature: int,
):
    data = frame.filter(pl.col("temperature") == baseline_temperature)
    if data.is_empty():
        return
    if "selection_size" not in data.columns:
        data = add_selection_parts(data)
    sizes = data["selection_size"].unique().sort().to_list()
    fig, axes = plt.subplots(
        len(sizes),
        1,
        figsize=(8, max(3, 2.8 * len(sizes))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, size in zip(axes, sizes):
        size_frame = data.filter(pl.col("selection_size") == size)
        for method in size_frame["selection_method"].unique().sort().to_list():
            selected = size_frame.filter(pl.col("selection_method") == method).sort(
                "correction_capacity"
            )
            ax.plot(
                selected["correction_capacity"],
                selected["empirical_failure_probability"],
                marker="o",
                label=f"{method} empirical",
            )
            ax.plot(
                selected["correction_capacity"],
                selected["independent_failure_probability"],
                linestyle="--",
                label=f"{method} independent",
            )
        ax.set_title(f"k = {size}")
        ax.set_ylabel("Failure probability")
        max_correction_capacity = selected["correction_capacity"].max()
        ax.set(xlim=(-1, max_correction_capacity + 1), ylim=(-0.05, 1.05))
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Correction capacity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def add_selection_labels(bit_table: pl.DataFrame, selections: dict[str, np.ndarray]):
    labels_by_bit = {}
    for selection, bit_ids in selections.items():
        for bit_id in bit_ids:
            labels_by_bit.setdefault(int(bit_id), []).append(selection)
    labels = [
        ",".join(labels_by_bit.get(int(bit_id), [])) or None
        for bit_id in bit_table["bit_id"].to_list()
    ]
    return bit_table.with_columns(pl.Series("selection", labels))


def selection_membership_frame(selections: dict[str, np.ndarray]):
    rows = []
    for selection, bit_ids in selections.items():
        method, size = parse_selection_name(selection)
        for bit_id in bit_ids:
            rows.append(
                {
                    "selection": selection,
                    "selection_method": method,
                    "selection_size": size,
                    "bit_id": int(bit_id),
                }
            )
    return pl.DataFrame(rows)


def add_selection_parts(frame: pl.DataFrame):
    parsed = [parse_selection_name(selection) for selection in frame["selection"]]
    return frame.with_columns(
        pl.Series("selection_method", [method for method, _ in parsed]),
        pl.Series("selection_size", [size for _, size in parsed]),
    )


def parse_selection_name(selection: str):
    if "_k" not in selection:
        return selection, None
    method, size = selection.rsplit("_k", 1)
    return method, int(size)


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _transform_values(values, label: str, transform: str):
    values = np.asarray(values, dtype=float)
    if transform == "none":
        return values, label
    if transform == "log1p":
        return np.log1p(np.clip(values, 0.0, None)), f"log1p({label})"
    if transform == "robust_z":
        median = np.nanmedian(values)
        q25, q75 = np.nanquantile(values, [0.25, 0.75])
        scale = q75 - q25
        if not np.isfinite(scale) or scale == 0.0:
            scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        return (values - median) / scale, f"Robust z-score of {label}"
    raise ValueError(f"Unknown transform: {transform}")


def _column_label(column: str):
    labels = {
        "abs_diff_mean": "Mean $|f_1 - f_2|$",
        "baseline_abs_diff_mean": "Room-temperature mean $|f_1 - f_2|$",
        "normalized_abs_diff_mean": "Mean normalized $|f_1 - f_2|$",
        "baseline_normalized_abs_diff_mean": (
            "Room-temperature mean normalized $|f_1 - f_2|$"
        ),
        "reliable_entropy": "Reliable Entropy",
    }
    return labels.get(column, column)
