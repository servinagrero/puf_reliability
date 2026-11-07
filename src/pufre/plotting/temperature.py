from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pufre.analysis.temperature import ROLES
from pufre.plotting.graph import ROLE_COLORS, role_colors


def plot_role_distribution(summary: pl.DataFrame, output_path: Path):
    temperatures = summary["temperature"].unique().sort().to_list()
    bottom = np.zeros(len(temperatures), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    for role in ROLES:
        data = role_series(summary, role, "fraction_bits", temperatures)
        ax.bar(temperatures, data, bottom=bottom, label=role, width=6)
        bottom += np.nan_to_num(data)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Fraction of bits")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_flip_rate_by_role(summary: pl.DataFrame, output_path: Path):
    temperatures = summary["temperature"].unique().sort().to_list()
    fig, ax = plt.subplots(figsize=(8, 4))
    for role in ROLES:
        mean = role_series(summary, role, "flip_rate_mean", temperatures)
        q25 = role_series(summary, role, "flip_rate_q25", temperatures)
        q75 = role_series(summary, role, "flip_rate_q75", temperatures)
        ax.plot(temperatures, mean, marker="o", label=role)
        ax.fill_between(temperatures, q25, q75, alpha=0.15)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Bit flip rate")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_role_jaccard(transitions: pl.DataFrame, output_path: Path):
    data = transitions.filter(pl.col("metric") == "jaccard")
    fig, ax = plt.subplots(figsize=(8, 4))
    for role in ROLES:
        role_data = data.filter(pl.col("baseline_role") == role).sort("temperature")
        ax.plot(role_data["temperature"], role_data["value"], marker="o", label=role)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Jaccard with baseline role set")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_role_change_fraction(transitions: pl.DataFrame, output_path: Path):
    data = transitions.filter(pl.col("metric") == "role_changed_fraction").sort(
        "temperature"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data["temperature"], data["value"], marker="o")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Fraction of bits changing role")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_graph_drift(drift: pl.DataFrame, output_path: Path, *, top_edge_k: int = 256):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        drift["temperature"],
        drift["cosine_with_baseline"],
        marker="o",
        label="Cosine",
    )
    ax.plot(
        drift["temperature"],
        drift["fixed_strong_edge_jaccard"],
        marker="o",
        label="Strong-edge Jaccard",
    )
    ax.plot(
        drift["temperature"],
        drift["top_edge_overlap"],
        marker="o",
        label=f"Top-{top_edge_k} overlap",
    )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Similarity to baseline")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_transfer_loss(transfer: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for source_temperature in transfer["source_temperature"].unique().sort().to_list():
        data = transfer.filter(pl.col("source_temperature") == source_temperature).sort(
            "target_temperature"
        )
        ax.plot(
            data["target_temperature"],
            data["nodewise_log_loss"],
            marker="o",
            label=f"fit {source_temperature}",
        )
    ax.set_xlabel("Evaluation temperature")
    ax.set_ylabel("Nodewise pseudo-log-loss")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_slope_by_role(
    susceptibility: pl.DataFrame,
    column: str,
    ylabel: str,
    output_path: Path,
):
    values = [
        finite_values(
            susceptibility.filter(pl.col("baseline_graph_role") == role)[
                column
            ].to_numpy()
        )
        for role in ROLES
    ]
    labels = [role for role, value in zip(ROLES, values) if value.size]
    values = [value for value in values if value.size]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Baseline graph role")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_strength_vs_slope(susceptibility: pl.DataFrame, output_path: Path):
    colors = role_colors(susceptibility["baseline_graph_role"].to_list())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        susceptibility["baseline_total_strength"],
        susceptibility["flip_rate_slope"],
        c=colors,
        s=16,
        alpha=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Baseline total graph strength")
    ax.set_ylabel("Flip-rate slope")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_margin_vs_flip_slope(susceptibility: pl.DataFrame, output_path: Path):
    colors = role_colors(susceptibility["baseline_graph_role"].to_list())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        susceptibility["baseline_mean_abs_margin"],
        susceptibility["flip_rate_slope"],
        c=colors,
        s=16,
        alpha=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Baseline mean absolute margin")
    ax.set_ylabel("Flip-rate slope")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def role_series(summary: pl.DataFrame, role: str, column: str, temperatures):
    values = []
    for temperature in temperatures:
        row = summary.filter(
            (pl.col("temperature") == temperature) & (pl.col("graph_role") == role)
        )
        values.append(float(row[column][0]) if row.height else np.nan)
    return np.asarray(values, dtype=float)


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def add_role_legend(ax):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=role)
        for role, color in ROLE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
