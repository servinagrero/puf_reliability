from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pufre import DATA_DIR, FIGURES_DIR, PROJECT_DIR, logger
from pufre.analysis.temperature import load_differential_hdf5_artifact
from pufre.error_metrics import holm_adjusted_p_values
from pufre.graph import ROLE_VALUES
from pufre.plotting.graph import ROLE_COLORS

INPUT_DIR = DATA_DIR / "ro_temperature"
ARTIFACT_PATH = DATA_DIR / "raw" / "RO_PUF.h5"
FIGURE_DIR = FIGURES_DIR / "host" / "ro_temperature"
TABLE_DIR = PROJECT_DIR / "tables" / "host" / "ro_temperature"
ROOM_TEMPERATURE = 25

try:
    plt.style.use("publications")
except OSError:
    pass


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    qc = pl.read_csv(INPUT_DIR / "temperature_qc.csv")
    independence = pl.read_csv(INPUT_DIR / "temperature_independence.csv")
    independence = independence.with_columns(
        pl.Series(
            "p_holm",
            holm_adjusted_p_values(independence["p_value"]),
        )
    )
    block = pl.read_csv(INPUT_DIR / "temperature_block_failure.csv")
    graph = pl.read_csv(INPUT_DIR / "temperature_graph_summary.csv")
    roles = pl.read_csv(INPUT_DIR / "ro_temperature_role_summary.csv")
    transitions = pl.read_csv(
        INPUT_DIR / "ro_temperature_role_transitions.csv"
    )
    transfer = pl.read_csv(INPUT_DIR / "ro_temperature_model_transfer.csv")
    graph_distribution = pl.read_csv(
        INPUT_DIR / "temperature_graph_bootstrap_distribution.csv"
    )
    graph_distances = pl.read_csv(
        INPUT_DIR / "temperature_graph_distance_distribution.csv"
    )
    error_counts = load_temperature_error_counts()

    plot_temperature_error_count_ecdf(
        error_counts,
        FIGURE_DIR / "temperature_error_count_ecdf.png",
    )
    plot_dependence_effects(
        independence,
        FIGURE_DIR / "temperature_dependence_effects.png",
    )
    plot_qc_effective_dimensions(
        qc,
        FIGURE_DIR / "temperature_effective_dimensions.png",
    )
    plot_block_failure(
        block,
        FIGURE_DIR / "temperature_block_failure.png",
    )
    plot_stable_edges(
        graph,
        FIGURE_DIR / "temperature_stable_edges.png",
    )
    plot_fixed_role_distribution(
        roles,
        FIGURE_DIR / "temperature_fixed_roles.png",
    )
    plot_role_change(
        transitions,
        FIGURE_DIR / "temperature_role_change.png",
    )
    plot_room_model_transfer(
        transfer,
        FIGURE_DIR / "temperature_room_model_transfer.png",
    )
    plot_temperature_graph_null_effects(
        graph_distribution,
        FIGURE_DIR / "temperature_graph_null_effects.png",
    )
    plot_null_corrected_graph_distances(
        graph_distances,
        FIGURE_DIR / "temperature_null_corrected_graph_distances.png",
    )
    write_temperature_summary(qc, independence, block)
    logger.info("Wrote HOST RO-temperature assets to %s", FIGURE_DIR)


def load_temperature_error_counts():
    """Load exact per-readout error counts for every temperature."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing temperature artifact {ARTIFACT_PATH}")

    artifact = load_differential_hdf5_artifact(
        ARTIFACT_PATH,
        room_temperature=ROOM_TEMPERATURE,
    )
    reference = artifact["reference"]
    counts = {}
    for temperature, responses in artifact["responses"].items():
        if responses.shape[0] != reference.shape[0]:
            raise ValueError(
                f"T={temperature} has {responses.shape[0]} devices; "
                f"expected {reference.shape[0]}"
            )
        if responses.shape[2] != reference.shape[1]:
            raise ValueError(
                f"T={temperature} has {responses.shape[2]} bits; "
                f"expected {reference.shape[1]}"
            )
        errors = responses != reference[:, np.newaxis, :]
        counts[temperature] = errors.sum(axis=2).reshape(-1).astype(int)
    return counts


def plot_temperature_error_count_ecdf(error_counts, output_path):
    """Plot the empirical error-count ECDF at every measured temperature."""
    temperatures = sorted(error_counts)
    normalization = mpl.colors.Normalize(
        vmin=min(temperatures),
        vmax=max(temperatures),
    )
    colormap = mpl.colors.LinearSegmentedColormap.from_list(
        "temperature_inferno",
        mpl.colormaps["inferno"](np.linspace(0.08, 0.78, 256)),
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for temperature in temperatures:
        counts = np.sort(error_counts[temperature])
        cumulative_probability = (
            np.arange(1, counts.size + 1, dtype=float) / counts.size
        )
        ax.step(
            counts,
            cumulative_probability,
            where="post",
            color=colormap(normalization(temperature)),
            linewidth=1.8 if temperature == ROOM_TEMPERATURE else 1.1,
            alpha=1.0 if temperature == ROOM_TEMPERATURE else 0.8,
        )

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=normalization, cmap=colormap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label(r"Temperature ($^\circ$C)")
    ax.set_xlabel("Bitflips per readout")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_dependence_effects(frame, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.5), sharex=True)
    for ax, metric, label in [
        (
            axes[0],
            "correlation_energy",
            "Correlation-energy ratio",
        ),
        (
            axes[1],
            "error_count_variance",
            "Error-count variance ratio",
        ),
    ]:
        data = frame.filter(pl.col("metric") == metric).sort("temperature")
        ax.plot(
            data["temperature"],
            data["observed_to_null_ratio"],
            color="tab:blue",
        )
        significant = data["p_holm"].to_numpy() < 0.05
        ax.scatter(
            data["temperature"],
            data["observed_to_null_ratio"],
            facecolors=[
                "tab:blue" if value else "white"
                for value in significant
            ],
            edgecolors="tab:blue",
            zorder=3,
        )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Temperature (C)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_qc_effective_dimensions(frame, output_path):
    data = frame.sort("temperature")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(
        data["temperature"],
        data["effective_independent_error_bits"] / data["n_bits"],
        marker="o",
        label="Effective bits / B",
    )
    ax.plot(
        data["temperature"],
        data["unique_error_ratio"],
        marker="s",
        label="Observed / expected unique rows",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Relative value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_block_failure(frame, output_path):
    temperatures = [
        temperature
        for temperature in [25, 80, 120, 150, 180]
        if temperature in frame["temperature"].unique().to_list()
    ]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(temperatures)))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for temperature, color in zip(temperatures, colors):
        data = frame.filter(pl.col("temperature") == temperature).sort(
            "correction_capacity"
        )
        ax.plot(
            data["correction_capacity"],
            data["empirical_block_failure"],
            color=color,
            label=f"{temperature} C empirical",
        )
        ax.plot(
            data["correction_capacity"],
            data["independent_block_failure"],
            color=color,
            linestyle="--",
        )
    ax.set_xlabel("Correction capacity")
    ax.set_ylabel("Block failure probability")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_stable_edges(frame, output_path):
    data = frame.sort("temperature")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.3), sharex=True)
    axes[0].plot(
        data["temperature"],
        data["observed_stable_directed_edges"],
        marker="o",
        label="Observed",
    )
    axes[0].plot(
        data["temperature"],
        data["null_stable_directed_edges"],
        marker="s",
        linestyle="--",
        label="Device-preserving null",
    )
    axes[0].set_ylabel("Stable directed edges")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        data["temperature"],
        data["stable_edge_ratio"],
        marker="o",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_ylabel("Observed / null")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fixed_role_distribution(frame, output_path):
    data = frame.filter(
        pl.col("role_definition") == "fixed_baseline_threshold"
    )
    temperatures = data["temperature"].unique().sort().to_list()
    positions = np.arange(len(temperatures))
    bottom = np.zeros(len(temperatures))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.8, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
        layout="constrained",
    )
    for role in ROLE_VALUES:
        observed = role_fraction_series(
            data,
            temperatures,
            role,
            graph_source="observed",
        )
        null = role_fraction_series(
            data,
            temperatures,
            role,
            graph_source="null",
        )
        axes[0].bar(
            positions,
            observed,
            bottom=bottom,
            width=0.85,
            color=ROLE_COLORS[role],
            label=role,
        )
        bottom += observed
        axes[1].plot(
            positions,
            observed - null,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=ROLE_COLORS[role],
            label=role,
        )
    axes[0].set_ylabel("Observed role fraction")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(
        fontsize=7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, [str(value) for value in temperatures])
    axes[1].set_xlabel(r"Temperature ($^\circ$C)")
    axes[1].set_ylabel("Observed $-$ null")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_role_change(frame, output_path):
    data = frame.filter(pl.col("metric") == "role_changed_fraction")
    observed = data.filter(pl.col("graph_source") == "observed").sort(
        "temperature"
    )
    null = data.filter(pl.col("graph_source") == "null").sort("temperature")

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.3), sharex=True)
    axes[0].plot(
        observed["temperature"],
        observed["value"],
        marker="o",
        label="Observed",
    )
    axes[0].plot(
        null["temperature"],
        null["value"],
        marker="s",
        linestyle="--",
        label="Device-preserving null",
    )
    axes[0].set_ylabel("Fraction changing role")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        observed["temperature"],
        observed["value"] - null["value"],
        marker="o",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_ylabel("Observed - null")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def role_fraction_series(frame, temperatures, role, *, graph_source):
    values = []
    for temperature in temperatures:
        row = frame.filter(
            (pl.col("temperature") == temperature)
            & (pl.col("graph_role") == role)
            & (pl.col("graph_source") == graph_source)
        )
        values.append(float(row["fraction_bits"][0]) if row.height else 0.0)
    return np.asarray(values)


def plot_room_model_transfer(frame, output_path):
    data = frame.filter(
        pl.col("source_temperature") == ROOM_TEMPERATURE
    ).sort("target_temperature")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(
        data["target_temperature"],
        data["log_loss_gain"],
        marker="o",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Evaluation temperature (C)")
    ax.set_ylabel("Log-loss gain over independent model")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_temperature_graph_null_effects(frame, output_path):
    metrics = [
        ("directed_edges", "Edge-count effect"),
        ("total_abs_weight", "Weight effect"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.3), sharex=True)
    temperatures = frame["temperature"].unique().sort().to_list()
    for ax, (metric, label) in zip(axes, metrics):
        medians, q25, q75 = [], [], []
        for temperature in temperatures:
            selected = frame.filter(pl.col("temperature") == temperature)
            observed = selected.filter(
                pl.col("graph_source") == "observed"
            ).sort("bootstrap")[metric].to_numpy()
            null = selected.filter(
                pl.col("graph_source") == "null"
            ).sort("bootstrap")[metric].to_numpy()
            scale = np.std(null, ddof=1)
            effects = (
                (observed - null) / scale
                if scale > 0
                else np.full(observed.shape, np.nan)
            )
            medians.append(np.nanmedian(effects))
            q25.append(np.nanquantile(effects, 0.25))
            q75.append(np.nanquantile(effects, 0.75))
        ax.plot(temperatures, medians, marker="o")
        ax.fill_between(temperatures, q25, q75, alpha=0.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Temperature (C)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_null_corrected_graph_distances(frame, output_path):
    metrics = [
        ("normalized_frobenius_distance", "Frobenius distance"),
        ("cosine_distance", "Cosine distance"),
        ("edge_jaccard_distance", "Edge-Jaccard distance"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.2), sharex=True)
    temperatures = frame["temperature"].unique().sort().to_list()
    for ax, (metric, label) in zip(axes, metrics):
        medians, q25, q75 = [], [], []
        for temperature in temperatures:
            selected = frame.filter(pl.col("temperature") == temperature)
            observed = selected.filter(
                pl.col("graph_source") == "observed"
            ).sort("bootstrap")[metric].to_numpy()
            null = selected.filter(
                pl.col("graph_source") == "null"
            ).sort("bootstrap")[metric].to_numpy()
            corrected = observed - null
            medians.append(np.nanmedian(corrected))
            q25.append(np.nanquantile(corrected, 0.25))
            q75.append(np.nanquantile(corrected, 0.75))
        ax.plot(temperatures, medians, marker="o")
        ax.fill_between(temperatures, q25, q75, alpha=0.2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Temperature (C)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_temperature_summary(qc, independence, block):
    correlation = independence.filter(
        pl.col("metric") == "correlation_energy"
    ).select(
        "temperature",
        pl.col("observed_to_null_ratio").alias("correlation_ratio"),
        pl.col("p_holm").alias("correlation_p"),
    )
    count_variance = independence.filter(
        pl.col("metric") == "error_count_variance"
    ).select(
        "temperature",
        pl.col("observed_to_null_ratio").alias("count_ratio"),
        pl.col("p_holm").alias("count_p"),
    )
    maximum_gap = (
        block.with_columns(
            pl.col("failure_gap").abs().alias("absolute_gap")
        )
        .group_by("temperature")
        .agg(pl.max("absolute_gap").alias("maximum_gap"))
    )
    summary = (
        qc.select(
            "temperature",
            "flip_rate_mean",
        )
        .join(correlation, on="temperature")
        .join(count_variance, on="temperature")
        .join(maximum_gap, on="temperature")
        .sort("temperature")
    )
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{rrrrr}",
        r"\toprule",
        (
            r"$T$ ($^\circ$C) & BER & "
            r"$R_{\mathrm{corr}}\;(p_{\mathrm{H}})$ & "
            r"$R_{\mathrm{count}}\;(p_{\mathrm{H}})$ & "
            r"$\max_\tau|\Delta P_\tau|$ \\"
        ),
        r"\midrule",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"{row['temperature']} & "
            f"{row['flip_rate_mean']:.3f} & "
            f"{row['correlation_ratio']:.2f} "
            f"({row['correlation_p']:.3f}) & "
            f"{row['count_ratio']:.2f} "
            f"({row['count_p']:.3f}) & "
            f"{row['maximum_gap']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"    \end{tabular}",
            (
                r"    \caption{Temperature-sweep diagnostics. Parentheses "
                r"contain Holm-adjusted permutation p-values across all 22 "
                r"temperature-specific tests.}"
            ),
            r"    \label{tab:temperature-summary}",
            r"\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "ro_temperature_summary.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
