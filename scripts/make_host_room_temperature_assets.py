import matplotlib as mpl
import numpy as np
import polars as pl

mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from pufre import DATA_DIR, FIGURES_DIR, PROJECT_DIR, logger
from pufre.datasets import families_by_name
from pufre.error_metrics import holm_adjusted_p_values
from pufre.graph import ROLE_VALUES
from pufre.metrics import bit_aliasing_entropy, temporal_instability_entropy
from pufre.plotting.graph import ROLE_COLORS

try:
    plt.style.use("publications")
except OSError:
    pass

FAMILIES = [
    "RO_Infineon",
    "RO_Fraunhofer_LEFT_LOWER",
    "RO_Maiti_Spartan3E",
    "RO_Spartan7_roomT",
    "SRAM_ESP32_roomT",
    "SRAM_STM32",
    "SRAM_XMC4500",
]
FAMILY_LABELS = {
    "RO_Fraunhofer_LEFT_LOWER": "Fraunhofer RO",
    "RO_Infineon": "Infineon RO",
    "RO_Maiti_Spartan3E": "Maiti RO",
    "RO_Spartan7_roomT": "Spartan-7 RO",
    "SRAM_ESP32_roomT": "ESP32 SRAM",
    "SRAM_STM32": "STM32 SRAM",
    "SRAM_XMC4500": "XMC4500 SRAM",
}
FAMILY_COLORS = dict(
    zip(
        FAMILIES,
        plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(FAMILIES))),
    )
)
ROLES = ROLE_VALUES
N_BITS = 256

QC_DIR = DATA_DIR / "error_qc"
GRAPH_DIR = DATA_DIR / "error_graph"
HYPOTHESIS_DIR = DATA_DIR / "error_hypothesis"
ARTIFACT_PATH = DATA_DIR / "puf_artifact.h5"
FIGURE_DIR = FIGURES_DIR / "host" / "room_temperature"
TABLE_DIR = PROJECT_DIR / "tables" / "host" / "room_temperature"


def main():
    data = load_cached_results()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    bit_quality, family_quality = load_puf_quality_metrics(data["roles"])
    readout_distributions = load_readout_error_distributions()

    # with pl.Config(tbl_cols=-1):
    #     print(
    #         family_quality.sort(["strength_reliable_entropy_rho"], descending=True).select(
    #             ["family", "mean_uniqueness", "mean_upc", "mean_stability", "mean_reliable_entropy", "strength_reliable_entropy_rho"]
    #         )
    #     )

    plot_role_flip_rate(data["roles"], FIGURE_DIR / "role_flip_rate.png")
    plot_hub_authority(data["roles"], FIGURE_DIR / "hub_authority_roles.png")
    plot_validation_gain(
        data["validation_trials"],
        FIGURE_DIR / "validation_gain.png",
    )
    plot_evidence_summary(
        data["hypothesis"],
        data["graph_distribution"],
        data["validation_trials"],
        FIGURE_DIR / "evidence_summary.png",
    )
    plot_block_failure_curves(
        data["block_failure"],
        FIGURE_DIR / "block_failure_curves.png",
    )
    plot_graph_null_effects(
        data["graph_distribution"],
        FIGURE_DIR / "graph_null_effects.png",
    )
    plot_puf_quality_distributions(
        bit_quality,
        family_quality,
        FIGURE_DIR / "puf_quality_distributions.png",
    )
    plot_readout_reliability_ridges(
        readout_distributions,
        FIGURE_DIR / "readout_reliability_ridges.png",
    )
    plot_readout_reliability_ecdf(
        readout_distributions,
        FIGURE_DIR / "readout_reliability_ecdf.png",
    )
    plot_error_count_distributions(
        readout_distributions,
        FIGURE_DIR / "error_count_distributions.png",
    )
    plot_exact_exceedance_curves(
        readout_distributions,
        FIGURE_DIR / "exact_exceedance_curves.png",
    )
    plot_exceedance_gap_curves(
        readout_distributions,
        FIGURE_DIR / "exceedance_gap_curves.png",
    )

    write_dataset_table(data["qc"], TABLE_DIR / "datasets.tex")
    write_dependence_table(data, TABLE_DIR / "dependence_summary.tex")
    write_role_table(data["roles"], TABLE_DIR / "role_summary.tex")
    write_independence_null_table(
        data["independence_null"],
        TABLE_DIR / "graph_independence_null.tex",
    )
    write_conditional_independence_table(
        data["hypothesis"],
        TABLE_DIR / "conditional_independence.tex",
    )
    write_puf_quality_table(
        family_quality,
        TABLE_DIR / "puf_quality_summary.tex",
    )
    logger.info("Wrote HOST room-temperature figures to %s", FIGURE_DIR)
    logger.info("Wrote HOST room-temperature tables to %s", TABLE_DIR)


def load_cached_results():
    paths = {
        "qc": QC_DIR / "artifact_qc.csv",
        "block_failure": QC_DIR / "block_failure_curve.csv",
        "validation_trials": QC_DIR / "validation_trials.csv",
        "independence_null": GRAPH_DIR / "independence_null_summary.csv",
        "hypothesis": HYPOTHESIS_DIR / "permutation_summary.csv",
        "graph_distribution": GRAPH_DIR / "graph_bootstrap_distribution.csv",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached result {path}. Run the corresponding study first."
            )

    frames = {name: select_families(pl.read_csv(path)) for name, path in paths.items()}
    frames["hypothesis"] = frames["hypothesis"].with_columns(
        pl.Series(
            "p_holm",
            holm_adjusted_p_values(frames["hypothesis"]["p_value"]),
        )
    )
    required = {"expected_independent_unique_rows"}
    missing = required - set(frames["qc"].columns)
    if missing:
        raise ValueError(
            "QC cache predates the device-conditional metrics. "
            "Rerun scripts/run_error_qc.py with --compute-mi."
        )
    frames["roles"] = {}
    for family in FAMILIES:
        path = GRAPH_DIR / family / "stable_bit_graph_roles.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing bootstrap-stable graph-role cache for {family}: {path}"
            )
        roles = pl.read_csv(path)
        if roles.height != N_BITS:
            raise ValueError(
                f"{family} contains {roles.height} graph nodes; expected {N_BITS}"
            )
        frames["roles"][family] = roles

    validation = frames["validation_trials"]
    for family in FAMILIES:
        n_trials = validation.filter(pl.col("family") == family).height
        if n_trials < 2:
            raise ValueError(
                f"{family} contains only {n_trials} predictive-validation trial(s)"
            )
    return frames


def load_puf_quality_metrics(roles):
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset artifact {ARTIFACT_PATH}")

    artifact = families_by_name(ARTIFACT_PATH)
    missing = set(FAMILIES) - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing families: {sorted(missing)}")

    bit_frames = []
    family_rows = []
    for family in FAMILIES:
        observations = artifact[family]
        reference = observations.reference
        responses = observations.responses
        n_devices = reference.shape[0]

        upc = bit_aliasing_entropy(reference)
        stability = 1.0 - temporal_instability_entropy(responses)
        reliable_entropy = upc * stability
        # ones = reference.sum(axis=0)
        # uniqueness = np.mean(
        #     2.0 * ones * (n_devices - ones)
        #     / (n_devices * (n_devices - 1))
        # )
        uniqueness = pdist(reference, "hamming")

        graph = roles[family].select(
            "bit_id",
            "graph_role",
            "total_strength",
        )
        bit_frame = pl.DataFrame(
            {
                "family": family,
                "bit_id": np.arange(reference.shape[1]),
                "upc": upc,
                "stability": stability,
                "reliable_entropy": reliable_entropy,
            }
        ).join(graph, on="bit_id", how="left")
        bit_frames.append(bit_frame)

        strength = bit_frame["total_strength"].to_numpy()
        family_rows.append(
            {
                "family": family,
                "uniqueness": uniqueness,
                "mean_uniqueness": float(np.mean(uniqueness)),
                "mean_upc": float(np.mean(upc)),
                "mean_stability": float(np.mean(stability)),
                "mean_reliable_entropy": float(np.mean(reliable_entropy)),
                "strength_upc_rho": safe_spearman(strength, upc),
                "strength_stability_rho": safe_spearman(strength, stability),
                "strength_reliable_entropy_rho": safe_spearman(
                    strength,
                    reliable_entropy,
                ),
            }
        )
    return pl.concat(bit_frames), pl.DataFrame(family_rows)


def load_readout_error_distributions():
    """Return exact empirical and independent error-count distributions."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Missing dataset artifact {ARTIFACT_PATH}")

    artifact = families_by_name(ARTIFACT_PATH)
    missing = set(FAMILIES) - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing families: {sorted(missing)}")

    distributions = {}
    for family in FAMILIES:
        observations = artifact[family]
        responses = np.asarray(observations.responses)
        reference = np.asarray(observations.reference)
        n_bits = responses.shape[-1]
        if n_bits != N_BITS:
            raise ValueError(
                f"{family} contains {n_bits} response bits; expected {N_BITS}"
            )

        errors = responses != reference[:, np.newaxis, :]
        counts = errors.sum(axis=2).reshape(-1).astype(int)
        empirical_pmf = np.bincount(
            counts,
            minlength=n_bits + 1,
        ).astype(float)
        empirical_pmf /= empirical_pmf.sum()

        per_device_flip_rate = errors.mean(axis=1)
        empirical_device_pmf = np.stack(
            [
                np.bincount(device_counts, minlength=n_bits + 1) / device_counts.size
                for device_counts in errors.sum(axis=2)
            ]
        )
        independent_device_pmf = np.stack(
            [
                poisson_binomial_pmf(probabilities)
                for probabilities in per_device_flip_rate
            ]
        )
        independent_pmf = independent_device_pmf.mean(axis=0)

        if not np.isclose(empirical_pmf.sum(), 1.0):
            raise ValueError(f"{family} empirical PMF is not normalized")
        if not np.isclose(independent_pmf.sum(), 1.0):
            raise ValueError(f"{family} independent PMF is not normalized")

        distributions[family] = {
            "counts": counts,
            "empirical_pmf": empirical_pmf,
            "independent_pmf": independent_pmf,
            "empirical_device_pmf": empirical_device_pmf,
            "independent_device_pmf": independent_device_pmf,
            "n_bits": n_bits,
        }
    return distributions


def poisson_binomial_pmf(probabilities):
    """Exact PMF for a sum of independent, non-identical Bernoulli variables."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    pmf = np.zeros(probabilities.size + 1, dtype=float)
    pmf[0] = 1.0
    for probability in probabilities:
        previous = pmf.copy()
        pmf *= 1.0 - probability
        pmf[1:] += previous[:-1] * probability
    return pmf


def safe_spearman(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def select_families(frame):
    selected = frame.filter(pl.col("family").is_in(FAMILIES))
    missing = set(FAMILIES) - set(selected["family"].to_list())
    if missing:
        raise ValueError(f"Cached results are missing families: {sorted(missing)}")
    return selected


def plot_role_flip_rate(roles, output_path):
    centers = np.arange(len(FAMILIES))
    width = 0.12
    role_center = (len(ROLES) - 1) / 2
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    for family_index, family in enumerate(FAMILIES):
        frame = roles[family]
        for role_index, role in enumerate(ROLES):
            selected = frame.filter(pl.col("graph_role") == role)
            if selected.is_empty():
                continue
            position = centers[family_index] + (role_index - role_center) * width
            boxes = ax.boxplot(
                [selected["flip_rate"].to_numpy()],
                positions=[position],
                widths=width * 0.8,
                showfliers=False,
                patch_artist=True,
            )
            boxes["boxes"][0].set_facecolor(ROLE_COLORS[role])
            boxes["boxes"][0].set_alpha(0.65)
            boxes["medians"][0].set_color("black")
    ax.set_xticks(
        centers,
        [FAMILY_LABELS[family].replace(" ", "\n", 1) for family in FAMILIES],
    )
    ax.set_ylabel("Bitflip rate")
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=ROLE_COLORS[role],
                label=role.title(),
            )
            for role in ROLES
        ],
        ncol=3,
        fontsize=9,
    )
    ax.set(ylim=(-0.02, 0.41))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hub_authority(roles, output_path):
    n_columns = 4
    n_rows = int(np.ceil((len(FAMILIES) + 1) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3.25 * n_rows),
        squeeze=False,
    )
    for ax, family in zip(axes.flat, FAMILIES):
        frame = roles[family]
        colors = [ROLE_COLORS[role] for role in frame["graph_role"]]
        ax.scatter(
            frame["relative_out_strength"],
            frame["relative_in_strength"],
            c=colors,
            s=12,
            edgecolor="none",
            alpha=0.75,
        )
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Relative source strength")
        ax.set_ylabel("Relative sink strength")
    legend_ax = axes.flat[len(FAMILIES)]
    legend_ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=ROLE_COLORS[role],
            label=role,
        )
        for role in ROLES
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        fontsize=12,
        frameon=False,
    )
    for ax in axes.flat[len(FAMILIES) + 1 :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_validation_gain(validation_trials, output_path):
    values = [
        validation_trials.filter(pl.col("family") == family)["log_loss_gain"].to_numpy()
        for family in FAMILIES
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.boxplot(values, vert=False, showfliers=False)
    ax.scatter(
        [np.mean(value) for value in values],
        np.arange(1, len(FAMILIES) + 1),
        color="tab:blue",
        marker="D",
        s=18,
        zorder=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(
        np.arange(1, len(FAMILIES) + 1),
        [FAMILY_LABELS[family] for family in FAMILIES],
    )
    ax.set_xlabel("Held-out log-loss gain over independent baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def paired_null_standardized_effects(frame, family, metric):
    selected = frame.filter(pl.col("family") == family)
    observed = (
        selected.filter(pl.col("graph_source") == "observed")
        .sort("bootstrap")[metric]
        .to_numpy()
    )
    null = (
        selected.filter(pl.col("graph_source") == "null")
        .sort("bootstrap")[metric]
        .to_numpy()
    )
    scale = np.std(null, ddof=1)
    if scale > 0:
        return (observed - null) / scale
    if np.allclose(observed, null):
        return np.zeros_like(observed, dtype=float)
    return np.full(observed.shape, np.nan)


def plot_evidence_summary(
    hypothesis,
    graph_distribution,
    validation_trials,
    output_path,
):
    positions = np.arange(len(FAMILIES))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 3.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]},
    )

    for metric, label, marker, color, offset in [
        ("correlation_energy", "Correlation energy", "o", "tab:blue", -0.12),
        ("error_count_variance", "Error-count variance", "s", "tab:orange", 0.12),
    ]:
        selected = hypothesis.filter(pl.col("metric") == metric)
        ratios = np.asarray(
            [
                selected.filter(pl.col("family") == family)["observed_to_null_ratio"][0]
                for family in FAMILIES
            ]
        )
        significant = np.asarray(
            [
                selected.filter(pl.col("family") == family)["p_holm"][0] < 0.05
                for family in FAMILIES
            ]
        )
        axes[0].scatter(
            ratios,
            positions + offset,
            marker=marker,
            s=34,
            facecolors=[color if value else "white" for value in significant],
            edgecolors=color,
            label=label,
            zorder=3,
        )
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Observed / null (1 = null)")
    axes[0].set_title("(a) Model-free tests")
    axes[0].legend(fontsize=7, loc="center right")

    graph_effects = [
        paired_null_standardized_effects(
            graph_distribution,
            family,
            "total_abs_weight",
        )
        for family in FAMILIES
    ]
    axes[1].boxplot(
        graph_effects,
        positions=positions,
        vert=False,
        showfliers=False,
        manage_ticks=False,
    )
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("Null-standardized weight (>0: observed > null)")
    axes[1].set_title("(b) Graph-null effect")

    gains = [
        validation_trials.filter(pl.col("family") == family)["log_loss_gain"].to_numpy()
        for family in FAMILIES
    ]
    axes[2].boxplot(
        gains,
        positions=positions,
        vert=False,
        showfliers=False,
        manage_ticks=False,
    )
    axes[2].scatter(
        [np.mean(value) for value in gains],
        positions,
        color="tab:blue",
        marker="D",
        s=18,
        zorder=3,
    )
    axes[2].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel(r"Held-out $\Delta\mathcal{L}$ (>0: graph better)")
    axes[2].set_title("(c) Predictive transfer")

    axes[0].set_yticks(
        positions,
        [FAMILY_LABELS[family] for family in FAMILIES],
    )
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_block_failure_curves(frame, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    family_handles = []
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(FAMILIES)))
    for color, family in zip(colors, FAMILIES):
        data = frame.filter(pl.col("family") == family).sort("correction_capacity")
        alpha = 0.4 if family != "RO_Fraunhofer_LEFT_LOWER" else 0.75
        ax.plot(
            data["correction_capacity"],
            data["empirical_block_failure"],
            color=color,
            linestyle="-",
            alpha=alpha,
        )
        ax.plot(
            data["correction_capacity"],
            data["independent_block_failure"],
            color=color,
            linestyle="--",
            alpha=alpha,
        )
        family_handles.append(
            plt.Line2D([0], [0], color=color, label=FAMILY_LABELS[family])
        )
    family_legend = ax.legend(
        handles=family_handles,
        # loc="upper right",
        ncol=2,
        fontsize=10,
    )
    ax.add_artist(family_legend)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", label="Empirical"),
            plt.Line2D([0], [0], color="black", linestyle="--", label="Independent"),
        ],
        loc="lower left",
        fontsize=10,
    )
    ax.set_xlabel("Correction capacity")
    ax.set_ylabel("Block-failure probability")
    ax.set(xlim=(-1, 28), ylim=(-0.01, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_readout_reliability_ridges(distributions, output_path):
    """Plot canonical per-readout reliability as normalized ridgelines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    offsets = np.arange(len(FAMILIES))[::-1]
    minimum_reliability = 1.0

    for offset, family in zip(offsets, FAMILIES):
        values = distributions[family]
        n_bits = values["n_bits"]
        reliability = 1.0 - np.arange(n_bits + 1) / n_bits
        order = np.argsort(reliability)
        reliability = reliability[order]
        empirical = values["empirical_pmf"][order]
        independent = values["independent_pmf"][order]
        scale = max(empirical.max(), independent.max())
        empirical = 0.8 * empirical / scale
        independent = 0.8 * independent / scale

        minimum_reliability = min(
            minimum_reliability,
            1.0 - values["counts"].max() / n_bits,
        )
        color = FAMILY_COLORS[family]
        ax.fill_between(
            reliability,
            offset,
            offset + empirical,
            color=color,
            alpha=0.45,
        )
        ax.plot(
            reliability,
            offset + empirical,
            color=color,
            linewidth=1.2,
        )
        ax.plot(
            reliability,
            offset + independent,
            color=color,
            linestyle="--",
            linewidth=1.0,
        )

    ax.set_yticks(
        offsets + 0.25,
        [FAMILY_LABELS[family] for family in FAMILIES],
    )
    ax.set_xlim(max(0.0, minimum_reliability - 0.03), 1.005)
    ax.set_xlabel(r"Per-readout reliability $1-\mathrm{IntraHD}$")
    ax.set_ylabel("")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", label="Empirical"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="Independent",
            ),
        ],
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_readout_reliability_ecdf(distributions, output_path):
    """Compare empirical and independent ECDFs of per-readout reliability."""
    fig, ax = plt.subplots(figsize=(8, 4))
    minimum_reliability = 1.0

    for family in FAMILIES:
        values = distributions[family]
        n_bits = values["n_bits"]
        reliability = 1.0 - np.arange(n_bits + 1) / n_bits
        order = np.argsort(reliability)
        reliability = reliability[order]
        empirical_cdf = np.cumsum(values["empirical_pmf"][order])
        independent_cdf = np.cumsum(values["independent_pmf"][order])
        minimum_reliability = min(
            minimum_reliability,
            1.0 - values["counts"].max() / n_bits,
        )

        color = FAMILY_COLORS[family]
        ax.step(
            reliability,
            empirical_cdf,
            where="post",
            color=color,
            label=FAMILY_LABELS[family],
        )
        ax.step(
            reliability,
            independent_cdf,
            where="post",
            color=color,
            linestyle="--",
            alpha=0.75,
        )

    family_legend = ax.legend(ncol=2, fontsize=9)
    ax.add_artist(family_legend)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", label="Empirical"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="Independent",
            ),
        ],
        loc="upper left",
        fontsize=9,
    )
    ax.set_xlim(max(0.0, minimum_reliability - 0.03), 1.005)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel(r"Per-readout reliability $1-\mathrm{IntraHD}$")
    ax.set_ylabel("Cumulative probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_count_distributions(distributions, output_path):
    """Plot empirical and independent error-count PMFs for each family."""
    fig, axes = plt.subplots(
        1,
        len(FAMILIES),
        figsize=(8, 2),
        squeeze=False,
    )

    for family_index, (ax, family) in enumerate(zip(axes.flat, FAMILIES)):
        values = distributions[family]
        empirical = values["empirical_pmf"]
        independent = values["independent_pmf"]
        error_counts = np.arange(empirical.size)
        visible = np.flatnonzero((empirical > 1e-4) | (independent > 1e-4))
        maximum = max(int(values["counts"].max()), int(visible[-1])) + 1
        maximum = min(maximum, values["n_bits"])

        ax.bar(
            error_counts[: maximum + 1],
            empirical[: maximum + 1],
            # color=FAMILY_COLORS[family],
            color="tab:blue",
            alpha=0.6,
            width=1.0,
            linewidth=0,
            label="Empirical",
        )
        ax.plot(
            error_counts[: maximum + 1],
            independent[: maximum + 1],
            color="black",
            linestyle="--",
            # linewidth=1.2,
            linewidth=0.75,
            label="Independent",
        )
        ax.set_title(FAMILY_LABELS[family], fontsize=7, pad=2)
        ax.tick_params(axis="both", labelsize=6, pad=1)
        if family_index == 0:
            ax.set_ylabel("Probability", fontsize=7)
        ax.set_xlim(-0.5, maximum + 0.5)

    fig.supxlabel(
        "Bitflips per readout",
        fontsize=7,
        y=0.1,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), pad=0.25, w_pad=0.5)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_exact_exceedance_curves(distributions, output_path):
    """Plot exact empirical and device-conditioned independent exceedance."""
    capacities = np.arange(27)
    fig, ax = plt.subplots(figsize=(8, 4))

    for family in FAMILIES:
        values = distributions[family]
        empirical = 1.0 - np.cumsum(values["empirical_pmf"])
        independent = 1.0 - np.cumsum(values["independent_pmf"])
        color = FAMILY_COLORS[family]
        ax.plot(
            capacities,
            empirical[capacities],
            color=color,
            label=FAMILY_LABELS[family],
        )
        ax.plot(
            capacities,
            independent[capacities],
            color=color,
            linestyle="--",
            alpha=0.75,
        )

    family_legend = ax.legend(ncol=2, fontsize=9)
    ax.add_artist(family_legend)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", label="Empirical"),
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                label="Independent",
            ),
        ],
        loc="lower left",
        fontsize=10,
    )
    ax.set_xlim(-0.5, capacities[-1] + 0.5)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel(r"Correction capacity $\tau$")
    ax.set_ylabel(r"Correction-capacity exceedance $P(K>\tau)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def device_bootstrap_exceedance_gap(
    values,
    capacities,
    *,
    n_bootstrap=2000,
    random_state=0,
):
    empirical = 1.0 - np.cumsum(values["empirical_pmf"])
    independent = 1.0 - np.cumsum(values["independent_pmf"])
    point = empirical[capacities] - independent[capacities]

    empirical_by_device = 1.0 - np.cumsum(
        values["empirical_device_pmf"],
        axis=1,
    )
    independent_by_device = 1.0 - np.cumsum(
        values["independent_device_pmf"],
        axis=1,
    )
    device_gaps = (
        empirical_by_device[:, capacities] - independent_by_device[:, capacities]
    )
    n_devices = device_gaps.shape[0]
    weights = np.random.default_rng(random_state).multinomial(
        n_devices,
        np.full(n_devices, 1.0 / n_devices),
        size=n_bootstrap,
    )
    samples = weights @ device_gaps / n_devices
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    if point.shape != lower.shape or np.any(lower > upper):
        raise RuntimeError("Invalid device-bootstrap exceedance interval")
    return point, lower, upper


def plot_exceedance_gap_curves(distributions, output_path):
    capacities = np.arange(27)
    fig, ax = plt.subplots(figsize=(8, 4))
    for family_index, family in enumerate(FAMILIES):
        point, lower, upper = device_bootstrap_exceedance_gap(
            distributions[family],
            capacities,
            random_state=family_index,
        )
        color = FAMILY_COLORS[family]
        ax.plot(
            capacities,
            point,
            color=color,
            linewidth=1.8 if family == "RO_Fraunhofer_LEFT_LOWER" else 1.2,
            label=FAMILY_LABELS[family],
        )
        ax.fill_between(
            capacities,
            lower,
            upper,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(-0.5, capacities[-1] + 0.5)
    ax.set_xlabel(r"Correction capacity $\tau$")
    ax.set_ylabel(
        r"Exceedance gap $\Delta P_\tau="
        r"P_{\mathrm{emp}}(K>\tau)-P_{\mathrm{ind}}(K>\tau)$"
    )
    ax.legend(ncol=2, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_graph_null_effects(frame, output_path):
    metrics = [
        ("directed_edges", "Bootstrap-fit edge count"),
        ("total_abs_weight", "Total absolute weight"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        effects = []
        for family in FAMILIES:
            effects.append(paired_null_standardized_effects(frame, family, metric))
        ax.boxplot(effects, vert=False, showfliers=False)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"Null-standardized {label.lower()}")
    axes[0].set_yticks(
        np.arange(1, len(FAMILIES) + 1),
        [FAMILY_LABELS[family] for family in FAMILIES],
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_puf_quality_distributions(bit_frame, family_frame, output_path):
    metrics = [
        ("upc", "UPC"),
        ("stability", "Stability"),
        ("reliable_entropy", "Reliable Entropy"),
    ]
    labels = [FAMILY_LABELS[family] for family in FAMILIES]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.5), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        values = [
            bit_frame.filter(pl.col("family") == family)[metric].to_numpy()
            for family in FAMILIES
        ]
        ax.boxplot(values, vert=False, showfliers=False)
        ax.set_xlabel(label)

    uniqueness = [
        family_frame.filter(pl.col("family") == family)["uniqueness"].to_numpy()[0]
        for family in FAMILIES
    ]
    axes[3].boxplot(uniqueness, vert=False, showfliers=False)
    axes[3].set_xlabel("Uniqueness")

    axes[0].set_yticks(np.arange(1, len(FAMILIES) + 1), labels)

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_puf_quality_table(frame, output_path):
    def value(number):
        return "--" if not np.isfinite(number) else f"{number:.3f}"

    rows = []
    for family in FAMILIES:
        row = frame.filter(pl.col("family") == family).row(0, named=True)
        rows.append(
            f"{FAMILY_LABELS[family]} & {row['mean_uniqueness']:.3f} & "
            f"{row['mean_upc']:.3f} & {row['mean_stability']:.3f} & "
            f"{row['mean_reliable_entropy']:.3f} & "
            f"{value(row['strength_upc_rho'])} & "
            f"{value(row['strength_stability_rho'])} & "
            f"{value(row['strength_reliable_entropy_rho'])} \\\\"
        )
    write_table(
        output_path,
        "lrrrrrrr",
        "Dataset & Uniqueness & UPC & Stability & RE & "
        "$\\rho(q,\\mathrm{UPC})$ & $\\rho(q,\\mathrm{Stability})$ & "
        "$\\rho(q,\\mathrm{RE})$",
        rows,
        "Classical inter-device uniqueness and mean per-bit quality metrics. "
        "Spearman coefficients compare each metric with total stable graph "
        "strength $q=q_{\\mathrm{in}}+q_{\\mathrm{out}}$.",
        "tab:puf-quality",
        wide=True,
    )


def write_dataset_table(frame, output_path):
    rows = []
    for family in FAMILIES:
        values = frame.filter(pl.col("family") == family).row(0, named=True)
        rows.append(
            f"{FAMILY_LABELS[family]} & {values['n_devices']} & "
            f"{values['n_samples']} & {values['n_bits']} & "
            f"{values['n_observations']} \\\\"
        )
    write_table(
        output_path,
        "lrrrr",
        "Dataset & Devices & Repetitions & Analyzed bits & Error rows",
        rows,
        "Room-temperature datasets used in the cross-family evaluation. "
        "Each analysis uses the same 256-bit response width.",
        "tab:room-datasets",
    )


def write_dependence_table(data, output_path):
    rows = []
    for family in FAMILIES:
        qc = data["qc"].filter(pl.col("family") == family).row(0, named=True)
        trials = data["validation_trials"].filter(pl.col("family") == family)
        correlation = (
            data["hypothesis"]
            .filter(
                (pl.col("family") == family)
                & (pl.col("metric") == "correlation_energy")
            )
            .row(0, named=True)
        )
        count_variance = (
            data["hypothesis"]
            .filter(
                (pl.col("family") == family)
                & (pl.col("metric") == "error_count_variance")
            )
            .row(0, named=True)
        )
        gain = float(trials["log_loss_gain"].mean())
        relative_gain = 100.0 * gain / float(trials["independent_log_loss"].mean())
        rows.append(
            f"{FAMILY_LABELS[family]} & {qc['flip_rate_mean']:.3f} & "
            f"{qc['unique_error_rows'] / qc['expected_independent_unique_rows']:.3f} & "
            f"{correlation['observed_to_null_ratio']:.2f} "
            f"({correlation['p_holm']:.3f}) & "
            f"{count_variance['observed_to_null_ratio']:.2f} "
            f"({count_variance['p_holm']:.3f}) & "
            f"{gain:.5f} & {relative_gain:.2f} \\\\"
        )
    write_table(
        output_path,
        "lrrrrrr",
        "Dataset & BER & "
        "$U_{\\mathrm{obs}}/\\mathbb{E}[U_{\\mathrm{ind}}]$ & "
        "$R_{\\mathrm{corr}}\\;(p_{\\mathrm{H}})$ & "
        "$R_{\\mathrm{count}}\\;(p_{\\mathrm{H}})$ & "
        "$\\Delta\\mathcal{L}\\;\\uparrow$ & "
        "$G_{\\mathrm{rel}}\\;(\\%)\\;\\uparrow$",
        rows,
        "Room-temperature dependence and predictive-transfer summary. "
        "Parentheses contain Holm-adjusted permutation p-values; "
        "$\\Delta\\mathcal{L}$ is the mean over 20 device-grouped splits. "
        "Upward arrows indicate better held-out predictive performance.",
        "tab:dependence-summary",
        wide=True,
    )


def write_role_table(roles, output_path):
    rows = []
    for family in FAMILIES:
        frame = roles[family]
        for role in ROLES:
            selected = frame.filter(pl.col("graph_role") == role)
            median = selected["flip_rate"].median()
            median_text = "--" if median is None else f"{median:.3f}"
            rows.append(
                f"{FAMILY_LABELS[family]} & {role.title()} & "
                f"{selected.height} & {median_text} \\\\"
            )
    write_table(
        output_path,
        "llrr",
        "Dataset & Role & Bits & Median bitflip rate",
        rows,
        "Bootstrap-stable graph roles and their held-out bitflip rates.",
        "tab:room-roles",
        wide=True,
    )


def write_independence_null_table(frame, output_path):
    rows = []
    for family in FAMILIES:
        values = frame.filter(pl.col("family") == family).row(0, named=True)
        observed_edges = values["observed_stable_directed_edges"]
        null_edges = values["null_stable_directed_edges"]
        ratio = observed_edges / null_edges if null_edges else float("inf")
        ratio_text = "$\\infty$" if not np.isfinite(ratio) else f"{ratio:.2f}"
        rows.append(
            f"{FAMILY_LABELS[family]} & {observed_edges} & {null_edges} & "
            f"{ratio_text} & {values['observed_disconnected_bits']} & "
            f"{values['null_disconnected_bits']} \\\\"
        )
    write_table(
        output_path,
        "lrrrrr",
        "Dataset & $|E|_{\\mathrm{obs}}$ & $|E|_{\\mathrm{null}}$ & Ratio & "
        "$N_{\\mathrm{disc,obs}}$ & $N_{\\mathrm{disc,null}}$",
        rows,
        "Bootstrap-stable graph density under observed errors and a "
        "device-preserving independent surrogate.",
        "tab:graph-independence-null",
    )


def write_conditional_independence_table(frame, output_path):
    rows = []
    for family in FAMILIES:
        correlation = frame.filter(
            (pl.col("family") == family) & (pl.col("metric") == "correlation_energy")
        ).row(0, named=True)
        count_variance = frame.filter(
            (pl.col("family") == family) & (pl.col("metric") == "error_count_variance")
        ).row(0, named=True)
        rows.append(
            f"{FAMILY_LABELS[family]} & "
            f"{correlation['observed_to_null_ratio']:.2f} & "
            f"{correlation['p_holm']:.3f} & "
            f"{count_variance['observed_to_null_ratio']:.2f} & "
            f"{count_variance['p_holm']:.3f} \\\\"
        )
    write_table(
        output_path,
        "lrrrr",
        "Dataset & $R_{\\mathrm{corr}}$ & $p_{\\mathrm{H,corr}}$ & "
        "$R_{\\mathrm{count}}$ & $p_{\\mathrm{H,count}}$",
        rows,
        "Model-free conditional-independence tests using within-device, "
        "per-bit permutations. P-values use Holm family-wise adjustment "
        "across the 14 room-temperature tests.",
        "tab:conditional-independence",
    )


def write_table(path, columns, header, rows, caption, label, *, wide=False):
    environment = "table*" if wide else "table"
    lines = [
        f"\\begin{{{environment}}}[t]",
        "    \\centering",
        "    \\small",
        f"    \\begin{{tabular}}{{{columns}}}",
        "        \\toprule",
        f"        {header} \\\\ ",
        "        \\midrule",
        *[f"        {row}" for row in rows],
        "        \\bottomrule",
        "    \\end{tabular}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"\\end{{{environment}}}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
