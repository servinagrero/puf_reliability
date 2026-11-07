from pathlib import Path

import matplotlib as mpl
import numpy as np
import polars as pl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("publications")

from pufre.analysis import spectral_metrics
from pufre.graph import compute_hits_centrality, compute_network_centrality
from pufre.hamiltonian import compute_avalanche_size_complete, hysteresis_loop


def plot_study_summary(summary: dict, output_dir: Path, *, prefix: str):
    """Generate the standard reusable figures for a saved LASSO study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _summary_frame(summary)

    plot_susceptibility(frame, output_dir / f"{prefix}_susceptibility.png")
    plot_reliability_vs_susceptibility(
        frame, output_dir / f"{prefix}_reliability_vs_susceptibility.png"
    )
    # plot_stability_vs_upc(frame, output_dir / f"{prefix}_stability_vs_upc.png")
    plot_reliable_entropy_vs_susceptibility(
        frame, output_dir / f"{prefix}_reliable_entropy_vs_susceptibility.png"
    )
    plot_reliable_entropy(frame, output_dir / f"{prefix}_reliable_entropy.png")
    plot_independence_pairplot(frame, output_dir / f"{prefix}_pairplot.png")


def plot_hamiltonian_study(data, hamiltonian, output_dir: Path, *, prefix: str):
    """Generate device-level Hamiltonian and graph figures from data and model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_idx = 0
    device_h = hamiltonian.h_intrinsic[device_idx]
    device_samples = data.crps[device_idx]
    device_ref = data.gt_polar[device_idx]
    J_sym = hamiltonian.J_sym

    reliability = np.mean(data.crps == data.gt_polar[:, np.newaxis, :], axis=(0, 1))
    L_lengths, _ = compute_avalanche_size_complete(J_sym, device_h)
    effective_field = device_h + (J_sym @ device_ref)
    alignment = effective_field * device_ref
    frustration = hamiltonian.local_frustration(device_ref)
    energy_ref = hamiltonian.hamiltonian_energy(device_ref.reshape(1, -1), device_h)[0]
    energies_samples = hamiltonian.hamiltonian_energy(device_samples, device_h)
    biases, magnetization = hysteresis_loop(J_sym, device_h, device_ref)
    hub_score, authority_score = compute_network_centrality(hamiltonian.W)
    hits_hub, hits_authority = compute_hits_centrality(hamiltonian.W)

    plot_energy_landscape(
        energies_samples,
        energy_ref,
        output_dir / f"{prefix}_energy_landscape.png",
    )
    plot_local_frustration(
        frustration,
        reliability,
        hamiltonian.h_magnitude,
        output_dir / f"{prefix}_local_frustration.png",
    )
    plot_effective_field_alignment(
        data.bit_ids,
        alignment,
        device_h,
        output_dir / f"{prefix}_effective_field_alignment.png",
    )
    plot_effective_field_vs_avalanches(
        effective_field,
        L_lengths,
        output_dir / f"{prefix}_effective_field_vs_avalanches.png",
    )
    plot_hysteresis(
        biases,
        magnetization,
        output_dir / f"{prefix}_hysteresis.png",
    )
    plot_graph_centrality(
        hub_score,
        authority_score,
        hamiltonian.h_magnitude,
        output_dir / f"{prefix}_hub_centrality.png",
    )
    plot_graph_centrality(
        hits_hub,
        hits_authority,
        hamiltonian.h_magnitude,
        output_dir / f"{prefix}_hub_centrality_hits.png",
        x_label="HITS hub score",
        y_label="HITS authority score",
    )


def plot_model_spectrum(model, output_dir: Path, *, prefix: str):
    """Plot eigenvalue histogram for a saved logistic PUF model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    J_sym = (model.J + model.J.T) / 2.0
    np.fill_diagonal(J_sym, 0.0)
    metrics = spectral_metrics(J_sym)

    fig, ax = plt.subplots(figsize=(6, 4))
    eigenvalues = np.asarray(metrics["eigenvalues"], dtype=float)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if eigenvalues.size == 0:
        plt.close(fig)
        return
    ax.hist(eigenvalues, bins=np.clip(eigenvalues.size, 1, 30))
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_eigenvalues.png", dpi=300)
    plt.close(fig)


def plot_susceptibility(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(frame["bit_id"], frame["susceptibility"], alpha=0.8)
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Susceptibility")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_reliability_vs_susceptibility(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        frame["reliability"],
        frame["susceptibility"],
        c=frame["h_magnitude"],
        s=8,
        alpha=0.75,
    )
    ax.set_xlabel("Reliability")
    ax.set_ylabel("Susceptibility")
    ax.set(xlim=(None, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_stability_vs_upc(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        frame["stability"],
        frame["upc"],
        c=frame["h_magnitude"],
        s=8,
        alpha=0.75,
    )
    ax.set_ylabel("Stability")
    ax.set_xlabel("UPC")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_reliable_entropy_vs_susceptibility(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        frame["susceptibility"],
        frame["reliable_entropy"],
        c=frame["h_magnitude"],
        s=8,
        alpha=0.75,
    )
    ax.set_ylabel("Reliable Entropy")
    ax.set_xlabel("Susceptibility")
    ax.set(ylim=(-0.01, 1.01))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_reliable_entropy(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(frame["bit_id"], frame["reliable_entropy"], alpha=0.8)
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Reliable Entropy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_independence_pairplot(frame: pl.DataFrame, output_path: Path):
    columns = [
        "independence_ratio",
        "reliability",
        "reliable_entropy",
        "h_magnitude",
        "interaction_strength",
        "susceptibility",
    ]
    existing = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = np.asarray(frame[column], dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 1 and np.nanmax(values) > np.nanmin(values):
            existing.append(column)

    if len(existing) < 2:
        return

    plot_frame = frame.select(existing).to_pandas()
    plot_frame = plot_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if plot_frame.shape[0] < 2:
        return

    diag_bins = min(16, max(1, int(np.sqrt(plot_frame.shape[0]))))
    try:
        grid = sns.pairplot(
            plot_frame,
            corner=True,
            diag_kws={"bins": diag_bins},
            plot_kws={"s": 8, "alpha": 0.65},
        )
    except ValueError:
        return
    grid.fig.tight_layout()
    grid.fig.savefig(output_path, dpi=300)
    plt.close(grid.fig)


def plot_energy_landscape(energies_samples, energy_ref, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    energies_samples = np.asarray(energies_samples, dtype=float)
    energies_samples = energies_samples[np.isfinite(energies_samples)]
    if energies_samples.size == 0:
        plt.close(fig)
        return

    energy_min = np.min(energies_samples)
    energy_max = np.max(energies_samples)
    if energy_max > energy_min and np.unique(energies_samples).size > 1:
        try:
            sns.kdeplot(energies_samples, fill=True, label="Noisy samples", ax=ax)
        except ValueError:
            ax.hist(
                energies_samples,
                bins=min(16, energies_samples.size),
                alpha=0.7,
                label="Noisy samples",
            )
    else:
        ax.hist(energies_samples, bins=1, alpha=0.7, label="Noisy samples")
    if np.isfinite(energy_ref):
        ax.axvline(energy_ref, linewidth=2, linestyle="--", label="Reference")
    ax.set_xlabel("Hamiltonian energy H(s)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_local_frustration(frustration, reliability, color_values, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.scatter(frustration, reliability, alpha=0.8, c=color_values, s=10)
    fig.colorbar(im, label="Intrinsic bias |h|")
    ax.set_xlabel("Local frustration energy")
    ax.set_ylabel("Bit reliability")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_effective_field_alignment(bit_ids, alignment, device_h, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    limit = np.nanmax(np.abs(device_h))
    if not np.isfinite(limit) or limit == 0:
        colors = None
    else:
        norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        colors = plt.get_cmap("coolwarm")(norm(device_h))

    rects = ax.bar(bit_ids, alignment, color=colors)

    sm = ScalarMappable(cmap=plt.get_cmap("coolwarm"), norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel("Intrinsic bias |h|")

    ax.set_xlabel("Bit index")
    ax.set_ylabel("Effective-field alignment")
    ax.set(xlim=(-2, len(bit_ids) + 2))
    ax.xaxis.set_major_locator(MultipleLocator(16))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_effective_field_vs_avalanches(effective_field, avalanche_lengths, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(effective_field, avalanche_lengths, alpha=0.75, s=10)
    ax.set_xlabel("Effective field")
    ax.set_ylabel("Avalanche size")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hysteresis(biases, magnetization, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(biases, magnetization)
    ax.set_xlabel("External field")
    ax.set_ylabel("Magnetization")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_graph_centrality(
    hub_score,
    authority_score,
    color_values,
    output_path: Path,
    *,
    x_label: str = "Hub score",
    y_label: str = "Authority score",
):
    hub_score = np.asarray(hub_score, dtype=float)
    authority_score = np.asarray(authority_score, dtype=float)
    color_values = np.asarray(color_values, dtype=float)
    mask = (
        np.isfinite(hub_score)
        & np.isfinite(authority_score)
        & np.isfinite(color_values)
    )
    if not np.any(mask):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.scatter(
        hub_score[mask],
        authority_score[mask],
        alpha=0.75,
        c=color_values[mask],
        s=10,
    )
    fig.colorbar(im, label="Intrinsic bias |h|")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rbm_study(
    summary: dict,
    precision: np.ndarray,
    output_dir: Path,
    *,
    prefix: str,
):
    """Generate reusable figures from saved RBM precision outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _summary_frame(summary)

    if "precision_strength" in frame.columns:
        plot_precision_strength(
            frame["bit_id"],
            frame["precision_strength"],
            output_dir / f"{prefix}_precision_strength.png",
        )
    if {"precision_diagonal", "precision_strength"}.issubset(set(frame.columns)):
        plot_precision_diagonal_vs_strength(
            frame["precision_diagonal"],
            frame["precision_strength"],
            output_dir / f"{prefix}_precision_diagonal_vs_strength.png",
        )
    if {"generated_mean", "generated_variance"}.issubset(set(frame.columns)):
        plot_generated_mean_vs_variance(
            frame["generated_mean"],
            frame["generated_variance"],
            output_dir / f"{prefix}_generated_mean_vs_variance.png",
        )

    plot_precision_heatmap(
        precision,
        output_dir / f"{prefix}_precision_matrix_offdiag.png",
        zero_diagonal=True,
    )
    plot_precision_spectrum(
        precision, output_dir / f"{prefix}_precision_eigenvalues.png"
    )


def plot_precision_strength(bit_ids, strength, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bit_ids, strength, alpha=0.8)
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Off-diagonal precision strength")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_diagonal_vs_strength(diagonal, strength, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(diagonal, strength, alpha=0.75, s=10)
    ax.set_xlabel("Precision diagonal")
    ax.set_ylabel("Off-diagonal precision strength")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_generated_mean_vs_variance(mean, variance, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(mean, variance, alpha=0.75, s=10)
    ax.set_xlabel("Generated response mean")
    ax.set_ylabel("Generated response variance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_heatmap(
    precision: np.ndarray,
    output_path: Path,
    *,
    zero_diagonal: bool = False,
):
    precision = np.asarray(precision, dtype=float).copy()
    precision[~np.isfinite(precision)] = 0.0
    if zero_diagonal:
        np.fill_diagonal(precision, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    limit = np.nanmax(np.abs(precision))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0
    im = ax.imshow(
        precision,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label="Precision")
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Bit index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_precision_spectrum(precision: np.ndarray, output_path: Path):
    metrics = spectral_metrics((precision + precision.T) / 2.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    eigenvalues = np.asarray(metrics["eigenvalues"], dtype=float)
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if eigenvalues.size == 0:
        plt.close(fig)
        return
    ax.hist(eigenvalues, bins=min(30, max(1, eigenvalues.size)))
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _summary_frame(summary: dict):
    bit_count = np.asarray(summary["bit_id"]).shape[0]
    columns = {}
    for key, value in summary.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == bit_count:
            columns[key] = arr
    bit_id = np.asarray(columns["bit_id"])
    local_bit_id = np.arange(bit_count)
    if not np.array_equal(bit_id, local_bit_id):
        columns.setdefault("physical_bit_id", bit_id)
        columns["bit_id"] = local_bit_id
    return pl.DataFrame(columns)
