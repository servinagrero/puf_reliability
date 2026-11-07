from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.model import NodewiseLogisticGraph

EDGE_STABILITY_SCHEMA = {
    "target": pl.Int64,
    "predictor": pl.Int64,
    "selection_probability": pl.Float64,
    "mean_weight": pl.Float64,
    "std_weight": pl.Float64,
    "mean_abs_weight": pl.Float64,
    "sign_stability": pl.Float64,
}


@dataclass
class EdgeStability:
    selection_probability: npt.NDArray
    mean_weight: npt.NDArray
    std_weight: npt.NDArray
    mean_abs_weight: npt.NDArray
    sign_stability: npt.NDArray
    adjacency_samples: npt.NDArray | None = None


def bootstrap_edge_stability(
    observations,
    *,
    n_bootstrap: int = 50,
    C: float | None = 0.1,
    edge_threshold: float = 1e-8,
    random_state: int = 0,
    max_iter: int = 500,
    independent_null: bool = False,
    n_jobs: int = 1,
) -> EdgeStability:
    """
    Refit directed error-graph models on device bootstraps and summarize edges.

    Devices are sampled with replacement and all repeated measurements from each
    sampled device are retained. This estimates graph stability under finite
    device sampling. When ``independent_null`` is enabled, each error-spin
    column is independently permuted within device after resampling. This
    preserves device-conditional bit marginals while destroying simultaneous
    cross-bit dependence.
    """
    rng = np.random.default_rng(random_state)
    surrogate_rng = np.random.default_rng(random_state + 1_000_003)
    n_bits = observations.n_bits
    weights = np.zeros((n_bootstrap, n_bits, n_bits), dtype=np.float32)
    unique_devices = np.arange(observations.n_devices)

    for idx in range(n_bootstrap):
        sampled_devices = rng.choice(
            unique_devices, size=unique_devices.size, replace=True
        )
        rows = np.concatenate(
            [
                np.flatnonzero(observations.device_ids == device)
                for device in sampled_devices
            ]
        )
        errors = observations.flat_errors()[rows]
        groups = observations.device_ids[rows]
        if independent_null:
            errors = _permute_columns_within_groups(errors, groups, surrogate_rng)

        estimator = NodewiseLogisticGraph(
            C=C,
            random_state=random_state + idx,
            max_iter=max_iter,
            n_jobs=n_jobs,
        )
        estimator.fit(errors, groups=groups)
        weights[idx] = estimator.model_.adjacency

    selected = np.abs(weights) > edge_threshold
    signs = np.sign(weights)
    sign_balance = np.abs(np.mean(np.where(selected, signs, 0.0), axis=0))
    return EdgeStability(
        selection_probability=selected.mean(axis=0),
        mean_weight=weights.mean(axis=0),
        std_weight=weights.std(axis=0),
        mean_abs_weight=np.abs(weights).mean(axis=0),
        sign_stability=sign_balance,
        adjacency_samples=weights,
    )


def _permute_columns_within_groups(errors, groups, rng):
    errors = np.asarray(errors)
    groups = np.asarray(groups)
    if errors.ndim != 2 or groups.shape != (errors.shape[0],):
        raise ValueError(
            "errors and groups must have shapes (samples, bits) and (samples,)"
        )

    permuted = errors.copy()
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        for bit in range(errors.shape[1]):
            permuted[indices, bit] = rng.permutation(errors[indices, bit])
    return permuted


def edge_stability_frame(
    stability: EdgeStability, *, min_probability: float = 0.0
) -> pl.DataFrame:
    """Convert bootstrap edge stability matrices to a directed edge table."""
    rows = []
    n_bits = stability.selection_probability.shape[0]
    for target in range(n_bits):
        for predictor in range(n_bits):
            if target == predictor:
                continue
            probability = stability.selection_probability[target, predictor]
            if probability <= min_probability:
                continue
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "selection_probability": float(probability),
                    "mean_weight": float(stability.mean_weight[target, predictor]),
                    "std_weight": float(stability.std_weight[target, predictor]),
                    "mean_abs_weight": float(
                        stability.mean_abs_weight[target, predictor]
                    ),
                    "sign_stability": float(
                        stability.sign_stability[target, predictor]
                    ),
                }
            )
    return pl.DataFrame(rows, schema=EDGE_STABILITY_SCHEMA)


def stable_adjacency(
    stability: EdgeStability, *, min_probability: float = 0.5
) -> npt.NDArray:
    """Probability-weighted signed adjacency from bootstrap stability."""
    A = stability.mean_weight * stability.selection_probability
    A[stability.selection_probability < min_probability] = 0.0
    np.fill_diagonal(A, 0.0)
    return A
