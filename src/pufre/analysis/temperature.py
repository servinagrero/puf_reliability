from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import spearmanr

from pufre.data import DifferentialObservations, PUFObservations
from pufre.graph import (
    ROLE_VALUES,
    classify_role,
    graph_role_frame,
    graph_role_thresholds,
    positive_quantile,
    prepare_adjacency,
)
from pufre.graph.model import DirectedErrorGraphModel, NodewiseLogisticGraph

ROLES = ROLE_VALUES


@dataclass
class TemperatureGraphStudyResult:
    bit_metrics: pl.DataFrame
    role_summary: pl.DataFrame
    graph_drift: pl.DataFrame
    role_transitions: pl.DataFrame
    bit_susceptibility: pl.DataFrame
    model_transfer: pl.DataFrame
    models: dict[int, DirectedErrorGraphModel]
    regularization: float


def load_differential_hdf5_artifact(
    path: Path,
    *,
    room_temperature: int = 25,
):
    """
    Load the differential RO-style HDF5 artifact produced by data/raw/etl.py.

    The returned response/count dictionaries include the room/reference condition
    under ``room_temperature`` so downstream studies can use it as the baseline.
    """
    with h5py.File(path, "r") as h5:
        required = {
            "reference",
            "reference_majority",
            "reference_freq1",
            "reference_freq2",
            "counts",
        }
        missing = required - set(h5.keys())
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")

        temperatures = sorted(int(name) for name in h5.keys() if name.isdigit())
        responses = {
            room_temperature: h5["reference"][:].astype(np.uint8),
            **{
                temperature: h5[str(temperature)][:].astype(np.uint8)
                for temperature in temperatures
            },
        }
        value1 = {
            room_temperature: h5["reference_freq1"][:].astype(float),
            **{
                temperature: h5["counts"]["freq1"][str(temperature)][:].astype(float)
                for temperature in temperatures
            },
        }
        value2 = {
            room_temperature: h5["reference_freq2"][:].astype(float),
            **{
                temperature: h5["counts"]["freq2"][str(temperature)][:].astype(float)
                for temperature in temperatures
            },
        }
        return {
            "reference": h5["reference_majority"][:].astype(np.uint8),
            "responses": responses,
            "value1": value1,
            "value2": value2,
        }


def fit_temperature_graph_study(
    *,
    reference: np.ndarray,
    responses_by_temperature: Mapping[int, np.ndarray],
    value1_by_temperature: Mapping[int, np.ndarray],
    value2_by_temperature: Mapping[int, np.ndarray],
    baseline_temperature: int,
    graph_C: float | None = None,
    max_iter: int = 500,
    max_tuning_bits: int = 8,
    cv_splits: int = 2,
    n_jobs: int = 1,
    role_quantile: float = 0.75,
    strong_edge_quantile: float = 0.9,
    top_edge_k: int = 256,
):
    bit_frames = []
    summary_frames = []
    models = {}
    observations_by_temperature = {}
    baseline_thresholds = None
    regularization = graph_C

    temperatures = sorted(responses_by_temperature)
    if baseline_temperature not in responses_by_temperature:
        raise ValueError(
            "baseline_temperature is missing from responses_by_temperature"
        )
    ordered_temperatures = [baseline_temperature] + [
        temperature
        for temperature in temperatures
        if temperature != baseline_temperature
    ]

    for temperature in ordered_temperatures:
        differential = DifferentialObservations.from_counts(
            value1_by_temperature[temperature],
            value2_by_temperature[temperature],
            reference=reference,
            metadata={"temperature": temperature},
        )
        observations = PUFObservations.from_binary(
            responses_by_temperature[temperature],
            reference,
            metadata={"temperature": temperature},
        )
        estimator = NodewiseLogisticGraph(
            C=regularization,
            max_iter=max_iter,
            max_tuning_bits=max_tuning_bits,
            cv_splits=cv_splits,
            n_jobs=n_jobs,
        )
        estimator.fit(
            observations.flat_errors(),
            groups=observations.device_ids,
        )
        model = estimator.model_
        if regularization is None:
            regularization = model.regularization
        models[temperature] = model
        observations_by_temperature[temperature] = observations

        bit_roles = graph_role_frame(
            model.adjacency,
            flip_rate=observations.error_binary.mean(axis=(0, 1)),
            low_quantile=1.0 - role_quantile,
            high_quantile=role_quantile,
        )
        if temperature == baseline_temperature:
            baseline_thresholds = role_thresholds(
                model.adjacency,
                role_quantile=role_quantile,
            )

        if baseline_thresholds is None:
            raise RuntimeError("Baseline graph-role thresholds were not initialized")
        bit_roles = add_fixed_roles(bit_roles, model.adjacency, baseline_thresholds)
        bit_roles = add_margin_columns(bit_roles, differential.normalized_margin)
        bit_roles = bit_roles.with_columns(pl.lit(temperature).alias("temperature"))
        bit_frames.append(bit_roles)
        summary_frames.append(summarize_temperature_roles(bit_roles, "graph_role"))
        summary_frames.append(
            summarize_temperature_roles(bit_roles, "fixed_baseline_graph_role")
        )

    bit_metrics = pl.concat(bit_frames).sort(["temperature", "bit_id"])
    role_summary = pl.concat(summary_frames).sort(
        ["role_definition", "temperature", "graph_role"]
    )
    return TemperatureGraphStudyResult(
        bit_metrics=bit_metrics,
        role_summary=role_summary,
        graph_drift=graph_drift_frame(
            models,
            baseline_temperature,
            strong_edge_quantile=strong_edge_quantile,
            top_edge_k=top_edge_k,
        ),
        role_transitions=role_transition_frame(bit_metrics, baseline_temperature),
        bit_susceptibility=bit_susceptibility_frame(bit_metrics, baseline_temperature),
        model_transfer=model_transfer_frame(models, observations_by_temperature),
        models=models,
        regularization=float(regularization),
    )


def role_thresholds(adjacency, *, role_quantile: float = 0.75):
    return graph_role_thresholds(
        adjacency,
        low_quantile=1.0 - role_quantile,
        high_quantile=role_quantile,
    )


def add_fixed_roles(bit_roles: pl.DataFrame, adjacency, thresholds: dict):
    A = prepare_adjacency(adjacency, absolute=True)
    fixed_roles = [
        classify_role(out_value, in_value, thresholds)
        for out_value, in_value in zip(A.sum(axis=0), A.sum(axis=1))
    ]
    return bit_roles.with_columns(
        pl.Series(
            "fixed_baseline_graph_role",
            [role.value for role in fixed_roles],
        )
    )


def add_margin_columns(bit_roles: pl.DataFrame, margin):
    mean_margin = margin.mean(axis=(0, 1))
    mean_abs_margin = np.abs(margin).mean(axis=(0, 1))
    margin_std = margin.std(axis=(0, 1))
    return bit_roles.with_columns(
        pl.Series("mean_margin", mean_margin),
        pl.Series("mean_abs_margin", mean_abs_margin),
        pl.Series("margin_std", margin_std),
    )


def summarize_temperature_roles(bit_roles: pl.DataFrame, role_column: str):
    temperature = int(bit_roles["temperature"][0])
    role_definition = (
        "fixed_baseline_threshold"
        if role_column == "fixed_baseline_graph_role"
        else "per_temperature_quantile"
    )
    rows = []
    for role in ROLES:
        role_bits = bit_roles.filter(pl.col(role_column) == role)
        rows.append(
            {
                "temperature": temperature,
                "role_definition": role_definition,
                "graph_role": role,
                "n_bits": role_bits.height,
                "fraction_bits": role_bits.height / bit_roles.height,
                **distribution_summary(role_bits, "flip_rate"),
                **distribution_summary(role_bits, "mean_abs_margin", "mean_abs_margin"),
                **distribution_summary(role_bits, "mean_margin", "mean_margin"),
            }
        )
    return pl.DataFrame(rows)


def distribution_summary(
    frame: pl.DataFrame, column: str, prefix: Optional[str] = None
):
    prefix = column if prefix is None else prefix
    if frame.is_empty():
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
        }
    values = frame[column].to_numpy()
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def graph_drift_frame(
    models: Mapping[int, DirectedErrorGraphModel],
    baseline_temperature: int,
    *,
    strong_edge_quantile: float = 0.9,
    top_edge_k: int = 256,
):
    baseline = models[baseline_temperature]
    baseline_adjacency = baseline.undirected_adjacency
    A0 = prepare_adjacency(baseline_adjacency, absolute=True)
    fixed_threshold = positive_quantile(A0, strong_edge_quantile)
    baseline_edges = strong_edge_mask(A0, fixed_threshold)
    baseline_top = top_edge_set(A0, top_edge_k)
    rows = []
    for temperature, model in models.items():
        adjacency = model.undirected_adjacency
        A = prepare_adjacency(adjacency, absolute=True)
        edges = strong_edge_mask(A, fixed_threshold)
        rows.append(
            {
                "temperature": temperature,
                "frobenius_from_baseline": float(
                    np.linalg.norm(adjacency - baseline_adjacency)
                ),
                "cosine_with_baseline": cosine_similarity(
                    baseline_adjacency, adjacency
                ),
                "pearson_with_baseline": pearson_similarity(
                    baseline_adjacency, adjacency
                ),
                "spearman_with_baseline": spearman_similarity(
                    baseline_adjacency, adjacency
                ),
                "sign_agreement_with_baseline": sign_agreement(
                    baseline_adjacency, adjacency
                ),
                "top_edge_overlap": len(baseline_top & top_edge_set(A, top_edge_k))
                / max(len(baseline_top), 1),
                "n_fixed_strong_edges": int(edges.sum()),
                "n_new_fixed_strong_edges": int(
                    np.logical_and(edges, ~baseline_edges).sum()
                ),
                "n_lost_fixed_strong_edges": int(
                    np.logical_and(baseline_edges, ~edges).sum()
                ),
                "fixed_strong_edge_jaccard": mask_jaccard(baseline_edges, edges),
            }
        )
    return pl.DataFrame(rows).sort("temperature")


def role_transition_frame(bit_metrics: pl.DataFrame, baseline_temperature: int):
    baseline = bit_metrics.filter(pl.col("temperature") == baseline_temperature)
    baseline_roles = baseline.select(["bit_id", "graph_role"]).rename(
        {"graph_role": "baseline_role"}
    )
    rows = []
    for temperature in bit_metrics["temperature"].unique().sort().to_list():
        current = bit_metrics.filter(pl.col("temperature") == temperature)
        joined = current.select(["bit_id", "graph_role"]).join(
            baseline_roles,
            on="bit_id",
        )
        for role in ROLES:
            base = set(
                joined.filter(pl.col("baseline_role") == role)["bit_id"].to_list()
            )
            current_role = set(
                joined.filter(pl.col("graph_role") == role)["bit_id"].to_list()
            )
            rows.append(
                {
                    "temperature": temperature,
                    "baseline_role": role,
                    "current_role": role,
                    "metric": "jaccard",
                    "value": set_jaccard(base, current_role),
                }
            )
        changed = joined.filter(pl.col("baseline_role") != pl.col("graph_role")).height
        rows.append(
            {
                "temperature": temperature,
                "baseline_role": "all",
                "current_role": "all",
                "metric": "role_changed_fraction",
                "value": changed / joined.height,
            }
        )
        for baseline_role in ROLES:
            for current_role in ROLES:
                count = joined.filter(
                    (pl.col("baseline_role") == baseline_role)
                    & (pl.col("graph_role") == current_role)
                ).height
                rows.append(
                    {
                        "temperature": temperature,
                        "baseline_role": baseline_role,
                        "current_role": current_role,
                        "metric": "transition_count",
                        "value": count,
                    }
                )
    return pl.DataFrame(rows).sort(["temperature", "metric", "baseline_role"])


def bit_susceptibility_frame(bit_metrics: pl.DataFrame, baseline_temperature: int):
    baseline = bit_metrics.filter(pl.col("temperature") == baseline_temperature)
    rows = []
    temperatures = bit_metrics["temperature"].unique().sort().to_numpy()
    for bit_id in bit_metrics["bit_id"].unique().sort().to_list():
        bit = bit_metrics.filter(pl.col("bit_id") == bit_id).sort("temperature")
        baseline_bit = baseline.filter(pl.col("bit_id") == bit_id)
        row = {
            "bit_id": bit_id,
            "baseline_graph_role": baseline_bit["graph_role"][0],
            "baseline_fixed_graph_role": baseline_bit["fixed_baseline_graph_role"][0],
            "baseline_flip_rate": float(baseline_bit["flip_rate"][0]),
            "baseline_out_strength": float(baseline_bit["out_strength"][0]),
            "baseline_in_strength": float(baseline_bit["in_strength"][0]),
            "baseline_total_strength": float(baseline_bit["total_strength"][0]),
            "baseline_mean_abs_margin": float(baseline_bit["mean_abs_margin"][0]),
        }
        for column in [
            "flip_rate",
            "out_strength",
            "in_strength",
            "total_strength",
            "mean_margin",
            "mean_abs_margin",
            "margin_std",
        ]:
            row[f"{column}_slope"] = linear_slope(temperatures, bit[column].to_numpy())
            row[f"{column}_spearman"] = safe_spearman(
                temperatures,
                bit[column].to_numpy(),
            )
        rows.append(row)
    return pl.DataFrame(rows).sort("bit_id")


def model_transfer_frame(
    models: Mapping[int, DirectedErrorGraphModel],
    observations_by_temperature: Mapping[int, PUFObservations],
):
    rows = []
    for source_temperature, model in models.items():
        source_errors = observations_by_temperature[source_temperature].flat_errors()
        independent_p = np.clip((source_errors > 0).mean(axis=0), 1e-6, 1.0 - 1e-6)
        for target_temperature, observations in observations_by_temperature.items():
            target_errors = observations.flat_errors()
            rows.append(
                {
                    "source_temperature": source_temperature,
                    "target_temperature": target_temperature,
                    "nodewise_log_loss": nodewise_log_loss(model, target_errors),
                    "independent_log_loss": independent_log_loss(
                        independent_p,
                        target_errors,
                    ),
                }
            )
    return pl.DataFrame(rows).sort(["source_temperature", "target_temperature"])


def nodewise_log_loss(model: DirectedErrorGraphModel, errors):
    y = (errors > 0).astype(float)
    logits = errors @ model.coefficients.T + model.intercepts
    p = np.clip(expit(logits), 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def independent_log_loss(probability, errors):
    y = (errors > 0).astype(float)
    p = np.clip(probability[np.newaxis, :], 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def strong_edge_mask(A, threshold):
    mask = np.triu(A >= threshold, k=1)
    return mask if threshold > 0 else np.zeros_like(mask, dtype=bool)


def top_edge_set(A, k):
    upper = np.triu_indices_from(A, k=1)
    values = A[upper]
    if values.size == 0:
        return set()
    k = min(k, values.size)
    order = np.argpartition(values, -k)[-k:]
    return set(zip(upper[0][order].tolist(), upper[1][order].tolist()))


def cosine_similarity(A, B):
    a = np.ravel(A)
    b = np.ravel(B)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def pearson_similarity(A, B):
    a = np.ravel(A)
    b = np.ravel(B)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman_similarity(A, B):
    value = spearmanr(np.ravel(A), np.ravel(B)).statistic
    return float(value) if np.isfinite(value) else np.nan


def sign_agreement(A, B):
    mask = (A != 0.0) | (B != 0.0)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.sign(A[mask]) == np.sign(B[mask])))


def mask_jaccard(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return float(np.logical_and(a, b).sum() / union)


def set_jaccard(a, b):
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def linear_slope(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], deg=1)[0])


def safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2 or np.std(y[mask]) == 0.0:
        return np.nan
    value = spearmanr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else np.nan
