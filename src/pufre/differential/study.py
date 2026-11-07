from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import polars as pl

from pufre.analysis.temperature import fit_temperature_graph_study
from pufre.data import DifferentialObservations, PUFObservations
from pufre.differential.features import (
    differential_bit_feature_frame,
    differential_quality_frame,
)
from pufre.differential.selection import (
    baseline_selection_indices,
    selection_block_failure_frame,
    selection_summary_frame,
)


@dataclass
class DifferentialTemperatureStudyResult:
    bit_table: pl.DataFrame
    role_summary: pl.DataFrame
    graph_drift: pl.DataFrame
    role_transitions: pl.DataFrame
    bit_susceptibility: pl.DataFrame
    model_transfer: pl.DataFrame
    selection_summary: pl.DataFrame
    selection_block_failure: pl.DataFrame
    selections: dict[str, np.ndarray]
    margin_target: float
    observations_by_temperature: dict[int, PUFObservations]
    models: dict
    regularization: float


def fit_differential_temperature_study(
    *,
    reference: np.ndarray,
    responses_by_temperature: Mapping[int, np.ndarray],
    value1_by_temperature: Mapping[int, np.ndarray],
    value2_by_temperature: Mapping[int, np.ndarray],
    baseline_temperature: int,
    selection_k: int = 256,
    block_size: int = 256,
    max_correction_capacity: int = 26,
    n_blocks: int = 1000,
    random_state: int = 0,
    graph_C: float | None = None,
    max_iter: int = 500,
    max_tuning_bits: int = 8,
    cv_splits: int = 2,
    n_jobs: int = 1,
    role_quantile: float = 0.75,
    strong_edge_quantile: float = 0.9,
    top_edge_k: int = 256,
):
    """Run the differential-PUF temperature study and return reusable tables."""
    graph_result = fit_temperature_graph_study(
        reference=reference,
        responses_by_temperature=responses_by_temperature,
        value1_by_temperature=value1_by_temperature,
        value2_by_temperature=value2_by_temperature,
        baseline_temperature=baseline_temperature,
        graph_C=graph_C,
        max_iter=max_iter,
        max_tuning_bits=max_tuning_bits,
        cv_splits=cv_splits,
        n_jobs=n_jobs,
        role_quantile=role_quantile,
        strong_edge_quantile=strong_edge_quantile,
        top_edge_k=top_edge_k,
    )

    bit_tables = []
    observations_by_temperature = {}
    for temperature in sorted(responses_by_temperature):
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
        observations_by_temperature[temperature] = observations

        features = differential_bit_feature_frame(
            differential,
            temperature=temperature,
        )
        quality = differential_quality_frame(
            observations,
            temperature=temperature,
        )
        graph = graph_result.bit_metrics.filter(
            pl.col("temperature") == temperature
        ).drop("flip_rate")
        bit_tables.append(
            features.join(
                quality,
                on=["temperature", "bit_id", "physical_bit_id"],
            ).join(
                graph,
                on=["temperature", "bit_id", "physical_bit_id"],
            )
        )

    bit_table = pl.concat(bit_tables).sort(["temperature", "bit_id"])
    bit_table = add_temperature_deltas(bit_table, baseline_temperature)
    baseline = bit_table.filter(pl.col("temperature") == baseline_temperature)
    selections, margin_target = baseline_selection_indices(
        baseline,
        k=selection_k,
        margin_column="abs_diff_mean",
    )

    return DifferentialTemperatureStudyResult(
        bit_table=bit_table,
        role_summary=graph_result.role_summary,
        graph_drift=graph_result.graph_drift,
        role_transitions=graph_result.role_transitions,
        bit_susceptibility=graph_result.bit_susceptibility,
        model_transfer=graph_result.model_transfer,
        selection_summary=selection_summary_frame(
            bit_table,
            selections,
            baseline_temperature=baseline_temperature,
        ),
        selection_block_failure=selection_block_failure_frame(
            observations_by_temperature,
            selections,
            block_size=block_size,
            max_correction_capacity=max_correction_capacity,
            n_blocks=n_blocks,
            random_state=random_state,
        ),
        selections=selections,
        margin_target=margin_target,
        observations_by_temperature=observations_by_temperature,
        models=graph_result.models,
        regularization=graph_result.regularization,
    )


def add_temperature_deltas(bit_table: pl.DataFrame, baseline_temperature: int):
    baseline = (
        bit_table.filter(pl.col("temperature") == baseline_temperature)
        .select(
            [
                "bit_id",
                "flip_rate",
                "stability",
                "reliable_entropy",
                "abs_diff_mean",
                "normalized_abs_diff_mean",
                "total_strength",
                "graph_role",
            ]
        )
        .rename(
            {
                "flip_rate": "baseline_flip_rate",
                "stability": "baseline_stability",
                "reliable_entropy": "baseline_reliable_entropy",
                "abs_diff_mean": "baseline_abs_diff_mean",
                "normalized_abs_diff_mean": "baseline_normalized_abs_diff_mean",
                "total_strength": "baseline_total_strength",
                "graph_role": "baseline_graph_role",
            }
        )
    )
    return bit_table.join(baseline, on="bit_id").with_columns(
        (pl.col("flip_rate") - pl.col("baseline_flip_rate")).alias(
            "delta_flip_rate"
        ),
        (pl.col("stability") - pl.col("baseline_stability")).alias(
            "delta_stability"
        ),
        (
            pl.col("reliable_entropy") - pl.col("baseline_reliable_entropy")
        ).alias("delta_reliable_entropy"),
        (pl.col("abs_diff_mean") - pl.col("baseline_abs_diff_mean")).alias(
            "delta_abs_diff_mean"
        ),
        (
            pl.col("normalized_abs_diff_mean")
            - pl.col("baseline_normalized_abs_diff_mean")
        ).alias("delta_normalized_abs_diff_mean"),
        (pl.col("total_strength") - pl.col("baseline_total_strength")).alias(
            "delta_total_strength"
        ),
        (pl.col("graph_role") != pl.col("baseline_graph_role")).alias(
            "role_changed_from_baseline"
        ),
    )
