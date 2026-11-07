import numpy as np
import polars as pl

from pufre.data import DifferentialObservations, PUFObservations
from pufre.metrics import (
    bit_aliasing_entropy,
    reliable_entropy_score,
    temporal_instability_entropy,
)


def differential_bit_feature_frame(
    observations: DifferentialObservations,
    *,
    temperature: int | float | None = None,
):
    """Aggregate physical differential features per bit."""
    value1 = np.asarray(observations.value1, dtype=float)
    value2 = np.asarray(observations.value2, dtype=float)
    diff = value1 - value2
    abs_diff = np.abs(diff)
    ratio = np.divide(
        value1,
        value2,
        out=np.full_like(value1, np.nan, dtype=float),
        where=value2 != 0.0,
    )
    log_ratio = np.log(np.clip(ratio, 1e-12, None))
    normalized_diff = diff / np.maximum(value1 + value2, 1.0)
    normalized_abs_diff = np.abs(normalized_diff)

    columns = {
        "bit_id": observations.bit_ids,
        "physical_bit_id": observations.metadata.get(
            "selected_bit_indices",
            observations.bit_ids,
        ),
        "value1_mean": _mean(value1),
        "value1_std": _std(value1),
        "value2_mean": _mean(value2),
        "value2_std": _std(value2),
        "diff_mean": _mean(diff),
        "diff_std": _std(diff),
        "abs_diff_mean": _mean(abs_diff),
        "abs_diff_std": _std(abs_diff),
        "ratio_mean": _mean(ratio),
        "ratio_std": _std(ratio),
        "log_ratio_mean": _mean(log_ratio),
        "log_ratio_std": _std(log_ratio),
        "normalized_diff_mean": _mean(normalized_diff),
        "normalized_diff_std": _std(normalized_diff),
        "normalized_abs_diff_mean": _mean(normalized_abs_diff),
        "normalized_abs_diff_std": _std(normalized_abs_diff),
    }
    if temperature is not None:
        columns["temperature"] = np.full(observations.n_bits, temperature)
    return pl.DataFrame(columns)


def differential_quality_frame(
    observations: PUFObservations,
    *,
    temperature: int | float | None = None,
):
    """Compute canonical per-bit PUF quality metrics for one condition."""
    responses = observations.response_binary
    reference = observations.reference_binary
    instability = temporal_instability_entropy(responses)
    upc = bit_aliasing_entropy(reference)
    reliable_entropy = reliable_entropy_score(responses, reference)
    flip_rate = observations.error_binary.mean(axis=(0, 1))

    response_probability = responses.mean(axis=1)
    columns = {
        "bit_id": observations.bit_ids,
        "physical_bit_id": observations.physical_bit_ids,
        "reference_bit_aliasing": reference.mean(axis=0),
        "response_bit_aliasing": responses.mean(axis=(0, 1)),
        "temporal_response_probability_mean": response_probability.mean(axis=0),
        "temporal_response_probability_std": response_probability.std(axis=0),
        "upc": upc,
        "temporal_instability_entropy": instability,
        "stability": 1.0 - instability,
        "reliable_entropy": reliable_entropy,
        "flip_rate": flip_rate,
    }
    if temperature is not None:
        columns["temperature"] = np.full(observations.n_bits, temperature)
    return pl.DataFrame(columns)


def _mean(values):
    return np.nanmean(values, axis=(0, 1))


def _std(values):
    return np.nanstd(values, axis=(0, 1))
