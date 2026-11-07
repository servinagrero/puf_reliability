from collections.abc import Mapping

import numpy as np
import polars as pl

from pufre.data import PUFObservations
from pufre.error_metrics import (
    effective_independent_bits,
    empirical_block_failure_curve,
    independent_block_failure_curve,
)
from pufre.graph import ROLE_VALUES

def baseline_selection_indices(
    baseline_bit_table: pl.DataFrame,
    *,
    k: int = 256,
    margin_column: str = "abs_diff_mean",
    extreme_margin_outlier_quantile: float = 0.95,
):
    """Build raw, Reliable-Entropy, middle-margin, and extreme-margin selections."""
    n_bits = baseline_bit_table.height
    k = min(k, n_bits)
    bit_ids = baseline_bit_table["bit_id"].to_numpy()
    reliable_entropy = baseline_bit_table["reliable_entropy"].to_numpy()
    margin = baseline_bit_table[margin_column].to_numpy()

    finite_margin = margin[np.isfinite(margin)]
    if finite_margin.size:
        target_margin = float(np.nanmedian(finite_margin))
        margin_order = np.argsort(np.abs(margin - target_margin))
    else:
        target_margin = np.nan
        margin_order = np.arange(n_bits)

    extreme_order = non_outlier_extreme_margin_order(
        margin,
        outlier_quantile=extreme_margin_outlier_quantile,
    )
    selections = {
        "raw_first_k": np.sort(bit_ids[:k].astype(int)),
        "top_reliable_entropy": np.sort(
            bit_ids[np.argsort(reliable_entropy)[-k:]].astype(int)
        ),
        "middle_abs_margin": np.sort(bit_ids[margin_order[:k]].astype(int)),
        "extreme_abs_margin": np.sort(bit_ids[extreme_order[:k]].astype(int)),
    }
    return selections, target_margin


def non_outlier_extreme_margin_order(
    margin: np.ndarray,
    *,
    outlier_quantile: float = 0.95,
):
    """
    Rank high-margin bits after removing the extreme outlier tail.

    This targets low-entropy, high-reliability bits without letting a small group
    of pathological physical-margin outliers dominate the selection.
    """
    margin = np.asarray(margin, dtype=float)
    finite = np.isfinite(margin)
    if not np.any(finite):
        return np.arange(margin.size)

    cutoff = float(np.nanquantile(margin[finite], outlier_quantile))
    candidates = np.flatnonzero(finite & (margin <= cutoff))
    if candidates.size == 0:
        candidates = np.flatnonzero(finite)
    return candidates[np.argsort(margin[candidates])[::-1]]


def selection_indices_for_sizes(
    baseline_bit_table: pl.DataFrame,
    *,
    sizes: list[int] | tuple[int, ...],
    margin_column: str = "abs_diff_mean",
    extreme_margin_outlier_quantile: float = 0.95,
):
    """Build raw, Reliable-Entropy, and margin selections for several widths."""
    selections = {}
    targets = {}
    for size in sizes:
        size_selections, target_margin = baseline_selection_indices(
            baseline_bit_table,
            k=size,
            margin_column=margin_column,
            extreme_margin_outlier_quantile=extreme_margin_outlier_quantile,
        )
        for name, bit_ids in size_selections.items():
            selections[f"{name}_k{int(size)}"] = bit_ids
        targets[int(size)] = target_margin
    return selections, targets


def selection_summary_frame(
    bit_table: pl.DataFrame,
    selections: Mapping[str, np.ndarray],
    *,
    baseline_temperature: int,
):
    rows = []
    for temperature in bit_table["temperature"].unique().sort().to_list():
        temp_frame = bit_table.filter(pl.col("temperature") == temperature)
        for name, bit_ids in selections.items():
            selected = temp_frame.filter(pl.col("bit_id").is_in(bit_ids.tolist()))
            row = {
                "temperature": temperature,
                "baseline_temperature": baseline_temperature,
                "selection": name,
                "n_bits": selected.height,
            }
            for column in [
                "upc",
                "stability",
                "reliable_entropy",
                "flip_rate",
                "abs_diff_mean",
                "normalized_abs_diff_mean",
                "total_strength",
                "out_strength",
                "in_strength",
            ]:
                row.update(_distribution_summary(selected, column))
            for role in ROLE_VALUES:
                row[f"role_fraction_{role}"] = (
                    selected.filter(pl.col("graph_role") == role).height
                    / max(selected.height, 1)
                )
            rows.append(row)
    return pl.DataFrame(rows).sort(["temperature", "selection"])


def selection_block_failure_frame(
    observations_by_temperature: Mapping[int, PUFObservations],
    selections: Mapping[str, np.ndarray],
    *,
    block_size: int = 256,
    max_correction_capacity: int = 26,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    rows = []
    for temperature, observations in observations_by_temperature.items():
        for selection, bit_ids in selections.items():
            local_block_size = min(block_size, len(bit_ids))
            if local_block_size == 0:
                continue
            error_binary = observations.error_binary[:, :, bit_ids].reshape(
                -1,
                len(bit_ids),
            )
            flip_rate = error_binary.mean(axis=0)
            empirical = empirical_block_failure_curve(
                error_binary,
                block_size=local_block_size,
                max_correction_capacity=max_correction_capacity,
                n_blocks=n_blocks,
                random_state=random_state,
            )
            independent = independent_block_failure_curve(
                flip_rate,
                block_size=local_block_size,
                max_correction_capacity=max_correction_capacity,
                n_blocks=n_blocks,
                random_state=random_state,
            )
            effective_bits = effective_independent_bits(error_binary)
            for capacity in range(max_correction_capacity + 1):
                rows.append(
                    {
                        "temperature": temperature,
                        "selection": selection,
                        "block_size": local_block_size,
                        "correction_capacity": capacity,
                        "empirical_failure_probability": float(
                            empirical[capacity]
                        ),
                        "independent_failure_probability": float(
                            independent[capacity]
                        ),
                        "failure_probability_excess": float(
                            empirical[capacity] - independent[capacity]
                        ),
                        "effective_independent_error_bits": effective_bits,
                    }
                )
    return pl.DataFrame(rows).sort(
        ["temperature", "selection", "correction_capacity"]
    )


def _distribution_summary(frame: pl.DataFrame, column: str):
    if column not in frame.columns or frame.is_empty():
        return {
            f"{column}_mean": np.nan,
            f"{column}_median": np.nan,
            f"{column}_q25": np.nan,
            f"{column}_q75": np.nan,
        }
    values = frame[column].to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{column}_mean": np.nan,
            f"{column}_median": np.nan,
            f"{column}_q25": np.nan,
            f"{column}_q75": np.nan,
        }
    return {
        f"{column}_mean": float(np.mean(values)),
        f"{column}_median": float(np.median(values)),
        f"{column}_q25": float(np.quantile(values, 0.25)),
        f"{column}_q75": float(np.quantile(values, 0.75)),
    }
