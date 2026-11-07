from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from sklearn.utils.parallel import Parallel, delayed

STATISTICS = (
    ("correlation_energy", "greater"),
    ("error_count_variance", "two-sided"),
)


def holm_adjusted_p_values(p_values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return Holm family-wise adjusted p-values in their original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")

    order = np.argsort(values)
    multipliers = values.size - np.arange(values.size)
    adjusted_sorted = np.maximum.accumulate(values[order] * multipliers)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


@dataclass(frozen=True)
class ConditionalIndependenceResult:
    observed: npt.NDArray
    null: npt.NDArray

    def summary_rows(self):
        rows = []
        for index, (name, alternative) in enumerate(STATISTICS):
            observed = float(self.observed[index])
            null = self.null[:, index]
            null_mean = float(np.mean(null))
            upper = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
            if alternative == "two-sided":
                lower = (1 + np.count_nonzero(null <= observed)) / (len(null) + 1)
                p_value = min(1.0, 2.0 * min(lower, upper))
            else:
                p_value = upper
            rows.append(
                {
                    "metric": name,
                    "alternative": alternative,
                    "observed": observed,
                    "null_mean": null_mean,
                    "null_std": float(np.std(null, ddof=1))
                    if len(null) > 1
                    else 0.0,
                    "null_q025": float(np.quantile(null, 0.025)),
                    "null_q975": float(np.quantile(null, 0.975)),
                    "observed_to_null_ratio": observed / null_mean
                    if null_mean
                    else float("inf"),
                    "p_value": float(p_value),
                    "n_permutations": len(null),
                }
            )
        return rows


def conditional_independence_test(
    error_binary: npt.NDArray,
    *,
    n_permutations: int = 199,
    random_state: int = 0,
    n_jobs: int = 1,
):
    """Test same-readout dependence conditional on device/bit marginals."""
    errors = _validate_error_tensor(error_binary)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero")

    observed = dependence_statistics(errors)
    seeds = np.random.SeedSequence(random_state).generate_state(
        n_permutations,
        dtype=np.uint64,
    )
    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_permutation_statistics)(errors, int(seed)) for seed in seeds
    )
    null = np.vstack(
        list(
            tqdm(
                results,
                total=n_permutations,
                desc="Conditional-independence permutations",
            )
        )
    )
    return ConditionalIndependenceResult(observed=observed, null=null)


def dependence_statistics(error_binary: npt.NDArray):
    """Return correlation energy and within-device error-count variance."""
    errors = _validate_error_tensor(error_binary).astype(float)
    centered = errors - errors.mean(axis=1, keepdims=True)
    flat = centered.reshape(-1, centered.shape[-1])
    norms = np.linalg.norm(flat, axis=0)
    variable = norms > 0.0

    if np.count_nonzero(variable) < 2:
        correlation_energy = 0.0
    else:
        values = flat[:, variable]
        denominator = np.outer(norms[variable], norms[variable])
        correlation = (values.T @ values) / denominator
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        correlation_energy = float(np.mean(upper**2))

    error_counts = errors.sum(axis=2)
    centered_counts = error_counts - error_counts.mean(axis=1, keepdims=True)
    error_count_variance = float(np.mean(centered_counts**2))
    return np.array([correlation_energy, error_count_variance])


def permute_columns_within_devices(
    error_binary: npt.NDArray,
    rng: np.random.Generator,
):
    """Independently permute every bit across readouts within each device."""
    errors = _validate_error_tensor(error_binary)
    order = np.argsort(rng.random(errors.shape), axis=1)
    return np.take_along_axis(errors, order, axis=1)


def _permutation_statistics(errors, seed):
    rng = np.random.default_rng(seed)
    return dependence_statistics(permute_columns_within_devices(errors, rng))


def _validate_error_tensor(error_binary):
    errors = np.asarray(error_binary, dtype=np.uint8)
    if errors.ndim != 3:
        raise ValueError("error_binary must have shape (device, sample, bit)")
    if not np.all(np.isin(errors, [0, 1])):
        raise ValueError("error_binary must be binary")
    return errors
