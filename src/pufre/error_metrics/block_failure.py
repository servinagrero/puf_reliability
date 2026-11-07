import numpy as np
import numpy.typing as npt

from pufre.error_metrics.qc import as_error_matrix


def check_block_size(block_size: int, n_bits: int):
    if block_size > n_bits:
        raise ValueError("block_size cannot exceed number of bits")


def empirical_block_failure_probability(
    error_binary: npt.NDArray,
    *,
    block_size: int = 256,
    correction_capacity: int = 8,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """
    Estimate ECC block failure from empirical correlated error rows.

    A block fails when the number of flipped bits exceeds ``correction_capacity``.
    """
    counts = empirical_block_error_counts(
        error_binary,
        block_size=block_size,
        n_blocks=n_blocks,
        random_state=random_state,
    )
    return block_failure_probability(counts, correction_capacity)


def empirical_block_failure_curve(
    error_binary: npt.NDArray,
    *,
    block_size: int = 256,
    max_correction_capacity: int = 26,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """Empirical correlated block-failure curve for capacities 0..T."""
    counts = empirical_block_error_counts(
        error_binary,
        block_size=block_size,
        n_blocks=n_blocks,
        random_state=random_state,
    )
    return block_failure_curve(counts, max_correction_capacity)


def empirical_block_error_counts(
    error_binary: npt.NDArray,
    *,
    block_size: int = 256,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """Sample empirical correlated block error counts from observed error rows."""
    X = as_error_matrix(error_binary)
    check_block_size(block_size, X.shape[1])

    rng = np.random.default_rng(random_state)
    counts = np.zeros(n_blocks, dtype=int)
    for idx in range(n_blocks):
        row = X[rng.integers(0, X.shape[0])]
        bits = rng.choice(X.shape[1], size=block_size, replace=False)
        counts[idx] = int(np.sum(row[bits]))
    return counts


def independent_block_failure_probability(
    flip_rate: npt.NDArray,
    *,
    block_size: int = 256,
    correction_capacity: int = 8,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """Block failure under pooled or device-conditional Bernoulli marginals."""
    counts = independent_block_error_counts(
        flip_rate,
        block_size=block_size,
        n_blocks=n_blocks,
        random_state=random_state,
    )
    return block_failure_probability(counts, correction_capacity)


def independent_block_failure_curve(
    flip_rate: npt.NDArray,
    *,
    block_size: int = 256,
    max_correction_capacity: int = 26,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """Independent pooled or device-conditional block-failure curve."""
    counts = independent_block_error_counts(
        flip_rate,
        block_size=block_size,
        n_blocks=n_blocks,
        random_state=random_state,
    )
    return block_failure_curve(counts, max_correction_capacity)


def independent_block_error_counts(
    flip_rate: npt.NDArray,
    *,
    block_size: int = 256,
    n_blocks: int = 1000,
    random_state: int = 0,
):
    """Sample independent errors from per-bit or device-by-bit flip rates."""
    flip_rate = np.asarray(flip_rate, dtype=float)
    if flip_rate.ndim not in (1, 2):
        raise ValueError("flip_rate must have shape (bit,) or (device, bit)")
    if np.any((flip_rate < 0.0) | (flip_rate > 1.0)):
        raise ValueError("flip_rate values must lie in [0, 1]")
    check_block_size(block_size, flip_rate.shape[-1])

    rng = np.random.default_rng(random_state)
    counts = np.zeros(n_blocks, dtype=int)
    for idx in range(n_blocks):
        rates = (
            flip_rate
            if flip_rate.ndim == 1
            else flip_rate[rng.integers(flip_rate.shape[0])]
        )
        bits = rng.choice(rates.size, size=block_size, replace=False)
        flips = rng.binomial(1, rates[bits])
        counts[idx] = int(np.sum(flips))
    return counts


def block_failure_probability(error_counts: npt.NDArray, correction_capacity: int):
    """Probability that a block contains more errors than can be corrected."""
    return float(np.mean(np.asarray(error_counts) > correction_capacity))


def block_failure_curve(
    error_counts: npt.NDArray,
    max_correction_capacity: int,
):
    """Failure probabilities for correction capacities 0..T."""
    counts = np.asarray(error_counts)
    capacities = np.arange(max_correction_capacity + 1)
    return np.mean(counts[:, np.newaxis] > capacities[np.newaxis, :], axis=0)
