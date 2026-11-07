import numpy as np
import numpy.typing as npt

from pufre.data import PUFObservations


def artifact_qc_summary(observations: PUFObservations):
    """Dataset-level diagnostics for an artifact-backed PUF family."""
    error_binary = observations.error_binary
    flip_rate = error_binary.mean(axis=(0, 1))
    per_device_flip_rate = error_binary.mean(axis=1)
    flat_errors = observations.flat_errors()
    variable_error_bits = int(np.sum((flip_rate > 0.0) & (flip_rate < 1.0)))

    return {
        "n_devices": observations.n_devices,
        "n_samples": observations.n_samples,
        "n_bits": observations.n_bits,
        "n_observations": int(flat_errors.shape[0]),
        "unique_error_rows": int(np.unique(flat_errors, axis=0).shape[0]),
        "variable_error_bits": variable_error_bits,
        "constant_error_bits": int(observations.n_bits - variable_error_bits),
        "flip_rate_min": float(np.min(flip_rate)),
        "flip_rate_mean": float(np.mean(flip_rate)),
        "flip_rate_max": float(np.max(flip_rate)),
        "per_device_flip_rate_min": float(np.min(per_device_flip_rate)),
        "per_device_flip_rate_mean": float(np.mean(per_device_flip_rate)),
        "per_device_flip_rate_max": float(np.max(per_device_flip_rate)),
    }


def effective_independent_bits(error_binary: npt.NDArray):
    """Participation-ratio estimate after removing device-specific means.

    This is descriptive, not a cryptographic entropy bound.
    """
    X = np.asarray(error_binary, dtype=float)
    if X.ndim == 3:
        X -= X.mean(axis=1, keepdims=True)
        X = X.reshape(-1, X.shape[-1])
    elif X.ndim == 2:
        X -= X.mean(axis=0, keepdims=True)
    else:
        raise ValueError("error_binary must be a 2D or 3D array")

    std = X.std(axis=0, keepdims=True)
    nonconstant = std.ravel() > 0.0
    if not np.any(nonconstant):
        return 0.0

    X[:, nonconstant] /= std[:, nonconstant]
    X[:, ~nonconstant] = 0.0
    corr = (X.T @ X) / X.shape[0]
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denom = np.sum(eigenvalues**2)
    if denom == 0.0:
        return 0.0
    return float((np.sum(eigenvalues) ** 2) / denom)


def expected_independent_unique_rows(
    error_binary: npt.NDArray,
    *,
    n_simulations: int = 200,
    random_state: int = 0,
):
    """Expected unique rows under device-conditional Bernoulli marginals."""
    X = np.asarray(error_binary, dtype=np.uint8)
    if X.ndim != 3:
        raise ValueError("error_binary must have shape (device, sample, bit)")
    if n_simulations < 1:
        raise ValueError("n_simulations must be positive")

    flip_rate = X.mean(axis=1, keepdims=True)
    rng = np.random.default_rng(random_state)
    counts = np.empty(n_simulations, dtype=float)
    for index in range(n_simulations):
        simulated = rng.random(X.shape) < flip_rate
        counts[index] = np.unique(
            simulated.reshape(-1, X.shape[-1]), axis=0
        ).shape[0]
    return float(counts.mean())


def as_error_matrix(error_binary: npt.NDArray):
    X = np.asarray(error_binary, dtype=np.uint8)
    if X.ndim != 2:
        raise ValueError("error_binary must be a 2D matrix")
    return X
