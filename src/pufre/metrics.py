import numpy as np
import numpy.typing as npt


def binary_entropy(p: npt.ArrayLike, *, eps: float = 1e-12):
    """Compute bitwise Shannon entropy for Bernoulli probabilities."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def bit_aliasing_entropy(gt_bin: npt.NDArray):
    """UPC-style entropy from the population distribution of golden bits."""
    return binary_entropy(np.mean(gt_bin, axis=0))


def temporal_instability_entropy(crps_bin: npt.NDArray):
    """
    Mean temporal entropy per bit.

    ``crps_bin`` has shape ``(n_devices, n_samples, n_bits)``. The entropy is
    computed over repeated samples for each device and averaged over devices.
    """
    response_probability = np.mean(crps_bin, axis=1)
    return np.mean(binary_entropy(response_probability), axis=0)


def reliable_entropy_score(crps_bin: npt.NDArray, gt_bin: npt.NDArray):
    """
    Rank bits by uniqueness and stability.

    Reliable Entropy = UPC * Stability
    UPC = H(bit aliasing across devices)
    Stability = 1 - H(average response over time)
    """
    upc = bit_aliasing_entropy(gt_bin)
    instability = temporal_instability_entropy(crps_bin)
    stability = 1.0 - instability
    return upc * stability


def bit_reliability(crps_polar: npt.NDArray, gt_polar: npt.NDArray):
    """Mean agreement with the golden response for each bit."""
    matches = crps_polar == gt_polar[:, np.newaxis, :]
    return np.mean(matches, axis=(0, 1))
