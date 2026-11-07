import numpy as np
import numpy.typing as npt


def pairwise_mutual_information(error_binary: npt.NDArray) -> npt.NDArray:
    """Pairwise MI, conditioned on device when given a 3D error tensor.

    A 2D input computes ordinary pooled MI. For a ``(device, sample, bit)``
    tensor, the result is the device-average conditional MI.
    """
    X = np.asarray(error_binary, dtype=float)
    if X.ndim == 2:
        return _binary_pairwise_mutual_information(X)
    if X.ndim != 3:
        raise ValueError("error_binary must be a 2D or 3D array")

    mi = np.zeros((X.shape[-1], X.shape[-1]), dtype=float)
    for device_errors in X:
        mi += _binary_pairwise_mutual_information(device_errors)
    return (mi / X.shape[0]).astype(np.float32)


def _binary_pairwise_mutual_information(X):
    n_samples, n_bits = X.shape
    p_one = X.mean(axis=0)
    p11 = (X.T @ X) / n_samples
    p10 = p_one[:, np.newaxis] - p11
    p01 = p_one[np.newaxis, :] - p11
    p00 = 1.0 - p11 - p10 - p01

    marginal_entropy = binary_entropy(p_one)
    joint_entropy = sum(
        negative_xlog2x(probability)
        for probability in (p00, p01, p10, p11)
    )
    mi = (
        marginal_entropy[:, np.newaxis]
        + marginal_entropy[np.newaxis, :]
        - joint_entropy
    )
    mi = np.clip(mi, 0.0, None)
    np.fill_diagonal(mi, 0.0)
    return mi


def binary_entropy(probability):
    return negative_xlog2x(probability) + negative_xlog2x(1.0 - probability)


def negative_xlog2x(value):
    value = np.asarray(value, dtype=float)
    result = np.zeros_like(value)
    positive = value > 0.0
    result[positive] = -value[positive] * np.log2(value[positive])
    return result
