"""Binary and polar PUF response encodings."""

import numpy as np
import numpy.typing as npt


def binary_to_polar(values: npt.ArrayLike):
    """Map binary ``{0, 1}`` values to polar ``{1, -1}`` values."""
    return np.where(np.asarray(values) == 0, 1, -1).astype(int)


def polar_to_binary(values: npt.ArrayLike):
    """Map polar ``{1, -1}`` values to binary ``{0, 1}`` values."""
    return (np.asarray(values) < 0).astype(int)
