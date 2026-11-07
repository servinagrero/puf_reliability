from __future__ import annotations

from pathlib import Path
from typing import Optional, Self, Union

import numpy as np
import numpy.typing as npt
import sparse
from sklearn.preprocessing import FunctionTransformer


class PUFData:
    """

    We use the following transformation from polar to binary

    {0, 1} -> (-1)^x -> {1, -1}

    Args:
        crps: 3d POLAR array of shape (n_devices, n_samples, n_bits)
        gt_polar: 2d polar array of reference responses

        gt_bin:
        X_bin_flat:
        Y_polar_expanded:
        Y_bin_expanded:
    """

    def __init__(
        self,
        crps: npt.NDArray,
        gt_polar: Optional[npt.NDArray] = None,
        *,
        bit_ids: Optional[npt.ArrayLike] = None,
        metadata: Optional[dict] = None,
    ):
        n_devices, n_samples, n_bits = crps.shape
        self.crps = crps
        self.bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
        self.metadata = {} if metadata is None else dict(metadata)

        # Ground Truth (Golden Key)
        if gt_polar is not None:
            self.gt_polar: npt.NDArray = gt_polar
        else:
            self.gt_polar = np.where(self.crps.mean(axis=1) >= 0, 1, -1).astype(
                int
            )  # {-1, 1}

        X_polar_flat, device_labels = flatten_crp_array(crps)

        self.X_polar_flat = X_polar_flat
        self.device_labels = device_labels

    def fit_lasso_model(self, *, order: int = 1, C: Optional[float] = None, **kwargs):
        """
        Fit a sparse conditional PUF model from this dataset.

        The returned model stores logistic coefficients as ``W`` and ``bias``.
        Use ``model.J`` and ``model.h`` for Ising-scaled Hamiltonian quantities.
        """
        from pufre.logreg import PUFStructureLearner

        learner = PUFStructureLearner(
            self.X_polar_flat,
            self.Y_bin_expanded,
            groups=self.device_labels,
            **kwargs,
        )
        return learner.fit(order=order, C=C)

    def select_bits(self, bit_ids: npt.ArrayLike) -> PUFData:
        """Return a new ``PUFData`` view restricted to selected bit columns."""
        bit_ids = np.asarray(bit_ids, dtype=int)
        metadata = dict(self.metadata)
        current_physical_ids = metadata.get("selected_bit_indices", self.bit_ids)
        metadata["selected_bit_indices"] = np.asarray(current_physical_ids)[bit_ids]
        metadata["n_bits"] = int(bit_ids.size)
        return PUFData(
            self.crps[:, :, bit_ids],
            self.gt_polar[:, bit_ids],
            bit_ids=np.arange(bit_ids.size),
            metadata=metadata,
        )

    @staticmethod
    def save_model(path: Union[str, Path], model):
        """Persist first-order weights and metadata from a learned PUF model."""
        np.savez_compressed(
            path,
            W=model.W,
            bias=model.bias,
            order=model.order,
            selected_challenges=np.array(model.selected_challenges, dtype=object),
            params=np.array(model.params, dtype=object),
            W_orders=np.array(model.W_orders, dtype=object),
        )

    @staticmethod
    def load_model(path: Union[str, Path]):
        """Load a model saved with ``save_model``."""
        from pufre.logreg import LassoPUFModel

        with np.load(path, allow_pickle=True) as data:
            return LassoPUFModel(
                W=data["W"],
                bias=data["bias"],
                order=int(data["order"]),
                selected_challenges=list(data["selected_challenges"]),
                params=list(data["params"]),
                W_orders=data["W_orders"].item(),
            )

    @classmethod
    def from_binary(
        cls,
        crps: npt.NDArray,
        gt_bin: Optional[npt.NDArray] = None,
        *,
        bit_ids: Optional[npt.ArrayLike] = None,
        metadata: Optional[dict] = None,
    ):
        if gt_bin is None:
            return cls(
                BinaryToPolarTransformer.transform(crps),
                bit_ids=bit_ids,
                metadata=metadata,
            )
        return cls(
            BinaryToPolarTransformer.transform(crps),
            BinaryToPolarTransformer.transform(gt_bin),
            bit_ids=bit_ids,
            metadata=metadata,
        )

    @classmethod
    def from_family_responses(cls, family):
        return cls.from_binary(
            family.responses,
            family.reference,
            metadata=family.metadata,
        )

    def select_by_bit_ids(
        self, bit_ids: npt.ArrayLike, *, expected_n_bits: Optional[int] = None
    ) -> Self:
        """
        Select artifact-backed PUF data by local study bit IDs.

        The artifact already stores the selected response/reference matrices. Physical source bit indices are metadata only; analysis code uses local coordinates ``0..n_bits-1``.

        Raises:
            Value error:
        """
        bit_ids = np.asarray(bit_ids, dtype=int)
        if expected_n_bits is not None and bit_ids.shape[0] != expected_n_bits:
            raise ValueError(
                f"saved bit selection has {bit_ids.shape[0]} bits, but expected "
                f"{expected_n_bits}"
            )

        if (
            self.crps.shape[2] == bit_ids.shape[0]
            and np.array_equal(self.bit_ids, bit_ids)
            or np.array_equal(bit_ids, np.arange(bit_ids.shape[0]))
        ):
            return self

        if expected_n_bits is not None and self.crps.shape[2] == expected_n_bits:
            return PUFData(
                self.crps,
                self.gt_polar,
                bit_ids=np.arange(expected_n_bits),
                metadata=self.metadata,
            ).with_metadata({"n_bits": int(expected_n_bits)})

        if np.any(bit_ids < 0) or np.any(bit_ids >= self.crps.shape[2]):
            raise ValueError("saved local bit selection is outside artifact width")
        return self.select_bits(bit_ids)

    def with_metadata(self, metadata: dict):
        self.metadata.update(metadata)
        return self

    @property
    def gt_bin(self):
        return (self.gt_polar < 0).astype(np.int8)  # {0, 1}

    @property
    def X_bin_flat(self):
        return (self.X_polar_flat < 0).astype(np.int8)  # {0, 1}

    @property
    def Y_polar_expanded(self):
        return self.gt_polar[self.device_labels]  # {-1, 1}

    @property
    def Y_bin_expanded(self):
        return self.gt_bin[self.device_labels]  # {0, 1}


def xlog2x(x):
    if x == 0 or x == 1:
        return 0.0
    return x * np.log2(x)


def shannon_ent(p):
    return -(xlog2x(p) + xlog2x(1 - p))


def helper_data_cost(n_bits: int, n_err: int):
    """
    n: Number of bits in the response
    n_err: Number of errors to correct
    """
    return {
        "ecc": np.log2(n_bits) * n_err,
        "bch": np.ceil(np.log2(n_bits + 1)) * n_err,
        "fe": n_bits * shannon_ent(n_err / n_bits),
    }


def binary_to_polar(X: npt.NDArray):
    return np.where(np.asarray(X) == 0, 1, -1).astype(int)


#: Binary {0, 1} to Polar {1, -1} Transformer
BinaryToPolarTransformer = FunctionTransformer(binary_to_polar, validate=False)


def polar_to_binary(X: npt.NDArray):
    # return ((X + 1) // 2).astype(int)
    return (X < 0).astype(int)


#: Polar {1, -1} to Binary {0, 1} Transformer
PolarToBinaryTransformer = FunctionTransformer(polar_to_binary, validate=False)


logit_to_binary = polar_to_binary


def flatten_crp_array(X: npt.NDArray):
    """Collapse a 3D array `X` into a 2D array `X_flat` and a vector `labels`

    This function assumes `X` has dimensions (m_devices, l_samples, n_bits)

    Returns:
        A tuple containing X_flat and device_labels:
            - X_flat a 2D matrix of size ((m*l), n), where every single row is one noisy observation from one of the devices.
            - device_labels is an array of length (m*l) that tells the model which device produced each row.
    """

    n_devices, n_samples, n_bits = X.shape

    # Flatten to (n_devices*n_samples, n_bits)
    X_flat = X.reshape(-1, n_bits)

    # Create labels: [0,0,0... (n_samples times), 1,1,1... (n_samples times), ..., m_devices-1...]
    labels = np.repeat(np.arange(n_devices), n_samples)

    return X_flat, labels


def prune_matrix(W: npt.NDArray, k: Union[int, float]):
    """ """
    pruned = np.zeros_like(W)

    n_samples, n_features = pruned.shape

    if isinstance(k, float):
        k = int(n_features * k)

    for row_id, row in enumerate(W):
        if np.count_nonzero(row) > k:
            threshold = np.sort(np.abs(row))[-k]
            mask = np.abs(row) >= threshold
            pruned[row_id, mask] = row[mask]
        else:
            pruned[row_id, :] = row

    return pruned


def convert_to_3d_tensor(W_2nd, n_bits: int):
    """
    Converts W_2nd dictionary into a 3rd-order sparse tensor.
    Shape: (n_bits, n_bits, n_bits)
    """
    coords = []
    data = []

    for i, interactions in W_2nd.items():
        for (j, k), weight in interactions.items():
            # Add both (i, j, k) and (i, k, j) because the interaction
            # of j*k is symmetric
            coords.append([i, j, k])
            data.append(weight)

            coords.append([i, k, j])
            data.append(weight)

    # Create the sparse COO tensor
    W_tensor = sparse.COO(
        np.array(coords).T, np.array(data), shape=(n_bits, n_bits, n_bits)
    )
    return W_tensor
