from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.base import prepare_adjacency
from pufre.graph.roles import Role


def effective_resistance_frame(
    W: npt.NDArray,
    *,
    bit_ids: Optional[npt.ArrayLike] = None,
    roles: Optional[npt.ArrayLike] = None,
):
    """Compute resistance distances on the symmetrized weighted graph."""
    A = prepare_adjacency(W, absolute=True)
    n_bits = A.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    roles = None if roles is None else np.asarray(roles)
    resistance = effective_resistance_matrix(A)
    finite = np.where(np.isfinite(resistance), resistance, np.nan)
    role_distances = {}
    if roles is not None:
        for role in Role:
            role_distances[f"resistance_to_nearest_{role}"] = nearest_values(
                finite,
                roles == role,
            )

    offdiag = finite.copy()
    np.fill_diagonal(offdiag, np.nan)
    mean_resistance = row_nanmean(offdiag)
    median_resistance = row_nanmedian(offdiag)
    columns = {
        "bit_id": bit_ids,
        "mean_effective_resistance": mean_resistance,
        "median_effective_resistance": median_resistance,
        "resistance_closeness": 1.0 / (mean_resistance + 1e-12),
    }
    columns.update(role_distances)
    return pl.DataFrame(columns)


def effective_resistance_matrix(A: npt.NDArray):
    n_bits = A.shape[0]
    resistance = np.full((n_bits, n_bits), np.inf, dtype=float)
    graph = nx.from_numpy_array(A)
    if graph.number_of_edges() == 0:
        np.fill_diagonal(resistance, 0.0)
        return resistance

    for component in nx.connected_components(graph):
        nodes = np.array(sorted(component))
        sub_A = A[np.ix_(nodes, nodes)]
        degrees = np.diag(sub_A.sum(axis=1))
        laplacian = degrees - sub_A
        laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
        diag = np.diag(laplacian_pinv)
        sub_resistance = diag[:, None] + diag[None, :] - 2.0 * laplacian_pinv
        sub_resistance = np.maximum(sub_resistance, 0.0)
        resistance[np.ix_(nodes, nodes)] = sub_resistance
    np.fill_diagonal(resistance, 0.0)
    return resistance


def nearest_values(values: npt.NDArray, mask: npt.NDArray):
    output = np.full(values.shape[0], np.nan, dtype=float)
    targets = np.flatnonzero(mask)
    for bit in range(values.shape[0]):
        candidates = targets[targets != bit]
        if candidates.size:
            finite = values[bit, candidates]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                output[bit] = float(np.min(finite))
    return output


def row_nanmean(values: npt.NDArray):
    output = np.full(values.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            output[idx] = float(np.mean(finite))
    return output


def row_nanmedian(values: npt.NDArray):
    output = np.full(values.shape[0], np.nan, dtype=float)
    for idx, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            output[idx] = float(np.median(finite))
    return output
