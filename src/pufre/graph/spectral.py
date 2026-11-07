from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.base import prepare_adjacency


def spectral_metric_frame(
    W_asym: npt.NDArray,
    *,
    bit_ids: Optional[npt.ArrayLike] = None,
):
    """Per-bit spectral coordinates for weighted error graphs."""
    directed = prepare_adjacency(W_asym, absolute=True).T
    undirected = (directed + directed.T) / 2.0
    n_bits = undirected.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    eigenvector = principal_eigenvector(undirected)
    fiedler = fiedler_vector(undirected)
    source_svd, sink_svd, singular_value = directed_svd_coordinates(directed)

    return pl.DataFrame(
        {
            "bit_id": bit_ids,
            "adjacency_eigenvector": eigenvector,
            "fiedler_coordinate": fiedler,
            "svd_source_coordinate": source_svd,
            "svd_sink_coordinate": sink_svd,
            "svd_primary_singular_value": np.full(n_bits, singular_value),
            "spectral_participation": source_svd**2 + sink_svd**2,
        }
    )


def principal_eigenvector(A: npt.NDArray):
    if not np.any(A):
        return np.zeros(A.shape[0], dtype=float)
    values, vectors = np.linalg.eigh(A)
    vector = vectors[:, np.argmax(values)]
    return np.abs(vector)


def fiedler_vector(A: npt.NDArray):
    if not np.any(A):
        return np.zeros(A.shape[0], dtype=float)
    graph = nx.from_numpy_array(A)
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    values, vectors = np.linalg.eigh(laplacian)
    if values.size < 2:
        return np.zeros(A.shape[0], dtype=float)
    return vectors[:, 1]


def directed_svd_coordinates(A: npt.NDArray):
    if not np.any(A):
        return (
            np.zeros(A.shape[0], dtype=float),
            np.zeros(A.shape[0], dtype=float),
            0.0,
        )
    left, singular_values, right_t = np.linalg.svd(A, full_matrices=False)
    return np.abs(right_t[0]), np.abs(left[:, 0]), float(singular_values[0])
