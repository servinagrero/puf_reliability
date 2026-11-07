from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.base import positive_quantile, prepare_adjacency
from pufre.graph.roles import Role


def avalanche_metric_frame(
    W_asym: npt.NDArray,
    *,
    bit_ids: Optional[npt.ArrayLike] = None,
    threshold_quantile: float = 0.9,
):
    """Estimate static error reachability in the directed influence graph."""
    influence = prepare_adjacency(W_asym, absolute=True).T
    n_bits = influence.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    threshold = positive_quantile(influence, threshold_quantile)
    binary = (
        (influence >= threshold).astype(float)
        if threshold > 0
        else np.zeros_like(influence)
    )
    np.fill_diagonal(binary, 0.0)
    graph = nx.from_numpy_array(binary, create_using=nx.DiGraph)

    downstream_weight = influence.sum(axis=1)
    upstream_weight = influence.sum(axis=0)
    rows = []
    for bit in range(n_bits):
        distances = nx.single_source_shortest_path_length(graph, bit, cutoff=3)
        reachable = {node: depth for node, depth in distances.items() if node != bit}
        reach_1 = sum(depth <= 1 for depth in reachable.values())
        reach_2 = sum(depth <= 2 for depth in reachable.values())
        reach_3 = sum(depth <= 3 for depth in reachable.values())
        rows.append(
            {
                "bit_id": bit_ids[bit],
                "reachable_1hop": reach_1,
                "reachable_2hop": reach_2,
                "reachable_3hop": reach_3,
                "reachability_fraction": reach_3 / max(n_bits - 1, 1),
                "downstream_weight": downstream_weight[bit],
                "upstream_weight": upstream_weight[bit],
                "avalanche_gain": downstream_weight[bit]
                / (upstream_weight[bit] + 1e-12),
                "avalanche_threshold": threshold,
            }
        )
    return pl.DataFrame(rows)


def role_distance_frame(
    W_asym: npt.NDArray,
    *,
    bit_ids: Optional[npt.ArrayLike] = None,
    roles: npt.ArrayLike,
    threshold_quantile: float = 0.9,
):
    """Shortest directed graph distances to role classes."""
    influence = prepare_adjacency(W_asym, absolute=True).T
    n_bits = influence.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    roles = np.asarray(roles)
    threshold = positive_quantile(influence, threshold_quantile)
    binary = (
        (influence >= threshold).astype(float)
        if threshold > 0
        else np.zeros_like(influence)
    )
    np.fill_diagonal(binary, 0.0)
    graph = nx.from_numpy_array(binary, create_using=nx.DiGraph)
    reverse_graph = graph.reverse(copy=True)

    columns = {"bit_id": bit_ids}
    for role in [Role.SOURCE, Role.SINK, Role.RELAY]:
        role_nodes = np.flatnonzero(roles == role)
        columns[f"distance_to_nearest_{role}"] = nearest_shortest_paths(
            graph,
            role_nodes,
            n_bits,
        )
        columns[f"distance_from_nearest_{role}"] = nearest_shortest_paths(
            reverse_graph,
            role_nodes,
            n_bits,
        )
    return pl.DataFrame(columns)


def nearest_shortest_paths(graph, targets: npt.NDArray, n_bits: int):
    if len(targets) == 0:
        return np.full(n_bits, np.nan, dtype=float)
    distances = np.full(n_bits, np.nan, dtype=float)
    for bit in range(n_bits):
        lengths = nx.single_source_shortest_path_length(graph, bit)
        values = [
            lengths[target] for target in targets if target != bit and target in lengths
        ]
        if values:
            distances[bit] = float(min(values))
    return distances
