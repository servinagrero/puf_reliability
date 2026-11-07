from typing import Tuple

import networkx as nx
import networkx.algorithms
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.base import positive_quantile, prepare_adjacency


def pagerank(A: npt.NDArray) -> npt.NDArray:
    if not np.any(A):
        return np.zeros(A.shape[0], dtype=float)
    graph = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    scores = nx.pagerank(graph, weight="weight")
    return np.array([scores[i] for i in range(A.shape[0])])


def community_metric_frame(
    W: npt.NDArray,
    *,
    bit_ids: npt.ArrayLike | None = None,
    threshold_quantile: float = 0.9,
) -> pl.DataFrame:
    """Compute undirected community, core, bridge, and clustering metrics."""
    A = prepare_adjacency(W, absolute=True)
    n_bits = A.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    threshold = positive_quantile(A, threshold_quantile)
    binary = (A >= threshold).astype(float) if threshold > 0 else np.zeros_like(A)
    np.fill_diagonal(binary, 0.0)
    thresholded_A = A * binary

    graph = nx.from_numpy_array(binary)
    weighted_graph = nx.from_numpy_array(thresholded_A)
    communities = community_labels(graph, n_bits)
    louvain_communities = louvain_labels(weighted_graph, n_bits)
    core_number = core_number_by_node(graph, n_bits)
    betweenness = nx.betweenness_centrality(graph, normalized=True)
    clustering = nx.clustering(weighted_graph, weight="weight")
    pagerank_values = pagerank(A)
    louvain_size = label_sizes(louvain_communities)
    within_strength, bridge_strength, participation = community_strengths(
        thresholded_A,
        louvain_communities,
    )

    return pl.DataFrame(
        {
            "bit_id": bit_ids,
            "community": communities,
            "louvain_community": louvain_communities,
            "louvain_community_size": louvain_size,
            "within_community_strength": within_strength,
            "bridge_strength": bridge_strength,
            "participation_coefficient": participation,
            "core_number": np.array([core_number[i] for i in range(n_bits)]),
            "betweenness": np.array([betweenness[i] for i in range(n_bits)]),
            "clustering": np.array([clustering[i] for i in range(n_bits)]),
            "pagerank": pagerank_values,
            "threshold": np.full(n_bits, threshold),
        }
    )


def compute_clustering_coefficient(
    J: npt.NDArray, threshold: float = 0.1
) -> Tuple[float, npt.NDArray]:
    A = prepare_adjacency(J, absolute=True)
    binary = (A > threshold).astype(float)
    graph = nx.from_numpy_array(binary)
    local = nx.clustering(graph)
    return nx.transitivity(graph), np.array([local[i] for i in range(len(local))])


def community_labels(graph: nx.Graph, n_bits: int) -> npt.NDArray:
    if graph.number_of_edges() == 0:
        return np.full(n_bits, -1, dtype=int)
    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    labels = np.full(n_bits, -1, dtype=int)
    for label, community in enumerate(communities):
        labels[list(community)] = label
    return labels


def louvain_labels(graph: nx.Graph, n_bits: int) -> npt.NDArray:
    if graph.number_of_edges() == 0:
        return np.full(n_bits, -1, dtype=int)
    try:
        communities = nx.algorithms.community.louvain_communities(
            graph,
            weight="weight",
            seed=0,
        )
    except (AttributeError, ImportError):
        communities = nx.algorithms.community.greedy_modularity_communities(
            graph,
            weight="weight",
        )
    labels = np.full(n_bits, -1, dtype=int)
    for label, community in enumerate(communities):
        labels[list(community)] = label
    return labels


def label_sizes(labels: npt.NDArray) -> npt.NDArray:
    if labels.size == 0:
        return labels
    values, counts = np.unique(labels, return_counts=True)
    size_by_label = dict(zip(values, counts))
    return np.array([size_by_label[label] if label >= 0 else 0 for label in labels])


def community_strengths(A: npt.NDArray, labels: npt.NDArray) -> Tuple[npt.NDArray, ...]:
    within = np.zeros(A.shape[0], dtype=float)
    bridge = np.zeros(A.shape[0], dtype=float)
    participation = np.zeros(A.shape[0], dtype=float)
    for bit in range(A.shape[0]):
        weights = A[bit] + A[:, bit]
        total = weights.sum()
        if labels[bit] < 0 or total == 0.0:
            continue
        same = labels == labels[bit]
        within[bit] = weights[same].sum()
        bridge[bit] = weights[~same].sum()
        label_strengths = [
            weights[labels == label].sum() for label in np.unique(labels) if label >= 0
        ]
        participation[bit] = 1.0 - np.sum((np.asarray(label_strengths) / total) ** 2)
    return within, bridge, participation


def core_number_by_node(graph: nx.Graph, n_bits: int) -> dict:
    if graph.number_of_edges() == 0:
        return {idx: 0 for idx in range(n_bits)}
    return nx.core_number(graph)
