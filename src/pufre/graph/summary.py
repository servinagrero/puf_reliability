import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre.graph.base import positive_quantile, prepare_adjacency


def graph_assortativity_metrics(W: npt.NDArray, bit_graph: pl.DataFrame):
    """Graph-level mixing metrics for role, reliability, and flip-rate structure."""
    A = prepare_adjacency(W, absolute=True)
    threshold = positive_quantile(A, 0.9)
    binary = (A >= threshold).astype(float) if threshold > 0 else np.zeros_like(A)
    np.fill_diagonal(binary, 0.0)
    graph = nx.from_numpy_array(binary)
    if graph.number_of_edges() == 0:
        return {
            "role_assortativity": np.nan,
            "flip_rate_assortativity": np.nan,
            "reliable_entropy_assortativity": np.nan,
            "mean_participation_coefficient": np.nan,
            "max_participation_coefficient": np.nan,
            "n_louvain_communities": 0,
        }

    roles = bit_graph["graph_role"].to_list()
    nx.set_node_attributes(
        graph, {idx: roles[idx] for idx in range(len(roles))}, "role"
    )
    metrics = {
        "role_assortativity": safe_attribute_assortativity(graph, "role"),
        "mean_participation_coefficient": float(
            bit_graph["participation_coefficient"].mean()
        ),
        "max_participation_coefficient": float(
            bit_graph["participation_coefficient"].max()
        ),
        "n_louvain_communities": bit_graph.filter(pl.col("louvain_community") >= 0)[
            "louvain_community"
        ].n_unique(),
    }
    for column in ("flip_rate", "reliable_entropy"):
        if column in bit_graph.columns:
            values = bit_graph[column].to_numpy()
            nx.set_node_attributes(
                graph,
                {idx: float(values[idx]) for idx in range(len(values))},
                column,
            )
            metrics[f"{column}_assortativity"] = safe_numeric_assortativity(
                graph,
                column,
            )
        else:
            metrics[f"{column}_assortativity"] = np.nan
    return metrics


def graph_summary_metrics(W: npt.NDArray):
    """Dataset-level graph metrics for a weighted association matrix."""
    A = prepare_adjacency(W, absolute=True)
    threshold = positive_quantile(A, 0.9)
    binary = (A >= threshold).astype(float) if threshold > 0 else np.zeros_like(A)
    np.fill_diagonal(binary, 0.0)
    graph = nx.from_numpy_array(binary)
    if graph.number_of_edges():
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        eigenvalues = np.linalg.eigvalsh(laplacian)
    else:
        eigenvalues = np.zeros(A.shape[0], dtype=float)
    return {
        "n_edges": int(np.count_nonzero(binary) // 2),
        "density": float(nx.density(graph)),
        "n_components": int(nx.number_connected_components(graph)),
        "largest_component_size": max(
            (len(c) for c in nx.connected_components(graph)),
            default=0,
        ),
        "algebraic_connectivity": float(eigenvalues[1])
        if eigenvalues.size > 1
        else 0.0,
        "adjacency_spectral_radius": float(np.max(np.abs(np.linalg.eigvalsh(A)))),
        "adjacency_energy": float(np.sum(np.abs(np.linalg.eigvalsh(A)))),
        "threshold": float(threshold),
    }


def safe_attribute_assortativity(graph, attribute: str):
    try:
        return float(nx.attribute_assortativity_coefficient(graph, attribute))
    except Exception:
        return np.nan


def safe_numeric_assortativity(graph, attribute: str):
    try:
        return float(nx.numeric_assortativity_coefficient(graph, attribute))
    except Exception:
        return np.nan
