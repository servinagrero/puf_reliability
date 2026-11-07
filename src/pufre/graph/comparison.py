import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import spearmanr

from pufre.graph.base import prepare_adjacency


def graph_bootstrap_metric_frame(adjacencies, *, edge_threshold=1e-8):
    samples = _validate_adjacency_samples(adjacencies)
    return pl.DataFrame(
        {
            "bootstrap": np.arange(samples.shape[0]),
            "directed_edges": np.count_nonzero(
                np.abs(samples) > edge_threshold,
                axis=(1, 2),
            ),
            "total_abs_weight": np.abs(samples).sum(axis=(1, 2)),
            "frobenius_norm": np.linalg.norm(samples, axis=(1, 2)),
        }
    )


def paired_graph_distance_frame(
    reference_adjacencies,
    target_adjacencies,
    *,
    edge_threshold=1e-8,
):
    reference = _validate_adjacency_samples(reference_adjacencies)
    target = _validate_adjacency_samples(target_adjacencies)
    if reference.shape != target.shape:
        raise ValueError("reference and target adjacency samples must match")

    difference_norm = np.linalg.norm(target - reference, axis=(1, 2))
    reference_norm = np.linalg.norm(reference, axis=(1, 2))
    target_norm = np.linalg.norm(target, axis=(1, 2))
    norm_sum = reference_norm + target_norm
    normalized_frobenius = np.divide(
        difference_norm,
        norm_sum,
        out=np.zeros_like(difference_norm),
        where=norm_sum > 0,
    )

    dot = np.sum(reference * target, axis=(1, 2))
    norm_product = reference_norm * target_norm
    cosine_similarity = np.divide(
        dot,
        norm_product,
        out=np.full_like(dot, np.nan),
        where=norm_product > 0,
    )

    reference_edges = np.abs(reference) > edge_threshold
    target_edges = np.abs(target) > edge_threshold
    intersection = np.logical_and(reference_edges, target_edges).sum(axis=(1, 2))
    union = np.logical_or(reference_edges, target_edges).sum(axis=(1, 2))
    edge_jaccard = np.divide(
        intersection,
        union,
        out=np.ones(intersection.shape, dtype=float),
        where=union > 0,
    )

    return pl.DataFrame(
        {
            "bootstrap": np.arange(reference.shape[0]),
            "normalized_frobenius_distance": normalized_frobenius,
            "cosine_distance": 1.0 - cosine_similarity,
            "edge_jaccard_distance": 1.0 - edge_jaccard,
        }
    )


def _validate_adjacency_samples(adjacencies):
    samples = np.asarray(adjacencies, dtype=float)
    if (
        samples.ndim != 3
        or samples.shape[1] != samples.shape[2]
        or samples.shape[0] == 0
    ):
        raise ValueError("adjacencies must have shape (samples, bits, bits)")
    return samples


def compare_mi_and_graph(mi: npt.NDArray, adjacency: npt.NDArray):
    """Return edge-level MI-vs-graph comparison and global correlations."""
    mi = prepare_adjacency(mi, absolute=True)
    graph_abs = prepare_adjacency(adjacency, absolute=True)
    upper = np.triu_indices_from(mi, k=1)
    mi_values = mi[upper]
    graph_values = graph_abs[upper]

    frame = pl.DataFrame(
        {
            "bit_i": upper[0],
            "bit_j": upper[1],
            "mutual_information": mi_values,
            "graph_abs_weight": graph_values,
            "mi_rank": rankdata(mi_values),
            "graph_rank": rankdata(graph_values),
        }
    )
    mask = np.isfinite(mi_values) & np.isfinite(graph_values)
    if np.sum(mask) < 2:
        return frame, {"pearson": np.nan, "spearman": np.nan}
    if np.std(mi_values[mask]) == 0.0 or np.std(graph_values[mask]) == 0.0:
        return frame, {"pearson": np.nan, "spearman": np.nan}

    pearson = float(np.corrcoef(mi_values[mask], graph_values[mask])[0, 1])
    spearman = float(spearmanr(mi_values[mask], graph_values[mask]).statistic)
    return frame, {"pearson": pearson, "spearman": spearman}


def rankdata(values: npt.NDArray) -> npt.NDArray:
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size)
    return ranks
