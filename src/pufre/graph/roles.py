from enum import StrEnum
from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
import polars as pl

from pufre import logger
from pufre.graph.base import positive_quantile, prepare_adjacency


class Role(StrEnum):
    DISCONNECTED = "disconnected"
    PERIPHERAL = "peripheral"
    SOURCE = "source"
    SINK = "sink"
    RELAY = "relay"
    INTERMEDIATE = "intermediate"


ROLE_VALUES = tuple(role.value for role in Role)


def compute_network_centrality(W_asym: npt.NDArray):
    """
    Return source-like and sink-like strengths for directed predictive weights.

    Rows are targets and columns are predictors, so column sums are predictive
    hub/source scores and row sums are authority/sink scores.
    """
    A = prepare_adjacency(W_asym, absolute=True)
    return A.sum(axis=0), A.sum(axis=1)


def compute_hits_centrality(W_asym: npt.NDArray):
    A = prepare_adjacency(W_asym, absolute=True)
    if not np.any(A):
        return compute_network_centrality(A)

    graph = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    try:
        hubs, authorities = nx.hits(graph, max_iter=100, normalized=True)
    except Exception as exc:
        logger.warning("HITS failed (%s). Using in/out strength instead.", exc)
        return compute_network_centrality(A)

    hubs_vec = np.array([hubs[i] for i in range(len(hubs))])
    auth_vec = np.array([authorities[i] for i in range(len(authorities))])
    return hubs_vec, auth_vec


def graph_role_frame(
    W_asym: npt.NDArray,
    *,
    bit_ids: Optional[npt.ArrayLike] = None,
    physical_bit_ids: Optional[npt.ArrayLike] = None,
    flip_rate: Optional[npt.ArrayLike] = None,
    reliable_entropy: Optional[npt.ArrayLike] = None,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
):
    """Classify bits from the IQR of positive incoming/outgoing strengths."""
    A = prepare_adjacency(W_asym, absolute=True)
    n_bits = A.shape[0]
    bit_ids = np.arange(n_bits) if bit_ids is None else np.asarray(bit_ids)
    physical_bit_ids = (
        bit_ids if physical_bit_ids is None else np.asarray(physical_bit_ids)
    )
    out_strength = A.sum(axis=0)
    in_strength = A.sum(axis=1)
    out_degree = np.count_nonzero(A, axis=0)
    in_degree = np.count_nonzero(A, axis=1)
    total_strength = out_strength + in_strength
    net_source_strength = out_strength - in_strength

    thresholds = graph_role_thresholds(
        W_asym,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    roles = np.array(
        [
            classify_role(out_value, in_value, thresholds)
            for out_value, in_value in zip(out_strength, in_strength)
        ],
        dtype=str,
    )

    columns = {
        "bit_id": bit_ids,
        "physical_bit_id": physical_bit_ids,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "out_strength": out_strength,
        "in_strength": in_strength,
        "relative_out_strength": relative_strength(out_strength),
        "relative_in_strength": relative_strength(in_strength),
        "total_strength": total_strength,
        "net_source_strength": net_source_strength,
        "source_ratio": out_strength / (in_strength + 1e-12),
        "sink_ratio": in_strength / (out_strength + 1e-12),
        "isolation_score": 1.0 / (total_strength + 1e-12),
        "graph_role": roles,
    }
    if flip_rate is not None:
        columns["flip_rate"] = np.asarray(flip_rate)
    if reliable_entropy is not None:
        columns["reliable_entropy"] = np.asarray(reliable_entropy)
    return pl.DataFrame(columns)


def graph_role_thresholds(
    W_asym: npt.NDArray,
    *,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
):
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("role quantiles must satisfy 0 <= low < high <= 1")
    A = prepare_adjacency(W_asym, absolute=True)
    out_strength = A.sum(axis=0)
    in_strength = A.sum(axis=1)
    return {
        "out_low": positive_quantile(out_strength, low_quantile),
        "out_high": positive_quantile(out_strength, high_quantile),
        "in_low": positive_quantile(in_strength, low_quantile),
        "in_high": positive_quantile(in_strength, high_quantile),
    }


def classify_role(out_strength: float, in_strength: float, thresholds: dict) -> Role:
    if out_strength == 0.0 and in_strength == 0.0:
        return Role.DISCONNECTED

    high_out = out_strength >= thresholds["out_high"] and out_strength > 0.0
    high_in = in_strength >= thresholds["in_high"] and in_strength > 0.0
    if high_out and high_in:
        return Role.RELAY
    if high_out:
        return Role.SOURCE
    if high_in:
        return Role.SINK

    low_out = out_strength <= thresholds["out_low"]
    low_in = in_strength <= thresholds["in_low"]
    if low_out and low_in:
        return Role.PERIPHERAL
    return Role.INTERMEDIATE


def relative_strength(strength):
    mean = np.mean(strength)
    if mean == 0.0:
        return np.zeros_like(strength)
    return strength / mean
