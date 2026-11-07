import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pufre.data import PUFObservations
from pufre.graph.model import DirectedErrorGraphModel, NodewiseLogisticGraph
from pufre.graph.roles import compute_network_centrality
from pufre.metrics import bit_reliability, reliable_entropy_score


@dataclass
class ErrorGraphStudyResult:
    name: str
    observations: PUFObservations
    model: DirectedErrorGraphModel
    summary: dict[str, npt.NDArray]


def fit_error_graph_study(
    observations: PUFObservations,
    *,
    name: str,
    C: float | None = None,
    max_tuning_bits: int = 32,
    cv_splits: int = 3,
    random_state: int = 0,
    n_jobs: int = 1,
):
    estimator = NodewiseLogisticGraph(
        C=C,
        max_tuning_bits=max_tuning_bits,
        cv_splits=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    estimator.fit(observations.flat_errors(), groups=observations.device_ids)
    model = estimator.model_
    return ErrorGraphStudyResult(
        name,
        observations,
        model,
        summarize_error_graph(observations, model),
    )


def summarize_error_graph(
    observations: PUFObservations,
    model: DirectedErrorGraphModel,
):
    error_binary = observations.error_binary
    response_binary = observations.response_binary
    reference_binary = observations.reference_binary
    hub_score, authority_score = compute_network_centrality(model.adjacency)
    return {
        "bit_id": observations.bit_ids,
        "physical_bit_id": observations.physical_bit_ids,
        "reliable_entropy": reliable_entropy_score(response_binary, reference_binary),
        "reliability": bit_reliability(observations.responses, observations.reference),
        "flip_rate": np.mean(error_binary, axis=(0, 1)),
        "temporal_flip_entropy": _binary_entropy(np.mean(error_binary, axis=1)).mean(
            axis=0
        ),
        "logistic_intercept": model.intercepts,
        "directed_edge_strength": np.sum(np.abs(model.adjacency), axis=0),
        "hub_score": hub_score,
        "authority_score": authority_score,
    }


def save_error_graph_study(result: ErrorGraphStudyResult, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / result.name
    np.savez_compressed(
        prefix.with_suffix(".model.npz"),
        coefficients=result.model.coefficients,
        intercepts=result.model.intercepts,
        regularization=result.model.regularization,
    )
    np.savez_compressed(
        prefix.with_suffix(".summary.npz"),
        **result.summary,  # ty: ignore[invalid-argument-type]
    )
    prefix.with_suffix(".metadata.json").write_text(
        json.dumps(_jsonable(result.observations.metadata), indent=2),
        encoding="utf-8",
    )


def load_error_graph_model(path: Path):
    with np.load(path, allow_pickle=True) as data:
        return DirectedErrorGraphModel(
            coefficients=data["coefficients"],
            intercepts=data["intercepts"],
            regularization=float(data["regularization"]),
        )


def _binary_entropy(probability: npt.ArrayLike, eps: float = 1e-12):
    probability = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return -(
        probability * np.log2(probability)
        + (1.0 - probability) * np.log2(1.0 - probability)
    )


def _jsonable(metadata: dict):
    output = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            output[key] = value.tolist()
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output
