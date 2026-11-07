import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from pufre.graph import compute_hits_centrality, compute_network_centrality
from pufre.hamiltonian import Hamiltonian, compute_order_parameter
from pufre.metrics import (
    bit_aliasing_entropy,
    bit_reliability,
    reliable_entropy_score,
    temporal_instability_entropy,
)
from pufre.pipeline import LogisticLassoPUF, ReliableEntropySelector
from pufre.utils import PUFData, prune_matrix


@dataclass
class PUFStudyResult:
    name: str
    data: PUFData
    model: object
    hamiltonian: Hamiltonian
    selected_bits: npt.NDArray
    summary: dict[str, npt.NDArray]


def fit_lasso_study(
    data: PUFData,
    *,
    name: str,
    order: int = 1,
    n_best_bits: Optional[int] = None,
    C: Optional[float] = None,
    max_neighbors: int = 12,
    glasso_alphas: int = 4,
    glasso_cv: int = 3,
    glasso_max_iter: int = 200,
):
    """Fit a complete sparse-logistic Hamiltonian study."""
    if n_best_bits is not None:
        selector = ReliableEntropySelector(n_bits=n_best_bits)
        data = selector.fit_transform(data)

    estimator = LogisticLassoPUF(
        order=order,
        C=C,
        max_neighbors=max_neighbors,
        glasso_alphas=glasso_alphas,
        glasso_cv=glasso_cv,
        glasso_max_iter=glasso_max_iter,
    )
    estimator.fit(data)
    model = estimator.model_

    W_pruned = prune_matrix(model.W, max_neighbors)
    hamiltonian = Hamiltonian(data.crps, W_pruned, model.bias)
    summary = summarize_hamiltonian(data, hamiltonian)

    return PUFStudyResult(
        name=name,
        data=data,
        model=model,
        hamiltonian=hamiltonian,
        selected_bits=np.arange(data.crps.shape[2]),
        summary=summary,
    )


def summarize_hamiltonian(data: PUFData, hamiltonian: Hamiltonian):
    """Collect metrics used across PUF studies."""
    gt_bin = data.gt_bin
    crps_bin = data.X_bin_flat.reshape(data.crps.shape)

    upc = bit_aliasing_entropy(gt_bin)
    stability = 1.0 - temporal_instability_entropy(crps_bin)

    hub_score, authority_score = compute_network_centrality(hamiltonian.W)
    hits_hub, hits_authority = compute_hits_centrality(hamiltonian.W)

    bit_id = np.arange(data.crps.shape[2])
    physical_bit_id = data.metadata.get("selected_bit_indices", bit_id)

    return {
        "bit_id": bit_id,
        "physical_bit_id": physical_bit_id,
        "upc": upc,
        "stability": stability,
        "reliable_entropy": reliable_entropy_score(crps_bin, gt_bin),
        "reliability": bit_reliability(data.crps, data.gt_polar),
        "device_order_parameter": compute_order_parameter(hamiltonian.M),
        "h_magnitude": hamiltonian.h_magnitude,
        "interaction_strength": hamiltonian.interaction_strength,
        "susceptibility": hamiltonian.susceptibility,
        "independence_ratio": np.mean(hamiltonian.independence_ratio, axis=0),
        "hub_score": hub_score,
        "authority_score": authority_score,
        "hits_hub": hits_hub,
        "hits_authority": hits_authority,
    }


def save_study_result(result: PUFStudyResult, directory: Path):
    """Persist model arrays and bitwise summary arrays for a study scenario."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / result.name

    PUFData.save_model(prefix.with_suffix(".model.npz"), result.model)
    np.savez_compressed(prefix.with_suffix(".summary.npz"), **result.summary)
    np.save(prefix.with_suffix(".selected_bits.npy"), result.selected_bits)
    prefix.with_suffix(".metadata.json").write_text(
        json.dumps(_jsonable_metadata(result.data.metadata), indent=2),
        encoding="utf-8",
    )


def _jsonable_metadata(metadata: dict):
    jsonable = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            jsonable[key] = value.tolist()
        elif isinstance(value, np.generic):
            jsonable[key] = value.item()
        else:
            jsonable[key] = value
    return jsonable
