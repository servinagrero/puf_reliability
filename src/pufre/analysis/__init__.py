import numpy as np
import polars as pl

from pufre.hamiltonian import Hamiltonian
from pufre.study import PUFStudyResult, summarize_hamiltonian
from pufre.utils import PUFData


def summary_to_frame(summary: dict):
    """Convert same-length bitwise summary arrays to a Polars DataFrame."""
    bitwise = {}
    for key, value in summary.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.shape[0] == np.asarray(summary["bit_id"]).shape[0]:
            bitwise[key] = arr
    bit_count = np.asarray(summary["bit_id"]).shape[0]
    local_bit_id = np.arange(bit_count)
    if not np.array_equal(np.asarray(bitwise["bit_id"]), local_bit_id):
        bitwise.setdefault("physical_bit_id", bitwise["bit_id"])
        bitwise["bit_id"] = local_bit_id
    return pl.DataFrame(bitwise)


def study_summary_frame(result: PUFStudyResult):
    """Return the bitwise summary table for a fitted study result."""
    frame = summary_to_frame(result.summary)
    return frame.with_columns(pl.lit(result.name).alias("study"))


def hamiltonian_summary_frame(data: PUFData, hamiltonian: Hamiltonian):
    """Compute and return the standard Hamiltonian bitwise summary table."""
    return summary_to_frame(summarize_hamiltonian(data, hamiltonian))


def spectral_metrics(J: np.ndarray):
    """Compute basic spectral metrics for a symmetric coupling matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    abs_eigenvalues = np.abs(eigenvalues)
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "spectral_radius": np.max(abs_eigenvalues),
        "spectral_gap": _spectral_gap(abs_eigenvalues),
    }


def _spectral_gap(abs_eigenvalues: np.ndarray):
    if abs_eigenvalues.size < 2:
        return 0.0
    ordered = np.sort(abs_eigenvalues)
    return ordered[-1] - ordered[-2]
