import numpy as np
import numpy.typing as npt
import polars as pl


def prepare_adjacency(W: npt.NDArray, *, absolute: bool = True) -> npt.NDArray:
    """Return a finite adjacency matrix with no self-loops."""
    W_asym = np.asarray(W, dtype=float).copy()
    W_asym = np.nan_to_num(W_asym, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(W_asym, 0.0)
    if absolute:
        W_asym = np.abs(W_asym)
    return W_asym


def positive_quantile(matrix: npt.NDArray, quantile: float) -> float:
    values = np.asarray(matrix, dtype=float)
    values = np.abs(values[np.isfinite(values)])
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, quantile))


def edge_frame(
    matrix: npt.NDArray,
    *,
    source: str,
    directed: bool = False,
    min_abs_weight: float = 0.0,
) -> pl.DataFrame:
    """Represent graph matrix entries as a tidy edge table."""
    A = prepare_adjacency(matrix, absolute=False)
    rows = []
    n_bits = A.shape[0]
    if directed:
        iterator = (
            (i, j)
            for i in range(n_bits)
            for j in range(n_bits)
            if i != j and abs(A[i, j]) > min_abs_weight
        )
    else:
        iterator = (
            (i, j)
            for i in range(n_bits)
            for j in range(i + 1, n_bits)
            if abs(A[i, j]) > min_abs_weight
        )

    for target, predictor in iterator:
        rows.append(
            {
                "target": target,
                "predictor": predictor,
                "weight": float(A[target, predictor]),
                "abs_weight": float(abs(A[target, predictor])),
                "source": source,
            }
        )
    return pl.DataFrame(rows)
