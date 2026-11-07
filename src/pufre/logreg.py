import itertools
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import numpy.typing as npt
import optuna
from scipy.sparse import dok_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.covariance import GraphicalLassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from tqdm import tqdm

from pufre import DATA_DIR, logger


@dataclass
class LassoPUFModel:
    """
    Sparse conditional model learned from PUF observations.

    Logistic coefficients are stored in logit units. For Ising mean-field
    Hamiltonian calculations use ``J = W / 2`` and ``h = bias / 2``.
    """

    W: npt.NDArray
    bias: npt.NDArray
    params: list[dict]
    order: int
    selected_challenges: list[npt.NDArray]
    W_orders: dict[int, dict[int, dict[tuple[int, ...], float]]] = field(
        default_factory=dict
    )

    @property
    def J(self):
        return self.W / 2.0

    @property
    def h(self):
        return self.bias / 2.0


def _make_logreg(C: float):
    return LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=C,
        max_iter=1000,
        tol=1e-3,
        warm_start=True,
    )


def _cv_for_labels(y: npt.NDArray, groups: Optional[npt.NDArray], n_splits: int = 3):
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < 2:
        return None

    n_splits = min(n_splits, min_class_count)
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)


def tune_global_C(
    X: npt.NDArray,
    Y: npt.NDArray,
    *,
    groups: Optional[npt.NDArray] = None,
    Cs: npt.NDArray = np.logspace(-3, 2, 12),
    max_bits: int = 50,
):
    """Tune one L1 regularization strength on representative nonconstant bits."""
    candidate_bits = [idx for idx in range(Y.shape[1]) if np.unique(Y[:, idx]).size > 1]
    if not candidate_bits:
        return 1.0, []

    candidate_bits = candidate_bits[:max_bits]
    scores = np.zeros(len(Cs), dtype=float)
    details = []

    for bit_idx in tqdm(candidate_bits, desc="Tuning global C"):
        x_target = np.delete(X, bit_idx, axis=1)
        y_target = Y[:, bit_idx]
        cv = _cv_for_labels(y_target, groups)
        if cv is None:
            continue

        bit_scores = []
        for c_idx, C in enumerate(Cs):
            clf = _make_logreg(float(C))
            if groups is None:
                score = cross_val_score(clf, x_target, y_target, cv=cv).mean()
            else:
                score = cross_val_score(
                    clf, x_target, y_target, cv=cv, groups=groups
                ).mean()
            scores[c_idx] += score
            bit_scores.append(score)

        details.append({"index": bit_idx, "scores": bit_scores})

    best_C = float(Cs[int(np.argmax(scores))])
    return best_C, details


def select_challenges_glasso(
    X: npt.NDArray,
    *,
    max_neighbors: Optional[int] = 12,
    alphas: int = 4,
    cv: int = 3,
    max_iter: int = 200,
):
    """
    Estimate a sparse precision graph and return candidate challenge neighbors.

    The graph is used only for feature selection. The final couplings still come
    from supervised logistic LASSO against the golden response bits.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    std = X.std(axis=0)
    if n_samples <= n_features or np.any(std == 0):
        logger.warning(
            "Skipping GLASSO because data is rank-risky "
            "(samples=%d, features=%d, constant_features=%d). "
            "Falling back to correlations.",
            n_samples,
            n_features,
            int(np.sum(std == 0)),
        )
        return select_challenges_correlation(X, max_neighbors=max_neighbors)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            estimator = GraphicalLassoCV(alphas=alphas, cv=cv, max_iter=max_iter)
            estimator.fit(X)
        precision = estimator.precision_.copy()
        np.fill_diagonal(precision, 0.0)
        if not np.all(np.isfinite(precision)):
            raise ValueError("GLASSO produced a non-finite precision matrix")
    except Exception as exc:
        logger.warning("GLASSO failed (%s). Falling back to correlations.", exc)
        return select_challenges_correlation(X, max_neighbors=max_neighbors)

    selected = []
    correlation_selected, _ = select_challenges_correlation(
        X, max_neighbors=max_neighbors
    )
    for row_idx, row in enumerate(precision):
        neighbors = np.flatnonzero(row)
        if neighbors.size == 0:
            neighbors = correlation_selected[row_idx]
        if max_neighbors is not None and neighbors.size > max_neighbors:
            order = np.argsort(np.abs(row[neighbors]))[-max_neighbors:]
            neighbors = np.sort(neighbors[order])
        selected.append(neighbors.astype(int))

    return selected, precision


def select_challenges_correlation(
    X: npt.NDArray,
    *,
    max_neighbors: Optional[int] = 12,
):
    """Fallback challenge selector based on absolute empirical correlations."""
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X_centered, axis=0)
    safe_norm = np.where(norm == 0, 1.0, norm)
    X_scaled = X_centered / safe_norm
    corr = X_scaled.T @ X_scaled
    corr[:, norm == 0] = 0.0
    corr[norm == 0, :] = 0.0
    np.fill_diagonal(corr, 0.0)

    selected = []
    for row in corr:
        neighbors = np.flatnonzero(row)
        if max_neighbors is not None and neighbors.size > max_neighbors:
            order = np.argsort(np.abs(row[neighbors]))[-max_neighbors:]
            neighbors = np.sort(neighbors[order])
        selected.append(neighbors.astype(int))

    return selected, corr


def build_ordered_features(X: npt.NDArray, neighbors: npt.NDArray, order: int):
    """
    Build feature products from selected neighbors up to ``order``.

    Returns the design matrix and a parallel list of feature terms. A first-order
    feature has one index, a second-order feature has two indices, and so on.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    terms = []
    blocks = []
    for degree in range(1, order + 1):
        degree_terms = list(itertools.combinations(neighbors, degree))
        if not degree_terms:
            continue

        block = np.empty((X.shape[0], len(degree_terms)), dtype=X.dtype)
        for col, term in enumerate(degree_terms):
            block[:, col] = np.prod(X[:, term], axis=1)
        blocks.append(block)
        terms.extend(degree_terms)

    if not blocks:
        return np.empty((X.shape[0], 0), dtype=X.dtype), []

    return np.hstack(blocks), terms


def _log_odds_bias(y: npt.NDArray, *, eps: float = 1e-6):
    p = np.clip(np.mean(y), eps, 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


class PUFStructureLearner:
    """
    End-to-end sparse PUF structure learner.

    It takes flattened challenge observations ``X`` and expanded golden labels
    ``Y``, uses GLASSO to pick candidate neighbors/challenges, then fits logistic
    LASSO for first-, second-, or n-order conditional features.
    """

    def __init__(
        self,
        X: npt.NDArray,
        Y: npt.NDArray,
        *,
        groups: Optional[npt.NDArray] = None,
        max_neighbors: int = 12,
        glasso_alphas: int = 4,
        glasso_cv: int = 3,
        glasso_max_iter: int = 200,
        Cs: npt.NDArray = np.logspace(-3, 2, 12),
        constant_bias: float = 10.0,
    ):
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")

        self.X = X
        self.Y = Y
        self.groups = groups
        self.max_neighbors = max_neighbors
        self.glasso_alphas = glasso_alphas
        self.glasso_cv = glasso_cv
        self.glasso_max_iter = glasso_max_iter
        self.Cs = Cs
        self.constant_bias = constant_bias
        self.selected_challenges: Optional[list[npt.NDArray]] = None
        self.precision_: Optional[npt.NDArray] = None

    def select_challenges(self):
        self.selected_challenges, self.precision_ = select_challenges_glasso(
            self.X,
            max_neighbors=self.max_neighbors,
            alphas=self.glasso_alphas,
            cv=self.glasso_cv,
            max_iter=self.glasso_max_iter,
        )
        return self.selected_challenges

    def fit(self, order: int = 1, C: Optional[float] = None):
        if self.selected_challenges is None:
            self.select_challenges()

        if C is None:
            C, tuning = tune_global_C(
                self.X,
                self.Y,
                groups=self.groups,
                Cs=self.Cs,
            )
        else:
            tuning = []

        n_bits = self.X.shape[1]
        W = np.zeros((n_bits, n_bits), dtype=np.float32)
        bias = np.zeros(n_bits, dtype=np.float32)
        params = []
        W_orders: dict[int, dict[int, dict[tuple[int, ...], float]]] = {
            degree: {} for degree in range(2, order + 1)
        }

        for idx in tqdm(range(n_bits), desc=f"Fitting {order}-order LASSO"):
            y_target = self.Y[:, idx]
            unique_classes = np.unique(y_target)
            neighbors = self.selected_challenges[idx]
            neighbors = neighbors[neighbors != idx]

            if unique_classes.size < 2:
                if unique_classes[0] == 1:
                    bias[idx] = self.constant_bias
                else:
                    bias[idx] = -self.constant_bias
                params.append(
                    {
                        "index": idx,
                        "C": None,
                        "best_score": 1.0,
                        "is_constant": True,
                        "n_features_used": 0,
                    }
                )
                continue

            if neighbors.size == 0:
                bias[idx] = _log_odds_bias(y_target)
                params.append(
                    {
                        "index": idx,
                        "C": None,
                        "best_score": None,
                        "is_constant": False,
                        "n_features_used": 0,
                    }
                )
                continue

            X_target, terms = build_ordered_features(self.X, neighbors, order)
            clf = _make_logreg(float(C))
            clf.fit(X_target, y_target)

            bias[idx] = clf.intercept_[0]
            for term, coef in zip(terms, clf.coef_[0]):
                if np.abs(coef) <= 1e-5:
                    continue
                if len(term) == 1:
                    W[idx, term[0]] = coef
                else:
                    W_orders[len(term)].setdefault(idx, {})[term] = float(coef)

            params.append(
                {
                    "index": idx,
                    "C": C,
                    "best_score": None,
                    "is_constant": False,
                    "n_features_used": X_target.shape[1],
                }
            )

        if tuning:
            params.append({"index": "global", "C": C, "tuning": tuning})

        return LassoPUFModel(
            W=W,
            bias=bias,
            params=params,
            order=order,
            selected_challenges=self.selected_challenges,
            W_orders=W_orders,
        )


def objective_logreg(trial: optuna.Trial, X: npt.NDArray, y: npt.NDArray):
    # We optimize for SPARSITY (L1) and Accuracy against the Golden Key
    C = trial.suggest_float("C", 1e-4, 1e3, log=True)

    clf = LogisticRegression(
        l1_ratio=1,
        solver="liblinear",
        C=C,
        max_iter=5000,
        tol=1e-3,
    )
    clf.fit(X, y)

    # We use a Stratified K-Fold to ensure we don't overfit to specific devices
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=skf)

    return np.mean(scores)


def extract_params_logreg(X: npt.NDArray, Y: npt.NDArray, *, n_trials=10, n_jobs=1):
    n_samples, n_bits = X.shape
    results = []
    pbar = tqdm(np.arange(n_bits))
    for i in pbar:
        y = Y[:, i]
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            # Bit is constant. No need to tune hyperparameters.
            # Return dummy values (they won't be used for training anyway)
            results.append(
                {
                    "index": i,
                    "C": None,  # Dummy
                    "best_score": 1.0,  # Perfect score conceptually
                    "is_constant": True,  # Flag for later
                }
            )
            continue

        # Features: The NOISY other bits
        x = np.delete(X, i, axis=1)
        fn = partial(objective_logreg, X=x, y=y)

        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

        results.append(
            {
                "index": i,
                "C": study.best_params["C"],
                "best_score": study.best_value,
                "is_constant": False,
            }
        )

    return results


# def fit_logreg(X: npt.NDArray, Y: npt.NDArray, *, n_trials=10, n_jobs=1):
#     n_samples, n_bits = X.shape
#     params = extract_params_logreg(X, Y, n_trials=n_trials, n_jobs=n_jobs)

#     W = np.zeros((n_bits, n_bits))
#     bias = np.zeros(n_bits)

#     pbar = tqdm(params, desc="Fitting LogReg")
#     for p in pbar:
#         idx = p["index"]

#         y_target = Y[:, idx]
#         unique_classes = np.unique(y_target)
#         if len(unique_classes) < 2:
#             # The bit is always 0 or always 1.
#             # Logic: If it's always 1, bias = +5. If always 0, bias = -5.
#             # We set a strong bias so the BP network treats it as an "Anchor".
#             constant_class = unique_classes[0]  # 0.0 or 1.0

#             # If class is 1 (value=1), bias should be positive.
#             # If class is 0 (value=0 or -1 in polar), bias should be negative.
#             # Assuming y_target is {0, 1}:
#             bias[idx] = 10.0 if constant_class == 1 else -10.0

#             # Weights stay 0.0
#             continue

#         X_target = np.delete(X, idx, axis=1)

#         clf = LogisticRegression(
#             C=p["C"],
#             l1_ratio=1,
#             solver="liblinear",
#             max_iter=5000,
#             tol=1e-3,
#         ).fit(X_target, y_target)

#         # Hard Pruning Logic
#         bias[idx] = clf.intercept_[0]

#         coeffs = clf.coef_[0]
#         W[idx, :idx] = coeffs[:idx]
#         W[idx, idx + 1 :] = coeffs[idx:]

#     return params, W, bias


def fit_logreg(
    X: npt.NDArray, Y: npt.NDArray, *, cte_bias: float = 10.0, n_trials=10, n_jobs=1
):
    """
    X: The noisy training data (n_samples * n_devices, n_bits)
    Y: The ground truth labels (n_samples * n_devices, n_bits)
          Note: Y is repeated for every sample of the same device.
    cte_bias: The bias to set for constant bits. By default 10.0 for 1 and -10 for 0
    """
    n_samples, n_bits = X.shape

    W = np.zeros((n_bits, n_bits))
    bias = np.zeros(n_bits)
    params = []

    pbar = tqdm(np.arange(n_bits), desc="Fitting LogReg")
    for idx in pbar:
        y_target = Y[:, idx]
        unique_classes = np.unique(y_target)

        if len(unique_classes) < 2:
            # We set a strong bias so the BP network treats it as an "Anchor".
            # If class is 1 (value=1), bias should be positive.
            # If class is 0 (value=0 or -1 in polar), bias should be negative.
            # Assuming y_target is {0, 1}:
            bias[idx] = cte_bias if unique_classes[0] == 1 else -cte_bias

            params.append(
                {
                    "index": idx,
                    "C": None,  # Dummy
                    "best_score": 1.0,
                    "is_constant": True,
                }
            )
            continue

        X_target = np.delete(X, idx, axis=1)

        fn = partial(objective_logreg, X=X_target, y=y_target)
        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
        best_C, best_score = study.best_params["C"], study.best_value

        # Fit Final Model
        clf = LogisticRegression(
            C=best_C, l1_ratio=1.0, solver="liblinear", max_iter=5000, tol=1e-3
        )
        clf.fit(X_target, y_target)

        bias[idx] = clf.intercept_[0]

        coeffs = clf.coef_[0]
        W[idx, :idx] = coeffs[:idx]
        W[idx, idx + 1 :] = coeffs[idx:]

        params.append(
            {
                "index": idx,
                "C": best_C,
                "best_score": best_score,
                "is_constant": False,
            }
        )

    return params, W, bias


###############################################################################
def objective_hierarchical(trial: optuna.Trial, X: npt.NDArray, y: npt.NDArray):
    C = trial.suggest_float("C", 1e-4, 1e3, log=True)

    clf = LogisticRegression(
        C=C,
        l1_ratio=1.0,  # Pure L1
        solver="liblinear",
        max_iter=5000,
        tol=1e-3,
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=skf)
    return np.mean(scores)


def convert_w2nd_to_sparse_matrix(W_2nd, n_bits):
    """
    Converts the hierarchical W_2nd dictionary into a 2D Sparse Matrix.
    Shape: (n_bits, n_pairs) where n_pairs = n_bits * (n_bits - 1) / 2
    """
    # 1. Generate the canonical list of all possible pairs (interaction-only)
    # This ensures our column indices are consistent across all devices.
    all_pairs = list(itertools.combinations(range(n_bits), 2))
    pair_to_idx = {pair: i for i, pair in enumerate(all_pairs)}
    n_pairs = len(all_pairs)

    # 2. Initialize a Dictionary of Keys (DOK) sparse matrix
    # DOK is the most efficient format for incrementally adding values.
    W_poly_sparse = dok_matrix((n_bits, n_pairs), dtype=np.float32)

    # 3. Fill the matrix from the dictionary
    for target_bit, interactions in W_2nd.items():
        for pair, weight in interactions.items():
            # Ensure the pair is sorted (j, k) where j < k to match itertools
            canonical_pair = tuple(sorted(pair))
            if canonical_pair in pair_to_idx:
                col_idx = pair_to_idx[canonical_pair]
                W_poly_sparse[target_bit, col_idx] = weight

    # 4. Convert to CSR (Compressed Sparse Row) for fast arithmetic
    return W_poly_sparse.tocsr(), all_pairs


def build_hierarchical_features(X: npt.NDArray, neighbors: npt.NDArray):
    """
    Builds customized linear and 2nd-degree polynomial features
    using ONLY the pre-selected physical neighbors.
    """
    n_samples = X.shape[0]
    n_neighbors = len(neighbors)

    # 1. Linear Features (The 1st-order shadow)
    X_linear = X[:, neighbors]

    # 2. Pairwise Polynomial Features (2nd-order)
    if n_neighbors >= 2:
        pairs = list(itertools.combinations(neighbors, 2))
        X_poly = np.empty((n_samples, len(pairs)), dtype=X.dtype)
        for idx, (j, k) in enumerate(pairs):
            X_poly[:, idx] = X[:, j] * X[:, k]
    else:
        pairs = []
        X_poly = np.empty((n_samples, 0), dtype=X.dtype)

    # Stack them: [Linear features ... Poly features ...]
    X_custom = np.hstack([X_linear, X_poly])
    return X_custom, pairs


def fit_hierarchical_poly(
    X: npt.NDArray, Y: npt.NDArray, W_pruned: npt.NDArray, *, n_trials=10, n_jobs=1
):
    """
    Fits a 2nd-order Logistic Regression strictly confined to the physical
    neighbors discovered by the 1st-order unpruned/pruned model.

    Args:
        X: Noisy polar data {-1, 1} of shape (n_samples, n_bits)
        Y: Ground truth binary labels {0, 1} of shape (n_samples, n_bits)
        W_pruned: The sparse 1st-order coupling matrix of shape (n_bits, n_bits)
    """
    n_samples, n_bits = X.shape

    # Initialize Outputs
    W_1st = np.zeros((n_bits, n_bits), dtype=np.float32)
    bias = np.zeros(n_bits, dtype=np.float32)
    W_2nd = {}  # Dictionary to hold polynomial weights: W_2nd[target_bit] = {(j, k): weight}
    params = []

    pbar = tqdm(np.arange(n_bits), desc="Fitting Hierarchical Poly")
    for idx in pbar:
        y_target = Y[:, idx]
        unique_classes = np.unique(y_target)

        # Identify neighbors from the 1st-order matrix
        # Exclude self-interactions just in case they slipped through
        neighbors = np.where(W_pruned[idx] != 0)[0]
        neighbors = neighbors[neighbors != idx]

        if len(unique_classes) < 2 or len(neighbors) == 0:
            bias[idx] = (
                10.0 if (len(unique_classes) > 0 and unique_classes[0] == 1) else -10.0
            )
            W_2nd[idx] = {}
            params.append(
                {"index": idx, "C": None, "best_score": 1.0, "is_constant": True}
            )
            continue

        # Build targeted features
        X_custom, poly_pairs = build_hierarchical_features(X, neighbors)

        # Tune Hyperparameters
        fn = partial(objective_hierarchical, X=X_custom, y=y_target)
        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)
        best_C = study.best_params["C"]

        # Fit Final Model
        clf = LogisticRegression(
            C=best_C, l1_ratio=1.0, solver="liblinear", max_iter=5000, tol=1e-3
        )
        clf.fit(X_custom, y_target)

        # Extract Coefficients
        bias[idx] = clf.intercept_[0]
        coeffs = clf.coef_[0]

        # First N coefficients belong to the linear terms (1st order)
        n_linear = len(neighbors)
        linear_coeffs = coeffs[:n_linear]
        W_1st[idx, neighbors] = linear_coeffs

        # Remaining coefficients belong to the polynomial pairs (2nd order)
        poly_coeffs = coeffs[n_linear:]
        poly_dict = {}
        for (j, k), weight in zip(poly_pairs, poly_coeffs):
            if np.abs(weight) > 1e-5:  # Keep it sparse
                poly_dict[(j, k)] = weight
        W_2nd[idx] = poly_dict

        params.append(
            {
                "index": idx,
                "C": best_C,
                "best_score": study.best_value,
                "is_constant": False,
                "n_features_used": X_custom.shape[1],
            }
        )

    return params, W_1st, W_2nd, bias
