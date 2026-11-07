from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm


@dataclass
class DirectedErrorGraphModel:
    """Directed same-readout predictive associations between PUF error bits.

    ``coefficients[target, predictor]`` is the logistic coefficient used to
    predict the target error spin from the predictor error spin. The model is
    descriptive and predictive; edge direction does not establish causality.
    """

    coefficients: npt.NDArray
    intercepts: npt.NDArray
    regularization: float
    fit_details: list[dict] = field(default_factory=list)

    @property
    def adjacency(self):
        adjacency = np.asarray(self.coefficients, dtype=float).copy()
        np.fill_diagonal(adjacency, 0.0)
        return adjacency

    @property
    def undirected_adjacency(self):
        """Symmetric projection for algorithms that require undirected input."""
        adjacency = self.adjacency
        return (adjacency + adjacency.T) / 2.0

    def conditional_flip_probability(self, errors: npt.NDArray):
        """Return P(error_i=-1 | errors_-i) for each row and target bit."""
        errors = np.asarray(errors, dtype=float)
        logits_agree = errors @ self.coefficients.T + self.intercepts
        return 1.0 - expit(logits_agree)


class NodewiseLogisticGraph:
    """Fit an L1-logistic directed predictive graph to error-spin rows."""

    def __init__(
        self,
        *,
        C: float | None = None,
        Cs: npt.ArrayLike | None = None,
        max_tuning_bits: int = 32,
        cv_splits: int = 3,
        random_state: int = 0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        n_jobs: int = 1,
    ):
        if n_jobs == 0:
            raise ValueError("n_jobs must be nonzero")
        self.C = C
        self.Cs = np.logspace(-3, 1, 9) if Cs is None else np.asarray(Cs, dtype=float)
        self.max_tuning_bits = max_tuning_bits
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X: npt.NDArray, groups: npt.NDArray | None = None):
        X = _validate_error_spins(X)
        groups = None if groups is None else np.asarray(groups)
        C = float(self.C) if self.C is not None else self._tune_C(X, groups)

        n_bits = X.shape[1]
        coefficients = np.zeros((n_bits, n_bits), dtype=np.float32)
        intercepts = np.zeros(n_bits, dtype=np.float32)
        details = []

        results = Parallel(n_jobs=self.n_jobs, return_as="generator")(
            delayed(self._fit_target)(X, bit_idx, C) for bit_idx in range(n_bits)
        )
        for bit_idx, columns, coefficient, intercept, detail in tqdm(
            results,
            total=n_bits,
            desc="Fitting directed error graph",
        ):
            intercepts[bit_idx] = intercept
            if columns is not None:
                coefficients[bit_idx, columns] = coefficient
            details.append(detail)

        details.sort(key=lambda detail: detail["bit_id"])

        self.model_ = DirectedErrorGraphModel(
            coefficients,
            intercepts,
            C,
            details,
        )
        return self

    def _fit_target(self, X, bit_idx: int, C: float):
        y = (X[:, bit_idx] > 0).astype(np.uint8)
        if np.unique(y).size < 2:
            intercept = logit(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
            return (
                bit_idx,
                None,
                None,
                float(intercept),
                {"bit_id": bit_idx, "is_constant": True, "n_nonzero": 0},
            )

        columns = np.delete(np.arange(X.shape[1]), bit_idx)
        classifier = self._make_classifier(C)
        classifier.fit(X[:, columns], y)
        coefficient = classifier.coef_[0]
        return (
            bit_idx,
            columns,
            coefficient,
            float(classifier.intercept_[0]),
            {
                "bit_id": bit_idx,
                "is_constant": False,
                "n_nonzero": int(np.count_nonzero(np.abs(coefficient) > 1e-8)),
            },
        )

    def _make_classifier(self, C: float):
        return LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def _tune_C(self, X: npt.NDArray, groups: npt.NDArray | None):
        rng = np.random.default_rng(self.random_state)
        variable_bits = np.flatnonzero(
            [np.unique(X[:, idx]).size > 1 for idx in range(X.shape[1])]
        )
        if variable_bits.size == 0:
            return 1.0
        tune_bits = rng.choice(
            variable_bits,
            size=min(self.max_tuning_bits, variable_bits.size),
            replace=False,
        )

        scores = np.zeros(self.Cs.size, dtype=float)
        counts = np.zeros(self.Cs.size, dtype=int)
        results = Parallel(n_jobs=self.n_jobs, return_as="generator")(
            delayed(self._score_tuning_bit)(X, groups, bit_idx)
            for bit_idx in tune_bits
        )
        for bit_scores, bit_counts in tqdm(
            results,
            total=tune_bits.size,
            desc="Tuning directed graph C",
        ):
            scores += bit_scores
            counts += bit_counts

        if not np.any(counts):
            return 1.0
        mean_scores = np.divide(
            scores,
            counts,
            out=np.full_like(scores, np.inf),
            where=counts > 0,
        )
        return float(self.Cs[int(np.argmin(mean_scores))])

    def _score_tuning_bit(self, X, groups, bit_idx: int):
        scores = np.zeros(self.Cs.size, dtype=float)
        counts = np.zeros(self.Cs.size, dtype=int)
        y = (X[:, bit_idx] > 0).astype(np.uint8)
        predictors = np.delete(X, bit_idx, axis=1)
        splitter = _splitter(y, groups, self.cv_splits)
        if splitter is None:
            return scores, counts

        for train_idx, test_idx in splitter.split(predictors, y, groups):
            if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
                continue
            for c_idx, candidate in enumerate(self.Cs):
                classifier = self._make_classifier(float(candidate))
                classifier.fit(predictors[train_idx], y[train_idx])
                probability = classifier.predict_proba(predictors[test_idx])
                scores[c_idx] += log_loss(y[test_idx], probability, labels=[0, 1])
                counts[c_idx] += 1
        return scores, counts


def _validate_error_spins(X: npt.NDArray):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("error spins must be a 2D matrix")
    if not np.all(np.isin(X, [-1, 1])):
        raise ValueError("error spins must be encoded as {-1, +1}")
    return X.astype(np.float32, copy=False)


def _splitter(y: npt.NDArray, groups: npt.NDArray | None, n_splits: int):
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(counts.min())
    if min_class_count < 2:
        return None
    if groups is not None:
        unique_groups = np.unique(groups)
        if unique_groups.size >= 2:
            return GroupKFold(n_splits=min(n_splits, unique_groups.size))
    return KFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=0)
