from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from pufre.data import PUFObservations
from pufre.graph.model import NodewiseLogisticGraph


@dataclass
class ErrorGraphValidationResult:
    family: str
    model: str
    log_loss: float
    brier: float
    mean_observed_flip_rate: float
    mean_predicted_flip_rate: float
    n_train: int
    n_test: int
    C: float


@dataclass
class ErrorGraphValidationTrial:
    family: str
    split: int
    independent_log_loss: float
    error_graph_log_loss: float
    log_loss_gain: float
    independent_brier: float
    error_graph_brier: float
    brier_gain: float
    n_train: int
    n_test: int
    n_train_devices: int
    n_test_devices: int
    C: float


@dataclass
class ErrorGraphDeviceValidationResult:
    family: str
    split: int
    device_id: int
    n_test: int
    independent_log_loss: float
    error_graph_log_loss: float
    log_loss_gain: float
    independent_brier: float
    error_graph_brier: float
    brier_gain: float


def validate_error_graph(
    observations: PUFObservations,
    *,
    family: str,
    C: float | None = 0.1,
    test_size: float = 0.25,
    random_state: int = 0,
    max_iter: int = 500,
    n_jobs: int = 1,
    max_tuning_bits: int = 8,
    cv_splits: int = 2,
):
    """
    Compare independent per-bit error rates against a held-out directed graph.

    Splits by device when at least two devices are available.
    """
    independent, graph, _, _ = validate_error_graph_split(
        observations,
        family=family,
        split=0,
        C=C,
        test_size=test_size,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=n_jobs,
        max_tuning_bits=max_tuning_bits,
        cv_splits=cv_splits,
    )
    return independent, graph


def validate_error_graph_split(
    observations: PUFObservations,
    *,
    family: str,
    split: int,
    C: float | None = 0.1,
    test_size: float = 0.25,
    random_state: int = 0,
    max_iter: int = 500,
    n_jobs: int = 1,
    max_tuning_bits: int = 8,
    cv_splits: int = 2,
):
    """Fit and score one device-held-out predictive-validation split."""
    X = observations.flat_errors()
    groups = observations.device_ids
    train_idx, test_idx = train_test_indices(
        X,
        groups=groups,
        test_size=test_size,
        random_state=random_state,
    )
    train = X[train_idx]
    test = X[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    y_train_flip = (train < 0).astype(np.uint8)
    y_test_flip = (test < 0).astype(np.uint8)
    flip_rate = np.clip(y_train_flip.mean(axis=0), 1e-6, 1.0 - 1e-6)
    independent_probability = np.broadcast_to(flip_rate, y_test_flip.shape)

    estimator = NodewiseLogisticGraph(
        C=C,
        random_state=random_state,
        max_iter=max_iter,
        n_jobs=n_jobs,
        max_tuning_bits=max_tuning_bits,
        cv_splits=cv_splits,
    )
    estimator.fit(train, groups=train_groups)
    graph_probability = estimator.model_.conditional_flip_probability(test)

    independent = score_predictions(
        y_test_flip,
        independent_probability,
        family=family,
        model="independent",
        n_train=train.shape[0],
        C=np.nan,
    )
    graph = score_predictions(
        y_test_flip,
        graph_probability,
        family=family,
        model="error_graph",
        n_train=train.shape[0],
        C=estimator.model_.regularization,
    )

    trial = ErrorGraphValidationTrial(
        family=family,
        split=split,
        independent_log_loss=independent.log_loss,
        error_graph_log_loss=graph.log_loss,
        log_loss_gain=independent.log_loss - graph.log_loss,
        independent_brier=independent.brier,
        error_graph_brier=graph.brier,
        brier_gain=independent.brier - graph.brier,
        n_train=train.shape[0],
        n_test=test.shape[0],
        n_train_devices=np.unique(train_groups).size,
        n_test_devices=np.unique(test_groups).size,
        C=estimator.model_.regularization,
    )

    device_results = []
    for device_id in np.unique(test_groups):
        selected = test_groups == device_id
        device_independent = score_predictions(
            y_test_flip[selected],
            independent_probability[selected],
            family=family,
            model="independent",
            n_train=train.shape[0],
            C=np.nan,
        )
        device_graph = score_predictions(
            y_test_flip[selected],
            graph_probability[selected],
            family=family,
            model="error_graph",
            n_train=train.shape[0],
            C=estimator.model_.regularization,
        )
        device_results.append(
            ErrorGraphDeviceValidationResult(
                family=family,
                split=split,
                device_id=int(device_id),
                n_test=int(np.count_nonzero(selected)),
                independent_log_loss=device_independent.log_loss,
                error_graph_log_loss=device_graph.log_loss,
                log_loss_gain=(
                    device_independent.log_loss - device_graph.log_loss
                ),
                independent_brier=device_independent.brier,
                error_graph_brier=device_graph.brier,
                brier_gain=device_independent.brier - device_graph.brier,
            )
        )

    return independent, graph, trial, device_results


def score_predictions(
    y_true, proba, *, family: str, model: str, n_train: int, C: float
):
    y = y_true.reshape(-1)
    p = np.clip(proba.reshape(-1), 1e-6, 1.0 - 1e-6)
    return ErrorGraphValidationResult(
        family=family,
        model=model,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        mean_observed_flip_rate=float(np.mean(y)),
        mean_predicted_flip_rate=float(np.mean(p)),
        n_train=int(n_train),
        n_test=int(y_true.shape[0]),
        C=float(C),
    )


def train_test_indices(X, *, groups, test_size: float, random_state: int):
    if np.unique(groups).size >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        return next(splitter.split(X, groups=groups))

    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(X))
