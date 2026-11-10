import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from pufre import DATA_DIR, logger
from pufre.analysis.temperature import (
    add_fixed_roles,
    bit_susceptibility_frame,
    graph_drift_frame,
    load_differential_hdf5_artifact,
    role_thresholds,
    role_transition_frame,
    summarize_temperature_roles,
)
from pufre.differential import (
    fit_differential_temperature_study,
    selection_summary_frame,
)
from pufre.error_metrics import (
    artifact_qc_summary,
    conditional_independence_test,
    effective_independent_bits,
    empirical_block_failure_curve,
    expected_independent_unique_rows,
    independent_block_failure_curve,
)
from pufre.graph import (
    ROLE_VALUES,
    DirectedErrorGraphModel,
    bootstrap_edge_stability,
    edge_stability_frame,
    graph_bootstrap_metric_frame,
    graph_role_frame,
    paired_graph_distance_frame,
    stable_adjacency,
)

ARTIFACT = Path(
    os.environ.get(
        "PUFRE_TEMPERATURE_ARTIFACT",
        DATA_DIR / "raw" / "RO_PUF.h5",
    )
)
OUTPUT_DIR = DATA_DIR / "ro_temperature"
MODEL_DIR = OUTPUT_DIR / "models"
STABILITY_DIR = OUTPUT_DIR / "stability"

ROOM_TEMPERATURE = 25
SELECTION_K = 256
BLOCK_SIZE = 256
MAX_CORRECTION_CAPACITY = 26
N_BLOCKS = 10_000
UNIQUE_SIMULATIONS = 200
N_PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 50
BOOTSTRAP_MIN_PROBABILITY = 0.5
BOOTSTRAP_EDGE_THRESHOLD = 1e-8
MAX_ITER = 500
MAX_TUNING_BITS = 8
CV_SPLITS = 2
N_JOBS = int(os.environ.get("PUFRE_N_JOBS", "8"))
ROLE_QUANTILE = 0.75
RANDOM_STATE = 0


def main():
    if N_JOBS <= 0:
        raise SystemExit("PUFRE_N_JOBS must be a positive integer")

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    STABILITY_DIR.mkdir(parents=True, exist_ok=True)

    artifact = load_differential_hdf5_artifact(
        ARTIFACT,
        room_temperature=ROOM_TEMPERATURE,
    )
    result = fit_differential_temperature_study(
        reference=artifact["reference"],
        responses_by_temperature=artifact["responses"],
        value1_by_temperature=artifact["value1"],
        value2_by_temperature=artifact["value2"],
        baseline_temperature=ROOM_TEMPERATURE,
        selection_k=SELECTION_K,
        block_size=BLOCK_SIZE,
        max_correction_capacity=MAX_CORRECTION_CAPACITY,
        n_blocks=N_BLOCKS,
        random_state=RANDOM_STATE,
        graph_C=None,
        max_iter=MAX_ITER,
        max_tuning_bits=MAX_TUNING_BITS,
        cv_splits=CV_SPLITS,
        n_jobs=N_JOBS,
        role_quantile=ROLE_QUANTILE,
    )

    (
        observed_graphs,
        null_graphs,
        observed_samples,
        null_samples,
    ) = fit_stable_graphs(
        result.observations_by_temperature,
        regularization=result.regularization,
        force=args.force,
    )
    baseline_role_thresholds = role_thresholds(
        observed_graphs[ROOM_TEMPERATURE],
        role_quantile=ROLE_QUANTILE,
    )
    stable_roles = stable_role_table(
        result.observations_by_temperature,
        observed_graphs,
        baseline_role_thresholds,
    )
    null_roles = stable_role_table(
        result.observations_by_temperature,
        null_graphs,
        baseline_role_thresholds,
    )
    bit_table = replace_graph_metrics(result.bit_table, stable_roles)
    null_bit_table = replace_graph_metrics(result.bit_table, null_roles)
    role_summary = pl.concat(
        [
            temperature_role_summary(bit_table).with_columns(
                pl.lit("observed").alias("graph_source")
            ),
            temperature_role_summary(null_bit_table).with_columns(
                pl.lit("null").alias("graph_source")
            ),
        ]
    ).sort(
        ["graph_source", "role_definition", "temperature", "graph_role"]
    )

    stable_models = {
        temperature: DirectedErrorGraphModel(
            coefficients=adjacency,
            intercepts=np.zeros(adjacency.shape[0]),
            regularization=result.regularization,
        )
        for temperature, adjacency in observed_graphs.items()
    }
    graph_drift = graph_drift_frame(
        stable_models,
        ROOM_TEMPERATURE,
    )
    role_transitions = pl.concat(
        [
            role_transition_frame(
                bit_table,
                ROOM_TEMPERATURE,
            ).with_columns(pl.lit("observed").alias("graph_source")),
            role_transition_frame(
                null_bit_table,
                ROOM_TEMPERATURE,
            ).with_columns(pl.lit("null").alias("graph_source")),
        ]
    ).sort(["graph_source", "temperature", "metric", "baseline_role"])
    write_graph_distribution_tables(observed_samples, null_samples)
    susceptibility = bit_susceptibility_frame(bit_table, ROOM_TEMPERATURE)
    transfer = result.model_transfer.with_columns(
        (
            pl.col("independent_log_loss")
            - pl.col("nodewise_log_loss")
        ).alias("log_loss_gain")
    )

    write_qc_tables(
        result.observations_by_temperature,
        observed_graphs,
        null_graphs,
        stable_roles,
        result.regularization,
    )
    bit_table.write_csv(OUTPUT_DIR / "ro_temperature_bit_table.csv")
    role_summary.write_csv(OUTPUT_DIR / "ro_temperature_role_summary.csv")
    graph_drift.write_csv(OUTPUT_DIR / "ro_temperature_graph_drift.csv")
    role_transitions.write_csv(
        OUTPUT_DIR / "ro_temperature_role_transitions.csv"
    )
    susceptibility.write_csv(
        OUTPUT_DIR / "ro_temperature_bit_susceptibility.csv"
    )
    transfer.write_csv(OUTPUT_DIR / "ro_temperature_model_transfer.csv")
    selection_summary_frame(
        bit_table,
        result.selections,
        baseline_temperature=ROOM_TEMPERATURE,
    ).write_csv(OUTPUT_DIR / "ro_temperature_selection_summary.csv")
    result.selection_block_failure.write_csv(
        OUTPUT_DIR / "ro_temperature_selection_block_failure.csv"
    )
    write_selection_table(
        result.selections,
        OUTPUT_DIR / "ro_temperature_selections.csv",
    )
    save_models(result.models)
    (OUTPUT_DIR / "regularization.txt").write_text(
        f"{result.regularization}\n",
        encoding="utf-8",
    )
    logger.info("Wrote RO temperature study to %s", OUTPUT_DIR)


def fit_stable_graphs(observations_by_temperature, *, regularization, force):
    observed_graphs = {}
    null_graphs = {}
    observed_samples = {}
    null_samples = {}
    configuration = {
        "cache_version": 2,
        "regularization": regularization,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_min_probability": BOOTSTRAP_MIN_PROBABILITY,
        "bootstrap_edge_threshold": BOOTSTRAP_EDGE_THRESHOLD,
        "random_state": RANDOM_STATE,
    }

    for temperature, observations in sorted(observations_by_temperature.items()):
        prefix = STABILITY_DIR / f"t{temperature}"
        config_path = prefix.with_suffix(".json")
        observed_path = STABILITY_DIR / f"t{temperature}.observed.npy"
        null_path = STABILITY_DIR / f"t{temperature}.null.npy"
        observed_samples_path = (
            STABILITY_DIR / f"t{temperature}.observed_bootstrap.npy"
        )
        null_samples_path = STABILITY_DIR / f"t{temperature}.null_bootstrap.npy"
        cache_valid = (
            not force
            and observed_path.exists()
            and null_path.exists()
            and observed_samples_path.exists()
            and null_samples_path.exists()
            and read_json(config_path) == configuration
        )
        if cache_valid:
            logger.info("Loading stable graphs at T=%s", temperature)
            observed_graphs[temperature] = np.load(observed_path)
            null_graphs[temperature] = np.load(null_path)
            observed_samples[temperature] = np.load(observed_samples_path)
            null_samples[temperature] = np.load(null_samples_path)
            continue

        logger.info("Bootstrapping observed graph at T=%s", temperature)
        observed = bootstrap_edge_stability(
            observations,
            n_bootstrap=BOOTSTRAP_REPLICATES,
            C=regularization,
            edge_threshold=BOOTSTRAP_EDGE_THRESHOLD,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            n_jobs=N_JOBS,
        )
        logger.info("Bootstrapping independence null at T=%s", temperature)
        null = bootstrap_edge_stability(
            observations,
            n_bootstrap=BOOTSTRAP_REPLICATES,
            C=regularization,
            edge_threshold=BOOTSTRAP_EDGE_THRESHOLD,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            independent_null=True,
            n_jobs=N_JOBS,
        )
        observed_graphs[temperature] = stable_adjacency(
            observed,
            min_probability=BOOTSTRAP_MIN_PROBABILITY,
        )
        null_graphs[temperature] = stable_adjacency(
            null,
            min_probability=BOOTSTRAP_MIN_PROBABILITY,
        )
        observed_samples[temperature] = observed.adjacency_samples
        null_samples[temperature] = null.adjacency_samples
        np.save(observed_path, observed_graphs[temperature])
        np.save(null_path, null_graphs[temperature])
        np.save(observed_samples_path, observed.adjacency_samples)
        np.save(null_samples_path, null.adjacency_samples)
        np.save(
            STABILITY_DIR / f"t{temperature}.observed_probability.npy",
            observed.selection_probability,
        )
        np.save(
            STABILITY_DIR / f"t{temperature}.null_probability.npy",
            null.selection_probability,
        )
        edge_stability_frame(
            observed,
            min_probability=BOOTSTRAP_MIN_PROBABILITY,
        ).write_csv(STABILITY_DIR / f"t{temperature}.observed_edges.csv")
        edge_stability_frame(
            null,
            min_probability=BOOTSTRAP_MIN_PROBABILITY,
        ).write_csv(STABILITY_DIR / f"t{temperature}.null_edges.csv")
        config_path.write_text(
            json.dumps(configuration, indent=2),
            encoding="utf-8",
        )
    return observed_graphs, null_graphs, observed_samples, null_samples


def write_graph_distribution_tables(observed_samples, null_samples):
    metric_frames = []
    distance_frames = []
    for temperature in sorted(observed_samples):
        for graph_source, samples in [
            ("observed", observed_samples[temperature]),
            ("null", null_samples[temperature]),
        ]:
            metric_frames.append(
                graph_bootstrap_metric_frame(
                    samples,
                    edge_threshold=BOOTSTRAP_EDGE_THRESHOLD,
                ).with_columns(
                    pl.lit(temperature).alias("temperature"),
                    pl.lit(graph_source).alias("graph_source"),
                )
            )
            reference = (
                observed_samples[ROOM_TEMPERATURE]
                if graph_source == "observed"
                else null_samples[ROOM_TEMPERATURE]
            )
            distance_frames.append(
                paired_graph_distance_frame(
                    reference,
                    samples,
                    edge_threshold=BOOTSTRAP_EDGE_THRESHOLD,
                ).with_columns(
                    pl.lit(temperature).alias("temperature"),
                    pl.lit(graph_source).alias("graph_source"),
                )
            )

    pl.concat(metric_frames).select(
        "temperature",
        "graph_source",
        "bootstrap",
        "directed_edges",
        "total_abs_weight",
        "frobenius_norm",
    ).write_csv(OUTPUT_DIR / "temperature_graph_bootstrap_distribution.csv")
    pl.concat(distance_frames).select(
        "temperature",
        "graph_source",
        "bootstrap",
        "normalized_frobenius_distance",
        "cosine_distance",
        "edge_jaccard_distance",
    ).write_csv(OUTPUT_DIR / "temperature_graph_distance_distribution.csv")


def stable_role_table(
    observations_by_temperature,
    graphs,
    baseline_thresholds,
):
    frames = []
    for temperature, observations in sorted(observations_by_temperature.items()):
        adjacency = graphs[temperature]
        roles = graph_role_frame(
            adjacency,
            bit_ids=observations.bit_ids,
            physical_bit_ids=observations.physical_bit_ids,
            flip_rate=observations.error_binary.mean(axis=(0, 1)),
            low_quantile=1.0 - ROLE_QUANTILE,
            high_quantile=ROLE_QUANTILE,
        )
        frames.append(
            add_fixed_roles(
                roles,
                adjacency,
                baseline_thresholds,
            ).with_columns(pl.lit(temperature).alias("temperature"))
        )
    return pl.concat(frames).sort(["temperature", "bit_id"])


def replace_graph_metrics(bit_table, stable_roles):
    keys = ["temperature", "bit_id", "physical_bit_id"]
    graph_columns = [
        column
        for column in stable_roles.columns
        if column not in keys and column in bit_table.columns
    ]
    return (
        bit_table.drop(graph_columns)
        .join(stable_roles, on=keys)
        .sort(["temperature", "bit_id"])
    )


def temperature_role_summary(bit_table):
    frames = []
    for temperature in bit_table["temperature"].unique().sort().to_list():
        frame = bit_table.filter(pl.col("temperature") == temperature)
        frames.append(summarize_temperature_roles(frame, "graph_role"))
        frames.append(
            summarize_temperature_roles(
                frame,
                "fixed_baseline_graph_role",
            )
        )
    return pl.concat(frames).sort(
        ["role_definition", "temperature", "graph_role"]
    )


def write_qc_tables(
    observations_by_temperature,
    observed_graphs,
    null_graphs,
    stable_roles,
    regularization,
):
    qc_rows = []
    independence_rows = []
    permutation_rows = []
    block_rows = []
    graph_rows = []

    for temperature, observations in sorted(observations_by_temperature.items()):
        error_binary = observations.error_binary
        flat_errors = observations.flat_error_binary()
        qc = artifact_qc_summary(observations)
        expected_unique = expected_independent_unique_rows(
            error_binary,
            n_simulations=UNIQUE_SIMULATIONS,
            random_state=RANDOM_STATE,
        )
        qc.update(
            {
                "temperature": temperature,
                "effective_independent_error_bits": effective_independent_bits(
                    error_binary
                ),
                "expected_independent_unique_rows": expected_unique,
                "unique_error_ratio": qc["unique_error_rows"] / expected_unique,
            }
        )
        qc_rows.append(qc)

        independence = conditional_independence_test(
            error_binary,
            n_permutations=N_PERMUTATIONS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        independence_rows.extend(
            {"temperature": temperature, **row}
            for row in independence.summary_rows()
        )
        permutation_rows.extend(
            {
                "temperature": temperature,
                "permutation": index,
                "correlation_energy": float(values[0]),
                "error_count_variance": float(values[1]),
            }
            for index, values in enumerate(independence.null)
        )

        empirical = empirical_block_failure_curve(
            flat_errors,
            block_size=BLOCK_SIZE,
            max_correction_capacity=MAX_CORRECTION_CAPACITY,
            n_blocks=N_BLOCKS,
            random_state=RANDOM_STATE,
        )
        independent = independent_block_failure_curve(
            error_binary.mean(axis=1),
            block_size=BLOCK_SIZE,
            max_correction_capacity=MAX_CORRECTION_CAPACITY,
            n_blocks=N_BLOCKS,
            random_state=RANDOM_STATE,
        )
        block_rows.extend(
            {
                "temperature": temperature,
                "correction_capacity": capacity,
                "empirical_block_failure": float(empirical[capacity]),
                "independent_block_failure": float(independent[capacity]),
                "failure_gap": float(empirical[capacity] - independent[capacity]),
            }
            for capacity in range(MAX_CORRECTION_CAPACITY + 1)
        )

        observed_edges = int(np.count_nonzero(observed_graphs[temperature]))
        null_edges = int(np.count_nonzero(null_graphs[temperature]))
        roles = stable_roles.filter(pl.col("temperature") == temperature)
        graph_rows.append(
            {
                "temperature": temperature,
                "regularization": regularization,
                "observed_stable_directed_edges": observed_edges,
                "null_stable_directed_edges": null_edges,
                "stable_edge_excess": observed_edges - null_edges,
                "stable_edge_ratio": (
                    observed_edges / null_edges
                    if null_edges
                    else float("inf") if observed_edges else 1.0
                ),
                "median_relative_out_strength": float(
                    roles["relative_out_strength"].median()
                ),
                "median_relative_in_strength": float(
                    roles["relative_in_strength"].median()
                ),
                **{
                    f"n_{role}_bits": roles.filter(
                        pl.col("graph_role") == role
                    ).height
                    for role in ROLE_VALUES
                },
            }
        )

    pl.DataFrame(qc_rows).write_csv(OUTPUT_DIR / "temperature_qc.csv")
    pl.DataFrame(independence_rows).write_csv(
        OUTPUT_DIR / "temperature_independence.csv"
    )
    pl.DataFrame(permutation_rows).write_csv(
        OUTPUT_DIR / "temperature_permutation_null.csv"
    )
    pl.DataFrame(block_rows).write_csv(
        OUTPUT_DIR / "temperature_block_failure.csv"
    )
    pl.DataFrame(graph_rows).write_csv(
        OUTPUT_DIR / "temperature_graph_summary.csv"
    )


def write_selection_table(selections, path):
    rows = [
        {"selection": selection, "bit_id": int(bit_id)}
        for selection, bit_ids in selections.items()
        for bit_id in bit_ids
    ]
    pl.DataFrame(rows).write_csv(path)


def save_models(models):
    for temperature, model in models.items():
        np.savez_compressed(
            MODEL_DIR / f"temperature_{temperature}.model.npz",
            coefficients=model.coefficients,
            intercepts=model.intercepts,
            regularization=model.regularization,
        )


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


if __name__ == "__main__":
    main()
