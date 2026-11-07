import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from pufre import DATA_DIR
from pufre.data import PUFObservations
from pufre.datasets import load_hdf5_artifact
from pufre.error_metrics import (
    artifact_qc_summary,
    effective_independent_bits,
    empirical_block_failure_curve,
    empirical_block_failure_probability,
    expected_independent_unique_rows,
    independent_block_failure_curve,
    independent_block_failure_probability,
    pairwise_mutual_information,
    validate_error_graph_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "error_qc")
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--correction-capacity", type=int, default=8)
    parser.add_argument("--max-correction-capacity", type=int, default=26)
    parser.add_argument("--n-blocks", type=int, default=1000)
    parser.add_argument("--unique-simulations", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--validation-repeats", type=int, default=1)
    parser.add_argument(
        "--repeated-validation-families",
        nargs="+",
        default=None,
        help=(
            "Families that receive all repeated validation splits. Other "
            "families retain split zero for compatibility."
        ),
    )
    parser.add_argument("--compute-mi", action="store_true")
    parser.add_argument("--skip-mi", action="store_true")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-tuning-bits", type=int, default=8)
    parser.add_argument("--cv-splits", type=int, default=2)
    args = parser.parse_args()
    if args.validation_repeats < 1:
        raise ValueError("--validation-repeats must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    block_rows = []
    validation_rows = []
    validation_trial_rows = []
    validation_device_rows = []
    repeated_families = (
        None
        if args.repeated_validation_families is None
        else set(args.repeated_validation_families)
    )
    for family in load_hdf5_artifact(args.artifact):
        observations = PUFObservations.from_family(family)
        error_binary = observations.error_binary
        flat_error_binary = observations.flat_error_binary()
        per_device_flip_rate = error_binary.mean(axis=1)
        qc = artifact_qc_summary(observations)
        qc.update(
            {
                "family": family.family,
                "effective_independent_error_bits": effective_independent_bits(
                    error_binary
                ),
                "expected_independent_unique_rows": expected_independent_unique_rows(
                    error_binary,
                    n_simulations=args.unique_simulations,
                    random_state=args.random_state,
                ),
                "empirical_block_failure": empirical_block_failure_probability(
                    flat_error_binary,
                    block_size=min(args.block_size, observations.n_bits),
                    correction_capacity=args.correction_capacity,
                    n_blocks=args.n_blocks,
                    random_state=args.random_state,
                ),
                "independent_block_failure": independent_block_failure_probability(
                    per_device_flip_rate,
                    block_size=min(args.block_size, observations.n_bits),
                    correction_capacity=args.correction_capacity,
                    n_blocks=args.n_blocks,
                    random_state=args.random_state,
                ),
            }
        )
        qc_rows.append(qc)

        block_size = min(args.block_size, observations.n_bits)
        empirical_curve = empirical_block_failure_curve(
            flat_error_binary,
            block_size=block_size,
            max_correction_capacity=args.max_correction_capacity,
            n_blocks=args.n_blocks,
            random_state=args.random_state,
        )
        independent_curve = independent_block_failure_curve(
            per_device_flip_rate,
            block_size=block_size,
            max_correction_capacity=args.max_correction_capacity,
            n_blocks=args.n_blocks,
            random_state=args.random_state,
        )
        for capacity, empirical, independent in zip(
            range(args.max_correction_capacity + 1),
            empirical_curve,
            independent_curve,
        ):
            block_rows.append(
                {
                    "family": family.family,
                    "block_size": block_size,
                    "correction_capacity": capacity,
                    "empirical_block_failure": float(empirical),
                    "independent_block_failure": float(independent),
                    "failure_gap": float(empirical - independent),
                }
            )

        if args.compute_mi and not args.skip_mi:
            mi = pairwise_mutual_information(error_binary)
            np.save(args.output_dir / f"{family.family}.pairwise_mi.npy", mi)

        if args.run_validation:
            n_repeats = (
                args.validation_repeats
                if repeated_families is None or family.family in repeated_families
                else 1
            )
            for split in range(n_repeats):
                independent, graph, trial, device_results = (
                    validate_error_graph_split(
                        observations,
                        family=family.family,
                        split=split,
                        C=args.C,
                        random_state=args.random_state + split,
                        max_iter=args.max_iter,
                        n_jobs=args.n_jobs,
                        max_tuning_bits=args.max_tuning_bits,
                        cv_splits=args.cv_splits,
                    )
                )
                if split == 0:
                    validation_rows.extend([asdict(independent), asdict(graph)])
                validation_trial_rows.append(asdict(trial))
                validation_device_rows.extend(
                    asdict(result) for result in device_results
                )

    pl.DataFrame(qc_rows).write_csv(args.output_dir / "artifact_qc.csv")
    pl.DataFrame(block_rows).write_csv(args.output_dir / "block_failure_curve.csv")
    if validation_rows:
        pl.DataFrame(validation_rows).write_csv(args.output_dir / "validation.csv")
        pl.DataFrame(validation_trial_rows).write_csv(
            args.output_dir / "validation_trials.csv"
        )
        pl.DataFrame(validation_device_rows).write_csv(
            args.output_dir / "validation_by_device.csv"
        )


if __name__ == "__main__":
    main()
