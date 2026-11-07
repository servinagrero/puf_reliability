import argparse
from pathlib import Path

import polars as pl

from pufre import DATA_DIR, logger
from pufre.data import PUFObservations
from pufre.datasets import load_hdf5_artifact
from pufre.graph.study import fit_error_graph_study, save_error_graph_study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--study-name", default="error_dependency")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--regularization-csv", type=Path, default=None)
    parser.add_argument("--max-tuning-bits", type=int, default=32)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.C is not None and args.regularization_csv is not None:
        parser.error("--C and --regularization-csv are mutually exclusive")

    output_dir = args.output_dir or (DATA_DIR / args.study_name)
    regularization = load_regularization(args.regularization_csv)
    frames = []
    for family in load_hdf5_artifact(args.artifact):
        logger.info("Running directed error-dependency study %s", family.family)
        family_C = regularization.get(family.family, args.C)
        result = fit_error_graph_study(
            PUFObservations.from_family(family),
            name=family.family,
            C=family_C,
            max_tuning_bits=args.max_tuning_bits,
            cv_splits=args.cv_splits,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        save_error_graph_study(result, output_dir / family.family)
        frames.append(
            pl.DataFrame(result.summary).with_columns(
                pl.lit(family.family).alias("family")
            )
        )

    if not frames:
        raise SystemExit(f"No families found in {args.artifact}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.concat(frames, how="diagonal_relaxed").write_csv(output_dir / "summary.csv")


def load_regularization(path: Path | None):
    if path is None:
        return {}
    frame = pl.read_csv(path).filter(pl.col("model") == "error_graph")
    required = {"family", "C"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["family"].n_unique() != frame.height:
        raise ValueError(f"Duplicate error_graph families in {path}")
    return dict(zip(frame["family"].to_list(), frame["C"].to_list()))


if __name__ == "__main__":
    main()
