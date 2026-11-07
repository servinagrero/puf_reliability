import argparse
from pathlib import Path

import polars as pl

from pufre import DATA_DIR, logger
from pufre.data import PUFObservations
from pufre.datasets import families_by_name
from pufre.error_metrics import conditional_independence_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "error_hypothesis",
    )
    parser.add_argument("--families", action="append", nargs="+", default=None)
    parser.add_argument("--n-permutations", type=int, default=199)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    families = families_by_name(args.artifact)
    selected = normalize_families(args.families) or set(families)
    summary_rows = []
    null_rows = []

    for family_name in sorted(selected):
        if family_name not in families:
            raise ValueError(f"Family {family_name!r} is missing from the artifact")
        logger.info("Testing conditional independence for %s", family_name)
        observations = PUFObservations.from_family(families[family_name])
        result = conditional_independence_test(
            observations.error_binary,
            n_permutations=args.n_permutations,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        summary_rows.extend(
            {"family": family_name, **row} for row in result.summary_rows()
        )
        null_rows.extend(
            {
                "family": family_name,
                "permutation": index,
                "correlation_energy": float(values[0]),
                "error_count_variance": float(values[1]),
            }
            for index, values in enumerate(result.null)
        )

    if not summary_rows:
        raise SystemExit("No families selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(
        args.output_dir / "permutation_summary.csv"
    )
    pl.DataFrame(null_rows).write_csv(args.output_dir / "permutation_null.csv")


def normalize_families(family_groups):
    if family_groups is None:
        return None
    return {family for group in family_groups for family in group}


if __name__ == "__main__":
    main()
