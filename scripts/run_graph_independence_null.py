import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pufre import DATA_DIR, logger
from pufre.data import PUFObservations
from pufre.datasets import families_by_name
from pufre.graph import (
    ROLE_VALUES,
    bootstrap_edge_stability,
    edge_stability_frame,
    graph_bootstrap_metric_frame,
    graph_role_frame,
    stable_adjacency,
)
from pufre.graph.study import load_error_graph_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--graph-dir", type=Path, default=DATA_DIR / "error_graph")
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=DATA_DIR / "error_dependency",
    )
    parser.add_argument("--families", action="append", nargs="+", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10)
    parser.add_argument("--bootstrap-C", type=float, default=None)
    parser.add_argument("--bootstrap-edge-threshold", type=float, default=1e-8)
    parser.add_argument("--bootstrap-min-probability", type=float, default=0.5)
    parser.add_argument("--role-quantile", type=float, default=0.75)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    families = families_by_name(args.artifact)
    selected = normalize_families(args.families) or set(families)
    rows = []
    distribution_frames = []
    for family_name in sorted(selected):
        if family_name not in families:
            raise ValueError(f"Family {family_name!r} is missing from the artifact")
        family_dir = args.graph_dir / family_name
        observed_path = family_dir / "stable_directed_adjacency.npy"
        observed_roles_path = family_dir / "stable_bit_graph_roles.csv"
        observed_samples_path = family_dir / "bootstrap_adjacencies.npy"
        if (
            not observed_path.exists()
            or not observed_roles_path.exists()
            or not observed_samples_path.exists()
        ):
            raise FileNotFoundError(
                f"Missing observed stable graph for {family_name}; run the graph study first"
            )

        model_path = args.study_dir / family_name / f"{family_name}.model.npz"
        if args.bootstrap_C is None:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Missing fitted model for regularization lookup: {model_path}"
                )
            regularization = load_error_graph_model(model_path).regularization
        else:
            regularization = args.bootstrap_C

        logger.info("Computing device-preserving independence null for %s", family_name)
        observations = PUFObservations.from_family(families[family_name])
        stability = bootstrap_edge_stability(
            observations,
            n_bootstrap=args.bootstrap_replicates,
            C=regularization,
            edge_threshold=args.bootstrap_edge_threshold,
            random_state=args.random_state,
            max_iter=args.max_iter,
            independent_null=True,
            n_jobs=args.n_jobs,
        )
        null_adjacency = stable_adjacency(
            stability,
            min_probability=args.bootstrap_min_probability,
        )
        null_roles = graph_role_frame(
            null_adjacency,
            bit_ids=observations.bit_ids,
            physical_bit_ids=observations.physical_bit_ids,
            flip_rate=observations.error_binary.mean(axis=(0, 1)),
            high_quantile=args.role_quantile,
        )

        family_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            family_dir / "null_stable_directed_adjacency.npy",
            null_adjacency,
        )
        np.save(
            family_dir / "null_edge_selection_probability.npy",
            stability.selection_probability,
        )
        np.save(
            family_dir / "null_bootstrap_adjacencies.npy",
            stability.adjacency_samples,
        )
        edge_stability_frame(
            stability,
            min_probability=args.bootstrap_min_probability,
        ).write_csv(family_dir / "null_edge_stability.csv")
        null_roles.write_csv(family_dir / "null_stable_bit_graph_roles.csv")

        observed_adjacency = np.load(observed_path)
        observed_roles = pl.read_csv(observed_roles_path)
        row = {
            "family": family_name,
            "regularization": regularization,
            "observed_stable_directed_edges": int(np.count_nonzero(observed_adjacency)),
            "null_stable_directed_edges": int(np.count_nonzero(null_adjacency)),
        }
        for role in ROLE_VALUES:
            row[f"observed_{role}_bits"] = observed_roles.filter(
                pl.col("graph_role") == role
            ).height
            row[f"null_{role}_bits"] = null_roles.filter(
                pl.col("graph_role") == role
            ).height
        rows.append(row)
        for graph_source, samples in [
            ("observed", np.load(observed_samples_path)),
            ("null", stability.adjacency_samples),
        ]:
            distribution_frames.append(
                graph_bootstrap_metric_frame(
                    samples,
                    edge_threshold=args.bootstrap_edge_threshold,
                ).with_columns(
                    pl.lit(family_name).alias("family"),
                    pl.lit(graph_source).alias("graph_source"),
                )
            )

    args.graph_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.graph_dir / "independence_null_summary.csv")
    pl.concat(distribution_frames).select(
        "family",
        "graph_source",
        "bootstrap",
        "directed_edges",
        "total_abs_weight",
        "frobenius_norm",
    ).write_csv(args.graph_dir / "graph_bootstrap_distribution.csv")


def normalize_families(family_groups):
    if family_groups is None:
        return None
    return {family for group in family_groups for family in group}


if __name__ == "__main__":
    main()
