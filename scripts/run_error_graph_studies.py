import argparse
from pathlib import Path

import numpy as np
import polars as pl

from pufre import DATA_DIR, logger
from pufre.data import PUFObservations
from pufre.datasets import families_by_name
from pufre.error_metrics import pairwise_mutual_information
from pufre.graph import (
    Role,
    avalanche_metric_frame,
    bootstrap_edge_stability,
    community_metric_frame,
    compare_mi_and_graph,
    edge_stability_frame,
    effective_resistance_frame,
    graph_assortativity_metrics,
    graph_role_frame,
    graph_summary_metrics,
    role_distance_frame,
    spectral_metric_frame,
    stable_adjacency,
)
from pufre.graph.study import load_error_graph_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DATA_DIR / "puf_artifact.h5")
    parser.add_argument("--study-dir", type=Path, default=DATA_DIR / "error_dependency")
    parser.add_argument("--qc-dir", type=Path, default=DATA_DIR / "error_qc")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "error_graph")
    parser.add_argument("--families", action="append", nargs="+", default=None)
    parser.add_argument("--role-quantile", type=float, default=0.75)
    parser.add_argument("--community-threshold-quantile", type=float, default=0.9)
    parser.add_argument("--reachability-threshold-quantile", type=float, default=0.9)
    parser.add_argument("--mi-edge-limit", type=int, default=5000)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-C", type=float, default=None)
    parser.add_argument("--bootstrap-edge-threshold", type=float, default=1e-8)
    parser.add_argument("--bootstrap-min-probability", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    families = families_by_name(args.artifact)
    selected_families = normalize_families(args.families)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for family_name, model_path, summary_path in iter_error_graph_models(
        args.study_dir
    ):
        if selected_families is not None and family_name not in selected_families:
            continue
        if family_name not in families:
            logger.warning(
                "Skipping %s because it is missing from artifact", family_name
            )
            continue

        logger.info("Computing graph study %s", family_name)
        family_dir = args.output_dir / family_name
        family_dir.mkdir(parents=True, exist_ok=True)

        model = load_error_graph_model(model_path)
        bootstrap_C = (
            model.regularization
            if args.bootstrap_C is None
            else args.bootstrap_C
        )
        observations = PUFObservations.from_family(families[family_name])
        summary = load_npz(summary_path)
        bit_roles = graph_role_frame(
            model.adjacency,
            bit_ids=summary["bit_id"],
            physical_bit_ids=summary["physical_bit_id"],
            flip_rate=summary["flip_rate"],
            reliable_entropy=summary["reliable_entropy"],
            high_quantile=args.role_quantile,
        )
        communities = community_metric_frame(
            model.undirected_adjacency,
            bit_ids=summary["bit_id"],
            threshold_quantile=args.community_threshold_quantile,
        )
        avalanche = avalanche_metric_frame(
            model.adjacency,
            bit_ids=summary["bit_id"],
            threshold_quantile=args.reachability_threshold_quantile,
        )
        spectral = spectral_metric_frame(model.adjacency, bit_ids=summary["bit_id"])
        resistance = effective_resistance_frame(
            model.undirected_adjacency,
            bit_ids=summary["bit_id"],
            roles=bit_roles["graph_role"].to_numpy(),
        )
        role_distance = role_distance_frame(
            model.adjacency,
            bit_ids=summary["bit_id"],
            roles=bit_roles["graph_role"].to_numpy(),
            threshold_quantile=args.reachability_threshold_quantile,
        )
        bit_graph = (
            bit_roles.join(communities, on="bit_id", how="left")
            .join(avalanche, on="bit_id", how="left")
            .join(spectral, on="bit_id", how="left")
            .join(resistance, on="bit_id", how="left")
            .join(role_distance, on="bit_id", how="left")
        )
        bit_graph.write_csv(family_dir / "bit_graph_metrics.csv")

        mi = load_or_compute_mi(args.qc_dir, family_name, observations)
        edge_compare, correlations = compare_mi_and_graph(
            mi, model.undirected_adjacency
        )
        edge_compare = edge_compare.sort(
            ["graph_abs_weight", "mutual_information"],
            descending=[True, True],
        )
        edge_compare.head(args.mi_edge_limit).write_csv(
            family_dir / "mi_vs_graph_edges.csv"
        )
        np.save(family_dir / "mi.npy", mi)

        graph_summary = graph_summary_metrics(model.undirected_adjacency)
        graph_summary.update(
            graph_assortativity_metrics(model.undirected_adjacency, bit_graph)
        )
        graph_summary.update(
            {
                "family": family_name,
                "regularization": model.regularization,
                "mi_graph_pearson": correlations["pearson"],
                "mi_graph_spearman": correlations["spearman"],
                **{
                    f"n_{role.value}_bits": bit_graph.filter(
                        pl.col("graph_role") == role.value
                    ).height
                    for role in Role
                },
            }
        )

        if args.bootstrap_replicates > 0:
            stability = bootstrap_edge_stability(
                observations,
                n_bootstrap=args.bootstrap_replicates,
                C=bootstrap_C,
                edge_threshold=args.bootstrap_edge_threshold,
                random_state=args.random_state,
                max_iter=args.max_iter,
                n_jobs=args.n_jobs,
            )
            stable_directed = stable_adjacency(
                stability,
                min_probability=args.bootstrap_min_probability,
            )
            stable_undirected = (stable_directed + stable_directed.T) / 2.0
            np.fill_diagonal(stable_undirected, 0.0)
            np.save(family_dir / "stable_directed_adjacency.npy", stable_directed)
            np.save(family_dir / "stable_undirected_adjacency.npy", stable_undirected)
            np.save(
                family_dir / "edge_selection_probability.npy",
                stability.selection_probability,
            )
            np.save(
                family_dir / "bootstrap_adjacencies.npy",
                stability.adjacency_samples,
            )
            edge_stability_frame(
                stability,
                min_probability=args.bootstrap_min_probability,
            ).write_csv(family_dir / "edge_stability.csv")
            stable_roles = graph_role_frame(
                stable_directed,
                bit_ids=summary["bit_id"],
                physical_bit_ids=summary["physical_bit_id"],
                flip_rate=summary["flip_rate"],
                reliable_entropy=summary["reliable_entropy"],
                high_quantile=args.role_quantile,
            )
            stable_roles.write_csv(family_dir / "stable_bit_graph_roles.csv")
            graph_summary["n_stable_directed_edges"] = int(
                np.count_nonzero(stable_directed)
            )

        summary_rows.append(graph_summary)

    if not summary_rows:
        raise SystemExit(
            f"No saved directed error-graph models found under {args.study_dir}"
        )
    pl.DataFrame(summary_rows).write_csv(args.output_dir / "graph_summary.csv")


def iter_error_graph_models(study_dir: Path):
    for model_path in sorted(study_dir.glob("*/*.model.npz")):
        family_dir = model_path.parent
        family_name = family_dir.name
        summary_path = family_dir / f"{family_name}.summary.npz"
        if summary_path.exists():
            yield family_name, model_path, summary_path


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_or_compute_mi(qc_dir: Path, family_name: str, observations: PUFObservations):
    mi_path = qc_dir / f"{family_name}.pairwise_mi.npy"
    if mi_path.exists():
        return np.load(mi_path)
    return pairwise_mutual_information(observations.error_binary)


def normalize_families(family_groups):
    if family_groups is None:
        return None
    return {family for group in family_groups for family in group}


if __name__ == "__main__":
    main()
