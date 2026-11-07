import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pufre import DATA_DIR, FIGURES_DIR
from pufre.plotting.graph import (
    plot_avalanche_vs_flip_rate,
    plot_community_participation,
    plot_distance_to_roles,
    plot_edge_stability_histogram,
    plot_graph_role_counts,
    plot_graph_summary_edges,
    plot_hub_authority_quadrants,
    plot_mi_graph_correlations,
    plot_mi_vs_graph_edges,
    plot_resistance_by_role,
    plot_role_flip_rate,
    plot_spectral_embedding,
    plot_stable_adjacency_heatmap,
)

try:
    plt.style.use("publications")
except OSError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=DATA_DIR / "error_graph")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR / "error_graph")
    parser.add_argument("--families", action="append", nargs="+", default=None)
    args = parser.parse_args()

    selected_families = normalize_families(args.families)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.graph_dir / "graph_summary.csv"
    if summary_path.exists():
        summary = pl.read_csv(summary_path)
        if selected_families is not None:
            summary = summary.filter(pl.col("family").is_in(selected_families))
        plot_graph_summary_edges(summary, args.figures_dir / "graph_edges.png")
        plot_mi_graph_correlations(
            summary,
            args.figures_dir / "mi_graph_correlations.png",
        )

    for family_dir in sorted(
        path for path in args.graph_dir.iterdir() if path.is_dir()
    ):
        family = family_dir.name
        if selected_families is not None and family not in selected_families:
            continue
        output_dir = args.figures_dir / family
        output_dir.mkdir(parents=True, exist_ok=True)
        bit_metrics = pl.read_csv(family_dir / "bit_graph_metrics.csv")
        stable_roles_path = family_dir / "stable_bit_graph_roles.csv"
        if not stable_roles_path.exists():
            raise FileNotFoundError(
                f"Missing bootstrap-stable role cache: {stable_roles_path}"
            )
        stable_roles = pl.read_csv(stable_roles_path)
        if stable_roles["bit_id"].to_list() != bit_metrics["bit_id"].to_list():
            raise ValueError(
                f"Stable and unbootstrapped bit identifiers differ for {family}"
            )

        plot_hub_authority_quadrants(
            stable_roles,
            output_dir / f"{family}_hub_authority_quadrants.png",
        )
        plot_graph_role_counts(
            stable_roles,
            output_dir / f"{family}_graph_role_counts.png",
        )
        plot_role_flip_rate(
            stable_roles,
            output_dir / f"{family}_role_flip_rate.png",
        )
        plot_avalanche_vs_flip_rate(
            bit_metrics,
            output_dir / f"{family}_avalanche_vs_flip_rate.png",
        )
        plot_resistance_by_role(
            bit_metrics,
            output_dir / f"{family}_resistance_by_role.png",
        )
        plot_spectral_embedding(
            bit_metrics,
            output_dir / f"{family}_spectral_embedding.png",
        )
        plot_distance_to_roles(
            bit_metrics,
            output_dir / f"{family}_distance_to_roles.png",
        )
        plot_community_participation(
            bit_metrics,
            output_dir / f"{family}_community_participation.png",
        )

        edge_compare_path = family_dir / "mi_vs_graph_edges.csv"
        if edge_compare_path.exists():
            plot_mi_vs_graph_edges(
                pl.read_csv(edge_compare_path),
                output_dir / f"{family}_mi_vs_graph_edges.png",
            )

        stable_adjacency_path = family_dir / "stable_undirected_adjacency.npy"
        if stable_adjacency_path.exists():
            plot_stable_adjacency_heatmap(
                np.load(stable_adjacency_path),
                output_dir / f"{family}_stable_adjacency.png",
            )

        edge_stability_path = family_dir / "edge_stability.csv"
        # Legacy zero-edge caches were written as a single newline.
        if edge_stability_path.exists() and edge_stability_path.stat().st_size > 1:
            plot_edge_stability_histogram(
                pl.read_csv(edge_stability_path),
                output_dir / f"{family}_edge_stability_histogram.png",
            )


def normalize_families(family_groups):
    if family_groups is None:
        return None
    return {family for group in family_groups for family in group}


if __name__ == "__main__":
    main()
