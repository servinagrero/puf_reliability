from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from pufre.graph import ROLE_VALUES, Role

ROLE_COLORS = {
    Role.DISCONNECTED.value: "black",
    Role.PERIPHERAL.value: "tab:gray",
    Role.SOURCE.value: "tab:blue",
    Role.SINK.value: "tab:orange",
    Role.RELAY.value: "tab:red",
    Role.INTERMEDIATE.value: "tab:green",
}


def plot_graph_summary_edges(frame: pl.DataFrame, output_path: Path):
    if frame.is_empty():
        return
    data = frame.select(["family", "n_edges", "density"]).sort("n_edges")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["family"], data["n_edges"])
    ax.set_xlabel("Edges above graph threshold")
    ax_top = ax.twiny()
    ax_top.plot(data["density"], np.arange(data.height), "o", color="tab:orange")
    ax_top.set_xlabel("Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mi_graph_correlations(frame: pl.DataFrame, output_path: Path):
    if frame.is_empty():
        return
    data = frame.select(["family", "mi_graph_spearman"]).sort("mi_graph_spearman")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["family"], data["mi_graph_spearman"])
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Spearman correlation between MI and graph weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hub_authority_quadrants(frame: pl.DataFrame, output_path: Path):
    colors = role_colors(frame["graph_role"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        frame["out_strength"],
        frame["in_strength"],
        c=colors,
        s=12,
        alpha=0.8,
    )
    del scatter
    ax.set_xlabel("Out strength / predictive hub score")
    ax.set_ylabel("In strength / predictive authority score")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_avalanche_vs_flip_rate(frame: pl.DataFrame, output_path: Path):
    required = {"flip_rate", "reachability_fraction", "avalanche_gain", "graph_role"}
    if not required.issubset(frame.columns):
        return
    colors = role_colors(frame["graph_role"].to_numpy())
    sizes = 10 + 50 * finite_scaled(frame["avalanche_gain"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        frame["flip_rate"],
        frame["reachability_fraction"],
        c=colors,
        s=sizes,
        alpha=0.75,
    )
    ax.set_xlabel("Flip rate")
    ax.set_ylabel("3-hop error reachability")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_resistance_by_role(frame: pl.DataFrame, output_path: Path):
    if "mean_effective_resistance" not in frame.columns:
        return
    roles = ROLE_VALUES
    role_values = [
        finite_values(
            frame.filter(pl.col("graph_role") == role)[
                "mean_effective_resistance"
            ].to_numpy()
        )
        for role in roles
    ]
    positions = [
        index + 1 for index, values in enumerate(role_values) if values.size
    ]
    values = [values for values in role_values if values.size]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, positions=positions, showfliers=False)
    ax.set_xticks(np.arange(1, len(roles) + 1), roles, rotation=30)
    ax.set_xlabel("Graph role")
    ax.set_ylabel("Mean effective resistance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_spectral_embedding(frame: pl.DataFrame, output_path: Path):
    required = {"fiedler_coordinate", "adjacency_eigenvector", "graph_role"}
    if not required.issubset(frame.columns):
        return
    colors = role_colors(frame["graph_role"].to_numpy())
    sizes = 10 + 50 * finite_scaled(frame["spectral_participation"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        frame["fiedler_coordinate"],
        frame["adjacency_eigenvector"],
        c=colors,
        s=sizes,
        alpha=0.75,
    )
    ax.set_xlabel("Fiedler coordinate")
    ax.set_ylabel("Adjacency eigenvector centrality")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_distance_to_roles(frame: pl.DataFrame, output_path: Path):
    columns = [
        "distance_to_nearest_source",
        "distance_to_nearest_sink",
        "distance_to_nearest_relay",
    ]
    if not set(columns).issubset(frame.columns):
        return
    labels = ["source", "sink", "relay"]
    values = [finite_values(frame[column].to_numpy()) for column in columns]
    labels = [label for label, value in zip(labels, values) if value.size]
    values = [value for value in values if value.size]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Target role")
    ax.set_ylabel("Directed graph distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_community_participation(frame: pl.DataFrame, output_path: Path):
    required = {
        "within_community_strength",
        "bridge_strength",
        "participation_coefficient",
        "graph_role",
    }
    if not required.issubset(frame.columns):
        return
    colors = role_colors(frame["graph_role"].to_numpy())
    sizes = 10 + 50 * finite_scaled(frame["participation_coefficient"].to_numpy())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        frame["within_community_strength"],
        frame["bridge_strength"],
        c=colors,
        s=sizes,
        alpha=0.75,
    )
    ax.set_xlabel("Within-community strength")
    ax.set_ylabel("Bridge strength")
    add_role_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_graph_role_counts(frame: pl.DataFrame, output_path: Path):
    observed = {
        role: count
        for role, count in frame.group_by("graph_role").len().iter_rows()
    }
    counts = [observed.get(role, 0) for role in ROLE_VALUES]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ROLE_VALUES,
        counts,
        color=role_colors(ROLE_VALUES),
    )
    ax.set_xlabel("Graph role")
    ax.set_ylabel("Bit count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_role_flip_rate(frame: pl.DataFrame, output_path: Path):
    if "flip_rate" not in frame.columns:
        return
    roles = ROLE_VALUES
    role_values = [
        frame.filter(pl.col("graph_role") == role)["flip_rate"].to_numpy()
        for role in roles
    ]
    positions = [
        index + 1 for index, values in enumerate(role_values) if values.size
    ]
    values = [values for values in role_values if values.size]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(values, positions=positions, showfliers=False)
    ax.set_xticks(np.arange(1, len(roles) + 1), roles, rotation=30)
    ax.set_xlabel("Graph role")
    ax.set_ylabel("Flip rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mi_vs_graph_edges(frame: pl.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        frame["mutual_information"],
        frame["graph_abs_weight"],
        s=6,
        alpha=0.5,
    )
    ax.set_xlabel("Pairwise error MI")
    ax.set_ylabel("Absolute undirected graph weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_stable_adjacency_heatmap(matrix, output_path: Path):
    matrix = np.asarray(matrix, dtype=float)
    limit = np.nanmax(np.abs(matrix))
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    fig.colorbar(im, ax=ax, label="Bootstrap-stable graph weight")
    ax.set_xlabel("Bit index")
    ax.set_ylabel("Bit index")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_edge_stability_histogram(frame: pl.DataFrame, output_path: Path):
    if frame.is_empty():
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frame["selection_probability"], bins=20)
    ax.set_xlabel("Bootstrap edge-selection probability")
    ax.set_ylabel("Directed edge count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def role_colors(roles):
    return [ROLE_COLORS.get(str(role), "black") for role in roles]


def add_role_legend(ax):
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=role)
        for role, color in ROLE_COLORS.items()
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def finite_scaled(values):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    span = values.max() - values.min()
    if span == 0.0:
        return np.zeros_like(values)
    return (values - values.min()) / span
