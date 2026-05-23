# src/visualization/plot_clusters.py

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)


def plot_cluster_counts(clustered_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Plot cluster counts as a bar chart.
    """
    if "cluster_label" not in clustered_df.columns:
        raise KeyError("clustered_df must contain 'cluster_label'.")
    if clustered_df.empty:
        raise ValueError("clustered_df is empty.")

    counts = clustered_df["cluster_label"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(counts))
    ax.bar(x_positions, counts.values)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(label) for label in counts.index])
    ax.set_title("Cluster Counts")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pca_projection(
    clustered_df: pd.DataFrame,
    feature_cols: list[str],
    output_path: str | Path,
) -> None:
    """
    Plot 2D PCA projection colored by cluster label.
    """
    if clustered_df.empty:
        raise ValueError("clustered_df is empty.")
    if "cluster_label" not in clustered_df.columns:
        raise KeyError("clustered_df must contain 'cluster_label'.")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")

    X = clustered_df[feature_cols].to_numpy(dtype=float)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    labels = clustered_df["cluster_label"].to_numpy()
    unique_labels = sorted(pd.unique(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for idx, cluster_label in enumerate(unique_labels):
        mask = labels == cluster_label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=18,
            alpha=0.75,
            label=f"Cluster {cluster_label}",
            color=cmap(idx),
        )

    ax.set_title("PCA Projection of Snapshot Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cluster_centroid_heatmap(
    centroids_df: pd.DataFrame,
    output_path: str | Path,
    max_features: int = 20,
) -> None:
    """
    Plot centroid heatmap using the features with highest variance across centroids.
    """
    if centroids_df.empty:
        raise ValueError("centroids_df is empty.")
    if "cluster_label" not in centroids_df.columns:
        raise KeyError("centroids_df must contain 'cluster_label'.")

    feature_cols = [c for c in centroids_df.columns if c != "cluster_label"]
    if not feature_cols:
        raise ValueError("No centroid feature columns available.")

    feature_var = centroids_df[feature_cols].var(axis=0).sort_values(ascending=False)
    selected_features = feature_var.head(max_features).index.tolist()

    heatmap_data = centroids_df.set_index("cluster_label")[selected_features]

    fig, ax = plt.subplots(figsize=(max(10, len(selected_features) * 0.5), 5))
    im = ax.imshow(heatmap_data.to_numpy(), aspect="auto", cmap="viridis")

    ax.set_title("Cluster Centroid Heatmap")
    ax.set_xlabel("Features")
    ax.set_ylabel("Cluster Label")
    ax.set_xticks(np.arange(len(selected_features)))
    ax.set_xticklabels(selected_features, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index.tolist())

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Centroid Value")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_default_cluster_plots(
    clustered_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save the default clustering plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_path = output_dir / "cluster_counts.png"
    pca_path = output_dir / "cluster_pca.png"
    heatmap_path = output_dir / "centroid_heatmap.png"

    plot_cluster_counts(clustered_df, counts_path)
    plot_pca_projection(clustered_df, feature_cols, pca_path)
    plot_cluster_centroid_heatmap(centroids_df, heatmap_path)

    return {
        "cluster_counts": counts_path,
        "cluster_pca": pca_path,
        "centroid_heatmap": heatmap_path,
    }