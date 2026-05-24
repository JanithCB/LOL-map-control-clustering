# src/clustering/cluster_analysis.py

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def cluster_size_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return size and proportion summary for each cluster.
    """
    if "cluster_label" not in clustered_df.columns:
        raise KeyError("clustered_df must contain a 'cluster_label' column.")

    total = len(clustered_df)
    if total == 0:
        raise ValueError("clustered_df is empty.")

    summary = (
        clustered_df["cluster_label"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("cluster_label")
        .reset_index(name="n_samples")
    )
    summary["proportion"] = summary["n_samples"] / total
    return summary


def top_features_per_cluster(
    centroids_df: pd.DataFrame,
    top_n: int = 8,
    exclude_cols: tuple[str, ...] = ("cluster_label",),
) -> pd.DataFrame:
    """
    For each cluster, return the top_n highest centroid features.
    """
    if "cluster_label" not in centroids_df.columns:
        raise KeyError("centroids_df must contain a 'cluster_label' column.")

    feature_cols = [c for c in centroids_df.columns if c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No centroid feature columns found.")

    rows: list[dict] = []

    for _, row in centroids_df.iterrows():
        cluster_label = int(row["cluster_label"])
        feature_values = (
            row[feature_cols]
            .sort_values(ascending=False)
            .head(top_n)
        )

        for rank, (feature, value) in enumerate(feature_values.items(), start=1):
            rows.append(
                {
                    "cluster_label": cluster_label,
                    "rank": rank,
                    "feature": feature,
                    "centroid_value": float(value),
                }
            )

    return pd.DataFrame(rows)


def representative_samples(
    clustered_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 6,
) -> pd.DataFrame:
    """
    Return the top_n closest samples to each cluster centroid.
    """
    if "cluster_label" not in clustered_df.columns:
        raise KeyError("clustered_df must contain a 'cluster_label' column.")
    if "cluster_label" not in centroids_df.columns:
        raise KeyError("centroids_df must contain a 'cluster_label' column.")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")

    missing_feature_cols = [c for c in feature_cols if c not in clustered_df.columns]
    if missing_feature_cols:
        raise KeyError(f"Missing feature columns in clustered_df: {missing_feature_cols}")

    results: list[dict] = []

    centroid_lookup = centroids_df.set_index("cluster_label")

    for cluster_label, group in clustered_df.groupby("cluster_label"):
        if cluster_label not in centroid_lookup.index:
            LOGGER.warning("Cluster %s missing in centroids_df; skipping.", cluster_label)
            continue

        centroid = centroid_lookup.loc[cluster_label, feature_cols].to_numpy(dtype=float)
        X = group[feature_cols].to_numpy(dtype=float)
        distances = np.linalg.norm(X - centroid, axis=1)

        group_out = group.copy()
        group_out["distance_to_centroid"] = distances
        group_out = group_out.sort_values("distance_to_centroid").head(top_n)

        for rank, (_, sample) in enumerate(group_out.iterrows(), start=1):
            row = {
                "cluster_label": int(cluster_label),
                "rank_in_cluster": rank,
                "distance_to_centroid": float(sample["distance_to_centroid"]),
            }

            if "image_id" in sample.index:
                row["image_id"] = sample["image_id"]
                # Derive image_path from image_id so GUI can load thumbnails
                row["image_path"] = f"mid_dataset/images/{sample['image_id']}"
            if "label_file" in sample.index:
                row["label_file"] = sample["label_file"]

            results.append(row)

    columns = ["cluster_label", "image_id", "rank_in_cluster", "distance_to_centroid"]
    if "label_file" in clustered_df.columns:
        columns.append("label_file")

    out_df = pd.DataFrame(results)
    existing_cols = [c for c in columns if c in out_df.columns]
    return out_df[existing_cols]


def cluster_profile_table(
    clustered_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a compact cluster profile table.
    """
    size_df = cluster_size_summary(clustered_df)

    if "cluster_label" not in centroids_df.columns:
        raise KeyError("centroids_df must contain a 'cluster_label' column.")

    feature_cols = [c for c in centroids_df.columns if c != "cluster_label"]
    centroid_abs_mean = centroids_df.copy()
    if feature_cols:
        centroid_abs_mean["avg_abs_centroid_value"] = (
            centroid_abs_mean[feature_cols].abs().mean(axis=1)
        )
        centroid_abs_mean = centroid_abs_mean[["cluster_label", "avg_abs_centroid_value"]]
        profile_df = size_df.merge(centroid_abs_mean, on="cluster_label", how="left")
    else:
        profile_df = size_df.copy()

    return profile_df.sort_values("cluster_label").reset_index(drop=True)


def save_cluster_analysis_tables(
    size_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save analysis tables to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size_path = output_dir / "cluster_sizes.csv"
    top_features_path = output_dir / "cluster_top_features.csv"
    samples_path = output_dir / "representative_samples.csv"

    size_df.to_csv(size_path, index=False)
    top_features_df.to_csv(top_features_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    return {
        "cluster_sizes": size_path,
        "cluster_top_features": top_features_path,
        "representative_samples": samples_path,
    }