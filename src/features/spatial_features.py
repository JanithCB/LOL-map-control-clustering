# src/features/spatial_features.py

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "image_id",
    "x_center_norm",
    "y_center_norm",
    "primary_zone",
    "map_side",
}

PRIMARY_ZONES = [
    "top_lane",
    "mid_lane",
    "bot_lane",
    "top_jungle",
    "bot_jungle",
    "river",
    "baron_zone",
    "dragon_zone",
    "blue_base",
    "red_base",
]


def compute_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute condensed pairwise Euclidean distances for an array of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2)

    Returns
    -------
    np.ndarray
        Condensed vector of pairwise distances.
        Returns empty array if fewer than 2 points.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    n_points = points.shape[0]
    if n_points < 2:
        return np.array([], dtype=float)

    distances: list[float] = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(float(dist))
    return np.array(distances, dtype=float)


def _validate_snapshot_columns(snapshot_df: pd.DataFrame) -> None:
    """
    Validate required columns for feature computation.
    """
    missing = REQUIRED_COLUMNS - set(snapshot_df.columns)
    if missing:
        raise KeyError(
            f"Snapshot DataFrame is missing required columns: {sorted(missing)}"
        )


def _safe_std(series: pd.Series) -> float:
    """
    Population std with safe fallback.
    """
    if len(series) == 0:
        return 0.0
    value = float(series.std(ddof=0))
    return 0.0 if np.isnan(value) else value


def _safe_mean(series: pd.Series) -> float:
    """
    Mean with safe fallback.
    """
    if len(series) == 0:
        return 0.0
    value = float(series.mean())
    return 0.0 if np.isnan(value) else value


def _count_primary_zones(snapshot_df: pd.DataFrame) -> dict[str, int]:
    """
    Count detections by primary zone.
    """
    counts = snapshot_df["primary_zone"].value_counts(dropna=False).to_dict()
    return {f"{zone}_count": int(counts.get(zone, 0)) for zone in PRIMARY_ZONES}


def _compute_lane_entropy(snapshot_df: pd.DataFrame) -> float:
    """
    Compute entropy across top/mid/bot lane counts.
    """
    lane_cols = ["top_lane", "mid_lane", "bot_lane"]
    counts = snapshot_df["primary_zone"].value_counts()
    lane_counts = np.array([counts.get(zone, 0) for zone in lane_cols], dtype=float)

    total = lane_counts.sum()
    if total == 0:
        return 0.0

    probs = lane_counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_snapshot_features(snapshot_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute one flat feature dictionary for a single minimap snapshot.

    Parameters
    ----------
    snapshot_df : pd.DataFrame
        Rows for a single image_id.

    Returns
    -------
    dict[str, Any]
        Flat dictionary of numeric snapshot features.
    """
    _validate_snapshot_columns(snapshot_df)

    if snapshot_df["image_id"].nunique() > 1:
        raise ValueError("compute_snapshot_features expects rows from exactly one image_id")

    n_detections = int(len(snapshot_df))

    x = snapshot_df["x_center_norm"].astype(float)
    y = snapshot_df["y_center_norm"].astype(float)

    if "bbox_area_norm" in snapshot_df.columns:
        bbox_area = snapshot_df["bbox_area_norm"].astype(float)
    else:
        bbox_area = pd.Series(np.zeros(n_detections, dtype=float), index=snapshot_df.index)

    points = snapshot_df[["x_center_norm", "y_center_norm"]].astype(float).to_numpy()
    pairwise = compute_pairwise_distances(points)

    zone_counts = _count_primary_zones(snapshot_df)

    jungle_count = zone_counts["top_jungle_count"] + zone_counts["bot_jungle_count"]
    river_count = zone_counts["river_count"]
    base_count = zone_counts["blue_base_count"] + zone_counts["red_base_count"]
    dragon_zone_count = zone_counts["dragon_zone_count"]
    baron_zone_count = zone_counts["baron_zone_count"]

    map_side_counts = snapshot_df["map_side"].value_counts(dropna=False).to_dict()
    blue_side_count = int(map_side_counts.get("blue_side", 0))
    red_side_count = int(map_side_counts.get("red_side", 0))
    center_count = int(map_side_counts.get("center", 0))

    mean_x = _safe_mean(x)
    mean_y = _safe_mean(y)
    std_x = _safe_std(x)
    std_y = _safe_std(y)

    bbox_area_norm_mean = _safe_mean(bbox_area)
    bbox_area_norm_std = _safe_std(bbox_area)

    avg_pairwise_distance = float(pairwise.mean()) if pairwise.size else 0.0
    min_pairwise_distance = float(pairwise.min()) if pairwise.size else 0.0
    max_pairwise_distance = float(pairwise.max()) if pairwise.size else 0.0

    lane_entropy = _compute_lane_entropy(snapshot_df)

    spread_score = float((std_x + std_y + avg_pairwise_distance) / 3.0)
    grouping_score = float(1.0 / (1.0 + avg_pairwise_distance))

    features: dict[str, Any] = {
        "n_detections": n_detections,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "bbox_area_norm_mean": bbox_area_norm_mean,
        "bbox_area_norm_std": bbox_area_norm_std,
        "avg_pairwise_distance": avg_pairwise_distance,
        "min_pairwise_distance": min_pairwise_distance,
        "max_pairwise_distance": max_pairwise_distance,
        **zone_counts,
        "lane_count_top": zone_counts["top_lane_count"],
        "lane_count_mid": zone_counts["mid_lane_count"],
        "lane_count_bot": zone_counts["bot_lane_count"],
        "jungle_count": jungle_count,
        "river_count": river_count,
        "base_count": base_count,
        "dragon_zone_count": dragon_zone_count,
        "baron_zone_count": baron_zone_count,
        "blue_side_count": blue_side_count,
        "red_side_count": red_side_count,
        "center_count": center_count,
        "lane_entropy": lane_entropy,
        "spread_score": spread_score,
        "grouping_score": grouping_score,
    }

    return features


def build_snapshot_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature row per image_id.

    Parameters
    ----------
    df : pd.DataFrame
        Detection-level dataframe.

    Returns
    -------
    pd.DataFrame
        Snapshot-level feature table.
    """
    _validate_snapshot_columns(df)

    rows: list[dict[str, Any]] = []

    for image_id, snapshot_df in df.groupby("image_id", sort=True):
        feature_row = compute_snapshot_features(snapshot_df)
        feature_row["image_id"] = image_id

        if "image_width" in snapshot_df.columns:
            widths = snapshot_df["image_width"].dropna()
            feature_row["image_width"] = int(widths.iloc[0]) if not widths.empty else np.nan

        if "image_height" in snapshot_df.columns:
            heights = snapshot_df["image_height"].dropna()
            feature_row["image_height"] = int(heights.iloc[0]) if not heights.empty else np.nan

        if "label_file" in snapshot_df.columns:
            labels = snapshot_df["label_file"].dropna()
            feature_row["label_file"] = labels.iloc[0] if not labels.empty else None

        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        return feature_df

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0.0)

    return feature_df


def scale_feature_table(
    feature_df: pd.DataFrame,
    exclude_cols: tuple[str, ...] = ("image_id", "label_file"),
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric columns for clustering.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Snapshot-level feature table.
    exclude_cols : tuple[str, ...]
        Columns to exclude from scaling.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled dataframe and fitted scaler.
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty; cannot scale empty feature table")

    scaled_df = feature_df.copy()

    numeric_cols = [
        col
        for col in scaled_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(scaled_df[col])
    ]

    if not numeric_cols:
        raise ValueError("No numeric columns available for scaling")

    scaler = StandardScaler()
    scaled_df[numeric_cols] = scaler.fit_transform(scaled_df[numeric_cols])

    return scaled_df, scaler


def feature_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive summary statistics for numeric feature columns.
    """
    if feature_df.empty:
        return pd.DataFrame()

    numeric_df = feature_df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T
    summary["missing_count"] = numeric_df.isna().sum()
    return summary.sort_index()