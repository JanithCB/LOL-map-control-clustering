# src/clustering/cluster_snapshots.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


def load_clustering_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load clustering YAML configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Clustering config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Clustering config must be a dictionary-like YAML structure.")

    return config


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: tuple[str, ...] = ("image_id", "label_file"),
) -> list[str]:
    """
    Return numeric columns suitable for clustering, excluding metadata columns.
    """
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for clustering.")

    return feature_cols


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    scale: bool = True,
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Extract numeric feature matrix and optionally standardize it.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot prepare feature matrix.")

    X = df[feature_cols].to_numpy(dtype=float)

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    return X, None


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Compute clustering metrics safely for a set of labels.
    """
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
        }

    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
    }


def fit_kmeans_models(
    X: np.ndarray,
    candidate_k: list[int],
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
) -> pd.DataFrame:
    """
    Fit KMeans for each candidate k and compute evaluation metrics.
    """
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to fit clustering models.")

    rows: list[dict[str, Any]] = []

    for k in candidate_k:
        if k < 2:
            LOGGER.warning("Skipping invalid k=%s; k must be >= 2", k)
            continue
        if k >= X.shape[0]:
            LOGGER.warning("Skipping k=%s because it is >= number of samples", k)
            continue

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = model.fit_predict(X)

        metric_row = {
            "method": "kmeans",
            "k": int(k),
            "inertia": float(model.inertia_),
        }
        metric_row.update(_safe_cluster_metrics(X, labels))
        rows.append(metric_row)

    if not rows:
        raise ValueError("No valid k values produced fitted KMeans models.")

    return pd.DataFrame(rows)


def select_best_k(metrics_df: pd.DataFrame, score_col: str = "silhouette_score") -> int:
    """
    Select the best k from a metrics table using the given score column.
    """
    if metrics_df.empty:
        raise ValueError("metrics_df is empty; cannot select best k.")

    if score_col not in metrics_df.columns:
        raise KeyError(f"Score column '{score_col}' not found in metrics_df.")

    valid_df = metrics_df.dropna(subset=[score_col])
    if valid_df.empty:
        raise ValueError(f"No valid rows available for score column '{score_col}'.")

    best_idx = valid_df[score_col].idxmax()
    return int(valid_df.loc[best_idx, "k"])


def _build_centroids_df(
    centers: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Build centroid DataFrame with cluster labels.
    """
    centroids_df = pd.DataFrame(centers, columns=feature_cols)
    centroids_df.insert(0, "cluster_label", np.arange(len(centroids_df), dtype=int))
    return centroids_df


def run_kmeans(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    Run end-to-end KMeans clustering on snapshot features.
    """
    feature_cols = get_feature_columns(df)

    scaling_enabled = bool(config.get("feature_scaling", {}).get("enabled", True))
    X, scaler = prepare_feature_matrix(df, feature_cols, scale=scaling_enabled)

    candidate_k = config["clustering"]["candidate_k"]
    random_state = int(config.get("random_state", 42))
    kmeans_cfg = config["clustering"].get("kmeans", {})

    metrics_df = fit_kmeans_models(
        X=X,
        candidate_k=candidate_k,
        random_state=random_state,
        n_init=int(kmeans_cfg.get("n_init", 20)),
        max_iter=int(kmeans_cfg.get("max_iter", 300)),
    )

    selected_k = select_best_k(metrics_df, score_col="silhouette_score")

    final_model = KMeans(
        n_clusters=selected_k,
        random_state=random_state,
        n_init=int(kmeans_cfg.get("n_init", 20)),
        max_iter=int(kmeans_cfg.get("max_iter", 300)),
    )
    labels = final_model.fit_predict(X)

    clustered_df = df.copy()
    clustered_df["cluster_label"] = labels.astype(int)

    centroids_df = _build_centroids_df(final_model.cluster_centers_, feature_cols)

    return {
        "method": "kmeans",
        "clustered_df": clustered_df,
        "metrics_df": metrics_df,
        "centroids_df": centroids_df,
        "model": final_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "selected_k": selected_k,
    }


def run_gmm(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """
    Optional Gaussian Mixture clustering.
    """
    feature_cols = get_feature_columns(df)

    scaling_enabled = bool(config.get("feature_scaling", {}).get("enabled", True))
    X, scaler = prepare_feature_matrix(df, feature_cols, scale=scaling_enabled)

    candidate_k = config["clustering"]["candidate_k"]
    random_state = int(config.get("random_state", 42))
    gmm_cfg = config["clustering"].get("gmm", {})

    rows: list[dict[str, Any]] = []
    fitted_models: dict[int, GaussianMixture] = {}

    for k in candidate_k:
        if k < 2 or k >= X.shape[0]:
            continue

        model = GaussianMixture(
            n_components=k,
            covariance_type=gmm_cfg.get("covariance_type", "full"),
            n_init=int(gmm_cfg.get("n_init", 5)),
            random_state=random_state,
        )
        labels = model.fit_predict(X)
        metrics = _safe_cluster_metrics(X, labels)

        row = {
            "method": "gmm",
            "k": int(k),
            "inertia": np.nan,
            **metrics,
        }
        rows.append(row)
        fitted_models[k] = model

    metrics_df = pd.DataFrame(rows)
    selected_k = select_best_k(metrics_df, score_col="silhouette_score")
    final_model = fitted_models[selected_k]
    labels = final_model.predict(X)

    clustered_df = df.copy()
    clustered_df["cluster_label"] = labels.astype(int)

    centroids_df = _build_centroids_df(final_model.means_, feature_cols)

    return {
        "method": "gmm",
        "clustered_df": clustered_df,
        "metrics_df": metrics_df,
        "centroids_df": centroids_df,
        "model": final_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "selected_k": selected_k,
    }


def run_hdbscan(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Optional HDBSCAN clustering.
    """
    if hdbscan is None:  # pragma: no cover
        LOGGER.warning("hdbscan is not installed; skipping HDBSCAN run.")
        return None

    feature_cols = get_feature_columns(df)

    scaling_enabled = bool(config.get("feature_scaling", {}).get("enabled", True))
    X, scaler = prepare_feature_matrix(df, feature_cols, scale=scaling_enabled)

    hdbscan_cfg = config["clustering"].get("hdbscan", {})
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(hdbscan_cfg.get("min_cluster_size", 25)),
        min_samples=int(hdbscan_cfg.get("min_samples", 10)),
    )
    labels = model.fit_predict(X)

    clustered_df = df.copy()
    clustered_df["cluster_label"] = labels.astype(int)

    valid_mask = labels != -1
    if valid_mask.sum() >= 2 and len(np.unique(labels[valid_mask])) >= 2:
        metrics = _safe_cluster_metrics(X[valid_mask], labels[valid_mask])
    else:
        metrics = {
            "silhouette_score": np.nan,
            "davies_bouldin_score": np.nan,
            "calinski_harabasz_score": np.nan,
        }

    metrics_df = pd.DataFrame(
        [
            {
                "method": "hdbscan",
                "k": int(len(set(labels)) - (1 if -1 in labels else 0)),
                "inertia": np.nan,
                **metrics,
            }
        ]
    )

    centroids_rows: list[dict[str, Any]] = []
    for cluster_label in sorted(set(labels)):
        if cluster_label == -1:
            continue
        cluster_points = clustered_df.loc[clustered_df["cluster_label"] == cluster_label, feature_cols]
        centroid = cluster_points.mean(axis=0)
        row = {"cluster_label": int(cluster_label), **centroid.to_dict()}
        centroids_rows.append(row)

    centroids_df = pd.DataFrame(centroids_rows)

    return {
        "method": "hdbscan",
        "clustered_df": clustered_df,
        "metrics_df": metrics_df,
        "centroids_df": centroids_df,
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "selected_k": int(len(set(labels)) - (1 if -1 in labels else 0)),
    }


def save_clustering_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """
    Save clustering outputs to CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clustered_path = output_dir / "clustered_features.csv"
    metrics_path = output_dir / "clustering_metrics.csv"
    centroids_path = output_dir / "cluster_centroids.csv"

    result["clustered_df"].to_csv(clustered_path, index=False)
    result["metrics_df"].to_csv(metrics_path, index=False)
    result["centroids_df"].to_csv(centroids_path, index=False)

    return {
        "clustered_features": clustered_path,
        "clustering_metrics": metrics_path,
        "cluster_centroids": centroids_path,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.
    """
    parser = argparse.ArgumentParser(
        description="Cluster minimap snapshot features into map-control patterns."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to snapshot_features.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/clustering_config.yaml",
        help="Path to clustering config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports/clustering",
        help="Directory to save clustering CSV outputs.",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=["kmeans", "gmm", "hdbscan"],
        help="Override default clustering method from config.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    """
    CLI entry point.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    df = pd.read_csv(args.input)
    config = load_clustering_config(args.config)

    method = args.method or config["clustering"].get("default_method", "kmeans")

    if method == "kmeans":
        result = run_kmeans(df, config)
    elif method == "gmm":
        result = run_gmm(df, config)
    elif method == "hdbscan":
        result = run_hdbscan(df, config)
        if result is None:
            raise RuntimeError("HDBSCAN was requested but is not available.")
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    saved_paths = save_clustering_outputs(result, args.output_dir)

    print("Clustering completed successfully.")
    print(f"Selected method: {result['method']}")
    print(f"Selected k: {result['selected_k']}")
    print("Saved outputs:")
    for key, path in saved_paths.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()