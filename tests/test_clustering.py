# tests/test_clustering.py

from __future__ import annotations

import pandas as pd
import pytest

from src.clustering.cluster_analysis import (
    cluster_size_summary,
    representative_samples,
    top_features_per_cluster,
)
from src.clustering.cluster_snapshots import (
    fit_kmeans_models,
    get_feature_columns,
    prepare_feature_matrix,
    run_kmeans,
    select_best_k,
)


def _make_synthetic_feature_df() -> pd.DataFrame:
    """
    Create a small clearly clusterable synthetic feature table.
    """
    rows = []

    cluster_a = [
        ("img_a1", "a1.txt", 0.10, 0.12, 0.15, 0.10, 0.20),
        ("img_a2", "a2.txt", 0.12, 0.10, 0.18, 0.12, 0.22),
        ("img_a3", "a3.txt", 0.11, 0.13, 0.17, 0.11, 0.19),
    ]

    cluster_b = [
        ("img_b1", "b1.txt", 0.85, 0.88, 0.75, 0.80, 0.78),
        ("img_b2", "b2.txt", 0.82, 0.86, 0.78, 0.82, 0.81),
        ("img_b3", "b3.txt", 0.87, 0.84, 0.76, 0.79, 0.80),
    ]

    for image_id, label_file, mean_x, mean_y, spread_score, avg_dist, grouping in cluster_a + cluster_b:
        rows.append(
            {
                "image_id": image_id,
                "label_file": label_file,
                "mean_x": mean_x,
                "mean_y": mean_y,
                "spread_score": spread_score,
                "avg_pairwise_distance": avg_dist,
                "grouping_score": grouping,
                "n_detections": 5,
            }
        )

    return pd.DataFrame(rows)


def _make_config() -> dict:
    return {
        "random_state": 42,
        "feature_scaling": {"enabled": True, "method": "standard"},
        "clustering": {
            "default_method": "kmeans",
            "candidate_k": [2, 3],
            "kmeans": {"n_init": 10, "max_iter": 200},
            "gmm": {"covariance_type": "full", "n_init": 3},
            "hdbscan": {"min_cluster_size": 2, "min_samples": 1},
        },
        "evaluation": {
            "use_silhouette": True,
            "use_davies_bouldin": True,
            "use_calinski_harabasz": True,
        },
    }


def test_get_feature_columns_returns_numeric_non_excluded_columns() -> None:
    df = _make_synthetic_feature_df()

    feature_cols = get_feature_columns(df)

    assert "image_id" not in feature_cols
    assert "label_file" not in feature_cols
    assert "mean_x" in feature_cols
    assert "spread_score" in feature_cols


def test_prepare_feature_matrix_returns_expected_shape() -> None:
    df = _make_synthetic_feature_df()
    feature_cols = get_feature_columns(df)

    X, scaler = prepare_feature_matrix(df, feature_cols, scale=True)

    assert X.shape == (len(df), len(feature_cols))
    assert scaler is not None


def test_fit_kmeans_models_returns_metrics_for_each_candidate_k() -> None:
    df = _make_synthetic_feature_df()
    feature_cols = get_feature_columns(df)
    X, _ = prepare_feature_matrix(df, feature_cols, scale=True)

    metrics_df = fit_kmeans_models(X, candidate_k=[2, 3], random_state=42)

    assert len(metrics_df) == 2
    assert {"k", "inertia", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"}.issubset(
        metrics_df.columns
    )


def test_select_best_k_returns_valid_tested_k() -> None:
    df = _make_synthetic_feature_df()
    feature_cols = get_feature_columns(df)
    X, _ = prepare_feature_matrix(df, feature_cols, scale=True)
    metrics_df = fit_kmeans_models(X, candidate_k=[2, 3], random_state=42)

    best_k = select_best_k(metrics_df)

    assert best_k in {2, 3}


def test_run_kmeans_returns_expected_result_structure() -> None:
    df = _make_synthetic_feature_df()
    config = _make_config()

    result = run_kmeans(df, config)

    assert "clustered_df" in result
    assert "metrics_df" in result
    assert "centroids_df" in result
    assert "selected_k" in result
    assert "cluster_label" in result["clustered_df"].columns
    assert "cluster_label" in result["centroids_df"].columns
    assert result["selected_k"] in {2, 3}


def test_cluster_size_summary_proportions_sum_to_one() -> None:
    df = _make_synthetic_feature_df()
    config = _make_config()
    result = run_kmeans(df, config)

    size_df = cluster_size_summary(result["clustered_df"])

    assert "proportion" in size_df.columns
    assert size_df["proportion"].sum() == pytest.approx(1.0)


def test_top_features_per_cluster_returns_rows_for_each_cluster() -> None:
    df = _make_synthetic_feature_df()
    config = _make_config()
    result = run_kmeans(df, config)

    top_df = top_features_per_cluster(result["centroids_df"], top_n=3)

    assert not top_df.empty
    assert {"cluster_label", "rank", "feature", "centroid_value"}.issubset(top_df.columns)
    assert top_df["cluster_label"].nunique() >= 1


def test_representative_samples_returns_at_least_one_sample_per_cluster() -> None:
    df = _make_synthetic_feature_df()
    config = _make_config()
    result = run_kmeans(df, config)

    samples_df = representative_samples(
        clustered_df=result["clustered_df"],
        centroids_df=result["centroids_df"],
        feature_cols=result["feature_cols"],
        top_n=2,
    )

    assert not samples_df.empty
    assert {"cluster_label", "image_id", "rank_in_cluster", "distance_to_centroid"}.issubset(samples_df.columns)
    assert samples_df["cluster_label"].nunique() == result["clustered_df"]["cluster_label"].nunique()