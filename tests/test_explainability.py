import pytest
import pandas as pd
from src.clustering.cluster_explainability import compute_cluster_explainability, export_explainability

@pytest.fixture
def sample_data():
    features = pd.DataFrame({
        "feat_a": [1.0, 2.0, 8.0, 9.0],
        "feat_b": [1.0, 1.5, 9.0, 9.5]
    })
    labels = pd.Series([0, 0, 1, 1])
    centroids = pd.DataFrame({
        "cluster_label": [0, 1],
        "feat_a": [1.5, 8.5],
        "feat_b": [1.25, 9.25]
    })
    return features, labels, centroids

def test_compute_cluster_explainability(sample_data):
    features, labels, centroids = sample_data
    reports = compute_cluster_explainability(features, labels, centroids)
    
    assert len(reports) == 2
    
    rep0 = reports[0]
    assert rep0["cluster_label"] == 0
    assert rep0["size"] == 2
    assert rep0["percentage"] == 50.0
    assert "top_distinguishing_features" in rep0
    assert "top_positive_deviations" in rep0
    # Spread should be computed
    assert isinstance(rep0["spread"], float)

def test_export_explainability(sample_data):
    features, labels, centroids = sample_data
    reports = compute_cluster_explainability(features, labels, centroids)
    
    df, json_export = export_explainability(reports)
    
    assert len(df) == 2
    assert "summary" in df.columns
    assert "top_distinguishing" in df.columns
    
    assert len(json_export) == 2
    assert "summary" in json_export[0]

def test_single_sample_cluster():
    features = pd.DataFrame({"feat_a": [1.0, 9.0], "feat_b": [1.0, 9.0]})
    labels = pd.Series([0, 1])
    centroids = pd.DataFrame({"cluster_label": [0, 1], "feat_a": [1.0, 9.0], "feat_b": [1.0, 9.0]})
    
    reports = compute_cluster_explainability(features, labels, centroids)
    assert len(reports) == 2
    # Variance of a single sample is usually NaN, the function should handle it (e.g., set to 0.0)
    assert pd.isna(reports[0]["spread"]) or reports[0]["spread"] == 0.0
