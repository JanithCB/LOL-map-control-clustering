import pytest
import pandas as pd
from src.clustering.evaluate_clustering import evaluate_clusters, evaluate_multiple_runs

@pytest.fixture
def sample_data():
    features = pd.DataFrame({
        "f1": [1.0, 1.1, 1.0, 9.0, 9.1, 9.0],
        "f2": [1.0, 1.0, 1.1, 9.0, 9.0, 9.1]
    })
    labels_good = pd.Series([0, 0, 0, 1, 1, 1])
    labels_single = pd.Series([0, 0, 0, 0, 0, 0])
    return features, labels_good, labels_single

def test_evaluate_clusters_valid(sample_data):
    features, labels_good, _ = sample_data
    metrics = evaluate_clusters(features, labels_good, {"algorithm": "kmeans"})
    
    assert metrics["algorithm"] == "kmeans"
    assert metrics["silhouette_score"] is not None
    assert metrics["davies_bouldin_score"] is not None
    assert metrics["calinski_harabasz_score"] is not None
    assert metrics["silhouette_score"] > 0.5  # Well separated clusters

def test_evaluate_clusters_single_cluster(sample_data):
    features, _, labels_single = sample_data
    metrics = evaluate_clusters(features, labels_single)
    
    assert metrics["silhouette_score"] is None
    assert metrics["davies_bouldin_score"] is None
    assert metrics["calinski_harabasz_score"] is None

def test_evaluate_multiple_runs(sample_data):
    features, labels_good, labels_single = sample_data
    runs = [
        {"features": features, "labels": labels_good, "metadata": {"run": 1}},
        {"features": features, "labels": labels_single, "metadata": {"run": 2}}
    ]
    
    df = evaluate_multiple_runs(runs)
    assert len(df) == 2
    assert "silhouette_score" in df.columns
    assert "run" in df.columns
    assert df.loc[0, "silhouette_score"] is not None
    
    # For single cluster, score should be None/NaN
    assert pd.isna(df.loc[1, "silhouette_score"]) or df.loc[1, "silhouette_score"] is None
