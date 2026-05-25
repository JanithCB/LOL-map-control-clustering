import pytest
import pandas as pd
from pathlib import Path
from gui.data_loader import (
    _load_explainability, 
    _load_evaluation, 
    load_app_data,
    AppData
)

def test_load_explainability_missing(tmp_path):
    # Should safely return an empty dict when files are missing
    result = _load_explainability(tmp_path)
    assert result == {}

def test_load_evaluation_missing(tmp_path):
    # Should safely return an empty dict when files are missing
    result = _load_evaluation(tmp_path)
    assert result == {}

def test_load_app_data_empty_dir(tmp_path):
    # Mock the required CSVs so load_app_data doesn't raise FileNotFoundError on base paths
    clust_dir = tmp_path / "outputs" / "reports" / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({"cluster_id": [], "size": [], "pct": []}).to_csv(clust_dir / "cluster_sizes.csv", index=False)
    pd.DataFrame({"cluster_id": [], "label": []}).to_csv(clust_dir / "cluster_labels.csv", index=False)
    pd.DataFrame({"cluster_id": [], "feature": [], "score": []}).to_csv(clust_dir / "cluster_top_features.csv", index=False)
    pd.DataFrame({"cluster_id": [], "image_id": []}).to_csv(clust_dir / "representative_samples.csv", index=False)
    
    comp_dir = tmp_path / "outputs" / "reports" / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cluster_id": [], "note": []}).to_csv(comp_dir / "qualitative_comparison_notes.csv", index=False)
    
    app_data = load_app_data(tmp_path)
    
    assert isinstance(app_data, AppData)
    assert len(app_data.cluster_infos) == 0
    assert app_data.evaluation_metrics == {}

def test_load_evaluation_with_data(tmp_path):
    # Verify that valid data is correctly parsed into a dict
    clust_dir = tmp_path / "outputs" / "reports" / "clustering"
    clust_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame({
        "algorithm": ["kmeans", "gmm"],
        "silhouette_score": [0.5, 0.6],
        "davies_bouldin_score": [1.2, 1.1]
    })
    df.to_csv(clust_dir / "clustering_evaluation.csv", index=False)
    
    eval_data = _load_evaluation(tmp_path)
    assert "kmeans" in eval_data
    assert eval_data["kmeans"]["silhouette_score"] == 0.5
    assert eval_data["gmm"]["davies_bouldin_score"] == 1.1
    # Check that missing columns are safely ignored
    assert "calinski_harabasz_score" not in eval_data["kmeans"]
