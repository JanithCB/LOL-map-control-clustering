import pytest
import pandas as pd
import numpy as np
from src.clustering.cluster_projection import compute_2d_projection

@pytest.fixture
def sample_feature_df():
    # TSNE default perplexity is 30, so we need > 30 samples
    n_samples = 35
    data = {
        "image_id": [f"img{i}" for i in range(n_samples)],
        "cluster_label": [i % 3 for i in range(n_samples)],
        "feature_1": np.random.rand(n_samples),
        "feature_2": np.random.rand(n_samples)
    }
    return pd.DataFrame(data)

def test_compute_2d_projection_tsne(sample_feature_df):
    # Use tsne to guarantee it runs without umap dependency
    proj_df = compute_2d_projection(sample_feature_df, method="tsne", random_state=42)
    
    assert len(proj_df) == len(sample_feature_df)
    assert "proj_x" in proj_df.columns
    assert "proj_y" in proj_df.columns
    assert "method" in proj_df.columns
    assert proj_df["method"].iloc[0] == "TSNE"
    assert "image_id" in proj_df.columns
    assert "cluster_label" in proj_df.columns
    assert list(proj_df["image_id"]) == list(sample_feature_df["image_id"])

def test_compute_2d_projection_empty():
    empty_df = pd.DataFrame(columns=["image_id", "feature_1"])
    with pytest.raises(ValueError, match="Input dataframe is empty"):
        compute_2d_projection(empty_df)

def test_compute_2d_projection_no_numeric(sample_feature_df):
    df_no_num = sample_feature_df[["image_id", "cluster_label"]].copy()
    # Assuming cluster_label is considered metadata and skipped for features
    with pytest.raises(ValueError, match="No numeric feature columns found"):
        compute_2d_projection(df_no_num)
