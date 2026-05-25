# src/clustering/cluster_projection.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

def compute_2d_projection(
    df: pd.DataFrame, 
    method: str = "umap", 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Computes a 2D cluster projection from the minimap feature matrix using UMAP or t-SNE.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")
        
    method = method.lower()
    
    # Identify feature columns (numeric) and metadata columns
    # We exclude standard metadata and cluster assignments.
    meta_cols = ["image_id", "image_width", "image_height", "label_file", "cluster_label"]
    feature_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    if not feature_cols:
        raise ValueError("No numeric feature columns found for projection.")
        
    X = df[feature_cols].to_numpy()
    
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
        except ImportError:
            LOGGER.warning("UMAP not installed. Falling back to t-SNE.")
            method = "tsne"
            
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        
    LOGGER.info(f"Computing 2D projection using {method.upper()} on {X.shape[0]} samples...")
    X_proj = reducer.fit_transform(X)
    
    # Construct output dataframe
    out_df = pd.DataFrame({
        "proj_x": X_proj[:, 0],
        "proj_y": X_proj[:, 1],
        "method": method.upper()
    })
    
    # Preserve cluster assignments and metadata
    for col in meta_cols:
        if col in df.columns:
            out_df[col] = df[col].values
            
    return out_df

def save_cluster_projection(
    proj_df: pd.DataFrame, 
    output_dir: str | Path,
    method: str
) -> Path:
    """
    Saves the computed projection CSV to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method_name = method.lower()
    file_path = output_dir / f"cluster_projection_{method_name}.csv"
    
    proj_df.to_csv(file_path, index=False)
    LOGGER.info(f"Saved {method_name.upper()} projection to {file_path}")
    
    return file_path
