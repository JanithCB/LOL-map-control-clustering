"""
Module to validate the integrity of key pipeline outputs.
"""
import logging
from pathlib import Path
import pandas as pd

LOGGER = logging.getLogger(__name__)

def check_file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()

def check_df_not_empty(df: pd.DataFrame) -> bool:
    return not df.empty

def check_required_columns(df: pd.DataFrame, required_cols: list) -> list:
    missing = [col for col in required_cols if col not in df.columns]
    return missing

def validate_pipeline_outputs(output_root: str = ".") -> dict:
    """
    Validates processed features, cluster assignments, representative_samples.csv,
    projection outputs, and macro summaries.
    
    Returns a dictionary report.
    """
    root = Path(output_root)
    report = {
        "status": "pass",
        "checks": []
    }
    
    def add_check(name: str, status: str, details: str = ""):
        report["checks"].append({"name": name, "status": status, "details": details})
        if status == "fail":
            report["status"] = "fail"
        elif status == "warn" and report["status"] == "pass":
            report["status"] = "warn"

    # Check processed features
    snapshot_features_path = root / "data" / "processed" / "snapshot_features.csv"
    if check_file_exists(snapshot_features_path):
        try:
            df = pd.read_csv(snapshot_features_path)
            if check_df_not_empty(df):
                add_check("snapshot_features_exists_and_not_empty", "pass")
                missing = check_required_columns(df, ["image_id"]) # Example required column
                if missing:
                    add_check("snapshot_features_columns", "warn", f"Missing columns: {missing}")
                else:
                    add_check("snapshot_features_columns", "pass")
            else:
                add_check("snapshot_features_exists_and_not_empty", "fail", "File is empty.")
        except Exception as e:
            add_check("snapshot_features_exists_and_not_empty", "fail", str(e))
    else:
        add_check("snapshot_features_exists", "fail", f"Missing {snapshot_features_path}")

    # Check cluster assignments & representative samples
    clustering_dir = root / "outputs" / "reports" / "clustering"
    clustered_data_path = clustering_dir / "clustered_features.csv"
    rep_samples_path = clustering_dir / "representative_samples.csv"
    
    if check_file_exists(clustered_data_path):
        try:
            df = pd.read_csv(clustered_data_path)
            if check_df_not_empty(df) and "cluster_label" in df.columns:
                 add_check("clustered_data_valid", "pass")
            else:
                 add_check("clustered_data_valid", "fail", "Empty or missing 'cluster_label' column.")
        except Exception as e:
             add_check("clustered_data_valid", "fail", str(e))
    else:
        add_check("clustered_data_valid", "warn", f"Missing {clustered_data_path} (clustering may not have run).")

    if check_file_exists(rep_samples_path):
        try:
            df = pd.read_csv(rep_samples_path)
            if check_df_not_empty(df) and "image_id" in df.columns:
                 add_check("representative_samples_valid", "pass")
            else:
                 add_check("representative_samples_valid", "fail", "Empty or missing 'image_id' column.")
        except Exception as e:
             add_check("representative_samples_valid", "fail", str(e))
    else:
        add_check("representative_samples_valid", "warn", f"Missing {rep_samples_path}")

    # Check macro summaries
    macro_dir = root / "outputs" / "reports" / "macro"
    macro_summary_path = macro_dir / "macro_feature_summary.csv"
    if check_file_exists(macro_summary_path):
        add_check("macro_summary_exists", "pass")
    else:
        add_check("macro_summary_exists", "warn", f"Missing {macro_summary_path} (macro analysis may not have run).")
        
    # Check Projection Outputs
    proj_tsne_path = clustering_dir / "cluster_projection_tsne.csv"
    proj_umap_path = clustering_dir / "cluster_projection_umap.csv"
    if check_file_exists(proj_tsne_path) or check_file_exists(proj_umap_path):
        add_check("projection_output_exists", "pass")
    else:
        add_check("projection_output_exists", "warn", f"Missing projection output in {clustering_dir}")

    # Example Image check
    plots_dir = root / "outputs" / "figures" / "clustering"
    if plots_dir.exists() and any(plots_dir.iterdir()):
        add_check("clustering_plots_exist", "pass")
    else:
        add_check("clustering_plots_exist", "warn", "No plots found in outputs/figures/clustering.")

    return report
