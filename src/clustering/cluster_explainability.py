import pandas as pd
import numpy as np

def compute_cluster_explainability(
    features: pd.DataFrame, 
    labels: pd.Series, 
    centroids: pd.DataFrame
) -> list[dict]:
    """
    Computes explainability metrics for each cluster.
    """
    explainability_reports = []
    
    global_mean = features.mean()
    total_samples = len(features)
    
    unique_labels = sorted(labels.unique())
    
    for cluster_id in unique_labels:
        cluster_mask = labels == cluster_id
        cluster_features = features[cluster_mask]
        
        cluster_size = int(len(cluster_features))
        cluster_pct = float((cluster_size / total_samples) * 100) if total_samples > 0 else 0.0
        
        if 'cluster_label' in centroids.columns:
            centroid_row = centroids[centroids['cluster_label'] == cluster_id].drop(columns=['cluster_label'])
        else:
            centroid_row = centroids.iloc[cluster_id:cluster_id+1]
            
        if centroid_row.empty:
            centroid_values = cluster_features.mean()
        else:
            centroid_values = centroid_row.iloc[0]
            
        deviations = centroid_values - global_mean
        abs_deviations = deviations.abs()
        top_distinguishing = abs_deviations.sort_values(ascending=False).head(5)
        
        pos_deviations = deviations[deviations > 0]
        top_positive = pos_deviations.sort_values(ascending=False).head(5)
        
        spread = float(cluster_features.var().mean())
        if pd.isna(spread):
            spread = 0.0
            
        report = {
            "cluster_label": int(cluster_id),
            "size": cluster_size,
            "percentage": cluster_pct,
            "spread": spread,
            "top_distinguishing_features": top_distinguishing.to_dict(),
            "top_positive_deviations": top_positive.to_dict()
        }
        explainability_reports.append(report)
        
    return explainability_reports

def generate_plain_language_summary(report: dict) -> str:
    """
    Generates a plain-language summary string from a cluster's explainability report.
    """
    cluster_id = report['cluster_label']
    pct = report['percentage']
    size = report['size']
    
    summary = f"Cluster {cluster_id} represents {pct:.1f}% of the data ({size} samples). "
    
    pos_dev = report.get('top_positive_deviations', {})
    if pos_dev:
        features_str = ", ".join([f"{k} (+{v:.2f})" for k, v in pos_dev.items()])
        summary += f"It is characterized by abnormally high values in: {features_str}. "
        
    disting = report.get('top_distinguishing_features', {})
    if disting:
        features_str = ", ".join([f"{k}" for k in disting.keys()])
        summary += f"The most distinguishing features overall compared to the global average are: {features_str}. "
        
    spread = report.get('spread', 0)
    summary += f"The within-cluster feature variance is {spread:.2f}."
    
    return summary

def export_explainability(reports: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns a DataFrame suitable for CSV export and a list of dictionaries for JSON export.
    """
    df_rows = []
    export_reports = []
    
    for r in reports:
        r_json = {
            "cluster_label": r["cluster_label"],
            "size": r["size"],
            "percentage": r["percentage"],
            "spread": r["spread"],
            "top_distinguishing_features": {k: float(v) for k, v in r["top_distinguishing_features"].items()},
            "top_positive_deviations": {k: float(v) for k, v in r["top_positive_deviations"].items()},
            "summary": generate_plain_language_summary(r)
        }
        export_reports.append(r_json)
        
        row = {
            "cluster_label": r["cluster_label"],
            "size": r["size"],
            "percentage": r["percentage"],
            "spread": r["spread"],
            "top_positive": str(list(r["top_positive_deviations"].keys())),
            "top_distinguishing": str(list(r["top_distinguishing_features"].keys())),
            "summary": r_json["summary"]
        }
        df_rows.append(row)
        
    return pd.DataFrame(df_rows), export_reports
