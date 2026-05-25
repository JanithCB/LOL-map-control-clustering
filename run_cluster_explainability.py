# Execution command: python run_cluster_explainability.py
import json
import logging
from pathlib import Path
import pandas as pd

from src.clustering.cluster_explainability import (
    compute_cluster_explainability,
    export_explainability
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    methods = ["kmeans", "gmm"]
    all_df_rows = []
    all_json_export = {}
    
    for method in methods:
        input_dir = Path(f"outputs/reports/{method}")
        features_path = input_dir / "clustered_features.csv"
        centroids_path = input_dir / "cluster_centroids.csv"
        
        if not features_path.exists() or not centroids_path.exists():
            logging.warning(f"Required files not found for {method}. Skipping.")
            continue
            
        logging.info(f"Processing explainability for {method}...")
        df = pd.read_csv(features_path)
        centroids = pd.read_csv(centroids_path)
        
        if "cluster_label" not in df.columns:
            logging.warning(f"No cluster_label in {features_path}. Skipping.")
            continue
            
        labels = df["cluster_label"]
        drop_cols = ["cluster_label", "matchId", "summonerName", "participantId", "teamId", "win", "role", "lane"]
        features = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        reports = compute_cluster_explainability(features, labels, centroids)
        df_export, json_export = export_explainability(reports)
        
        df_export.insert(0, "algorithm", method)
        all_df_rows.append(df_export)
        all_json_export[method] = json_export
        
    if not all_df_rows:
        logging.error("No valid runs to explain.")
        return
        
    final_df = pd.concat(all_df_rows, ignore_index=True)
    
    out_dir = Path("outputs/reports/clustering")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "cluster_explainability.csv"
    json_path = out_dir / "cluster_explainability.json"
    
    final_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_json_export, f, indent=4)
        
    logging.info(f"Saved explainability reports to {csv_path} and {json_path}")

if __name__ == "__main__":
    main()
