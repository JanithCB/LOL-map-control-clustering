# Execution command: python run_clustering_evaluation.py
import logging
from pathlib import Path
import pandas as pd

from src.clustering.evaluate_clustering import evaluate_multiple_runs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    methods = ["kmeans", "gmm"]
    runs_data = []
    
    for method in methods:
        input_dir = Path(f"outputs/reports/{method}")
        features_path = input_dir / "clustered_features.csv"
        
        if not features_path.exists():
            logging.warning(f"Could not find output for {method} at {features_path}")
            continue
            
        logging.info(f"Loading {method} data from {features_path}")
        df = pd.read_csv(features_path)
        
        if "cluster_label" not in df.columns:
            logging.warning(f"No cluster_label column in {features_path}")
            continue
            
        labels = df["cluster_label"]
        features = df.select_dtypes(include=['number']).drop(columns=["cluster_label"], errors='ignore')
        
        runs_data.append({
            "features": features,
            "labels": labels,
            "metadata": {"algorithm": method}
        })
        
    if not runs_data:
        logging.error("No valid clustering runs found to evaluate.")
        return
        
    logging.info("Computing evaluation metrics...")
    results_df = evaluate_multiple_runs(runs_data)
    
    out_dir = Path("outputs/reports/clustering")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clustering_evaluation.csv"
    
    results_df.to_csv(out_path, index=False)
    logging.info(f"Saved evaluation report to {out_path}")
    
    logging.info("\n--- Ranked Summary (by Silhouette Score, higher is better) ---")
    if "silhouette_score" in results_df.columns:
        summary = results_df.sort_values(by="silhouette_score", ascending=False)
        print(summary.to_string(index=False))
    else:
        print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
