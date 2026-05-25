# run_multi_clustering.py
import argparse
import logging
from pathlib import Path
import pandas as pd

from src.clustering.run_multi_clustering import run_and_save_algorithm
from src.clustering.cluster_snapshots import load_clustering_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run multiple clustering algorithms.")
    parser.add_argument("--input", type=str, default="outputs/reports/clustering/clustered_features.csv",
                        help="Path to clustered_features.csv or snapshot_features.csv to reuse dataset")
    parser.add_argument("--config", type=str, default="configs/clustering_config.yaml",
                        help="Path to config")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return
        
    logging.info(f"Loading feature dataset from {input_path}")
    df = pd.read_csv(input_path)
    
    # We strip out existing labels to just use the pure features
    if "cluster_label" in df.columns:
        df = df.drop(columns=["cluster_label"])
        
    config = load_clustering_config(args.config)
    
    methods = ["kmeans", "gmm"]
    for method in methods:
        output_dir = Path(f"outputs/reports/{method}")
        run_and_save_algorithm(df, config, method, output_dir)
        
    logging.info("Multi-algorithm clustering complete.")
    logging.info("Outputs saved to:")
    logging.info("  - outputs/reports/kmeans/")
    logging.info("  - outputs/reports/gmm/")

if __name__ == "__main__":
    main()
