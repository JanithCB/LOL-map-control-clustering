# run_projection.py
import argparse
import logging
from pathlib import Path
import pandas as pd

from src.clustering.cluster_projection import compute_2d_projection, save_cluster_projection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Compute and save 2D projection of minimap clusters.")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"], 
                        help="Projection method to use (default: umap)")
    parser.add_argument("--input", type=str, default="outputs/reports/clustering/clustered_features.csv",
                        help="Path to the processed clustered features CSV")
    parser.add_argument("--output-dir", type=str, default="outputs/reports/clustering",
                        help="Directory to save the projection CSV")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return
        
    logging.info(f"Loading input features from {input_path}...")
    df = pd.read_csv(input_path)
    
    if "cluster_label" not in df.columns:
        logging.error("Input data must contain 'cluster_label' column.")
        return
        
    proj_df = compute_2d_projection(df, method=args.method)
    
    actual_method = proj_df["method"].iloc[0]
    logging.info(f"Projection completed using {actual_method}.")
    
    out_path = save_cluster_projection(proj_df, args.output_dir, method=actual_method)
    logging.info(f"File saved successfully: {out_path}")

    # Generate the visualization
    from src.visualization.plot_clusters import plot_2d_projection
    plot_output = Path("outputs/figures/cluster_projection.png")
    plot_2d_projection(proj_df, plot_output)
    logging.info(f"Generated 2D scatter plot at {plot_output}")

if __name__ == "__main__":
    main()
