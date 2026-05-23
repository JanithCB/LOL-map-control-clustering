# Main pipeline entry point
#
# Runs the full end-to-end pipeline:
#   1. Parse YOLO labels → champion positions
#   2. Define map zones
#   3. Compute spatial features per snapshot
#   4. Cluster snapshots (k-means / GMM / HDBSCAN)
#   5. (Optional) Aggregate to game-level playstyles
#   6. Extract macro stats from ranked games
#   7. Compare clusters to macro patterns
#   8. Save outputs (models, figures, reports)
#
# Usage:
#   python run_pipeline.py
#   python run_pipeline.py --config configs/clustering_config.yaml
