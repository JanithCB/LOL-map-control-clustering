"""
Orchestration script to run the full League map-control clustering pipeline.

PowerShell execution command:
python run_all.py
"""

import yaml
import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_command(cmd: list):
    LOGGER.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        LOGGER.error(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        # Fail gracefully, but stop the pipeline for critical steps
        raise RuntimeError(f"Pipeline step failed: {' '.join(cmd)}")

def main():
    config_path = "configs/pipeline_config.yaml"
    if not Path(config_path).exists():
        LOGGER.error(f"Pipeline config not found at {config_path}")
        sys.exit(1)
        
    config = load_config(config_path)
    
    paths = config.get("paths", {})
    raw_paths = paths.get("raw", {})
    config_paths = paths.get("configs", {})
    out_paths = paths.get("output", {})
    pipeline = config.get("pipeline", {})

    LOGGER.info("Starting End-to-End Pipeline...")

    # 1. Base pipeline (parsing, spatial features, optional clustering & macro)
    base_cmd = [
        sys.executable, "run_pipeline.py",
        "--labels-dir", raw_paths.get("labels_dir", "mid_dataset/labels"),
        "--images-dir", raw_paths.get("images_dir", "mid_dataset/images"),
        "--names-path", raw_paths.get("names_path", "mid.names"),
        "--zones-config", config_paths.get("zones_config", "configs/map_zones.yaml"),
        "--output-root", out_paths.get("root", ".")
    ]
    
    if pipeline.get("run_clustering"):
        base_cmd.extend([
            "--run-clustering",
            "--clustering-config", config_paths.get("clustering_config", "configs/clustering_config.yaml"),
            "--clustering-output-dir", out_paths.get("clustering_dir", "outputs/reports/clustering"),
            "--plots-output-dir", out_paths.get("plots_dir", "outputs/figures/clustering")
        ])
        
    if pipeline.get("run_macro_analysis"):
        base_cmd.extend([
            "--run-macro-analysis",
            "--ranked-games-csv", raw_paths.get("ranked_games_csv", "games.csv"),
            "--macro-config", config_paths.get("macro_config", "configs/macro_config.yaml"),
            "--macro-output-dir", out_paths.get("macro_dir", "outputs/reports/macro"),
            "--comparison-output-dir", out_paths.get("comparison_dir", "outputs/reports/comparison")
        ])

    try:
        run_command(base_cmd)
    except RuntimeError as e:
        LOGGER.error(f"Base pipeline failed: {e}")
        sys.exit(1)

    # 2. Additional optional steps
    if pipeline.get("run_projections"):
        LOGGER.info("Running Projections...")
        proj_cmd = [
            sys.executable, "run_projection.py"
        ]
        # Allow it to fail gracefully if script is not fully updated
        try:
            run_command(proj_cmd)
        except RuntimeError as e:
            LOGGER.warning(f"Projection step encountered an issue or is missing: {e}")

    if pipeline.get("run_explainability"):
        LOGGER.info("Running Explainability & Evaluation...")
        exp_cmd = [sys.executable, "run_cluster_explainability.py"]
        try:
            run_command(exp_cmd)
        except RuntimeError as e:
            LOGGER.warning(f"Explainability step encountered an issue: {e}")

    LOGGER.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
