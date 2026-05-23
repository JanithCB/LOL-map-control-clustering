# run_pipeline.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.parse_yolo_labels import (
    dataset_summary,
    parse_dataset,
    save_parsed_labels,
)
from src.features.map_zones import label_points
from src.features.spatial_features import (
    build_snapshot_feature_table,
    feature_summary,
)

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for the MVP pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run MVP pipeline: parse YOLO labels, label map zones, and build spatial features."
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/raw/mid_dataset/labels",
        help="Directory containing YOLO label .txt files.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/mid_dataset/images",
        help="Directory containing minimap image files.",
    )
    parser.add_argument(
        "--names-path",
        type=str,
        default="data/raw/mid.names",
        help="Path to YOLO .names file.",
    )
    parser.add_argument(
        "--zones-config",
        type=str,
        default="configs/map_zones.yaml",
        help="Path to zone configuration YAML.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Project root where data/ and outputs/ directories live.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    """
    Create output directories and return output file paths.
    """
    interim_dir = output_root / "data" / "interim"
    processed_dir = output_root / "data" / "processed"
    reports_dir = output_root / "outputs" / "reports"

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "parsed_labels": interim_dir / "parsed_labels.csv",
        "labeled_points": interim_dir / "labeled_points.csv",
        "snapshot_features": processed_dir / "snapshot_features.csv",
        "feature_summary": reports_dir / "feature_summary.csv",
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_root = Path(args.output_root)
    outputs = ensure_output_dirs(output_root)

    LOGGER.info("Step 1/3: Parsing YOLO labels")
    parsed_df = parse_dataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        names_path=args.names_path,
    )
    save_parsed_labels(parsed_df, outputs["parsed_labels"])

    LOGGER.info("Step 2/3: Assigning map zones to detections")
    labeled_df = label_points(parsed_df, config_path=args.zones_config)
    labeled_df.to_csv(outputs["labeled_points"], index=False)

    LOGGER.info("Step 3/3: Building snapshot-level spatial features")
    feature_df = build_snapshot_feature_table(labeled_df)
    feature_df.to_csv(outputs["snapshot_features"], index=False)

    LOGGER.info("Generating feature summary report")
    summary_df = feature_summary(feature_df)
    summary_df.to_csv(outputs["feature_summary"], index=True)

    parsed_summary = dataset_summary(parsed_df)
    n_detections = parsed_summary["n_rows"]
    n_images = parsed_summary["n_images"]
    n_snapshot_rows = len(feature_df)

    print("\nMVP pipeline completed successfully.")
    print(f"Number of detections: {n_detections}")
    print(f"Number of images: {n_images}")
    print(f"Number of snapshot feature rows: {n_snapshot_rows}")
    print("\nSaved artifacts:")
    print(f"- Parsed labels: {outputs['parsed_labels']}")
    print(f"- Labeled points: {outputs['labeled_points']}")
    print(f"- Snapshot features: {outputs['snapshot_features']}")
    print(f"- Feature summary: {outputs['feature_summary']}")


if __name__ == "__main__":
    main()