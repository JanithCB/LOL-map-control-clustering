from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.parse_yolo_labels import parse_dataset, save_parsed_labels
from src.features.map_zones import label_points
from src.features.spatial_features import (
    build_snapshot_feature_table,
    feature_summary,
)

LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the minimap feature engineering pipeline."
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing YOLO label files.",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing image files.",
    )
    parser.add_argument(
        "--names-path",
        required=True,
        help="Path to class names file.",
    )
    parser.add_argument(
        "--zones-config",
        default="configs/map_zones.yaml",
        help="Path to zone config YAML.",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Project output root.",
    )

    parser.add_argument(
        "--run-clustering",
        action="store_true",
        help="Run clustering after spatial feature generation.",
    )
    parser.add_argument(
        "--clustering-config",
        default="configs/clustering_config.yaml",
        help="Path to clustering config YAML.",
    )
    parser.add_argument(
        "--clustering-output-dir",
        default="outputs/reports/clustering",
        help="Directory for clustering CSV outputs.",
    )
    parser.add_argument(
        "--plots-output-dir",
        default="outputs/figures/clustering",
        help="Directory for clustering plots.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    output_root = Path(args.output_root)
    interim_dir = output_root / "data" / "interim"
    processed_dir = output_root / "data" / "processed"
    reports_dir = output_root / "outputs" / "reports"

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parsed_labels_path = interim_dir / "parsed_labels.csv"
    labeled_points_path = interim_dir / "labeled_points.csv"
    snapshot_features_path = processed_dir / "snapshot_features.csv"
    feature_summary_path = reports_dir / "feature_summary.csv"

    LOGGER.info("Step 1/3: Parsing YOLO labels")
    parsed_df = parse_dataset(
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        names_path=args.names_path,
    )
    save_parsed_labels(parsed_df, parsed_labels_path)

    LOGGER.info("Step 2/3: Assigning map zones to detections")
    labeled_df = label_points(parsed_df, config_path=args.zones_config)
    labeled_df.to_csv(labeled_points_path, index=False)

    LOGGER.info("Step 3/3: Building snapshot-level spatial features")
    snapshot_df = build_snapshot_feature_table(labeled_df)
    snapshot_df.to_csv(snapshot_features_path, index=False)

    LOGGER.info("Generating feature summary report")
    summary_df = feature_summary(snapshot_df)
    summary_df.to_csv(feature_summary_path, index=False)

    print("\nMVP pipeline completed successfully.")
    print(f"Number of detections: {len(parsed_df)}")
    print(
        f"Number of images: "
        f"{parsed_df['image_id'].nunique() if 'image_id' in parsed_df.columns else 'N/A'}"
    )
    print(f"Number of snapshot feature rows: {len(snapshot_df)}")

    print("\nSaved artifacts:")
    print(f"- Parsed labels: {parsed_labels_path}")
    print(f"- Labeled points: {labeled_points_path}")
    print(f"- Snapshot features: {snapshot_features_path}")
    print(f"- Feature summary: {feature_summary_path}")

    if args.run_clustering:
        LOGGER.info("Starting optional clustering stage")

        from src.clustering.cluster_snapshots import (
            load_clustering_config,
            run_kmeans,
            save_clustering_outputs,
        )
        from src.clustering.cluster_analysis import (
            cluster_size_summary,
            top_features_per_cluster,
            representative_samples,
            save_cluster_analysis_tables,
        )
        from src.visualization.plot_clusters import save_default_cluster_plots

        clustering_config = load_clustering_config(args.clustering_config)
        clustering_result = run_kmeans(snapshot_df, clustering_config)

        clustering_paths = save_clustering_outputs(
            clustering_result,
            output_dir=args.clustering_output_dir,
        )

        size_df = cluster_size_summary(clustering_result["clustered_df"])
        top_features_df = top_features_per_cluster(
            clustering_result["centroids_df"],
            top_n=8,
        )
        samples_df = representative_samples(
            clustered_df=clustering_result["clustered_df"],
            centroids_df=clustering_result["centroids_df"],
            feature_cols=clustering_result["feature_cols"],
            top_n=6,
        )

        analysis_paths = save_cluster_analysis_tables(
            size_df=size_df,
            top_features_df=top_features_df,
            samples_df=samples_df,
            output_dir=args.clustering_output_dir,
        )

        plot_paths = save_default_cluster_plots(
            clustered_df=clustering_result["clustered_df"],
            centroids_df=clustering_result["centroids_df"],
            feature_cols=clustering_result["feature_cols"],
            output_dir=args.plots_output_dir,
        )

        print("\nClustering completed successfully.")
        print(f"Selected clustering method: {clustering_result['method']}")
        print(f"Selected k: {clustering_result['selected_k']}")

        print("\nSaved clustering CSV outputs:")
        for key, path in clustering_paths.items():
            print(f"- {key}: {path}")

        print("\nSaved clustering analysis tables:")
        for key, path in analysis_paths.items():
            print(f"- {key}: {path}")

        print("\nSaved clustering plots:")
        for key, path in plot_paths.items():
            print(f"- {key}: {path}")


if __name__ == "__main__":
    main()