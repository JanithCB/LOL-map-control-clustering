
from __future__ import annotations

import argparse

from src.clustering.interpret_clusters import run_interpretation


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for cluster interpretation."""
    parser = argparse.ArgumentParser(
        description="Run Task 1 cluster interpretation and export label files."
    )
    parser.add_argument(
        "--export-py",
        action="store_true",
        help="Also export src/clustering/cluster_labels.py for GUI integration.",
    )
    return parser


def main() -> None:
    """Run the cluster interpretation workflow from the repo root."""
    parser = build_arg_parser()
    args = parser.parse_args()

    outputs = run_interpretation(export_py=args.export_py)

    print(f"Saved interpretation sheet: {outputs['interpretation_sheet']}")
    print(f"Saved cluster labels CSV: {outputs['labels_csv']}")
    if "python_mapping" in outputs:
        print(f"Saved Python label mapping: {outputs['python_mapping']}")


if __name__ == "__main__":
    main()