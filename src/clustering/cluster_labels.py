# Path: src/clustering/cluster_labels.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_LABELS_CSV = Path("outputs/reports/clustering/cluster_labels.csv")

CLUSTER_LABELS: Dict[int, str] = {}
CLUSTER_DESCRIPTIONS: Dict[int, str] = {}


def load_cluster_labels(
    csv_path: str | Path = DEFAULT_LABELS_CSV,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load cluster labels and descriptions from a CSV file.

    Expected CSV columns:
    - cluster_id
    - label
    - short_description

    If the CSV does not exist, returns the module defaults.
    """
    path = Path(csv_path)

    if not path.exists():
        return CLUSTER_LABELS, CLUSTER_DESCRIPTIONS

    labels: Dict[int, str] = {}
    descriptions: Dict[int, str] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"cluster_id", "label", "short_description"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Missing required columns in {path}: {missing}"
            )

        for row in reader:
            cluster_id_raw = row.get("cluster_id", "").strip()
            if not cluster_id_raw:
                continue

            cluster_id = int(cluster_id_raw)
            labels[cluster_id] = row.get("label", "").strip()
            descriptions[cluster_id] = row.get("short_description", "").strip()

    return labels, descriptions


def get_cluster_label(
    cluster_id: int,
    csv_path: str | Path = DEFAULT_LABELS_CSV,
) -> str:
    """Return the human-readable label for a cluster, or a fallback name."""
    labels, _ = load_cluster_labels(csv_path)
    return labels.get(cluster_id, f"Cluster {cluster_id}")


def get_cluster_description(
    cluster_id: int,
    csv_path: str | Path = DEFAULT_LABELS_CSV,
) -> str:
    """Return the human-readable description for a cluster, or an empty string."""
    _, descriptions = load_cluster_labels(csv_path)
    return descriptions.get(cluster_id, "")