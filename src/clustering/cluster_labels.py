# Path: src/clustering/cluster_labels.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_LABELS_CSV = Path("outputs/reports/clustering/cluster_labels.csv")

CLUSTER_LABELS = {
    0: "Cluster pattern review",
    1: "Cluster pattern review",
    2: "Cluster pattern review",
    3: "Cluster pattern review",
    4: "Cluster pattern review",
}

CLUSTER_DESCRIPTIONS = {
    0: "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
    1: "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
    2: "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
    3: "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
    4: "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
}


def load_cluster_labels(csv_path: str | Path = DEFAULT_LABELS_CSV) -> Tuple[Dict[int, str], Dict[int, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return CLUSTER_LABELS, CLUSTER_DESCRIPTIONS

    labels: Dict[int, str] = {}
    descriptions: Dict[int, str] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cluster_id = int(row["cluster_id"])
            labels[cluster_id] = row["label"]
            descriptions[cluster_id] = row["short_description"]

    return labels, descriptions


def get_cluster_label(cluster_id: int, csv_path: str | Path = DEFAULT_LABELS_CSV) -> str:
    labels, _ = load_cluster_labels(csv_path)
    return labels.get(cluster_id, f"Cluster {cluster_id}")


def get_cluster_description(cluster_id: int, csv_path: str | Path = DEFAULT_LABELS_CSV) -> str:
    _, descriptions = load_cluster_labels(csv_path)
    return descriptions.get(cluster_id, "")
