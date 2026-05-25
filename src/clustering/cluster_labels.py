# Path: src/clustering/cluster_labels.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_LABELS_CSV = Path("outputs/reports/clustering/cluster_labels.csv")

CLUSTER_LABELS = {
    0: "River & Mid Roaming",
    1: "Blue Base Grouping",
    2: "Red Bot Lane Siege",
    3: "Bot Lane Skirmishing",
    4: "Red River Control",
}

CLUSTER_DESCRIPTIONS = {
    0: "Players are positioned around the mid lane and river, focusing on roaming and contesting neutral objectives.",
    1: "Players are grouped inside the blue team's base, likely defending a siege or respawning.",
    2: "Players are concentrated on the red side of the bot lane, pushing deep towers or setting up map pressure.",
    3: "Players are focused in the bot lane and lower map areas, typical for laning phase presence or early skirmishes.",
    4: "Players are occupying the river towards the red side, likely establishing vision control or setting up for neutral objectives.",
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
