
from __future__ import annotations

import pandas as pd

from src.clustering.interpret_clusters import (
    detect_size_column,
    export_labels_csv,
    extract_top_features,
    infer_label,
    normalize_cluster_column,
)


def test_normalize_cluster_column_accepts_cluster_id() -> None:
    df = pd.DataFrame({"cluster_id": [0, 1], "value": [10, 20]})
    result = normalize_cluster_column(df)
    assert "cluster_id" in result.columns
    assert list(result["cluster_id"]) == [0, 1]


def test_normalize_cluster_column_accepts_cluster() -> None:
    df = pd.DataFrame({"cluster": [2, 3], "value": [5, 6]})
    result = normalize_cluster_column(df)
    assert "cluster_id" in result.columns
    assert "cluster" not in result.columns
    assert list(result["cluster_id"]) == [2, 3]


def test_normalize_cluster_column_accepts_cluster_idx() -> None:
    df = pd.DataFrame({"cluster_idx": [4, 5], "value": [7, 8]})
    result = normalize_cluster_column(df)
    assert "cluster_id" in result.columns
    assert "cluster_idx" not in result.columns
    assert list(result["cluster_id"]) == [4, 5]


def test_detect_size_column_handles_size() -> None:
    df = pd.DataFrame({"cluster_id": [0, 1], "size": [100, 50]})
    assert detect_size_column(df) == "size"


def test_detect_size_column_handles_count() -> None:
    df = pd.DataFrame({"cluster_id": [0, 1], "count": [100, 50]})
    assert detect_size_column(df) == "count"


def test_detect_size_column_handles_n() -> None:
    df = pd.DataFrame({"cluster_id": [0, 1], "n": [100, 50]})
    assert detect_size_column(df) == "n"


def test_extract_top_features_long_format() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1],
            "feature": [
                "bot_lane_count",
                "dragon_pit_heat",
                "river_count",
                "mid_lane_count",
                "grouped_mid",
            ],
            "importance": [0.85, 0.92, 0.50, 0.77, 0.88],
        }
    )

    result = extract_top_features(df, top_n=2)

    assert 0 in result
    assert 1 in result
    assert result[0][0][0] == "dragon_pit_heat"
    assert result[0][1][0] == "bot_lane_count"
    assert result[1][0][0] == "grouped_mid"


def test_extract_top_features_wide_format() -> None:
    df = pd.DataFrame(
        {
            "cluster_id": [0, 1],
            "bot_lane_count": [0.90, 0.10],
            "dragon_pit_heat": [0.80, 0.05],
            "mid_lane_count": [0.20, 0.95],
            "grouped_mid": [0.15, 0.88],
        }
    )

    result = extract_top_features(df, top_n=2)

    assert result[0][0][0] == "bot_lane_count"
    assert result[0][1][0] == "dragon_pit_heat"
    assert result[1][0][0] == "mid_lane_count"
    assert result[1][1][0] == "grouped_mid"


def test_infer_label_bot_side_pattern() -> None:
    label, description = infer_label(
        feature_names=["bot_lane_count", "dragon_pit_heat", "river_count"],
        note_text="high bot river presence around dragon",
        size_pct=12.5,
    )
    assert label == "Bot-side objective setup"
    assert "bot-side" in description.lower() or "dragon" in description.lower()


def test_infer_label_top_side_pattern() -> None:
    label, description = infer_label(
        feature_names=["top_lane_count", "herald_pit_heat", "top_river_count"],
        note_text="herald setup on top side",
        size_pct=8.0,
    )
    assert label == "Top-side objective setup"
    assert "top-side" in description.lower() or "herald" in description.lower()


def test_infer_label_mid_group_pattern() -> None:
    label, description = infer_label(
        feature_names=["mid_lane_count", "mid_group_density"],
        note_text="grouped mid pressure with five players around mid tower",
        size_pct=9.0,
    )
    assert label == "5-man mid group"
    assert "mid" in description.lower()


def test_infer_label_invade_pattern() -> None:
    label, description = infer_label(
        feature_names=["enemy_jungle_count", "jungle_depth_score"],
        note_text="deep invade into enemy jungle entrances",
        size_pct=6.0,
    )
    assert label == "Deep invade"
    assert "invade" in description.lower() or "jungle" in description.lower()


def test_export_labels_csv_prefers_manual_values(tmp_path) -> None:
    interpretation_df = pd.DataFrame(
        [
            {
                "cluster_id": 0,
                "draft_label": "Standard laning",
                "draft_description": "Draft description",
                "manual_label": "Reviewed standard laning",
                "manual_description": "Reviewed description",
                "size": 100,
                "pct": 55.0,
                "review_status": "reviewed",
            },
            {
                "cluster_id": 1,
                "draft_label": "Bot-side objective setup",
                "draft_description": "Draft bot description",
                "manual_label": "",
                "manual_description": "",
                "size": 50,
                "pct": 27.5,
                "review_status": "needs_review",
            },
        ]
    )

    output_path = tmp_path / "cluster_labels.csv"
    exported = export_labels_csv(interpretation_df, output_path)

    assert output_path.exists()
    assert list(exported["label"]) == [
        "Reviewed standard laning",
        "Bot-side objective setup",
    ]
    assert list(exported["short_description"]) == [
        "Reviewed description",
        "Draft bot description",
    ]
    assert list(exported["cluster_id"]) == [0, 1]