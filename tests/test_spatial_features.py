# tests/test_spatial_features.py

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.features.map_zones import (
    assign_primary_zone,
    label_points,
    load_zones,
)
from src.features.spatial_features import (
    build_snapshot_feature_table,
    compute_snapshot_features,
    scale_feature_table,
)


def _write_test_zone_config(path: Path) -> None:
    config = {
        "metadata": {
            "map_name": "Summoner's Rift",
            "coordinate_space": "normalized",
            "version": 1,
        },
        "zones": {
            "top_lane": {"type": "rectangle", "xmin": 0.00, "ymin": 0.00, "xmax": 0.30, "ymax": 0.20},
            "mid_lane": {"type": "rectangle", "xmin": 0.40, "ymin": 0.40, "xmax": 0.60, "ymax": 0.60},
            "bot_lane": {"type": "rectangle", "xmin": 0.70, "ymin": 0.80, "xmax": 1.00, "ymax": 1.00},
            "top_jungle": {"type": "rectangle", "xmin": 0.00, "ymin": 0.20, "xmax": 0.35, "ymax": 0.50},
            "bot_jungle": {"type": "rectangle", "xmin": 0.65, "ymin": 0.50, "xmax": 1.00, "ymax": 0.80},
            "river": {"type": "rectangle", "xmin": 0.30, "ymin": 0.20, "xmax": 0.70, "ymax": 0.70},
            "baron_zone": {"type": "rectangle", "xmin": 0.00, "ymin": 0.00, "xmax": 0.15, "ymax": 0.15},
            "dragon_zone": {"type": "rectangle", "xmin": 0.85, "ymin": 0.85, "xmax": 1.00, "ymax": 1.00},
            "blue_base": {"type": "rectangle", "xmin": 0.00, "ymin": 0.85, "xmax": 0.15, "ymax": 1.00},
            "red_base": {"type": "rectangle", "xmin": 0.85, "ymin": 0.00, "xmax": 1.00, "ymax": 0.15},
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _make_labeled_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label_file": "img1.txt",
                "x_center_norm": 0.10,
                "y_center_norm": 0.10,
                "bbox_area_norm": 0.04,
                "image_width": 100,
                "image_height": 100,
            },
            {
                "image_id": "img1",
                "label_file": "img1.txt",
                "x_center_norm": 0.50,
                "y_center_norm": 0.50,
                "bbox_area_norm": 0.03,
                "image_width": 100,
                "image_height": 100,
            },
            {
                "image_id": "img1",
                "label_file": "img1.txt",
                "x_center_norm": 0.90,
                "y_center_norm": 0.90,
                "bbox_area_norm": 0.02,
                "image_width": 100,
                "image_height": 100,
            },
            {
                "image_id": "img2",
                "label_file": "img2.txt",
                "x_center_norm": 0.12,
                "y_center_norm": 0.30,
                "bbox_area_norm": 0.01,
                "image_width": 200,
                "image_height": 200,
            },
            {
                "image_id": "img2",
                "label_file": "img2.txt",
                "x_center_norm": 0.82,
                "y_center_norm": 0.72,
                "bbox_area_norm": 0.02,
                "image_width": 200,
                "image_height": 200,
            },
        ]
    )


def test_load_zones_loads_valid_yaml_rectangles(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)

    zones = load_zones(config_path)

    assert isinstance(zones, dict)
    assert "top_lane" in zones
    assert "river" in zones
    assert zones["top_lane"].xmin == pytest.approx(0.00)
    assert zones["dragon_zone"].xmax == pytest.approx(1.00)


def test_assign_primary_zone_returns_expected_zone_for_obvious_coordinates(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)
    zones = load_zones(config_path)

    assert assign_primary_zone(0.10, 0.10, zones) == "baron_zone"
    assert assign_primary_zone(0.50, 0.50, zones) == "river"
    assert assign_primary_zone(0.90, 0.90, zones) == "dragon_zone"


def test_label_points_adds_expected_columns(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)

    df = _make_labeled_input_df()
    labeled = label_points(df, config_path=config_path)

    assert "primary_zone" in labeled.columns
    assert "all_zones" in labeled.columns
    assert "map_side" in labeled.columns
    assert "near_major_objective" in labeled.columns

    assert len(labeled) == len(df)
    assert labeled["primary_zone"].notna().any()
    assert labeled["map_side"].isin(["blue_side", "red_side", "center"]).all()


def test_compute_snapshot_features_returns_core_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)

    df = _make_labeled_input_df()
    labeled = label_points(df, config_path=config_path)
    snapshot_df = labeled[labeled["image_id"] == "img1"]

    features = compute_snapshot_features(snapshot_df)

    expected_keys = {
        "n_detections",
        "mean_x",
        "mean_y",
        "std_x",
        "std_y",
        "avg_pairwise_distance",
        "top_lane_count",
        "mid_lane_count",
        "bot_lane_count",
        "jungle_count",
        "river_count",
        "base_count",
        "blue_side_count",
        "red_side_count",
        "center_count",
        "lane_entropy",
        "spread_score",
        "grouping_score",
    }

    assert expected_keys.issubset(features.keys())
    assert features["n_detections"] == 3


def test_build_snapshot_feature_table_creates_one_row_per_image_id(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)

    df = _make_labeled_input_df()
    labeled = label_points(df, config_path=config_path)

    feature_df = build_snapshot_feature_table(labeled)

    assert len(feature_df) == 2
    assert set(feature_df["image_id"]) == {"img1", "img2"}
    assert "image_width" in feature_df.columns
    assert "image_height" in feature_df.columns
    assert "label_file" in feature_df.columns


def test_scale_feature_table_preserves_image_id_and_scales_numeric_columns(tmp_path: Path) -> None:
    config_path = tmp_path / "map_zones.yaml"
    _write_test_zone_config(config_path)

    df = _make_labeled_input_df()
    labeled = label_points(df, config_path=config_path)
    feature_df = build_snapshot_feature_table(labeled)

    scaled_df, scaler = scale_feature_table(feature_df)

    assert "image_id" in scaled_df.columns
    assert list(scaled_df["image_id"]) == list(feature_df["image_id"])
    assert scaler is not None

    numeric_cols = [
        col for col in scaled_df.columns
        if col not in {"image_id", "label_file"} and pd.api.types.is_numeric_dtype(scaled_df[col])
    ]
    assert len(numeric_cols) > 0