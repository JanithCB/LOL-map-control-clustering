from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_ranked_data import validate_ranked_games_schema
from src.features.macro_features import (
    compute_match_macro_features,
    derive_macro_style_labels,
    summarize_macro_features,
    summarize_macro_styles,
)


@pytest.fixture
def sample_ranked_df() -> pd.DataFrame:
    data = {
        "gameId": [1, 2, 3, 4, 5],
        "gameDuration": [1200, 1600, 2200, 3000, 1800],
        "winner": [1, 2, 1, 2, 1],
        "t1_towerKills": [8, 2, 10, 4, 7],
        "t2_towerKills": [1, 9, 2, 11, 4],
        "t1_dragonKills": [3, 0, 4, 1, 2],
        "t2_dragonKills": [0, 4, 1, 4, 2],
        "t1_baronKills": [1, 0, 2, 0, 1],
        "t2_baronKills": [0, 1, 0, 2, 0],
        "t1_inhibitorKills": [1, 0, 2, 0, 1],
        "t2_inhibitorKills": [0, 2, 0, 3, 0],
        "t1_riftHeraldKills": [1, 0, 1, 0, 0],
        "t2_riftHeraldKills": [0, 1, 0, 1, 1],
        "firstBlood": [1, 2, 1, 2, 1],
        "firstTower": [1, 2, 1, 2, 1],
        "firstInhibitor": [1, 2, 1, 2, 1],
        "firstBaron": [1, 2, 1, 2, 1],
        "firstDragon": [1, 2, 1, 2, 1],
        "firstRiftHerald": [1, 2, 1, 2, 2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config() -> dict:
    return {
        "dataset": {"dataset_name": "Test"},
        "columns": {
            "game_id": "gameId",
            "game_duration": "gameDuration",
            "winner": "winner",
            "t1_tower_kills": "t1_towerKills",
            "t2_tower_kills": "t2_towerKills",
            "t1_dragon_kills": "t1_dragonKills",
            "t2_dragon_kills": "t2_dragonKills",
            "t1_baron_kills": "t1_baronKills",
            "t2_baron_kills": "t2_baronKills",
            "t1_inhibitor_kills": "t1_inhibitorKills",
            "t2_inhibitor_kills": "t2_inhibitorKills",
            "t1_rift_herald_kills": "t1_riftHeraldKills",
            "t2_rift_herald_kills": "t2_riftHeraldKills",
            "first_blood": "firstBlood",
            "first_tower": "firstTower",
            "first_inhibitor": "firstInhibitor",
            "first_baron": "firstBaron",
            "first_dragon": "firstDragon",
            "first_rift_herald": "firstRiftHerald",
        },
        "feature_engineering": {
            "compute_total_objectives": True,
            "compute_objective_differentials": True,
            "compute_duration_bins": True,
            "duration_bin_edges": [0, 1500, 2100, 2700, 999999],
            "duration_bin_labels": ["short", "standard", "long", "very_long"],
            "derive_macro_style_labels": True,
        }
    }


@pytest.fixture
def sample_column_map(sample_config) -> dict:
    return sample_config["columns"]


def test_validate_ranked_games_schema_passes(sample_ranked_df, sample_config):
    validate_ranked_games_schema(sample_ranked_df, sample_config)


def test_validate_ranked_games_schema_fails_missing(sample_ranked_df, sample_config):
    bad_df = sample_ranked_df.drop(columns=["gameDuration"])
    with pytest.raises(ValueError):
        validate_ranked_games_schema(bad_df, sample_config)


def test_compute_match_macro_features_columns(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    expected_cols = [
        "game_id", "t1_win", "t2_win", "t1_dragons", "t2_dragons", 
        "t1_objectives_total", "t2_objectives_total", "objective_diff", "duration_bin"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_t1_win_derived_correctly(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    assert df.loc[0, "t1_win"] == 1
    assert df.loc[0, "t2_win"] == 0
    assert df.loc[1, "t1_win"] == 0
    assert df.loc[1, "t2_win"] == 1


def test_objective_diff_computed(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    assert (df["t1_objectives_total"] - df["t2_objectives_total"] == df["objective_diff"]).all()


def test_duration_bin_populated(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    assert not df["duration_bin"].isna().any()


def test_derive_macro_style_labels_columns(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    style_df = derive_macro_style_labels(df)
    assert "objective_focus" in style_df.columns
    assert "neutral_focus" in style_df.columns
    assert "game_length_style" in style_df.columns


def test_summarize_macro_features_nonempty(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    summary = summarize_macro_features(df)
    assert not summary.empty


def test_summarize_macro_styles_keys(sample_ranked_df, sample_column_map, sample_config):
    df = compute_match_macro_features(sample_ranked_df, sample_column_map, sample_config)
    style_df = derive_macro_style_labels(df)
    summaries = summarize_macro_styles(style_df)
    assert "objective_focus" in summaries
    assert "neutral_focus" in summaries
    assert "game_length_style" in summaries