from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def compute_match_macro_features(
    df: pd.DataFrame, column_map: dict[str, str], config: dict[str, Any]
) -> pd.DataFrame:
    """Compute match-level macro features from the ranked games dataset."""
    macro_df = pd.DataFrame()
    
    macro_df["game_id"] = df[column_map.get("game_id", "gameId")]
    macro_df["game_duration_seconds"] = df[column_map.get("game_duration", "gameDuration")]
    macro_df["game_duration_minutes"] = macro_df["game_duration_seconds"] / 60.0
    
    winner_col = column_map.get("winner", "winner")
    macro_df["t1_win"] = (df[winner_col] == 1).astype(int)
    macro_df["t2_win"] = (df[winner_col] == 2).astype(int)
    
    for team in ["t1", "t2"]:
        for obj in ["dragons", "barons", "towers", "inhibitors", "rift_heralds"]:
            raw_col = column_map.get(f"{team}_{obj[:-1]}_kills", f"{team}_{obj[:-1]}Kills")
            if raw_col in df.columns:
                macro_df[f"{team}_{obj}"] = df[raw_col]
            else:
                macro_df[f"{team}_{obj}"] = 0
                LOGGER.warning(f"Missing column {raw_col} in dataset. Setting to 0.")
    
    for event in ["first_blood", "first_tower", "first_inhibitor", "first_baron", "first_dragon", "first_rift_herald"]:
        raw_col = column_map.get(event, event.replace("_", "").replace("first", "first").capitalize()) # Best effort default
        if raw_col in df.columns:
            macro_df[f"t1_{event}"] = (df[raw_col] == 1).astype(int)
            macro_df[f"t2_{event}"] = (df[raw_col] == 2).astype(int)
        else:
            macro_df[f"t1_{event}"] = 0
            macro_df[f"t2_{event}"] = 0
            LOGGER.warning(f"Missing column {raw_col} in dataset. Setting to 0.")

    feat_config = config.get("feature_engineering", {})
    
    if feat_config.get("compute_total_objectives", True):
        macro_df["t1_objectives_total"] = (
            macro_df["t1_dragons"] + macro_df["t1_barons"] + 
            macro_df["t1_towers"] + macro_df["t1_inhibitors"] + macro_df["t1_rift_heralds"]
        )
        macro_df["t2_objectives_total"] = (
            macro_df["t2_dragons"] + macro_df["t2_barons"] + 
            macro_df["t2_towers"] + macro_df["t2_inhibitors"] + macro_df["t2_rift_heralds"]
        )
        
    if feat_config.get("compute_objective_differentials", True):
        macro_df["objective_diff"] = macro_df.get("t1_objectives_total", 0) - macro_df.get("t2_objectives_total", 0)
        macro_df["dragon_diff"] = macro_df["t1_dragons"] - macro_df["t2_dragons"]
        macro_df["baron_diff"] = macro_df["t1_barons"] - macro_df["t2_barons"]
        macro_df["tower_diff"] = macro_df["t1_towers"] - macro_df["t2_towers"]

    if feat_config.get("compute_duration_bins", True):
        edges = feat_config.get("duration_bin_edges", [0, 1500, 2100, 2700, 999999])
        labels = feat_config.get("duration_bin_labels", ["short", "standard", "long", "very_long"])
        macro_df["duration_bin"] = pd.cut(
            macro_df["game_duration_seconds"], bins=edges, labels=labels, include_lowest=True
        )

    return macro_df


def derive_macro_style_labels(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical playstyle columns based on macro features."""
    df = macro_df.copy()
    
    if "t1_objectives_total" in df.columns and "t2_objectives_total" in df.columns:
        total_objs = df["t1_objectives_total"] + df["t2_objectives_total"]
        median_objs = total_objs.median()
        df["objective_focus"] = np.where(total_objs > median_objs, "objective_heavy", "objective_light")
    
    if all(c in df.columns for c in ["t1_dragons", "t2_dragons", "t1_barons", "t2_barons"]):
        total_dragons = df["t1_dragons"] + df["t2_dragons"]
        total_barons = df["t1_barons"] + df["t2_barons"]
        
        conditions = [
            (total_dragons > total_barons),
            (total_barons > total_dragons)
        ]
        choices = ["dragon_focused", "baron_focused"]
        df["neutral_focus"] = np.select(conditions, choices, default="balanced")
    
    if "duration_bin" in df.columns:
        df["game_length_style"] = df["duration_bin"].astype(str)
    
    return df


def summarize_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for numeric macro features."""
    numeric_df = macro_df.select_dtypes(include=[np.number])
    summary = numeric_df.describe().T
    summary.reset_index(inplace=True)
    summary.rename(columns={"index": "feature"}, inplace=True)
    summary["median"] = numeric_df.median().values
    return summary


def summarize_macro_styles(macro_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Generate frequency tables for macro style labels."""
    summaries = {}
    style_cols = ["objective_focus", "neutral_focus", "game_length_style"]
    
    for col in style_cols:
        if col in macro_df.columns:
            counts = macro_df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            counts["proportion"] = counts["count"] / len(macro_df)
            summaries[col] = counts
            
    return summaries


def save_macro_outputs(
    macro_df: pd.DataFrame, 
    macro_summary_df: pd.DataFrame, 
    style_summaries: dict[str, pd.DataFrame], 
    output_dir: str | Path
) -> dict[str, Path]:
    """Save all macro outputs to the specified directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    feats_path = out_dir / "ranked_macro_features.csv"
    macro_df.to_csv(feats_path, index=False)
    paths["features"] = feats_path
    
    sum_path = out_dir / "macro_feature_summary.csv"
    macro_summary_df.to_csv(sum_path, index=False)
    paths["summary"] = sum_path
    
    for key, style_df in style_summaries.items():
        style_path = out_dir / f"macro_style_{key}.csv"
        style_df.to_csv(style_path, index=False)
        paths[key] = style_path
        
    return paths