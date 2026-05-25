from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def load_macro_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration for ranked-games macro analysis.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Macro config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Macro config must load into a dictionary.")

    LOGGER.info("Loaded macro config from %s", config_path)
    return config


def load_ranked_games_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load ranked-games CSV into a DataFrame.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Ranked games CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Failed to read ranked games CSV: {csv_path}") from exc

    if df.empty:
        raise ValueError(f"Ranked games CSV is empty: {csv_path}")

    LOGGER.info("Loaded ranked games CSV from %s with %d rows", csv_path, len(df))
    return df


def get_required_macro_columns(config: dict[str, Any]) -> dict[str, str]:
    """
    Return the semantic-to-raw column mapping from config.
    """
    columns = config.get("columns")
    if not isinstance(columns, dict):
        raise ValueError("Config must contain a 'columns' dictionary.")

    required_semantic_keys = [
        "game_id",
        "game_duration",
        "winner",
        "t1_tower_kills",
        "t2_tower_kills",
        "t1_dragon_kills",
        "t2_dragon_kills",
        "t1_baron_kills",
        "t2_baron_kills",
        "t1_inhibitor_kills",
        "t2_inhibitor_kills",
        "t1_rift_herald_kills",
        "t2_rift_herald_kills",
        "first_blood",
        "first_tower",
        "first_inhibitor",
        "first_baron",
        "first_dragon",
        "first_rift_herald",
    ]

    missing_keys = [key for key in required_semantic_keys if key not in columns]
    if missing_keys:
        raise ValueError(
            f"Config is missing required semantic column mappings: {missing_keys}"
        )

    return columns


def validate_ranked_games_schema(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """
    Validate that all configured raw CSV columns exist in the DataFrame.
    """
    column_map = get_required_macro_columns(config)

    raw_columns = []
    for semantic_name, raw_name in column_map.items():
        if raw_name is None:
            continue
        raw_columns.append(raw_name)

    missing_columns = [col for col in raw_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Ranked games CSV is missing required columns: "
            f"{missing_columns}. "
            "Check configs/macro_config.yaml and your local CSV headers."
        )

    LOGGER.info("Ranked games schema validation passed.")


def load_and_validate_ranked_games(
    csv_path: str | Path,
    config_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load config, load ranked-games CSV, validate schema, and return both.
    """
    config = load_macro_config(config_path)
    df = load_ranked_games_csv(csv_path)
    validate_ranked_games_schema(df, config)

    return df, config