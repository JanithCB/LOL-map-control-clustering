# src/features/map_zones.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class RectZone:
    """
    Rectangle-based minimap zone defined in normalized [0, 1] coordinates.
    """

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    category: str | None = None

    def contains(self, x: float, y: float) -> bool:
        """
        Return True if a normalized point lies inside the zone.
        Bounds are inclusive.
        """
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


def zone_priority() -> list[str]:
    """
    Explicit zone priority used for assigning a single primary zone.

    Priority:
    1. bases
    2. objective zones
    3. river
    4. lanes
    5. jungle
    """
    return [
        "blue_base",
        "red_base",
        "baron_zone",
        "dragon_zone",
        "river",
        "top_lane",
        "mid_lane",
        "bot_lane",
        "top_jungle",
        "bot_jungle",
    ]


def _infer_category(zone_name: str) -> str:
    """
    Infer a simple semantic category from a zone name.
    """
    if zone_name in {"blue_base", "red_base"}:
        return "base"
    if zone_name in {"baron_zone", "dragon_zone"}:
        return "objective"
    if zone_name == "river":
        return "river"
    if zone_name in {"top_lane", "mid_lane", "bot_lane"}:
        return "lane"
    if zone_name in {"top_jungle", "bot_jungle"}:
        return "jungle"
    return "other"


def _validate_bounds(name: str, zone_dict: dict[str, Any]) -> None:
    """
    Validate that rectangle bounds exist and are within [0, 1].
    """
    required = {"xmin", "ymin", "xmax", "ymax"}
    missing = required - set(zone_dict.keys())
    if missing:
        raise ValueError(f"Zone '{name}' is missing required keys: {sorted(missing)}")

    xmin = float(zone_dict["xmin"])
    ymin = float(zone_dict["ymin"])
    xmax = float(zone_dict["xmax"])
    ymax = float(zone_dict["ymax"])

    for key, value in {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Zone '{name}' has {key}={value}, which is outside [0,1]")

    if xmin >= xmax:
        raise ValueError(f"Zone '{name}' has xmin >= xmax")
    if ymin >= ymax:
        raise ValueError(f"Zone '{name}' has ymin >= ymax")


def load_zones(config_path: str | Path) -> dict[str, RectZone]:
    """
    Load rectangle zones from YAML config.

    Parameters
    ----------
    config_path : str | Path
        Path to configs/map_zones.yaml

    Returns
    -------
    dict[str, RectZone]
        Mapping of zone name to RectZone object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Zone config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Zone config must be a dictionary-like YAML structure.")

    zones_data = config.get("zones")
    if not isinstance(zones_data, dict) or not zones_data:
        raise ValueError("Zone config must contain a non-empty 'zones' section.")

    zones: dict[str, RectZone] = {}
    for zone_name, zone_def in zones_data.items():
        if not isinstance(zone_def, dict):
            raise ValueError(f"Zone '{zone_name}' must be a mapping.")

        zone_type = zone_def.get("type", "rectangle")
        if zone_type != "rectangle":
            raise ValueError(
                f"Zone '{zone_name}' has unsupported type '{zone_type}'. "
                "Only rectangle zones are supported in this version."
            )

        _validate_bounds(zone_name, zone_def)

        zones[zone_name] = RectZone(
            name=zone_name,
            xmin=float(zone_def["xmin"]),
            ymin=float(zone_def["ymin"]),
            xmax=float(zone_def["xmax"]),
            ymax=float(zone_def["ymax"]),
            category=zone_def.get("category", _infer_category(zone_name)),
        )

    return zones


def assign_all_matching_zones(x: float, y: float, zones: dict[str, RectZone]) -> list[str]:
    """
    Return all zones that contain the given point.
    """
    return [zone_name for zone_name, zone in zones.items() if zone.contains(x, y)]


def assign_primary_zone(x: float, y: float, zones: dict[str, RectZone]) -> str | None:
    """
    Return the first matching zone based on explicit priority order.
    """
    matches = set(assign_all_matching_zones(x, y, zones))
    for zone_name in zone_priority():
        if zone_name in matches:
            return zone_name
    return None


def _infer_map_side(x: float, y: float, margin: float = 0.08) -> str:
    """
    Assign a coarse map side using a diagonal heuristic.

    We use the diagonal x + y = 1:
    - blue_side: clearly bottom-left side
    - red_side: clearly top-right side
    - center: near the diagonal

    Parameters
    ----------
    x, y : float
        Normalized coordinates.
    margin : float
        Distance band around the diagonal treated as center.
    """
    diagonal_score = x + y - 1.0

    if diagonal_score < -margin:
        return "blue_side"
    if diagonal_score > margin:
        return "red_side"
    return "center"


def label_points(
    df: pd.DataFrame,
    x_col: str = "x_center_norm",
    y_col: str = "y_center_norm",
    config_path: str | Path = "configs/map_zones.yaml",
) -> pd.DataFrame:
    """
    Label each point with zone-derived features.

    Adds:
    - primary_zone
    - all_zones
    - map_side
    - near_major_objective

    Parameters
    ----------
    df : pd.DataFrame
        Input detection dataframe.
    x_col : str
        Column containing normalized x coordinate.
    y_col : str
        Column containing normalized y coordinate.
    config_path : str | Path
        Path to YAML zone config.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Input DataFrame must contain columns '{x_col}' and '{y_col}'")

    zones = load_zones(config_path)
    out = df.copy()

    all_zones_values: list[list[str]] = []
    primary_zone_values: list[str | None] = []
    map_side_values: list[str] = []
    near_major_objective_values: list[bool] = []

    for x, y in zip(out[x_col], out[y_col]):
        if pd.isna(x) or pd.isna(y):
            all_matches: list[str] = []
            primary_match = None
            side = "center"
            near_objective = False
        else:
            x_f = float(x)
            y_f = float(y)
            all_matches = assign_all_matching_zones(x_f, y_f, zones)
            primary_match = assign_primary_zone(x_f, y_f, zones)
            side = _infer_map_side(x_f, y_f)
            near_objective = any(z in {"baron_zone", "dragon_zone"} for z in all_matches)

        all_zones_values.append(all_matches)
        primary_zone_values.append(primary_match)
        map_side_values.append(side)
        near_major_objective_values.append(near_objective)

    out["primary_zone"] = primary_zone_values
    out["all_zones"] = all_zones_values
    out["map_side"] = map_side_values
    out["near_major_objective"] = near_major_objective_values

    return out