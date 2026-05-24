# Path: gui/data_loader.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


CLUSTERING_DIR = Path("outputs") / "reports" / "clustering"
COMPARISON_DIR = Path("outputs") / "reports" / "comparison"

CLUSTER_SIZES_PATH = CLUSTERING_DIR / "cluster_sizes.csv"
CLUSTER_TOP_FEATURES_PATH = CLUSTERING_DIR / "cluster_top_features.csv"
CLUSTER_LABELS_PATH = CLUSTERING_DIR / "cluster_labels.csv"
REPRESENTATIVE_SAMPLES_PATH = CLUSTERING_DIR / "representative_samples.csv"
QUALITATIVE_NOTES_PATH = COMPARISON_DIR / "qualitative_comparison_notes.csv"


@dataclass
class RepresentativeSample:
    image_path: Optional[str] = None
    match_id: Optional[str] = None
    frame_id: Optional[str] = None
    timestamp: Optional[str] = None
    raw_summary: str = ""


@dataclass
class ClusterInfo:
    cluster_id: int
    label: str = ""
    description: str = ""
    size: Optional[float] = None
    pct: Optional[float] = None
    top_features: List[str] = field(default_factory=list)
    representative_samples: List[RepresentativeSample] = field(default_factory=list)
    notes: str = ""
    summary_bullets: List[str] = field(default_factory=list)


@dataclass
class AppData:
    cluster_infos: Dict[int, ClusterInfo] = field(default_factory=dict)
    global_notes: str = ""
    global_summary_bullets: List[str] = field(default_factory=list)
    quantitative_comparison: Optional[pd.DataFrame] = None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: str(col).strip() for col in df.columns})


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    matches = [col for col in candidates if col in df.columns]
    if not matches:
        return None
    if len(matches) > 1:
        return matches[0]
    return matches[0]


def _normalize_cluster_column(df: pd.DataFrame, required: bool = True) -> pd.DataFrame:
    df = _normalize_columns(df).copy()
    cluster_col = _find_column(df, ["cluster_id", "cluster", "cluster_idx", "cluster_label"])
    if cluster_col is None:
        if required:
            raise ValueError(
                f"Could not find cluster identifier column. Got columns: {list(df.columns)}"
            )
        return df
    if cluster_col != "cluster_id":
        df = df.rename(columns={cluster_col: "cluster_id"})
    return df


def _coerce_cluster_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cluster_id" in df.columns:
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
        df = df.dropna(subset=["cluster_id"])
        df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _detect_size_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    size_col = _find_column(df, ["size", "count", "n", "n_samples"])
    pct_col = _find_column(df, ["pct", "percentage", "proportion"])
    return size_col, pct_col


def _load_cluster_sizes(base_dir: Path) -> Dict[int, Dict[str, Optional[float]]]:
    path = base_dir / CLUSTER_SIZES_PATH
    df = _read_csv(path)
    df = _normalize_cluster_column(df)
    df = _coerce_cluster_ids(df)

    size_col, pct_col = _detect_size_columns(df)
    if size_col is None:
        raise ValueError(
            f"Could not detect size column in cluster_sizes.csv. Got columns: {list(df.columns)}"
        )

    result: Dict[int, Dict[str, Optional[float]]] = {}
    total = pd.to_numeric(df[size_col], errors="coerce").fillna(0).sum()

    for _, row in df.iterrows():
        cluster_id = int(row["cluster_id"])
        raw_size = pd.to_numeric(pd.Series([row[size_col]]), errors="coerce").iloc[0]
        size_value: Optional[float]
        pct_value: Optional[float]

        size_value = None if pd.isna(raw_size) else float(raw_size)

        if pct_col is not None:
            raw_pct = pd.to_numeric(pd.Series([row[pct_col]]), errors="coerce").iloc[0]
            pct_value = None if pd.isna(raw_pct) else float(raw_pct)
            if pct_value is not None and pct_value <= 1.0:
                pct_value *= 100.0
        else:
            if size_value is not None and total > 0:
                pct_value = float(size_value / total * 100.0)
            else:
                pct_value = None

        result[cluster_id] = {
            "size": size_value,
            "pct": None if pct_value is None else round(pct_value, 2),
        }

    return result


def _load_cluster_labels(base_dir: Path) -> Dict[int, Dict[str, str]]:
    path = base_dir / CLUSTER_LABELS_PATH
    df = _read_csv(path)
    df = _normalize_cluster_column(df)
    df = _coerce_cluster_ids(df)

    label_col = _find_column(df, ["label", "name", "cluster_name"])
    description_col = _find_column(df, ["short_description", "description", "notes"])

    if label_col is None:
        raise ValueError(
            f"Could not detect label column in cluster_labels.csv. Got columns: {list(df.columns)}"
        )

    result: Dict[int, Dict[str, str]] = {}
    for _, row in df.iterrows():
        cluster_id = int(row["cluster_id"])
        result[cluster_id] = {
            "label": _safe_text(row[label_col]),
            "description": _safe_text(row[description_col]) if description_col else "",
        }
    return result


def _load_top_features(base_dir: Path, top_n: int = 5) -> Dict[int, List[str]]:
    path = base_dir / CLUSTER_TOP_FEATURES_PATH
    df = _read_csv(path)
    df = _normalize_cluster_column(df)
    df = _coerce_cluster_ids(df)

    result: Dict[int, List[str]] = {}

    if "feature" in df.columns:
        metric_col = _find_column(df, ["importance", "weight", "score", "value", "centroid_value"])
        if metric_col is None:
            raise ValueError(
                "Could not detect feature metric column in cluster_top_features.csv "
                f"with long format. Got columns: {list(df.columns)}"
            )

        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

        for cluster_id, group in df.groupby("cluster_id"):
            top_rows = group.sort_values(metric_col, ascending=False).head(top_n)
            result[int(cluster_id)] = [
                f"{_safe_text(row['feature'])} ({float(row[metric_col]):.4f})"
                for _, row in top_rows.iterrows()
            ]
        return result

    numeric_feature_cols = [
        col for col in df.columns
        if col != "cluster_id" and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not numeric_feature_cols:
        raise ValueError(
            "Could not parse cluster_top_features.csv. Expected long format with "
            "'feature' column or wide format with numeric feature columns."
        )

    for _, row in df.iterrows():
        cluster_id = int(row["cluster_id"])
        scored = [(col, float(row[col])) for col in numeric_feature_cols]
        scored.sort(key=lambda item: abs(item[1]), reverse=True)
        result[cluster_id] = [f"{name} ({value:.4f})" for name, value in scored[:top_n]]

    return result


def _build_sample_summary(row: pd.Series, columns: List[str]) -> str:
    parts: List[str] = []
    for col in columns:
        text = _safe_text(row[col])
        if text:
            parts.append(f"{col}={text}")
    return " | ".join(parts)


def _load_representative_samples(base_dir: Path, top_n: int = 5) -> Dict[int, List[RepresentativeSample]]:
    path = base_dir / REPRESENTATIVE_SAMPLES_PATH
    df = _read_csv(path)
    df = _normalize_cluster_column(df)
    df = _coerce_cluster_ids(df)

    summary_cols = [
        col for col in [
            "image_path",
            "frame_path",
            "sample_path",
            "match_id",
            "game_id",
            "frame_id",
            "timestamp",
            "game_time",
            "sample_id",
        ]
        if col in df.columns
    ]
    if not summary_cols:
        summary_cols = [col for col in df.columns if col != "cluster_id"]

    result: Dict[int, List[RepresentativeSample]] = {}

    for cluster_id, group in df.groupby("cluster_id"):
        samples: List[RepresentativeSample] = []
        for _, row in group.head(top_n).iterrows():
            sample = RepresentativeSample(
                image_path=_safe_text(
                    row["image_path"]
                    if "image_path" in row.index
                    else row["frame_path"]
                    if "frame_path" in row.index
                    else row["sample_path"]
                    if "sample_path" in row.index
                    else ""
                ) or None,
                match_id=_safe_text(
                    row["match_id"] if "match_id" in row.index else row["game_id"] if "game_id" in row.index else ""
                ) or None,
                frame_id=_safe_text(
                    row["frame_id"] if "frame_id" in row.index else row["sample_id"] if "sample_id" in row.index else ""
                ) or None,
                timestamp=_safe_text(
                    row["timestamp"] if "timestamp" in row.index else row["game_time"] if "game_time" in row.index else ""
                ) or None,
                raw_summary=_build_sample_summary(row, summary_cols),
            )
            samples.append(sample)
        result[int(cluster_id)] = samples

    return result


def _load_qualitative_notes(base_dir: Path) -> tuple[Dict[int, str], str, Dict[int, List[str]], List[str]]:
    path = base_dir / QUALITATIVE_NOTES_PATH
    df = _read_csv(path)
    df = _normalize_columns(df)

    cluster_col = _find_column(df, ["cluster_id", "cluster", "cluster_idx", "cluster_label"])
    if cluster_col is None:
        text_chunks: List[str] = []
        bullets: List[str] = []
        for _, row in df.iterrows():
            row_parts = []
            for col in df.columns:
                text = _safe_text(row[col])
                if text:
                    row_parts.append(f"{col}: {text}")
                    bullets.append(f"• {text}")
            if row_parts:
                text_chunks.append(" | ".join(row_parts))
        return {}, "\n".join(text_chunks), {}, bullets

    df = _normalize_cluster_column(df)
    df = _coerce_cluster_ids(df)

    note_cols = [col for col in df.columns if col != "cluster_id"]
    notes_by_cluster: Dict[int, str] = {}
    bullets_by_cluster: Dict[int, List[str]] = {}

    for cluster_id, group in df.groupby("cluster_id"):
        row_texts: List[str] = []
        cluster_bullets: List[str] = []
        for _, row in group.iterrows():
            parts = []
            for col in note_cols:
                text = _safe_text(row[col])
                if text:
                    parts.append(text)
                    cluster_bullets.append(f"• {text}")
            if parts:
                row_texts.append(" | ".join(parts))
        notes_by_cluster[int(cluster_id)] = " || ".join(row_texts)
        bullets_by_cluster[int(cluster_id)] = cluster_bullets

    return notes_by_cluster, "", bullets_by_cluster, []


def _load_quantitative_comparison(base_dir: Path) -> Optional[pd.DataFrame]:
    comp_dir = base_dir / COMPARISON_DIR
    if not comp_dir.exists():
        return None
    for csv_file in comp_dir.glob("*.csv"):
        if csv_file.name != QUALITATIVE_NOTES_PATH.name:
            try:
                return pd.read_csv(csv_file)
            except Exception:
                pass
    return None


def get_cluster_display_name(cluster_info: ClusterInfo) -> str:
    label = cluster_info.label.strip() if cluster_info.label else ""
    if label:
        return f"Cluster {cluster_info.cluster_id} – {label}"
    return f"Cluster {cluster_info.cluster_id}"


def load_cluster_infos(base_dir: str | Path = ".") -> Dict[int, ClusterInfo]:
    base = Path(base_dir)

    sizes = _load_cluster_sizes(base)
    labels = _load_cluster_labels(base)
    top_features = _load_top_features(base)
    representative_samples = _load_representative_samples(base)
    notes_by_cluster, _, bullets_by_cluster, _ = _load_qualitative_notes(base)

    cluster_ids = sorted(
        set(sizes.keys())
        | set(labels.keys())
        | set(top_features.keys())
        | set(representative_samples.keys())
        | set(notes_by_cluster.keys())
    )

    cluster_infos: Dict[int, ClusterInfo] = {}
    for cluster_id in cluster_ids:
        size_info = sizes.get(cluster_id, {})
        label_info = labels.get(cluster_id, {})

        size_value = size_info.get("size")
        if isinstance(size_value, float) and size_value.is_integer():
            size_value = int(size_value)

        cluster_infos[cluster_id] = ClusterInfo(
            cluster_id=cluster_id,
            label=label_info.get("label", ""),
            description=label_info.get("description", ""),
            size=size_value,
            pct=size_info.get("pct"),
            top_features=top_features.get(cluster_id, []),
            representative_samples=representative_samples.get(cluster_id, []),
            notes=notes_by_cluster.get(cluster_id, ""),
            summary_bullets=bullets_by_cluster.get(cluster_id, [])
        )

    return cluster_infos


def load_app_data(base_dir: str | Path = ".") -> AppData:
    base = Path(base_dir)
    cluster_infos = load_cluster_infos(base)
    _, global_notes, _, global_bullets = _load_qualitative_notes(base)
    quant_comp = _load_quantitative_comparison(base)
    return AppData(
        cluster_infos=cluster_infos, 
        global_notes=global_notes,
        global_summary_bullets=global_bullets,
        quantitative_comparison=quant_comp
    )


def get_cluster_info(cluster_id: int, base_dir: str | Path = ".") -> Optional[ClusterInfo]:
    cluster_infos = load_cluster_infos(base_dir)
    return cluster_infos.get(cluster_id)