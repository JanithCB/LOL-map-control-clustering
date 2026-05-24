from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim column names for safer downstream access."""
    return df.rename(columns={col: str(col).strip() for col in df.columns})


def _find_cluster_column(df: pd.DataFrame) -> str:
    """Find the cluster identifier column."""
    candidates = ["cluster_id", "cluster", "cluster_idx"]
    matches = [col for col in candidates if col in df.columns]
    if not matches:
        raise ValueError(
            f"Could not find a cluster column. Expected one of {candidates}, got {list(df.columns)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous cluster columns found: {matches}")
    return matches[0]


def normalize_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the cluster column to cluster_id."""
    df = _normalize_columns(df).copy()
    cluster_col = _find_cluster_column(df)
    if cluster_col != "cluster_id":
        df = df.rename(columns={cluster_col: "cluster_id"})
    return df


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV or raise a clear file error."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_cluster_inputs(base_dir: str | Path = ".") -> Dict[str, pd.DataFrame]:
    """
    Load the four Task 1 input CSV files.

    Returns a dict with:
    - cluster_sizes
    - cluster_top_features
    - representative_samples
    - qualitative_comparison_notes
    """
    base = Path(base_dir)
    clustering_dir = base / "outputs" / "reports" / "clustering"
    comparison_dir = base / "outputs" / "reports" / "comparison"

    return {
        "cluster_sizes": normalize_cluster_column(
            _load_csv(clustering_dir / "cluster_sizes.csv")
        ),
        "cluster_top_features": normalize_cluster_column(
            _load_csv(clustering_dir / "cluster_top_features.csv")
        ),
        "representative_samples": normalize_cluster_column(
            _load_csv(clustering_dir / "representative_samples.csv")
        ),
        "qualitative_comparison_notes": normalize_cluster_column(
            _load_csv(comparison_dir / "qualitative_comparison_notes.csv")
        ),
    }


def _detect_size_column(df: pd.DataFrame) -> str:
    """Detect the size/count column in cluster_sizes.csv."""
    direct_candidates = ["size", "count", "n"]
    matches = [col for col in direct_candidates if col in df.columns]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous size columns found: {matches}")

    numeric_cols = [
        col for col in df.columns
        if col != "cluster_id" and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not detect size column. Expected one of {direct_candidates} or one numeric non-cluster column."
    )


def _extract_cluster_size_info(sizes_df: pd.DataFrame, cluster_id: int) -> Dict[str, Any]:
    """Extract size and percentage for a given cluster."""
    df = normalize_cluster_column(sizes_df).copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    size_col = _detect_size_column(df)

    row = df.loc[df["cluster_id"] == cluster_id]
    if row.empty:
        return {"size": None, "pct": None}

    size_value = float(row.iloc[0][size_col])
    total = float(pd.to_numeric(df[size_col], errors="coerce").fillna(0).sum())
    pct_value = round((size_value / total) * 100, 2) if total > 0 else None

    if size_value.is_integer():
        size_value = int(size_value)

    return {"size": size_value, "pct": pct_value}


def _extract_top_features(
    top_features_df: pd.DataFrame,
    cluster_id: int,
    top_n_features: int,
) -> List[str]:
    """Extract readable top features for one cluster."""
    df = normalize_cluster_column(top_features_df).copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    cluster_rows = df.loc[df["cluster_id"] == cluster_id].copy()

    if cluster_rows.empty:
        return []

    if "feature" in cluster_rows.columns:
        metric_candidates = ["importance", "weight", "score", "value"]
        metric_matches = [col for col in metric_candidates if col in cluster_rows.columns]
        if len(metric_matches) == 1:
            metric_col = metric_matches[0]
            cluster_rows[metric_col] = pd.to_numeric(cluster_rows[metric_col], errors="coerce").fillna(0.0)
            cluster_rows = cluster_rows.sort_values(metric_col, ascending=False)
            return [
                f"{row['feature']} ({float(row[metric_col]):.4f})"
                for _, row in cluster_rows.head(top_n_features).iterrows()
            ]

    numeric_feature_cols = [
        col for col in cluster_rows.columns
        if col != "cluster_id" and pd.api.types.is_numeric_dtype(cluster_rows[col])
    ]
    if not numeric_feature_cols:
        return []

    row = cluster_rows.iloc[0]
    scored = [(col, float(row[col])) for col in numeric_feature_cols]
    scored.sort(key=lambda item: abs(item[1]), reverse=True)

    return [f"{name} ({value:.4f})" for name, value in scored[:top_n_features]]


def _choose_representative_columns(df: pd.DataFrame) -> List[str]:
    """Choose the most useful columns for readable sample display."""
    preferred = [
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
    chosen = [col for col in preferred if col in df.columns]
    return chosen if chosen else [col for col in df.columns if col != "cluster_id"]


def _extract_representative_samples(
    reps_df: pd.DataFrame,
    cluster_id: int,
    top_n_samples: int,
) -> List[str]:
    """Extract readable representative sample strings for one cluster."""
    df = normalize_cluster_column(reps_df).copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    cluster_rows = df.loc[df["cluster_id"] == cluster_id].copy()

    if cluster_rows.empty:
        return []

    selected_cols = _choose_representative_columns(cluster_rows)
    outputs: List[str] = []

    for _, row in cluster_rows.head(top_n_samples).iterrows():
        parts = []
        for col in selected_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                parts.append(f"{col}={value}")
        if parts:
            outputs.append(" | ".join(parts))

    return outputs


def _extract_macro_notes(notes_df: pd.DataFrame, cluster_id: int) -> str:
    """Extract and combine qualitative notes for one cluster."""
    df = normalize_cluster_column(notes_df).copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
    cluster_rows = df.loc[df["cluster_id"] == cluster_id].copy()

    if cluster_rows.empty:
        return ""

    text_cols = [col for col in cluster_rows.columns if col != "cluster_id"]
    note_lines: List[str] = []

    for _, row in cluster_rows.iterrows():
        row_parts = []
        for col in text_cols:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                row_parts.append(str(value).strip())
        if row_parts:
            note_lines.append(" | ".join(row_parts))

    return " || ".join(note_lines)


def summarize_cluster(
    cluster_id: int,
    sizes_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    top_n_features: int = 8,
    top_n_samples: int = 5,
) -> Dict[str, Any]:
    """
    Build a readable summary dict for a single cluster.

    Returns fields useful for manual cluster naming in notebooks.
    """
    size_info = _extract_cluster_size_info(sizes_df, cluster_id)
    top_features = _extract_top_features(top_features_df, cluster_id, top_n_features)
    representative_samples = _extract_representative_samples(reps_df, cluster_id, top_n_samples)
    macro_notes = _extract_macro_notes(notes_df, cluster_id)

    return {
        "cluster_id": cluster_id,
        "size": size_info["size"],
        "pct": size_info["pct"],
        "top_features": "; ".join(top_features),
        "representative_samples": " || ".join(representative_samples),
        "macro_notes": macro_notes,
    }


def build_cluster_summary_table(
    sizes_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row-per-cluster summary table for manual review."""
    sizes = normalize_cluster_column(sizes_df).copy()
    sizes["cluster_id"] = pd.to_numeric(sizes["cluster_id"], errors="raise").astype(int)

    cluster_ids = sorted(sizes["cluster_id"].unique().tolist())
    rows = [
        summarize_cluster(
            cluster_id=cluster_id,
            sizes_df=sizes_df,
            top_features_df=top_features_df,
            reps_df=reps_df,
            notes_df=notes_df,
        )
        for cluster_id in cluster_ids
    ]
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def print_cluster_summary(
    cluster_id: int,
    sizes_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> None:
    """Print a notebook-friendly cluster summary."""
    summary = summarize_cluster(
        cluster_id=cluster_id,
        sizes_df=sizes_df,
        top_features_df=top_features_df,
        reps_df=reps_df,
        notes_df=notes_df,
    )

    print(f"Cluster {summary['cluster_id']}")
    print(f"Size: {summary['size']}")
    print(f"Pct: {summary['pct']}")
    print(f"Top features: {summary['top_features']}")
    print(f"Representative samples: {summary['representative_samples']}")
    print(f"Macro notes: {summary['macro_notes']}")