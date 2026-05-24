# Path: src/clustering/interpret_clusters.py

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_ROOT = Path(".")
CLUSTERING_DIR = REPO_ROOT / "outputs" / "reports" / "clustering"
COMPARISON_DIR = REPO_ROOT / "outputs" / "reports" / "comparison"
SRC_CLUSTERING_DIR = REPO_ROOT / "src" / "clustering"

CLUSTER_SIZES_PATH = CLUSTERING_DIR / "cluster_sizes.csv"
CLUSTER_TOP_FEATURES_PATH = CLUSTERING_DIR / "cluster_top_features.csv"
REPRESENTATIVE_SAMPLES_PATH = CLUSTERING_DIR / "representative_samples.csv"
QUAL_NOTES_PATH = COMPARISON_DIR / "qualitative_comparison_notes.csv"

INTERPRETATION_SHEET_PATH = CLUSTERING_DIR / "cluster_interpretation_sheet.csv"
LABELS_CSV_PATH = CLUSTERING_DIR / "cluster_labels.csv"
PY_LABELS_PATH = SRC_CLUSTERING_DIR / "cluster_labels.py"

DEFAULT_TOP_N_FEATURES = 8
DEFAULT_TOP_N_SAMPLES = 5


def ensure_output_dirs() -> None:
    """Ensure required output directories exist before writing files."""
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
    SRC_CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file or raise a clear error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names for more robust schema handling."""
    renamed = {col: str(col).strip() for col in df.columns}
    return df.rename(columns=renamed)


def find_first_matching_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    """
    Find the first matching column name from candidates.

    Raises:
        ValueError: If no candidate column exists, or if multiple columns from the
        same alias family create ambiguity in a context where one is expected.
    """
    matches = [col for col in candidates if col in df.columns]
    if not matches:
        raise ValueError(f"Could not find {label} column. Expected one of: {list(candidates)}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {label} columns found: {matches}")
    return matches[0]


def normalize_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename cluster identifier column to cluster_id."""
    df = normalize_column_names(df).copy()
    cluster_col = find_first_matching_column(
        df,
        candidates=("cluster_id", "cluster", "cluster_idx", "cluster_label"),
        label="cluster identifier",
    )
    if cluster_col != "cluster_id":
        df = df.rename(columns={cluster_col: "cluster_id"})
    return df


def detect_size_column(df: pd.DataFrame) -> str:
    """Detect the column representing cluster size/count."""
    df = normalize_column_names(df)
    direct_matches = [col for col in ("size", "count", "n", "n_samples") if col in df.columns]
    if len(direct_matches) == 1:
        return direct_matches[0]
    if len(direct_matches) > 1:
        raise ValueError(f"Ambiguous size/count columns found: {direct_matches}")

    numeric_candidates = [
        col for col in df.columns
        if col != "cluster_id" and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not detect cluster size column. Expected one of ['size', 'count', 'n'] "
        f"or exactly one numeric non-cluster column, found: {list(df.columns)}"
    )


def parse_cluster_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize cluster sizes into cluster_id, size, pct."""
    df = normalize_cluster_column(df).copy()
    size_col = detect_size_column(df)

    out = df[["cluster_id", size_col]].copy()
    out = out.rename(columns={size_col: "size"})
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="raise").astype(int)
    out["size"] = pd.to_numeric(out["size"], errors="raise")

    total = float(out["size"].sum())
    out["pct"] = (out["size"] / total * 100.0).round(2) if total > 0 else 0.0
    return out.sort_values("cluster_id").reset_index(drop=True)


def _detect_feature_value_column(df: pd.DataFrame) -> str:
    """Find the metric column in long-format feature tables."""
    candidates = [col for col in ("importance", "weight", "score", "value", "centroid_value") if col in df.columns]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous feature metric columns found: {candidates}")
    raise ValueError(
        "Could not detect feature metric column. Expected one of "
        "['importance', 'weight', 'score', 'value', 'centroid_value']."
    )


def extract_top_features(df: pd.DataFrame, top_n: int = DEFAULT_TOP_N_FEATURES) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extract per-cluster top features.

    Supports:
    - Long format: cluster_id, feature, importance/weight/score/value
    - Wide format: cluster_id + numeric feature columns
    """
    df = normalize_cluster_column(df).copy()

    if "feature" in df.columns:
        metric_col = _detect_feature_value_column(df)
        df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

        grouped: Dict[int, List[Tuple[str, float]]] = {}
        for cluster_id, group in df.groupby("cluster_id"):
            top_rows = group.sort_values(metric_col, ascending=False).head(top_n)
            grouped[int(cluster_id)] = [
                (str(row["feature"]), float(row[metric_col]))
                for _, row in top_rows.iterrows()
            ]
        return grouped

    numeric_feature_cols = [
        col for col in df.columns
        if col != "cluster_id" and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_feature_cols:
        raise ValueError(
            "Could not interpret cluster_top_features.csv. "
            "Expected long format with a 'feature' column or wide format with numeric feature columns."
        )

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)

    grouped = {}
    for _, row in df.iterrows():
        cluster_id = int(row["cluster_id"])
        scored_features = [(col, float(row[col])) for col in numeric_feature_cols]
        scored_features.sort(key=lambda item: abs(item[1]), reverse=True)
        grouped[cluster_id] = scored_features[:top_n]
    return grouped


def _pick_representative_columns(df: pd.DataFrame) -> List[str]:
    """Pick the most helpful columns for readable representative sample strings."""
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
    if chosen:
        return chosen

    fallback = [col for col in df.columns if col != "cluster_id"]
    if not fallback:
        raise ValueError("representative_samples.csv contains no usable columns besides cluster_id.")
    return fallback


def extract_representative_samples(
    df: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N_SAMPLES,
) -> Dict[int, List[str]]:
    """Extract readable representative sample summaries per cluster."""
    df = normalize_cluster_column(df).copy()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)

    selected_cols = _pick_representative_columns(df)
    grouped: Dict[int, List[str]] = {}

    for cluster_id, group in df.groupby("cluster_id"):
        rows: List[str] = []
        for _, row in group.head(top_n).iterrows():
            parts = []
            for col in selected_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    parts.append(f"{col}={value}")
            if parts:
                rows.append(" | ".join(parts))
        grouped[int(cluster_id)] = rows

    return grouped


def extract_macro_notes(df: pd.DataFrame) -> Dict[int, str]:
    """Collapse qualitative note rows into one readable string per cluster."""
    try:
        df = normalize_cluster_column(df).copy()
    except ValueError:
        return {}

    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="raise").astype(int)

    text_cols = [col for col in df.columns if col != "cluster_id"]
    if not text_cols:
        raise ValueError("qualitative_comparison_notes.csv has no note columns besides cluster_id.")

    grouped: Dict[int, str] = {}
    for cluster_id, group in df.groupby("cluster_id"):
        note_chunks: List[str] = []
        for _, row in group.iterrows():
            row_parts = []
            for col in text_cols:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    row_parts.append(str(value).strip())
            if row_parts:
                note_chunks.append(" | ".join(row_parts))
        grouped[int(cluster_id)] = " || ".join(note_chunks)

    return grouped


def format_feature_pairs(feature_pairs: Iterable[Tuple[str, float]]) -> str:
    """Format feature-value pairs into a readable semicolon-separated string."""
    return "; ".join(f"{name} ({value:.4f})" for name, value in feature_pairs)


def format_sample_strings(samples: Iterable[str]) -> str:
    """Format representative sample summaries into a readable string."""
    return " || ".join(sample for sample in samples if sample)


def tokenize_text(text: str) -> List[str]:
    """Tokenize a free-text cluster evidence blob into normalized tokens."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def infer_label(
    feature_names: Sequence[str],
    note_text: str,
    size_pct: float,
) -> Tuple[str, str]:
    """
    Infer a draft gameplay label from feature names, notes, and cluster size.

    This is heuristic on purpose: draft labels should be easy to review and edit,
    not treated as final truth.
    """
    blob = " ".join(feature_names) + " " + (note_text or "")
    tokens = set(tokenize_text(blob))

    has_bot = any(token in tokens for token in ("bot", "bottom", "dragon"))
    has_top = any(token in tokens for token in ("top", "herald", "baron"))
    has_mid = any(token in tokens for token in ("mid", "middle"))
    has_river = "river" in tokens
    has_jungle = "jungle" in tokens
    has_lane = any(token in tokens for token in ("lane", "laning", "top_lane", "mid_lane", "bot_lane"))
    has_group = any(token in tokens for token in ("group", "grouped", "five", "teamfight", "collapse", "stacked"))
    has_invade = any(token in tokens for token in ("invade", "invading", "deep", "enemy_jungle"))
    has_objective = any(token in tokens for token in ("dragon", "herald", "baron", "pit", "objective"))
    has_neutral_words = any(token in tokens for token in ("standard", "neutral", "default"))

    if has_bot and (has_river or has_jungle or has_objective):
        return (
            "Bot-side objective setup",
            "Strong bot-side presence around lane, river, jungle, or dragon zones suggests setup for objective pressure or control.",
        )

    if has_top and (has_river or has_jungle or has_objective):
        return (
            "Top-side objective setup",
            "Top-side concentration near lane, river, jungle, or herald/baron areas suggests setup for pressure or objective control.",
        )

    if has_mid and has_group:
        return (
            "5-man mid group",
            "Players are concentrated around mid with reduced side-lane spread, consistent with grouped mid pressure or a mid-focused fight setup.",
        )

    if has_invade and has_jungle:
        return (
            "Deep invade",
            "Heavy jungle and enemy-side map occupation suggests a coordinated invade or aggressive vision-control pattern.",
        )

    if has_river and has_group:
        return (
            "River contest setup",
            "The cluster shows grouped activity around river corridors, consistent with contesting vision, rotations, or nearby objectives.",
        )

    if has_objective and has_group:
        return (
            "Objective collapse",
            "Players are grouped near a major objective area, indicating a coordinated collapse or concentrated contest state.",
        )

    if (has_lane or has_neutral_words) and size_pct >= 35.0:
        return (
            "Standard laning",
            "A large, comparatively neutral lane-distributed state with limited evidence of strong grouped objective pressure.",
        )

    return (
        "Cluster pattern review",
        "This cluster appears distinct but should be manually named from its feature profile, representative samples, and qualitative notes.",
    )


def build_interpretation_sheet(
    sizes_df: pd.DataFrame,
    top_features_map: Dict[int, List[Tuple[str, float]]],
    sample_map: Dict[int, List[str]],
    notes_map: Dict[int, str],
) -> pd.DataFrame:
    """Build the combined review sheet used for human interpretation."""
    size_lookup = {
        int(row["cluster_id"]): {"size": row["size"], "pct": row["pct"]}
        for _, row in sizes_df.iterrows()
    }

    cluster_ids = sorted(
        set(size_lookup.keys())
        | set(top_features_map.keys())
        | set(sample_map.keys())
        | set(notes_map.keys())
    )

    rows = []
    for cluster_id in cluster_ids:
        size = size_lookup.get(cluster_id, {}).get("size")
        pct = size_lookup.get(cluster_id, {}).get("pct")

        feature_pairs = top_features_map.get(cluster_id, [])
        feature_names = [name for name, _ in feature_pairs]
        top_features_text = format_feature_pairs(feature_pairs)

        representative_samples_text = format_sample_strings(sample_map.get(cluster_id, []))
        macro_notes_text = notes_map.get(cluster_id, "")

        draft_label, draft_description = infer_label(
            feature_names=feature_names,
            note_text=macro_notes_text,
            size_pct=float(pct) if pct is not None else 0.0,
        )

        rows.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "pct": pct,
                "top_features": top_features_text,
                "representative_samples": representative_samples_text,
                "macro_notes": macro_notes_text,
                "draft_label": draft_label,
                "draft_description": draft_description,
                "manual_label": "",
                "manual_description": "",
                "review_status": "needs_review",
            }
        )

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def prefer_manual_or_draft(row: pd.Series, manual_col: str, draft_col: str) -> str:
    """Prefer manual value if present, otherwise draft value."""
    manual_value = row.get(manual_col, "")
    if pd.notna(manual_value) and str(manual_value).strip():
        return str(manual_value).strip()
    draft_value = row.get(draft_col, "")
    return str(draft_value).strip() if pd.notna(draft_value) else ""


def export_labels_csv(interpretation_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Export the final machine-readable labels CSV used by reports and GUI."""
    final_df = interpretation_df.copy()

    final_df["label"] = final_df.apply(
        lambda row: prefer_manual_or_draft(row, "manual_label", "draft_label"),
        axis=1,
    )
    final_df["short_description"] = final_df.apply(
        lambda row: prefer_manual_or_draft(row, "manual_description", "draft_description"),
        axis=1,
    )

    out = final_df[
        ["cluster_id", "label", "short_description", "size", "pct", "review_status"]
    ].copy()
    out.to_csv(path, index=False)
    return out


def export_python_mapping(labels_df: pd.DataFrame, path: Path) -> None:
    """Write a simple Python mapping module for later GUI use."""
    lines: List[str] = [
        "# Path: src/clustering/cluster_labels.py",
        "",
        "from __future__ import annotations",
        "",
        "import csv",
        "from pathlib import Path",
        "from typing import Dict, Tuple",
        "",
        'DEFAULT_LABELS_CSV = Path("outputs/reports/clustering/cluster_labels.csv")',
        "",
        "CLUSTER_LABELS = {",
    ]

    for _, row in labels_df.sort_values("cluster_id").iterrows():
        cluster_id = int(row["cluster_id"])
        label = str(row["label"]).replace('"', '\\"')
        lines.append(f'    {cluster_id}: "{label}",')
    lines.extend(["}", "", "CLUSTER_DESCRIPTIONS = {"])

    for _, row in labels_df.sort_values("cluster_id").iterrows():
        cluster_id = int(row["cluster_id"])
        description = str(row["short_description"]).replace('"', '\\"')
        lines.append(f'    {cluster_id}: "{description}",')

    lines.extend(
        [
            "}",
            "",
            "",
            "def load_cluster_labels(csv_path: str | Path = DEFAULT_LABELS_CSV) -> Tuple[Dict[int, str], Dict[int, str]]:",
            "    csv_path = Path(csv_path)",
            "    if not csv_path.exists():",
            "        return CLUSTER_LABELS, CLUSTER_DESCRIPTIONS",
            "",
            "    labels: Dict[int, str] = {}",
            "    descriptions: Dict[int, str] = {}",
            "",
            '    with csv_path.open("r", encoding="utf-8", newline="") as f:',
            "        reader = csv.DictReader(f)",
            "        for row in reader:",
            '            cluster_id = int(row["cluster_id"])',
            '            labels[cluster_id] = row["label"]',
            '            descriptions[cluster_id] = row["short_description"]',
            "",
            "    return labels, descriptions",
            "",
            "",
            "def get_cluster_label(cluster_id: int, csv_path: str | Path = DEFAULT_LABELS_CSV) -> str:",
            "    labels, _ = load_cluster_labels(csv_path)",
            '    return labels.get(cluster_id, f"Cluster {cluster_id}")',
            "",
            "",
            "def get_cluster_description(cluster_id: int, csv_path: str | Path = DEFAULT_LABELS_CSV) -> str:",
            "    _, descriptions = load_cluster_labels(csv_path)",
            '    return descriptions.get(cluster_id, "")',
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_interpretation(
    top_n_features: int = DEFAULT_TOP_N_FEATURES,
    top_n_samples: int = DEFAULT_TOP_N_SAMPLES,
    export_py: bool = False,
) -> Dict[str, Path]:
    """
    Run the Task 1 interpretation pipeline.

    Returns a dictionary of written output paths for downstream scripts or notebooks.
    """
    ensure_output_dirs()

    cluster_sizes_df = parse_cluster_sizes(load_csv(CLUSTER_SIZES_PATH))
    top_features_map = extract_top_features(load_csv(CLUSTER_TOP_FEATURES_PATH), top_n=top_n_features)
    representative_samples_map = extract_representative_samples(
        load_csv(REPRESENTATIVE_SAMPLES_PATH),
        top_n=top_n_samples,
    )
    macro_notes_map = extract_macro_notes(load_csv(QUAL_NOTES_PATH))

    interpretation_df = build_interpretation_sheet(
        sizes_df=cluster_sizes_df,
        top_features_map=top_features_map,
        sample_map=representative_samples_map,
        notes_map=macro_notes_map,
    )
    interpretation_df.to_csv(INTERPRETATION_SHEET_PATH, index=False)

    labels_df = export_labels_csv(interpretation_df, LABELS_CSV_PATH)

    outputs = {
        "interpretation_sheet": INTERPRETATION_SHEET_PATH,
        "labels_csv": LABELS_CSV_PATH,
    }

    if export_py:
        export_python_mapping(labels_df, PY_LABELS_PATH)
        outputs["python_mapping"] = PY_LABELS_PATH

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Interpret and name gameplay clusters from clustering report CSV files."
    )
    parser.add_argument(
        "--export-py",
        action="store_true",
        help="Also export src/clustering/cluster_labels.py for GUI integration.",
    )
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=DEFAULT_TOP_N_FEATURES,
        help=f"Number of top features to keep per cluster (default: {DEFAULT_TOP_N_FEATURES}).",
    )
    parser.add_argument(
        "--top-n-samples",
        type=int,
        default=DEFAULT_TOP_N_SAMPLES,
        help=f"Number of representative samples to keep per cluster (default: {DEFAULT_TOP_N_SAMPLES}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    outputs = run_interpretation(
        top_n_features=args.top_n_features,
        top_n_samples=args.top_n_samples,
        export_py=args.export_py,
    )

    print(f"Wrote interpretation sheet: {outputs['interpretation_sheet']}")
    print(f"Wrote label mapping CSV: {outputs['labels_csv']}")
    if "python_mapping" in outputs:
        print(f"Wrote Python label mapping: {outputs['python_mapping']}")

    return outputs


if __name__ == "__main__":
    main()