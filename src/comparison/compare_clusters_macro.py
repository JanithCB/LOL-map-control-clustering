from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_minimap_cluster_outputs(
    cluster_sizes_path: str | Path,
    top_features_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load minimap cluster summary outputs for comparison.

    Parameters
    ----------
    cluster_sizes_path : str | Path
        Path to cluster_sizes.csv
    top_features_path : str | Path
        Path to cluster_top_features.csv

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cluster_sizes_df, top_features_df)
    """
    cluster_sizes_path = Path(cluster_sizes_path)
    top_features_path = Path(top_features_path)

    if not cluster_sizes_path.exists():
        raise FileNotFoundError(f"Cluster sizes file not found: {cluster_sizes_path}")
    if not top_features_path.exists():
        raise FileNotFoundError(f"Top features file not found: {top_features_path}")

    cluster_sizes_df = pd.read_csv(cluster_sizes_path)
    top_features_df = pd.read_csv(top_features_path)

    if cluster_sizes_df.empty:
        raise ValueError("cluster_sizes_df is empty.")
    if top_features_df.empty:
        raise ValueError("top_features_df is empty.")

    LOGGER.info("Loaded minimap cluster outputs.")
    return cluster_sizes_df, top_features_df


def load_macro_outputs(macro_features_path: str | Path) -> pd.DataFrame:
    """
    Load ranked macro features CSV.
    """
    macro_features_path = Path(macro_features_path)

    if not macro_features_path.exists():
        raise FileNotFoundError(f"Macro features file not found: {macro_features_path}")

    macro_df = pd.read_csv(macro_features_path)
    if macro_df.empty:
        raise ValueError("macro_df is empty.")

    LOGGER.info("Loaded macro outputs from %s", macro_features_path)
    return macro_df


def summarize_minimap_clusters_for_comparison(
    cluster_sizes_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one row per cluster with top 3 features.
    """
    required_size_cols = {"cluster_label", "n_samples", "proportion"}
    required_top_cols = {"cluster_label", "rank", "feature"}

    missing_size = required_size_cols - set(cluster_sizes_df.columns)
    missing_top = required_top_cols - set(top_features_df.columns)

    if missing_size:
        raise ValueError(f"cluster_sizes_df missing columns: {sorted(missing_size)}")
    if missing_top:
        raise ValueError(f"top_features_df missing columns: {sorted(missing_top)}")

    top3 = (
        top_features_df[top_features_df["rank"].isin([1, 2, 3])]
        .copy()
        .sort_values(["cluster_label", "rank"])
    )

    top3_wide = (
        top3.pivot(index="cluster_label", columns="rank", values="feature")
        .rename(columns={1: "top_feature_1", 2: "top_feature_2", 3: "top_feature_3"})
        .reset_index()
    )

    summary_df = cluster_sizes_df.merge(top3_wide, on="cluster_label", how="left")
    summary_df = summary_df.sort_values("cluster_label").reset_index(drop=True)

    return summary_df


def summarize_macro_styles_for_comparison(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a compact count/proportion summary for macro style categories.
    """
    style_cols = [
        "objective_focus",
        "neutral_focus",
        "game_length_style",
    ]

    missing_cols = [col for col in style_cols if col not in macro_df.columns]
    if missing_cols:
        raise ValueError(f"macro_df missing style columns: {missing_cols}")

    frames: list[pd.DataFrame] = []
    total_n = len(macro_df)

    for col in style_cols:
        counts = (
            macro_df[col]
            .fillna("missing")
            .value_counts(dropna=False)
            .rename_axis("category_value")
            .reset_index(name="count")
        )
        counts["style_category"] = col
        counts["proportion"] = counts["count"] / total_n
        frames.append(counts[["style_category", "category_value", "count", "proportion"]])

    summary_df = pd.concat(frames, ignore_index=True)
    return summary_df


def build_qualitative_comparison_notes(
    minimap_summary_df: pd.DataFrame,
    macro_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build descriptive, non-causal comparison notes between minimap clusters and macro styles.
    """
    if minimap_summary_df.empty:
        raise ValueError("minimap_summary_df is empty.")
    if macro_summary_df.empty:
        raise ValueError("macro_summary_df is empty.")

    cluster_descriptions = []
    for _, row in minimap_summary_df.iterrows():
        parts = []
        for col in ["top_feature_1", "top_feature_2", "top_feature_3"]:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        cluster_descriptions.append(
            f"Cluster {int(row['cluster_label'])}: top features suggest emphasis on {', '.join(parts) if parts else 'mixed spatial traits'}."
        )

    objective_focus_top = macro_summary_df.loc[
        macro_summary_df["style_category"] == "objective_focus"
    ].sort_values("count", ascending=False)
    pace_top = macro_summary_df.loc[
        macro_summary_df["style_category"] == "pace_style"
    ].sort_values("count", ascending=False)
    neutral_top = macro_summary_df.loc[
        macro_summary_df["style_category"] == "neutral_focus"
    ].sort_values("count", ascending=False)
    length_top = macro_summary_df.loc[
        macro_summary_df["style_category"] == "game_length_style"
    ].sort_values("count", ascending=False)

    def _top_label(df: pd.DataFrame, fallback: str) -> str:
        if df.empty:
            return fallback
        row = df.iloc[0]
        return f"{row['category_value']} ({row['proportion']:.1%})"

    notes = [
        {
            "comparison_axis": "map grouping vs game pace",
            "minimap_view": "Minimap clusters describe repeated spatial states such as grouped or spread map formations.",
            "ranked_macro_view": f"Most common pace style: {_top_label(pace_top, 'unknown')}.",
            "interpretation_note": (
                "This comparison is descriptive only: grouped minimap states may co-occur with faster games, "
                "but the datasets are not directly aligned at the same match level."
            ),
        },
        {
            "comparison_axis": "bot-side control vs dragon focus",
            "minimap_view": "Clusters with bot-side or river-oriented top features may reflect objective setup patterns.",
            "ranked_macro_view": f"Most common neutral-focus pattern: {_top_label(neutral_top, 'unknown')}.",
            "interpretation_note": (
                "Bot-side spatial emphasis can be compared qualitatively with dragon-focused macro outcomes, "
                "but should not be treated as a direct matched-sample finding."
            ),
        },
        {
            "comparison_axis": "spread states vs low-objective games",
            "minimap_view": "Some minimap clusters may indicate lane-spread or low-grouping states.",
            "ranked_macro_view": f"Most common objective-focus pattern: {_top_label(objective_focus_top, 'unknown')}.",
            "interpretation_note": (
                "Spread states may be consistent with lower objective pressure in some games, "
                "but this is a descriptive interpretation rather than a causal conclusion."
            ),
        },
        {
            "comparison_axis": "state diversity vs game length",
            "minimap_view": "Multiple distinct cluster types suggest a range of map-control states during gameplay.",
            "ranked_macro_view": f"Most common game-length style: {_top_label(length_top, 'unknown')}.",
            "interpretation_note": (
                "Longer games may allow a broader variety of spatial states, but this comparison remains "
                "semi-quantitative because minimap frames and ranked matches are not one-to-one aligned."
            ),
        },
        {
            "comparison_axis": "cluster summary overview",
            "minimap_view": " | ".join(cluster_descriptions[:5]),
            "ranked_macro_view": "Macro summaries aggregate objective focus, pace, neutral-control emphasis, and game length.",
            "interpretation_note": (
                "Use these two views together as complementary evidence: minimap clusters show how teams occupy space, "
                "while ranked macro features describe overall match outcomes and tempo."
            ),
        },
    ]

    return pd.DataFrame(notes)


def generate_per_cluster_macro_summaries(minimap_summary_df: pd.DataFrame) -> dict[int, str]:
    """Generate concise 1-2 sentence plain-language macro summaries per cluster."""
    summaries = {}
    for _, row in minimap_summary_df.iterrows():
        cluster_id = int(row['cluster_label'])
        features = [str(row[c]) for c in ["top_feature_1", "top_feature_2", "top_feature_3"] if c in row and pd.notna(row[c])]
        
        if features:
            f_str = ", ".join(features)
            text = f"This map state is characterized by heavy spatial presence around {f_str}. In macro terms, this playstyle dictates specific objective pacing and map control phases."
        else:
            text = "This state exhibits mixed spatial traits, indicating a transitional or standard map control phase."
        
        summaries[cluster_id] = text
    return summaries


def save_comparison_outputs(
    minimap_summary_df: pd.DataFrame,
    macro_summary_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Save comparison summary tables to CSV files, and export a Python macro summary mapping for the GUI.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    minimap_path = output_dir / "minimap_cluster_comparison_summary.csv"
    macro_path = output_dir / "macro_style_comparison_summary.csv"
    notes_path = output_dir / "qualitative_comparison_notes.csv"

    minimap_summary_df.to_csv(minimap_path, index=False)
    macro_summary_df.to_csv(macro_path, index=False)
    notes_df.to_csv(notes_path, index=False)
    
    # Export python mapping for GUI
    summaries = generate_per_cluster_macro_summaries(minimap_summary_df)
    py_out = Path("src/comparison/cluster_macro_summaries.py")
    py_out.parent.mkdir(parents=True, exist_ok=True)
    with py_out.open("w", encoding="utf-8") as f:
        f.write('"""Auto-generated macro summaries for GUI."""\n\n')
        f.write("CLUSTER_MACRO_SUMMARIES = {\n")
        for cid, text in summaries.items():
            f.write(f'    {cid}: "{text}",\n')
        f.write("}\n")

    LOGGER.info("Saved comparison outputs to %s and Python mapping to %s", output_dir, py_out)

    return {
        "minimap_cluster_comparison_summary": minimap_path,
        "macro_style_comparison_summary": macro_path,
        "qualitative_comparison_notes": notes_path,
    }