from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_minimap_cluster_outputs(
    cluster_sizes_path: str | Path, top_features_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cluster size and top features CSVs."""
    cluster_sizes_path = Path(cluster_sizes_path)
    top_features_path = Path(top_features_path)
    
    if not cluster_sizes_path.exists():
        raise FileNotFoundError(f"Cluster sizes file not found: {cluster_sizes_path}")
    if not top_features_path.exists():
        raise FileNotFoundError(f"Top features file not found: {top_features_path}")
        
    cluster_sizes_df = pd.read_csv(cluster_sizes_path)
    top_features_df = pd.read_csv(top_features_path)
    return cluster_sizes_df, top_features_df


def load_macro_outputs(macro_features_path: str | Path) -> pd.DataFrame:
    """Load ranked macro features CSV."""
    macro_features_path = Path(macro_features_path)
    if not macro_features_path.exists():
        raise FileNotFoundError(f"Macro features file not found: {macro_features_path}")
    return pd.read_csv(macro_features_path)


def summarize_minimap_clusters_for_comparison(
    cluster_sizes_df: pd.DataFrame, top_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Create a one-row-per-cluster summary with top features."""
    summary_rows = []
    
    for _, row in cluster_sizes_df.iterrows():
        cluster_id = row.get("cluster_label", row.get("cluster_id"))
        top_feats = top_features_df[top_features_df["cluster_label"] == cluster_id].sort_values("rank")
        
        feat_names = top_feats["feature"].tolist()
        
        summary_rows.append({
            "cluster_label": cluster_id,
            "n_samples": row.get("count", row.get("n_samples", 0)),
            "proportion": row.get("proportion", 0),
            "top_feature_1": feat_names[0] if len(feat_names) > 0 else None,
            "top_feature_2": feat_names[1] if len(feat_names) > 1 else None,
            "top_feature_3": feat_names[2] if len(feat_names) > 2 else None,
        })
        
    return pd.DataFrame(summary_rows)


def summarize_macro_styles_for_comparison(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary table of macro styles."""
    style_cols = ["objective_focus", "neutral_focus", "game_length_style"]
    rows = []
    
    for col in style_cols:
        if col in macro_df.columns:
            counts = macro_df[col].value_counts()
            for val, count in counts.items():
                rows.append({
                    "category": col,
                    "value": val,
                    "count": count,
                    "proportion": count / len(macro_df)
                })
                
    return pd.DataFrame(rows)


def build_qualitative_comparison_notes(
    minimap_summary_df: pd.DataFrame, macro_summary_df: pd.DataFrame
) -> pd.DataFrame:
    """Create a report-style table of qualitative comparison notes."""
    notes = [
        {
            "comparison_axis": "Grouping/Spread Patterns vs Game Pace",
            "minimap_view": "Clusters showing tight 3-5 man groups vs spread 1-1-1-2 lanes.",
            "ranked_macro_view": "High objective taking and quick game lengths vs slow scaling.",
            "interpretation_note": "Early grouping may correlate with shorter 'snowball' games, but strict causal links require match-level alignment."
        },
        {
            "comparison_axis": "Bot-side Control vs Dragon Focus",
            "minimap_view": "Clusters with high champion counts in bot lane and bot river.",
            "ranked_macro_view": "Matches categorized as 'dragon_focused' in neutral focus.",
            "interpretation_note": "A strong bot-side minimap presence typically supports dragon-heavy match outcomes."
        },
        {
            "comparison_axis": "Lane Control vs Objective Totals",
            "minimap_view": "High lane presence vs high jungle/roaming presence.",
            "ranked_macro_view": "Objective_heavy vs objective_light matches.",
            "interpretation_note": "Matches with many objectives may feature map states with less strict laning and more roaming."
        },
        {
            "comparison_axis": "Base/Defensive Patterns vs Game Length",
            "minimap_view": "Clusters showing champions grouped near own base.",
            "ranked_macro_view": "Matches categorized as 'long' or 'very_long'.",
            "interpretation_note": "Late-game states naturally pull teams toward base defense/sieging."
        }
    ]
    return pd.DataFrame(notes)


def save_comparison_outputs(
    minimap_summary_df: pd.DataFrame, 
    macro_summary_df: pd.DataFrame, 
    comparison_notes_df: pd.DataFrame, 
    output_dir: str | Path
) -> dict[str, Path]:
    """Save the comparison output tables."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    m_path = out_dir / "minimap_cluster_comparison_summary.csv"
    minimap_summary_df.to_csv(m_path, index=False)
    paths["minimap_summary"] = m_path
    
    mac_path = out_dir / "macro_style_comparison_summary.csv"
    macro_summary_df.to_csv(mac_path, index=False)
    paths["macro_summary"] = mac_path
    
    notes_path = out_dir / "qualitative_comparison_notes.csv"
    comparison_notes_df.to_csv(notes_path, index=False)
    paths["notes"] = notes_path
    
    return paths
