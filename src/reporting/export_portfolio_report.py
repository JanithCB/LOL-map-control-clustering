import pandas as pd
from pathlib import Path
import datetime

def generate_markdown_report(
    eval_df: pd.DataFrame, 
    explain_df: pd.DataFrame, 
    story_df: pd.DataFrame,
    labels_df: pd.DataFrame = None
) -> str:
    """
    Assembles a comprehensive markdown report for the portfolio.
    """
    md = [
        "# League of Legends Macro Playstyle Clustering",
        f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d')}\n",
        "## 1. Executive Summary",
        "This report details the unsupervised learning approach used to categorize map-control patterns from League of Legends minimap snapshots. "
        "By translating spatial YOLOv3 detections into structured features, we segmented game states into distinct macro playstyles.\n"
    ]
    
    if eval_df is not None and not eval_df.empty:
        md.append("## 2. Clustering Evaluation")
        md.append("Algorithm performance based on quantitative metrics:\n")
        md.append(eval_df.to_markdown(index=False))
        md.append("\n")
        
    md.append("## 3. Discovered Playstyles (Clusters)\n")
    
    if story_df is not None and not story_df.empty:
        for _, row in story_df.iterrows():
            algo = row.get('algorithm', 'unknown')
            c_id = row['cluster_label']
            tendency = row.get('likely_macro_tendency', 'N/A')
            story = row.get('narrative_story', '')
            caution = row.get('caution', '')
            
            md.append(f"### {algo.upper()} - Cluster {c_id}")
            
            if labels_df is not None and not labels_df.empty:
                label_match = labels_df[(labels_df['cluster_id'] == c_id) & (labels_df['algorithm'] == algo)]
                if not label_match.empty:
                    label = label_match.iloc[0]['label']
                    md.append(f"**Alias**: {label}")
                    
            md.append(f"**Tendency**: {tendency}\n")
            md.append(f"{story}\n")
            md.append(f"_{caution}_\n")
            
    md.append("## 4. Visualizations & Projections")
    md.append("*(Include saved 2D projection images and feature distribution charts here)*\n")
    
    md.append("---\n*End of Report*")
    
    return "\n".join(md)
