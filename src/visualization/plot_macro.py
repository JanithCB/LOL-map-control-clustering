# src/visualization/plot_macro.py

import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)

def generate_macro_summary_charts(macro_features_path: str | Path, output_dir: str | Path):
    """
    Generates summary charts for objective stats and game pacing 
    from the ranked macro features data.
    """
    macro_features_path = Path(macro_features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not macro_features_path.exists():
        LOGGER.error(f"Cannot find macro features data at {macro_features_path}")
        return []
        
    df = pd.read_csv(macro_features_path)
    generated_files = []
    
    # 1. Game Duration (Pacing) Chart
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["game_duration_minutes"], bins=30, kde=True, ax=ax, color="#c89b3c")
    ax.set_title("Distribution of Game Duration")
    ax.set_xlabel("Game Duration (Minutes)")
    ax.set_ylabel("Number of Games")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    pacing_path = output_dir / "macro_pacing_chart.png"
    fig.savefig(pacing_path, dpi=150)
    plt.close(fig)
    generated_files.append(pacing_path)
    
    # 2. Objective Counts Chart (Dragon Focus)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="t1_dragons", data=df, ax=ax, color="#0AC8B9")
    ax.set_title("Team Dragon Count Distribution")
    ax.set_xlabel("Number of Dragons Taken by Team")
    ax.set_ylabel("Number of Games")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    obj_path = output_dir / "macro_objective_chart.png"
    fig.savefig(obj_path, dpi=150)
    plt.close(fig)
    generated_files.append(obj_path)
    
    LOGGER.info(f"Generated macro summary charts: {[str(p) for p in generated_files]}")
    return generated_files

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_macro_summary_charts(
        "outputs/reports/macro/ranked_macro_features.csv",
        "outputs/figures"
    )
