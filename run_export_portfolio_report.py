# Execution command: python run_export_portfolio_report.py
import logging
from pathlib import Path
import pandas as pd

from src.reporting.export_portfolio_report import generate_markdown_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def safe_read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def main():
    base_dir = Path("outputs/reports/clustering")
    eval_path = base_dir / "clustering_evaluation.csv"
    explain_path = base_dir / "cluster_explainability.csv"
    story_path = base_dir / "cluster_macro_story.csv"
    
    logging.info("Loading analysis outputs...")
    eval_df = safe_read_csv(eval_path)
    explain_df = safe_read_csv(explain_path)
    story_df = safe_read_csv(story_path)
    
    if story_df.empty:
        logging.warning("No macro story data found. Report may be incomplete.")
        
    logging.info("Generating markdown portfolio report...")
    md_content = generate_markdown_report(eval_df, explain_df, story_df)
    
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "portfolio_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logging.info(f"Successfully generated portfolio report at: {report_path}")

if __name__ == "__main__":
    main()
