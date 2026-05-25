# Execution command: python run_cluster_macro_story.py
import logging
from pathlib import Path
import pandas as pd

from src.comparison.generate_cluster_macro_story import generate_macro_stories

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    input_path = Path("outputs/reports/clustering/cluster_explainability.csv")
    out_dir = Path("outputs/reports/clustering")
    out_path = out_dir / "cluster_macro_story.csv"
    
    if not input_path.exists():
        logging.error(f"Explainability data not found at {input_path}")
        return
        
    logging.info(f"Loading explainability data from {input_path}")
    explainability_df = pd.read_csv(input_path)
    
    logging.info("Generating macro stories...")
    stories = generate_macro_stories(explainability_df)
    
    story_df = pd.DataFrame(stories)
    out_dir.mkdir(parents=True, exist_ok=True)
    story_df.to_csv(out_path, index=False)
    
    logging.info(f"Saved {len(story_df)} narrative summaries to {out_path}")

if __name__ == "__main__":
    main()
