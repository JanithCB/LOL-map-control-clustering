"""
Lightweight utility for linking minimap snapshots to ranked match IDs.
"""

import pandas as pd
from pathlib import Path
import shutil

def main():
    base_dir = Path("outputs/reports")
    clustering_dirs = [base_dir / "clustering", base_dir / "kmeans", base_dir / "gmm"]
    
    # Manual mappings (Replay ID to Match ID)
    manual_mappings = {
        "1": "3326086514"
    }

    for cdir in clustering_dirs:
        samples_path = cdir / "representative_samples.csv"
        if not samples_path.exists():
            continue
            
        df = pd.read_csv(samples_path)
        
        links = []
        for _, row in df.iterrows():
            image_id = str(row.get("image_id", ""))
            
            # Extract replay id if present. e.g. "irelia_1_00194" -> "1"
            parts = image_id.split("_")
            replay_id = None
            if len(parts) >= 3 and parts[-2].isdigit():
                replay_id = parts[-2]
                
            sample_id = image_id
            cluster_id = row.get("cluster_label", row.get("cluster_id", ""))
            
            # Apply mappings
            if replay_id in manual_mappings:
                match_id = manual_mappings[replay_id]
                method = "manual"
                confidence = "0.95"
                notes = "Confirmed mapping based on replay ID."
            else:
                match_id = ""
                method = "heuristic"
                confidence = "0.30"
                notes = "pending_manual_review"
                
            links.append({
                "sample_id": sample_id,
                "image_path": row.get("label_file", ""),
                "cluster_id": cluster_id,
                "linked_match_id": match_id,
                "link_method": method,
                "confidence": confidence,
                "notes": notes
            })
            
        links_df = pd.DataFrame(links)
        out_path = cdir / "replay_match_links.csv"
        links_df.to_csv(out_path, index=False)
        print(f"Wrote {len(links_df)} replay match links to {out_path}")

if __name__ == "__main__":
    main()
