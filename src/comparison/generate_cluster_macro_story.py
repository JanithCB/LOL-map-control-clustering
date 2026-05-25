import pandas as pd

def generate_macro_stories(
    explainability_df: pd.DataFrame, 
    macro_df: pd.DataFrame = None, 
    linkage_df: pd.DataFrame = None
) -> list[dict]:
    """
    Generates narrative summaries for each cluster based on explainability and macro data.
    """
    stories = []
    
    for _, row in explainability_df.iterrows():
        cluster_id = row['cluster_label']
        algo = row.get('algorithm', 'unknown')
        size = row['size']
        summary = row.get('summary', '')
        
        top_pos = row.get('top_positive', '')
        
        if 'Baron' in top_pos or 'Dragon' in top_pos or 'RiftHerald' in top_pos:
            tendency = "Objective-focused map control"
        elif 'Jungle' in top_pos or 'Camp' in top_pos:
            tendency = "Jungle invasion and resource denial"
        elif 'Turret' in top_pos or 'Inhibitor' in top_pos:
            tendency = "Aggressive pushing and siege"
        else:
            tendency = "Distributed or balanced map presence"
            
        story = (
            f"Cluster {cluster_id} represents {size} similar game states. "
            f"Based on the prominent features, the likely macro tendency is '{tendency}'. "
            f"Context: {summary} "
        )
        
        caution = "Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies."
        
        stories.append({
            "algorithm": algo,
            "cluster_label": cluster_id,
            "likely_macro_tendency": tendency,
            "narrative_story": story,
            "caution": caution
        })
        
    return stories
