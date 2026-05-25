import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

LOGGER = logging.getLogger(__name__)

def evaluate_clusters(features: pd.DataFrame, labels: pd.Series, algorithm_metadata: dict = None) -> dict:
    """
    Evaluates clustering quality using silhouette score, Davies-Bouldin index, 
    and Calinski-Harabasz score.
    """
    if algorithm_metadata is None:
        algorithm_metadata = {}
        
    metrics = algorithm_metadata.copy()
    
    unique_labels = labels.unique()
    if len(unique_labels) < 2:
        LOGGER.warning("Less than 2 clusters found. Cannot compute metrics.")
        metrics.update({
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None
        })
        return metrics

    try:
        metrics["silhouette_score"] = silhouette_score(features, labels)
    except Exception as e:
        LOGGER.warning(f"Could not compute silhouette score: {e}")
        metrics["silhouette_score"] = None
        
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(features, labels)
    except Exception as e:
        LOGGER.warning(f"Could not compute Davies-Bouldin score: {e}")
        metrics["davies_bouldin_score"] = None

    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(features, labels)
    except Exception as e:
        LOGGER.warning(f"Could not compute Calinski-Harabasz score: {e}")
        metrics["calinski_harabasz_score"] = None

    return metrics

def evaluate_multiple_runs(runs_data: list) -> pd.DataFrame:
    """
    runs_data: list of dicts with 'features', 'labels', 'metadata'
    """
    results = []
    for run in runs_data:
        res = evaluate_clusters(run['features'], run['labels'], run.get('metadata', {}))
        results.append(res)
    return pd.DataFrame(results)
