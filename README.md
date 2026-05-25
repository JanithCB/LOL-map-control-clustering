# League of Legends Macro Playstyle Clustering

## Project Overview
This project identifies and visualizes macro playstyles in *League of Legends* by clustering minimap-based spatial behavior (e.g., lane presence, roaming, grouping, side pressure). It links these localized map-control clusters to high-level match statistics from ranked games to provide actionable, interpretable macro insights.

## End-to-End Pipeline
The project follows a robust data pipeline transforming raw screenshots into insights:
1. **Minimap Labels / Detections**: YOLO object detection parses champion locations from minimap screenshots into spatial coordinates.
2. **Spatial Feature Extraction**: Coordinates are mapped to map zones (lanes, jungle, river, pits) to compute presence, grouping, and pressure metrics.
3. **Unsupervised Clustering**: K-means, GMM, or HDBSCAN cluster these snapshot features into distinct spatial states (e.g., "5-man mid group" or "Deep invade").
4. **Cluster Interpretation and Labeling**: Centroid profiles and representative samples are analyzed to assign human-readable labels and descriptions.
5. **Macro Comparison Outputs**: Cluster states are qualitatively and quantitatively linked to high-level ranked game statistics (e.g., objective takes, game pace).
6. **Desktop App Exploration Layer**: A custom PyQt5 desktop application provides an intuitive interface for exploring clusters, centroids, and notes.

## Key Outputs
The pipeline generates clean interpretation artifacts in `outputs/reports/`:
- `clustering/cluster_sizes.csv`: Cluster sizes and proportions.
- `clustering/cluster_top_features.csv`: The most defining spatial features for each cluster centroid.
- `clustering/representative_samples.csv`: Real match metadata and image paths for the best examples of each cluster.
- `clustering/cluster_labels.csv`: Final human-readable cluster labels and descriptions.
- `comparison/qualitative_comparison_notes.csv`: Notes detailing how each cluster relates to macro objectives.

## Desktop App
The PyQt5 desktop application provides a polished exploration showcase:
- **Cluster Panel (Left)**: Navigate through identified clusters with size and top-line metadata. Toggle between different clustering algorithms (e.g., KMeans, GMM) and instantly see evaluation metrics like Silhouette Score.
- **Preview Tab (Right)**: View the cluster's centroid via an interactive feature bar chart. Review the cluster explainability panel ("Why this cluster is distinct") and examine an adjustable grid of representative minimap thumbnails (sorted by centroid distance).
- **Macro Tab (Right)**: Read integrated cluster-specific macro stories and global context linking the pattern to match outcomes. Filter views to see objective/pacing charts or a quantitative comparison table.
- **Projection Tab (Right)**: Explore a 2D UMAP projection of the entire feature space, visualizing how individual clusters map onto the global distribution of map control states.

## Screenshots / GUI Preview
*(Screenshots will be added here once generated)*
- **Cluster Panel**: `![Cluster Panel Placeholder]()`
- **Preview Tab (with Thumbnails and Chart)**: `![Preview Tab Placeholder]()`
- **Macro Tab (Notes and Comparison Table)**: `![Macro Tab Placeholder]()`

## Repository Structure

```
LOL ML project/
├── data/                          # Raw & processed ranked data
├── mid_dataset/                   # Minimap dataset (Yadola) - images & YOLO labels
├── src/                           # Source code (features, clustering, viz)
├── gui/                           # Desktop application (PyQt5)
├── notebooks/                     # Jupyter notebooks for exploration
├── configs/                       # Configuration files (zones, app settings)
├── outputs/                       # Generated outputs (figures, reports)
├── tests/                         # Unit tests
└── run_*.py                       # Entry points (pipeline, gui)
```

## How to Run

1. **Run the Full Pipeline**: Execute `python run_pipeline.py` to extract features and run core clustering.
2. **Multi-Algorithm Clustering**: `python run_multi_clustering.py` to run KMeans, GMM, etc.
3. **Generate Explainability & Evaluation**: 
   - `python run_cluster_explainability.py`
   - `python run_clustering_evaluation.py`
4. **Generate Macro Stories & Portfolio**: 
   - `python run_cluster_macro_story.py`
   - `python run_export_portfolio_report.py`
5. **Launch the Desktop App**: `python run_gui.py` to visually explore the generated insights and evaluate algorithm performance.

## Project Highlights
- **Computer Vision Pipeline**: Custom YOLO parsing on a noisy domain-specific dataset.
- **Domain-Specific Spatial Feature Engineering**: Transforming raw (x,y) positions into strategic League of Legends concepts like "grouping", "split pushing", and "objective control".
- **Unsupervised Learning & Evaluation**: Discovering implicit structures in player behavior using multi-algorithm approaches (KMeans, GMM, HDBSCAN) and quantitatively evaluating them with Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
- **Automated Cluster Explainability**: Dynamically generating plain-language narrative summaries of what makes a cluster distinct compared to the global average.
- **2D Projection Visualization**: Mapping complex, high-dimensional map states into intuitive 2D UMAP interactive scatter plots.
- **Replay Linkage**: Linking static screenshots back to structural match metadata to surface concrete win rates, pacing, and macro patterns.
- **Automated Portfolio Reporting**: Programmatically exporting the entire analysis pipeline into a clean, markdown-formatted portfolio report.
- **PyQt Desktop Showcase**: A robust, interactive tool built with Python and Qt to demonstrate the clustering results directly to stakeholders.
