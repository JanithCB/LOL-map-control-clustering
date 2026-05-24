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
- **Cluster Panel (Left)**: Navigate through identified clusters with size and top-line metadata.
- **Preview Tab (Right)**: View the cluster's centroid via a feature bar chart and examine a grid of representative minimap thumbnails.
- **Macro Tab (Right)**: Read integrated cluster-specific macro notes and global context linking the pattern to match outcomes.

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

1. **Run the Full Pipeline**: Conceptually (if enabled), `python run_pipeline.py` executes data loading, feature extraction, and clustering.
2. **Cluster Interpretation**: `python -m src.clustering.interpret_clusters --export-py` generates the readable labels and CSV outputs.
3. **Launch the Desktop App**: `python -m gui.main_window` or `python run_gui.py` to explore the generated insights visually.

## Project Highlights
- **Computer Vision Pipeline**: Custom YOLO parsing on a noisy domain-specific dataset.
- **Domain-Specific Feature Engineering**: Transforming raw (x,y) positions into strategic League of Legends concepts.
- **Unsupervised Learning**: Discovering implicit structures in player behavior without explicit labeling.
- **Qualitative Interpretation Layer**: Bridging the gap between raw math (centroids) and human insights.
- **PyQt Desktop Showcase**: A robust, portfolio-ready interactive tool to demonstrate the clustering results.
