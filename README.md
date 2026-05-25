# League of Legends Macro Playstyle Clustering

## Project Overview
This project identifies and visualizes macro playstyles in *League of Legends* by clustering minimap-based spatial behavior (e.g., lane presence, roaming, grouping, side pressure). It links these localized map-control clusters to high-level match statistics from ranked games to provide actionable, interpretable macro insights.

## End-to-End Pipeline
The project follows a robust data pipeline transforming raw screenshots into insights:
1. **Minimap Labels / Detections**: YOLO object detection parses champion locations from minimap screenshots into spatial coordinates.
2. **Spatial Feature Extraction**: Coordinates are mapped to map zones (lanes, jungle, river, pits) to compute presence, grouping, and pressure metrics.
3. **Unsupervised Clustering**: K-means, GMM, or HDBSCAN cluster these snapshot features into distinct spatial states.
4. **Cluster Interpretation and Labeling**: Centroid profiles and representative samples are analyzed to assign human-readable labels.
5. **Macro Comparison Outputs**: Cluster states are qualitatively and quantitatively linked to high-level ranked game statistics.
6. **Desktop App Exploration Layer**: A custom PyQt5 desktop application provides an intuitive interface for exploring clusters.

## Key Outputs
The pipeline generates clean interpretation artifacts in `outputs/reports/`:
- `clustering/cluster_sizes.csv`: Cluster sizes and proportions.
- `clustering/cluster_top_features.csv`: The most defining spatial features for each cluster centroid.
- `clustering/representative_samples.csv`: Real match metadata and image paths.
- `clustering/cluster_labels.csv`: Final human-readable cluster labels and descriptions.
- `comparison/qualitative_comparison_notes.csv`: Notes detailing how each cluster relates to macro objectives.

## How to Run (Full Pipeline Reproducibility)

The entire pipeline is fully orchestrated and strictly reproducible using a single command:

```bash
# Run the full end-to-end pipeline (parsing, clustering, macro linking, evaluation)
python run_all.py
```
*Note: Configuration is centrally managed in `configs/pipeline_config.yaml`.*

To validate that the pipeline outputs are structurally sound and ready for the GUI, run the test suite and validation scripts:
```bash
python run_validate_outputs.py
pytest tests/
```

## Desktop App & Packaging

The PyQt5 desktop application provides a polished exploration showcase:
- **Cluster Panel**: Navigate through identified clusters with size and top-line metadata.
- **Preview Tab**: View interactive feature bar charts, read plain-language explainability summaries, and examine representative minimap thumbnails.
- **Macro Tab**: Read integrated cluster-specific macro stories linking the pattern to match outcomes.
- **Projection Tab**: Explore a 2D t-SNE/UMAP projection of the entire feature space.

**Running the App from Source:**
```bash
python run_gui.py
```

**Packaging the App (Windows executable):**
To compile the GUI into a standalone Windows executable for distribution:
1. Ensure PyInstaller is installed (`pip install pyinstaller`)
2. Run the build script:
```powershell
.\build_desktop_app.ps1
```
The executable will be generated in `dist/League_Macro_App/`.

## Demo Flow
If you are presenting this project, a suggested flow is:
1. **Introduce the Problem**: Explain how spatial data in esports is hard to quantify.
2. **Explain the Tech Stack**: Mention YOLO, Feature Engineering, and KMeans/GMM.
3. **Open the GUI**: Show how raw math is translated into an interactive tool.
4. **Highlight a Cluster**: Click a cluster to show its defining features and real-world minimap thumbnails.
5. **Show Macro Links**: Demonstrate how the static minimap clusters correlate to game pacing and win conditions in the Macro Tab.
6. **Validate with Projections**: Show the 2D projection scatter plot to prove the mathematical separation of the clusters.

*(A full conversational demo script is available in `outputs/reports/demo_script.md`)*

## Screenshots

*(To be added after packaging)*
- **Cluster Panel & Explainability**: `![Cluster Panel Placeholder]()`
- **Representative Map States**: `![Representative Thumbnails]()`
- **Global 2D Projection Space**: `![2D Projection]()`

## Project Highlights
- **Computer Vision Pipeline**: Custom YOLO parsing on a noisy domain-specific dataset.
- **Domain-Specific Spatial Feature Engineering**: Transforming raw (x,y) positions into strategic concepts.
- **Unsupervised Learning & Evaluation**: Multi-algorithm approaches (KMeans, GMM, HDBSCAN) evaluated rigorously.
- **Automated Cluster Explainability**: Dynamically generating plain-language narrative summaries of cluster distinctions.
- **2D Projection Visualization**: Mapping complex map states into intuitive 2D scatter plots.
- **Replay Linkage**: Linking static screenshots back to structural match metadata.
- **Software Engineering**: Fully tested, PyInstaller-ready, GUI-driven applied machine learning project.
