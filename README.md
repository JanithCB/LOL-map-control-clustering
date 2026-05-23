# League of Legends Macro Playstyle Clustering

## Project Goal
Identify and visualize macro playstyles in League of Legends by clustering minimap-based spatial behavior (lane presence, roaming, grouping, side pressure), and relate those clusters to high-level match statistics from ranked games.

## Project Structure

```
LOL ML project/
│
├── data/                          # Raw & processed data
│   ├── raw/                       # Symlinks or references to raw dataset files
│   ├── processed/                 # Cleaned/transformed data ready for features
│   └── interim/                   # Intermediate data (parsed labels, etc.)
│
├── mid_dataset/                   # Minimap dataset (Yadola) - YOLO format
│   ├── images/                    # Minimap screenshot images
│   └── labels/                    # YOLO label files (.txt)
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/                      # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── parse_yolo_labels.py   # Step 1: Parse YOLO labels → champion positions
│   │   ├── load_ranked_data.py    # Load & clean ranked games CSV
│   │   └── data_utils.py          # Shared data utilities
│   │
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── map_zones.py           # Step 2: Define map zones (lanes, jungle, river, pits)
│   │   ├── spatial_features.py    # Step 3: Compute spatial features per snapshot
│   │   └── macro_features.py      # Step 6: Extract macro stats from ranked games
│   │
│   ├── clustering/                # Clustering & analysis
│   │   ├── __init__.py
│   │   ├── cluster_snapshots.py   # Step 4: K-means / GMM / HDBSCAN on minimap features
│   │   ├── cluster_analysis.py    # Interpret clusters (centroids, samples)
│   │   └── playstyle_agg.py       # Step 5: Aggregate to game-level playstyles (optional)
│   │
│   ├── visualization/             # Plotting & visual analysis
│   │   ├── __init__.py
│   │   ├── plot_clusters.py       # t-SNE / UMAP cluster visualizations
│   │   ├── plot_minimap.py        # Draw champion positions on minimap
│   │   ├── plot_radar.py          # Radar/bar charts for cluster centroids
│   │   └── plot_macro.py          # Macro stats distribution charts
│   │
│   └── comparison/                # Linking minimap clusters ↔ ranked stats
│       ├── __init__.py
│       └── compare_clusters_macro.py  # Step 7: Qualitative/semi-quantitative comparison
│
├── gui/                           # Desktop application (PyQt5)
│   ├── __init__.py
│   ├── main_window.py             # Main application window
│   ├── cluster_panel.py           # Left panel: cluster list & stats
│   ├── preview_tab.py             # Right tab: minimap samples & centroid charts
│   ├── macro_tab.py               # Right tab: macro context from ranked data
│   ├── widgets/                   # Reusable custom widgets
│   │   ├── __init__.py
│   │   ├── minimap_viewer.py      # Widget to display minimap images
│   │   └── chart_widget.py        # Embedded matplotlib chart widget
│   └── resources/                 # Icons, stylesheets, etc.
│       └── style.qss              # Qt stylesheet
│
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_macro_comparison.ipynb
│
├── configs/                       # Configuration files
│   ├── map_zones.yaml             # Zone definitions (coordinates, polygons)
│   ├── clustering_config.yaml     # Clustering hyperparameters
│   └── app_config.yaml            # GUI app settings
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_parse_labels.py
│   ├── test_spatial_features.py
│   ├── test_clustering.py
│   └── test_macro_features.py
│
├── outputs/                       # Generated outputs
│   ├── figures/                   # Saved plots and visualizations
│   ├── models/                    # Saved clustering models (pickle, joblib)
│   └── reports/                   # Summary reports, tables
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup (optional)
├── run_pipeline.py                # Main script: runs full pipeline end-to-end
├── run_gui.py                     # Entry point to launch the desktop app
└── README.md                      # This file
```

## Pipeline Steps
1. **Parse YOLO labels** → champion (x, y) positions per snapshot
2. **Define map zones** → lanes, jungle, river, objective pits
3. **Compute spatial features** → lane presence, roaming, grouping, side pressure
4. **Cluster snapshots** → k-means / GMM / HDBSCAN on spatial features
5. **(Optional) Aggregate to playstyles** → game-level cluster distributions
6. **Extract macro stats** → objectives, pace, outcome from ranked data
7. **Compare clusters ↔ macro** → qualitative/semi-quantitative linking

## Datasets
- **Minimap Dataset (Yadola)**: `mid_dataset/` — images + YOLO labels
- **Ranked Games (EUW)**: `games.csv` — ~50k matches
- **Training Data (NA, optional)**: `training_data.csv` — 10k matches, 700+ features
- **Champion Info**: `champion_info.json`, `champion_info_2.json`
- **Summoner Spells**: `summoner_spell_info.json`
- **YOLO Weights**: `yolov3_LoL_champions.weights`
- **Class Names**: `mid.names`
