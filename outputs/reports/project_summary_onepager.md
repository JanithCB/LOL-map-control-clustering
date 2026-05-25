# League of Legends Macro Playstyle Clustering
**Project Summary & Technical One-Pager**

## 🎯 The Problem
In competitive *League of Legends*, high-level strategy (macro) dictates game outcomes, yet it is notoriously difficult to quantify. Playstyles like "1-3-1 Split Push," "Baron Dance," or "5-man Grouping" rely on spatial positioning and team coordination. Traditional box-score metrics fail to capture these spatial dynamics, leaving analysts and coaches without automated ways to classify map-control states.

## 💡 The Solution
This project introduces an **end-to-end ML pipeline** that extracts player positions from raw minimap screenshots, engineers domain-specific spatial features, and uses **unsupervised clustering** to automatically discover distinct map-control states. The findings are surfaced in a custom, interactive **PyQt desktop application** that links these abstract clusters back to tangible match footage and ranked game outcomes.

## 🛠️ Pipeline & Methods
1. **Computer Vision Extraction**: Utilized YOLO object detection to parse raw minimap images, outputting exact `(x, y)` coordinates for champions, objectives, and structures.
2. **Spatial Feature Engineering**: Mapped raw coordinates to strategic zones (lanes, jungle, river, bases) to calculate advanced metrics such as grouping density, spatial spread, and objective control.
3. **Unsupervised Learning**: Applied and evaluated multiple clustering algorithms (K-Means, Gaussian Mixture Models, HDBSCAN) using Silhouette Score and Davies-Bouldin indexes to find the optimal map-control groupings.
4. **Automated Explainability**: Developed logic to compare cluster centroids against the global mean, automatically generating plain-language summaries of what makes each cluster unique.
5. **Macro Correlation**: Joined the snapshot-level clusters with high-level ranked match metadata to determine how specific map states correlate with game pacing, win rates, and objective secures.

## 🚀 Key Outputs & Capabilities
- **Desktop Exploration UI**: A fully-featured PyQt5 application allowing stakeholders to visually explore clusters, review representative match thumbnails, and read auto-generated tactical summaries.
- **2D UMAP/t-SNE Projections**: High-dimensional map states are projected into interactive 2D scatter plots, visually verifying the structural integrity of the clusters.
- **Explainability Reports**: Clear, concise reporting detailing the most distinguishing spatial features for every identified strategy.
- **Strict Reproducibility**: The entire pipeline is orchestrated via a unified `pipeline_config.yaml` and `run_all.py`, backed by an automated Pytest suite.

## 🧠 Demonstrated Software Engineering & Data Science Techniques
- **End-to-End Orchestration**: Managing data flows from raw images to final UI assets.
- **Advanced Unsupervised Evaluation**: Moving beyond basic clustering by implementing rigorous metric evaluations across multiple algorithms.
- **Explainable AI (XAI)**: Demystifying black-box clusters into human-readable narratives.
- **Desktop Software Development**: Building responsive GUI applications (PyQt) separated from the core ML logic.
- **CI/CD Readiness**: Packaged via PyInstaller with integrated unit testing.
