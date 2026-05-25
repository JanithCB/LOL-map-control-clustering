# Path: gui/projection_canvas.py
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ProjectionCanvas(FigureCanvas):
    """Embeds a 2D matplotlib scatter plot inside PyQt5 that highlights the selected cluster."""
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.proj_df = None
        self._load_data()
        
    def _load_data(self):
        try:
            path = Path("outputs/reports/clustering/cluster_projection_umap.csv")
            if not path.exists():
                path = Path("outputs/reports/clustering/cluster_projection_tsne.csv")
            if path.exists():
                self.proj_df = pd.read_csv(path)
        except Exception:
            self.proj_df = None
            
    def update_cluster(self, selected_cluster_id: int = None):
        self.ax.clear()
        
        if self.proj_df is None or self.proj_df.empty:
            self.ax.text(0.5, 0.5, "2D cluster projection not generated yet.\nRun 'python run_projection.py'", 
                         ha="center", va="center", color="#666666", fontsize=9)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.draw()
            return
            
        labels = self.proj_df["cluster_label"].to_numpy()
        
        if selected_cluster_id is None:
            # Show all normally if nothing is selected
            self.ax.scatter(self.proj_df["proj_x"], self.proj_df["proj_y"], s=10, alpha=0.5, c=labels, cmap="tab10", edgecolors="none")
            title_suffix = ""
        else:
            # Highlight the selected cluster, mute others
            mask_other = labels != selected_cluster_id
            mask_sel = labels == selected_cluster_id
            
            self.ax.scatter(self.proj_df.loc[mask_other, "proj_x"], self.proj_df.loc[mask_other, "proj_y"], 
                            s=8, alpha=0.15, color="gray", edgecolors="none")
            self.ax.scatter(self.proj_df.loc[mask_sel, "proj_x"], self.proj_df.loc[mask_sel, "proj_y"], 
                            s=20, alpha=0.9, color="#0AC8B9", edgecolors="none")
            title_suffix = f" (Cluster {selected_cluster_id} Highlighted)"
                            
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        method = self.proj_df["method"].iloc[0] if "method" in self.proj_df.columns else "2D"
        self.ax.set_title(f"{method} Projection{title_suffix}")
        
        # Consistent styling with the rest of the dark UI
        self.fig.patch.set_facecolor('#010a13')
        self.ax.set_facecolor('#010a13')
        self.ax.title.set_color('#F0E6D2')
        self.ax.title.set_fontsize(10)
        for spine in self.ax.spines.values():
            spine.set_color('#1e2328')
            
        self.fig.tight_layout()
        self.draw()
