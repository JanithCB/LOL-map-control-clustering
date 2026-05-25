# Path: gui/projection_tab.py
# Summary: Displays the 2D dimensionality reduction scatter plot of the clusters.

from __future__ import annotations
from typing import Optional
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea

from gui.data_loader import AppData

class ProjectionTab(QWidget):
    """Tab to show the overall 2D map of clusters (UMAP / t-SNE)."""

    def __init__(
        self,
        app_data: AppData,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.app_data = app_data
        
        self.title_label = QLabel("2D Cluster Projection", self)
        self.desc_label = QLabel("UMAP / t-SNE scatter plot of all minimap snapshots.", self)
        self.image_label = QLabel(self)
        self.scroll_area = QScrollArea(self)
        
        self._build_ui()
        self._load_image()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.desc_label.setStyleSheet("color: #666; margin-bottom: 8px;")
        
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #010a13; border: 1px solid #1e2328; border-radius: 4px;")
        
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.desc_label)
        layout.addWidget(self.scroll_area)
        
        self.setLayout(layout)

    def _load_image(self) -> None:
        img_path = Path("outputs/figures/cluster_projection.png")
        if img_path.exists():
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                # Scale smoothly, cap size at 1200 so it fits nicely
                scaled_pixmap = pixmap.scaled(1200, 1200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("<i>Failed to render image format.</i>")
        else:
            self.image_label.setText(
                "<i>Projection image not found.<br>"
                "Run <b>python run_projection.py</b> from the root directory to generate it.</i>"
            )
