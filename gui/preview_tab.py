# Path: gui/preview_tab.py
# Summary: Preview tab for selected cluster thumbnails, summaries, and feature charts.
# Description: Displays a grid of representative sample thumbnails and a centroid feature bar chart.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QGridLayout,
    QDialog,
)

from gui.data_loader import AppData, ClusterInfo, RepresentativeSample, get_cluster_display_name
from gui.feature_chart import FeatureBarChart


class ClickableLabel(QLabel):
    def __init__(self, sample: RepresentativeSample, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._show_enlarged_dialog()

    def _show_enlarged_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Representative Sample")
        dialog.setMinimumSize(400, 400)
        layout = QVBoxLayout(dialog)
        
        if self.sample.image_path:
            img_label = QLabel()
            pixmap = QPixmap(self.sample.image_path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                img_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(img_label)
            
        meta_label = QLabel()
        meta_text = []
        if self.sample.image_path:
            meta_text.append(f"Image Path: {self.sample.image_path}")
        if self.sample.match_id:
            meta_text.append(f"Match ID: {self.sample.match_id}")
        if self.sample.frame_id:
            meta_text.append(f"Frame ID: {self.sample.frame_id}")
        if self.sample.timestamp:
            meta_text.append(f"Timestamp: {self.sample.timestamp}")
            
        if meta_text:
            meta_label.setText("\n".join(meta_text))
        else:
            meta_label.setText(self.sample.raw_summary or "No metadata available.")
            
        meta_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(meta_label)
        dialog.exec_()


class PreviewTab(QWidget):
    """Preview tab for selected cluster summaries, thumbnails, and centroid charts."""

    def __init__(
        self,
        app_data: AppData,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.app_data = app_data
        self.current_cluster_id: Optional[int] = None

        self.title_label = QLabel("Cluster preview", self)
        self.cluster_name_label = QLabel("Selected cluster: —", self)
        self.description_label = QLabel("Description: —", self)
        self.top_features_label = QLabel("Top features: —", self)
        
        self.feature_chart = FeatureBarChart(self)

        self.samples_title_label = QLabel("Representative samples", self)
        
        # Thumbnail grid setup
        self.thumbnail_scroll = QScrollArea(self)
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_container = QWidget()
        self.thumbnail_grid = QGridLayout(self.thumbnail_container)
        self.thumbnail_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the preview tab layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.description_label.setWordWrap(True)
        self.top_features_label.setWordWrap(True)

        top_layout = QVBoxLayout()
        top_layout.addWidget(self.title_label)
        top_layout.addWidget(self.cluster_name_label)
        top_layout.addWidget(self.description_label)
        top_layout.addWidget(self.top_features_label)
        
        # Add chart below the text
        top_layout.addWidget(self.feature_chart)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.samples_title_label)
        main_layout.addWidget(self.thumbnail_scroll)

        self.setLayout(main_layout)

    def set_cluster(self, cluster_id: int) -> None:
        """Update the preview content for the selected cluster."""
        self.current_cluster_id = cluster_id
        cluster_info = self.app_data.cluster_infos.get(cluster_id)

        if cluster_info is None:
            self._clear_view()
            return

        self._populate_view(cluster_info)

    def update_cluster(self, cluster_id: int) -> None:
        """Alias for compatibility with other calling styles."""
        self.set_cluster(cluster_id)

    def _populate_view(self, cluster_info: ClusterInfo) -> None:
        """Render cluster metadata, feature chart, and representative sample thumbnails."""
        self.cluster_name_label.setText(
            f"Selected cluster: {get_cluster_display_name(cluster_info)}"
        )

        description = cluster_info.description.strip() if cluster_info.description else "—"
        self.description_label.setText(f"Description: {description}")

        feature_subset = cluster_info.top_features[:5] if cluster_info.top_features else []
        features_text = ", ".join(feature_subset) if feature_subset else "—"
        self.top_features_label.setText(f"Top features: {features_text}")
        
        # Update feature chart
        self.feature_chart.update_features(feature_subset)

        self._rebuild_thumbnail_grid(cluster_info.representative_samples or [])

    def _rebuild_thumbnail_grid(self, samples: list[RepresentativeSample]) -> None:
        """Clear and rebuild the thumbnail grid for the given samples."""
        # Clear existing items
        while self.thumbnail_grid.count():
            child = self.thumbnail_grid.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not samples:
            no_samples_label = QLabel("No representative samples available.")
            self.thumbnail_grid.addWidget(no_samples_label, 0, 0)
            return

        columns = 3
        max_samples = 6

        for i, sample in enumerate(samples[:max_samples]):
            row = i // columns
            col = i % columns
            
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setAlignment(Qt.AlignCenter)
            
            image_label = ClickableLabel(sample, parent=container)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setToolTip("Click to enlarge")
            
            pixmap = None
            if sample.image_path:
                pixmap = QPixmap(sample.image_path)
            
            if pixmap and not pixmap.isNull():
                pixmap = pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(pixmap)
            else:
                image_label.setText(self._format_sample_metadata(sample))
                image_label.setWordWrap(True)
                image_label.setFixedSize(140, 140)
                image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9; padding: 4px;")
                image_label.setAlignment(Qt.AlignCenter)

            layout.addWidget(image_label)
            
            caption = QLabel(f"Sample {i+1}")
            caption.setAlignment(Qt.AlignCenter)
            layout.addWidget(caption)
            
            self.thumbnail_grid.addWidget(container, row, col)

    def _format_sample_metadata(self, sample: RepresentativeSample) -> str:
        """Create a readable text fallback when image is missing."""
        pieces = []
        if sample.image_path:
            pieces.append(f"Img: {sample.image_path.split('/')[-1]}")
        if sample.match_id:
            pieces.append(f"Match: {sample.match_id}")
        if sample.frame_id:
            pieces.append(f"Frame: {sample.frame_id}")
            
        if pieces:
            return "\n".join(pieces)
        if sample.raw_summary:
            return sample.raw_summary
        return "No Image"

    def _clear_view(self) -> None:
        """Reset preview content when no cluster is available."""
        self.cluster_name_label.setText("Selected cluster: —")
        self.description_label.setText("Description: —")
        self.top_features_label.setText("Top features: —")
        self.feature_chart.update_features([])
        self._rebuild_thumbnail_grid([])