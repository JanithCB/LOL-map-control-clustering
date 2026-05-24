
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.data_loader import AppData, ClusterInfo, RepresentativeSample, get_cluster_display_name


class PreviewTab(QWidget):
    """Preview tab for selected cluster summaries and representative sample metadata."""

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
        self.samples_title_label = QLabel("Representative samples", self)
        self.samples_list = QListWidget(self)
        self.sample_hint_label = QLabel(
            "Sample image paths will be shown here when available. "
            "For now, sample metadata is displayed as readable text.",
            self,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the preview tab layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.description_label.setWordWrap(True)
        self.top_features_label.setWordWrap(True)
        self.sample_hint_label.setWordWrap(True)
        self.sample_hint_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        layout.addWidget(self.title_label)
        layout.addWidget(self.cluster_name_label)
        layout.addWidget(self.description_label)
        layout.addWidget(self.top_features_label)
        layout.addWidget(self.samples_title_label)
        layout.addWidget(self.samples_list)
        layout.addWidget(self.sample_hint_label)

        self.setLayout(layout)

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
        """Render cluster metadata and representative sample summaries."""
        self.cluster_name_label.setText(
            f"Selected cluster: {get_cluster_display_name(cluster_info)}"
        )

        description = cluster_info.description.strip() if cluster_info.description else "—"
        self.description_label.setText(f"Description: {description}")

        feature_subset = cluster_info.top_features[:5] if cluster_info.top_features else []
        features_text = ", ".join(feature_subset) if feature_subset else "—"
        self.top_features_label.setText(f"Top features: {features_text}")

        self.samples_list.clear()
        if cluster_info.representative_samples:
            for sample in cluster_info.representative_samples:
                text = self._format_sample(sample)
                item = QListWidgetItem(text)
                self.samples_list.addItem(item)
        else:
            self.samples_list.addItem(QListWidgetItem("No representative samples available."))

    def _format_sample(self, sample: RepresentativeSample) -> str:
        """Create a readable one-line summary for a representative sample."""
        pieces = []

        if sample.image_path:
            pieces.append(f"image_path={sample.image_path}")
        if sample.match_id:
            pieces.append(f"match_id={sample.match_id}")
        if sample.frame_id:
            pieces.append(f"frame_id={sample.frame_id}")
        if sample.timestamp:
            pieces.append(f"timestamp={sample.timestamp}")

        if pieces:
            return " | ".join(pieces)

        if sample.raw_summary:
            return sample.raw_summary

        return "Representative sample"

    def _clear_view(self) -> None:
        """Reset preview content when no cluster is available."""
        self.cluster_name_label.setText("Selected cluster: —")
        self.description_label.setText("Description: —")
        self.top_features_label.setText("Top features: —")
        self.samples_list.clear()
        self.samples_list.addItem(QListWidgetItem("No representative samples available."))