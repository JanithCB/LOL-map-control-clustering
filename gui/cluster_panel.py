from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.data_loader import ClusterInfo, get_cluster_display_name


class ClusterPanel(QWidget):
    """Cluster selector and quick-detail panel for labeled clustering results."""

    clusterChanged = pyqtSignal(int)

    def __init__(
        self,
        cluster_infos: Dict[int, ClusterInfo],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.cluster_infos = dict(sorted(cluster_infos.items(), key=lambda item: item[0]))

        self.title_label = QLabel("Clusters", self)
        self.list_widget = QListWidget(self)
        self.selected_label = QLabel("Label: —", self)
        self.selected_size = QLabel("Size: —", self)
        self.selected_pct = QLabel("Percentage: —", self)
        self.selected_description = QLabel("Description: —", self)
        self.selected_features = QLabel("Top features: —", self)

        self._build_ui()
        self._populate_clusters()
        self._connect_signals()

    def _build_ui(self) -> None:
        """Create the cluster panel layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        self.selected_description.setWordWrap(True)
        self.selected_features.setWordWrap(True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.selected_label)
        layout.addWidget(self.selected_size)
        layout.addWidget(self.selected_pct)
        layout.addWidget(self.selected_description)
        layout.addWidget(self.selected_features)

        self.setLayout(layout)

    def _populate_clusters(self) -> None:
        """Populate the cluster list from preloaded cluster info objects."""
        self.list_widget.clear()

        for cluster_id, cluster_info in self.cluster_infos.items():
            display_name = get_cluster_display_name(cluster_info)
            item = QListWidgetItem(display_name)
            item.setData(Qt.UserRole, cluster_id)
            self.list_widget.addItem(item)

    def _connect_signals(self) -> None:
        """Connect selection events to the details panel and external signal."""
        self.list_widget.currentItemChanged.connect(self._on_current_item_changed)

    def _format_size(self, value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _format_pct(self, value: object) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return str(value)

    def _on_current_item_changed(
        self,
        current: Optional[QListWidgetItem],
        previous: Optional[QListWidgetItem],
    ) -> None:
        """Update quick stats and emit the selected cluster ID."""
        del previous

        if current is None:
            self._clear_details()
            return

        cluster_id = current.data(Qt.UserRole)
        if cluster_id is None:
            self._clear_details()
            return

        cluster_info = self.cluster_infos.get(int(cluster_id))
        if cluster_info is None:
            self._clear_details()
            return

        self._update_details(cluster_info)
        self.clusterChanged.emit(int(cluster_id))

    def _update_details(self, cluster_info: ClusterInfo) -> None:
        """Fill the quick details area for the selected cluster."""
        label_text = cluster_info.label.strip() if cluster_info.label else f"Cluster {cluster_info.cluster_id}"
        description_text = cluster_info.description.strip() if cluster_info.description else "—"

        features = cluster_info.top_features[:5] if cluster_info.top_features else []
        features_text = ", ".join(features) if features else "—"

        self.selected_label.setText(f"Label: {label_text}")
        self.selected_size.setText(f"Size: {self._format_size(cluster_info.size)}")
        self.selected_pct.setText(f"Percentage: {self._format_pct(cluster_info.pct)}")
        self.selected_description.setText(f"Description: {description_text}")
        self.selected_features.setText(f"Top features: {features_text}")

    def _clear_details(self) -> None:
        """Reset the quick details section."""
        self.selected_label.setText("Label: —")
        self.selected_size.setText("Size: —")
        self.selected_pct.setText("Percentage: —")
        self.selected_description.setText("Description: —")
        self.selected_features.setText("Top features: —")

    def select_first_cluster(self) -> None:
        """Select the first cluster if one exists."""
        if self.list_widget.count() > 0 and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)

    def set_cluster(self, cluster_id: int) -> None:
        """Programmatically select a cluster by ID."""
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.data(Qt.UserRole) == cluster_id:
                self.list_widget.setCurrentItem(item)
                return