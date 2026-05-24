
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from gui.data_loader import AppData, ClusterInfo, get_cluster_display_name


class MacroTab(QWidget):
    """Tab for qualitative macro notes and comparison context."""

    def __init__(
        self,
        app_data: AppData,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.app_data = app_data
        self.current_cluster_id: Optional[int] = None

        self.title_label = QLabel("Macro context", self)
        self.cluster_label = QLabel("Selected cluster: —", self)
        self.notes_heading = QLabel("Notes", self)
        self.notes_text = QTextEdit(self)

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the tab layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.cluster_label.setWordWrap(True)
        self.notes_heading.setWordWrap(True)

        self.notes_text.setReadOnly(True)
        self.notes_text.setPlaceholderText("No macro notes available.")

        layout.addWidget(self.title_label)
        layout.addWidget(self.cluster_label)
        layout.addWidget(self.notes_heading)
        layout.addWidget(self.notes_text)

        self.setLayout(layout)

    def set_cluster(self, cluster_id: int) -> None:
        """Update the macro notes panel for the selected cluster."""
        self.current_cluster_id = cluster_id
        cluster_info = self.app_data.cluster_infos.get(cluster_id)

        if cluster_info is None:
            self._clear_view()
            return

        self._populate_view(cluster_info)

    def _populate_view(self, cluster_info: ClusterInfo) -> None:
        """Show cluster-specific notes or fall back to global comparison notes."""
        self.cluster_label.setText(
            f"Selected cluster: {get_cluster_display_name(cluster_info)}"
        )

        cluster_notes = cluster_info.notes.strip() if cluster_info.notes else ""
        global_notes = self.app_data.global_notes.strip() if self.app_data.global_notes else ""

        if cluster_notes:
            self.notes_heading.setText("Cluster-specific notes")
            self.notes_text.setPlainText(cluster_notes)
            return

        if global_notes:
            self.notes_heading.setText("Global comparison notes")
            self.notes_text.setPlainText(global_notes)
            return

        self.notes_heading.setText("Notes")
        self.notes_text.setPlainText("No qualitative macro notes available for this cluster.")

    def _clear_view(self) -> None:
        """Reset the tab when the selected cluster is unavailable."""
        self.cluster_label.setText("Selected cluster: —")
        self.notes_heading.setText("Notes")
        self.notes_text.setPlainText("No qualitative macro notes available.")