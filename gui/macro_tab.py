# Path: gui/macro_tab.py
# Summary: Qualitative macro analysis and comparison context.
# Description: Shows cluster-specific notes, global notes, summary bullets, and a quantitative comparison table.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView

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
        
        self.summary_label = QLabel("This panel compares macro context and qualitative notes for the selected cluster.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 8px;")

        self.bullets_label = QLabel("", self)
        self.bullets_label.setWordWrap(True)

        self.notes_heading = QLabel("Notes", self)
        self.notes_text = QTextEdit(self)

        self.table_heading = QLabel("Quantitative Comparison", self)
        self.table_heading.setStyleSheet("font-weight: bold; margin-top: 8px;")
        self.comparison_table = QTableWidget(self)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the tab layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.cluster_label.setWordWrap(True)
        self.notes_heading.setWordWrap(True)
        self.notes_heading.setStyleSheet("font-weight: bold; margin-top: 8px;")

        self.notes_text.setReadOnly(True)
        self.notes_text.setPlaceholderText("No macro notes available.")

        layout.addWidget(self.title_label)
        layout.addWidget(self.cluster_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.bullets_label)
        layout.addWidget(self.notes_heading)
        layout.addWidget(self.notes_text)
        
        if self.app_data.quantitative_comparison is not None:
            layout.addWidget(self.table_heading)
            layout.addWidget(self.comparison_table)
            self._populate_table()
        else:
            self.table_heading.hide()
            self.comparison_table.hide()

        self.setLayout(layout)

    def _populate_table(self):
        """Populate the quantitative comparison table if data is available."""
        df = self.app_data.quantitative_comparison
        if df is None or df.empty:
            return
            
        self.comparison_table.setRowCount(len(df))
        self.comparison_table.setColumnCount(len(df.columns))
        self.comparison_table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                self.comparison_table.setItem(i, j, item)

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
        
        bullets = cluster_info.summary_bullets if cluster_info.summary_bullets else self.app_data.global_summary_bullets
        if bullets:
            self.bullets_label.setText("\n".join(bullets))
            self.bullets_label.show()
        else:
            self.bullets_label.hide()

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
        self.bullets_label.hide()
        self.notes_heading.setText("Notes")
        self.notes_text.setPlainText("No qualitative macro notes available.")