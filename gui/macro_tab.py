# Path: gui/macro_tab.py
# Summary: Qualitative macro analysis and comparison context.
# Description: Shows cluster-specific notes, global notes, summary bullets, and a quantitative comparison table.

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QComboBox

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
        
        self.view_filter = QComboBox(self)
        self.view_filter.addItems(["All Views", "Narrative View", "Objective Charts", "Comparison Table"])
        self.view_filter.currentIndexChanged.connect(self._on_view_changed)
        self.view_filter.setStyleSheet("background-color: #010a13; color: #F0E6D2; border: 1px solid #1e2328; padding: 2px;")
        
        self.cluster_label = QLabel("Selected cluster: —", self)
        
        self.narrative_heading = QLabel("Narrative Summary", self)
        self.narrative_heading.setStyleSheet("font-size: 15px; color: #c89b3c; margin-top: 16px; margin-bottom: 8px; border-bottom: 1px solid #1e2328; padding-bottom: 4px;")
        self.summary_label = QLabel("This panel compares macro context and qualitative notes for the selected cluster.", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #666; font-style: italic; margin-bottom: 8px;")

        self.bullets_label = QLabel("", self)
        self.bullets_label.setWordWrap(True)

        self.notes_heading = QLabel("Notes", self)
        self.notes_text = QTextEdit(self)

        self.table_heading = QLabel("Quantitative Comparison", self)
        self.table_heading.setStyleSheet("font-size: 15px; color: #c89b3c; margin-top: 16px; margin-bottom: 8px; border-bottom: 1px solid #1e2328; padding-bottom: 4px;")
        self.comparison_table = QTableWidget(self)
        self.comparison_table.setMaximumHeight(180)  # Reduce visual dominance
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setStyleSheet("""
            QTableWidget {
                background-color: #010a13;
                color: #F0E6D2;
                gridline-color: #1e2328;
                border: 1px solid #1e2328;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #091428;
                color: #c89b3c;
                font-weight: bold;
                padding: 6px;
                border: 1px solid #1e2328;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)

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
        self.notes_text.setStyleSheet("""
            QTextEdit {
                background-color: #091428; 
                color: #F0E6D2;
                border: 1px solid #1e2328;
                border-radius: 4px;
                padding: 12px;
            }
        """)
        self.notes_text.setPlaceholderText("No macro notes available.")

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()
        top_layout.addWidget(QLabel("View:"))
        top_layout.addWidget(self.view_filter)
        
        layout.addLayout(top_layout)
        layout.addWidget(self.cluster_label)
        
        self.narrative_widget = QWidget()
        narrative_layout = QVBoxLayout(self.narrative_widget)
        narrative_layout.setContentsMargins(0, 0, 0, 0)
        narrative_layout.addWidget(self.narrative_heading)
        narrative_layout.addWidget(self.summary_label)
        narrative_layout.addWidget(self.bullets_label)
        layout.addWidget(self.narrative_widget)
        
        # Load charts side-by-side
        from pathlib import Path
        
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(8)
        
        pacing_label = QLabel()
        pacing_label.setAlignment(Qt.AlignCenter)
        pacing_label.setStyleSheet("background-color: #010a13; border: 1px solid #1e2328; border-radius: 4px; padding: 4px;")
        pacing_path = Path("outputs/figures/macro_pacing_chart.png")
        if pacing_path.exists():
            pixmap = QPixmap(str(pacing_path))
            pacing_label.setPixmap(pixmap.scaled(450, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            pacing_label.setText("<i>Pacing chart not generated yet.</i>")
            
        obj_label = QLabel()
        obj_label.setAlignment(Qt.AlignCenter)
        obj_label.setStyleSheet("background-color: #010a13; border: 1px solid #1e2328; border-radius: 4px; padding: 4px;")
        obj_path = Path("outputs/figures/macro_objective_chart.png")
        if obj_path.exists():
            pixmap = QPixmap(str(obj_path))
            obj_label.setPixmap(pixmap.scaled(450, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            obj_label.setText("<i>Objective chart not generated yet.</i>")
            
        charts_layout.addWidget(pacing_label)
        charts_layout.addWidget(obj_label)

        self.charts_widget = QWidget()
        self.charts_widget.setLayout(charts_layout)
        layout.addWidget(self.charts_widget)

        self.notes_widget = QWidget()
        notes_layout = QVBoxLayout(self.notes_widget)
        notes_layout.setContentsMargins(0, 0, 0, 0)
        notes_layout.addWidget(self.notes_heading)
        notes_layout.addWidget(self.notes_text)
        layout.addWidget(self.notes_widget)
        
        if self.app_data.quantitative_comparison is not None:
            layout.addWidget(self.table_heading)
            layout.addWidget(self.comparison_table)
            self._populate_table()
        else:
            self.table_heading.hide()
            self.comparison_table.hide()

        self.setLayout(layout)

    def _on_view_changed(self, index: int) -> None:
        text = self.view_filter.currentText()
        show_all = text == "All Views"
        
        self.narrative_widget.setVisible(show_all or text == "Narrative View")
        self.notes_widget.setVisible(show_all or text == "Narrative View")
        self.charts_widget.setVisible(show_all or text == "Objective Charts")
        
        has_table = self.app_data.quantitative_comparison is not None
        self.table_heading.setVisible(has_table and (show_all or text == "Comparison Table"))
        self.comparison_table.setVisible(has_table and (show_all or text == "Comparison Table"))

    def _populate_table(self):
        """Populate the quantitative comparison table if data is available."""
        df = self.app_data.quantitative_comparison
        if df is None or df.empty:
            return
            
        self.comparison_table.setRowCount(len(df))
        self.comparison_table.setColumnCount(len(df.columns))
        
        # Rename columns to short readable titles
        col_mapping = {
            "cluster_label": "Cluster",
            "cluster_id": "Cluster",
            "metric": "Category",
            "feature": "Category",
            "value": "Value",
            "n_samples": "Count",
            "proportion": "Proportion",
            "pct": "Proportion"
        }
        headers = [col_mapping.get(str(c).lower(), str(c).replace("_", " ").title()) for c in df.columns]
        self.comparison_table.setHorizontalHeaderLabels(headers)
        
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                val = row[col]
                try:
                    float_val = float(val)
                    if str(col).lower() in ["proportion", "pct"]:
                        formatted_val = f"{float_val:.2f}"
                    elif float_val.is_integer():
                        formatted_val = str(int(float_val))
                    else:
                        formatted_val = f"{float_val:.2f}"
                except (ValueError, TypeError):
                    formatted_val = str(val)
                
                item = QTableWidgetItem(formatted_val)
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
            f"<b>Selected cluster:</b> {get_cluster_display_name(cluster_info)}"
        )

        macro_desc = cluster_info.macro_story if cluster_info.macro_story else "Qualitative interpretation."
        tendency = cluster_info.macro_tendency if cluster_info.macro_tendency else "Unknown Tendency"
        caution = cluster_info.macro_caution if cluster_info.macro_caution else ""

        summary_text = (
            f"<b>Playstyle Tendency:</b> {tendency}<br><br>"
            f"<b>Narrative:</b> {macro_desc}"
        )
        if caution:
            summary_text += f"<br><br><span style='color: #c89b3c;'><i>Note: {caution}</i></span>"
            
        self.summary_label.setText(summary_text)
        self.summary_label.setStyleSheet("color: #333333; font-size: 13px; margin-bottom: 12px; line-height: 1.4;")

        cluster_notes = cluster_info.notes.strip() if cluster_info.notes else ""
        global_notes = self.app_data.global_notes.strip() if self.app_data.global_notes else ""
        
        bullets = cluster_info.summary_bullets if cluster_info.summary_bullets else []
        if bullets:
            self.bullets_label.setText("\n".join(bullets))
            self.bullets_label.show()
        else:
            self.bullets_label.hide()

        if cluster_notes:
            self.notes_heading.setText("<b>Cluster-specific Playstyle Notes</b>")
            self.notes_heading.setStyleSheet("font-size: 15px; color: #c89b3c; margin-top: 16px; margin-bottom: 8px; border-bottom: 1px solid #1e2328; padding-bottom: 4px;")
            formatted_notes = cluster_notes.replace(" || ", "<br><br><b style='color:#0AC8B9;'>Sector Notes:</b><br>").replace(" | ", "<br>• ")
            self.notes_text.setHtml(f"<div style='line-height: 1.6; font-size: 13px;'>{formatted_notes}</div>")
            return

        if global_notes:
            self.notes_heading.setText("<b>Global Comparison Notes</b>")
            self.notes_heading.setStyleSheet("font-size: 15px; color: #c89b3c; margin-top: 16px; margin-bottom: 8px; border-bottom: 1px solid #1e2328; padding-bottom: 4px;")
            formatted_notes = global_notes.replace(" || ", "<br><br>").replace(" | ", "<br>• ")
            if formatted_notes and not formatted_notes.startswith("<br>"):
                formatted_notes = "• " + formatted_notes
            self.notes_text.setHtml(f"<div style='line-height: 1.6; font-size: 13px;'>{formatted_notes}</div>")
            return

        self.notes_heading.setText("Notes")
        self.notes_text.setPlainText("No qualitative macro notes available for this cluster.")

    def _clear_view(self) -> None:
        """Reset the tab when the selected cluster is unavailable."""
        self.cluster_label.setText("Selected cluster: —")
        self.bullets_label.hide()
        self.notes_heading.setText("Notes")
        self.notes_text.setPlainText("No qualitative macro notes available.")