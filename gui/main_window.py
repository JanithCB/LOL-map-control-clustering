
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.cluster_panel import ClusterPanel
from gui.data_loader import AppData, load_app_data
from gui.macro_tab import MacroTab
from gui.preview_tab import PreviewTab
from gui.projection_tab import ProjectionTab
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence


class MainWindow(QMainWindow):
    """Main application window for exploring labeled clustering results."""

    def __init__(self, base_dir: str | Path = ".", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.base_dir = Path(base_dir)
        self.app_data: Optional[AppData] = None

        self.setWindowTitle("LoL Map Control Cluster Explorer")
        self.resize(1280, 800)
        self.setStatusBar(QStatusBar(self))

        self.cluster_panel: Optional[ClusterPanel] = None
        self.preview_tab: Optional[PreviewTab] = None
        self.macro_tab: Optional[MacroTab] = None
        self.projection_tab: Optional[ProjectionTab] = None
        self.tabs: Optional[QTabWidget] = None

        self._build_base_ui()
        self._load_data_and_initialize()

    def _build_base_ui(self) -> None:
        """Create the shell layout first so failures can still be shown gracefully."""
        self.central = QWidget(self)
        self.setCentralWidget(self.central)

        self.central_layout = QVBoxLayout(self.central)
        self.central_layout.setContentsMargins(8, 8, 8, 8)
        self.central_layout.setSpacing(8)

        self.splitter = QSplitter(Qt.Horizontal, self.central)
        self.central_layout.addWidget(self.splitter)

        self.placeholder_left = QLabel("Loading cluster data...", self.splitter)
        self.placeholder_left.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.placeholder_left.setWordWrap(True)

        self.placeholder_right = QLabel("Preparing preview and macro panels...", self.splitter)
        self.placeholder_right.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.placeholder_right.setWordWrap(True)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.statusBar().showMessage("Ready", 3000)

    def _load_data_and_initialize(self) -> None:
        """Load app data once and wire the child widgets from that shared source."""
        try:
            self.app_data = load_app_data(self.base_dir)
        except FileNotFoundError as exc:
            self._show_load_error(
                title="Missing data files",
                message=(
                    "The cluster explorer could not find one or more required CSV outputs.\n\n"
                    f"{exc}\n\n"
                    "Run the pipeline and interpretation steps first, then reopen the GUI."
                ),
            )
            return
        except Exception as exc:
            self._show_load_error(
                title="Failed to load clustering data",
                message=(
                    "The cluster explorer could not load the clustering outputs.\n\n"
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            return

        self._build_loaded_ui()
        self._wire_signals()
        self._initialize_default_selection()

        cluster_count = len(self.app_data.cluster_infos) if self.app_data else 0
        self.statusBar().showMessage(f"Loaded {cluster_count} clusters.", 5000)

    def _build_loaded_ui(self) -> None:
        """Replace placeholders with the real cluster panel and tab widgets."""
        if self.app_data is None:
            return

        self._clear_splitter()

        self.cluster_panel = ClusterPanel(cluster_infos=self.app_data.cluster_infos, parent=self.splitter)

        self.tabs = QTabWidget(self.splitter)
        self.preview_tab = PreviewTab(app_data=self.app_data, parent=self.tabs)
        self.macro_tab = MacroTab(app_data=self.app_data, parent=self.tabs)
        self.projection_tab = ProjectionTab(app_data=self.app_data, parent=self.tabs)

        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.addTab(self.macro_tab, "Macro")
        self.tabs.addTab(self.projection_tab, "Projection")

        self.splitter.addWidget(self.cluster_panel)
        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([320, 900])

    def _wire_signals(self) -> None:
        """Connect selection changes from the cluster panel to the detail tabs."""
        if self.cluster_panel is None:
            return

        if self.preview_tab is not None:
            self.cluster_panel.clusterChanged.connect(self.preview_tab.set_cluster)

        if self.macro_tab is not None:
            self.cluster_panel.clusterChanged.connect(self.macro_tab.set_cluster)

        self.cluster_panel.clusterChanged.connect(self._on_cluster_changed)
        self.cluster_panel.algoChanged.connect(self._on_algo_changed)

    def _on_algo_changed(self, algo: str) -> None:
        self.reload_data(algo)

    def _initialize_default_selection(self) -> None:
        """Select the first cluster so the detail tabs have initial content."""
        if self.cluster_panel is None:
            return
        self.cluster_panel.select_first_cluster()

        # Add F5 reload shortcut
        self.reload_shortcut = QShortcut(QKeySequence("F5"), self)
        self.reload_shortcut.activated.connect(self.reload_data)

    def reload_data(self, algo: str = None) -> None:
        """Cleanly reload data and rebuild UI components."""
        self.statusBar().showMessage(f"Reloading {algo or 'default'} data from disk...", 2000)
        
        if algo:
            from gui import data_loader
            from pathlib import Path
            data_loader.CLUSTERING_DIR = Path("outputs") / "reports" / algo
        
        selected_id = None
        if self.cluster_panel and self.cluster_panel.list_widget.currentItem():
            selected_id = self.cluster_panel.list_widget.currentItem().data(Qt.UserRole)
            
        try:
            self.app_data = load_app_data(self.base_dir)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to reload data: {e}", 5000)
            return

        if self.cluster_panel:
            self.cluster_panel.cluster_infos = dict(sorted(self.app_data.cluster_infos.items(), key=lambda item: item[0]))
            self.cluster_panel._populate_clusters()
            if selected_id is not None:
                self.cluster_panel.set_cluster(selected_id)
            else:
                self.cluster_panel.select_first_cluster()
                
        if self.preview_tab:
            self.preview_tab.app_data = self.app_data
            if selected_id is not None:
                self.preview_tab.set_cluster(selected_id)
                
        if self.macro_tab:
            self.macro_tab.app_data = self.app_data
            self.macro_tab._populate_table()
            if selected_id is not None:
                self.macro_tab.set_cluster(selected_id)

        cluster_count = len(self.app_data.cluster_infos) if self.app_data else 0
        self.statusBar().showMessage(f"Successfully reloaded {cluster_count} clusters.", 4000)

    def _on_cluster_changed(self, cluster_id: int) -> None:
        """Update status feedback when the selected cluster changes."""
        if self.app_data is None:
            return

        cluster_info = self.app_data.cluster_infos.get(cluster_id)
        if cluster_info is None:
            self.statusBar().showMessage(f"Selected cluster {cluster_id}", 3000)
            return

        label = cluster_info.label.strip() if cluster_info.label else ""
        if label:
            self.statusBar().showMessage(
                f"Selected Cluster {cluster_id} – {label}",
                3000,
            )
        else:
            self.statusBar().showMessage(f"Selected Cluster {cluster_id}", 3000)

    def _show_load_error(self, title: str, message: str) -> None:
        """Display a friendly load failure message in both UI and dialog form."""
        self.statusBar().showMessage(title, 8000)
        self.placeholder_left.setText("Cluster data could not be loaded.")
        self.placeholder_right.setText(message)

        QMessageBox.warning(self, title, message)

    def _clear_splitter(self) -> None:
        """Remove placeholder widgets before inserting the real child widgets."""
        while self.splitter.count():
            widget = self.splitter.widget(0)
            if widget is None:
                break
            widget.setParent(None)


def main() -> None:
    """Launch the main desktop application."""
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("LoL Map Control Cluster Explorer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()