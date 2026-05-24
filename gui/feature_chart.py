# Path: gui/feature_chart.py
# Summary: Reusable PyQt widget that embeds a matplotlib bar chart.
# Description: Renders a simple horizontal bar chart of the selected cluster's top features.

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FeatureBarChart(QWidget):
    """A small widget to display a horizontal bar chart of top cluster features."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.error_label = QLabel("No feature chart available")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.error_label)
        self.setLayout(layout)

    def update_features(self, feature_strings: list[str]):
        """
        Parses a list of feature strings like 'bot_lane_count (0.8421)'
        and plots them as a horizontal bar chart.
        """
        self.ax.clear()

        names = []
        values = []

        for f_str in feature_strings:
            # Parse 'bot_lane_count (0.8421)'
            try:
                if '(' in f_str and ')' in f_str:
                    name_part, val_part = f_str.rsplit('(', 1)
                    name = name_part.strip()
                    val = float(val_part.replace(')', '').strip())
                    names.append(name)
                    values.append(val)
                else:
                    names.append(f_str)
                    values.append(0.0)
            except Exception:
                continue

        if not names or all(v == 0.0 for v in values):
            self.canvas.hide()
            self.error_label.show()
            return

        self.error_label.hide()
        self.canvas.show()

        # Reverse so highest is at the top of horizontal bar chart
        names.reverse()
        values.reverse()

        bars = self.ax.barh(names, values, color='#4C72B0')
        self.ax.set_xlabel('Centroid Value')
        self.ax.set_title('Top Features')
        self.figure.tight_layout()
        self.canvas.draw()
