# gui/widgets/chart_widget.py
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class ChartWidget(QWidget):
    """
    A reusable matplotlib chart widget with hover tooltip support.
    Can be used for bar charts, scatter plots, etc.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        
        # Dark theme styling
        self.figure.patch.set_facecolor('#010a13')
        self.axes = self.figure.add_subplot(111)
        self.axes.set_facecolor('#010a13')
        self.axes.tick_params(colors='#F0E6D2')
        for spine in self.axes.spines.values():
            spine.set_color('#1e2328')
            
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
        # Hover annotation
        self.annot = self.axes.annotate(
            "", xy=(0,0), xytext=(10,10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#091428", ec="#c89b3c", alpha=0.9),
            color="#F0E6D2",
            fontsize=9
        )
        self.annot.set_visible(False)
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        
        self.plot_elements = []
        self.hover_texts = []

    def hover(self, event):
        if event.inaxes == self.axes:
            for el, text in zip(self.plot_elements, self.hover_texts):
                cont, ind = el.contains(event)
                if cont:
                    # Depending on element type (e.g. BarContainer vs Line2D), get position
                    if hasattr(el, 'get_xy'):
                        pos = el.get_xy()
                    elif hasattr(el, 'get_xdata'):
                        x = el.get_xdata()[ind['ind'][0]]
                        y = el.get_ydata()[ind['ind'][0]]
                        pos = (x, y)
                    elif hasattr(el, 'patches'):
                        # For BarContainer, ind doesn't always cleanly map to the patch if we iterate the container.
                        # Actually we mapped elements to individual patches in plot_bar_chart.
                        pass
                        
                    if hasattr(el, 'get_x') and hasattr(el, 'get_y'): # it's a Patch (Bar)
                        x = el.get_x() + el.get_width() / 2
                        y = el.get_height()
                        self.annot.xy = (x, y)
                        
                    self.annot.set_text(text)
                    self.annot.set_visible(True)
                    self.canvas.draw_idle()
                    return
        self.annot.set_visible(False)
        self.canvas.draw_idle()

    def clear(self):
        self.axes.clear()
        self.plot_elements = []
        self.hover_texts = []
        self.annot = self.axes.annotate(
            "", xy=(0,0), xytext=(10,10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#091428", ec="#c89b3c", alpha=0.9),
            color="#F0E6D2",
            fontsize=9
        )
        self.annot.set_visible(False)
        self.axes.set_facecolor('#010a13')
        self.axes.tick_params(colors='#F0E6D2')
        for spine in self.axes.spines.values():
            spine.set_color('#1e2328')

    def plot_bar_chart(self, data: dict, title: str = "", ylabel: str = ""):
        self.clear()
        if not data:
            self.axes.text(0.5, 0.5, "No data available", 
                           ha='center', va='center', color='#F0E6D2')
            self.canvas.draw()
            return
            
        labels = list(data.keys())
        values = list(data.values())
        
        bars = self.axes.bar(labels, values, color='#c89b3c', edgecolor='#F0E6D2')
        self.axes.set_title(title, color='#c89b3c')
        self.axes.set_ylabel(ylabel, color='#F0E6D2')
        
        # Format x labels
        short_labels = [l.split('(')[0].strip() if '(' in l else l for l in labels]
        self.axes.set_xticks(range(len(labels)))
        self.axes.set_xticklabels(short_labels, rotation=45, ha='right')
        
        for bar, label, value in zip(bars, labels, values):
            self.plot_elements.append(bar)
            self.hover_texts.append(f"{label}\nValue: {value:.4f}")
            
        self.figure.tight_layout()
        self.canvas.draw()
