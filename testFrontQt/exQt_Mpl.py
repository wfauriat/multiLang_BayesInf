import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFrame
)
from PyQt5.QtCore import QObject, pyqtSignal, Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class LinearRegressionModel(QObject):
    def __init__(self):
        super().__init__()
        self.X = None
        self.y = None
        self.generate_new_data()

    def generate_new_data(self):
        self.X = np.linspace(0, 10, 50)
        self.y = 2 * self.X + 5 + np.random.normal(0, 1.5, size=50)

    def get_raw_data(self):
        return self.X, self.y

    def calculate_regression(self):
        x_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        numerator = np.sum((self.X - x_mean) * (self.y - y_mean))
        denominator = np.sum((self.X - x_mean)**2)
        if denominator == 0:
            m = 0
            b = y_mean
        else:
            m = numerator / denominator
            b = y_mean - m * x_mean
        x_fit = np.array([self.X.min(), self.X.max()])
        y_fit = m * x_fit + b

        return x_fit, y_fit, m, b

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, layout='tight')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.axes.set_title("1D Linear Regression")
        self.axes.set_xlabel("Feature X")
        self.axes.set_ylabel("Target Y")
        self.axes.grid(True, linestyle='--', alpha=0.7)

    def plot_data(self, X_data, y_data, X_fit, y_fit, m, b):
        self.axes.clear()
        self.axes.scatter(X_data, y_data, label='Raw Data', color='#4F46E5', s=10, alpha=0.7)
        self.axes.plot(X_fit, y_fit, label=f'Fit: y = {m:.2f}x + {b:.2f}', color='#DC2626', linewidth=3)
        self.axes.set_title("Linear Regression Visualization")
        self.axes.set_xlabel("Feature X")
        self.axes.set_ylabel("Target Y")
        self.axes.grid(True, linestyle='--', alpha=0.5)
        self.axes.legend(loc='upper left', frameon=True, shadow=True)
        
        self.draw()


class RegressionView(QMainWindow):

    plot_requested = pyqtSignal()

    def __init__(self, canvas):
        super().__init__()
        self.setWindowTitle("PyQt MVC Regression Plotter")
        self.setMinimumSize(800,600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        title_label = QLabel("Linear Regression with Matplotlib (MVC)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px; color: darkblue;")
        self.layout.addWidget(title_label)

        self.canvas = canvas
        canvas_frame = QFrame()
        canvas_frame.setFrameShape(QFrame.NoFrame)
        canvas_frame.setStyleSheet("""
            QFrame {
                background-color: #F8FAFC; 
                border: 2px solid #D1D5DB; 
                border-radius: 12px;
            }
        """)
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.addWidget(self.canvas)
        self.layout.addWidget(canvas_frame)

        self.status_label = QLabel("Click the button to generate new data, run the regression, and plot the result.")
        self.status_label.setStyleSheet("padding: 10px; font-style: italic; color: #374151;")
        self.layout.addWidget(self.status_label)

        self.plot_button = QPushButton("Generate New Data and Plot Regression")
        self.plot_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.plot_button.clicked.connect(self._button_clicked)
        self.layout.addWidget(self.plot_button)

    def _button_clicked(self):
        self.plot_requested.emit()

    def display_results(self, X_data, y_data, X_fit, y_fit, m, b):
        self.canvas.plot_data(X_data, y_data, X_fit, y_fit, m, b)
        status_text = (
            f"Regression Complete! | "
            f"Slope (m): <span style='color:#DC2626; font-weight:bold;'>{m:.4f}</span> | "
            f"Intercept (b): <span style='color:#DC2626; font-weight:bold;'>{b:.4f}</span>"
        )
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet("padding: 10px; font-weight: bold; color: #1F2937;")

class RegressionController(QObject):

    def __init__(self, model, view):
        super().__init__()
        self.model = model
        self.view = view

        self.view.plot_requested.connect(self.handle_plot_request)

    def handle_plot_request(self):

        print("Controller received plot request. Generating new data and calculating regression...")
        
        self.model.generate_new_data()
        X_data, y_data = self.model.get_raw_data()
        X_fit, y_fit, m, b = self.model.calculate_regression()
        self.view.display_results(X_data, y_data, X_fit, y_fit, m, b)
        print("View updated with regression results.")


if __name__ == '__main__':

    model = LinearRegressionModel()
    mpl_canvas = MplCanvas()
    view = RegressionView(canvas=mpl_canvas)
    view.show()
    controller = RegressionController(model=model, view=view)

    app = QApplication(sys.argv)
    sys.exit(app.exec_())
