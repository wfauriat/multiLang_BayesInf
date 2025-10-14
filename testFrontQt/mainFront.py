import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    """

    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Basic PyQt5 App")
        self.setMinimumSize(300,300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        title_label = QLabel("Welcome to your PyQt5 Application!")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; color: #333;")

        info_label = QLabel("This is the main window content area.")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size: 14px; margin-top: 10px;")

        self.action_button = QPushButton("Click Me")
        self.action_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.action_button.clicked.connect(self.button_clicked)


        layout.addWidget(title_label)
        layout.addWidget(info_label)
        # layout.addStretch()
        layout.addWidget(self.action_button)
        layout.addStretch() # Add stretch again for better centering

        central_widget.setLayout(layout)

    def button_clicked(self):
        print("Button was clicked!")
        self.action_button.setText("Clicked!")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
