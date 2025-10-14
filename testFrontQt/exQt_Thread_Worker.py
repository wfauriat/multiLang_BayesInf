import sys
import time # Used to simulate a long-running task
# Import necessary modules from PyQt5
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QMessageBox
)
from PyQt5.QtGui import QIcon
# QThread and pyqtSignal are crucial for safe threading
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal

# --- 1. Worker Class (The logic that runs in the separate thread) ---
class Worker(QObject):
    """
    Worker object that handles the background task.
    It must inherit from QObject to use Qt's signal/slot mechanism.
    """
    # Signals to communicate back to the main thread
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(str)

    def run_long_task(self):
        """
        A function to simulate a time-consuming operation.
        NOTE: This function does NOT touch any UI elements directly.
        """
        max_steps = 5
        self.result.emit("Task started...")
        
        for i in range(1, max_steps + 1):
            # Simulate work being done (e.g., calculation, network call)
            time.sleep(1)

            # Calculate and emit progress back to the main thread
            percentage = int((i / max_steps) * 100)
            self.progress.emit(percentage)
            print(f"Worker thread processing: {percentage}%")

        # Emit the final result and the finished signal
        final_message = f"Task completed successfully! Processed {max_steps} steps."
        self.result.emit(final_message)
        self.finished.emit()


# --- 2. Main Window Class (UI and Thread Management) ---
class MainWindow(QMainWindow):
    """
    Main application window class.
    Manages UI and orchestrates the worker thread execution.
    """
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.setWindowTitle("PyQt5 App with Worker Thread")
        self.setGeometry(100, 100, 550, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # --- UI Elements ---
        self.title_label = QLabel("Threaded Background Task Demo")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; color: #1e88e5;")
        
        self.status_label = QLabel("Click 'Start Task' to begin.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; margin-top: 10px; color: #555;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ccc;
                border-radius: 5px;
                background-color: #f0f0f0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                margin: 0.5px;
            }
        """)

        self.start_button = QPushButton("Start Long Task")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #03A9F4;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #0288D1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.start_button.clicked.connect(self.start_task)

        # --- Layout Arrangement ---
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addStretch()
        self.layout.addWidget(self.start_button)
        self.layout.addStretch()

        # Initialize the thread container (must be an instance variable)
        self.thread = None
        self.worker = None

    def start_task(self):
        """
        Initializes and starts the worker thread when the button is pressed.
        """
        # Disable the button to prevent multiple simultaneous tasks
        self.start_button.setEnabled(False)
        self.status_label.setText("Task running... Please wait.")
        self.progress_bar.setValue(0)

        # 1. Create a QThread object
        self.thread = QThread()
        # 2. Create the Worker object
        self.worker = Worker()
        # 3. Move the Worker object to the thread
        self.worker.moveToThread(self.thread)

        # 4. Connect signals to slots:
        
        # Connect the thread's started signal to the worker's long task method
        self.thread.started.connect(self.worker.run_long_task)

        # Connect worker signals to UI update slots (safe to update UI here)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.update_status)
        
        # Connect signals for cleanup and completion
        self.worker.finished.connect(self.thread.quit) # Stop the QThread's event loop
        self.worker.finished.connect(self.worker.deleteLater) # Delete the worker QObject
        self.thread.finished.connect(self.thread.deleteLater) # Delete the QThread object
        self.thread.finished.connect(self.task_finished) # Handle final UI updates

        # 5. Start the thread
        self.thread.start()
        
        # NOTE: Do NOT call worker.run_long_task() directly here! 
        # Calling it directly runs the task in the main thread and freezes the UI.

    def update_progress(self, value):
        """Slot to safely update the progress bar from the worker thread."""
        self.progress_bar.setValue(value)

    def update_status(self, message):
        """Slot to safely update the status label from the worker thread."""
        self.status_label.setText(message)
        
    def task_finished(self):
        """Slot called when the thread has finished and cleaned up."""
        self.start_button.setEnabled(True)
        # Final message is already set by worker.result signal
        QMessageBox.information(self, "Task Complete", "The background process has finished.")
        # Reset the thread and worker variables
        self.thread = None
        self.worker = None

def main():
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    # Ensure application is high DPI aware (good practice)
    # app.setAttribute(Qt.AA_EnableHighDpiScaling) # Use this line if on Qt 5.6+ and experiencing scaling issues

    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
