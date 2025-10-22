import sys
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QProgressBar,
                             QPushButton, QVBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal

# 1. Create a Worker Thread to run the long operation
class Worker(QThread):
    # Signal to update the progress bar value (emits an integer)
    progress_updated = pyqtSignal(int)

    def run(self):
        # Simulate a task with 100 steps
        for i in range(101):
            # Emit the current progress value (0 to 100)
            self.progress_updated.emit(i)
            # Simulate work being done (delay)
            time.sleep(0.05)
        
        # Optionally emit a final value or a completion signal
        print("Task completed!")

# 2. Main Application Window
class ProgressBarDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt QProgressBar Demo')
        self.setGeometry(100, 100, 300, 150)
        self.initUI()
        
    def initUI(self):
        # Create widgets
        self.pbar = QProgressBar(self)
        # Set the range (default is 0 to 100, but good practice to set it)
        self.pbar.setRange(0, 100) 
        
        self.startButton = QPushButton('Start Task', self)
        self.startButton.clicked.connect(self.start_task)
        
        # Create layout and add widgets
        vbox = QVBoxLayout()
        vbox.addWidget(self.pbar)
        vbox.addWidget(self.startButton)
        self.setLayout(vbox)
        
        # Initialize the Worker Thread
        self.worker = Worker()
        # Connect the worker's signal to the update method
        self.worker.progress_updated.connect(self.update_progress_bar)

    def start_task(self):
        # Reset the progress bar for a new task
        self.pbar.setValue(0)
        # Disable the button to prevent multiple starts
        self.startButton.setEnabled(False)
        # Start the background thread
        self.worker.start()
        
    def update_progress_bar(self, value):
        # This method runs in the main GUI thread because it is a slot 
        # connected to a signal from another thread.
        self.pbar.setValue(value)
        
        # Check if task is finished
        if value == 100:
            self.startButton.setEnabled(True)
            self.startButton.setText('Task Completed!')
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = ProgressBarDemo()
    demo.show()
    sys.exit(app.exec_())