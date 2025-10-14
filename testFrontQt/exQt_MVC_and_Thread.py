import sys
import time
import threading 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QListWidget, QPushButton, 
    QLineEdit, QListWidgetItem, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QObject, QThread, pyqtSignal, QMetaObject, Q_ARG, Q_RETURN_ARG
)
from PyQt5.QtCore import pyqtSlot

# ======================================================================
# 1. MODEL: Manages Application Data
# ======================================================================
class DataModel(QObject):
    """
    The Model manages the state and data of the application.
    It exposes signals when its data is changed.
    """
    # Signal emitted when the internal data list is updated
    data_updated = pyqtSignal(list)
    
    def __init__(self, initial_tasks=None):
        super().__init__()
        # Mock data structure: a list of dictionaries
        self._tasks = initial_tasks or [
            {"id": 1, "name": "Buy groceries", "status": "Pending"},
            {"id": 2, "name": "Write PyQt5 MVC app", "status": "Completed"},
            {"id": 3, "name": "Review documentation", "status": "Pending"},
        ]
        self._next_id = len(self._tasks) + 1

    def get_all_tasks(self):
        """Simulates fetching data from a database/API."""
        print("Model: Starting long task (1.5s delay) to get all tasks.") # DEBUG
        # This method could be lengthy, so it should be run by a worker.
        time.sleep(1.5) # Simulate latency
        print(f"Model: Finished long task. Returning {len(self._tasks)} tasks.") # DEBUG
        return self._tasks

    def add_task(self, name):
        """Adds a new task to the internal list."""
        if not name:
            return
        
        print(f"Model: Starting short task (0.5s delay) to add task '{name}'.") # DEBUG
        time.sleep(0.5) # Simulate write operation delay
        
        new_task = {"id": self._next_id, "name": name, "status": "Pending"}
        self._tasks.append(new_task)
        self._next_id += 1
        print(f"Model: Finished short task. Total tasks: {len(self._tasks)}.") # DEBUG
        return self._tasks

    def refresh(self):
        """Method called by the Controller to refresh the data."""
        # The result of get_all_tasks will be emitted by the worker signal
        return self.get_all_tasks()

# ======================================================================
# 2. WORKER (Subclassed QThread): Runs long operations off the main thread
# ======================================================================
class TaskThread(QThread):
    """
    QThread subclass responsible for executing the long-running Model methods.
    The 'run' method is always executed in the new thread.
    """
    # Signal to return the result of the task to the Controller
    result = pyqtSignal(object) 
    
    def __init__(self, method_to_run, args=None, parent=None):
        super().__init__(parent)
        self._method = method_to_run
        self._args = args if args is not None else []
        
    def run(self):
        """Execute the stored method and emit the result."""
        # DIAGNOSTIC PRINT: Check the thread ID where this function is running
        print(f"Worker: Running task on thread ID: {threading.get_ident()}") 
        try:
            # Execute the bound method with its arguments
            res = self._method(*self._args)
            self.result.emit(res)
        except Exception as e:
            # Handle potential errors
            self.result.emit(f"Error: {str(e)}")
        finally:
            # Note: No explicit signal needed here, as QThread.run() automatically 
            # emits the 'finished' signal when it exits.
            pass
            
# ======================================================================
# 3. VIEW: Displays the UI and captures user interaction
# ======================================================================
class TaskView(QMainWindow):
    """
    The View builds the GUI and emits signals for user actions.
    It knows nothing about the Model's data structure or logic.
    """
    # Signals for user interaction (Controller will listen to these)
    refresh_requested = pyqtSignal()
    add_requested = pyqtSignal(str) # Emits the text to be added
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 MVC Task Manager")
        self.setGeometry(200, 200, 600, 500)
        self.setStyleSheet(self._get_qss_style())
        
        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Title
        self.title_label = QLabel("Task List (MVC + Threading)")
        self.title_label.setObjectName("titleLabel")
        main_layout.addWidget(self.title_label)
        
        # Task List (Display component)
        self.task_list = QListWidget()
        self.task_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.task_list)

        # ----------------- Input/Action Panel -----------------
        input_panel = QHBoxLayout()
        
        # Add Task Widgets
        self.add_input = QLineEdit()
        self.add_input.setPlaceholderText("Enter new task name...")
        self.add_button = QPushButton("Add Task")
        self.add_button.setObjectName("addButton")
        self.add_button.clicked.connect(self._handle_add_task)

        # Refresh Button
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        
        # Assemble input panel
        input_panel.addWidget(self.add_input)
        input_panel.addWidget(self.add_button)
        input_panel.addWidget(self.refresh_button)
        
        main_layout.addLayout(input_panel)
        
        # Status Bar
        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("statusLabel")
        main_layout.addWidget(self.status_label)
        
        # Request initial data load
        QMetaObject.invokeMethod(self, 'request_initial_refresh', Qt.QueuedConnection)

    @pyqtSlot()
    def request_initial_refresh(self):
        """Helper to fire the initial refresh once the UI is set up."""
        self.refresh_requested.emit()

    def _handle_add_task(self):
        """Internal slot to process add button click and emit the requested signal."""
        task_name = self.add_input.text().strip()
        if task_name:
            self.add_requested.emit(task_name)
            self.add_input.clear()
        else:
            self.set_status("Please enter a task name.", "warning")

    def update_task_list(self, task_data: list):
        """
        Slot connected to the Model's data_updated signal.
        This is the only place the View directly interacts with the data structure.
        """
        self.task_list.clear()
        for task in task_data:
            status_text = f"[{task['status']}]"
            item_text = f"{status_text.ljust(15)} {task['name']}"
            item = QListWidgetItem(item_text)
            
            # Apply styling based on status
            if task['status'] == "Completed":
                item.setForeground(Qt.darkGreen)
            else:
                item.setForeground(Qt.darkRed)
            
            self.task_list.addItem(item)
        self.set_status(f"List updated! Found {len(task_data)} tasks.", "success")
        
    def set_status(self, message: str, level: str = "info"):
        """
        Updates the status bar with a message and temporary styling.
        """
        self.status_label.setText(message)
        
        if level == "loading":
            self._set_ui_busy(True)
            self.status_label.setStyleSheet("QLabel#statusLabel { color: #FFA000; font-weight: bold; }")
        elif level == "success":
            self._set_ui_busy(False)
            self.status_label.setStyleSheet("QLabel#statusLabel { color: #388E3C; font-weight: bold; }")
        elif level == "warning":
            self._set_ui_busy(False)
            self.status_label.setStyleSheet("QLabel#statusLabel { color: #D32F2F; font-weight: bold; }")
        else:
            self._set_ui_busy(False)
            self.status_label.setStyleSheet("QLabel#statusLabel { color: #555; }")

    def _set_ui_busy(self, is_busy):
        """Disables/Enables input elements while a background task runs."""
        self.add_input.setEnabled(not is_busy)
        self.add_button.setEnabled(not is_busy)
        self.refresh_button.setEnabled(not is_busy)

    def _get_qss_style(self):
        """Custom QSS for a clean, modern look."""
        return """
            QWidget {
                background-color: #f4f7f9;
                font-family: Inter, sans-serif;
            }
            QLabel#titleLabel {
                font-size: 28px;
                color: #2c3e50;
                padding: 15px;
                border-bottom: 2px solid #ecf0f1;
            }
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
            QLabel#statusLabel {
                font-size: 14px;
                padding: 5px;
            }
        """

# ======================================================================
# 4. CONTROLLER: The central nervous system
# ======================================================================
class Controller(QObject):
    """
    The Controller handles the flow of control and data between Model, View, and Workers.
    """
    def __init__(self, model: DataModel, view: TaskView):
        super().__init__()
        self._model = model
        self._view = view
        self._current_thread = None # To hold the reference to the active thread

        self._connect_signals()
        
    def _connect_signals(self):
        """Connects all signals from the View and Model to Controller slots."""
        
        # --- Connect View Signals to Controller Slots (User Actions) ---
        self._view.refresh_requested.connect(self.handle_refresh_request)
        self._view.add_requested.connect(self.handle_add_request)

        # --- Connect Model Signal to View Slot (Data Update) ---
        # When Model data changes, the View is updated directly by the Controller
        self._model.data_updated.connect(self._view.update_task_list)
        
    def _start_worker(self, worker_method, args=None, message="Processing task in background..."):
        """
        Generic helper to initialize and start a TaskThread.
        """
        if self._current_thread and self._current_thread.isRunning():
            self._view.set_status("A task is already running...", "warning")
            return
            
        # Use the specific message provided by the calling handler
        self._view.set_status(message, "loading")
        
        # 1. Create TaskThread instance (QThread subclass)
        self._current_thread = TaskThread(worker_method, args)
        
        # 2. Connect result handling and cleanup
        # Result signal comes from the custom TaskThread class
        self._current_thread.result.connect(self.handle_worker_result)
        
        # Standard QThread cleanup connections
        self._current_thread.finished.connect(self._current_thread.deleteLater)
        self._current_thread.finished.connect(lambda: setattr(self, '_current_thread', None))
        
        # 3. Start the thread
        self._current_thread.start()
        
    # --- Controller Slots: Command Handlers ---

    def handle_refresh_request(self):
        """Starts a worker to execute the Model's refresh method."""
        self._start_worker(self._model.get_all_tasks, message="Loading task list from Model...")

    def handle_add_request(self, task_name: str):
        """Starts a worker to execute the Model's add_task method."""
        self._start_worker(self._model.add_task, args=[task_name], message=f"Adding task '{task_name}' in background...")
        
    def handle_worker_result(self, data):
        """
        Generic slot that receives the output from ANY worker.
        It then checks the data type and tells the Model to emit an update signal.
        """
        print(f"Controller: Received result of type {type(data)}.") # DEBUG
        # DIAGNOSTIC PRINT: Check the thread ID where this slot is running (should be main thread)
        print(f"Controller: Handling result on thread ID: {threading.get_ident()}")
        
        if isinstance(data, list):
            # This means the worker successfully returned the new list of tasks.
            # We now tell the Model to broadcast this new data list via its signal.
            self._model.data_updated.emit(data)
        elif isinstance(data, str) and data.startswith("Error:"):
            self._view.set_status(data, "warning")
        else:
            self._view.set_status("Unknown result received from worker.", "warning")

# ======================================================================
# 5. Application Startup
# ======================================================================
if __name__ == '__main__':
    # DIAGNOSTIC PRINT: Print main thread ID
    print(f"Main Thread ID: {threading.get_ident()}")
    print("Application starting...")
    app = QApplication(sys.argv)
    
    # 1. Create Mock Data and Model
    mock_data = [
        {"id": 1, "name": "Prepare project report", "status": "Pending"},
        {"id": 2, "name": "Call team lead", "status": "Completed"},
        {"id": 3, "name": "Deploy new version", "status": "Pending"},
    ]
    data_model = DataModel(initial_tasks=mock_data)

    # 2. Create View
    task_view = TaskView()

    # 3. Create Controller (The glue)
    controller = Controller(data_model, task_view)

    # 4. Show the View and run the application
    task_view.show()
    sys.exit(app.exec())
