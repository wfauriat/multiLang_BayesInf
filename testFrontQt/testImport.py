import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Import File Example")
        
        # Set up a simple UI
        self.button = QPushButton("Import File", self)
        self.button.clicked.connect(self.openFileNameDialog)
        
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # Uncomment this to use the non-native Qt dialog
        
        # Open the file dialog and get the selected file name (path) and the filter used
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            "", # Default directory (empty string is usually fine)
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)", # File filter
            options=options
        )
        
        if file_name:
            # Call the function to handle the file content
            self.import_file_content(file_name)

    def import_file_content(self, file_path):
        """Reads the content of the file and processes it."""
        try:
            # Open the file in read mode ('r')
            with open(file_path, 'r') as f:
                content = f.read()
                
            print(f"Successfully imported file: {file_path}")
            print("--- Content Preview ---")
            # Displaying the first 200 characters for demonstration
            print(content[:200] + "..." if len(content) > 200 else content)
            print("-----------------------")
            
            # TODO: Add your actual file processing logic here
            # e.g., parsing CSV, XML, JSON, or loading image/data.

            QMessageBox.information(self, "Success", f"File '{file_path}' imported successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read file: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())