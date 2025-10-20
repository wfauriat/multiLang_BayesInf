import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic

from ui_mywindow import Ui_MainWindow 

class MyMainWindow(QMainWindow):   # Method 2 : imported from a .py file generate with CLI pyuic6
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_11.clicked.connect(self.open_window2)
    def open_window2(self):
        # print("ok")
        self.ui.window2 = SecondWindow()
        self.ui.window2.show()

class SecondWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regression Model Parameters")
        self.resize(600,400)


    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # window = uic.loadUi("BayesRegGUI.ui")    # Method 1 : direct runtime import conversion
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec())

