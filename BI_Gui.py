import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic 

from UIcomps.componentsGUI import MyMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    # window = uic.loadUi("./UIcomps/BayesRegGUI.ui")
    window.show()
    sys.exit(app.exec())