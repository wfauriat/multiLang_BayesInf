import sys
from PyQt5.QtWidgets import QApplication
# from PyQt5 import uic 

from UIcomps.componentsGUI import ViewMainUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ViewMainUI()
    # window = uic.loadUi("./UIcomps/BayesRegGUI.ui")
    window.show()
    sys.exit(app.exec())