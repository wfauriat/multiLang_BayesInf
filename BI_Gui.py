import sys
from PyQt5.QtWidgets import QApplication
# from PyQt5 import uic 

# from multiLang_BayesInf.UIcomps.componentsGUI import ModelUI, ViewMainUI, ControllerUI
from UIcomps.componentsGUI import ModelUI, ViewMainUI, ControllerUI


if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = ViewMainUI()
    # view = uic.loadUi("./UIcomps/BayesRegGUI.ui")
    view.show()
    model = ModelUI()
    controller = ControllerUI(model, view)
    sys.exit(app.exec())