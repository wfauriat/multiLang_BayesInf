from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMainWindow
from .baseLayout import Ui_MainWindow

class ModelUI(QObject):
    def __init__(self):
        super().__init__()
        pass

class ViewMainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

class ControlerUI(QObject):
    def __init__(self, model, view): 
        super().__init__()
        self.model = model
        self.view = view

