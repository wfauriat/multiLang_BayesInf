import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout
# from .baseLayout import Ui_MainWindow

from multiLang_BayesInf.UIcomps.baseLayout import Ui_MainWindow
from multiLang_BayesInf.cases_data.data_cases_def import PolynomialCase, HousingCase

from multiLang_BayesInf.pyBI.base import (
    UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar)
from multiLang_BayesInf.pyBI.inference import MHalgo, MHwGalgo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class ModelUI(QObject):
    def __init__(self):
        super().__init__()
        self.data_selected_case = "Polynomial"
        self.load_case()
        self.NMCMC = 20000
        self.Nthin = 20
        self.Nburn = 5000
        self.verbose = True
        self.MCalgo = None

    def load_case(self):
        if self.data_selected_case == "Polynomial":
            self.data_case = PolynomialCase()
            self.rndUs = [UnifVar([-3,3]) for _ in range(3)]
            self.rnds = HalfNormVar(param=0.5)
            self.obsvar = ObsVar(obs=self.data_case.ymes,
                                prev_model=self.data_case.form_fit, 
                                cond_var=self.data_case.xmes)
            self.bstart = np.array([self.rndUs[i].draw() for i in range(3)] + \
                        [float(self.rnds.draw())])
        elif self.data_selected_case == "Housing":
            self.data_case = HousingCase()
            self.rndUs = [NormVar([0, 1000000]), NormVar([0, 50000]), NormVar([0, 50000]),
                NormVar([0, 50000]), NormVar([0, 50000]), NormVar([0, 50000]),
                NormVar([0, 50000]), NormVar([0, 50000]), NormVar([0, 50000])]
            self.rnds = HalfNormVar(param=80000)
            self.obsvar = ObsVar(obs=np.c_[self.data_case.y_train],
                    prev_model=self.data_case.form_fit, 
                    cond_var=self.data_case.X_train)
            self.bstart = np.array([0]*9 + [80000])


class ViewMainUI(QMainWindow):

    computeSignal = pyqtSignal()
    NMCMCSignal = pyqtSignal(int)
    selectCaseSignal = pyqtSignal(str)


    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.pushCompute.clicked.connect(self._handle_pushCompute)
        # self.ui.lineEdMC1.editingFinished.connect(self._handle_NMCMC)
        self.ui.lineEdMC1.textChanged.connect(self._handle_NMCMC)
        self.ui.selectDataCase.currentTextChanged.connect(self._handle_select_case)

        self.ui.tab1Layout = QVBoxLayout(self.ui.tab)
        self.ui.tab2Layout = QVBoxLayout(self.ui.tab_2)
        self.ui.tab4Layout = QVBoxLayout(self.ui.tab_4)
        self.sc1 = MplCanvas(self.ui.tab, width=5, height=4, dpi=100)
        self.sc2 = MplCanvas(self.ui.tab_2, width=5, height=4, dpi=100)
        self.sc4 = MplCanvas(self.ui.tab_4, width=5, height=4, dpi=100)
        self.ui.tab1Layout.addWidget(self.sc1)
        self.ui.tab2Layout.addWidget(self.sc2)
        self.ui.tab4Layout.addWidget(self.sc4)

    def _handle_pushCompute(self):
        self.computeSignal.emit()

    def _handle_NMCMC(self):
        value = int(self.ui.lineEdMC1.text())
        self.NMCMCSignal.emit(value)

    def _handle_select_case(self):
        value = self.ui.selectDataCase.currentText()
        self.selectCaseSignal.emit(value)


class ControllerUI(QObject):
    def __init__(self, model, view): 
        super().__init__()
        self.model = model
        self.view = view
        self.view.computeSignal.connect(self._handle_pushCompute)
        self.view.NMCMCSignal.connect(self._getMCMC)
        self.view.selectCaseSignal.connect(self._select_case)

    def _handle_pushCompute(self):
        ### ADD A WORKER IN ORDER NOT TO BLOCK THE UI
        self.model.MCalgo = MHalgo(N=self.model.NMCMC,
                                   Nthin=self.model.Nthin,
                                   Nburn=self.model.Nburn,
                                   is_adaptive=True,
                                   verbose=self.model.verbose)
        # MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
        #                    verbose=verbose)
        self.model.MCalgo.initialize(self.model.obsvar,
                                     self.model.rndUs, 
                                     self.model.rnds)
        self.model.MCalgo.MCchain[0] = self.model.bstart
        self.model.MCalgo.state(0, set_state=True)
        MCout, llout = self.model.MCalgo.runInference()
        postpar = self.model.MCalgo.cut_chain
        if self.model.data_selected_case == "Polynomial":
            postY = np.array([self.model.data_case.form_fit(
                            self.model.data_case.xmes, bb)[0] for bb in postpar])
            self.view.sc1.axes.clear()
            self.view.sc1.axes.plot(self.model.data_case.xmes[:,0],
                                    postY.T[:,0], '.k', label="posterior prediction")
            self.view.sc1.axes.plot(self.model.data_case.xmes[:,0],
                                    postY.T, '.k')
            self.view.sc1.axes.plot(self.model.data_case.xmes[:,0],
                                    np.ravel(self.model.data_case.ymes), 'or',
                                    label="measured values")
            self.view.sc1.axes.legend()
        elif self.model.data_selected_case == "Housing":
            postY = np.array([[self.model.data_case.form_fit(
                              np.r_[[xx]], bb)[0] for bb in postpar] \
                            for xx in self.model.data_case.X_test[:100,:]])
            self.view.sc1.axes.clear()
            self.view.sc1.axes.plot(self.model.data_case.X_test[:100,-1],
                                    postY[:,0], '.k', label="posterior prediction")
            self.view.sc1.axes.plot(self.model.data_case.X_test[:100,-1],
                                    postY, '.k')
            self.view.sc1.axes.plot(self.model.data_case.X_test[:100,-1],
                                    np.ravel(self.model.data_case.y_test[:100]), 'or',
                                    label="measured values")
            self.view.sc1.axes.legend()
        self.view.sc1.draw()
        self.view.sc2.axes.clear()
        self.view.sc2.axes.scatter(MCout[:,0], MCout[:,1], c=llout, cmap="jet")
        self.view.sc2.draw()
        self.view.sc4.axes.clear()
        self.view.sc4.axes.plot(self.model.MCalgo.MCchain[:,0])
        self.view.sc4.draw()
        print("MAP" + str(self.model.MCalgo.MAP))

    @pyqtSlot(int)
    def _getMCMC(self, value):
        self.model.NMCMC = value

    @pyqtSlot(str)
    def _select_case(self, value):
        self.model.data_selected_case = value
        if self.model.data_selected_case == "Polynomial":
            print(value)
        elif self.model.data_selected_case == "Housing":
            print(value)
        self.model.load_case()
