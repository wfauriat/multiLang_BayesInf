import numpy as np

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from multiLang_BayesInf.UIcomps.baseLayout import Ui_MainWindow

from multiLang_BayesInf.cases_data.data_cases_def import PolynomialCase, HousingCase

from multiLang_BayesInf.pyBI.base import (
    UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar)
from multiLang_BayesInf.pyBI.inference import MHalgo, MHwGalgo

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

###############################################################################
### MODEL OBJECT
###############################################################################

class ModelUI(QObject):
    def __init__(self):
        super().__init__()
        self.data_selected_case = "Polynomial"
        self.NMCMC = 20000
        self.Nthin = 20
        self.Nburn = 5000
        self.verbose = True
        self.MCalgo = None
        self.MCsort = None
        self.LLsort = None
        self.postpar = None
        self.postY = None
        self.data_case = None
        self.rnds = None
        self.rndUs = None
        self.obsvar = None
        self.bstart = None
        self.selected_model = "Linear Polynomial"
        self.fitreg_model = None
        self.yreg_pred = None
        self.load_case()

    def load_case(self):
        if self.data_selected_case == "Polynomial":
            self.data_case = PolynomialCase()
            self.rndUs = [UnifVar([-3,3]) for _ in range(3)]
            self.rnds = HalfNormVar(param=0.5)
            self.obsvar = ObsVar(obs=np.c_[np.ravel(self.data_case.ymes)],
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
            self.obsvar = ObsVar(obs=np.c_[self.data_case.ymes],
                    prev_model=self.data_case.form_fit, 
                    cond_var=self.data_case.xmes)
            self.bstart = np.array([0]*9 + [80000])

    def post_treat_chains(self):
        idxrdn = np.random.permutation(
            np.minimum(self.MCalgo.cut_chain.shape[0],100))
        postpar_red = self.postpar[idxrdn]
        self.postY = np.array([[self.data_case.form_fit(
                            np.r_[[xx]], bb)[0] for bb in postpar_red] \
                        for xx in self.data_case.xmes])
        self.postMAP = np.array([self.data_case.form_fit(np.r_[[xx]],
                             self.MCalgo.MAP) for xx in self.data_case.xmes])
        self.postYeps = self.postY + np.random.randn(100) * postpar_red[:,-1]

    def regr_fit(self):
        if self.selected_model != "SVR":
            if self.selected_model == "Linear Polynomial":
                self.fitreg_model = LinearRegression()
            elif self.selected_model == "ElasticNet":
                self.fitreg_model = ElasticNet()
            elif self.selected_model == "RandomForest":
                self.fitreg_model = RandomForestRegressor()
            Xfit = self.data_case.xmes
            Yfit = self.data_case.ymes
            self.fitreg_model.fit(Xfit, Yfit)
            self.yreg_pred = self.fitreg_model.predict(
                self.data_case.xmes)
        elif self.selected_model == "SVR":
            self.fitreg_model = SVR(
                kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(self.data_case.xmes)
            y_train_scaled = scaler_y.fit_transform(
                self.data_case.ymes.reshape(-1, 1)).ravel()
            self.fitreg_model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = self.fitreg_model.predict(
                X_train_scaled)
            self.yreg_pred = scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).ravel()


###############################################################################
### VIEW OBJECT
###############################################################################

class ViewMainUI(QMainWindow):

    computeSignal = pyqtSignal()
    fitRegSignal = pyqtSignal()
    NMCMCSignal = pyqtSignal(int)
    selectCaseSignal = pyqtSignal(str)
    selectDimRSignal = pyqtSignal(int)
    selectDim1Signal = pyqtSignal(int)
    selectDim2Signal = pyqtSignal(int)
    selectRegModelSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.dimR = 0
        self.dim1 = 0
        self.dim2 = 0
        self.check1 = 1
        self.check2 = 1
        self.check3 = 1 
        self.check4 = 1
        self.check5 = 1

        self.ui.pushCompute.clicked.connect(self._handle_pushCompute)
        self.ui.lineEdMC1.textChanged.connect(self._handle_NMCMC)
        self.ui.selectDataCase.currentTextChanged.connect(self._handle_select_case)
        self.ui.selectRegMod.currentTextChanged.connect(self.handle_select_regmod)
        self.ui.pushFitReg.clicked.connect(self._handle_pushFitReg)
        self.ui.selectDimR.currentIndexChanged.connect(self._handle_select_dimR)
        self.ui.selectDim1.currentIndexChanged.connect(self._handle_select_dim1)
        self.ui.selectDim2.currentIndexChanged.connect(self._handle_select_dim2)
        self.ui.checkLegend1.clicked.connect(self._handel_checkLegend1)
        self.ui.checkLegend2.clicked.connect(self._handel_checkLegend2)
        self.ui.checkLegend3.clicked.connect(self._handel_checkLegend3)
        self.ui.checkLegend4.clicked.connect(self._handel_checkLegend4)
        self.ui.checkLegend5.clicked.connect(self._handel_checkLegend5)

        self.ui.tab1Layout = QVBoxLayout(self.ui.tab)
        self.ui.tab2Layout = QVBoxLayout(self.ui.tab_2)
        self.ui.tab3Layout = QVBoxLayout(self.ui.tab_3)
        self.ui.tab4Layout = QVBoxLayout(self.ui.tab_4)
        self.sc1 = MplCanvas(self.ui.tab, width=5, height=4, dpi=100)
        self.sc2 = MplCanvas(self.ui.tab_2, width=5, height=4, dpi=100)
        self.sc3 = MplCanvas(self.ui.tab_3, width=5, height=4, dpi=100)
        self.sc4 = MplCanvas(self.ui.tab_4, width=5, height=4, dpi=100)
        self.ui.tab1Layout.addWidget(self.sc1)
        self.ui.tab2Layout.addWidget(self.sc2)
        self.ui.tab3Layout.addWidget(self.sc3)
        self.ui.tab4Layout.addWidget(self.sc4)

    def _handle_pushCompute(self):
        self.computeSignal.emit()
    
    def _handle_pushFitReg(self):
        self.fitRegSignal.emit()

    def _handle_NMCMC(self):
        value = int(self.ui.lineEdMC1.text())
        self.NMCMCSignal.emit(value)

    def _handle_select_case(self):
        value = self.ui.selectDataCase.currentText()
        self.selectCaseSignal.emit(value)

    def handle_select_regmod(self):
        value = self.ui.selectRegMod.currentText()
        self.selectRegModelSignal.emit(value)

    def _handle_select_dimR(self):
        value = self.ui.selectDimR.currentIndex()
        self.selectDimRSignal.emit(value)
    def _handle_select_dim1(self):
        value = self.ui.selectDim1.currentIndex()
        self.selectDim1Signal.emit(value)
    def _handle_select_dim2(self):
        value = self.ui.selectDim2.currentIndex()
        self.selectDim2Signal.emit(value)

    def _update_dimR_selection(self, dim):
        self.ui.selectDimR.clear()
        for i in range(dim):
            self.ui.selectDimR.addItem(str(i))
    def _update_dimP_selection(self, dim):
        self.ui.selectDim1.clear()
        self.ui.selectDim2.clear()
        for i in range(dim):
            self.ui.selectDim1.addItem(str(i))
            self.ui.selectDim2.addItem(str(i))

    def _handel_checkLegend1(self, value):
        self.check1 = value
    def _handel_checkLegend2(self, value):
        self.check2 = value
    def _handel_checkLegend3(self, value):
        self.check3 = value
    def _handel_checkLegend4(self, value):
        self.check4 = value
    def _handel_checkLegend5(self, value):
        self.check5 = value


###############################################################################
### CONTROLLER OBJECT
###############################################################################

class ControllerUI(QObject):
    def __init__(self, model, view): 
        super().__init__()
        self.model = model
        self.view = view
        self.thread = None
        self.worker = None

        self.view.computeSignal.connect(self._handle_pushCompute)
        self.view.fitRegSignal.connect(self._handle_pushFitReg)
        self.view.NMCMCSignal.connect(self._getMCMC)
        self.view.selectCaseSignal.connect(self._select_case)
        self.view.selectRegModelSignal.connect(self._select_reg_model)
        self.view.selectDimRSignal.connect(self._select_dimR)
        self.view.selectDim1Signal.connect(self._select_dim1)
        self.view.selectDim2Signal.connect(self._select_dim2)
        self.view.ui.checkLegend1.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend2.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend3.clicked.connect(self.draw_plot_tabs)  
        self.view.ui.checkLegend4.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend5.clicked.connect(self.draw_plot_tabs)

    def _handle_pushCompute(self):
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
        self._compute_worker()

    def _handle_pushFitReg(self):
        self.model.regr_fit()
        self.draw_plot_tabs()

    def draw_plot_tabs(self):
        self.view.sc1.axes.clear()
        if self.view.check4:
            self.view.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postYeps[:100,0], '.b', label="posterior with noise")
            self.view.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postY[:100,0], '.k', label="posterior prediction")
            self.view.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postYeps[:100], '.b')
            self.view.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postY[:100], '.k')
        if self.view.check3: self.view.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    np.ravel(self.model.postMAP[:100]), '.g',
                    label="MAP")
        if self.view.check5: self.view.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    self.model.yreg_pred[:100], '.', color='orange',
                    label="regmod")
        if self.view.check1: self.view.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    np.ravel(self.model.data_case.ymes[:100]), 'or',
                    label="train values") 
        if self.view.check2: self.view.sc1.axes.plot(
                    self.model.data_case.X_test[:20,self.view.dimR],
                        np.ravel(self.model.data_case.y_test[:20]), 'sm', ms=3,
                        label="test values")
        self.view.sc1.axes.legend()
        self.view.sc1.draw()
        self.view.sc2.axes.clear()
        self.view.sc2.axes.scatter(self.model.MCsort[:,self.view.dim1],
                                   self.model.MCsort[:,self.view.dim2],
                                   c=self.model.LLsort, cmap="jet")
        self.view.sc2.draw()
        self.view.sc3.axes.clear()
        self.view.sc3.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                np.ravel(self.model.data_case.ymes[:100]) - \
                                np.ravel(self.model.postMAP[:100]), '.g')
        self.view.sc3.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                np.ravel(self.model.data_case.ymes[:100]) - \
                                np.ravel(self.model.yreg_pred[:100]), '.',
                                color='orange')
        self.view.sc3.draw()
        self.view.sc4.axes.clear()
        self.view.sc4.axes.plot(self.model.MCalgo.MCchain[:,self.view.dim1])
        self.view.sc4.draw()

    @pyqtSlot(int)
    def _getMCMC(self, value):
        self.model.NMCMC = value

    @pyqtSlot(str)
    def _select_case(self, value):
        self.view.ui.selectDimR.setCurrentIndex(0)
        self.model.data_selected_case = value
        self.model.load_case()

    @pyqtSlot(str)
    def _select_reg_model(self, value):
        self.model.selected_model = value

    @pyqtSlot(int)
    def _select_dimR(self, value):
        self.view.dimR = value
        self.draw_plot_tabs()
    @pyqtSlot(int)
    def _select_dim1(self, value):
        self.view.dim1 = value
        self.draw_plot_tabs()
    @pyqtSlot(int)
    def _select_dim2(self, value):
        self.view.dim2 = value
        self.draw_plot_tabs()

    def _compute_worker(self):
        self.thread = QThread()
        self.worker = ComputeWorker(self.model)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_worker_finished(self):
        self.view._update_dimR_selection(self.model.data_case.xmes.shape[1])
        self.view.ui.selectDimR.setCurrentIndex(0)
        self.view._update_dimP_selection(len(self.model.bstart))
        self.view.ui.selectDim1.setCurrentIndex(0)
        self.view.ui.selectDim2.setCurrentIndex(
            int(self.view.ui.selectDim1.currentIndex())+1)
        self.draw_plot_tabs()
        print("MAP" + str(self.model.MCalgo.MAP))


###############################################################################
### TEMPLATE CANVAS OBJECT
###############################################################################

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


###############################################################################
### COMPUTE WORK OBJECT
###############################################################################

class ComputeWorker(QObject):
    finished = pyqtSignal()
    # progress = pyqtSignal(int)
    ## TO-DO add progress bar (need extract current i from MCalgo.runInference)

    def __init__(self, model):
        super().__init__()
        self.model = model

    @pyqtSlot()
    def run(self):
        # be careful as worker has access to object
        # to modify it directly (not well isolated)
        self.model.MCalgo.runInference()
        self.model.regr_fit()
        self.model.postpar = self.model.MCalgo.cut_chain
        self.model.MCsort = self.model.MCalgo.idx_chain
        self.model.LLsort = self.model.MCalgo.cut_llchain[
            self.model.MCalgo.sorted_indices]
        self.model.post_treat_chains()
        self.finished.emit()
    
    # def regr_fit(self):
    #     if self.model.selected_model != "SVR":
    #         if self.model.selected_model == "Linear Polynomial":
    #             self.model.fitreg_model = LinearRegression()
    #         elif self.model.selected_model == "ElasticNet":
    #             self.model.fitreg_model = ElasticNet()
    #         elif self.model.selected_model == "RandomForest":
    #             self.model.fitreg_model = RandomForestRegressor()
    #         Xfit = self.model.data_case.xmes
    #         Yfit = self.model.data_case.ymes
    #         self.model.fitreg_model.fit(Xfit, Yfit)
    #         self.model.yreg_pred = self.model.fitreg_model.predict(
    #             self.model.data_case.xmes)
    #     elif self.model.selected_model == "SVR":
    #         self.model.fitreg_model = SVR(
    #             kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
    #         scaler_X = StandardScaler()
    #         scaler_y = StandardScaler()
    #         X_train_scaled = scaler_X.fit_transform(self.model.data_case.xmes)
    #         y_train_scaled = scaler_y.fit_transform(
    #             self.model.data_case.ymes.reshape(-1, 1)).ravel()
    #         self.model.fitreg_model.fit(X_train_scaled, y_train_scaled)
    #         y_pred_scaled = self.model.fitreg_model.predict(
    #             X_train_scaled)
    #         self.model.yreg_pred = scaler_y.inverse_transform(
    #             y_pred_scaled.reshape(-1, 1)).ravel()


