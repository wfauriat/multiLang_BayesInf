import numpy as np
from sklearn.model_selection import train_test_split

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5.QtGui import QDoubleValidator

from UIcomps.baseLayout import Ui_MainWindow

# from multiLang_BayesInf.UIcomps.baseLayout import Ui_MainWindow

# from multiLang_BayesInf.cases_data.data_cases_def import (
#     VoidCase, PolynomialCase, HousingCase)

# from multiLang_BayesInf.pyBI.base import (
#     UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar)
# from multiLang_BayesInf.pyBI.inference import MHalgo, MHwGalgo

from cases_data.data_cases_def import (
    VoidCase, PolynomialCase, HousingCase)

from pyBI.base import (
    UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar)
from pyBI.inference import MHalgo, MHwGalgo

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
        self.type_inf = 1
        self.verbose = True
        self.MCalgo = None
        self.MCsort = None
        self.LLsort = None
        self.postpar = None
        self.postY = None
        self.data_case = None
        self.custom_case = VoidCase()
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

        elif self.data_selected_case == "Custom":
            self.data_case = self.custom_case
            def form_fit(x, b):
                if x.ndim == 1:
                    x = x[np.newaxis, :]
                b = np.asarray(b)
                return b[0] + x @ b[1:] 
            self.data_case.form_fit = form_fit
            ymax = np.max(self.data_case.ymes)
            xdims = self.data_case.xmes.shape[1]
            self.rndUs = [NormVar([0, ymax]) for _ in range(xdims+1)]
            self.rnds = HalfNormVar(param=ymax)
            self.obsvar = ObsVar(obs=np.c_[self.data_case.ymes],
            prev_model=self.data_case.form_fit, 
            cond_var=self.data_case.xmes)
            self.bstart = np.array([0]*(xdims+1) + [ymax])

    def post_treat_chains(self):
        idxrdn = np.random.permutation(
            np.minimum(self.MCalgo.cut_chain.shape[0],100))
        postpar_red = self.postpar[idxrdn]
        self.postY = np.array([[self.data_case.form_fit(
                            np.r_[[xx]], bb)[0] for bb in postpar_red[:,:-1]] \
                        for xx in self.data_case.xmes])
        self.postMAP = np.array([self.data_case.form_fit(np.r_[[xx]],
                             self.MCalgo.MAP[:-1]) for xx in self.data_case.xmes])
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

validator = QDoubleValidator(-10e8, 10e8, 2)

class ViewMainUI(QMainWindow):

    computeSignal = pyqtSignal()
    fitRegSignal = pyqtSignal()
    NMCMCSignal = pyqtSignal(int)
    NthinSignal = pyqtSignal(int)
    NburnSignal = pyqtSignal(int)
    saveFileSignal = pyqtSignal()

    selectCaseSignal = pyqtSignal(str)
    openFileSignal = pyqtSignal()

    selectRegModelSignal = pyqtSignal(str)

    selectParamTuneSignal = pyqtSignal(int)
    selectDistTypeSignal = pyqtSignal(str)
    editPar1Signal = pyqtSignal(float)
    editPar2Signal = pyqtSignal(float)

    selectDimRSignal = pyqtSignal(int)
    selectDim1Signal = pyqtSignal(int)
    selectDim2Signal = pyqtSignal(int)

    radio1Signal = pyqtSignal(bool)
    radio2Signal = pyqtSignal(bool)

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
        self.check6 = 0
        self._update_param_selection(int(3+1)) # initially polynomial case
        self.activeParamTune = 0
        self.activePar1 = 0
        self.activePar2 = 1

        # compute panel
        self.ui.pushCompute.clicked.connect(self._handle_pushCompute)
        self.ui.lineEdMC1.textChanged.connect(self._handle_NMCMC)
        self.ui.lineEdMC2.textChanged.connect(self._handle_Nthin)
        self.ui.lineEdMC3.textChanged.connect(self._handle_Nburn)
        self.ui.radioStepType1.toggled.connect(self._handle_radio1)
        self.ui.radioStepType2.toggled.connect(self._handle_radio2)
        self.ui.pushExport.clicked.connect(self._handle_pushExport)
        # case panel
        self.ui.selectDataCase.activated.connect(self._handle_select_case)
        self.ui.pushImportData.clicked.connect(self._handle_pushImport)
        # reg panel
        self.ui.selectRegMod.activated.connect(self._handle_select_regmod)
        self.ui.pushFitReg.clicked.connect(self._handle_pushFitReg)
        # bayes panel
        self.ui.selectParamTune.activated.connect(self._handle_select_param)
        self.ui.selectDistType.activated.connect(self._handle_select_dist_type)
        self.ui.lineEdpar1.textChanged.connect(self._handle_Edpar1)
        # self.ui.lineEdpar1.setValidator(validator)
        self.ui.lineEdpar2.textChanged.connect(self._handle_Edpar2)
        # self.ui.lineEdpar2.setValidator(validator)
        # display panel
        self.ui.selectDimR.currentTextChanged.connect(self._handle_select_dimR)
        self.ui.selectDimR.activated.connect(self._handle_select_dimR)
        self.ui.selectDim2.currentTextChanged.connect(self._handle_select_dim2)
        self.ui.selectDim1.activated.connect(self._handle_select_dim1)
        self.ui.selectDim2.currentTextChanged.connect(self._handle_select_dim2)
        self.ui.selectDim2.activated.connect(self._handle_select_dim2)
        self.ui.checkLegend1.clicked.connect(self._handel_checkLegend1)
        self.ui.checkLegend2.clicked.connect(self._handel_checkLegend2)
        self.ui.checkLegend3.clicked.connect(self._handel_checkLegend3)
        self.ui.checkLegend4.clicked.connect(self._handel_checkLegend4)
        self.ui.checkLegend5.clicked.connect(self._handel_checkLegend5)
        self.ui.checkLegend6.clicked.connect(self._handel_checkLegend6)

    def _handle_pushCompute(self):
        self.computeSignal.emit()
    def _handle_NMCMC(self):
        value = int(self.ui.lineEdMC1.text())
        self.NMCMCSignal.emit(value)
    def _handle_Nthin(self):
        value = int(self.ui.lineEdMC2.text())
        self.NthinSignal.emit(value)
    def _handle_Nburn(self):
        value = int(self.ui.lineEdMC3.text())
        self.NburnSignal.emit(value)
    def _handle_radio1(self):
        value = self.ui.radioStepType1.isChecked()
        self.radio1Signal.emit(value)
    def _handle_radio2(self):
        value = self.ui.radioStepType2.isChecked()
        self.radio2Signal.emit(value)

    def _handle_select_case(self):
        value = self.ui.selectDataCase.currentText()
        self.selectCaseSignal.emit(value)

    def _handle_select_regmod(self):
        value = self.ui.selectRegMod.currentText()
        self.selectRegModelSignal.emit(value)
    def _handle_pushFitReg(self):
        self.fitRegSignal.emit()

    def _handle_select_param(self):
        value = self.ui.selectParamTune.currentIndex()
        self.selectParamTuneSignal.emit(value)
    def _handle_select_dist_type(self):
        value = self.ui.selectDistType.currentText()
        self.selectDistTypeSignal.emit(value)
    def _handle_Edpar1(self):
        try:
            value = float(self.ui.lineEdpar1.text())
        except:
            value = 0
        self.editPar1Signal.emit(value)
    def _handle_Edpar2(self):
        try:
            value = float(self.ui.lineEdpar2.text())
        except:
            value = 0
        self.editPar2Signal.emit(value)

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
    def _update_param_selection(self, dim):
        self.ui.selectParamTune.clear()
        for i in range(dim):
            self.ui.selectParamTune.addItem(str(i))

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
    def _handel_checkLegend6(self, value):
        self.check6 = value

    def _handle_pushImport(self):
        self.openFileSignal.emit()
    def _handle_pushExport(self):
        self.saveFileSignal.emit()


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

        self.view.NMCMCSignal.connect(self._setNMCMC)
        self.view.NthinSignal.connect(self._setNthin)
        self.view.NburnSignal.connect(self._setNburn)
        self.view.radio1Signal.connect(self._setradio1)
        self.view.radio2Signal.connect(self._setradio2)
        self.view.saveFileSignal.connect(self.saveFileNameDialog)

        self.view.selectCaseSignal.connect(self._select_case)
        self.view.openFileSignal.connect(self.openFileNameDialog)

        self.view.selectRegModelSignal.connect(self._select_reg_model)
        self.view.fitRegSignal.connect(self._handle_pushFitReg)

        self.view.selectParamTuneSignal.connect(self._select_param_tune)
        self.view.selectDistTypeSignal.connect(self._select_dist_type)
        self.view.editPar1Signal.connect(self._setEdpar1)
        self.view.editPar2Signal.connect(self._setEdpar2)

        self.view.selectDimRSignal.connect(self._select_dimR)
        self.view.selectDim1Signal.connect(self._select_dim1)
        self.view.selectDim2Signal.connect(self._select_dim2)
        self.view.ui.checkLegend1.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend2.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend3.clicked.connect(self.draw_plot_tabs)  
        self.view.ui.checkLegend4.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend5.clicked.connect(self.draw_plot_tabs)
        self.view.ui.checkLegend6.clicked.connect(self.draw_plot_tabs)


    def _handle_pushCompute(self):
        if self.model.type_inf:
            self.model.MCalgo = MHalgo(N=self.model.NMCMC,
                                    Nthin=self.model.Nthin,
                                    Nburn=self.model.Nburn,
                                    is_adaptive=True,
                                    verbose=self.model.verbose)
        else:
            self.model.MCalgo = MHwGalgo(N=self.model.NMCMC,
                                    Nthin=self.model.Nthin,
                                    Nburn=self.model.Nburn,
                                    is_adaptive=True,
                                    verbose=self.model.verbose)
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
        self.view.ui.sc1.axes.clear()
        if self.view.check4:
            self.view.ui.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postYeps[:100,0], '.b', label="posterior with noise")
            self.view.ui.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postY[:100,0], '.k', label="posterior prediction")
            self.view.ui.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postYeps[:100], '.b')
            self.view.ui.sc1.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                    self.model.postY[:100], '.k')
        if self.view.check3: self.view.ui.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    np.ravel(self.model.postMAP[:100]), '.g',
                    label="MAP")
        if self.view.check5: self.view.ui.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    self.model.yreg_pred[:100], '.', color='orange',
                    label="regmod")
        if self.view.check1: self.view.ui.sc1.axes.plot(
                    self.model.data_case.xmes[:100,self.view.dimR],
                    np.ravel(self.model.data_case.ymes[:100]), 'or',
                    label="train values") 
        if self.view.check2: self.view.ui.sc1.axes.plot(
                    self.model.data_case.X_test[:20,self.view.dimR],
                        np.ravel(self.model.data_case.y_test[:20]), 'sm', ms=3,
                        label="test values")
        self.view.ui.sc1.axes.legend()
        self.view.ui.sc1.draw()
        self.view.ui.sc2.axes.clear()
        self.view.ui.sc2.axes.scatter(self.model.MCsort[:,self.view.dim1],
                                   self.model.MCsort[:,self.view.dim2],
                                   c=self.model.LLsort, cmap="jet")
        self.view.ui.sc2.draw()
        self.view.ui.sc3.axes.clear()
        self.view.ui.sc3.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                np.ravel(self.model.data_case.ymes[:100]) - \
                                np.ravel(self.model.postMAP[:100]), '.g')
        self.view.ui.sc3.axes.plot(self.model.data_case.xmes[:100,self.view.dimR],
                                np.ravel(self.model.data_case.ymes[:100]) - \
                                np.ravel(self.model.yreg_pred[:100]), '.',
                                color='orange')
        self.view.ui.sc3.draw()
        self.view.ui.sc4.axes.clear()
        self.view.ui.sc4.axes.plot(self.model.MCalgo.MCchain[:,self.view.dim1])
        self.view.ui.sc4.draw()
        self.view.ui.sc5.axes.clear()
        self.view.ui.sc5.axes.hist(self.model.MCalgo.cut_chain[:,self.view.dim1],
                                   ec='k', alpha=0.6, bins=30)
        self.view.ui.sc5.twaxes.clear()
        xlims = self.view.ui.sc5.axes.get_xlim()
        if self.view.check6:
            if self.view.dim1 < len(self.model.bstart)-1:
                xp = np.linspace(self.model.rndUs[self.view.dim1].min,
                                    self.model.rndUs[self.view.dim1].max, 100)
                self.view.ui.sc5.twaxes.plot(xp, 
                                np.exp([self.model.rndUs[self.view.dim1].logprior(
                                    x) for x in xp]), 'r')
            else:
                xp = np.linspace(self.model.rnds.min,
                        self.model.rnds.max, 100)
                self.view.ui.sc5.twaxes.plot(xp, np.exp([self.model.rnds.logprior(
                                    x) for x in xp]), 'r')
        else:
            self.view.ui.sc5.twaxes.clear()
            self.view.ui.sc5.twaxes.set_xlim(xlims)
        self.view.ui.sc5.draw()

    @pyqtSlot(int)
    def _setNMCMC(self, value):
        self.model.NMCMC = value
    @pyqtSlot(int)
    def _setNthin(self, value):
        self.model.Nthin = value
    @pyqtSlot(int)
    def _setNburn(self, value):
        self.model.Nburn = value
    @pyqtSlot(bool)
    def _setradio1(self, value):
        self.model.type_inf = value
        self.view.ui.radioStepType1.setChecked(value)
        self.view.ui.radioStepType2.setChecked(not(value))
    @pyqtSlot(bool)
    def _setradio2(self, value):
        self.model.type_inf = not(value)
        self.view.ui.radioStepType1.setChecked(not(value))
        self.view.ui.radioStepType2.setChecked(value)

    def saveFileNameDialog(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.view, "Save File", "", "CSV Files (*.csv)")
        if file_path:
            self._export_result(file_path)

    @pyqtSlot(str)
    def _select_case(self, value):
        self.view.ui.selectDimR.setCurrentIndex(0)
        self.model.data_selected_case = value
        self.model.load_case()
        self.view._update_param_selection(len(self.model.rndUs)+1)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.view,
            "Open Data File", "", "CSV Files (*.csv);;All Files (*.*)",
            options=options
        )
        if file_name:
            self._handle_custom_case(file_name)

    def _handle_custom_case(self, file_name):
            imported_data = np.loadtxt(file_name, skiprows=0, delimiter=",")
            if self.view.ui.selectDataCase.findText("Custom") == -1:
                self.view.ui.selectDataCase.addItem("Custom")
            else: self.view.ui.selectDataCase.setItemText(2,"Custom")
            self.model.data_selected_case = "Custom"
            XX = imported_data[:,:-1]
            yy = imported_data[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(
                XX, yy, test_size=0.95, random_state=42)
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.model.custom_case.xmes = X_train
            self.model.custom_case.ymes = y_train
            self.model.custom_case.X_test = X_test
            self.model.custom_case.y_test = y_test

    def _export_result(self, file_name):
        np.savetxt(file_name[:-4] + "_par.csv",
                    np.vstack([self.model.postpar, 
                    self.model.MCalgo.MAP]))
        np.savetxt(file_name[:-4] + "_postY.csv",
                   self.model.postY)
        np.savetxt(file_name[:-4] + "_postYeps.csv",
                   self.model.postYeps)

    @pyqtSlot(str)
    def _select_reg_model(self, value):
        self.model.selected_model = value

    @pyqtSlot(int)
    def _select_param_tune(self, value):
        self.view.activeParamTune = value
        if self.view.activeParamTune < len(self.model.rndUs):
            self.view.ui.lineEdpar1.setText(
                str(self.model.rndUs[self.view.activeParamTune].param[0]))
            self.view.ui.lineEdpar2.setText(
                str(self.model.rndUs[self.view.activeParamTune].param[1]))
            if isinstance(self.model.rndUs[self.view.activeParamTune], NormVar):
                self.view.ui.selectDistType.setCurrentIndex(0)
            elif isinstance(self.model.rndUs[self.view.activeParamTune], UnifVar):
                self.view.ui.selectDistType.setCurrentIndex(1)
            elif isinstance(self.model.rndUs[self.view.activeParamTune], HalfNormVar):
                self.view.ui.selectDistType.setCurrentIndex(2)
        else:
            self.view.ui.lineEdpar1.setText(
                str(self.model.rnds.param))
            self.view.ui.lineEdpar2.setText("")
            self.view.ui.selectDistType.setCurrentIndex(2)
        
    @pyqtSlot(str)
    def _select_dist_type(self, value):
        if self.view.activeParamTune < len(self.model.rndUs):
            if value == "Normal":
                self.model.rndUs[self.view.activeParamTune] = \
                    NormVar([self.view.activePar1, self.view.activePar2])
            if value == "Uniform":
                self.model.rndUs[self.view.activeParamTune] = \
                    UnifVar([self.view.activePar1, self.view.activePar2])
            if value == "Half-Normal":
                self.model.rndUs[self.view.activeParamTune] = \
                    HalfNormVar(self.view.activePar1)
        else:
            self.model.rnds = HalfNormVar(self.view.activePar1)
            if (value == "Normal") | (value == "Uniform") :
                print("Error parameter should only be Half-Normal")  

    @pyqtSlot(float)
    def _setEdpar1(self, value):
        self.view.activePar1 = value

    @pyqtSlot(float)
    def _setEdpar2(self, value):
        self.view.activePar2 = value

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
        if self.model.type_inf: inf_res = "Block Step"
        else: inf_res = "Single Step"
        formatted_list = [f"{x:.2f}" for x in self.model.MCalgo.MAP]
        self.view.ui.labelResult1.setText(
            inf_res + "\n" +
            "Acceptation:\n" +
            str([f"{x:.2f}" for x in self.model.MCalgo.tacc]) + 
            "\n" +
            "MAP:\n" +
            "\n".join(formatted_list))


###############################################################################
### COMPUTE WORK OBJECT
###############################################################################

class ComputeWorker(QObject):
    finished = pyqtSignal()
    # progress = pyqtSignal(int)
    ## TO-DO add progress bar (need extract current i from MCalgo.runInference)
    # or pass a callback (as a signal emitter)

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
    
