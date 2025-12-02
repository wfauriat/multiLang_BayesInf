
import numpy as np

from cases_data.data_cases_def import (
    VoidCase, PolynomialCase, HousingCase)

from pyBI.base import (
    UnifVar, NormVar, HalfNormVar, ObsVar)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


###############################################################################
### MODEL OBJECT
###############################################################################

class BI_Model():
    def __init__(self):
        super().__init__()
        self.data_selected_case = "Polynomial"
        self.NMCMC = 10000
        self.Nthin = 20
        self.Nburn = 5000
        self.type_inf = 1
        self.verbose = True
        self.MCalgo = None
        self.MCsort = None
        self.LLsort = None
        self.postpar = None
        self.postMAP = None
        self.postY = None
        self.postYeps = None
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

