import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PolynomialCase():
    def __init__(self):
        
        def modeltrue(x,b):
            return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + \
                                1*x[:,1] + 0.02*x[:,0]**3)
        def modelfit(x,b):
            return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)
        
        self.true_generator = modeltrue
        self.form_fit = modelfit
        self.b0 = [2, -1, 2, 0]
        self.nslvl = 0.1
        self.nsp1 = 0.2
        self.biasp1 = -1
        self.xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                        self.biasp1+self.nsp1*np.c_[np.random.randn(10)]])
        self.ymes = modeltrue(self.xmes, self.b0)
        self.ymes += np.random.randn(self.xmes.shape[0])*self.nslvl

class HousingCase():
    def __init__(self):
        img = Image.open('multiLang_BayesInf/cases_data/cal_map.jpg')
        datacsv = pd.read_csv('multiLang_BayesInf/cases_data/cal_housing.data',
                               header=None, dtype='f8')
        tmp = np.loadtxt('multiLang_BayesInf/cases_data/cal_housing.domain',
                          delimiter=':', dtype='U20')

        df = pd.DataFrame(datacsv.values, columns=tmp[:,0])

        df.values[:,3] = df.values[:,3]/df['households'].values
        df.values[:,4] = df.values[:,4]/df['households'].values 
        df.values[:,6] = df['population'].values/df.values[:,6]

        census = np.vstack([
            df['households'].values<10,
            df['totalRooms'].values<15,
            df['totalBedrooms'].values<4, 
            df['population'].values<10000])

        censust = np.ravel([all(el) for el in census.T])

        valcens = df.values[censust,:]
        dfcens = pd.DataFrame(valcens, columns=df.columns)
        dims = len(dfcens.columns)
        XX = dfcens.values[:,:dims-1]
        yy = np.float32(dfcens.values[:,-1])

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(
            XX, yy, test_size=0.95, random_state=42)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        def modelfit(x,b):
            return np.atleast_2d(b[0] + \
                                b[1]*x[:,0] +
                                b[2]*x[:,1] +
                                b[3]*x[:,2] +
                                b[4]*x[:,3] +
                                b[5]*x[:,4] +
                                b[6]*x[:,5] +
                                b[7]*x[:,6] + 
                                b[8]*x[:,7])[0]

        self.form_fit = modelfit