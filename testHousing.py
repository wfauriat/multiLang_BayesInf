#%%###########################################################################
## IMPORT PACKAGES
##############################################################################
from sklearn.datasets import fetch_california_housing
import numpy as np
import scipy.stats as sst
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


np.set_printoptions(suppress=True, precision=4)


#%%###########################################################################
## LOADING DATA AND MAP AND PREPROCESSING
##############################################################################

# Number of Instances:    20640
# Number of Attributes:   8 numeric, predictive attributes and the target
# Attribute Information:  MedInc median income in block group
#                         HouseAge median house age in block group
#                         AveRooms average number of rooms per household
#                         AveBedrms average number of bedrooms per household
#                         Population block group population
#                         AveOccup average number of household members
#                         Latitude block group latitude
#                         Longitude block group longitude

#['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
#        'totalBedrooms', 'population', 'households', 'medianIncome',
#        'medianHouseValue']

img = Image.open('cal_map.jpg')

# housing = fetch_california_housing()
# df = pd.DataFrame(housing.data, columns=housing.feature_names)

datacsv = pd.read_csv('cal_housing.data', header=None, dtype='f8')
tmp = np.loadtxt('cal_housing.domain', delimiter=':', dtype='U20')

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

#%%###########################################################################
## VISUALISATION ON MAP
##############################################################################
dim = 8

filtsorted = dfcens.values[np.argsort(dfcens.iloc[:,dim]),:]
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((filtsorted[:,0]+120)*70+354,
                  -(filtsorted[:,1]-38)*70+312, c=filtsorted[:,dim],
            cmap='jet', marker='.', s=10, alpha=1)
fig.colorbar(cmap, ax=ax, label=df.columns[dim])





#%%###########################################################################
## HISTOGRAMS
##############################################################################

dims = len(dfcens.columns)

fig, ax = plt.subplots(2,int(dims/2), figsize=(10,4))
for i in range(int(dims/2)):
    ax[0,i].hist(dfcens.iloc[:,i], ec='k')
    ax[0,i].set_yticks([])
    ax[0,i].set_xlabel(dfcens.columns[i])
for i in range(int(dims/2),dims-1):
    ax[1,i-int(dims/2)].hist(dfcens.iloc[:,i], ec='k')
    ax[1,i-int(dims/2)].set_yticks([])
    ax[1,i-int(dims/2)].set_xlabel(dfcens.columns[i])
fig.tight_layout()

#%%###########################################################################
## 1D REGRESSIONS
##############################################################################
# dims = len(dfcens.columns)

# fig, ax = plt.subplots(2,int(dims/2), figsize=(10,4))
# for i in range(int(dims/2)):
#     sns.regplot(x=dfcens.iloc[:,i],  y=dfcens['medianHouseValue'],
#                 line_kws={'color':'r'}, ax=ax[0,i])
#     ax[0,i].set_xlabel(dfcens.columns[i])
# for i in range(int(dims/2),dims-1):
#     sns.regplot(x=dfcens.iloc[:,i],  y=dfcens['medianHouseValue'],
#                 line_kws={'color':'r'}, ax=ax[1,i-int(dims/2)])
#     ax[1,i-int(dims/2)].set_xlabel(dfcens.columns[i])
# fig.tight_layout()

#%%###########################################################################
## TEST WITH VARIOUS REGRESSION MODELS
##############################################################################

dims = len(dfcens.columns)
XX = dfcens.values[:,:dims-1]
# XX = np.float32(dfcens.values[:,2:-1])
yy = np.float32(dfcens.values[:,-1])

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.90)

X_lat_test = X_test[:,:2]
# X_train = X_train[:,2:] # to remove learning on latitude / longitude
# X_test = X_test[:,2:] # to remove learning on latitude / longitude

dimtrain = X_test.shape[1]

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
X_test_scaled = scaler_X.transform(X_test)

# model = Ridge(alpha=0.01)
model = LinearRegression()
# model = ElasticNet()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()

kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0]*dimtrain, (1e-2, 1e2)) + \
      WhiteKernel(noise_level_bounds=(1e-3, 1e3))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=1e-5, n_restarts_optimizer=0)

svr = SVR(
    kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50), # 2 hidden layers with 100 and 50 neurons
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=300,
        random_state=42
    ))
])

# mlp_pipeline.fit(X_train, y_train)

model.fit(X_train, y_train)
# svr.fit(X_train_scaled, y_train_scaled)
# gp.fit(X_train, y_train)

y_pred = model.predict(X_test)
# y_pred = mlp_pipeline.predict(X_test)
# y_pred, sigma = gp.predict(X_test, return_std=True)
# y_pred_scaled = svr.predict(X_test_scaled)
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_pred = np.minimum(y_pred, 500000)

r2 = r2_score(y_test, y_pred)

residuals = y_test-y_pred
mue, sigmae = sst.norm.fit(residuals)
mue1, sigmae1 = sst.norm.fit(np.abs(residuals[(y_test>0) & (y_test<100000)]))
mue2, sigmae2 = sst.norm.fit(np.abs(residuals[(y_test>100000) & (y_test<200000)]))
mue3, sigmae3 = sst.norm.fit(np.abs(residuals[(y_test>200000) & (y_test<300000)]))
mue4, sigmae4 = sst.norm.fit(np.abs(residuals[(y_test>300000) & (y_test<400000)]))
mue5, sigmae5 = sst.norm.fit(np.abs(residuals[(y_test>400000)]))


#%%###########################################################################
## VISUALISATION OF FIT AND RESIDUALS
##############################################################################

vdim = -1
nplot = 30
visu_samp = np.random.randint(X_test.shape[0],size=nplot)

fig, ax = plt.subplots(2,2,figsize=(10,7))
ax[0,0].plot(y_test, y_pred, '.b')
ax[0,0].plot(y_test[visu_samp], y_pred[visu_samp], '+m')
ax[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
ax[0,0].set_title('y-y')
ax[1,0].plot(X_test[visu_samp,vdim], y_test[visu_samp], '.r')
ax[1,0].plot(X_test[visu_samp,vdim], y_pred[visu_samp], '+m')
ax[1,0].stem(X_test[visu_samp,vdim], np.abs(y_test[visu_samp]- y_pred[visu_samp]),
              markerfmt='.k', linefmt='k')
ax[1,0].grid(ls=':')
ax[1,0].set_title('pred vs test')
ax[0,1].plot(y_test, np.abs(residuals), '.b')
ax[0,1].plot([0,100000],np.ones(2)*sigmae1,'o-r', lw=2)
ax[0,1].plot([100000,200000],np.ones(2)*sigmae2,'o-r', lw=2)
ax[0,1].plot([200000,300000],np.ones(2)*sigmae3,'o-r', lw=2)
ax[0,1].plot([300000,400000],np.ones(2)*sigmae4,'o-r', lw=2)
ax[0,1].plot([400000,500000],np.ones(2)*sigmae5,'o-r', lw=2)
ax[0,1].axhline(y=sigmae, color='k', ls='--')
ax[0,1].grid(ls=':')
ax[0,1].set_title('residuals')
ax[1,1].hist(residuals, ec='k', bins=30)
ax[1,1].grid(ls=':')
ax[1,1].set_yticks([])
ax[1,1].set_title('residuals')
ax1 = ax[1,1].twinx()
sns.kdeplot(residuals, ax=ax1, color='r', lw=2)
ax1.plot(np.sort(residuals),
         sst.norm(mue, sigmae).pdf(np.sort(residuals)), color='m', lw=3)
ax1.set_yticks([])
ax1.set_ylabel('')

print(f"R-squared score on test set: {r2:.2f}")
print(f"Std of error is : {sigmae:.2f}")
print(f"Std of abs error in [0, 100000k$] : {sigmae1:.2f}")
print(f"Std of abs error in [100000k$, 200000k$] : {sigmae2:.2f}")
print(f"Std of abs error in [200000k$, 300000k$] : {sigmae3:.2f}")
print(f"Std of abs error in [300000k$, 400000k$] : {sigmae4:.2f}")
print(f"Std of abs error in [400000k$, 500000k$] : {sigmae5:.2f}")



#%%###########################################################################
## VISUALISATION OF PREDICTION AND RESIDUALS ON MAP
##############################################################################

idsort = np.argsort(y_test) 

filtsorted = y_test[idsort]
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((X_lat_test[idsort,0]+120)*70+354,
                  -(X_lat_test[idsort,1]-38)*70+312, c=filtsorted,
            cmap='jet', marker='.', s=10, alpha=1, vmin=0)
fig.colorbar(cmap, ax=ax, label=df.columns[8])

idsort = np.argsort(y_pred) 

filtsorted = y_pred[idsort]
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((X_lat_test[idsort,0]+120)*70+354,
                  -(X_lat_test[idsort,1]-38)*70+312, c=filtsorted,
            cmap='jet', marker='.', s=10, alpha=1, vmin=0)
fig.colorbar(cmap, ax=ax, label=df.columns[8])

idsort = np.argsort(np.abs(residuals))

filtsorted = np.abs(residuals[idsort])
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((X_lat_test[idsort,0]+120)*70+354,
                  -(X_lat_test[idsort,1]-38)*70+312, c=filtsorted,
            cmap='jet', marker='.', s=10, alpha=1, vmin=0)
fig.colorbar(cmap, ax=ax, label='abs error')


#%%###########################################################################
## TEST OF BAYESIAN INFERENCE
##############################################################################

from itertools import chain

from pyBI.base import UnifVar, InvGaussVar, ObsVar
from pyBI.base import HGP, GaussLike
from pyBI.inference import MHalgo, MHwGalgo, InfAlgo, InfAlgo2, MHwGalgo2


Ndim = 8 + 1
# sinvg = [0.2, -0.1, 100000]
sinvg = [0.1, 10000, 1000000]

# sm = [100]*Ndim
sm = [1000, 100, 100, 10, 10, 10, 1, 10, 10]
smexp = 0.1
covProp = np.eye(Ndim)*1e-1
LLTprop = np.linalg.cholesky(covProp)

# rndUs = [UnifVar([-30000,30000]),
#          UnifVar([-30000,30000]),
#          UnifVar([-2000,2000]),
#          UnifVar([-2000,2000]),
#          UnifVar([-2000,2000]),
#          UnifVar([-20,20]),
#          UnifVar([-30000,20000]),
#          UnifVar([0,50000])
# ]
# rndUs = [UnifVar([-50000,50000]),
#          UnifVar([-50000,50000]),
#          UnifVar([-20000,20000]),
#          UnifVar([-20000,20000]),
#          UnifVar([-20000,20000]),
#          UnifVar([-200,200]),
#          UnifVar([-30000,30000]),
#          UnifVar([-50000,50000])
# ]
rndUs = [UnifVar([-3000,3000]),
         UnifVar([-3000,3000]),
         UnifVar([-5000,5000]),
         UnifVar([500,2000]),
         UnifVar([1000,5000]),
         UnifVar([-2000,2000]),
         UnifVar([-3,10]),
         UnifVar([-7000,-3000]),
         UnifVar([9000,13000])
]

def modelfit(x,b):
    return np.atleast_2d(b[0] + \
                         b[1]*x[:,0] +
                         b[2]*x[:,1] +
                         b[3]*x[:,2] +
                         b[4]*x[:,3] +
                         b[5]*x[:,4] +
                         b[6]*x[:,5] +
                         b[7]*x[:,6] + 
                         b[8]*x[:,7])[0][0]

rnds = InvGaussVar(param=sinvg) # A REMPLACER PAR HALFNORMAL

bstart = np.array([rndUs[i].draw() for i in range(Ndim)] + \
                    [float(rnds.draw())])

obsvar = ObsVar(obs=np.c_[y_train], prev_model=modelfit, cond_var=X_train)

NMCMC = 20000
Nburn = 5000
verbose = True

# MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
#                     verbose=verbose)
MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                    verbose=verbose)


tmp = obsvar.loglike(bstart[:Ndim],
                     rnds.diagSmat(s=bstart[Ndim],
                                N=obsvar.dimdata))
MCalgo.initialize(obsvar, rndUs, rnds)
MCalgo.sdisc = smexp
MCalgo.svar = sm
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCout, llout = MCalgo.runInference()


# %%

#%%############################################################################
# VISUALISATION OF INFERENCE RESULTS
###############################################################################

MCalgo.post_visupar()
MCalgo.hist_alldim()

print(MCalgo)
