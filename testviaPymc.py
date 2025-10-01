#%%###########################################################################
## IMPORT PACKAGES
##############################################################################

import numpy as np
np.set_printoptions(suppress=True, precision=5)
import scipy.stats as sst
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PIL import Image

import pymc as pm

#%%###########################################################################
## LOADING DATA AND MAP AND PREPROCESSING
##############################################################################

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

arrdata = np.array(dfcens.values, dtype=float)
dim = arrdata.shape[1] - 1
XX = arrdata[:,:dim]
yy = arrdata[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.8)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

#%%###########################################################################
## VISUALISATION ON MAP
##############################################################################

filtsorted = dfcens.values[np.argsort(dfcens.values[:,-1]),:]
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((filtsorted[:,0]+120)*70+354,
                  -(filtsorted[:,1]-38)*70+312, c=filtsorted[:,-1],
            cmap='jet', marker='.', s=10, alpha=1,
            vmin=0, vmax=500000)
fig.colorbar(cmap, ax=ax, label=df.columns[-1])


#%%###########################################################################
## DEFINITION OF PYMC OBJECTS
##############################################################################

import pymc as pm
import pytensor.tensor as pt


with pm.Model() as linear_model:
    intercept = pm.Normal("intercept", mu=0, sigma=1000)
    weights = pm.Normal("weights", mu=0, sigma=500, shape=dim)
    sigma = pm.HalfNormal("sigma", sigma=2000)
    mu = pt.dot(X_train, weights) + intercept
    # mu = pm.Deterministic("mu", intercept + pt.dot(XX, weights)) # other formulation
    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_train)


with linear_model:
    # Run the No-U-Turn Sampler (NUTS)
    idata = pm.sample(draws=2000, tune=2000, target_accept=0.8)
    # idata = pm.sample(draws=30000, tune=5000, step=pm.Metropolis())

print("\n--- Model Summary ---")
pm.summary(idata)

#%%###########################################################################
## PLOT INFERENCE RESULTS AND GIVE MAP
##############################################################################

import arviz as az

az.plot_trace(idata)

#%%

log_posterior = idata.sample_stats["lp"].values
flat_lp = log_posterior.flatten()
map_index_flat = np.argmax(flat_lp)

map_params = {}
for var_name, var_data in idata.posterior.items():
    flat_data = var_data.values.reshape(-1, *var_data.shape[2:])
    map_value = flat_data[map_index_flat]
    map_params[var_name] = map_value

print(*[str(el) + '\n' for el in map_params.values()])

# #%%
# trace0 = idata.posterior.data_vars['weights'].sel(chain=0)
# trace1 = idata.posterior.data_vars['weights'].sel(chain=1)

# vdim = 0
# fig, ax = plt.subplots()
# ax.plot(trace0[:,vdim])
# ax.plot(trace1[:,vdim])


#%%###########################################################################
## VISUALIZE SEVERAL CHAIN ON ONE DIMENSION
##############################################################################

trace0 = idata.posterior.sel(chain=0)['weights']
trace1 = idata.posterior.sel(chain=1)['weights']
trace2 = idata.posterior.sel(chain=2)['weights']
trace3 = idata.posterior.sel(chain=3)['weights']

vdim = 7
fig, ax = plt.subplots()
ax.plot(trace0[:,vdim])
ax.plot(trace1[:,vdim])
ax.plot(trace2[:,vdim])
ax.plot(trace3[:,vdim])


#%%###########################################################################
## COMPUTE AND VISUALIZE PREDICTIONS
##############################################################################

ypred = map_params['intercept'] + X_test @ map_params['weights']


fig, ax = plt.subplots()
ax.imshow(img)

idsort = np.argsort(ypred)
cbar = ax.scatter((X_test[idsort,0]+120)*70+354,
                 -(X_test[idsort,1]-38)*70+312,
                  c=ypred[idsort],
                   marker='.', cmap='jet', s=10, vmin=0, vmax=500000)
fig.colorbar(cbar, ax=ax)


#%%###########################################################################
## COMPUTE AND (APPROX) UNCERTAINTY IN PREDICTIONS (MAP + ERROR MAGNITUDE)
##############################################################################

smag = np.std(np.abs(sst.norm(loc=0, scale=map_params['sigma']).rvs(5000)))

idsort = np.argsort(X_test[:100,-1])
fig, ax = plt.subplots()
ax.plot(X_test[idsort,-1], y_test[idsort], 'or')
ax.plot(X_test[idsort,-1], ypred[idsort], '.b')
ax.plot(X_test[idsort,-1], ypred[idsort] + 2*smag, '+--b')
ax.plot(X_test[idsort,-1], ypred[idsort] - 2*smag, '+--b')


#%%###########################################################################
## COMPUTE AND UNCERTAINTY FROM FULL POSTERIOR
##############################################################################

postpar = np.hstack([
    idata.posterior.sel(chain=0)['intercept'].values[None,:].T,
    idata.posterior.sel(chain=0)['weights'].values,
    idata.posterior.sel(chain=0)['sigma'].values[None,:].T
])

def predy(x,b):
    return b[0] + np.sum(b[1:9]*x)

xtry = X_test[:100,:]
postY = np.array([[predy(xx, bb) for bb in postpar] for xx in xtry]).T
postYeps = postY + \
            sst.norm(loc=0, scale=postpar[:,-1]).rvs(size=(100,2000)).T

fig, ax = plt.subplots()
ax.plot(xtry[:,-1], postYeps.T, '.b')
ax.plot(xtry[:,-1], postY.T, '.k')
ax.plot(xtry[:,-1], y_test[:100] ,'or')


#%%###########################################################################
## MANUAL MAP
##############################################################################
manuMAP = {}
manuMAP['intercept'] = np.array([-333.1999960686905])
manuMAP['weights'] = np.array([-2069.93974,-4032.20164,1171.66251,
                                3856.20045, -137.02971, 1.4767,
                                -3553.31995, 15159.43436])
manuMAP['sigma'] = np.array(79482.6208348724)

yMAP = manuMAP['intercept'] + X_test @ manuMAP['weights']
 


