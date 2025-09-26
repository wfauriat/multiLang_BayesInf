#%%

from sklearn.datasets import fetch_california_housing
import numpy as np
import scipy.stats as sst
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

np.set_printoptions(suppress=True, precision=4)

#%%

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


#%%

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

# %%

dim = 8

filtsorted = dfcens.values[np.argsort(dfcens.iloc[:,dim]),:]
fig, ax = plt.subplots()
ax.imshow(img)
cmap = ax.scatter((filtsorted[:,0]+120)*70+354,
                  -(filtsorted[:,1]-38)*70+312, c=filtsorted[:,dim],
            cmap='jet', marker='.', s=10, alpha=1)
fig.colorbar(cmap, ax=ax, label=df.columns[dim])


# # %%

# fig, ax = plt.subplots()
# cmap = ax.scatter(dfcens['medianIncome'].values,
#                    dfcens['medianHouseValue'].values, 
#                   c=dfcens['totalRooms'].values, 
#            marker='.')
# fig.colorbar(cmap, ax=ax, label='totalRooms')


# %%
# HISTOGRAMS

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

# %%
# 1D REGRESSIONS

dims = len(dfcens.columns)

fig, ax = plt.subplots(2,int(dims/2), figsize=(10,4))
for i in range(int(dims/2)):
    sns.regplot(x=dfcens.iloc[:,i],  y=dfcens['medianHouseValue'],
                line_kws={'color':'r'}, ax=ax[0,i])
    ax[0,i].set_xlabel(dfcens.columns[i])
for i in range(int(dims/2),dims-1):
    sns.regplot(x=dfcens.iloc[:,i],  y=dfcens['medianHouseValue'],
                line_kws={'color':'r'}, ax=ax[1,i-int(dims/2)])
    ax[1,i-int(dims/2)].set_xlabel(dfcens.columns[i])
fig.tight_layout()

# %%
# TEST WITH VARIOUS MODELS

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# model = Ridge(alpha=0.01)
# model = LinearRegression()
model = ElasticNet()
# model = DecisionTreeRegressor()
# model = RandomForestClassifier(n_estimators=50, n_jobs=2, max_depth=30)

dims = len(dfcens.columns)
# XX = dfcens.values[:,:dims-1]
XX = dfcens.values[:,2:-1]
yy = dfcens.values[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.minimum(y_pred, 500000)

r2 = r2_score(y_test, y_pred)
mue, sigmae = sst.norm.fit(y_test-y_pred)

#%%

vdim = 5
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
ax[0,1].plot(y_test, y_test-y_pred, '.b')
ax[0,1].set_title('residuals')
ax[1,1].hist(y_test-y_pred, ec='k', bins=30)
ax[1,1].grid(ls=':')
ax[1,1].set_yticks([])
ax[1,1].set_title('residuals')
ax1 = ax[1,1].twinx()
sns.kdeplot(y_test-y_pred, ax=ax1, color='r', lw=2)
ax1.plot(np.sort(y_test-y_pred),
         sst.norm(mue, sigmae).pdf(np.sort(y_test-y_pred)), color='m', lw=3)
ax1.set_yticks([])
ax1.set_ylabel('')

print(f"R-squared score on test set: {r2:.2f}")
print(f"Std of error is : {sigmae:.2f}")

