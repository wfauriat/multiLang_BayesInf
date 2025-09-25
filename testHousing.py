#%%

from sklearn.datasets import fetch_california_housing
import numpy as np
import scipy.stats as sst
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

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




#%%

# housing = fetch_california_housing()
# # print(housing.data.shape, housing.target.shape)
# # print(housing.feature_names[0:6])
# df = pd.DataFrame(housing.data, columns=housing.feature_names)

datacsv = pd.read_csv('cal_housing.data', header=None, dtype='f8')
tmp = np.loadtxt('cal_housing.domain', delimiter=':', dtype='U20')

df = pd.DataFrame(datacsv.values, columns=tmp[:,0])

df.values[:,3] = df.values[:,3]/df['households'].values
df.values[:,4] = df.values[:,4]/df['households'].values 
df.values[:,6] = df['population'].values/df.values[:,6]

census = np.vstack([df['households'].values<10,
df['totalRooms'].values<15,
df['totalBedrooms'].values<4, 
df['population'].values<10000])

censust = np.ravel([all(el) for el in census.T])

valcens = df.values[censust,:]
dfcens = pd.DataFrame(valcens, columns=df.columns)

# %%

dim = 8
thresh = 1e6
# filtdata = df.iloc[df.iloc[:,dim].values<thresh,dim]
# filtdata = dfcens.values[:,dim]

# fig, ax = plt.subplots()
# ax.hist(filtdata, ec='k', bins=20)
# ax.set_xlabel(df.columns[dim])


filtsorted = dfcens.values[np.argsort(dfcens.iloc[:,dim]),:]
fig, ax = plt.subplots()
cmap = ax.scatter(filtsorted[:,0],filtsorted[:,1], c=filtsorted[:,dim],
            cmap='jet', marker='.')
fig.colorbar(cmap, ax=ax, label=df.columns[dim])


# %%

fig, ax = plt.subplots()
cmap = ax.scatter(dfcens['medianIncome'].values,
                   dfcens['medianHouseValue'].values, 
                  c=dfcens['totalRooms'].values, 
           marker='.')
fig.colorbar(cmap, ax=ax, label='totalRooms')

#%%

# Index(['longitude', 'latitude', 'housingMedianAge', 'totalRooms',
#        'totalBedrooms', 'population', 'households', 'medianIncome',
#        'medianHouseValue'],
#       dtype='object')#%%

# sns.regplot(dfcens['housingMedianAge'], dfcens['medianHouseValue'],
#              line_kws={'color':'r'})

# %%

dims = len(dfcens.columns)

fig, ax = plt.subplots(2,int(dims/2), figsize=(10,4))
for i in range(int(dims/2)):
    ax[0,i].hist(dfcens.iloc[:,i])
    ax[0,i].set_yticks([])
    ax[0,i].set_xlabel(dfcens.columns[i])
for i in range(int(dims/2),dims-1):
    ax[1,i-int(dims/2)].hist(dfcens.iloc[:,i])
    ax[1,i-int(dims/2)].set_yticks([])
    ax[1,i-int(dims/2)].set_xlabel(dfcens.columns[i])
fig.tight_layout()

# %%

dims = len(dfcens.columns)

fig, ax = plt.subplots(2,int(dims/2), figsize=(10,4))
for i in range(int(dims/2)):
    sns.regplot(dfcens.iloc[:,i],  dfcens['medianHouseValue'],
                line_kws={'color':'r'}, ax=ax[0,i])
    ax[0,i].set_xlabel(dfcens.columns[i])
for i in range(int(dims/2),dims-1):
    sns.regplot(dfcens.iloc[:,i],  dfcens['medianHouseValue'],
                line_kws={'color':'r'}, ax=ax[1,i-int(dims/2)])
    ax[1,i-int(dims/2)].set_xlabel(dfcens.columns[i])
fig.tight_layout()

# %%
