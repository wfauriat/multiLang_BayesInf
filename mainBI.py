#%%
import pyBI as pbi

import numpy as np
import matplotlib.pyplot as plt


#%%

def modeltrue(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 1*x[:,1]

def modelfit(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2


b0 = [2, -1, 2, 0]
nslvl = 0.2
nsp1 = 0.1
biasp1 = -1

xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
xplot[:,1] = 1

xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                 biasp1+nsp1*np.c_[np.random.randn(10)]])
ymes = modeltrue(xmes, b0)
ymes += np.random.randn(xmes.shape[0])*nslvl
