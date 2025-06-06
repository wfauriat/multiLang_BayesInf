import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as sst

def modeltrue(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 1*x[:,1]

def modelfit(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2

b0 = [2, -1, 2, 0]
nslvl = 0.2

smod = 0.2

xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                 np.c_[sst.norm(loc=0, scale=1).rvs(10)]])
ymes = modeltrue(xmes, b0)
ymes += sst.norm().rvs(xmes.shape[0])*nslvl

data = np.loadtxt('data.txt', delimiter=',')
LL = np.loadtxt('LL.txt')
idL = np.argsort(LL)
Ndim = data.shape[1]
MAP = data[idL[-1],:]

postY = np.array([sst.norm(loc=modelfit(xmes, data[i,:Ndim]),
                            scale=smod).rvs(xmes.shape[0])
                              for i in range(data.shape[0])])

print('MAP: ', MAP)

# plt.close('all')
fig, ax = plt.subplots(4,4)
for j in range(4):
    for i in range(4):
        if i<j:
            ax[i,j].scatter(data[idL,j], data[idL,i],
                             c=LL[idL], marker='.', cmap='jet')
            ax[i,j].plot(MAP[j], MAP[i], 'dk')
        elif i==j:
            ax[i,j].hist(data[:,j], edgecolor='k')
        else:
            ax[i,j].set_visible(False)

fig.savefig("cpppost.png", dpi=200)

fig2, ax2 = plt.subplots()
ax2.plot(xmes[:,0], postY[-1:,:].T, '.k', label='posterior calibrated')
ax2.plot(xmes[:,0], postY.T, '.k')
ax2.plot(xmes[:,0], ymes, '.r', label='available observation')

fig2.savefig("cpppostobs.png", dpi=200)

# plt.show()
