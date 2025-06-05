import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as sst

data = np.loadtxt('data.txt', delimiter=',')
LL = np.loadtxt('LL.txt')
idL = np.argsort(LL)

plt.close('all')
fig, ax = plt.subplots(3,3)
for j in range(3):
    for i in range(3):
        if i<j:
            ax[i,j].scatter(data[idL,j], data[idL,i],
                             c=LL[idL], marker='.', cmap='jet')
        elif i==j:
            ax[i,j].hist(data[:,j], edgecolor='k')
        else:
            ax[i,j].set_visible(False)
plt.show()


# MAP = np.loadtxt('MAP.txt', delimiter=',')

# rndfit = sst.multivariate_normal(
#     mean=np.array([MAP[0], MAP[1]]),
#     cov=np.array([[MAP[2]**2, MAP[2]*MAP[3]*MAP[4]],
#                   [MAP[2]*MAP[3]*MAP[4], MAP[3]**2]])
# ).rvs(100)

# plt.close('all')
# fig, ax = plt.subplots()
# ax.scatter(data[:,0], data[:,1], marker='.', color='k', label='observed')
# ax.scatter(rndfit[:,0], rndfit[:,1], marker='.', color='b', label='generated')
# ax.legend()
# plt.show()