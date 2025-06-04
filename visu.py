import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as sst

data = np.loadtxt('data.txt', delimiter=',')
MAP = np.loadtxt('MAP.txt', delimiter=',')

rndfit = sst.multivariate_normal(
    mean=np.array([MAP[0], MAP[1]]),
    cov=np.array([[MAP[2]**2, MAP[2]*MAP[3]*MAP[4]],
                  [MAP[2]*MAP[3]*MAP[4], MAP[3]**2]])
).rvs(100)

plt.close('all')
fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], marker='.', color='k', label='observed')
ax.scatter(rndfit[:,0], rndfit[:,1], marker='.', color='b', label='generated')
ax.legend()
plt.show()