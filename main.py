#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst

#%%

y = sst.norm().rvs(30)

fig, ax = plt.subplots()
ax.hist(y, ec='k')

# %%
