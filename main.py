#%%############################################################################
# REQUIREMENT PACKAGE IMPORT
###############################################################################
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as sst

# matplotlib.use('QtAgg')


#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

def model(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 4*x[:,1]

b0 = [2, -1, 2, 0]
nslvl = 0.2

xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
xplot[:,1] = 1

xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                 np.c_[sst.norm().rvs(10)]])
ymes = model(xmes, b0)
ymes += sst.norm().rvs(xmes.shape[0])*nslvl


#%%############################################################################
# DEFINITION OF BAYESIAN INFERENCE OBJECTS
###############################################################################

def loglike(obs, var, par, smod, model):
    return np.sum(
        sst.norm(loc=model(var, par), scale=smod).logpdf(obs))

def logprior(par, pl=-5, ph=5):
    return np.sum([
        sst.uniform(pl,ph-pl).logpdf(x) for x in par])

def rnv(m,s): return sst.norm(loc=m, scale=s).rvs()
def rnlv(m,s): return np.exp(sst.norm(loc=np.log(m), scale=s).rvs())


#%%############################################################################
# IMPLEMENTATION OF INFERENCE ALGORITHM
###############################################################################

## SPECIFICATION OF PRACTICAL APPLICATION
Ndim = 3
pl = -3
ph = 3

smod = sst.invgauss(0.2,1)

sexp = [0.5, 0.5, 0.5]
smexp = 0.01

bstart = np.array([sst.uniform(pl,ph-pl).rvs() for _ in range(Ndim)])

## PARAMETRIZATION OF MCMC
NMCMC = 12000
Nburn = 2000
Nthin = 10

## INITIALISATION OF MCMC
MCchain = np.zeros((NMCMC, Ndim+1))
llchain = np.zeros(NMCMC)
MCchain[0,:Ndim] = bstart
MCchain[0,Ndim] = smod.mean()
llchain[0] = loglike(ymes, xmes, MCchain[0,:Ndim], MCchain[0,Ndim],
                      model=model)
llold = llchain[0]
lpold = logprior(MCchain[0,:Ndim], pl, ph)
lsold = smod.logpdf(MCchain[0,Ndim])
lsold = 0

## RUN OF MCMC
for i in range(1,NMCMC):
    xprop = rnv(MCchain[i-1,:Ndim],sexp)
    so = MCchain[i-1,Ndim]
    sp = rnlv(so,smexp)
    llprop = loglike(ymes, xmes, xprop, sp, model=model)
    lpprop = logprior(xprop, pl, ph)
    lspp = smod.logpdf(sp)
    ldiff = llprop + lpprop + lspp - llold - lpold - lsold
    if ldiff > np.log(np.random.rand()):
        MCchain[i,:Ndim] = xprop
        MCchain[i,Ndim] = sp
        llchain[i] = llprop
        lpold = lpprop
        lsold = lspp
    else:
        MCchain[i,:Ndim] = MCchain[i-1,:Ndim]
        MCchain[i,Ndim] = MCchain[i-1,Ndim]
        llchain[i] = llchain[i-1]
    if i%500 == 0 : print(i)


#%%############################################################################
# POSTPROCESSING OF RAW OUTPUT OF MCMC
###############################################################################

MCf = MCchain[Nburn::Nthin]
llf = llchain[Nburn::Nthin]
idsort = np.argsort(llf)
MCchainS = MCf[idsort,:]
llchainS = llf[idsort]
Ntot = MCchainS.shape[0]
MAP = MCchainS[-1,:]
Nppost = 100
postY = np.array([sst.norm(loc=model(xmes, MCchainS[i,:Ndim]),
                            scale=MCchainS[i,Ndim]).rvs(xmes.shape[0])
                              for i in range(Ntot-Nppost, Ntot)])
postxplotm = np.array([model(xplot, MCchainS[i,:Ndim])
      for i in range(Ntot-Nppost,Ntot)])
postxplot = np.array([sst.norm(loc=model(xplot, MCchainS[i,:Ndim]),
                            scale=MCchainS[i,Ndim]).rvs(xplot.shape[0])
      for i in range(Ntot-Nppost,Ntot)])


#%%############################################################################
# VISUALISATION OF RESULT IN OBSERVATION SPACE
###############################################################################

fig, ax = plt.subplots()
# ax.plot(xplot[:,0], model(xplot, b0), '-b', label='true function')
ax.plot(xmes[:,0], postY[-1:,:].T, '.k', label='posterior calibrated')
ax.plot(xmes[:,0], postY[-100:,:].T, '.k')
ax.plot(xmes[:,0], ymes, '.r', label='available observation')
ax.fill_between(xplot[:,0],
                y1=postxplot.mean(axis=0) + 2*postxplot.std(axis=0),
                y2=postxplot.mean(axis=0) - 2*postxplot.std(axis=0),
                color='m', alpha=0.3,
                label="posterior")
ax.fill_between(xplot[:,0],
                y1=postxplotm.mean(axis=0) + 2*postxplotm.std(axis=0),
                y2=postxplotm.mean(axis=0) - 2*postxplotm.std(axis=0),
                  color='g', alpha=0.6,
                label="posterior model no noise")
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.legend()



#%%############################################################################
# VISUALISATION OF RESULT IN PARAMETER SPACE
###############################################################################

# lplot = -Ntot + 100
lplot = 0
startchain = False

fig, ax = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        if j>i:
            if startchain:
                ax[i,j].plot(MCchain[:Nburn,j], MCchain[:Nburn,i],'-k',
                    linewidth=0.5)
                # ax[i,j].plot(bstart[j], bstart[i], 'sb')
            ax[i,j].scatter(MCchainS[lplot:,j], MCchainS[lplot:,i],
                             c=llchainS[lplot:],
                            marker='.', cmap='jet')
            ax[i,j].plot(b0[j], b0[i], 's', color='k', ms=10, alpha=0.4)
            ax[i,j].plot(MAP[j], MAP[i], 'dk')
        elif j == i:
            ax[i,j].hist(MCchainS[:,j], edgecolor='k')
            ax[i,j].set_xlabel('b' + str(i))
        else:
            ax[i,j].set_visible(False)


#%%############################################################################
# VISUALISATION OF DIAGNOSTIC FOR MCMC CHAINS
###############################################################################

l = 3
Nc = int(Ntot/4)

burnzone = patches.Rectangle((0, pl), Nburn, ph-pl,
                               alpha=0.2, color='r', linestyle='')
okzone = patches.Rectangle((Nburn, pl), NMCMC-Nburn, ph-pl,
                               alpha=0.2, color='g', linestyle='')

# kdes = [sst.gaussian_kde(MCf[i*Nc:(i+1)*Nc,l]) for i in range(0,4)]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(MCchain[:,l],'-k')
# ax.plot(bstart[l], 'sk')
ax.plot(np.arange(Nburn, NMCMC, Nthin), MCf[:,l], '.b')
ax.add_patch(burnzone)
ax.add_patch(okzone)
ax.axhline(y=MAP[l], color='m', linestyle='--', label='MAP')
ax.axhline(y=b0[l], color='r', linestyle='-', label='true')
ax.set_xlabel('Step in chain')
ax.set_ylabel('parameter value')
ax.legend()

# fig, ax = plt.subplots()
# ax.hist(MCf[:,l], edgecolor='b', color='b', bins=20, alpha=0.2)
# axx = ax.twinx()
# for i in range(4):
#     axx.plot(np.linspace(pl,ph,200), kdes[i](np.linspace(pl,ph,200)), '-k')
# ax.axvline(x=MAP[l], color='m', linestyle='--', label='MAP')
# ax.axvline(x=b0[l], color='r', linestyle='-', label='true')
# ax.set_xlim(pl, ph)
# ax.legend()

# plt.show()
# %%
