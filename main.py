#%%############################################################################
# REQUIREMENT PACKAGE IMPORT
###############################################################################
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import scipy.stats as sst

# matplotlib.use('QtAgg')


#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

def modeltrue(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 2*x[:,1]

def modelfit(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2

b0 = [2, -1, 2, 0]
nslvl = 0.5

xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
xplot[:,1] = 1

xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                 np.c_[sst.norm().rvs(10)]])
ymes = modeltrue(xmes, b0)
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
def logsp(s): return np.max([smod.logpdf(s),-100])


def rnv(m,s): return sst.norm(loc=m, scale=s).rvs()
def rnlv(m,s): return np.exp(sst.norm(loc=np.log(m), scale=s).rvs())


#%%############################################################################
# IMPLEMENTATION OF INFERENCE ALGORITHM
###############################################################################

## SPECIFICATION OF PRACTICAL APPLICATION
Ndim = 3
pl = -5
ph = 5

smod = sst.invgauss(0.4,0.2)

sexp = [0.2, 0.2, 0.05]
smexp = 0.05

bstart = np.array([sst.uniform(pl,ph-pl).rvs() for _ in range(Ndim)])

## PARAMETRIZATION OF MCMC
NMCMC = 22000
Nburn = 2000
Nthin = 10

## INITIALISATION OF MCMC
MCchain = np.zeros((NMCMC, Ndim+1))
llchain = np.zeros(NMCMC)
MCchain[0,:Ndim] = bstart
MCchain[0,Ndim] = smod.mean()
llchain[0] = loglike(ymes, xmes, MCchain[0,:Ndim], MCchain[0,Ndim],
                      model=modelfit)
llold = llchain[0]
lpold = logprior(MCchain[0,:Ndim], pl, ph)
lsold = logsp(MCchain[0,Ndim])
nacc = 0

## RUN OF MCMC
for i in range(1,NMCMC):
    xprop = rnv(MCchain[i-1,:Ndim],sexp)
    sp = rnlv(MCchain[i-1,Ndim],smexp)
    llprop = loglike(ymes, xmes, xprop, sp, model=modelfit)
    lpprop = logprior(xprop, pl, ph)
    lspp = logsp(sp)
    ldiff = llprop + lpprop + lspp - llold - lpold - lsold
    if ldiff > np.log(np.random.rand()):
        MCchain[i,:Ndim] = xprop
        MCchain[i,Ndim] = sp
        llchain[i] = llprop
        llold = llprop
        lpold = lpprop
        lsold = lspp
        if i>Nburn: nacc += 1
    else:
        MCchain[i,:Ndim] = MCchain[i-1,:Ndim]
        MCchain[i,Ndim] = MCchain[i-1,Ndim]
        llchain[i] = llchain[i-1]
    if i%500 == 0 : print(i)
    # print("LLN", "{:.1f}".format(llprop),
    #       "LLO", "{:.1f}".format(llold),
    #     "LPN", "{:.1f}".format(lpprop),
    #     "LPO", "{:.1f}".format(lpold),
    #     "LSN", "{:.1f}".format(lspp),        
    #     "LSO", "{:.1f}".format(lsold), 
    #     "dif", "{:.2f}".format(ldiff), sep='|')

print("acceptation rate :", "{:.2f}".format(nacc/(NMCMC-Nburn)))


#%%############################################################################
# POSTPROCESSING OF RAW OUTPUT OF MCMC
###############################################################################

MCf = MCchain[Nburn::Nthin,:]
llf = llchain[Nburn::Nthin]
idsort = np.argsort(llf)
MCchainS = MCf[idsort,:]
llchainS = llf[idsort]
Ntot = MCchainS.shape[0]
MAP = MCchainS[-1,:]
Nppost = Ntot
postY = np.array([sst.norm(loc=modelfit(xmes, MCchainS[i,:Ndim]),
                            scale=MCchainS[i,Ndim]).rvs(xmes.shape[0])
                              for i in range(Ntot-Nppost, Ntot)])
postxplotm = np.array([modelfit(xplot, MCchainS[i,:Ndim])
      for i in range(Ntot-Nppost,Ntot)])
postxplot = np.array([sst.norm(loc=modelfit(xplot, MCchainS[i,:Ndim]),
                            scale=MCchainS[i,Ndim]).rvs(xplot.shape[0])
      for i in range(Ntot-Nppost,Ntot)])


#%%############################################################################
# VISUALISATION OF RESULT IN OBSERVATION SPACE
###############################################################################

l = 0

fig, ax = plt.subplots()
ax.plot(xmes[:,l], postY[-1:,:].T, '.k', label='posterior calibrated')
ax.plot(xmes[:,l], postY[-Nppost:,:].T, '.k')
ax.plot(xmes[:,l], ymes, '.r', label='available observation')
ax.plot(xplot[:,l], modeltrue(xplot, b0), '-b', label='true function')
ax.fill_between(xplot[:,l],
                y1=postxplot.mean(axis=0) + 2*postxplot.std(axis=0),
                y2=postxplot.mean(axis=0) - 2*postxplot.std(axis=0),
                color='m', alpha=0.3,
                label="posterior")
ax.fill_between(xplot[:,l],
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

kdes = [sst.gaussian_kde(MCf[i*Nc:(i+1)*Nc,l]) for i in range(0,4)]

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

fig, ax = plt.subplots()
ax.hist(MCf[:,l], edgecolor='b', color='b', bins=20, alpha=0.2)
axx = ax.twinx()
for i in range(4):
    axx.plot(np.linspace(pl,ph,200), kdes[i](np.linspace(pl,ph,200)), '-k')
ax.axvline(x=MAP[l], color='m', linestyle='--', label='MAP')
ax.axvline(x=b0[l], color='r', linestyle='-', label='true')
ax.set_xlim(pl, ph)
ax.legend()

# plt.show()

#%%############################################################################
# VISUALISATION WITH TWO DIMENSION CASE
###############################################################################

nplot = 20

xg, yg = np.meshgrid(np.linspace(0,6,nplot), np.linspace(-3,3,nplot))
zg = np.array([modeltrue(np.atleast_2d([el0, el1]), b0)
                for el0, el1 in zip(xg.ravel(),yg.ravel())]).reshape(
                    [xg.shape[0], yg.shape[0]])

postgrid = np.array([sst.norm(loc=modelfit(
                            np.vstack([xg.ravel(), yg.ravel()]).T, MCf[k,:Ndim]),
                                    scale=MCf[k,Ndim]).rvs()
                for k in range(Ntot)])

post2Dm = postgrid.mean(axis=0).reshape([xg.shape[0], yg.shape[0]])
post2Ds = postgrid.std(axis=0).reshape([xg.shape[0], yg.shape[0]])
    

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot_surface(xg, yg, zg, cmap='jet_r', alpha=0.5)
for k in range(xmes.shape[0]):
    ax.scatter(xmes[k,0], xmes[k,1], postY[-50:,k], c='k', marker='.')
ax.scatter(xg, yg, post2Dm + 2*post2Ds, marker='.', color='k', alpha=0.2)
ax.scatter(xg, yg, post2Dm - 2*post2Ds, marker='+', color='k', alpha=0.2)
ax.scatter(xmes[:,0], xmes[:,1], ymes, c=ymes, cmap='jet_r')


# %%
