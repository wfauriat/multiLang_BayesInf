#%%############################################################################
# REQUIREMENT PACKAGE IMPORT
###############################################################################
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits import mplot3d
import scipy.stats as sst

import time as time
# matplotlib.use('QtAgg')


#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

def modeltrue(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 0.01*x[:,1]

def modelfit(x,b):
    return b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2

b0 = [2, -1, 2, 0]
nslvl = 0.2

xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
xplot[:,1] = 1

xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                 np.c_[sst.norm(loc=0, scale=1).rvs(10)]])
ymes = modeltrue(xmes, b0)
ymes += sst.norm().rvs(xmes.shape[0])*nslvl


#%%############################################################################
# DEFINITION OF BAYESIAN INFERENCE OBJECTS
###############################################################################

def loglike(obs, var, par, smod, model):
    return np.sum(
        sst.norm(loc=model(var, par), scale=smod).logpdf(obs))

def logprior(par, dists):
    return np.sum([dists[i].logpdf(par) for i in range(len(par))])
def logsp(s,dist): return np.max([dist.logpdf(s),-100])

def rnv(m,s): return np.random.randn(len(m))*np.array(s)+np.array(m)
def rnlv(m,s): return np.exp(np.random.randn()*s+np.log(m))


#%%############################################################################
# IMPLEMENTATION OF INFERENCE ALGORITHM
###############################################################################

## SPECIFICATION OF PRACTICAL APPLICATION
Ndim = 3
pl = -5
ph = 5
punif = [sst.uniform(pl,ph-pl) for _ in range(Ndim)]
smod = sst.invgauss(0.4,0.2)

bstart = np.array([sst.uniform(pl,ph-pl).rvs() for _ in range(Ndim)])

## PARAMETRIZATION OF MCMC
NMCMC = 22000
Nburn = 2000
Nthin = 20
Ntune = 1000

sexp = [0.2, 0.2, 0.05]
# sexp = [1, 1, 0.2]
smexp = 0.05

## INITIALISATION OF MCMC
MCchain = np.zeros((NMCMC, Ndim+1))
llchain = np.zeros(NMCMC)
MCchain[0,:Ndim] = bstart
MCchain[0,Ndim] = smod.mean()
# MCchain[0,Ndim] = 0.2
llchain[0] = loglike(ymes, xmes, MCchain[0,:Ndim], MCchain[0,Ndim],
                      model=modelfit)
llold = llchain[0]
lpold = logprior(MCchain[0,:Ndim], dists=punif)
lsold = logsp(MCchain[0,Ndim], dist=smod)
# lsold = 0
nacc = 0
tvacc = []


## RUN OF MCMC
Nphase = [Ntune, NMCMC]
for Ncur in Nphase:
    if Ncur == NMCMC: ## re-initialization for active MCMC after tuning
        nacc = 0
        MCchain[0,:Ndim] = xprop
        MCchain[0,Ndim] = sp
        llold = llchain[0]
        lpold = logprior(MCchain[0,:Ndim], punif)
        lsold = logsp(MCchain[0,Ndim], dist=smod)
        # lsold = 0
    if Ncur == Ntune: tacc=0 
    for i in range(1,Ncur): ## one chain for tuning another after tuning
        xprop = rnv(MCchain[i-1,:Ndim],sexp)
        sp = rnlv(MCchain[i-1,Ndim],smexp)
        # sp = 0.2
        llprop = loglike(ymes, xmes, xprop, sp, model=modelfit)
        lpprop = logprior(xprop, punif)
        lspp = logsp(sp, dist=smod)
        # lspp = 0
        ldiff = llprop + lpprop + lspp - llold - lpold - lsold
        if ldiff > np.log(np.random.rand()):
            tacc +=1
            if i>Nburn: nacc += 1
            MCchain[i,:Ndim] = xprop
            MCchain[i,Ndim] = sp
            llchain[i] = llprop
            llold = llprop
            lpold = lpprop
            lsold = lspp
        else:
            MCchain[i,:Ndim] = MCchain[i-1,:Ndim]
            MCchain[i,Ndim] = MCchain[i-1,Ndim]
            llchain[i] = llchain[i-1]
        if (i%1000 == 0) & (Ncur == Ntune): print(i, tacc/100, sexp)
        if (i%1000 == 0) & (Ncur != Ntune): print(i)
        if (i%100 == 0) & (Ncur == Ntune): ## fully proportional tuning of step size
            tvacc.append(tacc/100)
            if (tacc/100)<0.3:
                # sexp[np.random.randint(3)] *= 0.9
                sexp = [0.9*sexp[i] for i in range(Ndim)]
            else:
                # sexp[np.random.randint(3)] *= 1.1
                sexp = [1.1*sexp[i] for i in range(Ndim)]
            tacc = 0
            

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
postMAP = sst.norm(loc=modelfit(xplot, MAP[:Ndim]),
                    scale=MAP[Ndim]).rvs(xplot.shape[0])

#%%############################################################################
# VISUALISATION OF RESULT IN OBSERVATION SPACE
###############################################################################

l = 0

fig, ax = plt.subplots()
ax.plot(xmes[:,l], postY[-1:,:].T, '.k', label='posterior calibrated')
ax.plot(xmes[:,l], postY[-Nppost:,:].T, '.k')
ax.plot(xmes[:,l], ymes, '.r', label='available observation')
ax.plot(xplot[:,l], modeltrue(xplot, b0), '-b', label='true function (y-slice)')
ax.plot(xplot[:,l], modelfit(xplot, MAP), '--k', label='MAP mean')
# ax.plot(xplot[:,l], postMAP, '+k', label='MAP draw')
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

fig.savefig("pythonpostobs.png", dpi=200)

#%%############################################################################
# VISUALISATION OF RESULT IN PARAMETER SPACE
###############################################################################

startchain = False

# fig, ax = plt.subplots(4,4, figsize=(8,8))
fig, ax = plt.subplots(4,4)
for i in range(4):
    for j in range(4):
        if j>i:
            if startchain:
                ax[i,j].plot(MCchain[:Nburn,j], MCchain[:Nburn,i],'-k',
                    linewidth=0.5)
            ax[i,j].scatter(MCchainS[:,j], MCchainS[:,i],
                             c=llchainS[:],
                            marker='.', cmap='jet')
            # ax[i,j].plot(b0[j], b0[i], 's', color='k', ms=10, alpha=0.4)
            ax[i,j].plot(MAP[j], MAP[i], 'dk')
        elif j == i:
            ax[i,j].hist(MCchainS[:,j], edgecolor='k')
            ax[i,j].set_xlabel('b' + str(i))
        else:
            ax[i,j].set_visible(False)

fig.savefig("pythonpost.png", dpi=200)

#%%############################################################################
# VISUALISATION OF DIAGNOSTIC FOR MCMC CHAINS
###############################################################################

diag = False

if diag:
    l = 0
    Nc = int(Ntot/4)

    burnzone = patches.Rectangle((0, pl), Nburn, ph-pl,
                                alpha=0.2, color='r', linestyle='')
    okzone = patches.Rectangle((Nburn, pl), NMCMC-Nburn, ph-pl,
                                alpha=0.2, color='g', linestyle='')

    kdes = [sst.gaussian_kde(MCf[i*Nc:(i+1)*Nc,l]) for i in range(0,4)]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(MCchain[:,l],'-k')
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

if False:    

    fig = plt.figure(figsize=(12,12))
    ax = plt.axes(projection='3d')
    ax.plot_surface(xg, yg, zg, cmap='jet_r', alpha=0.5)
    for k in range(xmes.shape[0]):
        ax.scatter(xmes[k,0], xmes[k,1], postY[-50:,k], c='k', marker='.')
    ax.scatter(xg, yg, post2Dm + 2*post2Ds, marker='.', color='k', alpha=0.4)
    ax.scatter(xg, yg, post2Dm - 2*post2Ds, marker='+', color='k', alpha=0.4)
    ax.scatter(xmes[:,0], xmes[:,1], ymes, c=ymes, cmap='jet_r')
    ax.view_init(0,-90)
    # ax.view_init(0,180)
    ax.set_proj_type('ortho')
    # ax.set_proj_type('persp')

# %%

# plt.show()
print('MAP: ', str(MAP))