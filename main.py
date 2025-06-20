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
                 biasp1+nsp1*np.c_[sst.norm(loc=0, scale=1).rvs(10)]])
ymes = modeltrue(xmes, b0)
ymes += sst.norm().rvs(xmes.shape[0])*nslvl


#%%############################################################################
# DEFINITION OF BAYESIAN INFERENCE OBJECTS
###############################################################################

def loglike(obs, var, par, smod, model):
    return np.sum(
        sst.norm(loc=model(var, par), scale=smod).logpdf(obs))


# sigma = np.diag((smod.mean()**2)*np.ones(ymes.shape[0]))
# # bstart = np.array([1.5,-0.7,1.5])
# bstart = np.array([1,1,1])

def loglikenp(obs, var, par, sigma, model):
    obs = obs.reshape(-1,1)
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive semi-definite "+
                         "(cannot perform Cholesky decomposition).")
    log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
    diff = model(var,par).reshape(-1,1) - obs
    y = np.linalg.solve(L, diff)
    mahalanobis_term = np.sum(y**2)
    log_likelihood = -0.5 * sigma.shape[0] * np.log(2*np.pi) + \
                    -0.5 * log_det_sigma + \
                    - 0.5 * mahalanobis_term
    return log_likelihood

# LOG1 = sst.multivariate_normal(
#     mean=modeltrue(xmes,bstart),cov=sigma).logpdf(ymes)
# LOG2 = loglikenp(ymes, xmes, bstart, sigma, modeltrue)
# print(LOG1, LOG2)

def logprior(par, dists):
    return np.sum([dists[i].logpdf(par) for i in range(len(par))])

def logpriornp(par, bounds):
    bcheck = [(p > b[0]) & (p < b[1]) for (p,b) in zip(par, bounds)]
    return 0 if all(bcheck) else -np.inf

# test = logpriornp(bstart, np.repeat(np.c_[pl, ph],3, axis=0))
# print(test)

def logsp(s,dist): return np.max([dist.logpdf(s),-100])
def logspnp(x, mu_scipy, loc, scale_scipy=1):
    x_shifted = x - loc
    nu = mu_scipy * scale_scipy
    lambda_val = scale_scipy
    log_pdf = (
            0.5 * np.log(lambda_val)
            - 0.5 * np.log(2 * np.pi)
            - 1.5 * np.log(x_shifted)
            - (lambda_val * np.power(x_shifted - nu, 2)) /
              (2 * np.power(nu, 2) * x_shifted)
        )
    return -np.inf if np.isnan(log_pdf) else log_pdf

# LOG1 = sst.invgauss(mu=0.4,loc=0.2).logpdf(0.6)
# print(LOG1)
# LOG2 = logspnp(-0.6, 0.4, 0.2)
# print(LOG2)

# def rnv(m,s): return np.random.randn(len(m))*np.array(s)+np.array(m)
def rnv1(m,s): return np.random.randn()*np.array(s)+m
def rnlv(m,s): return np.exp(np.random.randn()*s+np.log(m))

def rnvmultiD(m,Lmat):
    # Lmat = np.linalg.cholesky(Cov)
    return m + np.ravel(Lmat @ np.random.randn(m.shape[0],1))

def covToCorr(Sig):
    Dinv = np.diag(1/np.sqrt(np.diag(covProp)))
    return Dinv @ Sig @ Dinv

#%%############################################################################
# IMPLEMENTATION OF INFERENCE ALGORITHM
###############################################################################

## SPECIFICATION OF PRACTICAL APPLICATION
Ndim = 3
pl = -3
ph = 3
punif = [sst.uniform(pl,ph-pl) for _ in range(Ndim)]
# smod = sst.invgauss(0.4,0.2)
sinvg = [0.4, 0.3]

bstart = np.array([sst.uniform(pl,ph-pl).rvs() for _ in range(Ndim)])
# bstart = np.array([1.5,-0.7,1.5])

## PARAMETRIZATION OF MCMC
NMCMC = 25000
Nburn = 5000
Nthin = 20
Ntune = Nburn

covProp = np.eye(3)*1e-1
LLTprop = np.linalg.cholesky(covProp)
sm = np.array([0.1, 0.1, 0.1])

smexp = 0.1

## INITIALISATION OF MCMC
MCchain = np.zeros((NMCMC, Ndim+1))
llchain = np.zeros(NMCMC)
MCchain[0,:Ndim] = bstart
# MCchain[0,Ndim] = smod.mean()
MCchain[0,Ndim] = sinvg[0] + sinvg[1]
xprop = MCchain[0,:Ndim]
# MCchain[0,Ndim] = 0.2
# llchain[0] = loglike(ymes, xmes, MCchain[0,:Ndim], MCchain[0,Ndim],
#                       model=modelfit)
llchain[0] = loglikenp(ymes, xmes, MCchain[0,:Ndim],
                      np.diag((MCchain[0,Ndim]**2)*np.ones(ymes.shape[0])),
                      model=modelfit)
llold = llchain[0]
# lpold = logprior(MCchain[0,:Ndim], dists=punif)
lpold = logpriornp(MCchain[0,:Ndim], np.repeat(np.c_[pl, ph],3, axis=0))
# lsold = logsp(MCchain[0,Ndim], dist=smod)
lsold = logspnp(MCchain[0,Ndim], sinvg[0], sinvg[1])
# lsold = 0

nacc = 0
naccmultiD = np.zeros((Ndim+1)) 

talgo = "MHwG"
# talgo = "MHmultiD"

if talgo == "MHwG":
    ## RUN OF Adaptative MC Within Gibbs
    for i in range(1,NMCMC):
        xprop = np.copy(MCchain[i-1,:Ndim])
        sp = rnlv(MCchain[i-1,Ndim],smexp)
        # llprop = loglike(ymes, xmes, xprop, sp, model=modelfit)
        llprop = loglikenp(ymes, xmes, xprop,
                      np.diag((sp**2)*np.ones(ymes.shape[0])),
                      model=modelfit)
        # lspp = logsp(sp, dist=smod)
        lspp = logspnp(sp, sinvg[0], sinvg[1])
        ldiff = llprop + lspp - llold - lsold
        if ldiff > np.log(np.random.rand()):
            if i>Nburn: naccmultiD[Ndim] += 1
            MCchain[i,Ndim] = sp
            llchain[i] = llprop
            llold = llprop
            lsold = lspp
        else:
            MCchain[i,Ndim] = MCchain[i-1,Ndim]
            llchain[i] = llchain[i-1]
        idj = np.random.permutation(Ndim)
        for j in idj:
            xprop[j] = rnv1(MCchain[i-1,j],sm[j])
            # llprop = loglike(ymes, xmes, xprop, MCchain[i,Ndim], model=modelfit)
            llprop = loglikenp(ymes, xmes, xprop,
                np.diag((sp**2)*np.ones(ymes.shape[0])),
                model=modelfit)
            # lpprop = logprior(xprop, punif)
            lpprop = logpriornp(xprop, np.repeat(np.c_[pl, ph],3, axis=0))
            ldiff = llprop + lpprop - llold - lpold
            if ldiff > np.log(np.random.rand()):
                if i>Nburn: naccmultiD[j] += 1
                MCchain[i,j] = xprop[j]
                llchain[i] = llprop
                llold = llprop
                lpold = lpprop
            else:
                xprop[j] = MCchain[i-1,j]
                MCchain[i,j] = xprop[j]
                llchain[i] = llchain[i-1]
        if (i%1000 == 0): print(i)
        if (i%500 == 0) & (i<=Ntune):
            for j in idj:
                sm[j] = np.std(MCchain[i-500:i,j])*1
                # sm[j] = np.sqrt(np.var(MCchain[i-500:i,j])*2.38**2/(Ndim-1))
        
    print("acceptation rate :", ["{:.2f}".format(naccmultiD[k]/(NMCMC-Nburn))
                                for k in range(Ndim+1)])
    print(sm)


if talgo == "MHmultiD":
    ## RUN OF Adaptative Metrolis MC
    for i in range(1,NMCMC):
        xprop = rnvmultiD(MCchain[i-1,:Ndim],LLTprop)
        sp = rnlv(MCchain[i-1,Ndim],smexp)
        # sp = 0.2
        # llprop = loglike(ymes, xmes, xprop, sp, model=modelfit)
        llprop = loglikenp(ymes, xmes, xprop,
                np.diag((sp**2)*np.ones(ymes.shape[0])),
                model=modelfit)
        # lpprop = logprior(xprop, punif)
        lpprop = logpriornp(xprop, np.repeat(np.c_[pl, ph],3, axis=0))
        # lspp = logsp(sp, dist=smod)
        lspp = logspnp(sp, sinvg[0], sinvg[1])
        # lspp = 0
        ldiff = llprop + lpprop + lspp - llold - lpold - lsold
        if ldiff > np.log(np.random.rand()):
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
        if (i%1000 == 0) : print(i)
        if (i%500 == 0) & (i<=Ntune) :
            covProp = np.cov(MCchain[i-500:i,:Ndim].T) 
            LLTprop = np.linalg.cholesky(covProp * 2.38**2/(Ndim-1) +
                                        np.eye(Ndim)*1e-8)

    print("acceptation rate :", "{:.2f}".format(nacc/(NMCMC-Nburn)))
    print(np.sqrt(np.diag(covProp)))
    print(covProp)

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


# ax[0,1].set_xlim(-2.5,1)
# ax[0,2].set_xlim(1.5,3)
# ax[1,2].set_xlim(1.5,3)
# ax[0,3].set_xlim(0.2,1)
# ax[1,3].set_xlim(0.2,1)
# ax[2,3].set_xlim(0.2,1)
# ax[0,1].set_ylim(-1,3)
# ax[0,2].set_ylim(-1,3)
# ax[0,3].set_ylim(-1,3)
# ax[1,2].set_ylim(-2.5,1)
# ax[1,3].set_ylim(-2.5,1)
# ax[2,3].set_ylim(1.5,3)
# ax[0,0].set_xlim(-1,3)
# ax[1,1].set_xlim(-2.5,1)
# ax[2,2].set_xlim(1.5,3)
# ax[3,3].set_xlim(0.2,1)

fig.savefig("pythonpost.png", dpi=200)

#%%############################################################################
# VISUALISATION OF DIAGNOSTIC FOR MCMC CHAINS
###############################################################################

diag = True

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
# %%
