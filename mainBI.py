#%%

from itertools import chain

from pyBI.base import UnifVar, NormVar, InvGaussVar, HalfNormVar, ObsVar
from pyBI.base import HGP, GaussLike
from pyBI.inference import MHalgo, MHwGalgo, MHwGalgo2


import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from scipy.optimize import minimize 

np.set_printoptions(suppress=True, precision=5)

#%%####
# FOR DEVELOPMENT PURPOSES
#

# import importlib
# import pyBI.base
# new_mod = importlib.reload(pyBI.base)
# GaussLike = new_mod.GaussLike

# np.random.seed(123)
## REFAIRE LOG-LIKELIHOOD en distinguant MEAN(paramsA) et COV(paramsB)

#%%

casep = 0
# casep = 1
# casep = 2

# inftype = 'MH'
inftype = 'MHwG'

#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

if casep == 0:
    def modeltrue(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + \
                             1*x[:,1] + 0.02*x[:,0]**3)

    def modelfit(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)

    b0 = [2, -1, 2, 0]
    nslvl = 0.1
    nsp1 = 0.2
    biasp1 = -1

    xplot = np.repeat(np.c_[np.linspace(0,6,50)],2, axis=1)
    xplot[:,1] = 1

    xmes = np.hstack([np.c_[[0, 0.5, 1, 2, 2.5, 2.8, 4, 4.4, 5.2, 5.5]],
                    biasp1+nsp1*np.c_[np.random.randn(10)]])
    ymes = modeltrue(xmes, b0)
    ymes += np.random.randn(xmes.shape[0])*nslvl

    # DXtrain = xmes.max(axis=0) - xmes.min(axis=0)
    # lowD = DXtrain/20
    # highD = DXtrain*5
    # kernel = RBF([1.0]*2, 
    #             [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
    #     WhiteKernel(noise_level=0.001,
    #                 noise_level_bounds=(1e-6,np.var(ymes.ravel())*0.05))

    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    # gp.fit(xmes, ymes.ravel())

if casep == 1:
    def ishigami(x,b):
        return np.atleast_2d(np.sin(x[:,0]) + b[:,0]*np.sin(x[:,1])**2 +  \
               b[:,1]*np.sin(x[:,0])*x[:,2]**4)

    Xtrain = sst.qmc.scale(sst.qmc.LatinHypercube(d=5).random(100),
                            [-3, -3, -3, 4, 0.01], [3, 3, 3, 9, 0.3])
    ytrain = ishigami(Xtrain[:,:3],Xtrain[:,3:])

    XX = np.array([[0,0,0], [-1.3,0,-1], [0.2,1,-1],
                [2,1,-2], [3.4,1,-1], [2.5,-2,-1],
                [3,3,-2], [-1,-1,2], [2.7,-2,0],
                [1,1,-2]])
    b0 = np.array([7,0.1])

    DXtrain = Xtrain.max(axis=0) - Xtrain.min(axis=0)
    lowD = DXtrain/20
    highD = DXtrain*5
    kernel = RBF([1.0]*5, 
                [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
        WhiteKernel(noise_level=0.001,
                    noise_level_bounds=(1e-6,np.var(ytrain)*0.05))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gp.fit(Xtrain, ytrain.ravel())

    def modelgp(x, b):
        return np.atleast_2d(gp.predict(np.hstack([x,
                    np.repeat(np.r_[[b]],x.shape[0], axis=0)])).ravel())

    ytrue = ishigami(XX,
                np.repeat(np.atleast_2d(b0),XX.shape[0], axis=0))
    ypred = modelgp(XX, b0)
    Q2emp = 1-np.var(ytrue-ypred)/np.var(ypred)
    ymes = ytrue
           
    if True:
        fig, ax = plt.subplots()
        ax.plot(XX[:,0], ytrue[0,:], 'or')
        ax.plot(XX[:,0], ypred[0,:], '.b')
        axx = ax.twiny()
        axx.plot(ytrue[0,:], ypred[0,:], '+k')
        axx.plot(ytrue[0,:], ytrue[0,:], '+-k')
        ax.set_title('Q2emp=' + \
                    str(np.round(Q2emp,3)))
    print('Q2emp =', Q2emp)

    xmes = XX

if casep == 2:
    Nsamp = 50
    mu_true = np.array([1,2,4])
    cor_true = np.array([[1,     0.8,  -0.1],
                         [0.8,     1,  -0.5],
                         [-0.5, -0.1,    1]])
    s_true = np.array([0.2, 0.4, 0.5])
    cov_true = np.diag(s_true) @ cor_true @ np.diag(s_true)
    L_true = np.linalg.cholesky(cov_true)

    yrnd = mu_true + (L_true @ np.random.randn(mu_true.shape[0],Nsamp)).T

    def idfun(cond, par, N):
        return np.repeat(np.atleast_2d(par),N, axis=0)
    

#%%############################################################################
# DEFINITION OF BAYESIAN INFERENCE OBJECTS
###############################################################################

############## FULLY ANALYTICAL REGRESSION EXAMPLE
if casep == 0:
    Ndim = 3
    pl = -5
    ph = 5
    # sinvg = [0.2, -0.1, 2]
    # sexp = [0.4, 0.4, 0.05]
    # sdexp = 0.1
    sexp = [0.1, 0.1, 0.1]
    sdexp = 0.1
    covProp = np.eye(3)*1e-1
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([pl,ph]) for _ in range(3)]
    # rndUs = [NormVar([0, 1]) for _ in range(3)]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    obsvar = ObsVar(obs=ymes, prev_model=modelfit, cond_var=xmes)

    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    
############## ISHIGAMI AND GP MODEL SURROGATE FOR CALIBRATION
if casep == 1:
    Ndim = 2
    sinvg = [0.4, 0, 2]

    sexp = [0.2, 5e-2]
    sdexp = 0.2
    covProp = np.array([[0.2,0],[0,0.05]])
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([0,10]), UnifVar([0,1])]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    obsvar = ObsVar(obs=ymes, prev_model=modelgp, cond_var=XX)

    bstart = np.array([rndUs[i].draw() for i in range(2)] + \
                       [float(rnds.draw())])
    
############## STANDARD MULTIVARIATE PARAMETER ESTIMATION
if casep == 2:
    Ndim = 3
    sinvg = [0.2, 0.2, 1]
    sexp = [0.1, 0.1, 0.1]
    sdexp = 0.1
    rndUs = [UnifVar([0,6]), UnifVar([0,6]), UnifVar([0,6])]
    # rnds = InvGaussVar(param=sinvg)
    rnds = HalfNormVar(param=0.5)
    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    obsvar = ObsVar(obs=yrnd,
                prev_model=lambda v, x: idfun(v, x, yrnd.shape[0]),
                cond_var=[])

#%%############################################################################
# INSTANTIATION AND RUN OF INFERENCE ALGORITHM
###############################################################################

NMCMC = 20000
Nburn = 10000
verbose = True

# inftype = 'MHwG'
# bstart = np.array([rndUs[i].draw() for i in range(3)] + \
#                     [float(rnds.draw())])

if inftype == 'MH':
    MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                     verbose=verbose)
if inftype == 'MHwG':
    MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                       verbose=verbose)
# MCalgo.initialize(obsvar, rndUs, rnds, svar=sexp, sdisc=sdexp)
MCalgo.initialize(obsvar, rndUs, rnds, svar=1)
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCout, llout = MCalgo.runInference()


#%%############################################################################
# VISUALISATION OF INFERENCE RESULTS
###############################################################################

MCalgo.post_visupar()
MCalgo.hist_alldim()
MCalgo.diag_chain(0, show_prior=False)
MCalgo.diag_chain(3, show_prior=False)

print(MCalgo)


#%%############################################################################
# VISUALISATION FOR PURE STATISTICAL INFERENCE
###############################################################################

if casep == 2:
    print(sst.multivariate_normal(
        MCalgo.MAP[:3], rnds.diagSmat(MCalgo.MAP[3], 3)).logpdf(yrnd).sum())
    print(obsvar.loglike(MCalgo.MAP[:3],rnds.diagSmat(MCalgo.MAP[3], 3)))
    ypost = sst.multivariate_normal(mean=MCalgo.MAP[:3],
                                    cov=MCalgo.discrObj.diagSmat(
                                        MCalgo.MAP[3], yrnd.shape[1])).rvs(100)

    fig, ax = plt.subplots()
    ax.scatter(ypost[:,0], ypost[:,1], marker='x', color='k')
    ax.scatter(yrnd[:,0], yrnd[:,1], marker='o', color='b')


#%%############################################################################
# TEST OF FITTING GP 
###############################################################################

if (casep == 0) | (casep == 1):

    if casep == 1:
        del ishigami
        def ishigami(x,b):
            return np.atleast_2d(np.sin(x[:,0]) + b[0]*np.sin(x[:,1])**2 +  \
                b[1]*np.sin(x[:,0])*x[:,2]**4)

        modeltrue = ishigami


    gpobj = HGP(xmes, ymes.ravel())
    # gpobj.bounds[-1] = [1e0, 1e1]
    gpobj.mtune(N=10)

    if casep == 0:
        xtest = np.linspace(0.5,5.4,100)
        xtest = np.vstack([xtest, np.random.randn(xtest.shape[0])*nsp1+biasp1+0]).T
        ypred = gpobj.m_predict(xtest)
        spred = np.diag(gpobj.cov_predict(xtest, noise=True))

    ypred0 = gpobj.m_predict(xmes)
    spred0 = np.diag(gpobj.cov_predict(xmes, noise=True))

    fig, ax = plt.subplots(1,2, figsize=(9,4))
    ax[0].plot(xmes[:,0], ymes.ravel(), 'or')
    ax[0].plot(xmes[:,0], modeltrue(xmes, b0).ravel(), 'sk', alpha=0.2, ms=3)
    ax[0].plot(xmes[:,0], gpobj.m_predict(xmes), 'xb')
    ax[0].fill_between(x=xmes[:,0], y1=ypred0 + 2*spred0**0.5,
                    y2=ypred0 -2*spred0**0.5, alpha=0.2, color='b')

    if casep == 0:
        ax[0].plot(xtest[:,0], ypred, 'x-m', ms=2, lw=0.5)
        ax[0].plot(xtest[:,0], modeltrue(xtest, b0).ravel(), 'sm', alpha=0.2, ms=3)
        ax[0].fill_between(x=xtest[:,0], y1=ypred + 2*spred**0.5,
                    y2=ypred -2*spred**0.5, alpha=0.2, color='m')

    if casep == 0:
        ax[1].plot(modeltrue(xtest, b0).ravel(), ypred, 'ob')
        ax[1].plot(ypred, ypred, '-k')

    print(gpobj)


#%%############################################################################
# VERIFICATION OF GAUSSIAN LIKELIHOOD
###############################################################################

if casep == 0:

    btest = b0
    # btest = [-2, -1, 2]
    # btest = MCalgo.MAP

    LLobj = GaussLike(ymes, gmod=modelfit, x=xmes,
                    gpar=btest[:3], kpar=[1,1,1], spar=1)
    # LLobj.setpar(gpar=MCalgo.MAP, kpar=[5,1,1], spar=1)
    LLobj.setpar(gpar=MCalgo.MAP, kpar=gpobj.kpar, spar=gpobj.spar)
    # LLobj.setpar(gpar=[0, 0, 0], kpar=gpobj.kpar, spar=gpobj.spar)
    # LLobj.setpar(gpar=[0, 0, 0], kpar=[1, 1, 1], spar=1)
    print(LLobj.loglike())

    fig, ax = plt.subplots()
    y0 = LLobj.g_predict(xmes).ravel()
    y1 = LLobj.mgp.m_predict(xmes).ravel()
    y2 = LLobj.mean(xmes).ravel()
    ypostBI = MCalgo.post_obs()
    ax.plot(xmes[:,0], ypostBI.T, '.k', ms=3)
    ax.plot(xmes[:,0], ymes.ravel(), 'or', ms=8, label='observation')
    ax.plot(xmes[:,0], y0, 'g', label='trend')
    ax.plot(xmes[:,0], y1, 'm', label='gp correc, errmod')
    ax.plot(xmes[:,0], y2 + 2*np.diag(LLobj.cov(xmes)), '.b', label='post uncty')
    ax.plot(xmes[:,0], y2 - 2*np.diag(LLobj.cov(xmes)), '.b')
    ax.plot(xmes[:,0], y0+y1, 'b')
    ax.plot(xtest[:,0], np.ravel(LLobj.mean(xtest)), 'x-m', alpha=0.3)

    ax.legend()

#%%############################################################################
# INFERENCE WITH GAUSSIAN LIKELIHOOD MADE OF GP + LINEAR REGRESSION
###############################################################################

if casep == 0:

    Nburn2 = 10000

    pl = -5
    ph = 5
    sinvg = [0.2, -0.1, 2]
    rndUs = [UnifVar([pl,ph]) for _ in range(3)]
    # rndUs = [UnifVar([0.5,2.5]),UnifVar([-1.5,-0.5]), UnifVar([1.5,2.5]) ]
    rnds = InvGaussVar(param=[0.05, -0.01, 2])
    rndk = []
    rndk.append(UnifVar([1e-1, 200]))
    rndk.append(UnifVar([2e-1, 10]))
    rndk.append(UnifVar([1e-2, 3]))

    lvarobj = {'gpar': [],
            'kpar': [],
            'spar': []}

    lvarobj['gpar'].extend(rndUs)
    lvarobj['kpar'].extend(rndk)
    lvarobj['spar'].append(rnds)

    svar2 = [[0.1, 0.1, 0.1], [10, 1, 0.1], [0.05]]

    MCalgo2 = MHwGalgo2(NMCMC, Nthin=20, Nburn=Nburn2, is_adaptive=True,
                        verbose=verbose)
    MCalgo2.initialize(LLobj, lvarobj, svar2)
    MCout2, llout2 = MCalgo2.runInference()

    fig, ax = MCalgo2.post_visupar()
    fig.tight_layout(pad=0)
    MCalgo2.hist_alldim()


# %%

# from scipy.spatial.distance import cdist
# print(cdist(xmes[:,0][:,None], xmes[:,0][:,None], metric='sqeuclidean'))
# print(rXX(xmes[:,0][:,None], xmes[:,0][:,None]))
