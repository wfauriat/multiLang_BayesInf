#%%

from itertools import chain

from pyBI.base import UnifVar, InvGaussVar, ObsVar
from pyBI.base import HGP
from pyBI.inference import MHalgo, MHwGalgo, InfAlgo, InfAlgo2, MHwGalgo2


import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from scipy.optimize import minimize 

np.set_printoptions(suppress=True, precision=5)

# np.random.seed(123)
## REFAIRE LOG-LIKELIHOOD en distinguant MEAN(paramsA) et COV(paramsB)

#%%

case = 0
# case = 1
# case = 2

inftype = 'MH'
# inftype = 'MHwG'

#%%############################################################################
# DEFINITION OF APPLICATION / CALIBRATION CASE
###############################################################################

if case == 0:
    def modeltrue(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + 4*x[:,1])

    def modelfit(x,b):
        return np.atleast_2d(b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2)

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

    # DXtrain = xmes.max(axis=0) - xmes.min(axis=0)
    # lowD = DXtrain/20
    # highD = DXtrain*5
    # kernel = RBF([1.0]*2, 
    #             [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
    #     WhiteKernel(noise_level=0.001,
    #                 noise_level_bounds=(1e-6,np.var(ymes.ravel())*0.05))

    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    # gp.fit(xmes, ymes.ravel())

if case == 1:
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
    b3 = np.array([7,0.1])

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
                np.repeat(np.atleast_2d(b3),XX.shape[0], axis=0))
    ypred = modelgp(XX, b3)
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

if case == 2:
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
if case == 0:
    Ndim = 3
    pl = -3
    ph = 3
    sinvg = [0.2, -0.1, 2]

    sm = [0.4, 0.4, 0.05]
    smexp = 0.1
    covProp = np.eye(3)*1e-1
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([pl,ph]) for _ in range(3)]
    rnds = InvGaussVar(param=sinvg)
    obsvar = ObsVar(obs=ymes, prev_model=modelfit, cond_var=xmes)

    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    
############## ISHIGAMI AND GP MODEL SURROGATE FOR CALIBRATION
if case == 1:
    Ndim = 2
    sinvg = [0.4, 0, 2]

    sm = [0.2, 5e-2]
    smexp = 0.2
    covProp = np.array([[0.2,0],[0,0.05]])
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([0,10]), UnifVar([0,1])]
    rnds = InvGaussVar(param=sinvg)
    obsvar = ObsVar(obs=ymes, prev_model=modelgp, cond_var=XX)

    bstart = np.array([rndUs[i].draw() for i in range(2)] + \
                       [float(rnds.draw())])
    
############## STANDARD MULTIVARIATE PARAMETER ESTIMATION
if case == 2:
    Ndim = 3
    sinvg = [0.2, 0.2, 1]
    rndUs = [UnifVar([0,6]), UnifVar([0,6]), UnifVar([0,6])]
    rnds = InvGaussVar(param=sinvg)
    bstart = np.array([rndUs[i].draw() for i in range(3)] + \
                       [float(rnds.draw())])
    obsvar = ObsVar(obs=yrnd,
                prev_model=lambda v, x: idfun(v, x, yrnd.shape[0]),
                cond_var=[])

#%%############################################################################
# INSTANTIATION AND RUN OF INFERENCE ALGORITHM
###############################################################################

NMCMC = 20000
Nburn = 5000
verbose = True

if inftype == 'MH':
    MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                     verbose=verbose)
if inftype == 'MHwG':
    MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                       verbose=verbose)
# MCalgo.initialize(obsvar, rndUs, rnds, svar=sm, sdisc=smexp)
MCalgo.initialize(obsvar, rndUs, rnds)
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCout, llout = MCalgo.runInference()


#%%############################################################################
# VISUALISATION OF INFERENCE RESULTS
###############################################################################

MCalgo.post_visupar()
MCalgo.hist_alldim()

#%%
print(MCalgo)


#%%############################################################################
# VISUALISATION FOR PURE STATISTICAL INFERENCE
###############################################################################

if case == 2:
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

# xmes = XX
gpobj = HGP(xmes, ymes.ravel())
# gpobj.bounds[-1] = [1e0, 1e1]

gpobj.mtune(N=10)

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
ax[0].plot(xtest[:,0], ypred, 'x-m', ms=2, lw=0.5)
ax[0].plot(xtest[:,0], modeltrue(xtest, b0).ravel(), 'sm', alpha=0.2, ms=3)
ax[0].fill_between(x=xtest[:,0], y1=ypred + 2*spred**0.5,
                y2=ypred -2*spred**0.5, alpha=0.2, color='m')


ax[1].plot(modeltrue(xtest, b0).ravel(), ypred, 'ob')
ax[1].plot(ypred, ypred, '-k')

print(gpobj)

#%%

# fig, ax = MCalgo.post_visuobs(0)
# ax.plot(xtest[:,0], ypred, 'xm')
# ax.plot(xtest[:,0], modeltrue(xtest, b0).ravel(), 'sm', alpha=0.2)
# ax.fill_between(x=xtest[:,0], y1=ypred + 2*spred**0.5,
#                 y2=ypred-2*spred**0.5, alpha=0.2, color='m')
# ax.plot(xmes[:,0], gpobj.m_predict(xmes), 'xb')


#%%

# btest = b0
btest = [1, -0.5, 1]

class GaussLike():
    def __init__(self, obs, gmod=None, x=None):
        self.obs = obs
        self.x = x
        if gmod is None:
            self.gmod = lambda x, v: 0 
        else: self.gmod = gmod
        self.Ndata = obs.shape[0]
        self.dimdata = obs.shape[1]

    def setpar(self, gpar, kpar, spar):
        self.gpar = gpar # parameters of g(x,beta)
        self.kpar = kpar # parameters of s * exp (-d**2/t**2)
        self.spar = spar # diagonal parameter of sigma (or full Sigma)
    
    def g_predict(self, x, par=None):
        ptmp = self.gpar if par is None else par
        return self.gmod(x, ptmp) if ptmp is not None else \
              np.zeros(x.shape[0])
    
    def gp_init(self):
        self.mgp = HGP(self.x, np.ravel(self.obs - \
                       self.g_predict(self.x, par=self.gpar)))
        self.mgp.setpar(self.kpar, self.spar)
        # self.mgp.mtune()
    
    def mean(self, x):
        return self.g_predict(x) + self.mgp.m_predict(x)
    
    def cov(self, x):
        return self.mgp.cov_predict(x, noise=False) + \
                np.eye(x.shape[0])*self.spar

    def loglike(self):
        N = self.obs.shape[0]
        d = self.obs.shape[1]
        try:
            L = np.linalg.cholesky(self.cov(self.x))
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive semi-definite "+
                             "(cannot perform Cholesky decomposition).")
        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        diff = self.mean(self.x) - self.obs
        mahalanobis_term = np.zeros(N)
        for i in range(N):
            y = np.linalg.solve(L, diff[i, :])
            mahalanobis_term[i] = np.sum(y**2)
        log_likelihood = - (N * d / 2) * np.log(2 * np.pi) \
                    - (N / 2) * log_det_sigma \
                    - (1 / 2) * np.sum(mahalanobis_term)
        return log_likelihood

LLobj = GaussLike(ymes, gmod=modelfit, x=xmes)
LLobj.setpar(gpar=btest[:3], kpar=[5,1,1], spar=0.5)
LLobj.gp_init()
print(LLobj.loglike())

# fig, ax = plt.subplots()
# ax.plot(xmes[:,0], np.ravel(ymes), 'or')
# ax.plot(xmes[:,0], np.ravel(LLobj.mean(xmes)), 'xb')
# ax.plot(xmes[:,0], np.ravel(LLobj.mean(xmes)) + \
#         2* np.diag(LLobj.cov(xmes)), '.b')
# ax.plot(xmes[:,0], np.ravel(LLobj.mean(xmes)) - \
#         2* np.diag(LLobj.cov(xmes)), '.b')
# ax.plot(xtest[:,0], np.ravel(LLobj.mean(xtest)), 'x-m')


#%%
# how to appropriately store infered parameter so that it works for different
# application cases ?
# par_vect = [[fmod_par : "beta",
#              gpm_par : "theta",
#              extra_cov_par : "sigma" or "Sigma"]]
# cond_var or no cond_var : "x"

# print("\n".join([str(rndUs[i]) for i in range(3)]))
# print(str(rnds))


pl = -3
ph = 3
sinvg = [0.2, -0.1, 2]
# rndUs = [UnifVar([pl,ph]) for _ in range(3)]
rndUs = [UnifVar([0.5,2.5]),UnifVar([-1.5,-0.5]), UnifVar([1.5,2.5]) ]
rnds = InvGaussVar(param=[0.01, -0.01, 5])
rndk = []
rndk.append(UnifVar([1e-1, 100]))
rndk.append(UnifVar([2e-1, 20]))
rndk.append(UnifVar([1e-2, 2]))

lvarobj = {'gpar': [],
           'kpar': [],
           'spar': []}

lvarobj['gpar'].extend(rndUs)
lvarobj['kpar'].extend(rndk)
lvarobj['spar'].append(rnds)

svar2 = [[0.1, 0.1, 0.1], [10, 1, 0.1], [0.05]]

MCalgo2 = MHwGalgo2(NMCMC, Nthin=20, Nburn=Nburn, is_adaptive=True,
                       verbose=verbose)
MCalgo2.initialize(LLobj, lvarobj, svar2)
MCout, llout = MCalgo2.runInference()

# vdim = np.sum([len(lvarobj[el]) for el in lvarobj.keys()])

# print("\n".join([str(el) for el in lvarobj.values()]))

# NMCMC = 30

# MCchain = np.zeros((NMCMC, vdim)) # Initialisation to size on inference pb vdim

# def draw_all_vars():
#     rnd = np.zeros(vdim)        
#     i = 0
#     for el in lvarobj.keys():
#         for k in range(len(lvarobj[el])):
#             rnd[i] = lvarobj[el][k].draw()
#             i+=1
#     return rnd

# def place_var():
#     place_g = len(lvarobj['gpar'])
#     place_k = len(lvarobj['kpar'])
#     place_s = len(lvarobj['spar'])
#     return place_g, place_g + place_k, place_g + place_k + place_s

# def category_from_chain(x, i):
#     gpar = list(x[i][:place_var()[0]])
#     kpar = list(x[i][place_var()[0]:place_var()[1]])
#     spar = list(x[i][place_var()[1]:place_var()[2]])
#     return gpar, kpar, spar

# MCchain[0,:] = draw_all_vars()

# catElem = category_from_chain(MCchain, 0)
# # print(*catElement)
# LLobj.setpar(catElem[0], catElem[1], catElem[2])
# # LLobj.setpar(gpar=btest[:3], kpar=[5,1,1], spar=0.5)
# print("gpar : ", LLobj.gpar)
# print("kpar : ", LLobj.kpar)
# print("spar : ", LLobj.spar)

# log_state = LLobj.loglike()
# log_prior = [[lvarobj[el][k].logprior(catElem[j][k])
#                for k in range(len(lvarobj[el]))] 
#                  for (el,j) in zip(lvarobj.keys(),range(3))]
# log_prior_sum = np.sum(list(chain(*log_prior)))

# print("log state :", log_state)
# print("log_prior :", log_prior_sum)

# svar = [[0.1, 0.1, 0.1], [10, 1, 0.1], [0.05]]

# def proposal_all():
#     catProp = [[], [], []]
#     for (el,j) in zip(lvarobj.keys(), range(3)):
#         for k in range(len(lvarobj[el])):
#             catProp[j].append(lvarobj[el][k].proposal(
#                 catElem[j][k], svar[j][k]))
#     return catProp

# print()
# print(np.hstack([np.c_[list(chain(*catElem))],
#                   np.c_[list(chain(*proposal_all()))]]))






# for i in range(NMCMC):
#     MCchain[i,:] = np.array(list(chain(*[[float(lvarobj[el][k].draw()) 
#                               for k in range(len(lvarobj[el]))]
#                               for el in lvarobj.keys()])))
            



    # def initialize(self, obsObj: ObsVar, varObj: RandVar, discrObj: RandVar,
    #                 svar: float = 1., sdisc: Optional[float] = None,
    #                 Lblock: Optional[np.ndarray] = None):



# %%

# #%%
# ## CODE TO VERIFY GP IMPLEMENTATION
# # def fmod(x, beta):
# #     return beta[0]*x + beta[1]*x**2
# # fmod = modelfit
# b0 = [2, -1, 2, 0]
# theta = [1, 0.1]
# sigma = 1
# sn = 0.1
# # theta = [3.17435, 4.24163]
# # sigma = 39.76435
# # sn = 0.01


# xtest = np.linspace(0.5,5.4,20)
# xtest = np.vstack([xtest, np.random.randn(xtest.shape[0])*0.1-1]).T

# kpar = [sigma] + theta
# spar = sn

# def raXX(X, Y, scl=theta):
#     diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
#     scaled_diff_sq = (diff / scl)**2
#     squared_weighted_distance = np.sum(scaled_diff_sq, axis=2)
#     return squared_weighted_distance 

# def KaXX(X, theta=theta, sigma=[sigma, sn]):
#     return sigma[0]**2 * np.exp(-raXX(X,X,theta)/2) + \
#         np.eye(X.shape[0])*sigma[1]

# def KaxX(x, X, theta=theta, sigma=[sigma, sn]):
#     return sigma[0]**2 * np.exp(-raXX(x,X, theta)/2)

# XX = xmes
# k_XX = KaXX(xmes) + np.eye(xmes.shape[0])*1e-8
# L_XX = np.linalg.cholesky(k_XX)
# alpha = np.linalg.solve(L_XX.T, np.linalg.solve(L_XX, ymes.ravel()))

# def m_predict(x):
#     K_xX = KaxX(x, XX)
#     mu_pred = K_xX @ alpha
#     return mu_pred

# def cov_predict(x):
#     K_xX = KaxX(x, XX)
#     v = np.linalg.solve(L_XX, K_xX.T)
#     cov_pred = KaxX(x, x) - v.T @ v
#     return cov_pred

# ypred = m_predict(xtest)
# spred = np.diag(cov_predict(xtest))

# fig, ax = plt.subplots(1,2, figsize=(8,5))
# ax[0].plot(xmes[:,0], ymes.ravel(), 'or')
# ax[0].plot(xtest[:,0], modeltrue(xtest, b0).ravel(), 'sk', alpha=0.2)
# ax[0].plot(xmes[:,0], m_predict(xmes), '.-b')
# ax[0].plot(xtest[:,0], m_predict(xtest), 'xk')
# ax[0].plot(xtest[:,0], ypred + 2*spred, '.m', ms=2)
# ax[0].plot(xtest[:,0], ypred - 2*spred, '.m', ms=2)
# ax[1].plot(xmes[:,1], ymes.ravel(), 'or')
# ax[1].plot(xtest[:,1], modeltrue(xtest, b0).ravel(), 'sk', alpha=0.2)
# ax[1].plot(xmes[:,1], m_predict(xmes), '.-b')
# ax[1].plot(xtest[:,1], m_predict(xtest), 'xk')
# ax[1].plot(xtest[:,1], ypred + 2*spred, '.m', ms=2)
# ax[1].plot(xtest[:,1], ypred - 2*spred, '.m', ms=2)


# from scipy.spatial.distance import cdist
# print(cdist(xmes[:,0][:,None], xmes[:,0][:,None], metric='sqeuclidean'))
# print(rXX(xmes[:,0][:,None], xmes[:,0][:,None]))
