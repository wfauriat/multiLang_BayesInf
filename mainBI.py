#%%
from pyBI.base import UnifVar, InvGaussVar, ObsVar
from pyBI.inference import MHalgo, MHwGalgo, InfAlgo

import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


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



#%%

# how to appropriately store infered parameter so that it works for different
# application cases ?
# par_vect = [[fmod_par : "beta",
#              gpm_par : "theta",
#              extra_cov_par : "sigma" or "Sigma"]]
# cond_var or no cond_var : "x"



from scipy.optimize import minimize 

np.set_printoptions(suppress=True, precision=3)

class GaussLike():
    def __init__(self, obs, gmod=None, x=None):
        self.obs = obs
        self.x = x
        self.gmod = gmod
        self.Ndata = obs.shape[0]
        self.dimdata = obs.shape[1]

    def setpar(self, gpar, kpar, spar):
        self.gpar = gpar # parameters of g(x,beta)
        self.kpar = kpar # parameters of s * exp (-d**2/t**2)
        self.spar = spar # diagonal parameter of sigma (or full Sigma)
    
    def gpredict(self, x, par=None):
        ptmp = self.par if par is None else par
        return self.gmod(x, ptmp) if ptmp is not None else \
              np.zeros(x.shape[0])
    


class HGP():
    def __init__(self, kpar, spar, XX, y):
        self.kpar = kpar
        self.spar = spar
        self.XX = XX
        self.y = y
        self.cond()

    def cond(self):
        self.k_XX = self.KaXX(self.XX) + np.eye(self.XX.shape[0])*1e-8
        self.L_XX = np.linalg.cholesky(self.k_XX)
        self.alpha = np.linalg.solve(self.L_XX.T,
                                      np.linalg.solve(self.L_XX, self.y))

    def setpar(self, kpar, spar, cond=True):
        self.kpar = kpar
        self.spar = spar
        if cond: self.cond()
    
    def raXX(self, X, Y, scl=None):
        if scl is None: scl = self.kpar[1:]
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        scaled_diff_sq = (diff / scl)**2
        squared_weighted_distance = np.sum(scaled_diff_sq, axis=2)
        return squared_weighted_distance 

    def KaXX(self, X, theta=None, sigma=None):
        if theta is None: theta = self.kpar[:1]
        if sigma is None: sigma = [self.kpar[0], self.spar]
        return sigma[0]**2 * np.exp(-self.raXX(X,X,theta)/2) + \
            np.eye(X.shape[0])*sigma[1]
    
    def KaxX(self, x, X, theta=None, sigma=None):
        if theta is None: theta = self.kpar[:1]
        if sigma is None: sigma = [self.kpar[0], self.spar]
        return sigma[0]**2 * np.exp(-self.raXX(x,X, theta)/2)
    
    def m_predict(self, x):
        K_xX = self.KaxX(x, self.XX)
        mu_pred = K_xX @ self.alpha
        return mu_pred
    
    def cov_predict(self, x):
        K_xX = self.KaxX(x, self.XX)
        v = np.linalg.solve(self.L_XX.T, np.linalg.solve(self.L_XX, K_xX.T))
        cov_pred = self.KaxX(x, x, self.kpar[:1],
                        np.array([self.kpar[0], self.spar])) - v.T @ v
        return cov_pred
    
    def loglike(self, par=None):
        if par is None: params = np.hstack([self.kpar, self.spar])
        else: params = par
        d = self.XX.shape[1]
        N = self.XX.shape[0]
        s = np.exp(params[0])
        t = np.exp(params[1:1+d])
        sn = np.exp(params[1+d])
        # print(s,t,sn)
        K_XX = self.KaXX(self.XX, t, [s, sn])
        L = np.linalg.cholesky(K_XX)
        log_det_K_noisy = 2 * np.sum(np.log(np.diag(L)))
        alpha = np.linalg.solve(L.T,np.linalg.solve(L, self.y))
        data_fit_term = self.y.T @ alpha
        log_marginal_likelihood = -0.5 * data_fit_term - \
                        0.5 * log_det_K_noisy - \
                        0.5 * N * np.log(2 * np.pi)
        return -log_marginal_likelihood


# def fmod(x, beta):
#     return beta[0]*x + beta[1]*x**2
fmod = modelfit
b0 = [2, -1, 2, 0]
theta = np.array([0.5, 10])
sigma = np.array([1])
sn = np.array(0.01)


LLobj = GaussLike(ymes, gmod=modelfit, x=xmes)
LLobj.par = b0

gpobj = HGP(np.hstack([sigma, theta]), sn, xmes, ymes.ravel())


bounds = (
    (np.log(1e-3), np.log(1e2)),  # sigma_f: e.g., 0.001 to 1000
    (np.log(1e-3), np.log(1e1)),   # length_scale: e.g., 0.001 to 1000
    (np.log(1e-3), np.log(1e1)),    # length_scale: e.g., 0.001 to 1000
    (np.log(1e-8), np.log(1e-4))   # noise_variance: e.g., 1e-6 to 10
)

result = minimize(
    fun=gpobj.loglike,
    x0=np.log([np.var(ymes)*0.001, 1, 100, 0.01]),
    method='L-BFGS-B',
    bounds=bounds,
    # options={'disp': True, 'maxiter': 1000}
)

print(np.exp(result.x))

xtest = np.linspace(0.5,5.4,20)
xtest = np.vstack([xtest, np.random.randn(xtest.shape[0])*0.1-1]).T
# sigma = np.exp(result.x[0])
# theta = np.exp(result.x[1:3])
# sn = np.exp(result.x[3])

sigma = np.array([5])
theta = np.array([3.06, 1.16])
sn = np.array(0.5)

gpobj.setpar(np.hstack([sigma, theta]), sn)


# L = np.linalg.cholesky(KaXX(xmes, theta, [sigma, sn]))
# alpha = np.linalg.solve(L.T, np.linalg.solve(L, ymes.ravel()))

# mu_pred = KaxX(xtest, xmes, theta, sigma) @ alpha
# v = np.linalg.solve(L, KaxX(xtest, xmes, theta, sigma).T)
# cov_pred = KaxX(xtest, xtest, theta, sigma) - v.T @ v
# spred = np.diag(cov_pred)

ypred = gpobj.m_predict(xtest)
spred = np.diag(gpobj.cov_predict(xtest))


# fig, ax = plt.subplots()
# ax.plot(xmes[:,0], ymes.ravel(), 'or')
# ax.plot(xtest[:,0], ypred, 'x-b')
# ax.fill_between(x=xtest[:,0], y1=ypred + 2*spred**0.5,
#                 y2=ypred -2*spred**0.5, alpha=0.2, color='b')

fig, ax = plt.subplots()
ax.plot(xmes[:,0], ymes.ravel(), 'or')
ax.plot(xmes[:,0], gpobj.m_predict(xmes), 'x-b')
ax.fill_between(x=xmes[:,0],
                y1=gpobj.m_predict(xmes) + 
                    2*np.diag(gpobj.cov_predict(xmes))**0.5,
                y2=gpobj.m_predict(xmes) - 
                    2*np.diag(gpobj.cov_predict(xmes))**0.5,
                      alpha=0.2, color='b')


# from scipy.spatial.distance import cdist
# print(cdist(xmes[:,0][:,None], xmes[:,0][:,None], metric='sqeuclidean'))
# print(rXX(xmes[:,0][:,None], xmes[:,0][:,None]))

def gpm(x, theta, sigma, Kinv=None, X=None):
    return x*theta*sigma

def meanGauss(x, beta, theta, sigma, Kinv=None, X=None):
    return fmod(x, beta) + gpm(x, theta, sigma, Kinv, X)
def covGauss(x, theta, sigma, Kinv=None):
    return 4

# obs = ymes

# N = obs.shape[0]
# d = obs.shape[1]
# try:
#     L = np.linalg.cholesky(covf(x))
# except np.linalg.LinAlgError:
#     raise ValueError("Covariance matrix is not positive semi-definite "+
#                         "(cannot perform Cholesky decomposition).")
# log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
# diff = meanf(x) - obs
# mahalanobis_term = np.zeros(N)
# for i in range(N):
#     y = np.linalg.solve(L, diff[i, :])
#     mahalanobis_term[i] = np.sum(y**2)
# log_likelihood = - (N * d / 2) * np.log(2 * np.pi) \
#             - (N / 2) * log_det_sigma \
#             - (1 / 2) * np.sum(mahalanobis_term)