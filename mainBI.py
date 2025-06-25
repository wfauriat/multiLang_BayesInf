#%%
from pyBI.base import UnifVar, InvGaussVar, ObsVar
from pyBI.inference import MHalgo, MHwGalgo

import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


np.random.seed(123)

#%%

case = 0
# case = 1

inftype = 'MH'
# inftype = 'MHwG'

#%%

if case == 0:
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
                    biasp1+nsp1*np.c_[np.random.randn(10)]])
    ymes = modeltrue(xmes, b0)
    ymes += np.random.randn(xmes.shape[0])*nslvl

if case == 1:
    def ishigami(x,b):
        return np.c_[np.sin(x[:,0]) + b[:,0]*np.sin(x[:,1])**2 +  \
               b[:,1]*np.sin(x[:,0])*x[:,2]**4] 

    Xtrain = sst.qmc.scale(sst.qmc.LatinHypercube(d=5).random(100),
                            [-3, -3, -3, 4, 0.01], [3, 3, 3, 9, 0.3])
    DXtrain = Xtrain.max(axis=0) - Xtrain.min(axis=0)
    lowD = DXtrain/20
    highD = DXtrain*5
    ytrain = ishigami(Xtrain[:,:3],Xtrain[:,3:])

    XX = np.array([[0,0,0],
                [-1.3,0,-1],
                [0.2,1,-1],
                [2,1,-2],
                [3.4,1,-1],
                [2.5,-2,-1],
                [3,3,-2],
                [-1,-1,2],
                [2.7,-2,0],
                [1,1,-2]])
    b3 = np.array([7,0.1])

    kernel = RBF([1.0]*5, 
                [(el1, el2) for el1,el2 in zip(lowD, highD)]) + \
        WhiteKernel(noise_level=0.001,
                    noise_level_bounds=(1e-6,np.var(ytrain)*0.05))

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

    gp.fit(Xtrain, ytrain)

    def modelgp(x, b):
        return gp.predict(np.hstack([x,
                    np.repeat(np.r_[[b]],x.shape[0], axis=0)])).ravel()

    ytrue = ishigami(XX,
                np.repeat(np.atleast_2d(b3),XX.shape[0], axis=0)).ravel()
    ypred = modelgp(XX, b3)
    Q2emp = 1-np.var(ytrue-ypred)/np.var(ypred)
            
    if True:
        fig, ax = plt.subplots()
        ax.plot(XX[:,0], ytrue, 'or')
        ax.plot(XX[:,0], ypred, '.b')
        axx = ax.twiny()
        axx.plot(ytrue, ypred, '+k')
        axx.plot(ytrue, ytrue, '+-k')
        ax.set_title('Q2emp=' + \
                    str(np.round(Q2emp,3)))

    ymes = ishigami(XX, np.repeat(np.r_[[b3]],XX.shape[0], axis=0)).ravel()

#%%

if case == 0:
    Ndim = 3
    pl = -3
    ph = 3
    sinvg = [0.4, 0.3, 1]

    sm = [0.4, 0.4, 0.05]
    smexp = 0.1
    covProp = np.eye(3)*1e-1
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([pl,ph]) for _ in range(3)]
    rnds = InvGaussVar(param=sinvg)
    obsvar = ObsVar(obs=ymes, prev_model=modelfit, cond_var=xmes)

    bstart = np.array([2.2,-0.7,1.5, 0.5])

if case == 1:
    Ndim = 2
    sinvg = [0.4, 0.3, 1]

    sm = [0.2, 5e-2]
    smexp = 0.2
    covProp = np.array([[0.2,0],[0,0.05]])
    LLTprop = np.linalg.cholesky(covProp)

    rndUs = [UnifVar([0,10]), UnifVar([0,1])]
    rnds = InvGaussVar(param=sinvg)
    obsvar = ObsVar(obs=ymes, prev_model=modelgp, cond_var=XX)

    bstart = [2, 0., 0.5]


#%%

NMCMC = 20000

if inftype == 'MH':
    MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=5000, is_adaptive=True,
                     verbose=True)
if inftype == 'MHwG':
    MCalgo = MHwGalgo(NMCMC, Nthin=20, Nburn=5000, is_adaptive=True,
                       verbose=True)
MCalgo.initialize(obsvar, rndUs, rnds, svar=sm, sdisc=smexp)
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCout, llout = MCalgo.runInference()

#%%

fig0, ax0 = MCalgo.post_visupar()
fig1, ax1 = MCalgo.diag_chain(2)

