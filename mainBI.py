#%%
from pyBI.base import RandVar, ObsVar

import numpy as np
import matplotlib.pyplot as plt


#%%

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


#%%

Ndim = 3
pl = -3
ph = 3
bounds = np.repeat(np.c_[pl, ph],3, axis=0)
bstart = np.array([1.5,-0.7,1.5])
sinvg = [0.05, 0.01]
scinvg = 10

sm = np.array([0.2]*Ndim)
smexp = 0.2

#%%
rnd1 = RandVar(support=[pl,ph])
rnd2 = RandVar(support=[pl,ph])
rnd3 = RandVar(support=[pl,ph])
rnds = RandVar(param=sinvg + [scinvg])

obsvar = ObsVar(obs=ymes, prev_model=modelfit, cond_var=xmes)

s0 = rnds.param[0]*rnds.param[2] + rnds.param[1]
sigma_0 = np.diag(np.ones(ymes.shape[0])*s0**2)

print(rnd1.proposal_N(2, s=sm[0]))
print(obsvar.loglike(bstart[:3], sigma_0))


#%%

NMCMC = 25000
Nburn = 5000
Nthin = 20
Ntune = Nburn

Ndim = 3


MCchain = np.zeros((NMCMC, Ndim+1))
llchain = np.zeros(NMCMC)
MCchain[0,:Ndim] = bstart
MCchain[0,Ndim] = sinvg[0]*scinvg + sinvg[1]
xprop = MCchain[0,:Ndim]

llchain[0] = obsvar.loglike(bstart[:3], sigma_0)
llold = llchain[0]


for i in range(1,NMCMC):
    xprop = np.copy(MCchain[i-1,:Ndim])
    sp = rnlv(MCchain[i-1,Ndim],smexp)
    # llprop = loglike(ymes, xmes, xprop, sp, model=modelfit)
    llprop = loglikenp(ymes, xmes, xprop,
                    np.diag((sp**2)*np.ones(ymes.shape[0])),
                    model=modelfit)
    lspp = logspnp(sp, sinvg[0], sinvg[1], scale_scipy=scinvg)
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
        lpprop = logpriornp(xprop, bounds=bounds)
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