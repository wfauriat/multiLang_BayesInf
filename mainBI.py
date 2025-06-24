#%%
from pyBI.base import UnifVar, InvGaussVar, ObsVar
from pyBI.inference import InfAlgo, MHalgo

import numpy as np
import scipy.stats as sst

import matplotlib.pyplot as plt

np.random.seed(123)

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
sinvg = [0.4, 0.3, 1]
# sinvg = [0.05, 0.01, 10]

# sm = np.array([0.2]*Ndim)
sm = [0.4, 0.4, 0.05]
smexp = 0.1

covProp = np.eye(3)*1e-1
LLTprop = np.linalg.cholesky(covProp)

rndUs = [UnifVar([pl,ph]) for _ in range(3)]
rnds = InvGaussVar(param=sinvg)
obsvar = ObsVar(obs=ymes, prev_model=modelfit, cond_var=xmes)

bstart = np.array([2.2,-0.7,1.5, 0.5])


#%%

NMCMC = 20000

MCalgo = MHalgo(NMCMC, Nthin=20, Nburn=2000)
MCalgo.initialize(obsvar, rndUs, rnds, svar=sm, sdisc=smexp, Lblock=LLTprop)
MCalgo.MCchain[0] = bstart
MCalgo.state(0, set_state=True)
MCalgo.runInference()

print(MCalgo.nacc/(MCalgo.N - MCalgo.Nburn))

# nacc = 0
# 
# for i in range(1,MCalgo.N):
#     MCalgo.move_prop(i=i, x0=MCalgo.MCchain[i-1], Lblock=LLTprop)
#     llprop, lpprop = MCalgo.state(i)
#     ldiff = llprop + lpprop - MCalgo.log_state - MCalgo.prior_state
#     if ldiff > np.log(np.random.rand()):
#         if i>MCalgo.Nburn: nacc += 1
#         MCalgo.llchain[i] = llprop
#         MCalgo.log_state = llprop
#         MCalgo.prior_state = lpprop
#     else: MCalgo.stay(i)
#     if (i%1000 == 0) : print(i)
#     if (i%500 == 0) & (i<=MCalgo.Nburn) :
#         covProp = np.cov(MCalgo.MCchain[i-500:i,:MCalgo.Ndim].T) 
#         LLTprop = np.linalg.cholesky(covProp * 2.38**2/(Ndim-1) +
#                                     np.eye(Ndim)*1e-8)
#         MCalgo.Lblock = LLTprop
# 
# print(nacc / (NMCMC*0.9))


#%%

# for i in range(1,NMCMC):
#     MCalgo.move_prop(i=i, x0=MCalgo.MCchain[i-1])
#     llprop, lpprop = obsvar.loglike(xprop, rnds.diagSmat(s=sp, N=obsvar.Ndata))
#     ldiff = llprop + lspp - llold - lsold
#     if ldiff > np.log(np.random.rand()):
#         if i>Nburn: naccmultiD[Ndim] += 1
#         MCchain[i,Ndim] = sp
#         llchain[i] = llprop
#         llold = llprop
#         lsold = lspp
#     else:
#         MCchain[i,Ndim] = MCchain[i-1,Ndim]
#         llchain[i] = llchain[i-1]
#     idj = np.random.permutation(Ndim)
#     for j in idj:
#         xprop[j] = rndUs[j].proposal(MCchain[i-1,j],sm[j])
#         llprop = obsvar.loglike(xprop, rnds.diagSmat(s=sp, N=obsvar.Ndata))
#         lpprop = rndUs[j].logprior(xprop[j])
#         ldiff = llprop + lpprop - llold - lpold
#         if ldiff > np.log(np.random.rand()):
#             if i>Nburn: naccmultiD[j] += 1
#             MCchain[i,j] = xprop[j]
#             llchain[i] = llprop
#             llold = llprop
#             lpold = lpprop
#         else:
#             xprop[j] = MCchain[i-1,j]
#             MCchain[i,j] = xprop[j]
#             llchain[i] = llchain[i-1]
#     if (i%1000 == 0): print(i)

# # %%

MCf = MCalgo.MCchain[MCalgo.Nburn::MCalgo.Nthin,:]
llf = MCalgo.llchain[MCalgo.Nburn::MCalgo.Nthin]
idsort = np.argsort(llf)
MCchainS = MCf[idsort,:]
llchainS = llf[idsort]
Ntot = MCchainS.shape[0]
MAP = MCchainS[-1,:]
Nppost = Ntot
postY = np.array([sst.norm(loc=modelfit(xmes, MCchainS[i,:Ndim]),
                            scale=MCchainS[i,Ndim]).rvs(xmes.shape[0])
                              for i in range(Ntot-Nppost, Ntot)])


startchain = False

fig, ax = plt.subplots(Ndim+1,Ndim+1, figsize=(8,8))
for i in range(Ndim+1):
    for j in range(Ndim+1):
        if j>i:
            if startchain:
                ax[i,j].plot(MCalgo.MCchain[:MCalgo.Nburn,j],
                              MCalgo.MCchain[:MCalgo.Nburn,i],'-k',
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

plt.show()

# %%
