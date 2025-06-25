import numpy as np
from itertools import combinations
from warnings import warn
from matplotlib.pyplot import subplots

from scipy.stats import gaussian_kde

from .base import rnvmultiD

class InfAlgo():
    def __init__(self, N, Nthin=None, Nburn=0.1, is_adaptive=True,
                 verbose=True):
        self.Nburn = int(N*0.1) if Nburn == 0.1 else Nburn
        self.Nthin = Nthin if Nthin is not None else 1
        self.N = N + self.Nburn
        self.log_state = 0
        self.prior_state = 0
        self.is_adaptive = is_adaptive
        self.verbose = verbose

    def initialize(self, obsObj, varObj, discrObj, svar=1, sdisc=1,
                    Lblock=None):
        self.Ndim = len(varObj)
        self.varObj = varObj
        self.discrObj = discrObj
        self.obsObj = obsObj
        self.ran_chain = False
        self.MCchain = np.zeros((self.N, self.Ndim+1))
        self.llchain = np.zeros(self.N)
        self.svar = svar*np.ones(self.Ndim) if isinstance(svar,(int,float)) \
            else svar
        self.sdisc = sdisc
        if isinstance(Lblock,(int,float)):
            self.Lblock = np.eye(self.Ndim)
        elif Lblock is None:
            self.Lblock = np.diag(self.svar)
        else: self.Lblock = Lblock
        self.MCchain[0,:self.Ndim] = [varObj[i].draw() 
                                      for i in range(len(varObj))]
        self.MCchain[0,self.Ndim] = discrObj.draw()
        self.state(0, set_state=True)

    def state(self, i, set_state=False):
        log_state = self.obsObj.loglike(self.MCchain[i,:self.Ndim], 
                        self.discrObj.diagSmat(s=self.MCchain[i,self.Ndim],
                                       N=self.obsObj.Ndata))
        prior_state = np.sum([self.varObj[k].logprior(self.MCchain[i,k]) 
                              for k in range(len(self.varObj))]) + \
                              self.discrObj.logprior(self.MCchain[i,self.Ndim])
        if set_state:
            self.llchain[i] = log_state
            self.log_state = log_state
            self.prior_state = prior_state
        else: return log_state, prior_state
    
    def move_prop(self, x0=None, i=0, Lblock=None):
        if Lblock is not None: Ltmp = Lblock # rather for testing purposes
        else: Ltmp = self.Lblock# default behaviour
        if x0 is not None: # default behaviour
            xprop = list(rnvmultiD(np.array(x0[:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(x0[self.Ndim], self.sdisc)
        else: # rather for visualisation purpose
            xprop = list(rnvmultiD(np.array(self.MCchain[i,:self.Ndim]), Ltmp))
            sprop = self.discrObj.proposal(self.MCchain[i,self.Ndim],
                                            self.sdisc)
        return xprop + [sprop]
    
    def move_uni(self, di, x0):
        xprop = list(x0[:self.Ndim])
        sprop = x0[self.Ndim]
        if di < self.Ndim:
            xprop[di] = self.varObj[di].proposal(x0[di], self.svar[di])
        else: sprop = self.discrObj.proposal(x0[di], self.sdisc)
        return xprop + [sprop]
    
    def store(self, i, xprop, sprop):
        self.MCchain[i,:self.Ndim] = xprop
        self.MCchain[i,self.Ndim] = sprop
        
    def stay(self, i, j=None):
        if j is None:
            self.MCchain[i] = self.MCchain[i-1]
            self.llchain[i] = self.llchain[i-1]
        else: 
            self.MCchain[i,j] = self.MCchain[i-1,j]
            self.llchain[i] = self.log_state

    def thin_and_sort(self):
        if not self.ran_chain: 
            self.ran_chain = True
            self.raw_chain = self.MCchain
            self.cut_chain = self.raw_chain[self.Nburn::self.Nthin]
            self.cut_llchain = self.llchain[self.Nburn::self.Nthin]
            self.sorted_indices = np.argsort(self.cut_llchain)
            self.idx_chain = self.cut_chain[self.sorted_indices,:]
            self.MAP = self.idx_chain[-1]
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        else: 
            warn("Already ran and sorted")
            return self.idx_chain, self.cut_llchain[self.sorted_indices]
        
    def post_obs(self, model=None):
        Mtmp = model if model is not None else self.obsObj.prev_model
        postY = np.zeros((int((self.N - self.Nburn)/self.Nthin),
                          self.obsObj.Ndata))
        for i in range(postY.shape[0]):
            postY[i,:] = Mtmp(self.obsObj.cond_var,
                             self.idx_chain[i,:self.Ndim]) + \
                        np.random.randn()*self.idx_chain[i,self.Ndim]
        return postY
    
    def post_visuobs(self, di=0):
        fig, ax = subplots()
        postY = self.post_obs()
        ax.plot(self.obsObj.cond_var[:,di], postY.T, '.k')
        ax.plot(self.obsObj.cond_var[:,di], self.obsObj.obs, 'or')
        return fig, ax
    
    def post_visupar(self):
        MC = self.idx_chain
        LL = self.cut_llchain[self.sorted_indices]
        quadrans = list(combinations(
                ['v' + str(i) for i in range(self.Ndim)] + ['d'],2))
        quad_idx = list(combinations(list(range(self.Ndim+1)),2))
        Nbplot = int(np.ceil(np.sqrt(len(quadrans))))
        k = 0
        fig, ax = subplots(Nbplot, Nbplot)
        for i in range(Nbplot):
            for j in range(Nbplot):
                if k<len(quadrans):
                    ax[i,j].scatter(MC[:,quad_idx[k][0]], MC[:,quad_idx[k][1]],
                                    c=LL, cmap='jet', marker='.')
                    ax[i,j].set_xlabel(quadrans[k][0])
                    ax[i,j].set_ylabel(quadrans[k][1])
                    ax[i,j].plot(self.MAP[quad_idx[k][0]],
                                  self.MAP[quad_idx[k][1]], 'dk')
                    k += 1
                else: ax[i,j].set_visible(False)
        fig.tight_layout()
        return fig, ax
    
    def diag_chain(self, di=0):
        MC = self.cut_chain
        Nc = int(self.cut_chain.shape[0]/4)
        kdes = [gaussian_kde(MC[i*Nc:(i+1)*Nc,di]) for i in range(0,4)]
        fig, ax = subplots(2,1)
        ax[0].plot(self.raw_chain[:,di], '-k', lw=0.5)
        ax[1].hist(self.idx_chain[:,di], edgecolor='k', alpha=0.6)
        axx = ax[1].twinx()
        xi = np.linspace(MC[:,di].min(), MC[:,di].max())
        for i in range(4):
            axx.plot(xi, kdes[i].pdf(xi), 'k')
        return fig, ax



    

class MHalgo(InfAlgo):
    def __init__(self, N, Nthin=None, Nburn=0.1, is_adaptive=True,
                 verbose=True):
        super().__init__(N, Nthin, Nburn, is_adaptive, verbose)

    def runInference(self):
        self.nacc = 0 
        for i in range(1,self.N):
            prop = self.move_prop(self.MCchain[i-1])
            self.store(i, prop[:self.Ndim], prop[self.Ndim])
            llprop, lpprop = self.state(i)
            ldiff = llprop + lpprop - self.log_state - self.prior_state
            if ldiff > np.log(np.random.rand()):
                if i>self.Nburn: self.nacc += 1
                self.llchain[i] = llprop
                self.log_state = llprop
                self.prior_state = lpprop
            else: self.stay(i)
            if (i%1000 == 0) & (self.verbose) : print(i)
            if (i%500 == 0) & (i<=self.Nburn) & (self.is_adaptive):
                covProp = np.cov(self.MCchain[i-500:i,:self.Ndim].T) 
                LLTprop = np.linalg.cholesky(covProp * 2.38**2/(self.Ndim-1) +
                                            np.eye(self.Ndim)*1e-8)
                self.Lblock = LLTprop
        MCS = self.thin_and_sort()
        self.tacc = self.nacc/(self.N - self.Nburn)
        if self.verbose: print('acceptation rate', self.tacc)
        return MCS[0], MCS[1]
    

class MHwGalgo(InfAlgo):
    def __init__(self, N, Nthin=None, Nburn=0.1, is_adaptive=True,
                 verbose=False):
        super().__init__(N, Nthin, Nburn, is_adaptive, verbose)

    def runInference(self):
        self.nacc = np.zeros(self.Ndim+1) 
        for i in range(1,self.N):
            self.MCchain[i] = self.MCchain[i-1]
            id_rnd = np.random.permutation(self.Ndim+1)
            for idx in id_rnd:
                prop = self.move_uni(idx, self.MCchain[i])
                self.store(i, prop[:self.Ndim], prop[self.Ndim])
                llprop, lpprop = self.state(i)
                ldiff = llprop + lpprop - self.log_state - self.prior_state
                if ldiff > np.log(np.random.rand()):
                    if i>self.Nburn: self.nacc[idx] += 1
                    self.llchain[i] = llprop
                    self.log_state = llprop
                    self.prior_state = lpprop
                else: self.stay(i,idx)
            if (i%1000 == 0) & (self.verbose): print(i)
            if (i%500 == 0) & (i<=self.Nburn) & (self.is_adaptive):
                for idx in id_rnd:
                    if idx != self.Ndim:
                        self.svar[idx] = np.std(self.MCchain[i-500:i,idx])*1 
        MCS = self.thin_and_sort()
        self.tacc = self.nacc/(self.N - self.Nburn)
        if self.verbose: print('acceptation rate', self.tacc)
        return MCS[0], MCS[1]
    