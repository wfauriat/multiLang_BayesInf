import numpy as np
from .base import rnvmultiD

class InfAlgo():
    def __init__(self, N, Nthin=None, Nburn=0.1):
        self.Nburn = int(N*0.1) if Nburn == 0.1 else Nburn
        self.Nthin = Nthin if Nthin is not None else 1
        self.N = N + self.Nburn
        self.log_state = 0
        self.prior_state = 0

    def initialize(self, obsObj, varObj, discrObj, svar=1, sdisc=1,
                    Lblock=1):
        self.Ndim = len(varObj)
        self.varObj = varObj
        self.discrObj = discrObj
        self.obsObj = obsObj
        self.MCchain = np.zeros((self.N, self.Ndim+1))
        self.llchain = np.zeros(self.N)
        self.svar = svar*np.ones(self.Ndim) if isinstance(1.,(int,float)) \
            else svar
        self.sdisc = sdisc
        self.Lblock = np.eye(self.Ndim) if \
              Lblock is isinstance(1.,(int,float)) else Lblock
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
    
    def move_prop(self, i=None, x0=None, Lblock=None):
        if x0 is not None:
            if Lblock is None:
                xprop = [self.varObj[k].proposal(x0[k], self.svar[k]) 
                        for k in range(len(self.varObj))]
            elif isinstance(Lblock, np.ndarray):
                xprop = list(rnvmultiD(np.array(x0[:self.Ndim]), Lblock))
            else:
                xprop = list(rnvmultiD(np.array(x0[:self.Ndim]), self.Lblock))
            sprop = self.discrObj.proposal(x0[self.Ndim], self.sdisc)
            if i is not None:
                self.MCchain[i,:self.Ndim] = xprop
                self.MCchain[i,self.Ndim] = sprop
            return xprop + [sprop]
        else:
            if i is None: i=0
            xprop = [self.varObj[k].proposal(self.MCchain[i,k], self.svar[k]) 
                     for k in range(len(self.varObj))]
            sprop = self.discrObj.proposal(self.MCchain[i,self.Ndim],
                                            self.sdisc)
            return xprop + [sprop]
        
    def stay(self, i):
        self.MCchain[i] = self.MCchain[i-1]
        self.llchain[i] = self.llchain[i-1]

    

class MHalgo(InfAlgo):
    def __init__(self, N, Nthin=None, Nburn=0.1):
        super().__init__(N, Nthin, Nburn)

    def runInference(self):
        self.nacc = 0 
        for i in range(1,self.N):
            self.move_prop(i=i, x0=self.MCchain[i-1], Lblock=self.Lblock)
            llprop, lpprop = self.state(i)
            ldiff = llprop + lpprop - self.log_state - self.prior_state
            if ldiff > np.log(np.random.rand()):
                if i>self.Nburn: self.nacc += 1
                self.llchain[i] = llprop
                self.log_state = llprop
                self.prior_state = lpprop
            else: self.stay(i)
            if (i%1000 == 0) : print(i)
            if (i%500 == 0) & (i<=self.Nburn) :
                covProp = np.cov(self.MCchain[i-500:i,:self.Ndim].T) 
                LLTprop = np.linalg.cholesky(covProp * 2.38**2/(self.Ndim-1) +
                                            np.eye(self.Ndim)*1e-8)
                self.Lblock = LLTprop
        return self.MCchain