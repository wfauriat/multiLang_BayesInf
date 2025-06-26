import numpy as np

class RandVar():
    def __init__(self, param=None):
        if param is not None: self.param = param

    def logprior(self, x):
        return NotImplemented
    
    def proposal(self):
        return NotImplemented
    
    def draw(self):
        return NotImplemented


def rnvmultiD(m, Lmat):
    # Lmat = np.linalg.cholesky(Cov)
    return m + np.ravel(Lmat @ np.random.randn(m.shape[0],1))


class UnifVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = self.param[0]
        self.max = self.param[1]

    def logprior(self, x):
        bcheck = (x > self.param[0]) & (x < self.param[1])
        logp = 1/(self.param[1] - self.param[0]) if bcheck else -1000
        return logp
    
    def proposal(self, m=0, s=1):
        return np.random.randn()*np.array(s)+m
    
    def draw(self):
        return np.random.rand()*(self.param[1] - self.param[0]) + self.param[0]


class InvGaussVar(RandVar):
    def __init__(self, param=None):
        super().__init__(param)
        self.min = 0
        self.max = (self.param[0]*self.param[2] + self.param[1])*4
    
    def logprior(self, x, loc=None, scale=None):
        mu = self.param[0]
        loc = self.param[1] if loc is None else loc
        scale = self.param[2] if scale is None else scale
        x_shifted = x - loc
        nu = mu * scale
        lambda_val = scale
        log_pdf = (0.5 * np.log(lambda_val) - 0.5 * np.log(2 * np.pi)
                - 1.5 * np.log(x_shifted) - (lambda_val * np.power(
                    x_shifted - nu, 2)) / (2 * np.power(nu, 2) * x_shifted))
        return -1000 if np.isnan(log_pdf) else log_pdf

    def proposal(self, m, s):
        prop = np.exp(np.random.randn()*s+np.log(m)) 
        return prop if prop != -np.inf else 0.0

    def draw(self, N=1):
        return np.random.wald(mean=self.param[0]*self.param[2],
                              scale=self.param[2], size=N) + self.param[1]
    
    def mean(self):
        return self.param[0]*self.param[2] + self.param[1] if \
              len(self.param) > 2 else self.param[0] + self.param[1]

    def diagSmat(self, s=None, N=1):
        s0 = self.param[0]*self.param[2] + self.param[1] if s is None else s
        return np.eye(N)*s0**2


class ObsVar():
    def __init__(self, obs, prev_model, cond_var=None):
        self.obs = obs
        self.Ndata = obs.shape[0]
        if cond_var is not None: self.cond_var = cond_var
        self.prev_model = prev_model

    def loglike(self, par, sigma):
        obs = self.obs.reshape(-1,1)
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is not positive semi-definite "+
                             "(cannot perform Cholesky decomposition).")
        log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
        diff = self.prev_model(self.cond_var,par).reshape(-1,1) - obs
        y = np.linalg.solve(L, diff)
        mahalanobis_term = np.sum(y**2)
        log_likelihood = -0.5 * sigma.shape[0] * np.log(2*np.pi) + \
                        -0.5 * log_det_sigma + \
                        - 0.5 * mahalanobis_term
        return log_likelihood
