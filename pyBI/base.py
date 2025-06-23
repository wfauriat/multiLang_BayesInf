import numpy as np


# class random variable (infered or supplied parameter)
# - define prior
# - compute logprior (check if value within support)
# - generate proposal


# class observed variable
# - compute loglikelihood (for a given parameter value and a given conditional data)


# class inference strategy
# - define inference parameter (NMCMC, NBurn, Nthin)


# def loglikenp(obs, var, par, sigma, model):
#     obs = obs.reshape(-1,1)
#     try:
#         L = np.linalg.cholesky(sigma)
#     except np.linalg.LinAlgError:
#         raise ValueError("Covariance matrix is not positive semi-definite "+
#                          "(cannot perform Cholesky decomposition).")
#     log_det_sigma = 2.0 * np.sum(np.log(np.diag(L)))
#     diff = model(var,par).reshape(-1,1) - obs
#     y = np.linalg.solve(L, diff)
#     mahalanobis_term = np.sum(y**2)
#     log_likelihood = -0.5 * sigma.shape[0] * np.log(2*np.pi) + \
#                     -0.5 * log_det_sigma + \
#                     - 0.5 * mahalanobis_term
#     return log_likelihood

# def logpriornp(par, bounds):
#     bcheck = [(p > b[0]) & (p < b[1]) for (p,b) in zip(par, bounds)]
#     return 0 if all(bcheck) else -1000

# def logspnp(x, mu_scipy, loc, scale_scipy=10):
#     x_shifted = x - loc
#     nu = mu_scipy * scale_scipy
#     lambda_val = scale_scipy
#     log_pdf = (
#             0.5 * np.log(lambda_val)
#             - 0.5 * np.log(2 * np.pi)
#             - 1.5 * np.log(x_shifted)
#             - (lambda_val * np.power(x_shifted - nu, 2)) /
#               (2 * np.power(nu, 2) * x_shifted)
#         )
#     return -1000 if np.isnan(log_pdf) else log_pdf

# def log1(N):
#     return np.ones(N)