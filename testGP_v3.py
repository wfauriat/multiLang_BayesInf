#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize # For hyperparameter optimization

class GaussianProcess:
    def __init__(self, kernel_func, noise_variance=1e-8):
        """
        Initializes the Gaussian Process.

        Args:
            kernel_func: A callable function that takes two arrays (X1, X2)
                         and returns the kernel (covariance) matrix.
            noise_variance: The assumed noise variance (sigma_n^2). This is
                            often added to the diagonal of the covariance matrix
                            to ensure numerical stability and account for noise.
        """
        self.kernel_func = kernel_func
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K = None # Training kernel matrix
        self.L = None # Cholesky decomposition of K

    def fit(self, X_train, y_train):
        """
        Trains the Gaussian Process.

        Args:
            X_train (np.ndarray): Training input data (n_samples, n_features).
            y_train (np.ndarray): Training target values (n_samples,).
        """
        self.X_train = X_train
        self.y_train = y_train.flatten() # Ensure y_train is 1D

        # Compute the kernel matrix for training data
        self.K = self.kernel_func(X_train, X_train)

        # Add noise variance to the diagonal for numerical stability and observation noise
        self.K += self.noise_variance * np.eye(len(X_train))

        # Perform Cholesky decomposition for efficient and stable inversion
        # L is a lower-triangular matrix such that K = L @ L.T
        self.L = np.linalg.cholesky(self.K)

    def predict(self, X_test, return_std=True):
        """
        Makes predictions with the trained Gaussian Process.

        Args:
            X_test (np.ndarray): Test input data (n_samples_test, n_features).
            return_std (bool): If True, returns the standard deviation of the
                               predictions along with the mean.

        Returns:
            tuple:
                - y_mean (np.ndarray): Mean of the predictive distribution.
                - y_std (np.ndarray, optional): Standard deviation of the
                                                predictive distribution.
        """
        if self.X_train is None:
            raise RuntimeError("The model has not been fitted yet. Call .fit() first.")

        # Compute cross-covariance between test points and training points
        K_s = self.kernel_func(X_test, self.X_train)

        # Compute auto-covariance for test points
        K_ss = self.kernel_func(X_test, X_test)

        # Solve for alpha (K_inv @ y_train), but using Cholesky decomposition for stability
        # K_inv @ y = (L @ L.T)_inv @ y = L.T_inv @ L_inv @ y
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

        # Predictive mean
        y_mean = K_s @ alpha

        if not return_std:
            return y_mean

        # Predictive covariance (K_ss - K_s @ K_inv @ K_s.T)
        # Using Cholesky: v = L_inv @ K_s.T
        v = np.linalg.solve(self.L, K_s.T)
        y_variance = np.diag(K_ss) - np.sum(v**2, axis=0)

        # Ensure variance is non-negative due to potential numerical issues
        y_variance = np.maximum(0, y_variance)
        y_std = np.sqrt(y_variance)

        return y_mean, y_std

# ---------------------------------------------------
# RBF (Squared Exponential) Kernel Implementation
# ---------------------------------------------------
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) kernel, also known as Squared Exponential kernel.

    k(x_i, x_j) = sigma_f^2 * exp(-0.5 * ||x_i - x_j||^2 / length_scale^2)

    Args:
        X1 (np.ndarray): First set of points (n_samples1, n_features).
        X2 (np.ndarray): Second set of points (n_samples2, n_features).
        length_scale (float): The length scale parameter (l).
        sigma_f (float): The signal variance (amplitude) parameter (sigma_f).

    Returns:
        np.ndarray: The kernel (covariance) matrix.
    """
    # Calculate squared Euclidean distances between all pairs of points
    # (x_i - x_j)^2 = x_i^2 - 2*x_i*x_j + x_j^2
    # Sum of squares for X1 (row vector)
    sq_dist_X1 = np.sum(X1**2, axis=1)[:, np.newaxis]
    # Sum of squares for X2 (row vector)
    sq_dist_X2 = np.sum(X2**2, axis=1)[np.newaxis, :]
    # Dot product term
    dot_product = -2 * X1 @ X2.T

    sq_distances = sq_dist_X1 + dot_product + sq_dist_X2

    # Apply the RBF formula
    K = (sigma_f**2) * np.exp(-0.5 * sq_distances / (length_scale**2))
    return K

# ---------------------------------------------------
# Example Usage: Training and Prediction
# ---------------------------------------------------
#%%
# 1. Generate some synthetic data
def true_function(x):
    return np.sin(x) + 0.5 * np.sin(2 * x)

# Training data
X_train = np.array([0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5]).reshape(-1, 1)
# Add some noise to observations
y_train = true_function(X_train).flatten() + np.random.normal(0, 0.2, X_train.shape[0])

# Test data (for prediction)
X_test = np.linspace(-2, 12, 200).reshape(-1, 1)

# Set hyperparameters (these would ideally be optimized)
initial_length_scale = 2
initial_sigma_f = 1.0
noise_variance = 0.01 # This is sigma_n^2

# initial_length_scale = 0.9485
# initial_sigma_f = 0.7881
# noise_variance = 3.6055e-07 # This is sigma_n^2

from pyBI.base import HGP
gpobj = HGP(X_train, y_train)
kpar = [initial_sigma_f, initial_length_scale]
spar = noise_variance
gpobj.setpar([initial_sigma_f, initial_length_scale], noise_variance)
# gpobj.default_tune(update=True)

ypred =  gpobj.m_predict(X_test)
spred = np.diag(gpobj.cov_predict(X_test, noise=False))

# Create a lambda function for the kernel to fix hyperparameters for GP class
# In a full implementation, these would be attributes of the GP class and optimized
my_kernel = lambda X1, X2: rbf_kernel(X1, X2, length_scale=initial_length_scale, sigma_f=initial_sigma_f)

# Initialize and train the Gaussian Process
gp_model = GaussianProcess(kernel_func=my_kernel, noise_variance=noise_variance)
gp_model.fit(X_train, y_train)

# Make predictions
y_pred_mean, y_pred_std = gp_model.predict(X_test, return_std=True)

#%%
# ---------------------------------------------------
# Plotting the results
# ---------------------------------------------------
fig, ax = plt.subplots()
ax.plot(X_test, true_function(X_test), 'r:', label='True function')
ax.plot(X_train, y_train, 'ro', markersize=8, label='Observations')
ax.plot(X_test, y_pred_mean, 'b-', label='GP mean prediction')
ax.plot(X_test, ypred, 'm')
ax.fill_between(X_test.flatten(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 alpha=0.4, color='lightblue', label='95% confidence interval')
ax.fill_between(X_test.flatten(),
                 ypred - 2 * spred**0.5,
                 ypred + 2 * spred**0.5,
                 alpha=0.2, color='m', label='95% confidence interval')

ax.set_ylim(-2,2)

ax.set_title('Gaussian Process Regression (from scratch) with RBF Kernel')
ax.legend(loc='upper left')
ax.grid(True)
plt.show()

#%%

# ---------------------------------------------------
# Optional: Simple Hyperparameter Optimization (Gradient-Free)
# This is a very basic example; a real-world scenario would use more robust optimizers
# and potentially gradients of the log-marginal-likelihood.
# ---------------------------------------------------

def neg_log_marginal_likelihood(params, X, y):
    """
    Negative log-marginal-likelihood function for optimization.
    We minimize this to maximize the log-marginal-likelihood.

    Args:
        params (np.ndarray): Array of [log(length_scale), log(sigma_f), log(noise_variance)].
                             Using log-parameters for better optimization.
        X (np.ndarray): Training input data.
        y (np.ndarray): Training target values.

    Returns:
        float: The negative log-marginal-likelihood.
    """
    length_scale = np.exp(params[0])
    sigma_f = np.exp(params[1])
    noise_variance = np.exp(params[2])

    K = rbf_kernel(X, X, length_scale=length_scale, sigma_f=sigma_f)
    K += noise_variance * np.eye(len(X))

    try:
        L = np.linalg.cholesky(K)
        # log-marginal-likelihood = -0.5 * y.T @ K_inv @ y - sum(log(diag(L))) - 0.5 * N * log(2*pi)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y.flatten()))
        nll = 0.5 * y.flatten().T @ alpha + np.sum(np.log(np.diag(L))) + \
            0.5 * len(X) * np.log(2 * np.pi)
        return nll
    except np.linalg.LinAlgError:
        # Handle cases where K is not positive definite (e.g., due to bad hyperparameters)
        return np.inf

print("\n--- Hyperparameter Optimization (simple L-BFGS-B) ---")

# Initial guess for log-hyperparameters
# [log(length_scale), log(sigma_f), log(noise_variance)]
initial_params = np.log([initial_length_scale, initial_sigma_f, noise_variance])

# Bounds for optimization (optional but good practice to prevent extreme values)
bounds = [(np.log(1e-2), np.log(1e2)), # length_scale
          (np.log(1e-2), np.log(1e2)), # sigma_f
          (np.log(1e-8), np.log(1e0))] # noise_variance (can be small but not zero)

# Optimize using L-BFGS-B (a common gradient-free optimizer for bounded problems)
result = minimize(neg_log_marginal_likelihood, initial_params,
                  args=(X_train, y_train), method='L-BFGS-B', bounds=bounds)

optimized_params = np.exp(result.x)
optimized_length_scale, optimized_sigma_f, optimized_noise_variance = optimized_params

print(f"Optimization successful: {result.success}")
print(f"Optimized Length Scale: {optimized_length_scale:.4f}")
print(f"Optimized Sigma_f: {optimized_sigma_f:.4f}")
print(f"Optimized Noise Variance: {optimized_noise_variance:.4e}")
print(f"Minimized NLL: {result.fun:.4f}")

# Re-run GP with optimized hyperparameters
optimized_kernel = lambda X1, X2: rbf_kernel(X1, X2, length_scale=optimized_length_scale, sigma_f=optimized_sigma_f)
gp_optimized = GaussianProcess(kernel_func=optimized_kernel, noise_variance=optimized_noise_variance)
gp_optimized.fit(X_train, y_train)
y_pred_opt_mean, y_pred_opt_std = gp_optimized.predict(X_test, return_std=True)

# gpobj.setpar([optimized_sigma_f, optimized_length_scale], optimized_noise_variance)
gpobj.bounds[-1][1]=4e-2
gpobj.mtune(20)
print(gpobj)

ypred = gpobj.m_predict(X_test)
spred = np.diag(gpobj.cov_predict(X_test, noise=True))

# Plotting with optimized hyperparameters
plt.figure(figsize=(12, 7))
plt.plot(X_test, true_function(X_test), 'r:', label='True function')
plt.plot(X_train, y_train, 'ro', markersize=8, label='Observations')
plt.plot(X_test, y_pred_opt_mean, 'g-', label='GP mean prediction (Optimized)')
plt.plot(X_test, ypred)
plt.fill_between(X_test.flatten(),
                 y_pred_opt_mean - 2 * y_pred_opt_std,
                 y_pred_opt_mean + 2 * y_pred_opt_std,
                 alpha=0.5, color='lightgreen', label='95% confidence interval (Optimized)')
plt.fill_between(X_test.flatten(),
                 ypred - 2 * spred**0.5,
                 ypred + 2 * spred**0.5,
                 alpha=0.2, color='m', label='95% confidence interval')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Process Regression (Optimized Hyperparameters)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()