import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal

# Enable double precision for numerical stability
torch.set_default_dtype(torch.float64)

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# ================== Synthetic Data Generation ==================
G = 6.674e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
zt_true = 6.0    # True depth (m)
xt_true = 3.0    # True horizontal position (m)
Rt_true = 2.0    # True radius (m)
den = 2500.0     # Density contrast (kg/m³)

num_data = 1000   # Number of observation points
x_range = (-20.0, 20.0)
xx = np.linspace(*x_range, num_data).reshape(-1, 1)
stdn = 1.0       # Noise standard deviation (µGal)

# Generate synthetic gravity data (now with Rt as a variable)
dg = (G * den * (4/3) * np.pi * Rt_true**3 * zt_true) / (zt_true**2 + (xt_true - xx)**2)**1.5 * 1e8
dgn = dg + np.random.normal(0, stdn, (num_data, 1))

# Convert to PyTorch tensors
xx_tensor = torch.from_numpy(xx).double()
y_tensor = torch.from_numpy(dgn).double()

# ================== Forward Model ==================
def forward_model(params, xx):
    """Compute forward gravity model for given parameters"""
    xi = params[0]  # xt
    zi = params[1]  # zt
    Ri = params[2]  # Rt (now variable)
    numerator = G * den * (4/3) * np.pi * Ri**3 * zi
    denominator = (zi**2 + (xi - xx.T)**2)**1.5
    return (numerator * 1e8 / denominator).squeeze()

# ================== Log Probability Functions ==================
def log_prior(params):
    """Log of prior probability - simple uniform priors with positivity constraints"""
    xt, zt, Rt = params
    # Uniform priors with reasonable bounds
    if (0 < xt < 10) and (1 < zt < 15) and (0.5 < Rt < 5):
        return 0.0  # Constant log prior within bounds
    return -np.inf  # Zero probability outside bounds

def calculate_residuals(params, xx, y_obs):
    """Calculate residuals and misfit in one forward model call"""
    y_pred = forward_model(params, xx)
    residuals = y_obs - y_pred
    rms_misfit = np.sqrt(np.mean(residuals**2))
    return residuals, rms_misfit

def log_likelihood(params, xx, y_obs, sigma_noise):
    """Log likelihood of data given parameters"""
    residuals, _ = calculate_residuals(params, xx, y_obs)
    return -0.5 * np.sum(residuals**2) / sigma_noise**2

def log_posterior(params, xx, y_obs, sigma_noise):
    """Log of posterior probability (unnormalized)"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xx, y_obs, sigma_noise)

# ================== MCMC Implementation ==================
def metropolis_hastings(xx, y_obs, sigma_noise, n_samples=10000, burn_in=2000, initial_params=None, proposal_scale=0.1):
    """
    Metropolis-Hastings MCMC sampler for gravity inversion problem
    
    Args:
        xx: Observation positions (tensor)
        y_obs: Observed gravity data (tensor)
        sigma_noise: Noise standard deviation
        n_samples: Total number of samples to generate
        burn_in: Number of burn-in samples to discard
        initial_params: Starting parameters [xt, zt, Rt]
        proposal_scale: Scale factor for Gaussian proposal distribution
        
    Returns:
        samples: Array of posterior samples [n_samples, 3]
        acceptance_rate: Overall acceptance rate
        misfit_history: Array of misfit values during sampling
    """
    # Convert tensors to numpy arrays
    xx_np = xx.numpy()
    y_obs_np = y_obs.numpy().flatten()
    
    # Initialize parameters
    if initial_params is None:
        params = np.array([5.0, 8.0, 1.5])  # Reasonable initial guess
    else:
        params = np.array(initial_params)
    
    # Store samples and misfit history
    samples = np.zeros((n_samples + burn_in, 3))
    misfit_history = np.zeros(n_samples + burn_in)
    accepted = 0
    
    # Current log posterior and misfit
    _, current_misfit = calculate_residuals(params, xx_np, y_obs_np)
    current_log_post = log_posterior(params, xx_np, y_obs_np, sigma_noise)
    misfit_history[0] = current_misfit
    
    for i in range(1, n_samples + burn_in):
        # Generate proposal (Gaussian random walk)
        proposal = params + proposal_scale * np.random.randn(3)
        
        # Ensure positive values for zt and Rt
        proposal[1] = np.abs(proposal[1])  # zt > 0
        proposal[2] = np.abs(proposal[2])  # Rt > 0
        
        # Calculate log posterior and misfit for proposal in one call
        residuals, proposal_misfit = calculate_residuals(proposal, xx_np, y_obs_np)
        proposal_log_post = log_prior(proposal) - 0.5 * np.sum(residuals**2) / sigma_noise**2
        
        # Acceptance probability
        log_alpha = proposal_log_post - current_log_post
        if np.log(np.random.rand()) < log_alpha:
            # Accept proposal
            params = proposal
            current_log_post = proposal_log_post
            current_misfit = proposal_misfit
            if i >= burn_in:
                accepted += 1
        
        # Store sample and misfit
        samples[i] = params
        misfit_history[i] = current_misfit
    
    # Discard burn-in samples
    samples = samples[burn_in:]
    #misfit_history = misfit_history[burn_in:]
    acceptance_rate = accepted / n_samples
    
    return samples, acceptance_rate, misfit_history

# ================== Run MCMC Sampling ==================
# Run MCMC
post_samples, acceptance_rate, misfit_history = metropolis_hastings(
    xx_tensor, y_tensor, stdn, 
    n_samples=100000, 
    burn_in=20000,
    initial_params=[5.5, 8.5, 4.5],  # Near true values for faster convergence [3.0,6.0,2.0]
    proposal_scale=0.01  # Adjusted for good acceptance rate
)

print("\nMCMC Sampling Complete")
print(f"Acceptance rate: {acceptance_rate:.2f}")

# ================== Misfit Convergence Plot ==================
plt.figure(figsize=(12, 10))

# Linear scale plot
plt.subplot(2, 1, 1)
plt.plot(misfit_history, alpha=0.7)
plt.axhline(stdn, color='r', linestyle='--', label='Expected noise level')
plt.xlabel('MCMC Sample')
plt.ylabel('RMS Misfit (µGal)')
plt.title('Misfit Convergence During MCMC Sampling (Linear Scale)')
plt.legend()
plt.grid(True)

# Log scale plot
plt.subplot(2, 1, 2)
plt.semilogy(misfit_history, alpha=0.7)
plt.axhline(stdn, color='r', linestyle='--', label='Expected noise level')
plt.xlabel('MCMC Sample')
plt.ylabel('RMS Misfit (µGal) - Log Scale')
plt.title('Misfit Convergence During MCMC Sampling (Log Scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mcmc_misfit_convergence_combined.png')
plt.show()

# ================== Posterior Analysis ==================
print("\nPosterior Statistics:")
print(f"Mean: {post_samples.mean(axis=0)}")
print(f"Std:  {post_samples.std(axis=0)}")
print(f"True: {[xt_true, zt_true, Rt_true]}")

# Visualization - now with 3 parameters
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
params = ['xt (m)', 'zt (m)', 'Rt (m)']
true_vals = [xt_true, zt_true, Rt_true]
for i in range(3):
    axs[i].hist(post_samples[:, i], bins=50, density=True, alpha=0.7)
    axs[i].axvline(true_vals[i], color='r', linestyle='--', label='True value')
    axs[i].set_title(params[i])
    axs[i].set_xlim(true_vals[i]-0.2, true_vals[i]+0.2)  # Wider range than flow for MCMC
    axs[i].legend()
    axs[i].set_xlabel('Parameter Value')
    axs[i].set_ylabel('Probability Density (%)')
    axs[i].grid(False)
plt.tight_layout()
plt.savefig('posterior_marginals_3params_mcmc.png')
plt.show()

# Trace plots for convergence diagnostics
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
param_names = ['xt', 'zt', 'Rt']
for i in range(3):
    axs[i].plot(post_samples[:, i], alpha=0.7)
    axs[i].axhline(true_vals[i], color='r', linestyle='--')
    axs[i].set_ylabel(param_names[i])
    axs[i].grid(True)
axs[-1].set_xlabel('MCMC Sample')
plt.suptitle('Trace Plots for MCMC Convergence Diagnostics')
plt.xlabel('MCMC Sample')
plt.tight_layout()
plt.savefig('mcmc_trace_plots.png')
plt.show()

# Autocorrelation plots
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
max_lag = 100
for i in range(3):
    acf = np.correlate(post_samples[:, i] - post_samples[:, i].mean(), 
                       post_samples[:, i] - post_samples[:, i].mean(), mode='full')
    acf = acf[acf.size//2:acf.size//2+max_lag+1]
    acf = acf / acf[0]
    axs[i].plot(acf, marker='o', markersize=4)
    axs[i].axhline(0, color='black', linestyle='-')
    axs[i].set_ylabel(param_names[i])
    axs[i].grid(True)
axs[-1].set_xlabel('Lag')
plt.suptitle('Autocorrelation Plots')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.savefig('mcmc_autocorrelation.png')
plt.show()

# Posterior predictive check
preds = np.zeros((100, num_data))  # Store 100 posterior predictive samples
for i in range(100):
    # Randomly select a posterior sample
    idx = np.random.randint(0, len(post_samples))
    params = post_samples[idx]
    preds[i] = forward_model(params, xx_tensor.numpy())

# Compute mean and credible intervals
mean_pred = preds.mean(axis=0)
std_pred = preds.std(axis=0)
lower = mean_pred - 2*std_pred
upper = mean_pred + 2*std_pred

plt.figure(figsize=(10, 6))
plt.plot(xx, dgn, 'k.', alpha=0.3, label='Noisy Data')
plt.plot(xx, dg, 'r-', lw=2, label='True Signal')
plt.plot(xx, mean_pred, 'b-', lw=2, label='Posterior Mean')
plt.fill_between(xx.flatten(), lower, upper, color='blue', alpha=0.2, label='95% Credible Interval')
plt.xlabel('Position (m)')
plt.ylabel('Gravity Anomaly (µGal)')
plt.title('Posterior Predictive Check (MCMC)')
plt.legend()
plt.grid(True)
plt.savefig('posterior_predictive_check_mcmc.png')
plt.show()