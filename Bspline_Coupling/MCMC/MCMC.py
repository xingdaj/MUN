# DRAM MCMC with SEM forward model 
# -------------------------------------------------------
# MH MCMC based inversion script with Torch-based simulator core
# - Uses real forward wave simulator for observations and for likelihood computation.
# - Implements Delayed Rejection Adaptive Metropolis (DRAM) MCMC for sampling from the posterior.
# - All wave-equation SEM computations in run_simulation() are in PyTorch.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
from scipy.interpolate import BSpline
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sem_waveform.core import SEMSimulation
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_velocity_on_sem_nodes


# -----------------------------
# Precision & constants
# -----------------------------
torch.set_default_dtype(torch.float64)
EPS = 1e-14
LIKELIHOOD_TEMP = 20.0  # keep original temperature scaling used in the old file

# -------------------------------------step 1: MCMC hyperparameters  ---------------------------------
pmean = 0.0              # initial center
pstd  = 50.0                # initial Gaussian std
num_chains = 1              # number of parallel chains
num_samples = 3000         # total samples per chain
num_burnin = 2000           # burn-in samples
num_thin = 1                # thinning interval
proposal_std = 10.0         # standard deviation for Gaussian proposal

num_samples = num_samples + 1 # fixed!
# DRAM specific parameters
dram_adapt_start = 1000     # when to start adaptation
dram_adapt_interval = 100   # how often to adapt
dram_scale = 2.38           # scaling factor for proposal covariance
dram_epsilon = 1e-6         # regularization for covariance matrix
dram_delayed_rejection_stages = 2  # number of DR stages
dram_dr_scales = [1.0, 0.25, 0.0625] # scaling factors for each DR stage

obs_noise_std = 0.5 * 1e-8  # realistic setting
time0 = time.perf_counter()


# -------------------------------------step 2: B-spline model  ---------------------------------
# ================== B-spline & True Control Points Setup (same as NSR81) ==================
ctrl_pts_original = np.array([
    [0, 0],
    [200, 0],
    [250, 50],
    [300, 100],
    [100, 300],
    [-100, 200]
]) - np.array([100, 100])  # center coordinates

tm   = 0.0
tstd = 50.0
np.random.seed(42)
true_offset = np.random.normal(loc=tm, scale=tstd, size=(6, 2))  # Gaussian offsets per control point
ctrl_pts_true = ctrl_pts_original + true_offset

k = 3  # cubic
ctrl_pts = np.vstack([ctrl_pts_true, np.tile(ctrl_pts_true[0], (k, 1))])
n = len(ctrl_pts) - 1
total_knots = n + k + 2
knots = np.zeros(total_knots)
knots[:k+1] = 0
knots[-k-1:] = 1
inner_knots = np.linspace(0, 1, n - k + 2)[1:-1]
knots[k+1:-k-1] = inner_knots

spline = BSpline(knots, ctrl_pts, k, extrapolate=False)
t_curve = np.linspace(knots[k], knots[-(k+1)], 500)
curve_points = spline(t_curve)

plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ro--', markersize=8, label='True Control Points')
ax.plot(curve_points[:, 0], curve_points[:, 1], 'r-', lw=2, label='B-spline Boundary')
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Cubic B-spline Velocity Model Structure (True Positions)')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.savefig('Bspline_model_structure_real.png', dpi=150)
plt.close()

time1 = time.perf_counter()
print(f"Step 1: B-spline setup time: {time1 - time0:.2f}s")

# ================== SEM configuration (copied from NSR81 for parity) ==================
sem_config = {
    'domain': {
        'xmin': -1000, 'xmax': 1000,
        'zmin': -1000, 'zmax': 1000,
        'nelem_x': 50, 'nelem_z': 50
    },
    'time': {
        'total_time': 0.4,
        'dt': 0.00008
    },
    'source': {
        'position': (0.0, 100.0),
        'frequency': 25.0,
        'amplitude': 1.0
    },
    'receivers': {
        'num_receivers': 40,
        'radius': 400.0
    },
    'method': {
        'polynomial_order': 5,
        'pml_thickness': 300.0
    },
    'velocity': {
        'inside_velocity': 2000.0,
        'outside_velocity': 3000.0,
        'control_points': ctrl_pts_original,
        'perturbations': true_offset,
        'spline_samples': 800,
        'tau': 10.0
    },
    'output': {
        'save_wavefield': False,
        'save_seismograms': True,
        'visualize': False,
        'output_dir': 'sem_output',
        'snapshot_interval': 10**9
    }
}

# Pre-create mesh & coords for the "velocity_model_true.png" figure (as NSR81 does)
global_coords, _, _ = create_global_mesh(
    xmin=sem_config['domain']['xmin'],
    xmax=sem_config['domain']['xmax'],
    zmin=sem_config['domain']['zmin'],
    zmax=sem_config['domain']['zmax'],
    nelem_x=sem_config['domain']['nelem_x'],
    nelem_z=sem_config['domain']['nelem_z'],
    npol=sem_config['method']['polynomial_order']
)

# Build velocity model using true parameters with SEM grid (just for plotting)
ctrl6 = ctrl_pts_true
velocity_model, signed_dist, extras = build_velocity_on_sem_nodes(
    nodes_xy   = global_coords,
    ctrl6_xy   = ctrl6,
    v_inside   = sem_config['velocity']['inside_velocity'],
    v_outside  = sem_config['velocity']['outside_velocity'],
    tau        = sem_config['velocity']['tau'],
    samples    = sem_config['velocity']['spline_samples'],
    newton_steps = 7
)

# Plot velocity model with true boundary (velocity_model_true.png)
plt.figure(figsize=(10, 8))
x_coords = global_coords[:, 0]
z_coords = global_coords[:, 1]
scatter = plt.scatter(x_coords, z_coords, c=velocity_model, cmap='seismic', s=1)
plt.colorbar(scatter, label='Velocity (m/s)')
plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', lw=2, label='Boundary')
source_pos = sem_config['source']['position']
plt.plot(source_pos[0], source_pos[1], 'k*', markersize=15, label='Source', markeredgecolor='red', markeredgewidth=1.0)
receiver_radius = sem_config['receivers']['radius']
num_receivers = sem_config['receivers']['num_receivers']
angles = np.linspace(0, 2*np.pi, num_receivers, endpoint=False)
receiver_x = receiver_radius * np.cos(angles) + source_pos[0]
receiver_z = receiver_radius * np.sin(angles) + source_pos[1]
plt.plot(receiver_x, receiver_z, 'b^', markersize=8, label='Receivers', markeredgecolor='black', markeredgewidth=0.5)
ctrl_pts_true_closed = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
plt.plot(ctrl_pts_true_closed[:, 0], ctrl_pts_true_closed[:, 1], 'ro--',
         markersize=8, label='B-spline Nodes', markeredgecolor='black', markeredgewidth=0.5)
xmin = sem_config['domain']['xmin']; xmax = sem_config['domain']['xmax']
zmin = sem_config['domain']['zmin']; zmax = sem_config['domain']['zmax']
plt.xlim(xmin, xmax); plt.ylim(zmin, zmax)
plt.title('Velocity Model with True Boundary')
plt.xlabel('X (m)'); plt.ylabel('Z (m)')
plt.legend()
plt.savefig('velocity_model_true.png', dpi=150)
plt.close()

# -------------------------------------step 3: SEM forward  ---------------------------------
# ================== Observed data generation (SEM forward) ==================
true_params = torch.from_numpy(true_offset.reshape(-1)).double()

def run_simulation(ctrl_params, noise_std, visualize=False):
    """
    SEM forward used for observation generation and for each MCMC likelihood call.
    Returns:
        noisy_data: (nt, num_receivers) torch.double
        clean_data: (nt, num_receivers) torch.double
    """
    if hasattr(ctrl_params, 'detach'):
        ctrl_params_np = ctrl_params.detach().cpu().numpy().reshape(-1, 2)
    else:
        ctrl_params_np = ctrl_params.reshape(-1, 2)

    # Update perturbations in config and run SEM
    sem_config['velocity']['perturbations'] = ctrl_params_np
    sim = SEMSimulation(sem_config)
    results = sim.run()  # expects: receiver_data (nt, nr), dt, nt

    clean_data = torch.from_numpy(results['receiver_data']).double()
    dt = results['dt']; nt = results['nt']

    if noise_std is None or float(noise_std) == 0.0:
        noisy = clean_data.clone()
    else:
        noisy = clean_data + torch.normal(mean=torch.zeros_like(clean_data), std=noise_std)

    if visualize:
        time_axis = np.arange(nt) * dt
        plt.figure(figsize=(15, 4))
        plt.plot(time_axis, clean_data[:, 0].numpy(), label='clean')
        if noise_std is not None and float(noise_std) > 0:
            plt.plot(time_axis, noisy[:, 0].numpy(), label='noisy')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.title('Example receiver waveform (R1)')
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig('example_sem_waveform.png', dpi=160)
        plt.close()

    return noisy, clean_data

# Generate observed data with SEM
print("Generating observed data with SEM (this may take some seconds)...")
noisy_obs, clean_obs = run_simulation(true_params, noise_std=obs_noise_std, visualize=False)
np.save("noisy_data_real.npy", noisy_obs.numpy())
np.save("clean_data_real.npy", clean_obs.numpy())
print("Observations saved to disk.")
time2 = time.perf_counter()
print(f"Step 2: SEM sim setup time: {time2 - time1:.2f}s")

# Quick diagnostic plot of observations
obs_dt = sem_config['time']['dt']
obs_nt = noisy_obs.shape[0]
time_axis = np.arange(obs_nt) * obs_dt
plt.figure(figsize=(15, 4 * sem_config['receivers']['num_receivers']))
for i in range(sem_config['receivers']['num_receivers']):
    plt.subplot(sem_config['receivers']['num_receivers'], 1, i+1)
    plt.plot(time_axis, clean_obs[:, i].numpy(), 'k-', lw=1.5, label='Clean')
    plt.plot(time_axis, noisy_obs[:, i].numpy(), 'r-', lw=1.0, alpha=0.7, label='Noisy')
    #plt.ylim(-1.5*1e-7, 1.5*1e-7)
    plt.title(f'Receiver R{i+1}')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    if i == 0: plt.legend()
plt.tight_layout()
plt.savefig('real_noise_comparison.png', dpi=200)
plt.close()

# -------------------------------------step 4: MCMC ---------------------------------
# ================== DRAM MCMC Sampler ==================
class DRAM_MCMC_Sampler:
    """Delayed Rejection Adaptive Metropolis MCMC sampler."""
    def __init__(self, log_posterior, initial_proposal_std=10.0, 
                 adapt_start=100, adapt_interval=50, scale=2.38, epsilon=1e-6,
                 dr_stages=2, dr_scales=None):
        self.log_posterior = log_posterior
        self.initial_proposal_std = initial_proposal_std
        self.adapt_start = adapt_start
        self.adapt_interval = adapt_interval
        self.scale = scale
        self.epsilon = epsilon
        self.dr_stages = dr_stages
        self.dr_scales = dr_scales if dr_scales is not None else [1.0] * dr_stages
        self.covariance = None
        self.sample_mean = None
        self.sample_cov = None
        self.n_samples = 0
        
    def adapt_proposal(self, samples):
        if len(samples) < 2:
            return
        samples_array = np.array(samples)
        self.sample_mean = np.mean(samples_array, axis=0)
        self.sample_cov = np.cov(samples_array.T)
        dim = len(self.sample_mean)
        regularized_cov = self.sample_cov + self.epsilon * np.eye(dim)
        self.covariance = (self.scale ** 2 / dim) * regularized_cov
        
    def propose(self, current_params, stage=0, param_idx=None):
        current_np = current_params.numpy().copy()
        proposed_np = current_np.copy()
        scale_factor = self.dr_scales[stage] if stage < len(self.dr_scales) else 1.0
        if self.covariance is not None and self.n_samples >= self.adapt_start:
            proposal_std = np.sqrt(self.covariance[param_idx, param_idx]) * scale_factor
            proposed_np[param_idx] = current_np[param_idx] + np.random.normal(0, proposal_std)
        else:
            proposed_np[param_idx] = current_np[param_idx] + np.random.normal(0, self.initial_proposal_std * scale_factor)
        return torch.from_numpy(proposed_np).double()
    
    def log_proposal_density(self, from_params, to_params, stage=0, param_idx=None):
        from_np = from_params.numpy(); to_np = to_params.numpy()
        scale_factor = self.dr_scales[stage] if stage < len(self.dr_scales) else 1.0
        if self.covariance is not None and self.n_samples >= self.adapt_start:
            proposal_std = np.sqrt(self.covariance[param_idx, param_idx]) * scale_factor
            dist = Normal(torch.tensor(from_np[param_idx]), proposal_std)
        else:
            std = self.initial_proposal_std * scale_factor
            dist = Normal(torch.tensor(from_np[param_idx]), std)
        return dist.log_prob(torch.tensor(to_np[param_idx]))
    
    def delayed_rejection(self, current_params, current_log_prob, stage, samples_history, param_idx):
        if stage >= self.dr_stages:
            return current_params, current_log_prob, False
        proposed_params = self.propose(current_params, stage, param_idx)
        proposed_log_prob = self.log_posterior(proposed_params)
        log_alpha1 = proposed_log_prob - current_log_prob
        log_alpha2 = 0.0
        for prev_stage in range(stage):
            prev_proposal = samples_history[prev_stage]['proposed']
            prev_lp = self.log_proposal_density(samples_history[prev_stage]['current'], prev_proposal, prev_stage, param_idx)
            cur_lp  = self.log_proposal_density(samples_history[prev_stage]['current'], proposed_params, stage, param_idx)
            log_alpha2 += torch.log(1 - torch.exp(samples_history[prev_stage]['log_alpha'])) \
                        - torch.log(1 - torch.exp(proposed_log_prob - samples_history[prev_stage]['current_log_prob']))
            log_alpha2 += (cur_lp - prev_lp)
        log_accept_ratio = log_alpha1 + log_alpha2
        if torch.log(torch.rand(1)) < log_accept_ratio:
            return proposed_params, proposed_log_prob, True
        else:
            samples_history.append({
                'current': current_params,
                'current_log_prob': current_log_prob,
                'proposed': proposed_params,
                'log_alpha': log_alpha1
            })
            return self.delayed_rejection(current_params, current_log_prob, stage + 1, samples_history, param_idx)
    
    def sample(self, initial_params, num_samples, num_burnin=0, num_chains=1):
        chains = []
        log_posterior_chains = []
        acceptance_rates = []
        stage_acceptance_rates = [[] for _ in range(self.dr_stages + 1)]
        starts = []
        for chain_idx in range(num_chains):
            print(f"Running chain {chain_idx+1}/{num_chains}")
            current_params = initial_params.clone() + torch.randn_like(initial_params) * 0.1
            starts.append(current_params.clone())
            current_log_prob = self.log_posterior(current_params)
            samples = []
            log_post_samples = []
            all_samples_history = []
            accept_count = 0
            total_iterations = 0
            stage_counts = [0] * (self.dr_stages + 1)
            stage_accepts = [0] * (self.dr_stages + 1)
            chain_start_time = time.perf_counter()
            last_print_time = chain_start_time
            for i in range(num_samples + num_burnin):
                all_samples_history.append(current_params.numpy())
                if (i + 1) % self.adapt_interval == 0 and len(all_samples_history) >= self.adapt_start:
                    self.adapt_proposal(all_samples_history)
                    self.n_samples = len(all_samples_history)
                param_idx = np.random.randint(0, len(current_params))
                total_iterations += 1
                proposed_params = self.propose(current_params, 0, param_idx)
                proposed_log_prob = self.log_posterior(proposed_params)
                log_accept_ratio = proposed_log_prob - current_log_prob
                stage_counts[0] += 1
                if torch.log(torch.rand(1)) < log_accept_ratio:
                    current_params = proposed_params
                    current_log_prob = proposed_log_prob
                    accept_count += 1
                    stage_accepts[0] += 1
                else:
                    samples_history = [{
                        'current': current_params,
                        'current_log_prob': current_log_prob,
                        'proposed': proposed_params,
                        'log_alpha': log_accept_ratio
                    }]
                    result_params, result_log_prob, accepted = self.delayed_rejection(current_params, current_log_prob, 1, samples_history, param_idx)
                    if accepted:
                        current_params = result_params
                        current_log_prob = result_log_prob
                        accept_count += 1
                        accepted_stage = len(samples_history)
                        stage_accepts[accepted_stage] += 1
                    for stage_idx in range(1, len(samples_history) + 1):
                        stage_counts[stage_idx] += 1
                samples.append(current_params.clone())
                log_post_samples.append(current_log_prob.item())
                if (i + 1) % 1 == 0:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - chain_start_time
                    iteration_time = current_time - last_print_time
                    last_print_time = current_time
                    acceptance_rate = accept_count / total_iterations
                    stage_rates = []
                    for j in range(len(stage_counts)):
                        if stage_counts[j] > 0:
                            stage_rates.append(stage_accepts[j] / stage_counts[j])
                        else:
                            stage_rates.append(0.0)
                    iterations_done = i + 1
                    iterations_total = num_samples + num_burnin
                    time_per_iteration = elapsed_time / iterations_done
                    time_remaining = time_per_iteration * (iterations_total - iterations_done)
                    print(f"Chain {chain_idx+1}, Iteration {i+1}/{num_samples+num_burnin}, "
                          f"Acceptance rate: {acceptance_rate:.3f}, "
                          f"Stage rates: {[f'{r:.3f}' for r in stage_rates]}, "
                          f"Elapsed: {elapsed_time:.1f}s, "
                          f"ETA: {time_remaining:.1f}s, "
                          f"Iteration time: {iteration_time:.4f}s")
            chains.append(torch.stack(samples))
            log_posterior_chains.append(log_post_samples)
            acceptance_rates.append(accept_count / total_iterations)
            for j in range(len(stage_counts)):
                if stage_counts[j] > 0:
                    stage_acceptance_rates[j].append(stage_accepts[j] / stage_counts[j])
                else:
                    stage_acceptance_rates[j].append(0.0)
        return chains, log_posterior_chains, acceptance_rates, stage_acceptance_rates, starts

# ================== Log Posterior (prior + SEM likelihood) ==================
def log_posterior(params):
    prior_mean = torch.zeros_like(params)
    prior_std = torch.ones_like(params) * tstd
    log_prior = torch.sum(Normal(prior_mean, prior_std).log_prob(params))
    simulated_data, _ = run_simulation(params, noise_std=None)
    eff_sigma = obs_noise_std * LIKELIHOOD_TEMP
    log_likelihood = torch.sum(Normal(simulated_data, eff_sigma).log_prob(noisy_obs))
    return log_prior + log_likelihood

# -------------------------------------step 5: Iteration starting  ---------------------------------
# ================== Run DRAM MCMC Sampling ==================
print("Starting DRAM MCMC sampling...")
time3 = time.perf_counter()

dram_sampler = DRAM_MCMC_Sampler(
    log_posterior,
    initial_proposal_std=proposal_std,
    adapt_start=dram_adapt_start,
    adapt_interval=dram_adapt_interval,
    scale=dram_scale,
    epsilon=dram_epsilon,
    dr_stages=dram_delayed_rejection_stages,
    dr_scales=dram_dr_scales
)

initial_params = true_params + (torch.ones(12, dtype=torch.float64) - 0.5) * 2 * pmean +  torch.ones(12, dtype=torch.float64) * pstd

chains, log_posterior_chains, acceptance_rates, stage_acceptance_rates, starts = dram_sampler.sample(
    initial_params,
    num_samples=num_samples,
    num_burnin=num_burnin,
    num_chains=num_chains
)

chains_np = [chain.numpy() for chain in chains]
all_log_posterior = np.concatenate(log_posterior_chains)

time4 = time.perf_counter()
print(f"DRAM MCMC sampling time: {time4 - time3:.2f}s")
print(f"Overall acceptance rates: {acceptance_rates}")
for i, stage_rates in enumerate(stage_acceptance_rates):
    print(f"Stage {i} acceptance rates: {stage_rates}")
print(f"Total elapsed time: {time4 - time0:.2f}s")

# -------------------------------------step 6: Posterior distribution ---------------------------------
# ================== Analysis and Visualization ==================
all_samples = np.vstack([chain[num_burnin:] for chain in chains_np])
posterior_mean = np.mean(all_samples, axis=0)
posterior_std  = np.std(all_samples, axis=0)

print("Posterior means:", posterior_mean)
print("Posterior stds:", posterior_std)
print("True values:", true_params.numpy())

initial_params_used = starts[0].detach().cpu().double()

# Trace plots
plt.figure(figsize=(15, 12))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    for chain_idx, chain in enumerate(chains_np):
        plt.plot(range(len(chain)), chain[:, i], alpha=1.0, label=f'Chain {chain_idx+1}' if i == 0 else "")
    plt.axhline(y=true_params[i].item(), color='r', linestyle='--', label='True value' if i == 0 else "")
    plt.axvline(x=num_burnin, color='orange', linestyle='--', linewidth=2, label='Burn-in end' if i == 0 else "")
    plt.ylabel(f'Param {i+1}'); plt.xlabel('Iteration')
    if i == 0: plt.legend()
plt.tight_layout()
plt.savefig('dram_mcmc_trace_plots.png', dpi=150)
plt.close()

# Posterior histograms
plt.figure(figsize=(15, 12))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.hist(all_samples[:, i], bins=30, density=True, alpha=0.7, label='Posterior')
    plt.axvline(true_params[i].item(), color='r', linestyle='--', linewidth=2, label='True value')
    plt.axvline(initial_params_used[i].item(), color='g', linestyle='--', linewidth=2, label='Initial value')
    plt.axvline(posterior_mean[i], color='b', linestyle='-', linewidth=2, label='Posterior mean')
    plt.xlabel(f'Param {i+1}'); plt.ylabel('Density'); plt.legend()
plt.tight_layout()
plt.savefig('dram_posterior_distributions.png', dpi=150)
plt.close()

# === Posterior velocity models figure ===
# Build B-spline curves for true, initial, mean models
ctrl_pts_true_m = ctrl_pts_original + true_offset
initial_offset  = initial_params_used.reshape(-1, 2).numpy()
ctrl_pts_init_m = ctrl_pts_original + initial_offset
mean_offset     = posterior_mean.reshape(-1, 2)
ctrl_pts_mean_m = ctrl_pts_original + mean_offset

fig, axes = plt.subplots(2, 2, figsize=(12, 10)); axes = axes.ravel()

# True
ctrl_closed_true = np.vstack([ctrl_pts_true_m, ctrl_pts_true_m[0]])
axes[0].plot(ctrl_closed_true[:, 0], ctrl_closed_true[:, 1], 'ro--', markersize=8, label='Control Points')
spline_true = BSpline(knots, np.vstack([ctrl_pts_true_m, np.tile(ctrl_pts_true_m[0], (k, 1))]), k)
curve_points_true = spline_true(t_curve)
axes[0].plot(curve_points_true[:, 0], curve_points_true[:, 1], 'r-', lw=2, label='B-spline Boundary')
axes[0].set_title('True Model'); axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Z (m)')
axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3); axes[0].legend()

# Initial
ctrl_closed_init = np.vstack([ctrl_pts_init_m, ctrl_pts_init_m[0]])
axes[1].plot(ctrl_closed_init[:, 0], ctrl_closed_init[:, 1], 'go--', markersize=8, label='Control Points')
spline_init = BSpline(knots, np.vstack([ctrl_pts_init_m, np.tile(ctrl_pts_init_m[0], (k, 1))]), k)
curve_points_init = spline_init(t_curve)
axes[1].plot(curve_points_init[:, 0], curve_points_init[:, 1], 'g-', lw=2, label='B-spline Boundary')
axes[1].set_title('Initial Model'); axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Z (m)')
axes[1].set_aspect('equal'); axes[1].grid(True, alpha=0.3); axes[1].legend()

# Mean posterior
ctrl_closed_mean = np.vstack([ctrl_pts_mean_m, ctrl_pts_mean_m[0]])
axes[2].plot(ctrl_closed_mean[:, 0], ctrl_closed_mean[:, 1], 'bo--', markersize=8, label='Control Points')
spline_mean = BSpline(knots, np.vstack([ctrl_pts_mean_m, np.tile(ctrl_pts_mean_m[0], (k, 1))]), k)
curve_points_mean = spline_mean(t_curve)
axes[2].plot(curve_points_mean[:, 0], curve_points_mean[:, 1], 'b-', lw=2, label='B-spline Boundary')
axes[2].set_title('Mean Posterior Model'); axes[2].set_xlabel('X (m)'); axes[2].set_ylabel('Z (m)')
axes[2].set_aspect('equal'); axes[2].grid(True, alpha=0.3); axes[2].legend()

# Comparison
axes[3].plot(curve_points_true[:, 0], curve_points_true[:, 1], 'r-', lw=3, label='True boundary')
axes[3].plot(ctrl_closed_true[:, 0], ctrl_closed_true[:, 1], 'ro--', markersize=6, alpha=0.7)
axes[3].plot(curve_points_init[:, 0], curve_points_init[:, 1], 'g--', lw=2, label='Initial boundary')
axes[3].plot(ctrl_closed_init[:, 0], ctrl_closed_init[:, 1], 'go--', markersize=6, alpha=0.7)
axes[3].plot(curve_points_mean[:, 0], curve_points_mean[:, 1], 'b-', lw=2, label='Mean posterior boundary')
axes[3].plot(ctrl_closed_mean[:, 0], ctrl_closed_mean[:, 1], 'bo--', markersize=6, alpha=0.7)
axes[3].set_title('Boundary Comparison'); axes[3].set_xlabel('X (m)'); axes[3].set_ylabel('Z (m)')
axes[3].set_aspect('equal'); axes[3].grid(True, alpha=0.3); axes[3].legend()
plt.tight_layout(); plt.savefig('dram_posterior_velocity_models.png', dpi=150); plt.close()

# === MCMC misfit curve (RMS) ===
print("Calculating MCMC misfit curve...")
misfit_values = []; rms_values = []
all_samples_with_burnin = np.vstack([chain.numpy() for chain in chains])
for i, sample in enumerate(all_samples_with_burnin):
    if i % 10 == 0:
        sample_params = torch.from_numpy(sample)
        simulated_data, _ = run_simulation(sample_params, noise_std=None)
        squared_diff = (simulated_data - noisy_obs) ** 2
        mse = torch.mean(squared_diff).item()
        rms = np.sqrt(mse)
        misfit_values.append((i, mse))
        rms_values.append((i, rms))

misfit_iterations, misfits = zip(*misfit_values)
rms_iterations, rms_errors = zip(*rms_values)
burnin_end_index = num_burnin
post_burnin_rms = [rms for i, rms in zip(rms_iterations, rms_errors) if i >= num_burnin]
final_rms = rms_errors[-1]
mean_rms = np.mean(post_burnin_rms) if post_burnin_rms else final_rms
std_rms  = np.std(post_burnin_rms) if post_burnin_rms else 0.0
std_error_rms = std_rms / np.sqrt(len(post_burnin_rms)) if post_burnin_rms else 0.0

print(f"Final RMS error: {final_rms:.6e}")
print(f"Mean RMS error (post-burnin): {mean_rms:.6e} ± {std_error_rms:.6e} (standard error)")
print(f"RMS error std (post-burnin): {std_rms:.6e}")

plt.figure(figsize=(12, 6))
plt.plot(rms_iterations, rms_errors, 'b-', alpha=0.7, label='RMS Error')
plt.axhline(y=mean_rms, color='b', linestyle='--', alpha=0.8, label=f'Mean RMS (post-burnin) : {mean_rms:.3e}')
plt.axhline(y=obs_noise_std, color='r', linestyle='--', alpha=0.8, label=f'Noise STD : {obs_noise_std:.3e}')
plt.fill_between(rms_iterations, 
                 np.array(rms_errors) - std_error_rms, 
                 np.array(rms_errors) + std_error_rms, 
                 color='gray', alpha=0.3, label=f'±1 SE: {std_error_rms:.3e}')
plt.axvline(x=burnin_end_index, color='orange', linestyle='--', linewidth=2, label='Burn-in end')
plt.xlabel('Iteration'); plt.ylabel('RMS Error'); plt.yscale('log')
plt.title(f'MCMC RMS Error Curve (Complete Process)\\nFinal RMS: {final_rms:.3e}, Mean (post-burnin): {mean_rms:.3e} ± {std_error_rms:.3e}')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig('dram_mcmc_misfit_curve.png', dpi=150); plt.close()

# === Waveform comparison: mean / initial / true ===
print("Plotting mean posterior waveform comparison...")
mean_params = torch.from_numpy(posterior_mean).double()
mean_simulated, _ = run_simulation(mean_params, noise_std=None)
initial_simulated, _ = run_simulation(initial_params_used, noise_std=None)
true_simulated, _ = run_simulation(true_params, noise_std=None)

plt.figure(figsize=(15, 4 * sem_config['receivers']['num_receivers']))
for j in range(sem_config['receivers']['num_receivers']):
    plt.subplot(sem_config['receivers']['num_receivers'], 1, j + 1)
    plt.plot(time_axis, clean_obs[:, j].numpy(), 'k-', lw=1.5, label='True (clean)')
    plt.plot(time_axis, noisy_obs[:, j].numpy(), 'r-', lw=1.0, alpha=0.7, label='Observed (noisy)')
    plt.plot(time_axis, initial_simulated[:, j].numpy(), 'g--', lw=1.0, label='Initial model')
    plt.plot(time_axis, mean_simulated[:, j].numpy(), 'b--', lw=1.0, label='Mean posterior')
    plt.title(f'Receiver R{j+1} - Waveform Comparison')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3); 
    if j == 0: plt.legend()
plt.tight_layout(); plt.savefig('dram_mean_posterior_waveform_comparison.png', dpi=200); plt.close()
print("Waveform comparison plots saved.")

# Save bundle
np.savez('dram_mcmc_results.npz', 
         chains=chains_np, 
         true_params=true_params.numpy(),
         initial_params=initial_params_used.numpy(),
         posterior_mean=posterior_mean,
         posterior_std=posterior_std,
         acceptance_rates=acceptance_rates,
         stage_acceptance_rates=stage_acceptance_rates,
         rms_curve=np.array(rms_values),
         mse_curve=np.array(misfit_values))
print("DRAM MCMC analysis complete. Results saved to dram_mcmc_results.npz")
