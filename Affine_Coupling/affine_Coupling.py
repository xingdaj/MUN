# - Final Normalizing Flow script with SEM-based simulator core.
# - Uses SEM wave simulator for observations and for ELBO computation.
# - Flow uses ActNorm + AffineCoupling for numerical stability and invertibility.
# - All wave-equation computations now use the Spectral Element Method.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from scipy.interpolate import BSpline
from matplotlib.patches import Patch

# Import SEM modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sem_waveform.core import SEMSimulation  # need to implement run_forward_and_adjoint in core
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_velocity_on_sem_nodes
print("Using SEMSimulation from:", SEMSimulation.__module__, SEMSimulation.__qualname__)


# ================================ Preparation =====================================
_SIM_CACHE = {}
def _sim_key_from_cfg(cfg):
    d = cfg['domain']
    m = cfg['method']
    r = cfg['receivers']
    # Extract pml_thickness directly from sem_config
    pml_thickness = m['pml_thickness']

    return (
        d['xmin'], d['xmax'], d['zmin'], d['zmax'],
        d['nelem_x'], d['nelem_z'],
        m['polynomial_order'], pml_thickness,
        r['num_receivers'], r['radius'],
        cfg['time']['total_time'], cfg['time'].get('dt', None),
    )

def get_or_make_sim(cfg):
    key = _sim_key_from_cfg(cfg)
    sim = _SIM_CACHE.get(key)
    if sim is None:
        sim = SEMSimulation(cfg)
        _SIM_CACHE[key] = sim
    return sim

# ================================ step 0: Optimization parameters =====================================
# Use double precision for improved numerical stability
torch.set_default_dtype(torch.float64)
EPS64 = 1e-12
EPS32 = 1e-8
EPS = EPS64 
DEBUG_BATCH = 64

# Training hyperparameters (start small for debugging; increase for final runs)
num_epochs = 200        # increase to e.g. 100+ for real training
min_elbo_samples = 4    # must be >= 2; number of samples for ELBO early; increase later
max_elbo_samples = 8    # the maximum number of samples
learning_rate = 5e-4
num_flows = 16
#K_bins = 8  # not used in affine coupling, kept for compatibility
obs_noise_std = 0.5 * 1e-8  # same as earlier realistic setting

# Gradient clipping configuration
max_grad_norm = 50.0
clip_gradient = True
monitor_gradient = True

np.random.seed(42)
# ================================ step 1: B-spline & True Control Points Setup =====================================
time0 = time.perf_counter()
# These are used only by the real simulator to build geometry from 12 parameters.
ctrl_pts_original = np.array([
    [-100, -100],
    [100, -100],
    [150, -50],
    [200, 0],
    [0, 200],
    [-200, 100]
])  # center coordinates

tm   = 0.0
tstd = 50.0
true_offset = np.random.normal(loc=tm, scale=tstd, size=(6, 2))  # Gaussian offsets per control point
ctrl_pts_true = ctrl_pts_original + true_offset

k = 3  # cubic B-spline degree
ctrl_pts = np.vstack([ctrl_pts_true, np.tile(ctrl_pts_true[0], (k, 1))])
n = len(ctrl_pts) - 1
total_knots = n + k + 2
knots = np.zeros(total_knots)
knots[:k+1] = 0
knots[-k-1:] = 1
inner_knots = np.linspace(0, 1, n - k + 2)[1:-1]
knots[k+1:-k-1] = inner_knots

spline = BSpline(knots, ctrl_pts, k, extrapolate=False)
t_curve = np.linspace(knots[k], knots[-(k+1)], 800)
curve_points = spline(t_curve)

# 6 closed B-spline spline
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

# ========================================= step 2: Simulation grid & receivers ======================================
# SEM configuration
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
        'tau': 10.0,
        'spline_samples': 800  # Global sampling count
   
    },
    'output': {
        'save_wavefield': False,
        'save_seismograms': True,
        'visualize': False,
        'output_dir': 'sem_output',
        'snapshot_interval': 10**9   
    }
}

# Create SEM grid from configuration
global_coords, global_connectivity, node_map = create_global_mesh(
    xmin=sem_config['domain']['xmin'],
    xmax=sem_config['domain']['xmax'],
    zmin=sem_config['domain']['zmin'],
    zmax=sem_config['domain']['zmax'],
    nelem_x=sem_config['domain']['nelem_x'],
    nelem_z=sem_config['domain']['nelem_z'],
    npol=sem_config['method']['polynomial_order']
)

# Build velocity model using true parameters with SEM grid
# Assume you have:
# sem.global_coords  -> (N,2)  SEM node coordinates
# ctrl6              -> (6,2) B-spline control points（or sem_config['velocity']['control_points']）
# v_in, v_out, tau   -> inside/outside velocities and transition width

# Build velocity model using true parameters with SEM grid
# ctrl6_xy requires absolute control point coordinates (6×2)
ctrl6 = ctrl_pts_true  # = ctrl_pts_original + true_offset, shape (6,2)
velocity_model, signed_dist, extras = build_velocity_on_sem_nodes(
    nodes_xy   = global_coords,                                  # (N,2)
    ctrl6_xy   = ctrl6,                                          # (6,2)
    v_inside   = sem_config['velocity']['inside_velocity'],
    v_outside  = sem_config['velocity']['outside_velocity'],
    tau        = sem_config['velocity']['tau'],                   
    samples    = sem_config['velocity']['spline_samples'],        
    newton_steps = 7
)

###### Plot the velocity model
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

# True parameter vector (12 dims = 6 control points × 2 coords)
true_params = torch.from_numpy(true_offset.reshape(-1)).double()  # Flattened 12-dim Gaussian offsets

# ====================================== step 3: SEM forward simulation (for obs) ====================================
def run_simulation(ctrl_params, noise_std, visualize=False):
    """
    Only used for generating observations and visualization (no backpropagation).
    """
    if hasattr(ctrl_params, 'detach'):
        ctrl_params_np = ctrl_params.detach().cpu().numpy().reshape(-1, 2)
    else:
        ctrl_params_np = ctrl_params.reshape(-1, 2)

    abs_ctrl = ctrl_pts_original + ctrl_params_np
    sem_config['velocity']['perturbations'] = ctrl_params_np
    sim = SEMSimulation(sem_config)
    results = sim.run()

    clean_data = torch.from_numpy(results['receiver_data']).double()
    dt = results['dt']; nt = results['nt']

    if noise_std is None or float(noise_std) == 0.0:
        noisy_data = clean_data.clone()
    else:
        noisy_data = clean_data + torch.normal(mean=torch.zeros_like(clean_data), std=noise_std)

    return noisy_data, clean_data, dt, nt


print("Generating observed data with SEM simulator (this may take some seconds)...")
noisy_obs, clean_obs, actual_dt, actual_nt = run_simulation(true_params, noise_std=obs_noise_std, visualize=False)
# Save observation time-grid parameters
obs_dt = actual_dt
obs_nt = actual_nt
np.save("noisy_data_real.npy", noisy_obs.numpy())
np.save("clean_data_real.npy", clean_obs.numpy())
print(f"SEM observations saved to disk. dt={actual_dt:.6f}, nt={actual_nt}")

time2 = time.perf_counter()
print(f"Step 2: SEM sim setup time: {time2 - time1:.2f}s")

time_axis = np.arange(actual_nt) * actual_dt
plt.figure(figsize=(10, 4 * sem_config['receivers']['num_receivers']))
for i in range(sem_config['receivers']['num_receivers']):
    plt.subplot(sem_config['receivers']['num_receivers'], 1, i+1)
    plt.plot(time_axis, clean_obs[:, i].numpy(), 'b-', lw=2, label='True')
    plt.plot(time_axis, noisy_obs[:, i].numpy(), 'r-', lw=1.0, alpha=0.7, label='Noisy')
    plt.ylim(-1.5*1e-7, 1.5*1e-7)
    plt.xlim(0.0, actual_nt * actual_dt)
    plt.title(f'Receiver R{i+1}')
    plt.xlabel('Time(s)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3); plt.legend()
plt.savefig('real_noise_comparison.png', dpi=200)
plt.close()

time3 = time.perf_counter()
print(f"Step 3: generated observations. Time: {time3 - time2:.2f}s")

# ================== step 4: Custom Autograd Function to connect SEM adjoint gradient to Normalizing Flow ==================
class SEMLikelihoodAdjointFn(torch.autograd.Function):
    """
    Forward: call SEM forward + adjoint to return log-likelihood (scalar).
    Backward: inject SEM adjoint d logLik / d z into z's gradient (propagates into NF parameters).
    Requires sem_waveform.core.SEMSimulation to implement run_forward_and_adjoint(perturbations, y_obs, noise_var).
    """
    @staticmethod
    def forward(ctx, z, y_obs, noise_std, sem_cfg_dict, obs_dt, obs_nt):
        # z: (12,), y_obs: (nt, nrec)
        if z.dim() != 1:
            raise ValueError("SEMLikelihoodAdjointFn expects z shape (12,), got {}".format(z.shape))

        z_off = z.detach().cpu().numpy().reshape(-1, 2)
        abs_ctrl = ctrl_pts_original + z_off  # <<<<<<<<<<<<<< key line
        cfg = dict(sem_cfg_dict); cfg['velocity'] = dict(cfg['velocity'])
        #cfg['velocity']['perturbations'] = abs_ctrl
        #cfg['velocity']['perturbations'] = z_off

        # Prepare to call SEM adjoint
        #z_np = z.detach().cpu().numpy().reshape(-1, 2)
        #cfg = dict(sem_cfg_dict)  # shallow copy
        #cfg['velocity'] = dict(cfg['velocity'])
        #cfg['velocity']['perturbations'] = z_np

        #sim = SEMSimulation(cfg)
        sim = get_or_make_sim(cfg)

        if not hasattr(sim, "run_forward_and_adjoint"):
            raise NotImplementedError(
                "SEMSimulation.run_forward_and_adjoint(...) is not implemented. "
                "Please implement this method in sem_waveform/core.py so that it returns: "
                "{'loglik': float, 'grad_wrt_ctrl': (12,), 'clean_data': (nt,nrec), 'dt': float, 'nt': int}"
            )

        # NOTE: y_obs must be numpy (core typically uses numpy)
        y_obs_np = y_obs.detach().cpu().numpy()
        # allow float or torch.Tensor
        if isinstance(noise_std, torch.Tensor):
            noise_var = float(noise_std.detach().cpu().item() ** 2 + EPS)
        else:
            noise_var = float(noise_std) ** 2 + EPS

        out = sim.run_forward_and_adjoint({
            'bspline_ctrl': abs_ctrl,  # abs_ctrl
            'y_obs': y_obs_np,
            'obs_dt': float(obs_dt),
            'obs_nt': int(obs_nt),
            'noise_std': np.sqrt(noise_var)
        })

        # Extract loglik and gradient wrt 12-dim parameters
        loglik = float(out['loglik'])
        grad_wrt_ctrl = np.asarray(out['grad_wrt_ctrl'], dtype=np.float64).reshape(-1)

        # Save to ctx for backward use
        ctx.save_for_backward(torch.from_numpy(grad_wrt_ctrl).to(z.device))
        # Return torch scalar
        return torch.tensor(loglik, dtype=torch.float64, device=z.device)

    @staticmethod
    def backward(ctx, grad_output):
        (grad_wrt_ctrl_tensor,) = ctx.saved_tensors  # shape: (12,)
        # Multiply upstream gradient with provided d logLik / d z
        grad_z = grad_output * grad_wrt_ctrl_tensor
        # No gradient needed for y_obs, noise_std, sem_cfg_dict
        return grad_z, None, None, None, None, None


def sem_loglik_with_adjoint(z, y_obs, noise_std, obs_dt, obs_nt):
    """
    Wrapper that automatically handles time-axis alignment.
    """
    # Ensure y_obs is generated under the current configuration
    result = SEMLikelihoodAdjointFn.apply(z, y_obs, noise_std, sem_config, obs_dt, obs_nt)
    return result

# ================== step 5: ActNorm / AffineCoupling / NormalizingFlow  ==================
class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.logs = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.initialized = False

    def initialize_parameters(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=0)
            # Use unbiased=False to avoid div-by-zero when batch=1
            if x.shape[0] == 1:
                # When batch=1, std should be 0; avoid NaN
                std = torch.zeros_like(mean)
            else:
                std = torch.std(x, dim=0, unbiased=False)  # key fix

            std = torch.where(std < 1e-6, torch.ones_like(std) * 1e-6, std)
            self.bias.data = (-mean).clone()
            self.logs.data = torch.log(1.0 / (std.clone() + 1e-12)) 
            self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
        y = (x + self.bias.unsqueeze(0)) * torch.exp(self.logs.unsqueeze(0))
        logabsdet = torch.sum(self.logs) * torch.ones(x.shape[0], dtype=torch.float64)
        return y, logabsdet

    def inverse(self, y):
        # (x, logabsdet)
        x = y * torch.exp(-self.logs.unsqueeze(0)) - self.bias.unsqueeze(0)
        logabsdet = -torch.sum(self.logs) * torch.ones(y.shape[0], dtype=torch.float64)
        return x, logabsdet  #

class AffineCoupling(nn.Module):
    def __init__(self, mask, transform_net_create_fn, scale_factor=0.2):
        super().__init__()
        self.register_buffer('mask', mask.clone().detach().double())
        self.in_dim = int((self.mask <= 0).sum().item())
        self.out_dim = int((self.mask > 0).sum().item())
        self.scale_factor = scale_factor
        self.transform_net = transform_net_create_fn(max(1, self.in_dim), self.out_dim * 2)

    def forward(self, inputs):
        identity = inputs[:, self.mask <= 0]
        to_transform = inputs[:, self.mask > 0]
        if self.out_dim == 0:
            return inputs, torch.zeros(inputs.shape[0], dtype=torch.float64)
        if self.in_dim == 0:
            dummy = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
            params = self.transform_net(dummy)
        else:
            params = self.transform_net(identity)
        params = params.view(inputs.shape[0], self.out_dim, 2)
        un_s = params[:, :, 0]; t = params[:, :, 1]
        s = torch.tanh(un_s) * self.scale_factor
        y_transformed = to_transform * torch.exp(s) + t
        logabsdet = torch.sum(s, dim=1)
        outputs = inputs.clone()
        outputs[:, self.mask > 0] = y_transformed
        return outputs, logabsdet

    def inverse(self, inputs):
        identity = inputs[:, self.mask <= 0]
        to_transform = inputs[:, self.mask > 0]
        if self.out_dim == 0:
            return inputs, torch.zeros(inputs.shape[0], dtype=torch.float64)
        if self.in_dim == 0:
            dummy = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
            params = self.transform_net(dummy)
        else:
            params = self.transform_net(identity)
        params = params.view(inputs.shape[0], self.out_dim, 2)
        un_s = params[:, :, 0]; t = params[:, :, 1]
        s = torch.tanh(un_s) * self.scale_factor
        x_inverted = (to_transform - t) * torch.exp(-s)
        logabsdet = -torch.sum(s, dim=1)
        outputs = inputs.clone()
        outputs[:, self.mask > 0] = x_inverted
        return outputs, logabsdet  #
    
class NormalizingFlow(nn.Module):
    def __init__(self, dim=12, num_flows=6, base_mean=None, base_cov=None):
        super().__init__()
        self.dim = dim
        if base_mean is None:
            base_mean = torch.zeros(dim)
        if base_cov is None:
            base_cov = torch.eye(dim)
        self.register_buffer('base_mean', base_mean.double())
        self.register_buffer('base_cov', base_cov.double())
        self.register_buffer('L_tril', torch.linalg.cholesky(base_cov.double()))
        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        for i in range(num_flows):
            self.actnorms.append(ActNorm(dim))
            mask = torch.ones(dim, dtype=torch.double)
            mask[::2] = 0.0
            if i % 2 == 1:
                mask = 1.0 - mask

            def create_net(in_features, out_features):
                net = nn.Sequential(
                    nn.Linear(int(in_features), 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, int(out_features))
                )
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        nn.init.constant_(layer.bias, 0.0)
                return net

            coupling = AffineCoupling(mask=mask, transform_net_create_fn=create_net, scale_factor=0.2)
            self.couplings.append(coupling)

    def forward(self, n_samples=1):
        eps = torch.randn(n_samples, self.dim, dtype=torch.float64)
        base_samples = self.base_mean.unsqueeze(0) + eps @ self.L_tril.T
        x = base_samples
        log_det_total = torch.zeros(n_samples, dtype=torch.float64)
        for act, cp in zip(self.actnorms, self.couplings):
            x, lnd1 = act.forward(x); log_det_total = log_det_total + lnd1
            x, lnd2 = cp.forward(x);  log_det_total = log_det_total + lnd2
        return x, log_det_total

    def log_prob(self, z):
        x = z.clone()
        log_det_total = torch.zeros(z.shape[0], dtype=torch.float64)
        # 
        for act, cp in reversed(list(zip(self.actnorms, self.couplings))):
            x, lnd1 = cp.inverse(x); log_det_total = log_det_total + lnd1  # x
            x, lnd2 = act.inverse(x); log_det_total = log_det_total + lnd2  # x
        
        base = MultivariateNormal(self.base_mean, covariance_matrix=self.base_cov)
        log_base = base.log_prob(x)
        return log_base + log_det_total

# =============================== ELBO  =======================================
def compute_elbo(model, y_obs, noise_std, prior_mean=0.0, prior_std=1.0,
                 current_epoch=None, total_epochs=None, min_samples=8, max_samples=16):
    if (current_epoch is not None) and (total_epochs is not None) and (total_epochs > 1):
        t = current_epoch / (total_epochs - 1)
        n_samples = int(round((1 - t) * min_samples + t * max_samples))
    else:
        n_samples = min_samples

    # q(z)
    z_samples, _ = model.forward(n_samples=n_samples)
    log_q_z = model.log_prob(z_samples).unsqueeze(1)

    # log p(z)
    prior_dist = torch.distributions.Normal(loc=prior_mean, scale=prior_std)
    log_p_z = torch.sum(prior_dist.log_prob(z_samples), dim=1, keepdim=True)

    # log p(y|z) —— uses SEM adjoint providing d logLik / dz
    log_p_y_given_z = []
    for i in range(n_samples):
        z_i = z_samples[i]
        ll_i = sem_loglik_with_adjoint(z_i, noisy_obs, obs_noise_std, obs_dt, obs_nt)  # differentiable w.r.t z_i
        log_p_y_given_z.append(ll_i)
    log_p_y_given_z = torch.stack(log_p_y_given_z, dim=0).unsqueeze(1)

    # ELBO
    elbo_terms = log_p_y_given_z + log_p_z - log_q_z
    elbo = torch.mean(elbo_terms)
    return elbo, torch.mean(log_p_y_given_z), torch.mean(log_p_z), torch.mean(log_q_z)

# ================== Gradient monitoring functions ==================
def get_gradient_stats(model):
    """Collect gradient statistics."""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'norm': grad.norm().item()
            }
    return grad_stats

def plot_gradient_history(gradient_history, filename='gradient_history_real.png'):
    """Plot gradient history."""
    if not gradient_history:
        return
        
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot([gh['total_norm'] for gh in gradient_history])
    plt.title('Total Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot([gh['max_grad'] for gh in gradient_history])
    plt.title('Max Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot([gh['mean_grad'] for gh in gradient_history])
    plt.title('Mean Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    plt.plot([gh['gradient_exploded'] for gh in gradient_history])
    plt.title('Gradient Explosion Events')
    plt.xlabel('Epoch')
    plt.ylabel('Exploded (1=Yes)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    # Plot gradient norms per layer
    if 'layer_norms' in gradient_history[0]:
        layer_names = list(gradient_history[0]['layer_norms'].keys())
        for layer in layer_names:
            norms = [gh['layer_norms'][layer] for gh in gradient_history if layer in gh['layer_norms']]
            plt.plot(norms, label=layer)
        plt.title('Gradient Norms by Layer')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    clipped_ratio = [gh.get('clipped_ratio', 0) for gh in gradient_history]
    plt.plot(clipped_ratio)
    plt.title('Gradient Clipping Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Clipped Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ================================= Step 6: Training setup ====================================
### prior = Gaussian
prior_mean = torch.zeros(12, dtype=torch.float64)
prior_std = torch.tensor(tstd, dtype=torch.float64)

model = NormalizingFlow(dim=12, num_flows=num_flows, base_mean=prior_mean, base_cov=(prior_std**2)*torch.eye(12))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

os.makedirs("intermediate_posteriors_real", exist_ok=True)

elbo_history = []; log_likelihood_history = []; log_prior_history = []; log_base_history = []
gradient_history = []

def plot_posterior_samples(epoch, post_samples_np):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ro--', markersize=8, label='True Control Points')
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'r-', lw=2, label='True Boundary')

    prior_mean_np = prior_mean.cpu().numpy().reshape(-1, 2)
    ctrl_pts_prior = prior_mean_np + ctrl_pts_original
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0:1]])
    ax.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], 'ko--', markersize=8, alpha=0.7, label='Prior Mean Points')
    spline_prior = BSpline(knots, np.vstack([ctrl_pts_prior, np.tile(ctrl_pts_prior[0], (k, 1))]), k)
    curve_points_prior = spline_prior(t_curve)
    ax.plot(curve_points_prior[:, 0], curve_points_prior[:, 1], 'k-', lw=2, alpha=0.7, label='Prior Mean Boundary')

    nplot = min(200, post_samples_np.shape[0])
    for i in range(nplot):
        sample = post_samples_np[i].reshape(-1, 2)
        ctrl_pts_sample = sample + ctrl_pts_original
        ax.plot(ctrl_pts_sample[:, 0], ctrl_pts_sample[:, 1], 'bo', markersize=4, alpha=0.08)
        ctrl_closed = np.vstack([ctrl_pts_sample, np.tile(ctrl_pts_sample[0], (k, 1))])
        spline_sample = BSpline(knots, ctrl_closed, k)
        curve_points_sample = spline_sample(t_curve)
        ax.plot(curve_points_sample[:, 0], curve_points_sample[:, 1], 'b--', lw=0.5, alpha=0.02)

    mean_sample = post_samples_np.mean(axis=0).reshape(-1, 2)
    mean_ctrl_pts = mean_sample + ctrl_pts_original
    mean_ctrl_closed = np.vstack([mean_ctrl_pts, np.tile(mean_ctrl_pts[0], (k, 1))])
    spline_mean = BSpline(knots, mean_ctrl_closed, k)
    curve_points_mean = spline_mean(t_curve)

    ax.plot(mean_ctrl_closed[:, 0], mean_ctrl_closed[:, 1], 'bo--', markersize=8, alpha=0.9, label='Posterior Mean Points')
    ax.plot(curve_points_mean[:, 0], curve_points_mean[:, 1], 'b-', lw=2, alpha=0.9, label='Posterior Mean Boundary')

    proxy_post = Patch(facecolor='blue', alpha=0.3, label='Posterior Samples')
    proxy_prior = Patch(facecolor='black', alpha=0.3, label='Prior Mean')
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [proxy_post, proxy_prior] + handles[-6:]

    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'VI Results at Epoch {epoch}\\n(True=Red, Posterior=Blue, Prior=Black, Mean=Blue)')
    ax.legend(handles=custom_handles); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'intermediate_posteriors_real/posterior_boundary_epoch_{epoch:04d}.png', dpi=200)
    plt.close()

# ================================= Step 7: Training starting ====================================
print("Starting training...")

grdnum = 200 # gradient test interval
for epoch in range(num_epochs):
    sem_config['method']['VERIFY_PROJECTION'] = (epoch % grdnum == 0)
    print('='*100) 
    t_epoch0 = time.perf_counter()
    optimizer.zero_grad()

    elbo, log_likelihood, log_prior, log_base = compute_elbo(
        model, noisy_obs, obs_noise_std, prior_mean, prior_std,
        current_epoch=epoch, total_epochs=num_epochs,
        min_samples=min_elbo_samples, max_samples=max_elbo_samples
    )

    loss = -elbo
    loss.backward()

    # Gradient monitoring
    grad_stats = get_gradient_stats(model)
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
    max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
    mean_grad = torch.mean(torch.stack([p.grad.abs().mean() for p in model.parameters() if p.grad is not None])).item()

    gradient_exploded = 0
    if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1e10:
        gradient_exploded = 1
        print(f"Epoch {epoch}: Gradient exploded! Norm: {total_norm.item()}")

    # Gradient clipping
    if clip_gradient and total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(scale)
        clipped_ratio = scale.item() if isinstance(scale, torch.Tensor) else float(scale)
        print(f"[Grad Scaling] total_norm={total_norm.item():.2e}, scaled by {scale:.3f}")
    else:
        clipped_ratio = 1.0

    #if clip_gradient and total_norm > max_grad_norm:
    #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    #    clipped_ratio = max_grad_norm / total_norm.item()
    #else:
    #    clipped_ratio = 1.0

    # Collect layer-wise gradient norms
    #layer_norms = {}
    #for name, param in model.named_parameters():
    #    if param.grad is not None:
    #        layer_norms[name] = torch.norm(param.grad).item()

    gradient_history.append({
        'epoch': epoch,
        'total_norm': total_norm.item(),
        'max_grad': max_grad,
        'mean_grad': mean_grad,
        'gradient_exploded': gradient_exploded,
        'clipped_ratio': clipped_ratio,
    #    'layer_norms': layer_norms
    })

    optimizer.step()

    elbo_history.append(elbo.item())
    log_likelihood_history.append(log_likelihood.item())
    log_prior_history.append(log_prior.item())
    log_base_history.append(log_base.item())

    t_epoch1 = time.perf_counter()
    epoch_time = t_epoch1 - t_epoch0

    if epoch % 1 == 0:
        print(f"Epoch {epoch:4d}/{num_epochs}: ELBO = {elbo.item():.4f}, "
              f"log_likelihood = {log_likelihood.item():.4f}, "
              f"log_prior = {log_prior.item():.4f}, "
              f"log_base = {log_base.item():.4f}, "
              f"grad_norm = {total_norm.item():.4f}, "
              f"time = {epoch_time:.2f}s")

    if epoch % 10 == 0:
        with torch.no_grad():
            post_samples, _ = model.forward(n_samples=5000)
            post_samples_np = post_samples.cpu().numpy()
            plot_posterior_samples(epoch, post_samples_np)

# ================================= Step 8: Utilities to render requested figures  ====================================
def _save_nf_posterior_distributions(posterior_samples, true_params_flat, filename='nf_posterior_distributions.png'):
    """
    Draws 12 histograms with prior mean , true value, and posterior mean.
    """
    initial_params = np.zeros(12)  # prior mean
    posterior_mean = np.mean(posterior_samples, axis=0)

    plt.figure(figsize=(15, 12))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.hist(posterior_samples[:, i], bins=30, density=True, alpha=0.7, label='NF Posterior')
        plt.axvline(true_params_flat[i], color='r', linestyle='--', linewidth=2, label='True value')
        plt.axvline(initial_params[i], color='g', linestyle='--', linewidth=2, label='Prior mean')
        plt.axvline(posterior_mean[i], color='b', linestyle='-', linewidth=2, label='Posterior mean')
        plt.xlabel(f'Param {i+1}'); plt.ylabel('Density')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def _save_nf_boundary_comparison(posterior_samples, filename='nf_boundary_comparison.png'):
    """
    Draws true/prior/posterior-mean boundaries + a cloud of sample boundaries and nodes.
    """
    # Mean offset -> posterior mean control points
    mean_offset = np.mean(posterior_samples, axis=0).reshape(-1, 2)
    ctrl_pts_mean = ctrl_pts_original + mean_offset

    # True
    ctrl_closed_true = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
    spline_true = BSpline(knots, np.vstack([ctrl_pts_true, np.tile(ctrl_pts_true[0], (k, 1))]), k)
    curve_points_true = spline_true(t_curve)

    # Prior 
    ctrl_pts_prior = ctrl_pts_original + np.zeros((6, 2))
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0]])
    spline_prior = BSpline(knots, np.vstack([ctrl_pts_prior, np.tile(ctrl_pts_prior[0], (k, 1))]), k)
    curve_points_prior = spline_prior(t_curve)

    # Posterior mean
    ctrl_closed_mean = np.vstack([ctrl_pts_mean, ctrl_pts_mean[0]])
    spline_mean = BSpline(knots, np.vstack([ctrl_pts_mean, np.tile(ctrl_pts_mean[0], (k, 1))]), k)
    curve_points_mean = spline_mean(t_curve)

    plt.figure(figsize=(12, 10))
    # True (red)
    plt.plot(curve_points_true[:, 0], curve_points_true[:, 1], 'r-', lw=3, label='True boundary')
    plt.plot(ctrl_closed_true[:, 0], ctrl_closed_true[:, 1], 'ro--', markersize=8, alpha=0.9, label='True nodes')

    # Prior (green)
    plt.plot(curve_points_prior[:, 0], curve_points_prior[:, 1], 'g--', lw=2, label='Prior mean boundary')
    plt.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], 'go--', markersize=6, alpha=0.7, label='Prior mean nodes')

    # Posterior mean (blue)
    plt.plot(curve_points_mean[:, 0], curve_points_mean[:, 1], 'b-', lw=2, label='Mean posterior boundary')
    plt.plot(ctrl_closed_mean[:, 0], ctrl_closed_mean[:, 1], 'bo--', markersize=6, alpha=0.7, label='Mean posterior nodes')

    # Sample boundaries/nodes in gray
    total_samples = len(posterior_samples)
    num_sample_boundaries = 1000
    step = max(1, total_samples // num_sample_boundaries)
    for i in range(0, total_samples, step):
        sample_offset = posterior_samples[i].reshape(-1, 2)
        ctrl_pts_sample = ctrl_pts_original + sample_offset
        ctrl_closed_sample = np.vstack([ctrl_pts_sample, np.tile(ctrl_pts_sample[0], (k, 1))])
        spline_sample = BSpline(knots, ctrl_closed_sample, k)
        curve_points_sample = spline_sample(t_curve)
        plt.plot(curve_points_sample[:, 0], curve_points_sample[:, 1], 'gray', lw=0.5, alpha=0.1, label='Sample boundaries' if i == 0 else "")
        plt.plot(ctrl_closed_sample[:, 0], ctrl_closed_sample[:, 1], 'gray', marker='o', linestyle='none', markersize=2, alpha=0.05, label='Sample nodes' if i == 0 else "")

    # Extra node cloud (light gray)
    all_sample_nodes = []
    for i in range(0, total_samples, max(1, total_samples // 1000)):
        sample_offset = posterior_samples[i].reshape(-1, 2)
        ctrl_pts_sample = ctrl_pts_original + sample_offset
        all_sample_nodes.extend(ctrl_pts_sample)
    all_sample_nodes = np.array(all_sample_nodes)
    plt.scatter(all_sample_nodes[:, 0], all_sample_nodes[:, 1], c='gray', alpha=0.02, s=10, label='Node distribution')

    plt.title('Boundary Comparison with Normalizing Flow Posterior Samples and Nodes')
    plt.xlabel('X (m)'); 
    plt.ylabel('Z (m)')
    plt.grid(True, alpha=0.3); 
    plt.legend(); 
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ================================= Step 9: Final posterior distribution  ====================================
print("Training finished. Generating final posterior samples...")
with torch.no_grad():
    post_samples, _ = model.forward(n_samples=5000)
    post_samples_np = post_samples.cpu().numpy()
    plot_posterior_samples(num_epochs, post_samples_np)

    # Save as numpy file
    np.save("posterior_samples.npy", post_samples_np)
    print(f"Posterior samples saved with shape: {post_samples_np.shape}")

    # Save as model
    torch.save(model.state_dict(), "trained_flow_model.pth")
    print("Trained model saved to trained_flow_model.pth")

    # === generate 'nf_posterior_distributions.png' and 'nf_boundary_comparison.png'
    try:
        _save_nf_posterior_distributions(
            posterior_samples=post_samples_np,
            true_params_flat=true_params.detach().cpu().numpy().reshape(-1),
            filename='nf_posterior_distributions.png'
        )
        print("Saved 'nf_posterior_distributions.png'")
    except Exception as e:
        print("Failed to save 'nf_posterior_distributions.png':", e)

    try:
        _save_nf_boundary_comparison(
            posterior_samples=post_samples_np,
            filename='nf_boundary_comparison.png'
        )
        print("Saved 'nf_boundary_comparison.png'")
    except Exception as e:
        print("Failed to save 'nf_boundary_comparison.png':", e)

# Plot ELBO history
plt.figure(figsize=(10, 6))
plt.plot(elbo_history, 'b-', lw=2, label='ELBO')
plt.plot(log_likelihood_history, 'r-', lw=2, label='log p(y|z)')
plt.plot(log_prior_history, 'g-', lw=2, label='log p(z)')
plt.plot(log_base_history, 'm-', lw=2, label='log q(z)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('ELBO and Components During Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('elbo_history_real.png', dpi=150)
plt.close()

# Plot gradient history
plot_gradient_history(gradient_history)

print('='*100) 
print("All done!")
