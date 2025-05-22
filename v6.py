import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, LogNormal

# Enable double precision for numerical stability
torch.set_default_dtype(torch.float64)

# ================== Synthetic Data Generation ==================
G = 6.674e-11  # Gravitational constant (m³ kg⁻¹ s⁻²)
zt_true = 6.0    # True depth (m)
xt_true = 3.0    # True horizontal position (m)
Rt_true = 2.0    # True radius (m)
den = 2500.0     # Density contrast (kg/m³)

num_data = 100   # Number of observation points
x_range = (-20.0, 20.0)
xx = np.linspace(*x_range, num_data).reshape(-1, 1)
#stdn = 0.5       # Noise standard deviation (µGal)
stdn = 0.01
# Generate synthetic gravity data
dg = (G * den * (4/3) * np.pi * Rt_true**3 * zt_true) / (zt_true**2 + (xt_true - xx)**2)**1.5 * 1e8
dgn = dg + np.random.normal(0, stdn, (num_data, 1))

# Plot synthetic data
fig1 = plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(xx, dg, '-r*', lw=2, label='True')
plt.plot(xx, dgn, 'ko', markersize=3, label='Noisy')
plt.xlabel('Horizontal Location (m)', fontsize=12, fontproperties='serif')
plt.ylabel('Gravity Anomaly (µGal)', fontsize=12, fontproperties='serif')
plt.title('Synthetic Gravity Observations', fontsize=14, fontproperties='serif')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
circle = plt.Circle((xt_true, zt_true), Rt_true, color='k', fill=True)
ax = plt.gca()
ax.add_patch(circle)
plt.xlim(x_range[0], x_range[1])
plt.ylim(0, zt_true*2)
plt.gca().invert_yaxis()  # Reverse Y-axis for depth
plt.xlabel('Horizontal Location (m)', fontsize=12, fontproperties='serif')
plt.ylabel('Depth (m)', fontsize=12, fontproperties='serif')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure1.png')
plt.show()

# Convert to PyTorch tensors
xx_tensor = torch.from_numpy(xx).double()
y_tensor = torch.from_numpy(dgn).double()

# ================== Improved Coupling Layer ==================
class CouplingLayer(nn.Module):
    """Affine coupling layer with dynamic split handling"""
    def __init__(self, dim, hidden_dim=128, split_dim=1, swap=False):
        super().__init__()
        self.dim = dim
        self.split_dim = split_dim
        self.swap = swap
        
        # Dynamic input dimension based on split configuration
        self.net_input_dim = dim - split_dim if swap else split_dim
        
        self.scale_net = nn.Sequential(
            nn.Linear(self.net_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim - split_dim)
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(self.net_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim - split_dim)
        )

    def forward(self, x):
        if self.swap:
            x1, x2 = x[:, self.split_dim:], x[:, :self.split_dim]
        else:
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        z2 = x2 * torch.exp(s) + t
        
        if self.swap:
            z = torch.cat([z2, x1], dim=1)
        else:
            z = torch.cat([x1, z2], dim=1)
            
        log_det = s.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        if self.swap:
            z1, z2 = z[:, self.split_dim:], z[:, :self.split_dim]
        else:
            z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
            
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        x2 = (z2 - t) * torch.exp(-s)
        
        if self.swap:
            return torch.cat([x2, z1], dim=1)
        else:
            return torch.cat([z1, x2], dim=1)

# ================== Corrected Flow Model ==================
class NormalizingFlow(nn.Module):
    """Normalizing flow with alternating coupling directions"""
    def __init__(self, dim, num_flows=8):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        
        # Learn Cholesky factor with improved initialization
        self.L_tril = nn.Parameter(torch.eye(dim) * 0.1)
        self.base_mean = nn.Parameter(torch.tensor([1.0, 2.0, 1.0]))
        
        # Initialize Cholesky structure
        with torch.no_grad():
            self.L_tril.data = self.L_tril.tril()
            self.L_tril.data.diagonal().abs_().clamp_(min=0.01)
        
        # Create alternating coupling layers
        self.flows = nn.ModuleList()
        for i in range(num_flows):
            split_dim = 1 if i%2 == 0 else 2
            swap = (i%2 == 1)  # Alternate split directions
            self.flows.append(
                CouplingLayer(dim, split_dim=split_dim, swap=swap)
            )

    def get_covariance(self):
        """Construct positive-definite covariance matrix"""
        L = self.L_tril.tril()
        L = L + torch.eye(self.dim, device=L.device) * 1e-6
        return L @ L.T

    def forward(self, n_samples=1):
        """Sampling with alternating transformations"""
        cov = self.get_covariance()
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(n_samples, self.dim)
        z = self.base_mean + eps @ L.T
        
        log_det_total = 0.0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, z):
        """Compute probability via inverse pass"""
        log_det_total = 0.0
        x = z.clone()
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        cov = self.get_covariance()
        base_dist = MultivariateNormal(self.base_mean, covariance_matrix=cov)
        return base_dist.log_prob(x) + log_det_total

# ================== Physics-Constrained Forward Model ==================
def forward_model(params, xx):
    """Vectorized forward model with parameter constraints"""
    xi = params[:, 0].unsqueeze(-1)  # xt can be negative
    zi = torch.exp(params[:, 1].unsqueeze(-1))  # Ensure zt > 0
    Ri = torch.exp(params[:, 2].unsqueeze(-1))  # Ensure Rt > 0
    
    xx_br = xx.T  # [1, num_data]
    numerator = G * den * (4/3) * np.pi * Ri**3 * zi
    denominator = (zi**2 + (xi - xx_br)**2).pow(1.5)
    return (numerator * 1e8 / denominator).squeeze(-1)

# ================== Enhanced ELBO Calculation ==================
#def compute_elbo(model, y, xx, sigma_noise=0.5, n_samples=128):
    """ELBO with improved sampling and physics-aware priors"""
    z_samples, _ = model(n_samples)
    dg_pred = forward_model(z_samples, xx)
    
    # Physics-informed priors (LogNormal)
    log_prior_zt = LogNormal(6.0, 1.0).log_prob(z_samples[:, 1]).sum()  # 2.0,0.5
    log_prior_Rt = LogNormal(2.0, 1.0).log_prob(z_samples[:, 2]).sum()  # 1.0,0.3
    log_prior = log_prior_zt + log_prior_Rt
    
    # Likelihood and entropy terms
    log_likelihood = -0.5 * torch.sum((y.T - dg_pred)**2) / sigma_noise**2
    log_q = model.log_prob(z_samples)
    
    return (log_likelihood + log_prior - log_q).mean()


def compute_elbo(model, y, xx, sigma_noise=0.5, n_samples=128):
    """ELBO with explicit priors for all parameters"""
    z_samples, _ = model(n_samples)
    
    # Transform latent variables to physical parameters
    xt = z_samples[:, 0]  # Directly use latent value for xt (no constraint)
    zt = torch.exp(z_samples[:, 1])  # Ensure zt > 0
    Rt = torch.exp(z_samples[:, 2])  # Ensure Rt > 0
    
    # Compute forward model predictions
    params = torch.stack([xt, torch.log(zt), torch.log(Rt)], dim=1)  # Inverse transform for flow input
    dg_pred = forward_model(params, xx)
    
    # Define priors in PHYSICAL SPACE
    log_prior_xt = torch.distributions.Normal(0.0, 10.0).log_prob(xt).sum()        # xt ~ N(0, 10)
    log_prior_zt = LogNormal(6.0, 1.0).log_prob(zt).sum()                          # zt ~ LogN(6.0, 1.0)
    log_prior_Rt = LogNormal(2.0, 1.0).log_prob(Rt).sum()                          # Rt ~ LogN(2.0, 1.0)
    
    # Jacobian correction for exp transforms (zt and Rt)
    log_jacobian = z_samples[:, 1].sum() + z_samples[:, 2].sum()  # log|det(J)| = z1 + z2
    
    # Total prior = physical-space priors + Jacobian correction
    log_prior = log_prior_xt + log_prior_zt + log_prior_Rt + log_jacobian
    
    # Likelihood term
    log_likelihood = -0.5 * torch.sum((y.T - dg_pred)**2) / sigma_noise**2
    
    # Variational posterior probability (latent space)
    log_q = model.log_prob(z_samples)
    
    return (log_likelihood + log_prior - log_q).mean()

# ================== Training Configuration ==================
model = NormalizingFlow(dim=3, num_flows=8)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
elbo_history = []

# ================== Training Loop ==================
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Maintain Cholesky structure
    with torch.no_grad():
        model.L_tril.data = model.L_tril.tril()
        model.L_tril.data.diagonal().abs_().clamp_(min=0.01)
    
    # Compute and optimize ELBO
    elbo = compute_elbo(model, y_tensor, xx_tensor, stdn)
    (-elbo).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Track progress
    elbo_history.append(elbo.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, ELBO: {elbo.item():7.2f}')

# ================== Posterior Analysis ==================
with torch.no_grad():
    post_samples, _ = model(n_samples=10000)
    # Transform latent variables to physical space
    post_samples[:, 1] = torch.exp(post_samples[:, 1])  # zt
    post_samples[:, 2] = torch.exp(post_samples[:, 2])  # Rt
    post_samples_np = post_samples.numpy()

print("\nPosterior Statistics:")
print(f"Mean: {post_samples_np.mean(axis=0)}")
print(f"Std:  {post_samples_np.std(axis=0)}")
print(f"True: {[xt_true, zt_true, Rt_true]}")

# ================== Visualization ==================
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
params = ['xt (m)', 'zt (m)', 'Rt (m)']
true_vals = [xt_true, zt_true, Rt_true]

for i in range(3):
    axs[i].hist(post_samples_np[:, i], bins=50, density=True, alpha=0.7)
    axs[i].axvline(true_vals[i], color='r', linestyle='--', label='True value')
    axs[i].set_title(params[i])
    axs[i].legend()

plt.tight_layout()
plt.savefig('posterior_marginals.png')
plt.show()

# ELBO convergence plot
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(elbo_history, color='blue', lw=1.5)
#plt.plot(elbo_history)
plt.title('ELBO Convergence')
plt.xlabel('Training Epoch')
plt.ylabel('ELBO Value')
plt.grid(True)
#plt.savefig('elbo_convergence.png')
#plt.show()


# ================== ELBO Log10 Scale Plot ==================
#plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 2)
log_elbo = np.log10(np.abs(elbo_history) + 1e-16)  # np.sign(elbo_history)*
plt.plot(log_elbo, color='darkorange', lw=1.5)
plt.title(r'$\log_{10}$(|ELBO|) Convergence', fontsize=14)
plt.xlabel('Training Epoch', fontsize=12)
plt.ylabel(r'$\log_{10}$(|ELBO|)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbo_convergence.png', dpi=150)
plt.show()

# ================== Posterior Predictive Check ==================
# Extract a subset of posterior samples and compute forward predictions
n_samples_plot = 10000
post_samples_subset = post_samples_np[:n_samples_plot, :]

# Convert zt and Rt back to log space for forward_model input
xt_plot = post_samples_subset[:, 0]
zt_plot_log = np.log(post_samples_subset[:, 1])  # Convert zt to log space
rt_plot_log = np.log(post_samples_subset[:, 2])  # Convert Rt to log space
params_plot = np.column_stack((xt_plot, zt_plot_log, rt_plot_log))
params_plot_tensor = torch.from_numpy(params_plot).double()

# Compute forward predictions
with torch.no_grad():
    dg_pred_plot = forward_model(params_plot_tensor, xx_tensor).numpy()

# Fix dimension mismatch
xx_plot = xx.squeeze()  # Convert from (100, 1) to (100,)
dg_pred_plot = dg_pred_plot.T  # Transpose to (100, 1000)

# Plot comparison
plt.figure(figsize=(12, 7))

# Plot all posterior prediction curves
for i in range(n_samples_plot):
    plt.plot(xx_plot, dg_pred_plot[:, i],  # Correct dimension pairing
             color='gray', lw=0.5, alpha=0.03)

# Plot key components
plt.plot(xx_plot, dg.squeeze(), 'r-', lw=3, label='True Data', zorder=4)
plt.plot(xx_plot, dgn.squeeze(), 'ko', markersize=6, label='Noisy Data', zorder=3)
plt.plot(xx_plot, dg_pred_plot.mean(axis=1),  # Proper mean calculation
         'b--', lw=2.5, label='Posterior Mean', zorder=2)

# Add confidence interval
lower = np.percentile(dg_pred_plot, 5, axis=1)
upper = np.percentile(dg_pred_plot, 95, axis=1)
plt.fill_between(xx_plot, lower, upper, color='skyblue', alpha=0.3, label='90% CI', zorder=1)

plt.xlabel('Horizontal Location (m)', fontsize=12)
plt.ylabel('Gravity Anomaly (µGal)', fontsize=12)
plt.title('Corrected Posterior Predictive Check', fontsize=14)
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('corrected_forward_comparison.png', dpi=150)
plt.show()