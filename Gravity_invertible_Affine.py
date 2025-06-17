import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import scipy.linalg

# Enable double precision for numerical stability
torch.set_default_dtype(torch.float64)

# seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# ================== GLOW Flow Implementation ==================
class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution layer from GLOW paper"""
    def __init__(self, dim, use_lu=True):
        super().__init__()
        self.dim = dim
        self.use_lu = use_lu
        
        # Initialize weight matrix
        weight = torch.randn(dim, dim)
        if use_lu:
            # LU decomposition initialization
            np_p, np_l, np_u = scipy.linalg.lu(weight.numpy())
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            
            self.register_buffer('p', torch.tensor(np_p))
            self.register_buffer('sign_s', torch.tensor(np_sign_s))
            
            self.l = nn.Parameter(torch.tensor(np_l))
            self.u = nn.Parameter(torch.tensor(np_u))
            self.log_s = nn.Parameter(torch.tensor(np_log_s))
        else:
            # Direct parameterization
            self.weight = nn.Parameter(weight)
    
    def forward(self, z):
        if self.use_lu:
            # Reconstruct weight matrix from LU decomposition
            l = torch.tril(self.l, diagonal=-1) + torch.eye(self.dim)
            u = torch.triu(self.u, diagonal=1)
            log_s = self.log_s
            weight = self.p @ l @ (u + torch.diag(self.sign_s * torch.exp(log_s)))
        else:
            weight = self.weight
        
        # Apply transformation
        z_out = z @ weight.T
        log_det = torch.slogdet(weight)[1]
        return z_out, log_det * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
    
    def inverse(self, z):
        if self.use_lu:
            # Reconstruct weight matrix from LU decomposition
            l = torch.tril(self.l, diagonal=-1) + torch.eye(self.dim)
            u = torch.triu(self.u, diagonal=1)
            log_s = self.log_s
            weight = self.p @ l @ (u + torch.diag(self.sign_s * torch.exp(log_s)))
        else:
            weight = self.weight
        
        # Apply inverse transformation
        weight_inv = torch.inverse(weight)
        z_out = z @ weight_inv.T
        log_det = -torch.slogdet(weight)[1]
        return z_out, log_det * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)

class ActNorm(nn.Module):
    """Activation normalization layer from GLOW paper"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.initialized = False
    
    def forward(self, z):
        if not self.initialized:
            # Initialize parameters based on first batch
            with torch.no_grad():
                mean = z.mean(0)
                std = z.std(0)
                self.shift.data.copy_(-mean)
                self.log_scale.data.copy_(torch.log(1.0 / (std + 1e-6)))
                self.initialized = True
        
        scale = torch.exp(self.log_scale)
        z_out = scale * (z + self.shift)
        log_det = torch.sum(self.log_scale) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        return z_out, log_det
    
    def inverse(self, z):
        scale = torch.exp(self.log_scale)
        z_out = z / scale - self.shift
        log_det = -torch.sum(self.log_scale) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
        return z_out, log_det

class AffineCouplingBlock(nn.Module):
    """Affine coupling layer from GLOW paper"""
    def __init__(self, dim, hidden_dim=64, scale=True):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.split_dim = dim // 2
        
        # Neural network to predict shift and log-scale
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * (2 if scale else 1))
        )
        
        # Initialize last layer with zeros
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    
    def forward(self, z):
        z1, z2 = z.split([self.split_dim, self.dim - self.split_dim], dim=1)
        params = self.net(z1)
        
        if self.scale:
            shift, log_scale = params.chunk(2, dim=1)
            z2_out = z2 * torch.exp(log_scale) + shift
            log_det = torch.sum(log_scale, dim=1)
        else:
            shift = params
            z2_out = z2 + shift
            log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        
        z_out = torch.cat([z1, z2_out], dim=1)
        return z_out, log_det
    
    def inverse(self, z):
        z1, z2 = z.split([self.split_dim, self.dim - self.split_dim], dim=1)
        params = self.net(z1)
        
        if self.scale:
            shift, log_scale = params.chunk(2, dim=1)
            z2_out = (z2 - shift) * torch.exp(-log_scale)
            log_det = -torch.sum(log_scale, dim=1)
        else:
            shift = params
            z2_out = z2 - shift
            log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        
        z_out = torch.cat([z1, z2_out], dim=1)
        return z_out, log_det

class GLOWFlow(nn.Module):
    """GLOW normalizing flow model"""
    def __init__(self, dim=3, num_flows=8, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        
        # Base distribution parameters
        self.L_tril = nn.Parameter(torch.eye(dim) * np.sqrt(0.1))
        self.base_mean = nn.Parameter(torch.zeros(dim))
        
        with torch.no_grad():
            self.L_tril.data = self.L_tril.tril()
            self.L_tril.data.diagonal().abs_()
        
        # Create GLOW flow layers
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ActNorm(dim))
            self.flows.append(Invertible1x1Conv(dim))
            self.flows.append(AffineCouplingBlock(dim, hidden_dim))
    
    def get_covariance(self):
        """Compute covariance matrix from Cholesky factor"""
        L = self.L_tril.tril() + torch.eye(self.dim) * 1e-6
        return L @ L.T
    
    def forward(self, n_samples=1):
        """Sample from the flow and compute log determinant"""
        cov = self.get_covariance()
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(n_samples, self.dim)
        z = self.base_mean + eps @ L.T
        log_det_total = torch.zeros(n_samples, dtype=torch.float64)
        
        # Apply all flow transformations
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        
        # Apply exp transform to zt (index 1) and Rt (index 2) to ensure positivity
        z_transformed = z.clone()
        z_transformed[:, 1] = torch.exp(z[:, 1])  # zt
        z_transformed[:, 2] = torch.exp(z[:, 2])  # Rt
        
        # Adjust log determinant for the exp transform
        log_det_total += z[:, 1] + z[:, 2]  # Jacobian of exp is exp, so log|J| is z
        return z_transformed, log_det_total
    
    def log_prob(self, z):
        """Compute log probability of samples under the flow"""
        # Inverse transform for zt and Rt
        z_inverse = z.clone()
        z_inverse[:, 1] = torch.log(z[:, 1])  # zt
        z_inverse[:, 2] = torch.log(z[:, 2])  # Rt
        
        log_det_total = torch.zeros(z.shape[0], dtype=torch.float64)
        x = z_inverse.clone()
        
        # Apply inverse flow transformations in reverse order
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
        
        cov = self.get_covariance()
        base_dist = MultivariateNormal(self.base_mean, covariance_matrix=cov)
        
        # Adjust log probability for the exp transform
        log_prob = base_dist.log_prob(x) + log_det_total
        log_prob -= torch.log(z[:, 1])  # Jacobian adjustment for zt
        log_prob -= torch.log(z[:, 2])  # Jacobian adjustment for Rt
        return log_prob

# ================== Forward Model ==================
def forward_model(params, xx):
    """Compute forward gravity model for given parameters"""
    xi = params[:, 0].unsqueeze(-1)  # xt
    zi = params[:, 1].unsqueeze(-1)  # zt
    Ri = params[:, 2].unsqueeze(-1)  # Rt (now variable)
    numerator = G * den * (4/3) * np.pi * Ri**3 * zi
    denominator = (zi**2 + (xi - xx.T)**2).pow(1.5)
    return (numerator * 1e8 / denominator).squeeze(-1)

# ================== ELBO Calculation ==================
def compute_elbo(model, y, xx, sigma_noise, n_samples=32):
    """Compute Evidence Lower Bound (ELBO)"""
    z_samples, _ = model(n_samples)
    dg_pred = forward_model(z_samples, xx)
    y_br = y.T
    
    log_likelihood = -0.5 * torch.sum((y_br - dg_pred)**2, dim=1) / sigma_noise**2
    log_prior = -0.5 * torch.sum(z_samples**2, dim=1)  # Simple isotropic Gaussian prior
    log_q = model.log_prob(z_samples)
    return (log_likelihood + log_prior - log_q).mean()

# ================== Training Loop ==================
model = GLOWFlow(dim=3, num_flows=10)  # Now using GLOW flow
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2000 #######################################################################
elbo_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    with torch.no_grad():
        model.L_tril.data = model.L_tril.tril()
        model.L_tril.data.diagonal().abs_()
    
    elbo = compute_elbo(model, y_tensor, xx_tensor, stdn)
    (-elbo).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()
    elbo_history.append(elbo.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, ELBO: {elbo.item():7.2f}')

# ================== Posterior Analysis ==================
with torch.no_grad():
    post_samples, _ = model(n_samples=10000)
    post_samples_np = post_samples.numpy()

print("\nPosterior Statistics:")
print(f"Mean: {post_samples_np.mean(axis=0)}")
print(f"Std:  {post_samples_np.std(axis=0)}")
print(f"True: {[xt_true, zt_true, Rt_true]}")

# Visualization - now with 3 parameters
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
params = ['xt (m)', 'zt (m)', 'Rt (m)']
true_vals = [xt_true, zt_true, Rt_true]
for i in range(3):
    axs[i].hist(post_samples_np[:, i], bins=50, density=True, alpha=0.7)
    axs[i].axvline(true_vals[i], color='r', linestyle='--', label='True value')
    axs[i].set_title(params[i])
    axs[i].set_xlim(true_vals[i]-0.2, true_vals[i]+0.2)
    axs[i].legend()
plt.tight_layout()
plt.savefig('posterior_marginals_3params_glow.png')
plt.show()

# ELBO convergence plots
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(elbo_history)
plt.title('ELBO Convergence (Linear Scale)')
plt.xlabel('Training Epoch')
plt.ylabel('ELBO Value')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.log10(-np.array(elbo_history)))  # Plot log of negative ELBO (since ELBO is negative)
plt.title('ELBO Convergence (Log Scale)')
plt.xlabel('Training Epoch')
plt.ylabel('log10(-ELBO)')
plt.grid(True)
plt.tight_layout()
plt.savefig('elbo_convergence_comparison_glow.png')
plt.show()

# Posterior predictive check
with torch.no_grad():
    # Get posterior predictive samples
    post_samples, _ = model(n_samples=100)
    preds = forward_model(post_samples, xx_tensor).numpy()
    
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
plt.title('Posterior Predictive Check (GLOW)')
plt.legend()
plt.grid(True)
plt.savefig('posterior_predictive_check_glow.png')
plt.show()