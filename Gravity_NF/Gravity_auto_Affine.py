import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

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

# Generate synthetic gravity data
dg = (G * den * (4/3) * np.pi * Rt_true**3 * zt_true) / (zt_true**2 + (xt_true - xx)**2)**1.5 * 1e8
dgn = dg + np.random.normal(0, stdn, (num_data, 1))

# Convert to PyTorch tensors
xx_tensor = torch.from_numpy(xx).double()
y_tensor = torch.from_numpy(dgn).double()

# ================== Fixed Autoregressive Implementation ==================
class MaskedAffineAutoregressive(nn.Module):
    """Fixed implementation with correct dimension handling"""
    def __init__(self, features, hidden_features, num_blocks=2):
        super().__init__()
        self.features = features
        
        # Initialize layers with proper dimension transitions
        layers = []
        # First layer maps features -> hidden_features
        layers.append(nn.Linear(features, hidden_features))
        layers.append(nn.ReLU())
        
        # Additional blocks maintain hidden_features
        for _ in range(num_blocks - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        
        # Final layer outputs scale and shift (features*2)
        layers.append(nn.Linear(hidden_features, features * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        params = self.net(inputs)
        transformed, logabsdet = self._affine_transform(inputs, params, inverse=False)
        return transformed, logabsdet

    def inverse(self, inputs):
        params = self.net(inputs)
        transformed, logabsdet = self._affine_transform(inputs, params, inverse=True)
        return transformed, logabsdet

    def _affine_transform(self, inputs, transform_params, inverse=False):
        batch_size = inputs.shape[0]
        transform_params = transform_params.view(batch_size, self.features, 2)
        scale = torch.sigmoid(transform_params[..., 0] + 2.0) + 1e-3
        shift = transform_params[..., 1]
        
        if not inverse:
            return scale * inputs + shift, torch.log(scale).sum(-1)
        else:
            return (inputs - shift) / scale, -torch.log(scale).sum(-1)

# ================== Normalizing Flow Model ==================
class NormalizingFlow(nn.Module):
    def __init__(self, dim, num_flows):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        
        # Use diagonal covariance for more stability
        self.log_diag = nn.Parameter(torch.zeros(dim))  # Log of diagonal elements
        self.base_mean = nn.Parameter(torch.zeros(dim))
        
        # Create autoregressive flow layers
        self.flows = nn.ModuleList([
            MaskedAffineAutoregressive(
                features=dim,
                hidden_features=64,
                num_blocks=2
            )
            for _ in range(num_flows)
        ])

    def get_covariance(self):
        """Compute diagonal covariance matrix with numerical stability"""
        return torch.diag(torch.exp(self.log_diag) + 1e-6)

    def forward(self, n_samples=1):
        """Sample from the flow with improved numerical stability"""
        cov = self.get_covariance()
        L = torch.diag(torch.sqrt(torch.diag(cov)))  # Diagonal matrix square root
        
        eps = torch.randn(n_samples, self.dim)
        z = self.base_mean + eps @ L.T
        log_det_total = 0.0
        
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        
        # Apply exp transform to zt and Rt to ensure positivity
        z_transformed = z.clone()
        z_transformed[:, 1] = torch.exp(z[:, 1])  # zt
        z_transformed[:, 2] = torch.exp(z[:, 2])  # Rt
        log_det_total += z[:, 1].sum() + z[:, 2].sum()
        return z_transformed, log_det_total

    def log_prob(self, z):
        """Compute log probability of samples under the flow"""
        z_inverse = z.clone()
        z_inverse[:, 1] = torch.log(z[:, 1])  # zt
        z_inverse[:, 2] = torch.log(z[:, 2])  # Rt
        
        log_det_total = 0.0
        x = z_inverse.clone()
        
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
        
        cov = self.get_covariance()
        base_dist = MultivariateNormal(self.base_mean, covariance_matrix=cov)
        log_prob = base_dist.log_prob(x) + log_det_total
        log_prob -= torch.log(z[:, 1]).sum()  # Jacobian adjustment for zt
        log_prob -= torch.log(z[:, 2]).sum()  # Jacobian adjustment for Rt
        return log_prob

# ================== Forward Model ==================
def forward_model(params, xx):
    """Compute forward gravity model for given parameters"""
    xi = params[:, 0].unsqueeze(-1)  # xt
    zi = params[:, 1].unsqueeze(-1)  # zt
    Ri = params[:, 2].unsqueeze(-1)  # Rt
    numerator = G * den * (4/3) * np.pi * Ri**3 * zi
    denominator = (zi**2 + (xi - xx.T)**2).pow(1.5)
    return (numerator * 1e8 / denominator).squeeze(-1)

# ================== ELBO Calculation ==================
def compute_elbo(model, y, xx, sigma_noise, n_samples=32):
    """Compute Evidence Lower Bound (ELBO) with numerical stability"""
    z_samples, _ = model(n_samples)
    dg_pred = forward_model(z_samples, xx)
    y_br = y.T
    
    # Clip predictions to avoid extreme values
    dg_pred = torch.clamp(dg_pred, -1e6, 1e6)
    
    log_likelihood = -0.5 * torch.sum((y_br - dg_pred)**2, dim=1) / sigma_noise**2
    log_prior = -0.5 * torch.sum(z_samples**2, dim=1)  # Simple isotropic Gaussian prior
    log_q = model.log_prob(z_samples)
    
    # Clip ELBO components to avoid extreme values
    log_likelihood = torch.clamp(log_likelihood, -1e6, 1e6)
    log_prior = torch.clamp(log_prior, -1e6, 1e6)
    log_q = torch.clamp(log_q, -1e6, 1e6)
    
    return (log_likelihood + log_prior - log_q).mean()

# ================== Training Loop ==================
model = NormalizingFlow(dim=3, num_flows=10)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight decay
num_epochs = 2000
elbo_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Clip gradients and parameters for stability
    with torch.no_grad():
        model.log_diag.data = torch.clamp(model.log_diag.data, -5, 5)
        model.base_mean.data = torch.clamp(model.base_mean.data, -10, 10)
    
    elbo = compute_elbo(model, y_tensor, xx_tensor, stdn)
    (-elbo).backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01) #0.1
    #torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.05)  #0.1
    
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

# Visualization
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
plt.xlabel('Parameter Value')
plt.ylabel('probability density(%)')
plt.savefig('posterior_marginals_3params.png')
plt.show()

# ELBO convergence plots
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(elbo_history)
plt.title('ELBO Convergence (Linear Scale)')
plt.xlabel('Iterations')
plt.ylabel('ELBO Value')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.log10(-np.array(elbo_history)))
plt.title('ELBO Convergence (Log Scale)')
plt.xlabel('Iterations')
plt.ylabel('log10(-ELBO)')
plt.grid(True)
plt.tight_layout()
plt.savefig('elbo_convergence_comparison.png')
plt.show()

# Posterior predictive check
with torch.no_grad():
    post_samples, _ = model(n_samples=100)
    preds = forward_model(post_samples, xx_tensor).numpy()
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
plt.title('Posterior Predictive Check')
plt.legend()
plt.grid(True)
plt.savefig('posterior_predictive_check.png')
plt.show()