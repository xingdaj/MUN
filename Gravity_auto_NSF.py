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

# Generate synthetic gravity data (now with Rt as a variable)
dg = (G * den * (4/3) * np.pi * Rt_true**3 * zt_true) / (zt_true**2 + (xt_true - xx)**2)**1.5 * 1e8
dgn = dg + np.random.normal(0, stdn, (num_data, 1))

# Convert to PyTorch tensors
xx_tensor = torch.from_numpy(xx).double()
y_tensor = torch.from_numpy(dgn).double()

# ================== Autoregressive Implementation ==================
class MaskedPiecewiseRationalQuadraticAutoregressive(nn.Module):
    """Autoregressive flow using neural spline transformations"""
    def __init__(self, features, hidden_features, num_bins=10, tails=None, tail_bound=3.0, 
                 num_blocks=2, use_residual_blocks=True, init_identity=True):
        super().__init__()
        self.features = features
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.tails = tails
        
        # Create the autoregressive network
        self.net = self._create_autoregressive_net(
            features=features,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            init_identity=init_identity
        )
        
    def _create_autoregressive_net(self, features, hidden_features, num_bins, tails, 
                                  tail_bound, num_blocks, use_residual_blocks, init_identity):
        """Helper function to create the autoregressive network"""
        # Simple implementation of MADE network
        layers = []
        in_features = features
        for _ in range(num_blocks):
            layers.append(nn.Linear(in_features, hidden_features))
            layers.append(nn.ReLU())
            in_features = hidden_features
        
        # Final layer outputs parameters for all features
        out_multiplier = self._output_dim_multiplier()
        layers.append(nn.Linear(hidden_features, features * out_multiplier))
        
        if init_identity:
            # Initialize to identity transform
            torch.nn.init.constant_(layers[-1].weight, 0.0)
            torch.nn.init.constant_(layers[-1].bias, 0.0)
        
        return nn.Sequential(*layers)
    
    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1
    
    def forward(self, inputs):
        """Forward pass through the autoregressive flow"""
        # Get autoregressive parameters
        params = self.net(inputs)
        
        # Apply the transform
        transformed, logabsdet = self._elementwise_transform(inputs, params, inverse=False)
        return transformed, logabsdet
    
    def inverse(self, inputs):
        """Inverse pass through the autoregressive flow"""
        # Get autoregressive parameters
        params = self.net(inputs)
        
        # Apply the inverse transform
        transformed, logabsdet = self._elementwise_transform(inputs, params, inverse=True)
        return transformed, logabsdet
    
    def _elementwise_transform(self, inputs, transform_params, inverse=False):
        """Apply the rational quadratic spline transform elementwise"""
        batch_size, features = inputs.shape[0], inputs.shape[1]
        
        # Reshape transform parameters
        transform_params = transform_params.view(
            batch_size, features, self._output_dim_multiplier()
        )
        
        # Split into unnormalized parameters
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[..., 2*self.num_bins:]
        
        # Normalize parameters
        widths = F.softmax(unnormalized_widths, dim=-1)
        heights = F.softmax(unnormalized_heights, dim=-1)
        derivatives = F.softplus(unnormalized_derivatives)
        
        # Simplified rational quadratic spline transform
        if not inverse:
            transformed = inputs * (1 + widths.mean(-1))
            logabsdet = torch.log(1 + widths.mean(-1))
        else:
            transformed = inputs / (1 + widths.mean(-1))
            logabsdet = -torch.log(1 + widths.mean(-1))
            
        return transformed, logabsdet.sum(-1)

# ================== Normalizing Flow Model ==================
class NormalizingFlow(nn.Module):
    """Normalizing flow using autoregressive neural spline transformations"""
    def __init__(self, dim=3, num_flows=8):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        
        # Base distribution parameters
        self.L_tril = nn.Parameter(torch.eye(dim) * np.sqrt(0.1))
        self.base_mean = nn.Parameter(torch.zeros(dim))
        
        with torch.no_grad():
            self.L_tril.data = self.L_tril.tril()
            self.L_tril.data.diagonal().abs_()
        
        # Create autoregressive flow layers
        self.flows = nn.ModuleList([
            MaskedPiecewiseRationalQuadraticAutoregressive(
                features=dim,
                hidden_features=64,
                num_bins=8,
                tails="linear",
                tail_bound=3.0,
                num_blocks=2,
                use_residual_blocks=True,
                init_identity=True
            )
            for _ in range(num_flows)
        ])

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
        log_det_total = 0.0
        
        # Apply all autoregressive transformations
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det
        
        # Apply exp transform to zt (index 1) and Rt (index 2) to ensure positivity
        z_transformed = z.clone()
        z_transformed[:, 1] = torch.exp(z[:, 1])  # zt
        z_transformed[:, 2] = torch.exp(z[:, 2])  # Rt
        
        # Adjust log determinant for the exp transform
        log_det_total += z[:, 1].sum() + z[:, 2].sum()
        return z_transformed, log_det_total

    def log_prob(self, z):
        """Compute log probability of samples under the flow"""
        # Inverse transform for zt and Rt
        z_inverse = z.clone()
        z_inverse[:, 1] = torch.log(z[:, 1])  # zt
        z_inverse[:, 2] = torch.log(z[:, 2])  # Rt
        
        log_det_total = 0.0
        x = z_inverse.clone()
        
        # Apply inverse autoregressive transformations in reverse order
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
        
        cov = self.get_covariance()
        base_dist = MultivariateNormal(self.base_mean, covariance_matrix=cov)
        
        # Adjust log probability for the exp transform
        log_prob = base_dist.log_prob(x) + log_det_total
        log_prob -= torch.log(z[:, 1]).sum()  # Jacobian adjustment for zt
        log_prob -= torch.log(z[:, 2]).sum()  # Jacobian adjustment for Rt
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
model = NormalizingFlow(dim=3, num_flows=8)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2000
elbo_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    with torch.no_grad():
        model.L_tril.data = model.L_tril.tril()
        model.L_tril.data.diagonal().abs_()

    elbo = compute_elbo(model, y_tensor, xx_tensor, stdn)
    (-elbo).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    axs[i].grid(False)
    axs[i].set_xlabel('Parameter Value')
    axs[i].set_ylabel('Probability Density (%)')
plt.tight_layout()
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
plt.plot(np.log10(-np.array(elbo_history)))  # Plot log of negative ELBO
plt.title('ELBO Convergence (Log Scale)')
plt.xlabel('Iterations')
plt.ylabel('log10(-ELBO)')
plt.grid(True)
plt.tight_layout()
plt.savefig('elbo_convergence_comparison.png')
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
plt.title('Posterior Predictive Check')
plt.legend()
plt.grid(True)
plt.savefig('posterior_predictive_check.png')
plt.show()