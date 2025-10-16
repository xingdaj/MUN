import numpy as np
import torch


def setup_pml(xmin, xmax, zmin, zmax, pml_thickness_m, global_coords, dt, R=1e-8, c_max= None):
    """
    Setup PML for SEM domain with cubic ramp, return both indices and mask
    
    Args:
        xmin, xmax, zmin, zmax: Domain boundaries
        pml_thickness_m: PML thickness in meters
        global_coords: Global coordinates of all nodes
        dt: Time step size (required for decay factor calculation)
        R: Theoretical reflection coefficient
    
    Returns:
        pml_indices: Tensor indices of PML nodes
        pml_mask: Boolean mask of PML nodes (NumPy array)
        decay_factors: Precomputed decay factors for all nodes
    """

    if c_max is None:
        raise ValueError("c_max must be provided from config")
    
    #c_max = 3000.0  # conservative estimate
    m = 3  # cubic ramp
    
    δ = pml_thickness_m
    c_max = float(c_max)
    d0 = -(m + 1) * c_max * np.log(R) / (2.0 * δ)

    σx = np.zeros(len(global_coords))
    σz = np.zeros(len(global_coords))

    x = global_coords[:, 0]
    z = global_coords[:, 1]

    # left/right boundaries
    left  = (x < xmin + δ)
    right = (x > xmax - δ)
    σx[left]  = d0 * ((xmin + δ - x[left]) / δ) ** m
    σx[right] = d0 * ((x[right] - (xmax - δ)) / δ) ** m

    # bottom/top boundaries
    bottom = (z < zmin + δ)
    top    = (z > zmax - δ)
    σz[bottom] = d0 * ((zmin + δ - z[bottom]) / δ) ** m
    σz[top]    = d0 * ((z[top] - (zmax - δ)) / δ) ** m

    σ = σx + σz
    pml_mask = σ > 0.0  # Boolean mask (NumPy)
    decay_factors = np.exp(-σ * dt)  # Precompute decay factors for all nodes
    
    # Convert to torch tensors for GPU compatibility
    pml_indices = torch.where(torch.from_numpy(pml_mask))[0]
    decay_factors_torch = torch.from_numpy(decay_factors)
    
    return pml_indices, pml_mask, decay_factors_torch


def apply_precomputed_pml_all_layers(u_prev, u_curr, u_next, pml_mask, decay_factors):
    """
    Apply precomputed PML damping to all three time layers using NumPy mask
    """
    u_prev[pml_mask] *= decay_factors[pml_mask]
    u_curr[pml_mask] *= decay_factors[pml_mask]
    u_next[pml_mask] *= decay_factors[pml_mask]
    
    return u_prev, u_curr, u_next


def apply_pml_torch(u_prev, u_curr, u_next, pml_indices, decay_factors):
    """
    Apply PML damping using precomputed indices for in-place scaling (GPU-friendly)
    """
    if len(pml_indices) > 0:
        decay_vals = decay_factors[pml_indices]
        u_prev[pml_indices] *= decay_vals
        u_curr[pml_indices] *= decay_vals
        u_next[pml_indices] *= decay_vals
    
    return u_prev, u_curr, u_next


def apply_pml_torch_optimized(u_prev, u_curr, u_next, pml_indices, decay_factors):
    """
    Optimized version: single memory access for all three arrays
    """
    if len(pml_indices) > 0:
        decay_vals = decay_factors[pml_indices]
        
        # Single indexing operation for better performance
        u_prev_pml = u_prev[pml_indices]
        u_curr_pml = u_curr[pml_indices]
        u_next_pml = u_next[pml_indices]
        
        u_prev_pml *= decay_vals
        u_curr_pml *= decay_vals
        u_next_pml *= decay_vals
        
        # Assign back (this is still in-place for the original arrays)
        u_prev[pml_indices] = u_prev_pml
        u_curr[pml_indices] = u_curr_pml
        u_next[pml_indices] = u_next_pml
    
    return u_prev, u_curr, u_next