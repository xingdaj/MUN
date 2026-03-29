import torch
import numpy as np
import scipy.sparse as sp
from .utils import gll_points, lagrange_basis

def setup_receivers(src_x, src_z, receiver_radius, num_receivers, xmin, zmin, dx_elem, dz_elem, nelem_x, nelem_z, global_connectivity, global_coords, npol):
    """Setup receiver coordinates and interpolation weights, and build sparse interpolation matrix (both NumPy and PyTorch versions)"""
    ngll = npol + 1
    gll_nodes = gll_points(npol)  # Precompute GLL nodes once
    #npoints = len(global_connectivity) * ngll * ngll  # Total number of nodes
    npoints = len(global_coords) 
    
    # Create circular receiver array
    angles = np.linspace(0, 2 * np.pi, num_receivers, endpoint=False)
    receiver_coords_sem = [
        (src_x + receiver_radius * np.cos(theta), src_z + receiver_radius * np.sin(theta))
        for theta in angles
    ]
    
    # Precompute Lagrange basis functions for all GLL indices
    # We'll compute these on the fly for each receiver, but we can precompute the basis functions for each GLL index
    # Alternatively, we can precompute a 2D array of basis functions for each possible xi/eta and GLL index, but that might be too large.
    # Instead, we keep the current approach but ensure we use the precomputed gll_nodes.
    
    # Build sparse interpolation matrix R (nrec × npoints)
    row_indices = []
    col_indices = []
    data = []
    
    for rec_idx, (rx, rz) in enumerate(receiver_coords_sem):
        # Find the element containing the receiver
        ex = int((rx - xmin) / dx_elem)
        ez = int((rz - zmin) / dz_elem)
        ex = max(0, min(ex, nelem_x - 1))
        ez = max(0, min(ez, nelem_z - 1))
        
        elem_idx = ez * nelem_x + ex
        elem_nodes = global_connectivity[elem_idx]
        
        # Map receiver to reference element coordinates
        x0 = xmin + ex * dx_elem
        z0 = zmin + ez * dz_elem
        
        # Transform to reference coordinates [-1, 1]
        xi = 2.0 * (rx - x0) / dx_elem - 1.0
        eta = 2.0 * (rz - z0) / dz_elem - 1.0
        
        # Compute shape function values at receiver location
        for j in range(ngll):
            for i in range(ngll):
                node_idx = j * ngll + i
                global_node_idx = elem_nodes[node_idx]
                weight = lagrange_basis(xi, i, gll_nodes) * lagrange_basis(eta, j, gll_nodes)
                
                row_indices.append(rec_idx)
                col_indices.append(global_node_idx)
                data.append(weight)
    
    # Create sparse matrix in SciPy format
    R_sparse = sp.csr_matrix((data, (row_indices, col_indices)), 
                            shape=(num_receivers, npoints), dtype=np.float64)
    
    # Also create PyTorch sparse tensor version for efficient interpolation
    R_torch = create_torch_interpolation_matrix(R_sparse)
    
    # Also keep the traditional format for backward compatibility (if needed)
    receiver_interp_info_sem = []
    for rec_idx, (rx, rz) in enumerate(receiver_coords_sem):
        # Extract weights for this receiver from sparse matrix
        row_data = R_sparse[rec_idx, :].toarray().flatten()
        non_zero_indices = np.where(row_data != 0)[0]
        weights = row_data[non_zero_indices]
        
        receiver_interp_info_sem.append({
            'node_indices': non_zero_indices,
            'weights': weights,
            'global_coords': (rx, rz)
        })
    
    return receiver_coords_sem, receiver_interp_info_sem, R_sparse, R_torch

def interpolate_to_receivers_sem(u, receiver_interp_info):
    """Interpolate SEM field values to receiver locations using shape functions (NumPy version)"""
    receiver_data = np.zeros(len(receiver_interp_info))
    
    for i, info in enumerate(receiver_interp_info):
        elem_values = u[info['node_indices']]
        receiver_data[i] = np.dot(info['weights'], elem_values)
    
    return receiver_data

def interpolate_to_receivers_torch(u_torch, R_torch):
    """Interpolate to receivers using PyTorch sparse matrix multiplication"""
    # Ensure u_torch is a column vector for multiplication
    if u_torch.dim() == 1:
        u_torch = u_torch.unsqueeze(1)
    # Sparse matrix multiplication
    result = torch.sparse.mm(R_torch, u_torch)
    return result.squeeze()

def create_torch_interpolation_matrix(R_sparse):
    """Convert scipy sparse matrix to PyTorch sparse tensor (precomputed during setup)"""
    R_coo = R_sparse.tocoo()
    values = torch.from_numpy(R_coo.data).double()
    indices = torch.from_numpy(np.vstack([R_coo.row, R_coo.col])).long()
    shape = R_coo.shape
    R_torch = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float64)
    return R_torch.coalesce()  # Coalesce for efficient multiplication