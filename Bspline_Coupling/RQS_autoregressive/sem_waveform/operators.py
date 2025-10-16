import numpy as np
import scipy.sparse as sp
import torch
from .utils import gll_points, compute_gll_weights, differentiation_matrix_gll, lagrange_basis

def precompute_sem_operators(dx_elem, dz_elem, npol):
    """Precompute constant SEM operators"""
    ngll = npol + 1
    gll_nodes = gll_points(npol)
    gll_w = compute_gll_weights(gll_nodes)
    
    # Build derivative matrix
    D = differentiation_matrix_gll(gll_nodes)
    
    # Precompute Lagrange basis values at GLL nodes
    L = np.zeros((ngll, ngll))
    for j in range(ngll):
        for n in range(ngll):
            L[j, n] = lagrange_basis(gll_nodes[j], n, gll_nodes)
    
    # Convert to float64 for consistency
    L32 = L.astype(np.float64, copy=False)
    D32 = D.astype(np.float64, copy=False)
    w32 = gll_w.astype(np.float64, copy=False)

    # Geometric transform coefficients
    sxi = np.float64(2.0 / dx_elem)
    seta = np.float64(2.0 / dz_elem)
    J = np.float64((dx_elem * dz_elem) * 0.25)  # Jacobian determinant

    # Gradient operator matrices
    Gx = sxi * np.kron(L32, D32)   # (ngll*ngll, ngll*ngll)
    Gz = seta * np.kron(D32, L32)  # (ngll*ngll, ngll*ngll)

    # Interpolation matrix
    S_interp = np.kron(L32, L32)   # (ngll*ngll, ngll*ngll)

    # 2D integration weight vector
    w2D = (w32[:, None] * w32[None, :]).astype(np.float64)  # (ngll, ngll)
    w2D = (w2D * J).reshape(-1).astype(np.float64)          # (ngll*ngll,)

    return Gx, Gz, S_interp, w2D

def assemble_mass_matrix_lumped_acoustic(velocity_model, global_connectivity, dx_elem, dz_elem, npol, npoints):
    """Assemble lumped mass matrix for acoustic wave equation"""
    ngll = npol + 1
    gll_nodes = gll_points(npol)
    gll_w = compute_gll_weights(gll_nodes)
    
    diag = np.zeros(npoints)
    Jx = dx_elem / 2.0
    Jz = dz_elem / 2.0
    J  = Jx * Jz

    for elem in global_connectivity:
        for j in range(ngll):
            for i in range(ngll):
                gi = elem[j*ngll + i]
                w = gll_w[i] * gll_w[j] * J
                diag[gi] += w 

    return sp.diags(diag, format='csr')

# Modified assemble_mass_matrix_lumped_torch
def assemble_mass_matrix_lumped_torch(global_connectivity, dx_elem, dz_elem, npol, npoints):
    """Assemble lumped mass matrix diagonal vector using PyTorch"""
    ngll = npol + 1
    gll_nodes = gll_points(npol)
    gll_w = compute_gll_weights(gll_nodes)
    
    diag = torch.zeros(npoints, dtype=torch.float64)
    Jx = dx_elem / 2.0
    Jz = dz_elem / 2.0
    J = Jx * Jz

    for elem in global_connectivity:
        for j in range(ngll):
            for i in range(ngll):
                gi = elem[j*ngll + i]
                w = gll_w[i] * gll_w[j] * J
                diag[gi] += w

    # Return the diagonal vector instead of a matrix
    return diag

def assemble_stiffness_matrix_acoustic(velocity_model, global_connectivity, Gx, Gz, S_interp, w2D, npol, npoints):
    """Assemble stiffness matrix for acoustic wave equation"""
    ngll = npol + 1
    ng = ngll * ngll  # nodes per element
    
    rows, cols, data = [], [], []  # COO sparse components
    v_all = velocity_model.astype(np.float64, copy=False)

    # Loop over all elements
    for elem in global_connectivity:
        elem_nodes = np.asarray(elem)  # local-to-global mapping
        c2_nodal = (v_all[elem_nodes] ** 2).astype(np.float64, copy=False)  # c^2 at nodes

        # Interpolate to integration points
        c2_q = S_interp @ c2_nodal  # (ng,)

        # Weight × c2
        w_c = w2D * c2_q  # (ng,)

        # Element stiffness
        Ke = (Gx * w_c[:, None]).T @ Gx
        Ke += (Gz * w_c[:, None]).T @ Gz

        # Collect into sparse structure
        ga = np.repeat(elem_nodes, ng)  # row indices
        gb = np.tile(elem_nodes, ng)    # col indices
        rows.append(ga)
        cols.append(gb)
        data.append(Ke.reshape(-1))

    # Build sparse matrix
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    K = sp.coo_matrix((data, (rows, cols)), shape=(npoints, npoints), dtype=np.float64).tocsr()
    return K

def accumulate_stiffness_sensitivity_wrt_velocity(u_field, v_field,
                                                  velocity_model,
                                                  global_connectivity,
                                                  Gx, Gz, S_interp, w2D,
                                                  npol, npoints):
    """
    Compute the instantaneous sensitivity dJ/dc (w.r.t. nodal velocity).
    Returns a NumPy array (npoints,) to match existing callers.
    """
    ngll = npol + 1
    ng   = ngll * ngll

    # --- Ensure everything is Torch float64 on CPU (no autograd graph needed) ---
    device = torch.device('cpu')

    def to_t64(a):
        # Accept torch.Tensor or np.ndarray
        if isinstance(a, torch.Tensor):
            return a.to(dtype=torch.float64, device=device)
        a = np.asarray(a)
        return torch.from_numpy(a).to(dtype=torch.float64, device=device)

    u_all = to_t64(u_field)
    v_all = to_t64(v_field)
    c_all = to_t64(velocity_model)

    Gx_t      = to_t64(Gx)
    Gz_t      = to_t64(Gz)
    S_interp_t= to_t64(S_interp)
    w2D_t     = to_t64(w2D).reshape(-1)

    dJ_dc2_nodal = torch.zeros(npoints, dtype=torch.float64, device=device)

    # --- Element loop (matches assemble_stiffness_matrix_acoustic discretization) ---
    for elem in global_connectivity:
        elem = np.asarray(elem, dtype=np.int64)

        u_e = u_all[elem]     # (ng,)
        v_e = v_all[elem]     # (ng,)
        c_e = c_all[elem]     # (ng,)

        # c^2 at nodes -> at integration points
        c2_q = S_interp_t @ (c_e * c_e)  # (ng,)

        # Weights × c^2 at integration points
        #w_c = w2D_t * c2_q  # (ng,)

        # Gradients at integration points
        u_x = Gx_t @ u_e
        v_x = Gx_t @ v_e
        u_z = Gz_t @ u_e
        v_z = Gz_t @ v_e

        # Element contribution at integration points
        #g_q = w_c * (u_x * v_x + u_z * v_z)  # (ng,)
        g_q = w2D_t * (u_x * v_x + u_z * v_z)

        # Back to nodal c^2 derivative
        dJ_dc2_local = S_interp_t.T @ g_q  # (ng,)

        # Accumulate to nodal array
        dJ_dc2_nodal[elem] += dJ_dc2_local

    # dJ/dc = 2 c ⊙ dJ/d(c^2)
    dJ_dc = 2.0 * c_all * dJ_dc2_nodal

    # Return NumPy for callers that expect ndarray
    return dJ_dc.detach().cpu().numpy()

def assemble_stiffness_matrix_torch(velocity_model_torch, global_connectivity, Gx, Gz, S_interp, w2D, npol, npoints):
    """Assemble stiffness matrix using PyTorch with efficient sparse format"""
    ngll = npol + 1
    ng = ngll * ngll  # nodes per element
    
    # Convert operators to PyTorch
    Gx_torch = torch.from_numpy(Gx).double()
    Gz_torch = torch.from_numpy(Gz).double()
    S_interp_torch = torch.from_numpy(S_interp).double()
    w2D_torch = torch.from_numpy(w2D).double()
    
    # Use lists for COO format components
    rows_list, cols_list, data_list = [], [], []
    
    # Precompute element contributions
    for elem_idx, elem in enumerate(global_connectivity):
        elem_nodes = torch.tensor(elem, dtype=torch.long)
        c2_nodal = (velocity_model_torch[elem_nodes] ** 2)
        
        # Interpolate to integration points
        c2_q = S_interp_torch @ c2_nodal
        
        # Weight × c2
        w_c = w2D_torch * c2_q
        
        # Element stiffness
        Ke = (Gx_torch * w_c.unsqueeze(1)).T @ Gx_torch
        Ke += (Gz_torch * w_c.unsqueeze(1)).T @ Gz_torch
        
        # Collect sparse components
        ga = elem_nodes.repeat_interleave(ng)
        gb = elem_nodes.repeat(ng)
        rows_list.append(ga)
        cols_list.append(gb)
        data_list.append(Ke.reshape(-1))
    
    # Concatenate all components
    rows = torch.cat(rows_list)
    cols = torch.cat(cols_list)
    data = torch.cat(data_list)
    
    # Create COO format
    indices = torch.stack([rows, cols])
    K_coo = torch.sparse_coo_tensor(indices, data, (npoints, npoints), dtype=torch.float64)
    
    # Coalesce the COO tensor (this is where the error was occurring)
    K_coo = K_coo.coalesce()
    
    # Return as COO format instead of converting to CSR
    return K_coo

def stiffness_matrix_vector_product(u, velocity_model, global_connectivity, 
                                   Gx, Gz, S_interp, w2D, npol, npoints):
    """
    Compute K @ u without assembling the global stiffness matrix.
    
    Parameters
    ----------
    u : array-like, shape (npoints,)
        Input vector
    velocity_model : array-like, shape (npoints,)
        Nodal velocity c(x)
    global_connectivity : list/array of shape (nelem, ngll*ngll)
        Element-to-global node mapping
    Gx, Gz : np.ndarray, shape (ng, ng)
        Precomputed gradient operators
    S_interp : np.ndarray, shape (ng, ng)
        Interpolation matrix from nodes to integration points
    w2D : np.ndarray, shape (ng,)
        2D integration weights (including Jacobian)
    npol : int
        Polynomial order
    npoints : int
        Total number of global nodes
    
    Returns
    -------
    Ku : np.ndarray, shape (npoints,)
        Result of stiffness matrix-vector product K @ u
    """
    ngll = npol + 1
    ng = ngll * ngll
    
    # Initialize output
    Ku = np.zeros(npoints, dtype=np.float64)
    v_all = velocity_model.astype(np.float64, copy=False)
    u_all = u.astype(np.float64, copy=False)
    
    # Loop over elements
    for elem in global_connectivity:
        elem_nodes = np.asarray(elem)
        
        # Extract local values
        u_local = u_all[elem_nodes]
        c2_nodal = (v_all[elem_nodes] ** 2)
        
        # Interpolate to integration points
        c2_q = S_interp @ c2_nodal
        
        # Weight × c2
        w_c = w2D * c2_q
        
        # Compute local stiffness-vector product
        # Equivalent to: K_e @ u_local = Gx^T @ (w_c * (Gx @ u_local)) + Gz^T @ (w_c * (Gz @ u_local))
        Gx_u = Gx @ u_local
        Gz_u = Gz @ u_local
        
        Ku_local = Gx.T @ (w_c * Gx_u) + Gz.T @ (w_c * Gz_u)
        
        # Assemble into global vector
        Ku[elem_nodes] += Ku_local
    
    return Ku

def stiffness_matrix_vector_product_torch(u, velocity_model, global_connectivity,
                                         Gx, Gz, S_interp, w2D, npol, npoints):
    """
    PyTorch version of matrix-free K @ u computation.
    """
    ngll = npol + 1
    ng = ngll * ngll
    
    # Convert to tensors if needed
    if not isinstance(u, torch.Tensor):
        u = torch.as_tensor(u, dtype=torch.float64)
    if not isinstance(velocity_model, torch.Tensor):
        velocity_model = torch.as_tensor(velocity_model, dtype=torch.float64)
    
    # Convert operators to PyTorch
    Gx_torch = torch.as_tensor(Gx, dtype=torch.float64)
    Gz_torch = torch.as_tensor(Gz, dtype=torch.float64)
    S_interp_torch = torch.as_tensor(S_interp, dtype=torch.float64)
    w2D_torch = torch.as_tensor(w2D, dtype=torch.float64)
    
    # Initialize output
    Ku = torch.zeros(npoints, dtype=torch.float64, device=u.device)
    
    # Loop over elements
    for elem in global_connectivity:
        elem_nodes = torch.as_tensor(elem, dtype=torch.long)
        
        # Extract local values
        u_local = u[elem_nodes]
        c2_nodal = velocity_model[elem_nodes] ** 2
        
        # Interpolate to integration points
        c2_q = S_interp_torch @ c2_nodal
        
        # Weight × c2
        w_c = w2D_torch * c2_q
        
        # Compute local stiffness-vector product
        Gx_u = Gx_torch @ u_local
        Gz_u = Gz_torch @ u_local
        
        Ku_local = Gx_torch.T @ (w_c * Gx_u) + Gz_torch.T @ (w_c * Gz_u)
        
        # Assemble into global vector
        Ku.index_add_(0, elem_nodes, Ku_local)
    
    return Ku

def stiffness_matrix_vector_product_torch_optimized(u, velocity_model, global_connectivity,
                                                   Gx_torch, Gz_torch, S_interp_torch, w2D_torch, 
                                                   npol, npoints, device='cpu'):
    """
    Optimized PyTorch matrix-free K@u computation, avoid global matrix assembly
    Use preloaded operator tensors, support GPU acceleration
    """
    ngll = npol + 1
    ng = ngll * ngll
    
    # Ensure input is on the correct device
    if isinstance(u, torch.Tensor):
        u_tensor = u.to(device=device, dtype=torch.float64)
    else:
        u_tensor = torch.as_tensor(u, dtype=torch.float64, device=device)
    
    if isinstance(velocity_model, torch.Tensor):
        velocity_model_tensor = velocity_model.to(device=device, dtype=torch.float64)
    else:
        velocity_model_tensor = torch.as_tensor(velocity_model, dtype=torch.float64, device=device)
    
    # Initialize output (on the same device)
    Ku = torch.zeros(npoints, dtype=torch.float64, device=device)
    
    # Precompute operators (if not already on correct device)
    Gx_torch = Gx_torch.to(device=device, dtype=torch.float64)
    Gz_torch = Gz_torch.to(device=device, dtype=torch.float64)
    S_interp_torch = S_interp_torch.to(device=device, dtype=torch.float64)
    w2D_torch = w2D_torch.to(device=device, dtype=torch.float64)
    
    # Batch process elements (reduce Python loop overhead)
    batch_size = min(32, len(global_connectivity))  # Adjustable batch size
    
    for batch_start in range(0, len(global_connectivity), batch_size):
        batch_end = min(batch_start + batch_size, len(global_connectivity))
        batch_elems = global_connectivity[batch_start:batch_end]
        
        # Preallocate memory for batch
        batch_size_current = len(batch_elems)
        u_local_batch = torch.zeros(batch_size_current, ng, dtype=torch.float64, device=device)
        c2_nodal_batch = torch.zeros(batch_size_current, ng, dtype=torch.float64, device=device)
        elem_nodes_batch = []
        
        # Collect batch data
        for i, elem in enumerate(batch_elems):
            elem_nodes = torch.as_tensor(elem, dtype=torch.long, device=device)
            elem_nodes_batch.append(elem_nodes)
            u_local_batch[i] = u_tensor[elem_nodes]
            c2_nodal_batch[i] = velocity_model_tensor[elem_nodes] ** 2
        
        # Batch compute: interpolate to integration points
        c2_q_batch = torch.bmm(S_interp_torch.unsqueeze(0).expand(batch_size_current, -1, -1), 
                              c2_nodal_batch.unsqueeze(-1)).squeeze(-1)
        
        # weights × c²
        w_c_batch = w2D_torch.unsqueeze(0) * c2_q_batch
        
        # Batch compute: gradient
        Gx_u_batch = torch.bmm(Gx_torch.unsqueeze(0).expand(batch_size_current, -1, -1), 
                              u_local_batch.unsqueeze(-1)).squeeze(-1)
        Gz_u_batch = torch.bmm(Gz_torch.unsqueeze(0).expand(batch_size_current, -1, -1), 
                              u_local_batch.unsqueeze(-1)).squeeze(-1)
        
        # Batch compute: local stiffness matrix-vector product
        term1 = torch.bmm(Gx_torch.T.unsqueeze(0).expand(batch_size_current, -1, -1), 
                         (w_c_batch.unsqueeze(-1) * Gx_u_batch.unsqueeze(-1))).squeeze(-1)
        term2 = torch.bmm(Gz_torch.T.unsqueeze(0).expand(batch_size_current, -1, -1), 
                         (w_c_batch.unsqueeze(-1) * Gz_u_batch.unsqueeze(-1))).squeeze(-1)
        
        Ku_local_batch = term1 + term2
        
        # Assemble into global vector
        for i, elem_nodes in enumerate(elem_nodes_batch):
            Ku.index_add_(0, elem_nodes, Ku_local_batch[i])
    
    return Ku

def stiffness_matrix_vector_product_torch_simple(u, velocity_model, global_connectivity,
                                                Gx_torch, Gz_torch, S_interp_torch, w2D_torch,
                                                npol, npoints, device='cpu'):
    """
    Simplified matrix-free version (easier to debug)
    """
    ngll = npol + 1
    
    if isinstance(u, torch.Tensor):
        u_tensor = u.to(device=device, dtype=torch.float64)
    else:
        u_tensor = torch.as_tensor(u, dtype=torch.float64, device=device)
    
    if isinstance(velocity_model, torch.Tensor):
        velocity_model_tensor = velocity_model.to(device=device, dtype=torch.float64)
    else:
        velocity_model_tensor = torch.as_tensor(velocity_model, dtype=torch.float64, device=device)
    
    Ku = torch.zeros(npoints, dtype=torch.float64, device=device)
    
    # Ensure operators are on the correct device
    Gx_torch = Gx_torch.to(device=device, dtype=torch.float64)
    Gz_torch = Gz_torch.to(device=device, dtype=torch.float64)
    S_interp_torch = S_interp_torch.to(device=device, dtype=torch.float64)
    w2D_torch = w2D_torch.to(device=device, dtype=torch.float64)
    
    for elem in global_connectivity:
        elem_nodes = torch.as_tensor(elem, dtype=torch.long, device=device)
        u_local = u_tensor[elem_nodes]
        c2_nodal = velocity_model_tensor[elem_nodes] ** 2
        
        # Interpolate to integration points
        c2_q = S_interp_torch @ c2_nodal
        
        # weights × c²
        w_c = w2D_torch * c2_q
        
        # Compute local stiffness matrix-vector product
        Gx_u = Gx_torch @ u_local
        Gz_u = Gz_torch @ u_local
        
        Ku_local = Gx_torch.T @ (w_c * Gx_u) + Gz_torch.T @ (w_c * Gz_u)
        
        # Assemble
        Ku.index_add_(0, elem_nodes, Ku_local)
    
    return Ku























