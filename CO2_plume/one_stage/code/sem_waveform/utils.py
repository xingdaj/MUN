import numpy as np
from scipy.special import legendre
from scipy.optimize import newton

def gll_points(n):
    """Return Gauss-Lobatto-Legendre points for polynomial order n"""
    if n == 0:
        return np.array([0.0])
    elif n == 1:
        return np.array([-1.0, 1.0])
    
    poly_deriv = legendre(n).deriv()
    x = np.cos(np.pi * np.arange(n+1) / n)
    x = np.sort(x)
    
    for i in range(1, n):
        x[i] = newton(poly_deriv, x[i])
    
    x[0] = -1.0
    x[-1] = 1.0
    return x

def compute_gll_weights(nodes):
    """Compute weights for GLL quadrature"""
    n = len(nodes) - 1
    weights = np.zeros_like(nodes)
    for i in range(len(nodes)):
        weights[i] = 2 / (n * (n+1) * legendre(n)(nodes[i])**2)
    return weights

def lagrange_basis(x, i, nodes):
    """Evaluate i-th Lagrange basis polynomial at points x"""
    result = np.ones_like(x)
    for j in range(len(nodes)):
        if j != i:
            result *= (x - nodes[j]) / (nodes[i] - nodes[j] + 1e-12)
    return result

def barycentric_weights(x):
    """Correct computation of barycentric weights"""
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (x[j] - x[k])
    return w

def differentiation_matrix_gll(x):
    """Construct differentiation matrix using correct barycentric weights"""
    w = barycentric_weights(x)
    n = len(x)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / (w[i] * (x[i] - x[j]))
        D[i, i] = -np.sum(D[i, :])
    
    return D


def check_cfl_condition(velocity_model, dt, npol, dx_elem, dz_elem, vmax):
    """
    Check CFL condition for SEM method
    Returns: True if stable, False if unstable, and recommended dt
    """
    # === checkvelocity_modelwhethercontainsnon-finite values ===
    if not np.all(np.isfinite(velocity_model)):
        raise ValueError(f"[CFL] velocity_model contains non-finite values (NaN/Inf). "
                        f"Min: {np.nanmin(velocity_model):.6f}, Max: {np.nanmax(velocity_model):.6f}, "
                        f"NaN count: {np.sum(np.isnan(velocity_model))}")
    
    max_velocity = float(np.max(velocity_model))
    dim = 2.0

    # SEM CFL: dt <= C * h_min / ( c_max * (p+1)^2 * sqrt(dim) )
    p = npol  # polynomial order
    h_min_sem = min(dx_elem, dz_elem) / (p + 1)  # smallest GLL spacing
    C_sem = 0.35  # empirical constant for SEM
    dt_max_sem = C_sem * h_min_sem / (max_velocity * (p + 1) * np.sqrt(dim))
    
    #print(f"\nSEM CFL check:")
    #print(f"  h_min_sem = {h_min_sem:.6f} m, c_max = {max_velocity:.1f} m/s")
    #print(f"  p = {p}, dim = {dim}")
    #print(f"  dt_current = {dt:.6f} s, dt_max_sem = {dt_max_sem:.6f} s")
    #print(f"  Stability: {'OK' if dt <= dt_max_sem else 'UNSTABLE'}")
    
    if dt > dt_max_sem:
        print(f"  WARNING: Reduce dt to <= {dt_max_sem:.6f} s for SEM stability")
    
    return dt <= dt_max_sem, dt_max_sem


def adjust_time_step_for_stability(velocity_model, current_dt, total_time, npol, dx_elem, dz_elem, vmax):
    """
    Adjust time step to satisfy SEM CFL condition
    """
    stable, dt_limit = check_cfl_condition(velocity_model, current_dt, npol, dx_elem, dz_elem, vmax)
    
    if not stable:
        print(f"WARNING: dt={current_dt:.6f} exceeds stability limit {dt_limit:.6f}")
        print(f"Reducing dt to {dt_limit:.6f}")
        new_dt = float(dt_limit)
        new_nt = int(np.ceil(total_time / new_dt))
        print(f"Adjusted nt to {new_nt} to preserve total_time ≈ {total_time:.3f}s")
        return new_dt, new_nt
    else:
        #print("CFL condition satisfied for SEM")
        return current_dt, int(np.ceil(total_time / current_dt))