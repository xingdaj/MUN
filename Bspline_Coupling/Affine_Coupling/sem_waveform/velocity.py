import numpy as np
from scipy.interpolate import BSpline
from matplotlib.path import Path

# ----------------------------------------------------------------------------------------------------------------------
# Utility: Construct closed B-spline (6 control points + replicate first control point 3 times; open interval end knots)
# ----------------------------------------------------------------------------------------------------------------------
def _closed_bspline_from_ctrl(ctrl6_xy: np.ndarray, degree: int = 3):
    """
    ctrl6_xy: (6,2) float64
    return: (spline, spline_d1, spline_d2, knots, ctrl_closed)
            where spline can be vectorized evaluate (...,2)
    """
    ctrl6_xy = np.asarray(ctrl6_xy, dtype=np.float64).reshape(6, 2)
    k = degree
    # replicate first control point 3 times and append to the end
    ctrl_closed = np.vstack([ctrl6_xy, np.tile(ctrl6_xy[0], (k, 1))])  # (9,2)

    # open interval end knots vector [0..0, inner..., 1..1]
    n = ctrl_closed.shape[0] - 1  # n = 8
    # total knots = n + k + 2 = 8 + 3 + 2 = 13
    knots = np.zeros(n + k + 2, dtype=np.float64)
    knots[:k+1] = 0.0
    knots[-(k+1):] = 1.0
    # inner knots equally spaced (consistent with old code)
    inner = np.linspace(0.0, 1.0, n - k + 2)[1:-1]  # (n-k) = 5 intervals → 4 inner knots
    knots[k+1:-(k+1)] = inner

    spline    = BSpline(knots, ctrl_closed, k, extrapolate=False)         # (..,2)
    spline_d1 = spline.derivative(1)  # C'(u)
    spline_d2 = spline.derivative(2)  # C''(u)
    return spline, spline_d1, spline_d2, knots, ctrl_closed

# ----------------------------------------------------------------
# Core: Newton signed distance to true spline (vectorized+chunked)
# ----------------------------------------------------------------
def signed_distance_to_spline_newton(
    coords_xy: np.ndarray,     # (N,2)
    ctrl6_xy: np.ndarray,      # (6,2)
    samples: int = 800,        # initial sampling number (suggest 400~1200)
    newton_steps: int = 7,     # Newton steps per initial guess (5~8 is enough)
    topk: int = 1,             # select top-k initial values per point (1 is enough, can set 2 for more stability)
    chunk: int = 20000,        # chunked to avoid memory peaks
    return_extra: bool = True  # whether to return u_star/closest/tangent/normal
):
    """
    return:
      signed_dist: (N,)
      if return_extra=True, also return:
      u_star: (N,), closest: (N,2), tangent: (N,2), normal_out: (N,2)
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    N = coords_xy.shape[0]

    # construct closed spline and derivatives
    spline, s1, s2, knots, ctrl_closed = _closed_bspline_from_ctrl(ctrl6_xy, degree=3)

    # uniformly spaced initial values (do not repeat endpoints to avoid numerical degeneration)
    u0_grid = np.linspace(knots[3], knots[-4], samples, dtype=np.float64)  # valid interval
    C_u0 = spline(u0_grid)  # (S,2)

    # for inside/outside sign: use dense polygon sampling inside/outside test (robust and consistent with your original logic)
    poly_path = Path(C_u0, closed=True)

    # result buffer
    signed_dist = np.empty(N, dtype=np.float64)
    u_star      = np.empty(N, dtype=np.float64)
    closest     = np.empty((N, 2), dtype=np.float64)
    tangent     = np.empty((N, 2), dtype=np.float64)
    normal_out  = np.empty((N, 2), dtype=np.float64)

    # chunked processing
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X = coords_xy[s:e]  # (M,2)
        M = X.shape[0]

        # 1) select initial value: use nearest sampled point as candidate (can select top-k)
        #    dist^2(X, C(u0))
        #    broadcasting here (M,1,2)-(1,S,2) → (M,S,2)
        diff = X[:, None, :] - C_u0[None, :, :]          # (M,S,2)
        dist2 = np.sum(diff * diff, axis=2)              # (M,S)
        if topk == 1:
            idx0 = np.argmin(dist2, axis=1)             # (M,)
            U0 = u0_grid[idx0][:, None]                 # (M,1)
        else:
            idx0 = np.argpartition(dist2, topk-1, axis=1)[:, :topk]  # (M,topk)
            U0 = u0_grid[idx0]                                       # (M,topk)

        # 2) Newton refinement (iterate independently for each initial value)
        # initialization
        if topk == 1:
            u = U0.copy()   # (M,1)
        else:
            u = U0.copy()   # (M,topk)

        for _ in range(newton_steps):
            # evaluate C(u), C'(u), C''(u)
            Cu  = spline(u)    # (M,1,2) or (M,topk,2)
            Cp  = s1(u)
            Cpp = s2(u)
            r   = Cu - X[:, None, :]               # residual (M,*,2)
            # φ'(u) = r·Cp
            phi1 = np.sum(r * Cp, axis=2, keepdims=False)             # (M,*) 
            # φ''(u) = ||Cp||^2 + r·Cpp
            phi2 = np.sum(Cp * Cp, axis=2) + np.sum(r * Cpp, axis=2)  # (M,*)
            # damped/safe division
            phi2 = np.where(np.abs(phi2) < 1e-14, 1e-14, phi2)
            du = phi1 / phi2
            u = u - du
            # wrap back to valid interval (closed): equivalent to [0,1] ring
            # knots[3], knots[-4] are endpoints of valid parameter interval; simply clip here
            u = np.clip(u, knots[3], knots[-4])

        # 3) select best initial value (if topk>1)
        if topk > 1:
            Cu  = spline(u)                    # (M,topk,2)
            r   = Cu - X[:, None, :]
            dist2_refined = np.sum(r * r, axis=2)  # (M,topk)
            best = np.argmin(dist2_refined, axis=1)   # (M,)
            u_best = u[np.arange(M), best]            # (M,)
        else:
            u_best = u[:, 0]                          # (M,)

        # 4) final values: closest point, tangent, normal, distance
        Cb  = spline(u_best)           # (M,2)
        Tb  = s1(u_best)               # (M,2)
        Tb_norm = np.linalg.norm(Tb, axis=1, keepdims=True) + 1e-15
        t_hat = Tb / Tb_norm           # unit tangent (M,2)
        # first take "left normal" (rotate tangent +90°)
        #n_left = np.stack([-t_hat[:,1], t_hat[:,0]], axis=1)   # (M,2)
        rvec = X - Cb
        dist = np.linalg.norm(rvec, axis=1)

        # 5) sign: consistent with old implementation, use Path.contains as inside (negative)
        inside = poly_path.contains_points(X, radius=1e-12)  # (M,)
        sign = np.where(inside, -1.0, 1.0)
        sdist = sign * dist

        rvec = X - Cb                             # from curve point to node
        dist = np.linalg.norm(rvec, axis=1)
        eps  = 1e-15
        n_geom = rvec / (dist[:, None] + eps)     # normal direction of distance
        n_out  = sign[:, None] * n_geom  

        # 6) align "outer normal" with sign (for easier ADJ usage later)
        #    if inside(negative), then n_out should point outward; we use sign * n_left
        #    n_out = sign[:, None] * n_left

        # write back
        signed_dist[s:e] = sdist
        u_star[s:e]      = u_best
        closest[s:e, :]  = Cb
        tangent[s:e, :]  = t_hat
        normal_out[s:e,:]= n_out

    if return_extra:
        return signed_dist, u_star, closest, tangent, normal_out
    else:
        return signed_dist


# -----------------------------------------------------------------
# Entry: Build velocity (replace original polyline distance logic)
# -----------------------------------------------------------------
def build_velocity_on_sem_nodes(
    nodes_xy: np.ndarray,  # (N,2)
    ctrl6_xy: np.ndarray,  # (6,2)
    v_inside: float, v_outside: float,
    tau: float,
    samples: int = 800,
    newton_steps: int = 7
):
    """
    Based on the signed distance to the "true spline closest point" for smooth velocity transition.
    keep consistent with old API: return velocity(N,) and signed_dist(N,)
    Also return (u_star, closest, tangent, normal_out) for ADJ use.
    """
    nodes_xy = np.asarray(nodes_xy)
    if nodes_xy.ndim == 1:
        if nodes_xy.size % 2 == 0:
            nodes_xy = nodes_xy.reshape(-1, 2)
        else:
            raise ValueError(f"'nodes_xy' must be (N,2); got 1D of size {nodes_xy.size}.")
    if nodes_xy.ndim != 2 or nodes_xy.shape[1] != 2:
        raise ValueError(f"'nodes_xy' must be (N,2); got shape {nodes_xy.shape}")
    ctrl6_xy = np.asarray(ctrl6_xy, dtype=np.float64).reshape(6, 2)

    sd, u_star, Cclosest, t_hat, n_out = signed_distance_to_spline_newton(
        coords_xy=nodes_xy, ctrl6_xy=ctrl6_xy,
        samples=samples, newton_steps=newton_steps,
        topk=1, return_extra=True
    )

    # smooth interface: v = vmax + (vmin-vmax) * sigmoid(-sd/tau)
    tau_safe = max(float(tau), 1e-12)
    x = np.clip(sd / tau_safe, -50.0, 50.0)      # note this is sd/tau
    sig = 1.0 / (1.0 + np.exp(x))                # = σ(-sd/τ)
    v = v_outside + (v_inside - v_outside) * sig

    return v, sd, {'u_star': u_star, 'closest': Cclosest,
                   'tangent': t_hat, 'normal_out': n_out}