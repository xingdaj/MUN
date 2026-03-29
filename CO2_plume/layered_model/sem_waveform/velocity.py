
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
    ctrl6_xy = np.asarray(ctrl6_xy, dtype=np.float64).reshape(-1, 2)
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

    # for inside/outside sign: use dense polygon sampling inside/outside test (robust and consistent with original logic)
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
        diff = X[:, None, :] - C_u0[None, :, :]          # (M,S,2)
        dist2 = np.sum(diff * diff, axis=2)              # (M,S)
        if topk == 1:
            idx0 = np.argmin(dist2, axis=1)             # (M,)
            U0 = u0_grid[idx0][:, None]                 # (M,1)
        else:
            idx0 = np.argpartition(dist2, topk-1, axis=1)[:, :topk]  # (M,topk)
            U0 = u0_grid[idx0]                                       # (M,topk)

        # 2) Newton refinement
        u = U0.copy()
        for _ in range(newton_steps):
            Cu  = spline(u)    # (M,*,2)
            Cp  = s1(u)
            Cpp = s2(u)
            r   = Cu - X[:, None, :]
            phi1 = np.sum(r * Cp, axis=2)             # (M,*)
            phi2 = np.sum(Cp * Cp, axis=2) + np.sum(r * Cpp, axis=2)
            phi2 = np.where(np.abs(phi2) < 1e-14, 1e-14, phi2)
            du = phi1 / phi2
            u = u - du
            u = np.clip(u, knots[3], knots[-4])

        # 3) select best initial value (if topk>1)
        if topk > 1:
            Cu  = spline(u)                    # (M,topk,2)
            r   = Cu - X[:, None, :]
            dist2_refined = np.sum(r * r, axis=2)  # (M,topk)
            best = np.argmin(dist2_refined, axis=1)
            u_best = u[np.arange(M), best]
        else:
            u_best = u[:, 0]

        # 4) final values: closest point, tangent, normal, distance
        Cb  = spline(u_best)           # (M,2)
        Tb  = s1(u_best)               # (M,2)
        Tb_norm = np.linalg.norm(Tb, axis=1, keepdims=True) + 1e-15
        t_hat = Tb / Tb_norm           # unit tangent (M,2)

        rvec = X - Cb
        dist = np.linalg.norm(rvec, axis=1)

        # 5) sign via inside test (negative inside)
        inside = poly_path.contains_points(X, radius=1e-12)
        sign = np.where(inside, -1.0, 1.0)
        sdist = sign * dist

        # outward normal aligned with sign
        n_geom = rvec / (dist[:, None] + 1e-15)
        n_out  = sign[:, None] * n_geom

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
# Original API: Build velocity by single inside/outside values across a spline
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
    ctrl6_xy = np.asarray(ctrl6_xy, dtype=np.float64).reshape(-1, 2)

    sd, u_star, Cclosest, t_hat, n_out = signed_distance_to_spline_newton(
        coords_xy=nodes_xy, ctrl6_xy=ctrl6_xy,
        samples=samples, newton_steps=newton_steps,
        topk=1, return_extra=True
    )

    # smooth interface: v = vmax + (vmin-vmax) * sigmoid(-sd/tau)
    tau_safe = max(float(tau), 1e-12)
    x = np.clip(sd / tau_safe, -50.0, 50.0)      # note: sd/tau
    sig = 1.0 / (1.0 + np.exp(x))                # = σ(-sd/τ)
    v = v_outside + (v_inside - v_outside) * sig

    return v, sd, {'u_star': u_star, 'closest': Cclosest,
                   'tangent': t_hat, 'normal_out': n_out}

# -----------------------------------------------------------------
# NEW: Layered background model (horizontal layers) + optional spline anomaly
# -----------------------------------------------------------------
def build_layered_background(nodes_xy: np.ndarray, interfaces_z: np.ndarray, velocities: np.ndarray):
    """
    Build a horizontally layered velocity background.
    Parameters
    ----------
    nodes_xy : (N,2)
        Global SEM node coordinates (x,z). We treat `z` as the vertical coordinate.
    interfaces_z : (L-1,)
        Monotonically increasing z-values of layer interfaces (top to bottom). Length L-1 makes L layers.
    velocities : (L,)
        Velocity (m/s) for each layer from top to bottom.
    Returns
    -------
    v_layer : (N,)
        Layered velocity at each node.
    Notes
    -----
    - We interpret "depths" as *z-coordinates* in the model (units same as mesh).
    - Example: interfaces_z=[-600, 0, 300] gives 4 layers: z<-600, [-600,0), [0,300), z>=300.
    """
    nodes_xy = np.asarray(nodes_xy, dtype=np.float64)
    z = nodes_xy[:, 1]
    interfaces_z = np.asarray(interfaces_z, dtype=np.float64).reshape(-1)
    velocities   = np.asarray(velocities,   dtype=np.float64).reshape(-1)

    if interfaces_z.size + 1 != velocities.size:
        raise ValueError("len(velocities) must be len(interfaces_z)+1 (L layers).")

    # Determine layer index for each z by counting how many interfaces are <= z
    # (interfaces assumed sorted ascending)
    idx = np.searchsorted(interfaces_z, z, side='right')
    v_layer = velocities[idx]
    return v_layer

def build_velocity_layered_with_anomaly(
    nodes_xy: np.ndarray,
    interfaces_z: np.ndarray,
    layer_velocities: np.ndarray,
    ctrl6_xy: np.ndarray = None,
    v_inside: float = None,
    tau: float = 10.0,
    samples: int = 800,
    newton_steps: int = 7,
    blend: str = "smooth"  # "smooth" (sigmoid) or "replace"
):
    """
    Compose a layered background with an optional closed B-spline anomaly.
    - If ctrl6_xy is None, returns pure layered model.
    - If ctrl6_xy is provided, we either "smooth"-blend or "replace" inside the curve.
    Returns
    -------
    v : (N,)
        Final velocity field.
    sd : (N,) or None
        Signed distance to the spline (if anomaly provided), else None.
    extras : dict or None
        Closest points / tangents / normals (if anomaly provided), else None.
    """
    v_layer = build_layered_background(nodes_xy, interfaces_z, layer_velocities)

    if ctrl6_xy is None:
        return v_layer, None, None

    # signed distance to anomaly
    sd, u_star, Cclosest, t_hat, n_out = signed_distance_to_spline_newton(
        coords_xy=nodes_xy, ctrl6_xy=np.asarray(ctrl6_xy, dtype=np.float64).reshape(-1,2),
        samples=samples, newton_steps=newton_steps, topk=1, return_extra=True
    )

    if blend == "replace":
        v = v_layer.copy()
        inside = sd < 0.0
        v[inside] = float(v_inside)
    else:  # smooth default
        tau_safe = max(float(tau), 1e-12)
        x = np.clip(sd / tau_safe, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(x))   # σ(-sd/τ)
        # smoothly blend layered background with anomaly velocity
        v = v_layer + (float(v_inside) - v_layer) * sig

    extras = {'u_star': u_star, 'closest': Cclosest, 'tangent': t_hat, 'normal_out': n_out}
    return v, sd, extras