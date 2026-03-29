import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import BSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ========================= Plot style =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.titlesize"] = 22
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 16

# ========================= RQS-consistent closed cubic B-spline =========================
k = 3  # cubic

def build_closed_bspline(ctrl_pts_base, num_samples=800):
    """
    Closed cubic B-spline (matches RQS approach):
    ctrl_closed = [ctrl_pts_base; ctrl_pts_base[0] repeated k times]
    clamped knots in [0,1]
    """
    ctrl_pts_base = np.asarray(ctrl_pts_base, dtype=float)
    ctrl_closed = np.vstack([ctrl_pts_base, np.tile(ctrl_pts_base[0], (k, 1))])

    n = len(ctrl_closed) - 1
    total_knots = n + k + 2
    knots = np.zeros(total_knots, dtype=float)
    knots[:k+1] = 0.0
    knots[-k-1:] = 1.0

    inner_knots = np.linspace(0.0, 1.0, n - k + 2)[1:-1]
    knots[k+1:-k-1] = inner_knots

    t_curve = np.linspace(knots[k], knots[-(k+1)], num_samples)
    spline = BSpline(knots, ctrl_closed, k, extrapolate=False)
    curve_points = spline(t_curve)
    return knots, t_curve, curve_points

# ========================= Geometry utilities: signed distance =========================
def point_in_polygon(x, y, poly_xy):
    poly = np.asarray(poly_xy, dtype=float)
    xp = poly[:, 0]
    yp = poly[:, 1]
    n = poly.shape[0]
    inside = np.zeros_like(x, dtype=bool)

    j = n - 1
    for i in range(n):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        intersect = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi)
        inside ^= intersect
        j = i
    return inside

def point_to_segments_distance(px, py, seg_x0, seg_y0, seg_x1, seg_y1):
    px = px[:, None]
    py = py[:, None]

    x0 = seg_x0[None, :]
    y0 = seg_y0[None, :]
    x1 = seg_x1[None, :]
    y1 = seg_y1[None, :]

    vx = x1 - x0
    vy = y1 - y0
    wx = px - x0
    wy = py - y0

    vv = vx * vx + vy * vy + 1e-30
    t = (wx * vx + wy * vy) / vv
    t = np.clip(t, 0.0, 1.0)

    projx = x0 + t * vx
    projy = y0 + t * vy
    dx = px - projx
    dy = py - projy
    d2 = dx * dx + dy * dy
    return np.sqrt(np.min(d2, axis=1))

def signed_distance_to_closed_polyline(px, py, poly_xy):
    poly = np.asarray(poly_xy, dtype=float)
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    d = point_to_segments_distance(px, py, x0, y0, x1, y1)
    inside = point_in_polygon(px, py, poly)
    return np.where(inside, -d, d)

# ========================= Domain & grid =========================
# Include z<0 in the grid (for completeness), but we will NOT plot z<0.
xmin, xmax = -300.0, 2000.0
zmin, zmax = -300.0, 1000.0

dx, dz = 10.0, 10.0
x = np.arange(xmin, xmax + dx, dx)
z = np.arange(zmin, zmax + dz, dz)
X, Z = np.meshgrid(x, z)  # (nz, nx)

# ========================= Background layered model (RQS-consistent for z>=0) =========================
# From your original code: interfaces_z = [0, 200, 500, 700] (z>=0 gives 4 layers to 1000)
interfaces_z = np.array([0.0, 200.0, 500.0, 700.0], dtype=float)

# For plotting (z>=0), use 4 layers velocities:
# 0-200: 1800; 200-500: 2400; 500-700: 2900; 700-1000: 3200
vel_layers = np.array([1800.0, 2400.0, 2900.0, 3200.0], dtype=float)

# Build V (background). Fill z<0 with something (won't be shown anyway).
V = np.empty_like(Z, dtype=float)
V[Z < 0.0] = 340.0  # matches original top/PML layer but will be cropped in plot

V[(Z >= 0.0) & (Z < interfaces_z[1])] = vel_layers[0]
V[(Z >= interfaces_z[1]) & (Z < interfaces_z[2])] = vel_layers[1]
V[(Z >= interfaces_z[2]) & (Z < interfaces_z[3])] = vel_layers[2]
V[(Z >= interfaces_z[3])] = vel_layers[3]

# ========================= 6-node B-spline anomaly =========================
co2_center = np.array([850.0, 350.0], dtype=float)
K1 = 6

def make_ellipse_control_points(center, num_points, rx, rz, rotation_deg=0.0):
    center = np.asarray(center, dtype=float)
    ang = np.linspace(0.0, 2.0*np.pi, num_points, endpoint=False)
    pts = np.stack([rx*np.cos(ang), rz*np.sin(ang)], axis=1)
    th = np.deg2rad(rotation_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    return pts + center[None, :]

# 6 nodes (ellipse-like); adjust rx/rz/rotation if needed
ctrl_pts_true = make_ellipse_control_points(
    center=co2_center,
    num_points=K1,
    rx=250.0,
    rz=70.0,
    rotation_deg=0.0
)

# B-spline boundary curve
_, _, curve_points = build_closed_bspline(ctrl_pts_true, num_samples=800)

# Anomaly velocity & smooth boundary thickness
v_inside = 2000.0
tau = 10.0  # smaller -> sharper boundary

# Signed distance and blending
px = X.ravel()
py = Z.ravel()
sd = signed_distance_to_closed_polyline(px, py, curve_points).reshape(X.shape)

indicator = 0.5 * (1.0 + np.tanh((-sd) / (tau + 1e-12)))  # ~1 inside, ~0 outside
V = V * (1.0 - indicator) + v_inside * indicator

# ========================= Acquisition geometry =========================
source_positions = np.array([
    [0.0, 10.0],
    [850.0, 10.0],
    [1700.0, 10.0]
], dtype=float)

receiver_positions = np.column_stack([
    np.linspace(0.0, 1700.0, 171),
    np.zeros(171)
])

# ========================= Plot (crop z>=0) =========================
mask_z = z >= 0.0
z_plot = z[mask_z]
V_plot = V[mask_z, :]

fig, ax = plt.subplots(figsize=(12, 6))  # flatter is natural with axis equal

im = ax.pcolormesh(
    x, z_plot, V_plot,
    cmap="turbo",
    shading="nearest"   # prevents white grid lines
)

# --- Colorbar matched to axes height (no "too long") ---
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3.5%", pad=0.08)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Velocity (m/s)")

# Boundary (B-spline curve)
ax.plot(curve_points[:, 0], curve_points[:, 1], "r-", lw=3, label="Boundary")

# Sources (white outline + red fill)
ax.scatter(
    source_positions[:, 0],
    source_positions[:, 1],
    marker="*",
    s=420,
    facecolor="red",
    edgecolor="white",
    linewidth=2.5,
    zorder=10,
    label="Sources"
)

# Receivers (white outline + blue fill)
ax.scatter(
    receiver_positions[:, 0],
    receiver_positions[:, 1],
    marker="^",
    s=60,                    # 比 source 小
    facecolor="#1f77b4",     # Matplotlib 默认蓝（干净）
    edgecolor="red",       # 白色描边，任何背景都清楚
    linewidth=1.2,
    zorder=8,
    label="Receivers"
)

# B-spline nodes (control points)
ctrl_closed = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
ax.plot(
    ctrl_closed[:, 0], ctrl_closed[:, 1],
    "ro--", markersize=10,
    label="B-spline Nodes",
    markeredgecolor="black", markeredgewidth=0.8
)

# Limits (z<0 removed already)
ax.set_xlim(0.0, 2000.0)
ax.set_ylim(0.0, zmax)
ax.invert_yaxis()

# Equal scaling + tight (axis tight)
ax.set_aspect("equal", adjustable="box")
ax.autoscale(enable=True, axis="both", tight=True)

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Layered Velocity Model with True Boundary")

ax.grid(False)
ax.legend(loc="upper right", frameon=True)

fig.tight_layout()
fig.savefig("velocity_model_smooth.png", dpi=600, bbox_inches="tight")
plt.close(fig)

print("Saved: velocity_model_true_clean.png")
