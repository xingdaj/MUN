# Final Normalizing Flow script with SEM-based simulator core
# - Uses SEM wave simulator for observations and for ELBO computation.
# - Flow uses ActNorm + piecewise rational-quadratic spline coupling for invertibility.
# - All wave-equation computations still use the Spectral Element Method.
# - Contact email: xingdaj@mun.ca.

# ============================================================================
# Part 1. Imports and module loading
# ============================================================================
import os
import sys
import time
import copy
import json
import random
import traceback
import shutil
import subprocess as _subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from scipy.interpolate import BSpline
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sem_waveform.core import SEMSimulation
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_velocity_layered_with_anomaly

# ============================================================================
# Part 2. User-adjustable parameters
# Put tunable parameters here for easier debugging and modification.
# ============================================================================
SEED = int(os.environ.get("SEED", "42"))
CPU_NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", "8"))

STAGE = str(os.environ.get("STAGE", "1")).strip()
try:
    STAGE_INT = int(STAGE)
except Exception as exc:
    raise ValueError(f"Invalid STAGE={STAGE!r}. Expected '1' or '2'.") from exc
if STAGE_INT not in (1, 2):
    raise ValueError(f"Invalid STAGE_INT={STAGE_INT}. This script supports only Stage-1 and Stage-2.")

# Geometry / model parameters
co2_center = np.array([850.0, 350.0], dtype=float)
K1 = 6
K_MAX = 20
K2_OVERRIDE = os.environ.get("K2_OVERRIDE", None)
K2 = int(K2_OVERRIDE) if (K2_OVERRIDE is not None and str(K2_OVERRIDE).strip() != "") else K_MAX
DIM_Z = 2 * K2
TSTD_DEFAULT = float(os.environ.get("TSTD_DEFAULT", "50.0"))

# Fixed Gaussian prior scales used by the two-stage inversion
FIXED_SIGMA_STAGE1 = float(os.environ.get("FIXED_SIGMA_STAGE1", "50.0"))  ###########
FIXED_SIGMA_STAGE2 = float(os.environ.get("FIXED_SIGMA_STAGE2", "20.0"))  ###########

# Regularization parameters
FUSE_LAMBDA = float(os.environ.get("FUSE_LAMBDA", str(10.0 / TSTD_DEFAULT)))
STAGE1_FUSE_SCALE = float(os.environ.get("STAGE1_FUSE_SCALE", "0.1"))
FUSE_EPS = float(os.environ.get("FUSE_EPS", "1e-3"))
FUSE_LAMBDA_EFF = FUSE_LAMBDA * (STAGE1_FUSE_SCALE if (STAGE_INT == 1) else 1.0)

# Stage-2 / cache / output control
AUTO_RUN_STAGE2 = (os.environ.get("AUTO_RUN_STAGE2", "1").strip() == "1")
BASE_CTRL_PATH = os.environ.get("BASE_CTRL_PATH", "").strip()
OBS_CACHE_DIR = os.environ.get("OBS_CACHE_DIR", "").strip()
if not OBS_CACHE_DIR:
    OBS_CACHE_DIR = "obs_cache"
os.makedirs(OBS_CACHE_DIR, exist_ok=True)

# SEM setup
DOMAIN_XMIN = float(os.environ.get("DOMAIN_XMIN", "-300"))
DOMAIN_XMAX = float(os.environ.get("DOMAIN_XMAX", "2000"))
DOMAIN_ZMIN = float(os.environ.get("DOMAIN_ZMIN", "-300"))
DOMAIN_ZMAX = float(os.environ.get("DOMAIN_ZMAX", "1000"))
NELEM_X = int(os.environ.get("NELEM_X", "30"))
NELEM_Z = int(os.environ.get("NELEM_Z", "30"))
TOTAL_TIME = float(os.environ.get("TOTAL_TIME", "1.2"))
DT = float(os.environ.get("DT", "0.80e-4"))
POLYNOMIAL_ORDER = int(os.environ.get("POLYNOMIAL_ORDER", "5"))
PML_THICKNESS = float(os.environ.get("PML_THICKNESS", "200.0"))
ADJ_HISTORY_DTYPE = os.environ.get("ADJ_HISTORY_DTYPE", "float32")
SOURCE_POSITIONS = [[0.0, 10.0], [850.0, 10.0], [1700.0, 10.0]]
SOURCE_FREQUENCY = float(os.environ.get("SOURCE_FREQUENCY", "20.0"))
SOURCE_AMPLITUDE = float(os.environ.get("SOURCE_AMPLITUDE", "1.0"))
RECEIVER_XMIN = float(os.environ.get("RECEIVER_XMIN", "0.0"))
RECEIVER_XMAX = float(os.environ.get("RECEIVER_XMAX", "1700.0"))
RECEIVER_DX = float(os.environ.get("RECEIVER_DX", "10.0"))
RECEIVER_Z = float(os.environ.get("RECEIVER_Z", "0.0"))
RECEIVER_POSITIONS = [[float(x), RECEIVER_Z] for x in np.arange(RECEIVER_XMIN, RECEIVER_XMAX + 0.5 * RECEIVER_DX, RECEIVER_DX)]
VEL_VMIN = float(os.environ.get("VEL_VMIN", "1500.0"))
VEL_VMAX = float(os.environ.get("VEL_VMAX", "3500.0"))
VEL_TAU = float(os.environ.get("VEL_TAU", "10.0"))
VEL_SPLINE_SAMPLES = int(os.environ.get("VEL_SPLINE_SAMPLES", "400"))
LAYER_INTERFACES_Z = [0.0, 200.0, 500.0, 700.0]
LAYER_VELOCITIES = [340.0, 1800.0, 2400.0, 2900.0, 3200.0]
ANOMALY_ENABLED = True
ANOMALY_V_INSIDE = float(os.environ.get("ANOMALY_V_INSIDE", "2000.0"))
ANOMALY_TAU = float(os.environ.get("ANOMALY_TAU", "10.0"))
ANOMALY_SPLINE_SAMPLES = int(os.environ.get("ANOMALY_SPLINE_SAMPLES", "400"))
ANOMALY_BLEND = os.environ.get("ANOMALY_BLEND", "smooth")

# Training parameters
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1"))  ##### print
PLOT_EVERY = int(os.environ.get("PLOT_EVERY", "10"))   ##### save
learning_rate = float(os.environ.get("LEARNING_RATE", "5e-4"))
num_flows = int(os.environ.get("NUM_FLOWS", "16"))
K_bins = int(os.environ.get("K_BINS", "12"))
min_elbo_samples = int(os.environ.get("MIN_ELBO_SAMPLES", "4")) #################
max_elbo_samples = int(os.environ.get("MAX_ELBO_SAMPLES", "8")) #################
obs_noise_std = float(os.environ.get("OBS_NOISE_STD", str(0.5e-8)))
max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", "50.0"))
clip_gradient = os.environ.get("CLIP_GRADIENT", "1").strip() == "1"
monitor_gradient = os.environ.get("MONITOR_GRADIENT", "1").strip() == "1"
torch.set_default_dtype(torch.float64)
EPS64 = 1e-12
EPS = EPS64
num_epochs = int(os.environ.get('NUM_EPOCHS', '0') or 0)
if num_epochs <= 0:
    if STAGE_INT == 1:
        num_epochs = int(os.environ.get("NUM_EPOCHS_STAGE1", "200"))### Stage-1 iteration  
    elif STAGE_INT == 2:
        num_epochs = int(os.environ.get("NUM_EPOCHS_STAGE2", "100"))##### Stage-2 iteration
    else:
        raise ValueError(f"Invalid STAGE_INT={STAGE_INT}. This script supports only Stage-1 and Stage-2.")

# ============================================================================
# Part 3. Core implementation
# ============================================================================

# ----------------- Threading configuration for CPU parallelism -----------------
try:
    torch.set_num_threads(CPU_NUM_THREADS)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_NUM_THREADS))


def set_global_seed(seed: int) -> None:
    """Best-effort global seeding for reproducibility across numpy/torch/random."""
    try:
        seed = int(seed)
    except Exception:
        seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    print(f"[SEED] {seed}")

set_global_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
    )
)
print(f"[DEVICE] {device}")

print("Using SEMSimulation from:", SEMSimulation.__module__, SEMSimulation.__qualname__)
_SIM_CACHE = {}


def _sim_key_from_cfg(cfg):
    d = cfg['domain']
    m = cfg['method']
    r = cfg['receivers']
    t = cfg['time']
    s = cfg['source']
    vel = cfg['velocity']
    meth = cfg['method']
    key_core = (
        d['xmin'], d['xmax'], d['zmin'], d['zmax'],
        d['nelem_x'], d['nelem_z'],
        m['polynomial_order'],
        m['pml_thickness'],
        len(r['positions']),
        t['total_time'], t['dt'],
    )
    src_positions = tuple(tuple(float(v) for v in pos) for pos in s['positions'])
    pos_single = tuple(float(v) for v in s.get('position', [0.0, 0.0]))
    key_extra = (
        pos_single,
        src_positions,
        float(s['frequency']),
        float(s['amplitude']),
        tuple(float(v) for v in vel['layers']['velocities']),
        tuple(float(v) for v in vel['layers']['interfaces_z']),
        bool(meth.get('VERIFY_PROJECTION', False)),
    )
    return key_core + key_extra


def get_or_make_sim(cfg):
    key = _sim_key_from_cfg(cfg)
    sim = _SIM_CACHE.get(key)
    if sim is None:
        sim = SEMSimulation(cfg)
        _SIM_CACHE[key] = sim
    return sim


def build_single_source_cfg(base_cfg, src_pos):
    cfg_k = copy.deepcopy(base_cfg)
    cfg_k['source'] = cfg_k['source'].copy()
    cfg_k['source']['position'] = [float(src_pos[0]), float(src_pos[1])]
    return cfg_k

def make_ellipse_control_points(center, num_points, rx, rz, rotation_deg=0.0):
    """Generate num_points control points on an ellipse around 'center'.
    center: (2,) array-like, [x_c, z_c]
    rx: horizontal semi-axis
    rz: vertical semi-axis (z positive downward)
    rotation_deg: rotation angle in degrees (counter-clockwise)
    """
    center = np.asarray(center, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    # Base ellipse in local coordinates
    pts = np.stack([rx * np.cos(angles), rz * np.sin(angles)], axis=1)
    # Rotation
    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    pts = pts @ R.T
    # Shift to center
    pts = pts + center[None, :]
    return pts
# TRUE model baseline control points (K1 nodes, used to generate synthetic data)
ctrl_pts_true_base = make_ellipse_control_points(
    center=co2_center,
    num_points=K1,
    rx=220.0,   # horizontal extent of CO2 plume
    rz=80.0,    # vertical extent (thinner)
    rotation_deg=0.0
)
# INITIAL / inversion model baseline control points (K2 nodes, optimized via NF)
ctrl_pts_init_base = make_ellipse_control_points(
    center=co2_center,
    num_points=K2,
    rx=220.0,
    rz=80.0,
    rotation_deg=0.0
)
# SEM configuration assembled from Part 2 parameters
sem_config = {
    'domain': {
        'xmin': DOMAIN_XMIN,
        'xmax': DOMAIN_XMAX,
        'zmin': DOMAIN_ZMIN,
        'zmax': DOMAIN_ZMAX,
        'nelem_x': NELEM_X,
        'nelem_z': NELEM_Z,
    },
    'time': {
        'total_time': TOTAL_TIME,
        'dt': DT,
    },
    'source': {
        'positions': copy.deepcopy(SOURCE_POSITIONS),
        'frequency': SOURCE_FREQUENCY,
        'amplitude': SOURCE_AMPLITUDE,
    },
    'receivers': {
        'positions': copy.deepcopy(RECEIVER_POSITIONS),
    },
    'method': {
        'polynomial_order': POLYNOMIAL_ORDER,
        'pml_thickness': PML_THICKNESS,
        'adj_history_dtype': ADJ_HISTORY_DTYPE,
    },
    'velocity': {
        'vmin': VEL_VMIN,
        'vmax': VEL_VMAX,
        'tau': VEL_TAU,
        'spline_samples': VEL_SPLINE_SAMPLES,
        'control_points': ctrl_pts_true_base.tolist(),
        'perturbations': None,
        'layers': {
            'interfaces_z': copy.deepcopy(LAYER_INTERFACES_Z),
            'velocities': copy.deepcopy(LAYER_VELOCITIES),
        },
        'anomaly': {
            'enabled': ANOMALY_ENABLED,
            'control_points': ctrl_pts_true_base.tolist(),
            'perturbations': None,
            'v_inside': ANOMALY_V_INSIDE,
            'tau': ANOMALY_TAU,
            'spline_samples': ANOMALY_SPLINE_SAMPLES,
            'blend': ANOMALY_BLEND,
        },
    },
    'output': {
        'save_wavefield': False,
        'save_seismograms': True,
        'visualize': False,
        'output_dir': 'sem_output',
        'snapshot_interval': 10**9,
    },
}
# ================================ step 1: B-spline & True Control Points Setup =====================================
time0 = time.perf_counter()
tstd = TSTD_DEFAULT
# --- TRUE model control-point offsets (K1 nodes) ---
true_offset = np.random.normal(loc=0.0, scale=tstd, size=(K1, 2))  # Gaussian offsets per TRUE control point
ctrl_pts_true = ctrl_pts_true_base + true_offset  # (K1, 2)
# Global spline degree
k = 3  # cubic B-spline degree
def build_closed_bspline(ctrl_pts_base, num_samples=800):
    """Build a closed cubic B-spline curve from a set of control points.
    Returns:
        knots      : knot vector
        t_curve    : parameter samples
        curve_pts  : sampled points on the curve (num_samples, 2)
    """
    ctrl_closed = np.vstack([ctrl_pts_base, np.tile(ctrl_pts_base[0], (k, 1))])
    n = len(ctrl_closed) - 1
    total_knots = n + k + 2
    knots = np.zeros(total_knots)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    inner_knots = np.linspace(0, 1, n - k + 2)[1:-1]
    knots[k+1:-k-1] = inner_knots
    t_curve = np.linspace(knots[k], knots[-(k+1)], num_samples)
    spline = BSpline(knots, ctrl_closed, k, extrapolate=False)
    curve_points = spline(t_curve)
    return knots, t_curve, curve_points
# Build B-spline for the TRUE boundary (K1 control points)
_, _, curve_points_true = build_closed_bspline(ctrl_pts_true, num_samples=800)
plt.figure(figsize=(10, 6))
ax = plt.gca()
ctrl_pts_true_closed = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
ax.plot(ctrl_pts_true_closed[:, 0], ctrl_pts_true_closed[:, 1], 'ro--', markersize=8, label='True Control Points')
ax.plot(curve_points_true[:, 0], curve_points_true[:, 1], 'r-', lw=2, label='B-spline Boundary')
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
ax.set_title('Cubic B-spline Velocity Model Structure (True Positions)')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.savefig('Bspline_model_structure_real.png', dpi=300)
plt.close()
time1 = time.perf_counter()
print(f" Step 1: B-spline setup time: {time1 - time0:.2f}s")
# After we know true_offset, fill perturbations into sem_config
sem_config['velocity']['perturbations'] = true_offset.tolist()
sem_config['velocity']['anomaly']['perturbations'] = true_offset.tolist()
# ========================================= step 2: Simulation grid & receivers ======================================
# Create SEM grid from configuration
global_coords, _, _ = create_global_mesh(
    xmin=sem_config['domain']['xmin'],
    xmax=sem_config['domain']['xmax'],
    zmin=sem_config['domain']['zmin'],
    zmax=sem_config['domain']['zmax'],
    nelem_x=sem_config['domain']['nelem_x'],
    nelem_z=sem_config['domain']['nelem_z'],
    npol=sem_config['method']['polynomial_order']
)
# Build layered velocity model with anomaly using true parameters
layers_cfg = sem_config['velocity']['layers']
ana_cfg = sem_config['velocity']['anomaly']
interfaces = np.asarray(layers_cfg['interfaces_z'], dtype=float)
vlay = np.asarray(layers_cfg['velocities'], dtype=float)
ctrl_true_anomaly = ctrl_pts_true  # true anomaly boundary
v_inside = float(ana_cfg['v_inside'])
tau = float(ana_cfg['tau'])
spls = int(ana_cfg['spline_samples'])
blend = str(ana_cfg.get('blend', 'smooth'))
# Create the velocity_params dictionary expected by SEMSimulation
velocity_params = {
    'vmin': float(sem_config['velocity']['vmin']),
    'vmax': float(sem_config['velocity']['vmax']),
    'control_points': ctrl_pts_true_base.tolist(),
    'perturbations': true_offset.tolist(),
    'inside_velocity': v_inside,
    'outside_velocity': vlay[-1],  # bottom layer velocity as outside
    'layers': {
        'interfaces_z': interfaces.tolist(),
        'velocities': vlay.tolist()
    }
}
# Add velocity_params to sem_config
sem_config['velocity_params'] = velocity_params
velocity_model, _, _ = build_velocity_layered_with_anomaly(
    nodes_xy=global_coords,
    interfaces_z=interfaces,
    layer_velocities=vlay,
    ctrl6_xy=ctrl_true_anomaly,
    v_inside=v_inside,
    tau=tau,
    samples=spls,
    newton_steps=7,
    blend=blend,
)
# Plot the velocity model
plt.figure(figsize=(10, 8))
x_coords = global_coords[:, 0]
z_coords = global_coords[:, 1]
scatter = plt.scatter(x_coords, z_coords, c=velocity_model, cmap='turbo', s=5, marker='s', linewidths=0)
plt.colorbar(scatter, label='Velocity (m/s)')
plt.plot(curve_points_true[:, 0], curve_points_true[:, 1], 'r-', lw=2, label='Boundary')
# Plot source positions
source_positions = sem_config['source']['positions']
for i, src_pos in enumerate(source_positions):
    plt.plot(src_pos[0], src_pos[1], 'k*', markersize=15,
             label='Sources' if i == 0 else "", markeredgecolor='red', markeredgewidth=1.0)
# Plot receiver positions
receiver_positions = np.asarray(sem_config['receivers']['positions'])
plt.plot(receiver_positions[:, 0], receiver_positions[:, 1], 'b^', markersize=8,
         label='Receivers', markeredgecolor='black', markeredgewidth=0.5)
# Plot true B-spline nodes
ctrl_pts_true_closed = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
plt.plot(ctrl_pts_true_closed[:, 0], ctrl_pts_true_closed[:, 1], 'ro--',
         markersize=8, label='B-spline Nodes', markeredgecolor='black', markeredgewidth=0.5)
xmin = sem_config['domain']['xmin']; xmax = sem_config['domain']['xmax']
zmin = sem_config['domain']['zmin']; zmax = sem_config['domain']['zmax']
plt.xlim(xmin, xmax)
plt.ylim(zmin, zmax)
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()  # top shallower like MCMC
plt.title('Layered Velocity Model with True Boundary')
plt.xlabel('X (m)'); plt.ylabel('Z (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('velocity_model_true.png', dpi=300)
plt.close()
# True parameter vector (2 * K1 dims = K1 control points × 2 coordinates)
true_params = torch.from_numpy(true_offset.reshape(-1)).double()  # Flattened 12-dim Gaussian offsets
# ====================================== step 3: SEM forward simulation (for obs) ====================================
def run_simulation(ctrl_params, noise_std):
    """Generate observed data with the multi-source SEM simulator (no backpropagation)."""
    if hasattr(ctrl_params, 'detach'):
        ctrl_params_np = ctrl_params.detach().cpu().numpy().reshape(-1, 2)
    else:
        ctrl_params_np = ctrl_params.reshape(-1, 2)
    # Update perturbations in config AND velocity_params
    sem_config['velocity']['anomaly']['perturbations'] = ctrl_params_np.tolist()
    sem_config['velocity']['perturbations'] = ctrl_params_np.tolist()
    sem_config['velocity_params']['perturbations'] = ctrl_params_np.tolist()
    waveforms_list = []
    dt = None
    nt = None
    source_positions = sem_config['source']['positions']
    for k_src, pos in enumerate(source_positions):
        sx, sz = map(float, pos)
        # Create config for this source
        cfg_k = build_single_source_cfg(sem_config, [sx, sz])
        sim = get_or_make_sim(cfg_k)
        results = sim.run()
        wf = results['receiver_data']  # (nt, nr)
        if dt is None:
            dt = float(results['dt'])
            nt = int(results['nt'])
        waveforms_list.append(wf)
    # Concatenate receiver traces from all sources
    if len(waveforms_list) == 1:
        data_all = waveforms_list[0]
    else:
        data_all = np.concatenate(waveforms_list, axis=1)  # (nt, n_sources * n_receivers)
    clean_data = torch.from_numpy(data_all).double()
    if noise_std is None or float(noise_std) == 0.0:
        noisy_data = clean_data.clone()
    else:
        noisy_data = clean_data + torch.normal(mean=torch.zeros_like(clean_data), std=noise_std)
    return noisy_data, clean_data, dt, nt

def build_obs_cache_signature():
    """Return a lightweight signature so stale cached observations are not reused silently."""
    return {
        'seed': int(SEED),
        'obs_noise_std': float(obs_noise_std),
        'domain': {
            'xmin': float(DOMAIN_XMIN),
            'xmax': float(DOMAIN_XMAX),
            'zmin': float(DOMAIN_ZMIN),
            'zmax': float(DOMAIN_ZMAX),
            'nelem_x': int(NELEM_X),
            'nelem_z': int(NELEM_Z),
        },
        'time': {
            'total_time': float(TOTAL_TIME),
            'dt': float(DT),
        },
        'source': {
            'positions': [[float(x), float(z)] for x, z in SOURCE_POSITIONS],
            'frequency': float(SOURCE_FREQUENCY),
            'amplitude': float(SOURCE_AMPLITUDE),
        },
        'receivers': {
            'positions': [[float(x), float(z)] for x, z in RECEIVER_POSITIONS],
        },
        'layers': {
            'interfaces_z': [float(v) for v in LAYER_INTERFACES_Z],
            'velocities': [float(v) for v in LAYER_VELOCITIES],
        },
        'anomaly': {
            'v_inside': float(ANOMALY_V_INSIDE),
            'tau': float(ANOMALY_TAU),
            'spline_samples': int(ANOMALY_SPLINE_SAMPLES),
            'blend': str(ANOMALY_BLEND),
        },
        'true_offset': np.asarray(true_offset, dtype=float).tolist(),
    }


def cache_signature_matches(meta, expected):
    return isinstance(meta, dict) and meta.get('signature') == expected


# ======================= Observations cache (shared across stages) =======================
_noisy_path = os.path.join(OBS_CACHE_DIR, "noisy_data_real.npy")
_clean_path = os.path.join(OBS_CACHE_DIR, "clean_data_real.npy")
_meta_path  = os.path.join(OBS_CACHE_DIR, "obs_meta.json")
_obs_signature = build_obs_cache_signature()
_use_cached_obs = False

if os.path.exists(_noisy_path) and os.path.exists(_clean_path) and os.path.exists(_meta_path):
    try:
        with open(_meta_path, "r") as _f:
            _meta = json.load(_f)
        _use_cached_obs = cache_signature_matches(_meta, _obs_signature)
        if _use_cached_obs:
            print(f"Loading cached observations from: {OBS_CACHE_DIR}")
            noisy_obs = torch.from_numpy(np.load(_noisy_path)).double()
            clean_obs = torch.from_numpy(np.load(_clean_path)).double()
            obs_dt = float(_meta["obs_dt"])
            obs_nt = int(_meta["obs_nt"])
        else:
            print(f"Cached observations in {OBS_CACHE_DIR} do not match the current configuration. Regenerating.")
    except Exception as _e:
        print(f"Failed to read cached observations cleanly ({_e}). Regenerating.")
        _use_cached_obs = False

if not _use_cached_obs:
    print("Generating observed data with SEM simulator (this may take some seconds)...")
    noisy_obs, clean_obs, actual_dt, actual_nt = run_simulation(true_params, noise_std=obs_noise_std)
    obs_dt = actual_dt
    obs_nt = actual_nt
    np.save(_noisy_path, noisy_obs.numpy())
    np.save(_clean_path, clean_obs.numpy())
    with open(_meta_path, "w") as _f:
        json.dump(
            {
                "obs_dt": float(obs_dt),
                "obs_nt": int(obs_nt),
                "signature": _obs_signature,
            },
            _f,
            indent=2,
        )
    print(f"SEM observations cached. dt={obs_dt:.6f}, nt={obs_nt}")
time2 = time.perf_counter()
print(f"Step 2: SEM sim setup time: {time2 - time1:.2f}s")
time_axis = np.arange(obs_nt) * obs_dt
n_traces = noisy_obs.shape[1]
plt.figure(figsize=(15, 3 * min(n_traces, 20)))
for i in range(min(n_traces, 20)):
    plt.subplot(min(n_traces, 20), 1, i+1)
    plt.plot(time_axis, clean_obs[:, i].numpy(), 'k-', lw=1.5, label='Clean')
    plt.plot(time_axis, noisy_obs[:, i].numpy(), 'r-', lw=1.0, alpha=0.7, label='Noisy')
    plt.title(f'Trace {i+1}')
    plt.xlabel('Time(s)'); plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.savefig('real_noise_comparison.png', dpi=300)
plt.close()
time3 = time.perf_counter()
print(f"Step 3: generated observations. Time: {time3 - time2:.2f}s")
# ================================ Step 4: Custom Autograd Function to connect SEM adjoint gradient to Normalizing Flow ==================
class SEMLikelihoodAdjointFn(torch.autograd.Function):
    """
    Forward: call SEM forward + adjoint to return log-likelihood (scalar).
    Backward: inject SEM adjoint d logLik / d z into z's gradient (propagates into NF parameters).
    Requires sem_waveform.core.SEMSimulation to implement run_forward_and_adjoint.
    """
    @staticmethod
    def forward(ctx, z, y_obs, noise_std, sem_cfg_dict, obs_dt, obs_nt):
        # z: (DIM_Z,), y_obs: (nt, n_traces)
        if z.dim() != 1:
            raise ValueError(
                f"SEMLikelihoodAdjointFn expects a 1D z, got shape {z.shape}"
            )
        # 1) Turn z into absolute control points (offset from original)
        z_off = z.detach().cpu().numpy().reshape(-1, 2)
        abs_ctrl = ctrl_pts_init_base + z_off  # use K2-node INITIAL / inversion model
        # 2) Build base cfg for SEM simulator (independent of this z)
        # We deliberately do NOT write the NF sample (z_off) into the config.
        # All dependence on z enters only through the 'bspline_ctrl' argument
        # passed to run_forward_and_adjoint, so that the cached SEMSimulation
        # instances are re-used safely across different z samples.
        cfg_base = copy.deepcopy(sem_cfg_dict)
        # 3) Observation + noise
        y_obs_np = y_obs.detach().cpu().numpy()
        if isinstance(noise_std, torch.Tensor):
            noise_var = float(noise_std.detach().cpu().item() ** 2) + EPS
        else:
            noise_var = float(noise_std) ** 2 + EPS
        # 4) Multi-source bookkeeping (always multi-source / multi-receiver)
        src_positions = cfg_base['source'].get('positions', None)
        if not src_positions:
            raise ValueError(
                "SEMLikelihoodAdjointFn: 'source.positions' must be provided and non-empty."
            )
        n_traces = y_obs_np.shape[1]
        nrec = len(cfg_base['receivers']['positions'])
        nsrc = len(src_positions)
        if n_traces != nsrc * nrec:
            raise ValueError(
                f"Multi-source likelihood: expected {nsrc * nrec} traces "
                f"({nsrc} sources * {nrec} receivers) but got {n_traces}."
            )
        total_loglik = 0.0
        grad_wrt_ctrl_accum = np.zeros(z.numel(), dtype=np.float32)
        dt_sim = None
        nt_sim = None
        # Parallel helper for a single source (kept inside forward so outer structure stays unchanged)
        def _run_one_source(k_pos):
            k, pos = k_pos
            start_col = k * nrec
            end_col = (k + 1) * nrec
            y_obs_k = y_obs_np[:, start_col:end_col]
            cfg_k = build_single_source_cfg(cfg_base, pos)
            sim_k = get_or_make_sim(cfg_k)
            if not hasattr(sim_k, "run_forward_and_adjoint"):
                raise NotImplementedError(
                    "SEMSimulation.run_forward_and_adjoint(...) is not implemented. "
                    "Please implement this method in sem_waveform/core.py so that it returns: "
                    "{'loglik': float, 'grad_wrt_ctrl': (DIM_Z,), 'dt': float, 'nt': int}"
                )
            out_k = sim_k.run_forward_and_adjoint({
                'bspline_ctrl': abs_ctrl,
                'y_obs': y_obs_k,
                'obs_dt': float(obs_dt),
                'obs_nt': int(obs_nt),
                'noise_std': np.sqrt(noise_var)
            })
            return (
                float(out_k["loglik"]),
                np.asarray(out_k["grad_wrt_ctrl"], dtype=np.float32).reshape(-1),
                out_k["dt"],
                out_k["nt"],
            )
        from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
        tasks = list(enumerate(src_positions))
        max_workers = min(len(tasks), 3) if tasks else 0
        if max_workers < 1:
            raise ValueError(
                "SEMLikelihoodAdjointFn: 'source.positions' must be provided and non-empty."
            )
        with _ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run_one_source, tasks))
        for loglik_k, grad_k, dt_k, nt_k in results:
            total_loglik += loglik_k
            grad_wrt_ctrl_accum += grad_k
            if dt_sim is None:
                dt_sim = dt_k
                nt_sim = nt_k
            else:
                if (abs(dt_sim - dt_k) > 1e-12) or (nt_sim != nt_k):
                    raise ValueError("Inconsistent dt/nt between source runs")
        # All source blocks are kept only for consistency checks above.
        # No combined clean-data array is needed downstream here.
        # 5) Save gradient for backward
        ctx.save_for_backward(torch.from_numpy(grad_wrt_ctrl_accum).to(z.device, dtype=torch.float32))
        # Return scalar log-likelihood
        return torch.tensor(total_loglik, dtype=torch.float32, device=z.device)
    @staticmethod
    def backward(ctx, grad_output):
        (grad_wrt_ctrl_tensor,) = ctx.saved_tensors  # shape: (DIM_Z,)
        # Multiply upstream gradient with provided d logLik / d z
        grad_z = grad_output * grad_wrt_ctrl_tensor
        # No gradient needed for y_obs, noise_std, sem_cfg_dict
        return grad_z, None, None, None, None, None
def sem_loglik_with_adjoint(z, y_obs, noise_std, obs_dt, obs_nt):
    """Thin wrapper around the SEM adjoint likelihood call."""
    # Evaluate the SEM adjoint log-likelihood under the current configuration.
    result = SEMLikelihoodAdjointFn.apply(z, y_obs, noise_std, sem_config, obs_dt, obs_nt)
    return result
# =================================== step 5: ActNorm ========================================
class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.logs = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.initialized = False
    def initialize_parameters(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=0)
            # correction=0 (unbiased=False)
            if x.shape[0] == 1:
                std = torch.zeros_like(mean)
            else:
                std = torch.std(x, dim=0, unbiased=False)
            std = torch.where(std < 1e-6, torch.ones_like(std) * 1e-6, std)
            self.bias.data = (-mean).clone()
            self.logs.data = torch.log(1.0 / (std.clone() + 1e-12))
            self.initialized = True
    def forward(self, x):
        if not self.initialized:
            self.initialize_parameters(x)
        y = (x + self.bias.unsqueeze(0)) * torch.exp(self.logs.unsqueeze(0))
        logabsdet = torch.sum(self.logs) * torch.ones(x.shape[0], dtype=torch.float64)
        return y, logabsdet
    def inverse(self, y):
        x = y * torch.exp(-self.logs.unsqueeze(0)) - self.bias.unsqueeze(0)
        logabsdet = -torch.sum(self.logs) * torch.ones(y.shape[0], dtype=torch.float64)
        return x, logabsdet
class FixedPermutation(nn.Module):
    """
    Fixed random permutation。
    """
    def __init__(self, dim, seed=12345):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(dim, generator=g)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(dim)
        self.register_buffer('perm', perm.to(torch.long))
        self.register_buffer('inv', inv.to(torch.long))
    def forward(self, x):
        return x[:, self.perm], torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    def inverse(self, y):
        return y[:, self.inv], torch.zeros(y.shape[0], dtype=y.dtype, device=y.device)
# ================== Piecewise Rational Quadratic Coupling (NSF-style) ==================
# Rational-quadratic spline coupling utilities
def _searchsorted(bin_locations, inputs):
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
def _normalize_bin_params(unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                          min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
    # widths / heights -> simplex
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * widths.shape[-1]) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = cumwidths / cumwidths[..., -1:].clamp_min(1e-12)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * heights.shape[-1]) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = cumheights / cumheights[..., -1:].clamp_min(1e-12)
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    return widths, cumwidths, heights, cumheights, derivatives
def _rqs_transform(inputs, widths, cumwidths, heights, cumheights, derivatives,
                   inverse=False, left=-3.0, right=3.0, bottom=-3.0, top=3.0,
                   newton_iters=8, eps=1e-12):
    """
    Rational-quadratic spline transform.
    Conventions:
        inputs      : (N, 1)
        widths      : (N, K)      bin width
        cumwidths   : (N, K+1)    row sum = 1
        heights     : (N, K)      bin height
        cumheights  : (N, K+1)    height sum = 1
        derivatives : (N, K+1)    node derivative（>= min_derivative）
    Returns:
        outputs   : (N, 1)
        logabsdet : (N, 1)
    """
    # output
    outputs = inputs.clone()
    logabsdet = torch.zeros_like(inputs)
    # ---------- forward：x -> y ----------
    if not inverse:
        # x
        inside = ((inputs >= left) & (inputs <= right)).squeeze(-1)
        if not torch.any(inside):
            return outputs, logabsdet
        x = inputs[inside, 0]
        # normalization [0,1]
        x_scaled = (x - left) / (right - left)
        x_scaled = x_scaled.clamp(0.0, 1.0)
        cw_rows = cumwidths[inside]     # (M, K+1)
        w_rows = widths[inside]         # (M, K)
        ch_rows = cumheights[inside]    # (M, K+1)
        h_rows = heights[inside]        # (M, K)
        d_rows = derivatives[inside]    # (M, K+1)
        bin_ids = _searchsorted(cw_rows, x_scaled).clamp(min=0, max=w_rows.shape[-1] - 1)
        gather = bin_ids.unsqueeze(-1)  # (M,1)
        xk = cw_rows.gather(-1, gather).squeeze(-1)                  # (M,)
        wk = w_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        yk = ch_rows.gather(-1, gather).squeeze(-1)                  # (M,)
        hk = h_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        dk = d_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        dk1 = d_rows.gather(-1, (bin_ids + 1).unsqueeze(-1)).squeeze(-1)
        # t ∈ [0,1]
        t = ((x_scaled - xk) / (wk + eps)).clamp(0.0, 1.0)
        # a,b,c（Durkan RQ-NSF）
        a = (hk + eps) / (wk + eps)   # slope ratio Δ = h/w
        b = dk
        c = dk1
        # s(t) = (a t^2 + b t (1-t)) / (a + (b + c - 2a) t (1-t))
        num = a * t * t + b * t * (1.0 - t)
        den = a + (b + c - 2.0 * a) * t * (1.0 - t)
        s = num / (den + eps)
        # y_scaled = yk + hk * s
        y_scaled = yk + hk * s
        y = y_scaled * (top - bottom) + bottom
        outputs[inside, 0] = y
        # dy/dx = (top-bottom)/(right-left) * [a^2 * (c t^2 + 2a t(1-t) + b (1-t)^2)] / [a + (b+c-2a) t(1-t)]^2
        deriv_num = (a * a) * (c * t * t + 2.0 * a * t * (1.0 - t) + b * (1.0 - t) * (1.0 - t))
        deriv_den = (den * den) + eps
        dydx = (top - bottom) / (right - left) * deriv_num / deriv_den
        logabsdet[inside, 0] = torch.log(dydx.clamp_min(1e-12))
        return outputs, logabsdet
    # ---------- inverse：y -> x ----------
    else:
        inside = ((inputs >= bottom) & (inputs <= top)).squeeze(-1)
        if not torch.any(inside):
            return outputs, logabsdet
        y = inputs[inside, 0]
        y_scaled = (y - bottom) / (top - bottom)
        y_scaled = y_scaled.clamp(0.0, 1.0)
        cw_rows = cumwidths[inside]     # (M, K+1)
        w_rows = widths[inside]         # (M, K)
        ch_rows = cumheights[inside]    # (M, K+1)
        h_rows = heights[inside]        # (M, K)
        d_rows = derivatives[inside]    # (M, K+1)
        # bin
        bin_ids = _searchsorted(ch_rows, y_scaled).clamp(min=0, max=h_rows.shape[-1] - 1)
        gather = bin_ids.unsqueeze(-1)
        xk = cw_rows.gather(-1, gather).squeeze(-1)                  # (M,)
        wk = w_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        yk = ch_rows.gather(-1, gather).squeeze(-1)                  # (M,)
        hk = h_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        dk = d_rows.gather(-1, gather).squeeze(-1)                   # (M,)
        dk1 = d_rows.gather(-1, (bin_ids + 1).unsqueeze(-1)).squeeze(-1)
        a = (hk + eps) / (wk + eps)
        b = dk
        c = dk1
        # s = (y_scaled - yk) / hk ∈ [0,1]
        s = ((y_scaled - yk) / (hk + eps)).clamp(0.0, 1.0)
        # Newton iteration: [0,1], t：  f(t) = s(t) - s = 0
        # s(t) = num/den，num=a t^2 + b t(1-t), den=a + (b+c-2a) t(1-t)
        t = s.clone()  # initial
        for _ in range(newton_iters):
            num = a * t * t + b * t * (1.0 - t)
            den = a + (b + c - 2.0 * a) * t * (1.0 - t)
            f = num / (den + eps) - s
            # f'(t) = (num' * den - num * den') / den^2
            num_t = 2.0 * a * t + b * (1.0 - 2.0 * t)
            den_t = (b + c - 2.0 * a) * (1.0 - 2.0 * t)
            fp = (num_t * den - num * den_t) / ((den * den) + eps)
            t = (t - f / (fp + eps)).clamp(0.0, 1.0)
        # t -> x
        x_scaled = xk + t * wk
        x_back = x_scaled * (right - left) + left
        outputs[inside, 0] = x_back
        # log|det| = -log(dy/dx)
        den = a + (b + c - 2.0 * a) * t * (1.0 - t)
        deriv_num = (a * a) * (c * t * t + 2.0 * a * t * (1.0 - t) + b * (1.0 - t) * (1.0 - t))
        deriv_den = (den * den) + eps
        dydx = (top - bottom) / (right - left) * deriv_num / deriv_den
        logabsdet[inside, 0] = -torch.log(dydx.clamp_min(1e-12))
        return outputs, logabsdet
class PiecewiseRationalQuadraticCoupling(nn.Module):
    """
    Monotonic rational-quadratic spline coupling (NSF style) with linear tails on [-B, B].
    """
    def __init__(self, mask, transform_net_create_fn, num_bins=8, tails="linear", tail_bound=10.0,
                 min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        super().__init__()
        self.register_buffer('mask', mask.clone().detach().double())
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = float(tail_bound)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        in_features = int((self.mask <= 0).sum().item())
        out_features = int((self.mask > 0).sum().item())
        # (w, h, d_left, d_right)：bin = K
        # Parameter = 2K + (K+1) = 3K + 1
        self.param_multiplier = 3 * self.num_bins + 1
        self.transform_net = transform_net_create_fn(max(1, in_features), out_features * self.param_multiplier)
    def _split_params(self, params, out_dim):
        # params: (B, out_dim * (3K+1)) → (B, out_dim, 3K+1)
        p = params.view(params.shape[0], out_dim, self.param_multiplier)
        K = self.num_bins
        un_w = p[..., :K]
        un_h = p[..., K:2 * K]
        un_d_all = p[..., 2 * K:]      # (B, out_dim, K+1)
        return un_w, un_h, un_d_all
    def _rqs(self, to_transform, cond_params, inverse=False):
        B = self.tail_bound
        left, right, bottom, top = -B, B, -B, B
        un_w, un_h, un_d_all = cond_params
        widths, cumwidths, heights, cumheights, derivatives = _normalize_bin_params(
            un_w, un_h, un_d_all, self.min_bin_width, self.min_bin_height, self.min_derivative
        )
        # Parameter broadcast -> (batch, out_dim, K)
        x = to_transform
        # util -> (B*out_dim, K)
        Bsz, D = x.shape
        K = self.num_bins
        reshape2 = lambda t: t.reshape(Bsz * D, -1)
        reshape_back = lambda t: t.reshape(Bsz, D)
        w = reshape2(widths); cw = reshape2(cumwidths)
        h = reshape2(heights); ch = reshape2(cumheights)
        d = reshape2(derivatives)
        xin = reshape2(x)
        yout, lad = _rqs_transform(xin, w, cw, h, ch, d, inverse=inverse,
                                   left=left, right=right, bottom=bottom, top=top)
        return reshape_back(yout), reshape_back(lad)
    def forward(self, inputs):
        identity = inputs[:, self.mask <= 0]
        to_transform = inputs[:, self.mask > 0]
        out_dim = to_transform.shape[1]
        if out_dim == 0:
            return inputs, torch.zeros(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
        if identity.shape[1] == 0:
            dummy = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
            params = self.transform_net(dummy)
        else:
            params = self.transform_net(identity)
        cond_params = self._split_params(params, out_dim)
        y, logabsdet = self._rqs(to_transform, cond_params, inverse=False)
        outputs = inputs.clone()
        outputs[:, self.mask > 0] = y
        return outputs, torch.sum(logabsdet, dim=1)
    def inverse(self, inputs):
        identity = inputs[:, self.mask <= 0]
        to_transform = inputs[:, self.mask > 0]
        out_dim = to_transform.shape[1]
        if out_dim == 0:
            return inputs, torch.zeros(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
        if identity.shape[1] == 0:
            dummy = torch.zeros(inputs.shape[0], 1, dtype=inputs.dtype, device=inputs.device)
            params = self.transform_net(dummy)
        else:
            params = self.transform_net(identity)
        cond_params = self._split_params(params, out_dim)
        x, logabsdet = self._rqs(to_transform, cond_params, inverse=True)
        outputs = inputs.clone()
        outputs[:, self.mask > 0] = x
        return outputs, torch.sum(logabsdet, dim=1)
# ================== NormalizingFlow (uses ActNorm + rational-quadratic spline coupling) ==================
class NormalizingFlow(nn.Module):
    def __init__(self, dim=12, num_flows=6, base_mean=None, base_cov=None):
        super().__init__()
        self.dim = dim
        if base_mean is None:
            base_mean = torch.zeros(dim)
        if base_cov is None:
            base_cov = torch.eye(dim)
        self.register_buffer('base_mean', base_mean.double())
        self.register_buffer('base_cov', base_cov.double())
        self.register_buffer('L_tril', torch.linalg.cholesky(base_cov.double()))
        self.actnorms = nn.ModuleList()
        self.couplings = nn.ModuleList()
        self.perms = nn.ModuleList()
        for i in range(num_flows):
            self.actnorms.append(ActNorm(dim))
            mask = torch.ones(dim, dtype=torch.double)
            mask[::2] = 0.0
            if i % 2 == 1:
                mask = 1.0 - mask
            def create_net(in_features, out_features):
                net = nn.Sequential(
                    nn.Linear(int(in_features), 128, dtype=torch.float64),
                    nn.ReLU(),
                    nn.Linear(128, 128, dtype=torch.float64),
                    nn.ReLU(),
                    nn.Linear(128, int(out_features), dtype=torch.float64)
                )
                for layer in net:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                        nn.init.constant_(layer.bias, 0.0)
                return net
            # Rational-quadratic spline coupling layer (NSF style).
            coupling = PiecewiseRationalQuadraticCoupling(
                mask=mask,
                transform_net_create_fn=create_net,
                num_bins=K_bins,
                tails="linear",
                tail_bound=tstd * 1.5
            )
            self.couplings.append(coupling)
            if i < num_flows - 1:
                self.perms.append(FixedPermutation(dim, seed=12345 + i))
        assert len(self.perms) == max(0, num_flows - 1)
    def forward(self, n_samples=1):
        eps = torch.randn(n_samples, self.dim, dtype=torch.float64)
        base_samples = self.base_mean.unsqueeze(0) + eps @ self.L_tril.T
        x = base_samples
        log_det_total = torch.zeros(n_samples, dtype=torch.float64)
        for layer_idx, (act, cp) in enumerate(zip(self.actnorms, self.couplings)):
            x, lnd1 = act.forward(x); log_det_total = log_det_total + lnd1
            x, lnd2 = cp.forward(x);  log_det_total = log_det_total + lnd2
            if layer_idx < len(self.perms):           # <<<
                x, lnd3 = self.perms[layer_idx].forward(x)
                log_det_total = log_det_total + lnd3  # logdet = 0
        return x, log_det_total
    def log_prob(self, z):
        # inverse：Coupling and ActNorm
        L = len(self.couplings)
        x = z.clone()
        log_det_total = torch.zeros(z.shape[0], dtype=torch.float64)
        for layer_idx in reversed(range(L)):
            # 1) Final layer: Coupling + ActNorm
            cp = self.couplings[layer_idx]
            act = self.actnorms[layer_idx]
            x, lnd1 = cp.inverse(x);  log_det_total = log_det_total + lnd1
            x, lnd2 = act.inverse(x); log_det_total = log_det_total + lnd2
            # 2) Other layer：perm -> layer_idx-1
            perm_idx = layer_idx - 1
            if perm_idx >= 0:
                x, lndp = self.perms[perm_idx].inverse(x)  # logdet = 0
                log_det_total = log_det_total + lndp
        base = MultivariateNormal(self.base_mean, covariance_matrix=self.base_cov)
        log_base = base.log_prob(x)
        return log_base + log_det_total
# ================== ELBO (multi-source likelihood via SEM adjoint) ==================
def compute_elbo(model, y_obs, noise_std,
                 stage=1,
                 current_epoch=None, total_epochs=None,
                 min_samples=8, max_samples=16):
    """Compute the ELBO for the current two-stage fixed-sigma inversion.

    The likelihood is evaluated with the SEM adjoint backend.
    The prior on z uses a stage-dependent fixed Gaussian scale.
    An optional fused penalty is applied to adjacent control-point offsets.
    """
    if (current_epoch is not None) and (total_epochs is not None) and (total_epochs > 1):
        t = current_epoch / (total_epochs - 1)
        n_samples = int(round((1 - t) * min_samples + t * max_samples))
    else:
        n_samples = int(min_samples)

    z_samples, _ = model.forward(n_samples=n_samples)
    log_q_z = model.log_prob(z_samples).unsqueeze(1)

    if float(FUSE_LAMBDA) > 0.0:
        z_pairs = z_samples.view(n_samples, K2, 2)
        idx_next = (torch.arange(K2, device=z_pairs.device) + 1) % K2
        diffs = z_pairs[:, idx_next, :] - z_pairs
        mags = torch.sqrt(torch.sum(diffs * diffs, dim=2) + (float(FUSE_EPS) ** 2))
        fuse_pen = torch.sum(mags, dim=1, keepdim=True)
        log_p_fuse = -float(FUSE_LAMBDA_EFF) * fuse_pen
    else:
        log_p_fuse = torch.zeros_like(log_q_z)

    if int(stage) == 1:
        sigma_k = torch.full(
            (K2,),
            float(FIXED_SIGMA_STAGE1),
            device=z_samples.device,
            dtype=z_samples.dtype,
        )
    elif int(stage) == 2:
        sigma_k = torch.full(
            (K2,),
            float(FIXED_SIGMA_STAGE2),
            device=z_samples.device,
            dtype=z_samples.dtype,
        )
    else:
        raise ValueError(f"Invalid stage={stage}. This script supports only Stage-1 and Stage-2.")

    sigma_vec = sigma_k.repeat_interleave(2)
    prior_dist = torch.distributions.Normal(loc=torch.zeros_like(sigma_vec), scale=sigma_vec)
    log_p_z = torch.sum(prior_dist.log_prob(z_samples), dim=1)

    log_p_y_given_z = []
    for i in range(n_samples):
        z_i = z_samples[i]
        ll_i = sem_loglik_with_adjoint(z_i, y_obs, noise_std, obs_dt, obs_nt)
        log_p_y_given_z.append(ll_i)
    log_p_y_given_z = torch.stack(log_p_y_given_z, dim=0).unsqueeze(1)

    elbo_terms = log_p_y_given_z + log_p_z + log_p_fuse - log_q_z
    elbo = torch.mean(elbo_terms)
    return (
        elbo,
        torch.mean(log_p_y_given_z),
        torch.mean(log_p_z),
        torch.mean(log_p_fuse),
        torch.mean(log_q_z),
    )
# ================== Gradient monitoring functions ==================
def get_gradient_stats(model):
    """Collect gradient statistics."""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'norm': grad.norm().item()
            }
    return grad_stats
def plot_gradient_history(gradient_history, filename='gradient_history_real.png'):
    """Plot gradient history."""
    if not gradient_history:
        return
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot([gh['total_norm'] for gh in gradient_history])
    plt.title('Total Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 2)
    plt.plot([gh['max_grad'] for gh in gradient_history])
    plt.title('Max Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.plot([gh['mean_grad'] for gh in gradient_history])
    plt.title('Mean Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 4)
    plt.plot([gh['gradient_exploded'] for gh in gradient_history])
    plt.title('Gradient Explosion Events')
    plt.xlabel('Epoch')
    plt.ylabel('Exploded (1=Yes)')
    plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 5)
    if 'layer_norms' in gradient_history[0]:
        layer_names = list(gradient_history[0]['layer_norms'].keys())
        for layer in layer_names:
            norms = [gh['layer_norms'][layer] for gh in gradient_history if layer in gh['layer_norms']]
            plt.plot(norms, label=layer)
        plt.title('Gradient Norms by Layer')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        plt.yscale('log')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    plt.subplot(2, 3, 6)
    clipped_ratio = [gh.get('clipped_ratio', 0) for gh in gradient_history]
    plt.plot(clipped_ratio)
    plt.title('Gradient Clipping Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Clipped Ratio')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
# ========================================== Step 6: Training setup =================================
# The z-space prior is centered at zero in both stages.
# Stage-2 changes the geometric prior center by updating ctrl_pts_init_base.
prior_mean = torch.zeros(DIM_Z, dtype=torch.float64)
base_mean = torch.zeros(DIM_Z, dtype=torch.float64)
base_cov = torch.eye(DIM_Z, dtype=torch.float64)
model = NormalizingFlow(dim=DIM_Z, num_flows=num_flows, base_mean=base_mean, base_cov=base_cov)

print(f"[STAGE-{STAGE_INT}] Fixed sigma = {(FIXED_SIGMA_STAGE1 if STAGE_INT == 1 else FIXED_SIGMA_STAGE2):g}")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Remove only the old top-level legacy directory from older script versions.
# Do not remove the stage-specific snapshot directory used below, otherwise
# the periodic posterior images saved every PLOT_EVERY epochs will disappear.
_legacy_dir = "intermediate_posteriors_real"
if os.path.isdir(_legacy_dir):
    shutil.rmtree(_legacy_dir, ignore_errors=True)

elbo_history = []
log_likelihood_history = []
log_prior_z_history = []
log_q_history = []
log_p_fuse_history = []
gradient_history = []

def plot_posterior_samples(epoch, post_samples_np, save_path=None):
    """Plot the true boundary, prior mean boundary, and posterior sample cloud.

    The save_path argument is optional. When provided, the figure is written to disk;
    otherwise it is simply rendered and closed.
    """
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    # TRUE boundary (K1 control points)
    ctrl_true_closed = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
    _, _, curve_true = build_closed_bspline(ctrl_pts_true)
    ax.plot(ctrl_true_closed[:, 0], ctrl_true_closed[:, 1], 'ro--', markersize=8, label='True Control Points')
    ax.plot(curve_true[:, 0], curve_true[:, 1], 'r-', lw=2, label='True Boundary')
    # PRIOR mean boundary in z-space (zero offset around ctrl_pts_init_base)
    prior_mean_np = prior_mean.cpu().numpy().reshape(-1, 2)  # (K2, 2)
    ctrl_pts_prior = ctrl_pts_init_base + prior_mean_np
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0:1]])
    ax.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], 'ko--', markersize=8, alpha=0.7,
            label='Prior Mean Points')
    _, _, curve_prior = build_closed_bspline(ctrl_pts_prior)
    ax.plot(curve_prior[:, 0], curve_prior[:, 1], 'k-', lw=2, alpha=0.7, label='Prior Mean Boundary')
    # Posterior sample boundaries (K2 nodes)
    post_samples_np = np.asarray(post_samples_np)
    nplot = min(200, post_samples_np.shape[0])
    for i in range(nplot):
        offset_i = post_samples_np[i].reshape(-1, 2)  # (K2, 2)
        ctrl_pts_i = ctrl_pts_init_base + offset_i
        ctrl_closed_i = np.vstack([ctrl_pts_i, ctrl_pts_i[0:1]])
        _, _, curve_i = build_closed_bspline(ctrl_pts_i)
        ax.plot(ctrl_pts_i[:, 0], ctrl_pts_i[:, 1], 'bo', markersize=4, alpha=0.08)
        ax.plot(curve_i[:, 0], curve_i[:, 1], 'b--', lw=0.5, alpha=0.02)
    # Posterior mean boundary (K2 nodes)
    if post_samples_np.size > 0:
        mean_sample = post_samples_np.mean(axis=0).reshape(-1, 2)
        mean_ctrl_pts = ctrl_pts_init_base + mean_sample
        mean_ctrl_closed = np.vstack([mean_ctrl_pts, mean_ctrl_pts[0:1]])
        _, _, curve_mean = build_closed_bspline(mean_ctrl_pts)
        ax.plot(mean_ctrl_closed[:, 0], mean_ctrl_closed[:, 1], 'bo--', markersize=8, alpha=0.9,
                label='Posterior Mean Points')
        ax.plot(curve_mean[:, 0], curve_mean[:, 1], 'b-', lw=2, alpha=0.9,
                label='Posterior Mean Boundary')
    # Legend proxy artists for posterior samples and the prior mean.
    proxy_post = Patch(facecolor='blue', alpha=0.3, label='Posterior Samples')
    proxy_prior = Patch(facecolor='black', alpha=0.3, label='Prior Mean')
    handles, labels = ax.get_legend_handles_labels()
    custom_handles = [proxy_post, proxy_prior] + handles[-8:]
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f'VI Results at Epoch {epoch}\n(True=Red, Posterior=Blue, Prior=Black, Mean=Blue)')
    ax.legend(handles=custom_handles)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # top shallower like MCMC
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160)
    plt.close()
# ================================= Step 7: Training loop and stage handoff =====================
print("Starting training...")
outdir = f"stage{STAGE}_K_{K2:02d}"
os.makedirs(outdir, exist_ok=True)
snapshot_dir = os.path.join(outdir, "intermediate_posteriors_real")
os.makedirs(snapshot_dir, exist_ok=True)
# ======================= Stage-2: load base control points (prior mean geometry) =======================
# Design:
#   - Stage-2 re-centers the geometric prior by replacing ctrl_pts_init_base
#     with Stage-1 posterior-mean or pruned control points.
#   - By default K2 starts from K_MAX, but auto-pruning/override may reduce it.
if STAGE_INT == 2:
    _base_path = BASE_CTRL_PATH
    if _base_path == "":
        if STAGE_INT == 2:
            # Default for Stage-2: use the Stage-1 posterior mean first,
            # then the pruned control points, then older legacy filenames.
            _candidates = [
                os.path.join(f"stage1_K_{K_MAX:02d}", "stage1_mean_ctrl_pts.npy"),
                os.path.join(f"stage1_K_{K_MAX:02d}", "stage1_pruned_ctrl_pts.npy"),
                os.path.join(f"stage1_K_{K_MAX:02d}", "kept_ctrl_pts.npy"),
                os.path.join(f"stage1_K_{K_MAX:02d}", "merged_ctrl_pts.npy"),
            ]
        _base_path = next((p for p in _candidates if os.path.exists(p)), "")
    if (not _base_path) or (not os.path.exists(_base_path)):
        if STAGE_INT == 2:
            raise FileNotFoundError(
                f"[STAGE-2] BASE_CTRL_PATH not found: {_base_path}. "
                f"Set BASE_CTRL_PATH to Stage-1's stage1_mean_ctrl_pts.npy (preferred), "
                f"stage1_pruned_ctrl_pts.npy, or an older legacy file from Stage-1."
            )
    _base = np.load(_base_path)
    if _base.ndim != 2 or _base.shape[1] != 2:
        raise ValueError(f"[STAGE-{STAGE_INT}] Invalid base ctrl pts shape: {_base.shape} (expected (K2,2))")
    if _base.shape[0] != K2:
        raise ValueError(
            f"[STAGE-{STAGE_INT}] K2 mismatch: env K2={K2} but BASE_CTRL has K={_base.shape[0]}. "
            f"Set K2_OVERRIDE={_base.shape[0]} or provide a matching base."
        )
    ctrl_pts_init_base = _base.astype(float)
    print(f"[STAGE-{STAGE_INT}] Loaded ctrl_pts_init_base from: {_base_path}  (K2={K2})")
# For speed, disable heavy projection tests by default.
# Set VERIFY_PROJECTION_DEBUG=True if you want periodic adjoint/FD checks (much slower).
VERIFY_PROJECTION_DEBUG = False
grdnum = 200  # projection/grad test interval
last_print_time = time.perf_counter()
for epoch in range(num_epochs):
    if VERIFY_PROJECTION_DEBUG:
        sem_config['method']['VERIFY_PROJECTION'] = (epoch % grdnum == 0)
    else:
        sem_config['method']['VERIFY_PROJECTION'] = False
    optimizer.zero_grad()
    elbo, log_likelihood, log_p_z, log_p_fuse, log_q_z = compute_elbo(
        model, noisy_obs, obs_noise_std,
        stage=STAGE_INT,
        current_epoch=epoch, total_epochs=num_epochs,
        min_samples=min_elbo_samples, max_samples=max_elbo_samples
    )
    loss = -elbo
    loss.backward()
    # Gradient clipping (robustness)
    params = [p for p in model.parameters() if getattr(p, "grad", None) is not None]
    if clip_gradient and params:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in params]))
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-12)
            for p in params:
                p.grad.detach().mul_(scale)
    else:
        total_norm = torch.tensor(0.0, dtype=torch.float64)
    optimizer.step()
    # Record histories (Stage-1 and Stage-2)
    elbo_history.append(float(elbo.detach().cpu()))
    log_likelihood_history.append(float(log_likelihood.detach().cpu()))
    log_prior_z_history.append(float(log_p_z.detach().cpu()))
    log_p_fuse_history.append(float(log_p_fuse.detach().cpu()))
    log_q_history.append(float(log_q_z.detach().cpu()))
    if monitor_gradient:
        _grad_stats = get_gradient_stats(model)
        if _grad_stats:
            _layer_norms = {name: stats['norm'] for name, stats in _grad_stats.items()}
            _total_norm_val = float(total_norm.detach().cpu()) if isinstance(total_norm, torch.Tensor) else float(total_norm)
            _max_grad_val = max(stats['max'] for stats in _grad_stats.values())
            _mean_grad_val = float(np.mean([stats['mean'] for stats in _grad_stats.values()]))
            gradient_history.append({
                'total_norm': _total_norm_val,
                'max_grad': _max_grad_val,
                'mean_grad': _mean_grad_val,
                'gradient_exploded': int(_total_norm_val > max_grad_norm),
                'layer_norms': _layer_norms,
                'clipped_ratio': float(min(1.0, max_grad_norm / (_total_norm_val + 1e-12))) if clip_gradient and _total_norm_val > 0 else 0.0,
            })
    if (epoch % PRINT_EVERY == 0) or (epoch == num_epochs - 1):
        now = time.perf_counter()
        dt_print = now - last_print_time
        last_print_time = now
        print(f"[TRAIN] epoch {epoch:4d}/{num_epochs} | ELBO={elbo.item():.3f} | loglike={log_likelihood.item():.3f} | "
              f"logp(z)={log_p_z.item():.3f} | logq={log_q_z.item():.3f} | grad_norm={total_norm.item():.2e} | Δt={dt_print:6.2f}s")
    # Quick boundary snapshots (save every PLOT_EVERY epochs).
    if (epoch % PLOT_EVERY == 0) or (epoch == num_epochs - 1):
        with torch.no_grad():
            snapshot_samples, _ = model.forward(n_samples=5000)
            snapshot_path = os.path.join(snapshot_dir, f"posterior_boundary_epoch_{epoch:04d}.png")
            plot_posterior_samples(epoch, snapshot_samples.cpu().numpy(), save_path=snapshot_path)
# ---- posterior samples for final analysis ----
posterior_draw = 5000
with torch.no_grad():
    post_samples, _ = model.forward(n_samples=posterior_draw)
    post_samples_np = post_samples.cpu().numpy()

print("[DONE] Finished.")

# ---- save raw ELBO-component data (Stage-1 and Stage-2) ----
# We already save the rendered PNGs, but for paper-quality replotting we also
# persist the underlying time series in a compact .npz file.
try:
    _elbo_npz_path = os.path.join(outdir, f"stage{STAGE_INT}_elbo_components_raw.npz")
    np.savez_compressed(
        _elbo_npz_path,
        stage=int(STAGE_INT),
        num_epochs=int(num_epochs),
        epoch=np.arange(len(elbo_history), dtype=np.int32),
        elbo=np.asarray(elbo_history, dtype=np.float64),
        E_log_p_y_given_z=np.asarray(log_likelihood_history, dtype=np.float64),
        E_log_p_z_given_sigma=np.asarray(log_prior_z_history, dtype=np.float64),
        neg_E_log_q_z=np.asarray([-v for v in log_q_history], dtype=np.float64),
        # Optional term (present in this script): fused/TV regularization contribution
        E_log_p_fuse=np.asarray(log_p_fuse_history, dtype=np.float64),
    )
    print(f"[OUTPUT] Saved raw ELBO-component data: {_elbo_npz_path}")
except Exception as _e:
    print(f"[WARN] Failed to save raw ELBO-component data: {_e}")
# ---- plots: ELBO components ----
plt.figure(figsize=(10, 5))
plt.plot(elbo_history, label="ELBO")
plt.plot(log_likelihood_history, label="E[log p(y|z)]")
plt.plot(log_prior_z_history, label="E[log p(z)]")
plt.plot([-v for v in log_q_history], label="-E[log q(z)]")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("ELBO and components")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "elbo_components.png"), dpi=300)
plt.close()
print(f"[OUTPUT] Saved outputs in folder: {outdir}")
# ================================= Step 8: Utilities to render requested figures  ====================================
def _save_nf_posterior_distributions(posterior_samples, true_params_flat, filename=os.path.join(outdir, 'nf_posterior_distributions.png')):
    """Draw histograms of each parameter with prior mean, true value (if available), and posterior mean.
    Works for arbitrary DIM_Z = posterior_samples.shape[1]. If true_params_flat has
    fewer entries than DIM_Z (e.g. K1 != K2), only the first len(true_params_flat)
    dimensions will have a red "true" line.
    """
    posterior_samples = np.asarray(posterior_samples)
    true_params_flat = np.asarray(true_params_flat).ravel()
    dim = posterior_samples.shape[1]
    initial_params = np.zeros(dim)  # prior mean in z-space
    posterior_mean = np.mean(posterior_samples, axis=0)
    ncols = 4
    nrows = int(np.ceil(dim / ncols))
    plt.figure(figsize=(4 * ncols, 3 * nrows))
    for i in range(dim):
        plt.subplot(nrows, ncols, i + 1)
        plt.hist(posterior_samples[:, i], bins=30, density=True, alpha=0.7, label='NF Posterior')
        if i < true_params_flat.size:
            plt.axvline(true_params_flat[i], color='r', linestyle='--', linewidth=2, label='True value')
        plt.axvline(initial_params[i], color='g', linestyle='--', linewidth=2, label='Prior mean')
        plt.axvline(posterior_mean[i], color='b', linestyle='-', linewidth=2, label='Posterior mean')
        plt.xlabel(f'Param {i+1}')
        plt.ylabel('Density')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
def _save_nf_boundary_comparison(posterior_samples, filename=os.path.join(outdir, 'nf_boundary_comparison.png')):
    """Draw true/prior/posterior-mean boundaries + a cloud of sample boundaries and nodes.
    TRUE boundary uses K1 control points (ctrl_pts_true).
    Prior / posterior boundaries use K2 control points (ctrl_pts_init_base) and
    NF offsets living in DIM_Z = 2 * K2.
    """
    posterior_samples = np.asarray(posterior_samples)
    total_samples, dim = posterior_samples.shape
    # Posterior mean (in z-space, DIM_Z)
    mean_offset = np.mean(posterior_samples, axis=0).reshape(-1, 2)  # (K2, 2)
    ctrl_pts_mean = ctrl_pts_init_base + mean_offset  # (K2, 2)
    _, _, curve_points_mean = build_closed_bspline(ctrl_pts_mean)
    # TRUE boundary (K1 control points, from synthetic model)
    ctrl_closed_true = np.vstack([ctrl_pts_true, ctrl_pts_true[0]])
    _, _, curve_points_true_local = build_closed_bspline(ctrl_pts_true)
    # PRIOR boundary (zero offset in z-space, i.e. pure ctrl_pts_init_base)
    prior_offset = np.zeros((K2, 2))
    ctrl_pts_prior = ctrl_pts_init_base + prior_offset
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0]])
    _, _, curve_points_prior = build_closed_bspline(ctrl_pts_prior)
    # Prepare figure
    plt.figure(figsize=(12, 10))
    # True (red)
    plt.plot(curve_points_true_local[:, 0], curve_points_true_local[:, 1], 'r-', lw=3, label='True boundary')
    plt.plot(ctrl_closed_true[:, 0], ctrl_closed_true[:, 1], 'ro--', markersize=8, alpha=0.9, label='True nodes')
    # Prior (green, K2 nodes)
    plt.plot(curve_points_prior[:, 0], curve_points_prior[:, 1], 'g--', lw=2, label='Prior mean boundary')
    plt.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], 'go--', markersize=6, alpha=0.7,
             label='Prior mean nodes')
    # Posterior mean (blue, K2 nodes)
    ctrl_closed_mean = np.vstack([ctrl_pts_mean, ctrl_pts_mean[0]])
    plt.plot(curve_points_mean[:, 0], curve_points_mean[:, 1], 'b-', lw=2, label='Mean posterior boundary')
    plt.plot(ctrl_closed_mean[:, 0], ctrl_closed_mean[:, 1], 'bo--', markersize=6, alpha=0.7,
             label='Mean posterior nodes')
    # Sample boundaries/nodes in gray (sub-sampled for readability)
    num_sample_boundaries = 1000
    step = max(1, total_samples // num_sample_boundaries)
    for i in range(0, total_samples, step):
        sample_offset = posterior_samples[i].reshape(-1, 2)  # (K2, 2)
        ctrl_pts_sample = ctrl_pts_init_base + sample_offset
        ctrl_closed_sample = np.vstack([ctrl_pts_sample, ctrl_pts_sample[0]])
        _, _, curve_points_sample = build_closed_bspline(ctrl_pts_sample)
        plt.plot(curve_points_sample[:, 0], curve_points_sample[:, 1], 'gray', lw=0.5, alpha=0.1,
                 label='Sample boundaries' if i == 0 else "")
        plt.plot(ctrl_closed_sample[:, 0], ctrl_closed_sample[:, 1], 'gray',
                 marker='o', linestyle='none', markersize=2, alpha=0.05,
                 label='Sample nodes' if i == 0 else "")
    # Extra node cloud (light gray) for K2 nodes
    all_sample_nodes = []
    for i in range(0, total_samples, max(1, total_samples // 1000)):
        sample_offset = posterior_samples[i].reshape(-1, 2)
        ctrl_pts_sample = ctrl_pts_init_base + sample_offset
        all_sample_nodes.extend(ctrl_pts_sample)
    all_sample_nodes = np.array(all_sample_nodes)
    plt.scatter(all_sample_nodes[:, 0], all_sample_nodes[:, 1],
                c='gray', alpha=0.02, s=10, label='Node distribution (K2)')
    plt.title('Boundary Comparison with Normalizing Flow Posterior Samples and Nodes')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.gca().invert_yaxis()  # top shallower like MCMC
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
# ================================= Step 9: Final posterior distribution  ====================================
print("Training finished. Saving final posterior outputs...")
final_boundary_path = os.path.join(outdir, f"posterior_boundary_epoch_{num_epochs:04d}_final.png")
plot_posterior_samples(num_epochs, post_samples_np, save_path=final_boundary_path)

post_npy_path = os.path.join(outdir, "posterior_samples.npy")
np.save(post_npy_path, post_samples_np)
print(f"Posterior samples saved: {post_npy_path} | shape: {post_samples_np.shape}")

# Generate final posterior figures from the same saved posterior sample set.
try:
    _save_nf_posterior_distributions(
        posterior_samples=post_samples_np,
        true_params_flat=true_params.detach().cpu().numpy().reshape(-1),
        filename=os.path.join(outdir, 'nf_posterior_distributions.png')
    )
    print("Saved 'nf_posterior_distributions.png'")
except Exception as e:
    print("Failed to save 'nf_posterior_distributions.png':", e)
try:
    _save_nf_boundary_comparison(
        posterior_samples=post_samples_np,
        filename=os.path.join(outdir, 'nf_boundary_comparison.png')
    )
    print("Saved 'nf_boundary_comparison.png'")
except Exception as e:
    print("Failed to save 'nf_boundary_comparison.png':", e)

# Optional training-history plot in the stage output folder.
elbo_history_path = os.path.join(outdir, 'elbo_history.png')
plt.figure(figsize=(10, 6))
plt.plot(elbo_history, 'b-', lw=2, label='ELBO')
plt.plot(log_likelihood_history, 'r-', lw=2, label='log p(y|z)')
plt.plot(log_prior_z_history, 'g-', lw=2, label='log p(z)')
plt.plot(log_q_history, 'm-', lw=2, label='log q(z)')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('ELBO and Components During Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(elbo_history_path, dpi=300)
plt.close()

# Plot gradient history if the list is populated.
if gradient_history:
    plot_gradient_history(gradient_history, filename=os.path.join(outdir, 'gradient_history.png'))

# ---------------- Save posterior-mean control points for each stage ----------------
try:
    _mean_off = post_samples_np.mean(axis=0).reshape(-1, 2)
    _ctrl_mean = (ctrl_pts_init_base + _mean_off).astype(float)
    np.save(os.path.join(outdir, f"stage{STAGE_INT}_mean_ctrl_pts.npy"), _ctrl_mean)
    np.save(os.path.join(outdir, f"stage{STAGE_INT}_mean_offset.npy"), _mean_off.astype(float))
    print(f"[STAGE-{STAGE_INT}] Saved posterior-mean ctrl pts: stage{STAGE_INT}_mean_ctrl_pts.npy")
except Exception as _e:
    print(f"[STAGE-{STAGE_INT}] WARNING: failed to save posterior-mean ctrl pts: {_e}")
# ======================= Auto launch Stage-2 =======================
# Stage-1 -> Stage-2: rerun this script with STAGE_INT=2 using Stage-1 mean control points by default;
# if available, pruned control points are also accepted as a fallback/alternative input.
# NOTE: AUTO_RUN_STAGE2 is already defined near the top with .strip(); do NOT redefine it here.

def _run_next(_env):
    try:
        _subprocess.run([sys.executable, __file__], env=_env, check=True)
    except Exception as _e:
        print(f"[AUTO] Failed to launch Stage-2: {_e}")

if (STAGE_INT == 1) and AUTO_RUN_STAGE2:
    # ======================= Two-stage posterior-variance pruning =======================

    # --------- User-tunable knobs (put alpha up front) ----------###########################################################################
    PRUNE_ALPHA = float(os.environ.get("PRUNE_ALPHA", "1.5"))   # quantile-threshold multiplier
    MIN_KEEP = int(os.environ.get("PRUNE_MIN_KEEP", "6"))       # safety valve
    MAX_DROP_FRAC = float(os.environ.get("PRUNE_MAX_DROP_FRAC", "0.70"))  # safety: do not drop >70% by default

    stage1_dir = outdir
    mean_ctrl_path = os.path.join(stage1_dir, "stage1_mean_ctrl_pts.npy")
    post_path = os.path.join(stage1_dir, "posterior_samples.npy")

    if (not os.path.exists(mean_ctrl_path)) or (not os.path.exists(post_path)):
        print("[PRUNE] Missing stage1_mean_ctrl_pts.npy or posterior_samples.npy; skipping Stage-2 auto-run.")
    else:
        ctrl_mean = np.load(mean_ctrl_path)
        post = np.load(post_path)  # (Ns, 2*K2)

        if post.ndim != 2 or post.shape[1] % 2 != 0:
            raise ValueError(f"[PRUNE] Unexpected posterior shape: {post.shape}")

        K2_stage1 = post.shape[1] // 2
        if ctrl_mean.shape[0] != K2_stage1:
            print(f"[PRUNE][WARN] ctrl_mean has K={ctrl_mean.shape[0]} but posterior implies K={K2_stage1}. Using posterior K.")
            ctrl_mean = np.asarray(ctrl_pts_init_base, dtype=float)[:K2_stage1].copy()

        off = post.reshape(post.shape[0], K2_stage1, 2)  # (Ns, K, 2)
        var_xy = np.var(off, axis=0, ddof=0)             # (K, 2) in offset-space
        std_xy = np.sqrt(var_xy + 1e-12)

        # --- Estimate local tangents/normals from posterior-mean polygon ---
        p = np.asarray(ctrl_mean, dtype=float)
        K = p.shape[0]
        p_im1 = p[(np.arange(K) - 1) % K]
        p_ip1 = p[(np.arange(K) + 1) % K]
        t = p_ip1 - p_im1
        t_norm = np.linalg.norm(t, axis=1) + 1e-12
        t_hat = t / t_norm[:, None]
        n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)  # rotate +90 deg

        # --- Project posterior offsets onto normal/tangent directions ---
        proj_n = np.sum(off * n_hat[None, :, :], axis=2)  # (Ns, K)
        proj_t = np.sum(off * t_hat[None, :, :], axis=2)  # (Ns, K)
        var_n = np.var(proj_n, axis=0, ddof=0)            # (K,)
        var_t = np.var(proj_t, axis=0, ddof=0)            # (K,)

        # ---------------- Anti-artifact pruning score (std_n-based) ----------------
        USE_CURV = int(os.environ.get("PRUNE_USE_CURV", "0")) == 1
        CURV_WEIGHT = float(os.environ.get("PRUNE_CURV_WEIGHT", "1.0"))

        std_n = np.std(proj_n, axis=0, ddof=0)  # (K,), units: m
        med_std_n = float(np.median(std_n)) + 1e-12
        std_n_norm = std_n / med_std_n

        # Optional: curvature proxy
        curv = np.zeros(K, dtype=float)
        if USE_CURV:
            ctrl_mean_arr = np.asarray(ctrl_mean, dtype=float)
            for _i in range(K):
                _im1 = (_i - 1) % K
                _ip1 = (_i + 1) % K
                p0 = ctrl_mean_arr[_im1]
                p1 = ctrl_mean_arr[_i]
                p2 = ctrl_mean_arr[_ip1]
                s1 = float(np.linalg.norm(p1 - p0)) + 1e-12
                s2 = float(np.linalg.norm(p2 - p1)) + 1e-12
                t1 = (p1 - p0) / s1
                t2 = (p2 - p1) / s2
                ct = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
                dtheta = float(np.arccos(ct))
                curv[_i] = dtheta / (0.5 * (s1 + s2) + 1e-12)

            med_curv = float(np.median(curv)) + 1e-12
            curv_norm = curv / med_curv
            score = std_n_norm + CURV_WEIGHT * curv_norm
        else:
            score = std_n_norm

        # ---------------- Robust quantile-based pruning threshold ----------------
        PRUNE_Q = float(os.environ.get("PRUNE_Q", "0.65"))  # recommended: 0.6–0.7
        score_q = float(np.quantile(score, PRUNE_Q))

        thr_score = PRUNE_ALPHA * score_q

        print(f"[PRUNE] Using quantile threshold: q={PRUNE_Q:.2f}, "
            f"score_q={score_q:.4g}, thr={thr_score:.4g}")

        # Primary keep: retain nodes whose sensitivity score is within the scaled quantile threshold
        keep = (score <= thr_score)
        keep_idx = np.where(keep)[0].tolist()

# ---------------- Occam / evidence quick-check ----------------
# Goal: decide which nodes are "dispensable" by comparing expected log-joint
# (log-likelihood + log-prior) BEFORE Stage-2 starts.
#
# STRICT-DELETE mode (requested):
#   - We REALLY delete a node: K -> K-1 (not "set dx,dz to 0").
#   - We rebuild the closed control-point set (cyclic order) and re-run forward.
#   - Greedy: test nodes one-by-one against the CURRENT model (updates baseline after each accepted drop).
#   - Deterministic expectation: use {mean, mean±1σ, mean±2σ} (5 representative samples).
#   - Default OCCAM_TOL=0: only drop if the expected log-joint does NOT get worse.
#
# Notes:
#   - This Occam check is used only for Stage-1 -> Stage-2 transition pruning.
#   - It does NOT change the NF training dimension; it only selects which ctrl points
#     are carried into Stage-2 by writing a pruned ctrl-pt file.

DO_OCCAM_PRUNE = (int(os.environ.get("DO_OCCAM_PRUNE", "1")) == 1)
OCCAM_TOL = float(os.environ.get("OCCAM_TOL", "0.0"))#################################################################
MAX_OCCAM_DROPS = int(os.environ.get("MAX_OCCAM_DROPS", "999"))
STRICT_OCCAM_FAIL = (os.environ.get("STRICT_OCCAM_FAIL", "0").strip() == "1")
OCCAM_MODE = os.environ.get("OCCAM_MODE", "strict_delete").strip().lower()  # "strict_delete" or "freeze_zero"

# Deterministic samples: multipliers for (mean + k * std)
# default: 0, +1, -1, +2, -2  (total 5)
_ks_str = os.environ.get("OCCAM_KSIGS", "0,1,-1,2,-2").strip()
try:
    OCCAM_KSIGS = [float(s) for s in _ks_str.split(",") if s.strip() != ""]
except Exception:
    OCCAM_KSIGS = [0.0, 1.0, -1.0, 2.0, -2.0]
if len(OCCAM_KSIGS) == 0:
    OCCAM_KSIGS = [0.0]

if DO_OCCAM_PRUNE and (STAGE_INT == 1) and AUTO_RUN_STAGE2:
    # Robustness: Occam pruning refines the primary keep_idx produced above from
    # score <= PRUNE_ALPHA * score_q. We defensively verify those inputs exist.
    _occam_have_keep_idx = ('keep_idx' in locals())
    _occam_have_score = ('score' in locals())
    if (not _occam_have_keep_idx) or (not _occam_have_score):
        _msg = "[PRUNE][OCCAM] ERROR: prerequisites missing: " + (
            ("keep_idx " if not _occam_have_keep_idx else "") +
            ("score " if not _occam_have_score else "")
        ).strip() + ". Skipping Occam."
        print(_msg)
        if STRICT_OCCAM_FAIL:
            raise RuntimeError(_msg)
        # Fallback: if keep_idx is missing, keep everything (no pruning) rather than risk accidental drops.
        if not _occam_have_keep_idx:
            # Robust fallback: infer K from available arrays/vars (avoid NameError if K is out of scope)
            _K_local = None
            try:
                _K_local = int(globals().get('K', -1))
                if _K_local is not None and _K_local > 0:
                    pass
                else:
                    _K_local = None
            except Exception:
                _K_local = None
            if _K_local is None:
                try:
                    # Prefer Stage-2 init ctrl pts if present
                    if 'ctrl_pts_init_base' in locals() and hasattr(ctrl_pts_init_base, 'shape'):
                        _K_local = int(ctrl_pts_init_base.shape[0])
                    elif 'ctrl_mean' in locals():
                        _cm = np.asarray(ctrl_mean)
                        if _cm.ndim >= 2:
                            _K_local = int(_cm.shape[0])
                    elif 'ctrl_pts' in locals() and hasattr(ctrl_pts, 'shape'):
                        _K_local = int(ctrl_pts.shape[0])
                except Exception:
                    _K_local = None
            if _K_local is None:
                # Last resort: do nothing (skip defining keep_idx) rather than crash
                print('[PRUNE][OCCAM] WARN: could not infer K for fallback keep_idx; leaving keep_idx undefined')
            else:
                keep_idx = list(range(_K_local))
    else:
        try:

            occam_log_path = os.path.join(outdir, "occam_prune_log.txt")

            def _occam_log(_msg: str):
                print(_msg)
                try:
                    with open(occam_log_path, "a", encoding="utf-8") as _f:
                        _f.write(_msg + "\n")
                except Exception:
                    pass

            _occam_log("=" * 100)
            _occam_log("[PRUNE][OCCAM] Evidence-check (greedy drop) - verbose audit log")
            _occam_log(
                f"[PRUNE][OCCAM] K={K} | MIN_KEEP={MIN_KEEP} | OCCAM_TOL={OCCAM_TOL:g} | "
                f"MAX_OCCAM_DROPS={MAX_OCCAM_DROPS} | MODE={OCCAM_MODE} | KSIGS={OCCAM_KSIGS}"
            )

            # ---------------- Score-threshold pruning audit ----------------
            try:
                _occam_log("[PRUNE][SCORE] Score-threshold pruning (pre-Occam)")
                _occam_log(f"[PRUNE][SCORE] PRUNE_ALPHA={PRUNE_ALPHA:g} | score_q={score_q:.6g} | thr={thr_score:.6g}")
                _occam_log(f"[PRUNE][SCORE] keep = (score <= thr) => {int(np.sum(keep))}/{K} kept before Occam")
                _occam_log("[PRUNE][SCORE] Per-node details: node | score | std_n_norm | keep_by_score")
                for _i in range(K):
                    _occam_log(f"[PRUNE][SCORE] node={_i:02d} | score={float(score[_i]):.6g} | std_n_norm={float(std_n_norm[_i]):.6g} | keep={int(keep[_i])}")
                _ord = np.argsort(score)
                _occam_log("[PRUNE][SCORE] Nodes sorted by score (low -> high): " + ",".join([f"{int(i):02d}" for i in _ord.tolist()]))
            except Exception as _score_e:
                _occam_log("[PRUNE][SCORE] WARNING: failed to write score audit: " + str(_score_e))

            # Use posterior samples (prefer in-memory post_samples_np if present)
            post_samples_np_local = locals().get("post_samples_np", None)
            if post_samples_np_local is None:
                post_samples_np_local = post  # fallback (Stage-1 list/np array)
            post_samples_np_local = np.asarray(post_samples_np_local, dtype=float)
            if post_samples_np_local.ndim != 2 or post_samples_np_local.shape[1] != 2 * K:
                raise ValueError(f"post_samples_np_local has unexpected shape: {post_samples_np_local.shape}, expected (N, {2*K})")

            z_mean = post_samples_np_local.mean(axis=0).astype(np.float32)              # (2K,)
            z_std  = post_samples_np_local.std(axis=0, ddof=0).astype(np.float32)      # (2K,)

            # Prior std for z dims: use Stage-1 fixed sigma (consistent with this 2-stage, script)
            sigma_z = float(os.environ.get("FIXED_SIGMA_STAGE1", os.environ.get("TSTD_DEFAULT", "50.0")))
            sigma_z = float(sigma_z)
            if sigma_z <= 0:
                raise ValueError("sigma_z must be positive")

            # ---- No-grad SEM log-likelihood for an arbitrary ctrl-point set (K can change) ----
            # This bypasses SEMLikelihoodAdjointFn (which assumes fixed K and uses ctrl_pts_init_base + z_off).
            # Keep this path deterministic and self-contained because Occam strict-delete is already expensive.
            occam_noise_std = float(obs_noise_std)
            if not np.isfinite(occam_noise_std) or occam_noise_std <= 0.0:
                raise ValueError(f"Invalid occam_noise_std={occam_noise_std}. Expected a positive finite value.")

            def _sem_loglik_no_grad_from_ctrl(abs_ctrl_pts: np.ndarray, y_obs_tensor: torch.Tensor) -> float:
                abs_ctrl_pts = np.asarray(abs_ctrl_pts, dtype=np.float32)
                if abs_ctrl_pts.ndim != 2 or abs_ctrl_pts.shape[1] != 2:
                    raise ValueError(
                        f"Occam strict-delete: abs_ctrl_pts must have shape (K_active, 2), got {abs_ctrl_pts.shape}"
                    )
                if abs_ctrl_pts.shape[0] < int(MIN_KEEP):
                    raise ValueError(
                        f"Occam strict-delete: active control points K_active={abs_ctrl_pts.shape[0]} < MIN_KEEP={int(MIN_KEEP)}"
                    )

                cfg_base = copy.deepcopy(sem_config)
                y_obs_np = y_obs_tensor.detach().cpu().numpy()
                noise_var = float(occam_noise_std) ** 2 + EPS

                src_positions = cfg_base['source'].get('positions', None)
                if not src_positions:
                    raise ValueError("Occam strict-delete: 'source.positions' missing/empty in sem_config")
                nrec = len(cfg_base['receivers']['positions'])
                nsrc = len(src_positions)
                if y_obs_np.ndim != 2:
                    raise ValueError(f"Occam strict-delete: y_obs must be 2D, got shape {y_obs_np.shape}")
                if y_obs_np.shape[1] != nsrc * nrec:
                    raise ValueError(
                        f"Occam strict-delete: expected {nsrc*nrec} traces ({nsrc}*{nrec}), got {y_obs_np.shape[1]}"
                    )

                total_ll = 0.0
                for k, pos in enumerate(src_positions):
                    start_col = k * nrec
                    end_col = (k + 1) * nrec
                    y_obs_k = y_obs_np[:, start_col:end_col]
                    cfg_k = build_single_source_cfg(cfg_base, pos)
                    sim_k = get_or_make_sim(cfg_k)
                    out_k = sim_k.run_forward_and_adjoint({
                        'bspline_ctrl': abs_ctrl_pts,
                        'y_obs': y_obs_k,
                        'obs_dt': float(obs_dt),
                        'obs_nt': int(obs_nt),
                        'noise_std': float(np.sqrt(noise_var))
                    })
                    ll_k = float(out_k["loglik"])
                    if not np.isfinite(ll_k):
                        raise FloatingPointError(
                            f"Occam strict-delete: non-finite source loglik encountered for source {k}: {ll_k}"
                        )
                    total_ll += ll_k
                if not np.isfinite(total_ll):
                    raise FloatingPointError(f"Occam strict-delete: non-finite total loglik encountered: {total_ll}")
                return float(total_ll)

            # ---- Helper: build abs ctrl pts for an active index set and a z-offset vector ----
            def _abs_ctrl_from_z(active_idx: list[int], z_vec: np.ndarray) -> np.ndarray:
                idx = np.asarray(active_idx, dtype=int)
                if idx.ndim != 1:
                    raise ValueError(f"Occam strict-delete: active_idx must be 1D, got shape {idx.shape}")
                if idx.size == 0:
                    raise ValueError("Occam strict-delete: active_idx is empty")
                if np.any(idx < 0) or np.any(idx >= K):
                    raise IndexError(f"Occam strict-delete: active_idx contains out-of-range entries: {idx.tolist()}")

                z_off = np.asarray(z_vec, dtype=np.float32).reshape(K, 2)  # (K,2) in original Stage-1 indexing
                base_all = np.asarray(ctrl_pts_init_base, dtype=np.float32)
                if base_all.shape != (K, 2):
                    raise ValueError(
                        f"Occam strict-delete: ctrl_pts_init_base has shape {base_all.shape}, expected ({K}, 2)"
                    )
                base = base_all[idx, :]  # (K_active,2)
                off = z_off[idx, :]      # (K_active,2)
                abs_ctrl = base + off
                if not np.all(np.isfinite(abs_ctrl)):
                    raise FloatingPointError("Occam strict-delete: abs_ctrl contains non-finite values")
                return abs_ctrl

            # ---- Prior on active dims only (Gaussian with sigma_z) ----
            def _logprior_active(z_vec: np.ndarray, active_idx: list[int]) -> float:
                idx = np.asarray(active_idx, dtype=int)
                if idx.size == 0:
                    raise ValueError("Occam strict-delete: active_idx is empty in _logprior_active")
                z_off = np.asarray(z_vec, dtype=np.float32).reshape(K, 2)
                zz = z_off[idx, :].reshape(-1)  # (2*K_active,)
                lp = np.sum(-0.5 * (zz / sigma_z) ** 2 - np.log(sigma_z) - 0.5 * np.log(2.0 * np.pi))
                if not np.isfinite(lp):
                    raise FloatingPointError(f"Occam strict-delete: non-finite log-prior encountered: {lp}")
                return float(lp)

            # ---- Deterministic representative batch in z-space ----
            def _det_z_batch() -> list[np.ndarray]:
                zs = []
                for ksig in OCCAM_KSIGS:
                    z_rep = z_mean + float(ksig) * z_std
                    if not np.all(np.isfinite(z_rep)):
                        raise FloatingPointError("Occam strict-delete: representative z batch contains non-finite values")
                    zs.append(np.asarray(z_rep, dtype=np.float32))
                if len(zs) == 0:
                    raise ValueError("Occam strict-delete: OCCAM_KSIGS produced an empty representative batch")
                return zs

            det_zs = _det_z_batch()

            # Expected log-joint for a given active set
            def _expected_logjoint_active(active_idx: list[int]) -> float:
                if len(active_idx) == 0:
                    raise ValueError("Occam strict-delete: active_idx is empty in _expected_logjoint_active")
                vals = []
                for z_vec in det_zs:
                    abs_ctrl = _abs_ctrl_from_z(active_idx, z_vec)
                    ll = _sem_loglik_no_grad_from_ctrl(abs_ctrl, noisy_obs)
                    lp = _logprior_active(z_vec, active_idx)
                    val = float(ll + lp)
                    if not np.isfinite(val):
                        raise FloatingPointError(
                            f"Occam strict-delete: non-finite log-joint encountered for active_idx={active_idx}"
                        )
                    vals.append(val)
                if len(vals) == 0:
                    raise ValueError("Occam strict-delete: no representative samples were evaluated")
                return float(np.mean(vals))

            # ---- Initialize active set from score-threshold pruning ----
            # (keep_idx was computed above from score <= PRUNE_ALPHA * score_q)
            keep_idx = list(keep_idx)  # ensure list
            if len(keep_idx) < int(MIN_KEEP):
                raise ValueError(
                    f"Occam strict-delete: initial keep_idx has only {len(keep_idx)} nodes, below MIN_KEEP={int(MIN_KEEP)}"
                )
            active = np.zeros(K, dtype=bool)
            active[np.asarray(keep_idx, dtype=int)] = True

            # Baseline is the CURRENT active set (not full K)
            active_idx = np.where(active)[0].tolist()
            base_val = _expected_logjoint_active(active_idx)

            _occam_log(f"[PRUNE][OCCAM] Baseline expected log-joint E0={float(base_val):+.6f} (active set size={len(active_idx)})")

            order_drop = np.argsort(-score)  # highest score first (try to drop unstable nodes first)
            drops_done = 0

            _occam_log(f"[PRUNE][OCCAM] Initial keep_idx ({len(active_idx)}/{K}): {active_idx}")

            # Greedy loop
            for node in order_drop.tolist():
                if not active[node]:
                    continue
                if int(active.sum()) <= int(MIN_KEEP):
                    break
                if drops_done >= int(MAX_OCCAM_DROPS):
                    break

                # Test strict deletion: remove this node from the active set
                test_active_idx = [i for i in np.where(active)[0].tolist() if i != int(node)]

                if OCCAM_MODE == "freeze_zero":
                    # Back-compat mode: emulate old behavior (NOT strict delete)
                    # Equivalent to: keep node but force its offset to zero in all representative samples.
                    # Still uses full ctrl set, so not used in your strict workflow.
                    def _expected_logjoint_freeze(node_to_freeze: int, act_idx: list[int]) -> float:
                        vals = []
                        for z_vec in det_zs:
                            z_mod = np.array(z_vec, copy=True)
                            j0 = 2 * int(node_to_freeze)
                            z_mod[j0:j0+2] = 0.0
                            abs_ctrl = _abs_ctrl_from_z(act_idx, z_mod)
                            ll = _sem_loglik_no_grad_from_ctrl(abs_ctrl, noisy_obs)
                            lp = _logprior_active(z_mod, act_idx)
                            vals.append(ll + lp)
                        return float(np.mean(vals))
                    test_val = _expected_logjoint_freeze(int(node), active_idx)
                else:
                    test_val = _expected_logjoint_active(test_active_idx)

                base_before = float(base_val)

                dV = float(test_val - base_before)

                # OCCAM_TOL=0 => strict: only drop when test_val >= base_val
                if dV >= -float(OCCAM_TOL):
                    active[node] = False
                    active_idx = test_active_idx
                    base_val = float(test_val)  # greedy update uses CURRENT model
                    drops_done += 1
                    _occam_log(f"[PRUNE][OCCAM] step={drops_done:03d} DROP node={node:02d} | E_before={base_before:+.6f} | E_after={float(test_val):+.6f} | ΔE={dV:+.6f} | K_now={len(active_idx)}")
                else:
                    _occam_log(f"[PRUNE][OCCAM] KEEP node={node:02d} | E_base={base_before:+.6f} | E_test={float(test_val):+.6f} | ΔE={dV:+.6f} | K_now={len(active_idx)}")

            keep_idx = np.where(active)[0].tolist()
            keep = active.copy()
            _occam_log(f"[PRUNE][OCCAM] Final keep after evidence-check: {len(keep_idx)}/{K}")

        except Exception as _e:
            print("[PRUNE][OCCAM] ERROR: evidence-check crashed.")
            traceback.print_exc()
            if STRICT_OCCAM_FAIL:
                raise
            print("[PRUNE][OCCAM] WARNING: falling back to score-threshold pruning due to Occam failure. Reason:", _e)

            if 'score' not in locals():
                _msg2 = "[PRUNE][OCCAM] ERROR: fallback requires 'score' but it is not defined. Keeping all nodes (no pruning)."
                print(_msg2)
                if STRICT_OCCAM_FAIL:
                    raise RuntimeError(_msg2)
                keep_idx = list(range(K))
                keep = np.ones(K, dtype=bool)
            else:
                # Safety valve: keep at least MIN_KEEP nodes
                if len(keep_idx) < MIN_KEEP:
                    order = np.argsort(score)  # ascending
                    keep_idx = order[:MIN_KEEP].tolist()
                    keep = np.zeros(K, dtype=bool)
                    keep[keep_idx] = True

                # Safety valve: do not drop too many nodes
                max_drop = int(np.floor(MAX_DROP_FRAC * K))
                if (K - len(keep_idx)) > max_drop:
                    target_keep = K - max_drop
                    order = np.argsort(score)
                    keep_idx = order[:target_keep].tolist()
                    keep = np.zeros(K, dtype=bool)
                    keep[keep_idx] = True

            # ---- Report pruning diagnostics ----
            print("=" * 100)
            print("[PRUNE] Stage-1 anti-artifact pruning (score threshold + optional curvature + Occam)")
            print(f"[PRUNE] K_stage1={K} | PRUNE_ALPHA={PRUNE_ALPHA:.3g} | score_q={score_q:.4g} | thr={thr_score:.4g}")
            print(f"[PRUNE] keep={len(keep_idx)}/{K} | drop={K-len(keep_idx)} | MIN_KEEP={MIN_KEEP}")
            print("[PRUNE] Per-node stats: idx | std_dx std_dz | std_n | score | keep")
            for kidx in range(K):
                print(
                    f"  {kidx:02d} | {std_xy[kidx,0]:8.3f} {std_xy[kidx,1]:8.3f} | "
                    f"{std_n[kidx]:8.3f} | {score[kidx]:8.3f} | {'KEEP' if keep[kidx] else 'DROP'}"
                )

        # ---------------- Materialize pruned control points for Stage-2 ----------------
        _mean_ctrl_abs = np.asarray(ctrl_mean, dtype=float)
        pruned_ctrl = _mean_ctrl_abs[np.asarray(keep_idx, dtype=int)]
        pruned_path = os.path.join(outdir, "stage1_pruned_ctrl_pts.npy")
        np.save(pruned_path, pruned_ctrl)
        print(f"[PRUNE] Saved pruned ctrl pts for Stage-2: {pruned_path} | shape={pruned_ctrl.shape}")

        _env = os.environ.copy()
        _env["STAGE"] = "2"
        _env["K2_OVERRIDE"] = str(int(len(keep_idx)))
        _env["BASE_CTRL_PATH"] = pruned_path
        _env["OBS_CACHE_DIR"] = OBS_CACHE_DIR
        _run_next(_env)

print(f"[DONE] Completed Stage-{STAGE_INT}. Outputs in: {outdir}")
