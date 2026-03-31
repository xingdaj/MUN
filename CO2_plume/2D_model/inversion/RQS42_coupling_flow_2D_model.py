# Final (slow but real) Normalizing Flow script with SEM-based simulator core
# - Uses SEM wave simulator for observations and for ELBO computation.
# - Flow now uses ActNorm + Piecewise Rational Quadratic (PRQ) Coupling for invertibility.
# - All wave-equation computations still use the Spectral Element Method.

# ============================================================================
# Part 1. Imports and module loading
# ============================================================================
import copy
import faulthandler
import glob
import json
import math
import os
import random
import subprocess as _subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import BSpline
from torch.distributions import MultivariateNormal

faulthandler.enable(all_threads=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sem_waveform.core import SEMSimulation
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_background_from_grid, build_velocity_2d_background_with_anomaly

# ============================================================================
# Part 2. User-adjustable parameters
# Put frequently tuned parameters here for easier maintenance.
# ============================================================================
SEED = int(os.environ.get("SEED", "0"))
CPU_NUM_THREADS = int(os.environ.get("CPU_NUM_THREADS", "8"))

STAGE = str(os.environ.get("STAGE", "1")).strip()
try:
    STAGE_INT = int(STAGE)
except Exception:
    STAGE_INT = 1
STAGE_INT = 1 if STAGE_INT <= 1 else 2

# Geometry / control-point setup
co2_center = np.array([2800.0, 2250.0], dtype=float)
K_MAX = 20
K2_OVERRIDE = os.environ.get("K2_OVERRIDE", None)
K2 = int(K2_OVERRIDE) if (K2_OVERRIDE is not None and str(K2_OVERRIDE).strip() != "") else K_MAX
DIM_Z = 2 * K2

# Prior / stage-transition settings
# Current workflow: fixed-sigma two-stage inversion (no ARD / learned per-node sigma).
# The optional fuse penalty is kept for compatibility, but it is disabled by default (FUSE_LAMBDA=0).
FUSE_LAMBDA = 0.0
STAGE1_FUSE_SCALE = float(os.environ.get("STAGE1_FUSE_SCALE", "0.05"))
FUSE_EPS = 1e-3
FUSE_LAMBDA_EFF = FUSE_LAMBDA * (STAGE1_FUSE_SCALE if (STAGE_INT == 1) else 1.0)

# Stage / cache control
# AUTO_RUN_STAGE2 controls whether Stage-1 automatically launches the reduced Stage-2 run.
# In the current workflow, Occam refinement is only executed when this handoff is enabled.
AUTO_RUN_STAGE2 = (os.environ.get("AUTO_RUN_STAGE2", "1").strip() == "1")
BASE_CTRL_PATH = os.environ.get("BASE_CTRL_PATH", "").strip()
OBS_CACHE_DIR = os.environ.get("OBS_CACHE_DIR", "").strip()
FORCE_REGEN_OBS_CACHE = (os.environ.get("FORCE_REGEN_OBS_CACHE", "1").strip() == "1")
if not OBS_CACHE_DIR:
    OBS_CACHE_DIR = "obs_cache"
os.makedirs(OBS_CACHE_DIR, exist_ok=True)

# Default model-data directory
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model1"

# Grid / background files
VP_SMOOTH_PATH = os.environ.get("VP_SMOOTH_PATH", str(MODEL_DIR / "vp_true.npy"))
VP_PRE_PATH = os.environ.get("VP_PRE_PATH", str(MODEL_DIR / "vp_init.npy"))
VP_META_PATH = os.environ.get("VP_META_PATH", str(MODEL_DIR / "vp_meta.json"))

# Misfit window applied consistently to observations and synthetics
MISFIT_WINDOW = {
    "enabled": True,
    "tmin": 0.2,
    "tmax": None,
    "window_len": 1.5,
    "decim": 1,
    "pick_noise_sec": 0.01,
    "pick_k_sigma": 8.0,
    "pick_buffer_cycles": 2.0,
}

# Initial ellipse for the inversion model
ELLIPSE_RX_INIT = float(os.environ.get("ELLIPSE_RX_INIT", "1100.0"))
ELLIPSE_RZ_INIT = float(os.environ.get("ELLIPSE_RZ_INIT", "260.0"))
ELLIPSE_ROT_INIT_DEG = float(os.environ.get("ELLIPSE_ROT_INIT_DEG", "0.0"))

# Diagnostics used later in the script
DEBUG_ALIGN_VP = (os.environ.get("DEBUG_ALIGN_VP", "1").strip() == "1")
DEBUG_VP_CHECK = (os.environ.get("DEBUG_VP_CHECK", "1").strip() == "1")
DEBUG_PLOT_INIT = (os.environ.get("DEBUG_PLOT_INIT", "1").strip() == "1")

# SEM domain / acquisition parameters
DOMAIN_XMIN = float(os.environ.get("DOMAIN_XMIN", "0.0"))
DOMAIN_XMAX = float(os.environ.get("DOMAIN_XMAX", "6500.0"))
DOMAIN_ZMIN = float(os.environ.get("DOMAIN_ZMIN", "1000.0"))
DOMAIN_ZMAX = float(os.environ.get("DOMAIN_ZMAX", "3200.0"))
NELEM_X = int(os.environ.get("NELEM_X", "46"))
NELEM_Z = int(os.environ.get("NELEM_Z", "25"))
TOTAL_TIME = float(os.environ.get("TOTAL_TIME", "1.8"))
DT = float(os.environ.get("DT", "9.0e-5"))
SOURCE_POSITIONS = [
    [500.0, 1310.0],
    [1500.0, 1310.0],
    [2500.0, 1310.0],
    [3500.0, 1310.0],
    [4500.0, 1310.0],
    [5500.0, 1310.0]
    #[6000.0, 1310.0],
]
SOURCE_FREQUENCY = float(os.environ.get("SOURCE_FREQUENCY", "10.0"))
SOURCE_AMPLITUDE = float(os.environ.get("SOURCE_AMPLITUDE", "1.0"))
RECEIVER_XMIN = int(os.environ.get("RECEIVER_XMIN", "500"))
RECEIVER_XMAX = int(os.environ.get("RECEIVER_XMAX", "6000"))
RECEIVER_DX = int(os.environ.get("RECEIVER_DX", "50"))
RECEIVER_Z = float(os.environ.get("RECEIVER_Z", "1300.0"))
RECEIVER_POSITIONS = [[float(x), RECEIVER_Z] for x in range(RECEIVER_XMIN, RECEIVER_XMAX + 1, RECEIVER_DX)]
POLYNOMIAL_ORDER = int(os.environ.get("POLYNOMIAL_ORDER", "5"))
PML_THICKNESS = float(os.environ.get("PML_THICKNESS", "250.0"))
ADJ_HISTORY_DTYPE = os.environ.get("ADJ_HISTORY_DTYPE", "float32")
VP_NX = int(os.environ.get("VP_NX", "2000"))
VP_NZ = int(os.environ.get("VP_NZ", "800"))
VP_INTERP = os.environ.get("VP_INTERP", "linear")
VEL_VMIN = float(os.environ.get("VEL_VMIN", "1500.0"))
VEL_VMAX = float(os.environ.get("VEL_VMAX", "3500.0"))
VEL_TAU = float(os.environ.get("VEL_TAU", "10.0"))
VEL_SPLINE_SAMPLES = int(os.environ.get("VEL_SPLINE_SAMPLES", "400"))
ANOMALY_ENABLED = (os.environ.get("ANOMALY_ENABLED", "1").strip() == "1")
ANOMALY_V_INSIDE = float(os.environ.get("ANOMALY_V_INSIDE", "3500.0"))
ANOMALY_TAU = float(os.environ.get("ANOMALY_TAU", "10.0"))
ANOMALY_SPLINE_SAMPLES = int(os.environ.get("ANOMALY_SPLINE_SAMPLES", "400"))
ANOMALY_BLEND = os.environ.get("ANOMALY_BLEND", "smooth")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "sem_output")
SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", str(10**9)))

# Training / optimization parameters
TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)
EPS64 = 1e-12
EPS = EPS64
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', '0') or 0)
if NUM_EPOCHS <= 0:
    if STAGE_INT == 1:
        NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS_STAGE1", "200"))####################
    elif STAGE_INT == 2:
        NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS_STAGE2", "200"))#####################
    else:
        raise ValueError(f"Invalid STAGE_INT={STAGE_INT}. This script supports only Stage-1 and Stage-2.")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1"))
PLOT_EVERY = int(os.environ.get("PLOT_EVERY", "10"))#########################
# Shared zoom window for boundary diagnostic plots.
ZOOM_WINDOW = (
    float(os.environ.get("ZOOM_XMIN", "0.0")),
    float(os.environ.get("ZOOM_XMAX", "6500.0")),
    float(os.environ.get("ZOOM_ZMIN", "1000.0")),
    float(os.environ.get("ZOOM_ZMAX", "3000.0")),
)
MIN_ELBO_SAMPLES = int(os.environ.get("MIN_ELBO_SAMPLES", "4"))
MAX_ELBO_SAMPLES = int(os.environ.get("MAX_ELBO_SAMPLES", "8"))  ########################
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-4"))
NUM_FLOWS = int(os.environ.get("NUM_FLOWS", "16"))
K_BINS = int(os.environ.get("K_BINS", "12"))
OBS_NOISE_STD = float(os.environ.get("OBS_NOISE_STD", "1.0e-7"))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "50.0"))
CLIP_GRADIENT = (os.environ.get("CLIP_GRADIENT", "1").strip() == "1")

# Post-training merging
MERGE_DIST_FRAC = float(os.environ.get("MERGE_DIST_FRAC", "0.50"))
MERGE_DIFF_FRAC = float(os.environ.get("MERGE_DIFF_FRAC", "0.50"))
MIN_MERGED_POINTS = int(os.environ.get("MIN_MERGED_POINTS", "4"))

FIXED_SIGMA_STAGE1 = float(os.environ.get("FIXED_SIGMA_STAGE1", "300.0"))
FIXED_SIGMA_STAGE2 = float(os.environ.get("FIXED_SIGMA_STAGE2", "100.0"))

# ============================================================================
# Part 3. Core implementation
# ============================================================================
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


def build_closed_bspline(ctrl_pts: np.ndarray, n_samples: int = 400):
    """Build a closed cubic B-spline curve from control points."""
    pts = np.asarray(ctrl_pts, dtype=float).reshape(-1, 2)
    n_pts = pts.shape[0]

    if n_pts == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.empty((0, 2), dtype=float)
    if n_pts == 1:
        curve = np.repeat(pts, int(max(1, n_samples)), axis=0)
        return np.array([0.0, 1.0], dtype=float), np.linspace(0.0, 1.0, curve.shape[0]), curve
    if n_pts == 2:
        ctrl_closed = np.vstack([pts, pts[0:1]])
        t_curve = np.linspace(0.0, 1.0, int(max(3, n_samples)))
        seg01 = (1.0 - t_curve[:, None]) * ctrl_closed[0] + t_curve[:, None] * ctrl_closed[1]
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=float), t_curve, seg01

    k = min(3, n_pts - 1)
    ctrl_closed = np.vstack([pts, np.tile(pts[0], (k, 1))])
    n = len(ctrl_closed) - 1
    total_knots = n + k + 2

    knots = np.zeros(total_knots, dtype=float)
    knots[:k + 1] = 0.0
    knots[-(k + 1):] = 1.0

    n_inner = total_knots - 2 * (k + 1)
    if n_inner > 0:
        inner_knots = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
        knots[k + 1:-(k + 1)] = inner_knots

    t_curve = np.linspace(knots[k], knots[-(k + 1)], int(max(3, n_samples)))
    spline = BSpline(knots, ctrl_closed, k, extrapolate=False)
    curve = np.asarray(spline(t_curve), dtype=float)
    return knots, t_curve, curve


try:
    torch.set_num_threads(CPU_NUM_THREADS)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_NUM_THREADS))

set_global_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
    )
)
print(f"[DEVICE] {device}")
DEVICE = device
print("Using SEMSimulation from:", SEMSimulation.__module__, SEMSimulation.__qualname__)


def merge_control_points(ctrl_pts, offsets, rep_scale=None, dist_frac=None, diff_frac=None, min_points=None):
    """Merge neighboring control points for a closed curve."""
    ctrl_pts = np.asarray(ctrl_pts, dtype=float)
    offsets = np.asarray(offsets, dtype=float)
    if dist_frac is None:
        dist_frac = float(MERGE_DIST_FRAC)
    if diff_frac is None:
        diff_frac = float(MERGE_DIFF_FRAC)
    if min_points is None:
        min_points = int(MIN_MERGED_POINTS)
    K = ctrl_pts.shape[0]
    assert offsets.shape == (K, 2)
    dists = np.linalg.norm(np.roll(ctrl_pts, -1, axis=0) - ctrl_pts, axis=1)
    med_dist = float(np.median(dists)) + 1e-12
    dist_thr = float(dist_frac) * med_dist
    doffs = np.linalg.norm(np.roll(offsets, -1, axis=0) - offsets, axis=1)
    med_sig = float(np.median(rep_scale)) + 1e-12 if rep_scale is not None else float(np.median(doffs) + 1e-12)
    diff_thr = float(diff_frac) * med_sig
    merge_edge = (dists < dist_thr) | (doffs < diff_thr)
    visited = np.zeros(K, dtype=bool)
    groups = []
    i = 0
    while not np.all(visited):
        while visited[i]:
            i = (i + 1) % K
        group = [i]
        visited[i] = True
        j = i
        while merge_edge[j]:
            jn = (j + 1) % K
            if visited[jn]:
                break
            group.append(jn)
            visited[jn] = True
            j = jn
        groups.append(group)
        i = (i + 1) % K
    if len(groups) >= 2:
        last_idx = groups[-1][-1]
        first_idx = groups[0][0]
        if merge_edge[last_idx] and (first_idx == (last_idx + 1) % K):
            groups[0] = groups[-1] + groups[0]
            groups.pop(-1)
    if len(groups) < int(min_points):
        while len(groups) < int(min_points):
            sizes = [len(g) for g in groups]
            gi = int(np.argmax(sizes))
            g = groups[gi]
            if len(g) <= 1:
                break
            mid = len(g) // 2
            groups[gi] = g[:mid]
            groups.insert(gi + 1, g[mid:])
    merged_pts = np.asarray([np.mean(ctrl_pts[g, :], axis=0) for g in groups], dtype=float)
    stats = dict(
        dist_thr=dist_thr,
        diff_thr=diff_thr,
        med_dist=med_dist,
        med_sig=med_sig,
        dists=dists,
        doffs=doffs,
        merge_edge=merge_edge,
        merge_by_dist=(dists < dist_thr),
        merge_by_diff=(doffs < diff_thr),
    )
    return merged_pts, groups, stats


_SIM_CACHE = {}

def _sim_key_from_cfg(cfg):
    """Build a stable cache key for SEMSimulation."""
    dom = cfg['domain']
    tm = cfg['time']
    src = cfg['source']
    rec = cfg['receivers']
    vel = cfg['velocity']
    meth = cfg.get('method', {})

    src_positions = tuple(tuple(p) for p in src.get('positions', []))
    rec_positions = tuple(tuple(p) for p in rec.get('positions', []))

    if 'position' in src and src['position'] is not None:
        pos_single = tuple(src['position'])
    else:
        pos_single = tuple(src_positions[0]) if len(src_positions) > 0 else (0.0, 0.0)

    key_core = (
        float(dom['xmin']), float(dom['xmax']), float(dom['zmin']), float(dom['zmax']),
        int(dom['nelem_x']), int(dom['nelem_z']),
        float(tm['dt']), float(tm['total_time']),
        pos_single,
        src_positions, rec_positions,
        float(src.get('frequency', 0.0)),
    )

    bg = vel.get('background_2d', vel.get('background', {})) or {}
    vp_path = str(bg.get('vp_path', 'vp_smooth.npy'))
    bg_meta = (
        vp_path,
        float(bg.get('xmin', dom['xmin'])), float(bg.get('xmax', dom['xmax'])), int(bg.get('nx', 0)),
        float(bg.get('zmin', dom['zmin'])), float(bg.get('zmax', dom['zmax'])), int(bg.get('nz', 0)),
        str(bg.get('interp', 'linear')),
    )

    an = vel.get('anomaly', {}) or {}
    an_meta = (
        bool(an.get('enabled', True)),
        float(an.get('v_inside', vel.get('vmin', 0.0))),
        float(an.get('tau', vel.get('tau', 0.0))),
        int(an.get('spline_samples', vel.get('spline_samples', 0))),
        str(an.get('blend', 'smooth')),
        bool(meth.get('VERIFY_PROJECTION', False)),
    )
    return key_core + bg_meta + an_meta


def get_or_make_sim(cfg):
    key = _sim_key_from_cfg(cfg)
    sim = _SIM_CACHE.get(key)
    if sim is None:
        sim = SEMSimulation(cfg)
        _SIM_CACHE[key] = sim
    return sim


def make_ellipse_control_points(center, num_points, rx, rz, rotation_deg=0.0):
    """Generate control points on an ellipse around the given center."""
    center = np.asarray(center, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    pts = np.stack([rx * np.cos(angles), rz * np.sin(angles)], axis=1)
    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    pts = pts + center[None, :]
    return pts


ctrl_pts_init_base = make_ellipse_control_points(
    center=co2_center,
    num_points=K2,
    rx=ELLIPSE_RX_INIT,
    rz=ELLIPSE_RZ_INIT,
    rotation_deg=ELLIPSE_ROT_INIT_DEG,
)
def _fft_bandpass(arr, dt, f_lo, f_hi, roll=0.20, time_axis=0):
    """Fast, dependency-free bandpass via rFFT + cosine tapers."""
    a = np.asarray(arr)
    x = np.moveaxis(a, time_axis, 0)
    nt = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(nt, d=float(dt))
    f_lo = float(f_lo); f_hi = float(f_hi)
    roll = float(roll)
    mask = np.zeros_like(freqs, dtype=float)
    if f_hi <= 0 or f_hi <= f_lo:
        return arr
    passband = (freqs >= f_lo) & (freqs <= f_hi)
    mask[passband] = 1.0
    if roll > 0:
        f1 = max(0.0, f_lo * (1.0 - roll))
        f2 = f_lo
        if f2 > f1:
            idx = (freqs >= f1) & (freqs < f2)
            phi = (freqs[idx] - f1) / (f2 - f1)
            mask[idx] = 0.5 * (1.0 - np.cos(np.pi * phi))
        f3 = f_hi
        f4 = f_hi * (1.0 + roll)
        if f4 > f3:
            idx = (freqs > f3) & (freqs <= f4)
            phi = (freqs[idx] - f3) / (f4 - f3)
            mask[idx] = 0.5 * (1.0 + np.cos(np.pi * phi))
    Xf = X * mask[:, None]
    xf = np.fft.irfft(Xf, n=nt, axis=0)
    xf = np.moveaxis(xf, 0, time_axis)
    return xf


def _apply_window_and_decim_np(traces_np, dt, tmin, tmax, decim, time_axis=0):
    """Apply [tmin,tmax] cropping and time decimation along the *time axis*.

    This script stores traces as (nt, n_traces) i.e., time is axis=0.
    We keep a flexible implementation in case shapes are transposed.
    Returns:
        win_traces, dt_new, nt_new, i0, i1
    """
    arr = np.asarray(traces_np)
    if arr.ndim < 1:
        raise ValueError(f"traces_np must have at least 1 dimension, got shape={arr.shape}")
    nt = arr.shape[time_axis]

    # Compute index range
    if tmin is None:
        i0 = 0
    else:
        i0 = int(round(float(tmin) / float(dt)))
        i0 = max(0, min(i0, nt - 1))
    if tmax is None:
        i1 = nt
    else:
        i1 = int(round(float(tmax) / float(dt))) + 1
        i1 = max(i0 + 1, min(i1, nt))

    # Slice along time axis
    sl = [slice(None)] * arr.ndim
    sl[time_axis] = slice(i0, i1)
    win = arr[tuple(sl)]

    # Decimate along time axis
    if decim is None or int(decim) < 2:
        return win, float(dt), int(win.shape[time_axis]), int(i0), int(i1)

    decim = int(decim)
    sl = [slice(None)] * win.ndim
    sl[time_axis] = slice(None, None, decim)
    win = win[tuple(sl)]
    return win, float(dt) * decim, int(win.shape[time_axis]), int(i0), int(i1)

def _pick_first_break_tmin_np(noisy_obs_np, dt, f0, pick_noise_sec, k_sigma, buffer_cycles, time_axis=0):
    """Lightweight first-break picker (run ONCE) returning a single global tmin.

    We pick on a small subset of traces for speed. Data are assumed to be shaped (nt, n_traces)
    with time on axis=0 by default.
    """
    arr = np.asarray(noisy_obs_np)
    if arr.ndim != 2:
        # Try to flatten everything except time axis
        nt = arr.shape[time_axis]
        arr2 = np.moveaxis(arr, time_axis, 0).reshape(nt, -1)
    else:
        arr2 = arr if time_axis == 0 else np.moveaxis(arr, time_axis, 0)

    nt, ntr = arr2.shape
    n0 = max(8, int(round(float(pick_noise_sec) / float(dt))))
    n0 = min(n0, max(8, nt // 4))

    # Subsample a few representative traces
    X = arr2
    if ntr > 8:
        idx = np.linspace(0, ntr - 1, 8).astype(int)
        X = arr2[:, idx]  # (nt, 8)

    env = np.abs(X)  # (nt, nsel)
    mu = env[:n0, :].mean(axis=0, keepdims=True)
    sd = env[:n0, :].std(axis=0, keepdims=True) + 1e-12
    thr = mu + float(k_sigma) * sd

    first_idx = []
    for j in range(env.shape[1]):
        above = np.where(env[:, j] > thr[0, j])[0]
        first_idx.append(int(above[0]) if above.size else int(n0))
    it_fb = max(first_idx)  # conservative
    t_fb = it_fb * float(dt)

    tmin = t_fb + (float(buffer_cycles) / max(float(f0), 1e-6))
    return float(tmin)


def build_base_sem_config(ctrl_points):
    """Assemble the SEM configuration from Part 2 parameters."""
    cfg = {
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
            'background_2d': {
                'vp_path': VP_SMOOTH_PATH,
                'xmin': DOMAIN_XMIN,
                'xmax': DOMAIN_XMAX,
                'nx': VP_NX,
                'zmin': DOMAIN_ZMIN,
                'zmax': DOMAIN_ZMAX,
                'nz': VP_NZ,
                'interp': VP_INTERP,
            },
            'vmin': VEL_VMIN,
            'vmax': VEL_VMAX,
            'tau': VEL_TAU,
            'spline_samples': VEL_SPLINE_SAMPLES,
            'control_points': ctrl_points.tolist(),
            'perturbations': None,
            'anomaly': {
                'enabled': ANOMALY_ENABLED,
                'control_points': ctrl_points.tolist(),
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
            'output_dir': OUTPUT_DIR,
            'snapshot_interval': SNAPSHOT_INTERVAL,
        },
    }
    cfg.setdefault('method', {})['misfit_window'] = MISFIT_WINDOW
    return cfg


# --------------------------------------------------------------------------------------
# Build two SEM configurations:
#   - sem_cfg_obs: used ONLY to generate observed data y_obs from VP_SMOOTH_PATH (vp_smooth.npy)
#   - sem_cfg_inv: used for inversion/training; background is VP_PRE_PATH (vp_pre.npy) + B-spline anomaly
# --------------------------------------------------------------------------------------

def _apply_vp_meta(cfg: dict, vp_path: str, meta_path: str | None, label: str):
    """Load vp_meta.json and synchronize BOTH grid shape and physical coordinate bounds.

    In this workflow, vp_meta.json is the single source of truth for:
      - vp grid shape (nx, nz)
      - physical bounds (xmin, xmax, zmin, zmax)

    Why:
      - If the SEM domain keeps [0, 6500]×[0, 4000] while the vp grid is defined on
        [-300, 6800]×[-300, 4000], plotting and interpolation will look correct in shape
        but coordinates will be wrong (your current symptom).

    Policy:
      - Overwrite cfg['domain'] bounds using meta when present.
      - Also overwrite cfg['velocity']['background_2d'] bounds to match.
      - Still keep nelem_x/nelem_z and time/source/receiver geometry unchanged.
    """
    bg2d = cfg.get('velocity', {}).get('background_2d', {}) or {}
    if meta_path is None or meta_path == "":
        meta_path = str(Path(vp_path).with_name("vp_meta.json"))

    if not os.path.exists(meta_path):
        print(f"[WARN] {label} meta not found: {meta_path}. Using cfg bounds.")
        return

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {label} meta '{meta_path}': {e}. Using cfg bounds.")
        return

    # --- grid shape ---
    if 'nx' in meta:
        bg2d['nx'] = int(meta['nx'])
    if 'nz' in meta:
        bg2d['nz'] = int(meta['nz'])

    # --- physical bounds: overwrite SEM domain AND background_2d bounds ---
    for k in ('xmin', 'xmax', 'zmin', 'zmax'):
        if k in meta:
            try:
                mv = float(meta[k])
                cfg['domain'][k] = mv
                bg2d[k] = mv
            except Exception:
                pass

    cfg['velocity']['background_2d'] = bg2d

    print(f"[OK] Loaded {label} meta: {os.path.basename(meta_path)} (nx={bg2d.get('nx','?')}, nz={bg2d.get('nz','?')})")
    print(f"     -> Domain set to X=[{cfg['domain']['xmin']},{cfg['domain']['xmax']}], Z=[{cfg['domain']['zmin']},{cfg['domain']['zmax']}] (from meta)")
# ======================================================================================
# vp_pre alignment helper
# ======================================================================================
def load_vp_grid_with_alignment(
    vp_path: str,
    *,
    ref_vp_path: str | None = None,
    save_aligned_path: str | None = None,
    heuristic_threshold: float = 50.0,
    verbose: bool = True,
):
    """Load vp grid (nz,nx) and ensure its vertical orientation matches the reference.

    Why this exists:
      - In your workflow, vp_smooth.npy and vp_pre.npy *should* share the same storage convention.
      - However, it is very easy for one file to be saved with z increasing upward (top row = deep)
        while the plotting / interpolation assumes z increasing downward (top row = shallow).
      - If this happens, projecting vp_pre onto SEM nodes will look like a Y-axis flip.

    Strategy:
      1) If ref_vp_path is provided, choose the orientation (as-is vs flipud) that best matches ref
         in L2 sense (after a light downsample).
      2) Otherwise, fall back to a physically motivated heuristic:
           if mean(vp[top]) > mean(vp[bottom]) + threshold  -> likely flipped -> flipud.
         (Your log showed top mean ~5278 > bottom mean ~4905, which triggers this.)

    Returns:
      vp_grid_aligned : np.ndarray (float64) shape (nz,nx)
      out_path        : str (path to aligned file if saved, else original vp_path)
      info            : dict with alignment diagnostics
    """
    vp_path = str(vp_path)
    vp = np.load(vp_path).astype(np.float64)

    if vp.ndim != 2:
        raise ValueError(f"vp grid must be 2D (nz,nx). Got shape={vp.shape} from {vp_path}")

    def _downsample(a, max_nz=200, max_nx=200):
        nz, nx = a.shape
        sz = max(1, int(np.ceil(nz / max_nz)))
        sx = max(1, int(np.ceil(nx / max_nx)))
        return a[::sz, ::sx]

    info = {
        'vp_path': vp_path,
        'shape': tuple(vp.shape),
        'flipped_ud': False,
        'mode': None,
        'score_keep': None,
        'score_flip': None,
        'top_mean': float(np.mean(vp[0, :])),
        'bottom_mean': float(np.mean(vp[-1, :])),
    }

    vp_keep = vp
    vp_flip = np.flipud(vp)

    # ---- Mode 1: align to reference by minimizing L2 misfit ----
    if ref_vp_path is not None and str(ref_vp_path).strip() != "" and os.path.exists(ref_vp_path):
        ref = np.load(str(ref_vp_path)).astype(np.float64)
        if ref.shape != vp.shape:
            # still try (compare after independent downsample), but warn
            if verbose:
                print(f"[WARN] ref_vp shape {ref.shape} != vp shape {vp.shape}. Will compare after downsampling.")
        a = _downsample(vp_keep)
        b = _downsample(vp_flip)
        r = _downsample(ref)

        # if shapes differ after downsample, crop to common min extents
        nz = min(a.shape[0], r.shape[0])
        nx = min(a.shape[1], r.shape[1])
        a = a[:nz, :nx]; b = b[:nz, :nx]; r = r[:nz, :nx]

        score_keep = float(np.mean((a - r) ** 2))
        score_flip = float(np.mean((b - r) ** 2))
        info['score_keep'] = score_keep
        info['score_flip'] = score_flip
        info['mode'] = 'ref_l2'

        if score_flip < score_keep:
            vp_aligned = vp_flip
            info['flipped_ud'] = True
        else:
            vp_aligned = vp_keep

    # ---- Mode 2: heuristic (no reference) ----
    else:
        info['mode'] = 'heuristic_top_vs_bottom'
        if info['top_mean'] > info['bottom_mean'] + float(heuristic_threshold):
            vp_aligned = vp_flip
            info['flipped_ud'] = True
        else:
            vp_aligned = vp_keep

    # ---- Optional: save aligned grid to disk and return the aligned path ----
    out_path = vp_path
    if save_aligned_path is not None and str(save_aligned_path).strip() != "":
        save_aligned_path = str(save_aligned_path)
        try:
            np.save(save_aligned_path, vp_aligned.astype(np.float64))
            out_path = save_aligned_path
            if verbose:
                print(f"[OK] Saved aligned vp grid -> {save_aligned_path} (flipud={info['flipped_ud']}, mode={info['mode']})")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to save aligned vp grid to {save_aligned_path}: {e}. Using in-memory aligned grid.")

    if verbose:
        if info['mode'] == 'ref_l2':
            print(f"[ALIGN vp] mode=ref_l2 flipud={info['flipped_ud']} score_keep={info['score_keep']:.3e} score_flip={info['score_flip']:.3e}")
        else:
            print(f"[ALIGN vp] mode=heuristic flipud={info['flipped_ud']} top_mean={info['top_mean']:.2f} bottom_mean={info['bottom_mean']:.2f} thr={heuristic_threshold}")
    return vp_aligned, out_path, info
# ======================================================================================


def _make_obs_and_inv_configs(ctrl_points):
    """Create the paired SEM configs used by the workflow.

    Observation config:
      - background = vp_smooth.npy
      - no injected anomaly

    Inversion config:
      - background = vp_pre.npy
      - injected B-spline anomaly with zero initial perturbation
    """
    base_cfg = build_base_sem_config(ctrl_points)
    obs_cfg = copy.deepcopy(base_cfg)
    inv_cfg = copy.deepcopy(base_cfg)

    obs_cfg['velocity']['background_2d']['vp_path'] = VP_SMOOTH_PATH
    inv_cfg['velocity']['background_2d']['vp_path'] = VP_PRE_PATH

    obs_cfg['velocity'].setdefault('anomaly', {})
    obs_cfg['velocity']['anomaly']['enabled'] = False
    obs_cfg['velocity'].pop('control_points', None)
    obs_cfg['velocity'].pop('perturbations', None)
    obs_cfg['velocity']['anomaly'].pop('control_points', None)
    obs_cfg['velocity']['anomaly'].pop('perturbations', None)

    inv_cfg['velocity']['anomaly']['enabled'] = True
    inv_cfg['velocity']['anomaly']['v_inside'] = ANOMALY_V_INSIDE
    zero_perturb = np.zeros((K2, 2), dtype=float).tolist()
    inv_cfg['velocity']['perturbations'] = zero_perturb
    inv_cfg['velocity']['anomaly']['perturbations'] = zero_perturb
    return obs_cfg, inv_cfg


def _sync_background_bounds_to_domain(cfg):
    """Keep background_2d bounds identical to the SEM domain bounds."""
    bg = cfg['velocity'].get('background_2d', {}) or {}
    for key in ('xmin', 'xmax', 'zmin', 'zmax'):
        bg[key] = float(cfg['domain'][key])
    cfg['velocity']['background_2d'] = bg


def _validate_matching_domains(obs_cfg, inv_cfg, tol=1e-6):
    """Obs and inversion configs must use the same physical SEM bounds."""
    dom_obs = obs_cfg['domain']
    dom_inv = inv_cfg['domain']
    for key in ('xmin', 'xmax', 'zmin', 'zmax'):
        if abs(float(dom_obs[key]) - float(dom_inv[key])) > tol:
            raise ValueError(
                f"Domain mismatch between vp_smooth and vp_pre meta for '{key}': "
                f"obs={dom_obs[key]} vs inv={dom_inv[key]}.\n"
                f"Please ensure vp_smooth.npy and vp_pre.npy share identical bounds "
                f"(and their meta json agree)."
            )

sem_cfg_obs, sem_cfg_inv = _make_obs_and_inv_configs(ctrl_pts_init_base)

try:
    _apply_vp_meta(sem_cfg_obs, VP_SMOOTH_PATH, VP_META_PATH, label="vp_smooth")
    _apply_vp_meta(sem_cfg_inv, VP_PRE_PATH, VP_META_PATH, label="vp_pre")
    _validate_matching_domains(sem_cfg_obs, sem_cfg_inv)
except Exception as e:
    print(f"[WARN] Meta sync failed: {e}. Continuing with current cfg bounds.")

for _cfg in (sem_cfg_obs, sem_cfg_inv):
    _sync_background_bounds_to_domain(_cfg)

# Align vp_pre orientation to vp_smooth once, then reuse the aligned path downstream.
try:
    _bg_i = sem_cfg_inv['velocity'].get('background_2d', {}) or {}
    _vp_pre_path = str(_bg_i.get('vp_path', VP_PRE_PATH))
    _vp_smooth_path = str(sem_cfg_obs['velocity'].get('background_2d', {}).get('vp_path', VP_SMOOTH_PATH))
    _aligned_path = os.path.splitext(_vp_pre_path)[0] + "_aligned.npy"
    _, _aligned_path_out, _ = load_vp_grid_with_alignment(
        _vp_pre_path,
        ref_vp_path=_vp_smooth_path,
        save_aligned_path=_aligned_path,
        heuristic_threshold=50.0,
        verbose=DEBUG_ALIGN_VP,
    )
    sem_cfg_inv['velocity']['background_2d']['vp_path'] = _aligned_path_out
except Exception as _e_align:
    print(f"[WARN] vp_pre alignment failed: {_e_align}. Continuing with original vp_pre path.")

# Alias used below: the rest of the script refers to the inversion configuration as sem_config.
sem_config = sem_cfg_inv

# ================================ Step 0: Configuration assembly ================================
num_epochs = NUM_EPOCHS
min_elbo_samples = MIN_ELBO_SAMPLES
max_elbo_samples = MAX_ELBO_SAMPLES
learning_rate = LEARNING_RATE
num_flows = NUM_FLOWS
K_bins = K_BINS
obs_noise_std = OBS_NOISE_STD
max_grad_norm = MAX_GRAD_NORM
clip_gradient = CLIP_GRADIENT
# ================================ step 1: B-spline & True Control Points Setup =====================================
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
ana_cfg = sem_config['velocity']['anomaly']
v_inside = float(ana_cfg['v_inside'])
tau = float(ana_cfg['tau'])
spls = int(ana_cfg['spline_samples'])
blend = str(ana_cfg.get('blend', 'smooth'))
# -------------------- Build velocity model on SEM nodes (2D background + spline anomaly) --------------------
# Background: interpolate vp_smooth.npy onto SEM nodes
bg_cfg = sem_cfg_obs['velocity'].get('background_2d', sem_cfg_obs['velocity'].get('background', {}))
vp_path = bg_cfg.get('vp_path', 'vp_smooth.npy')
vp_grid = np.load(vp_path).astype(np.float64)

# Build x/z grid coordinates (fallback to metadata)
xmin_bg = float(bg_cfg.get('xmin', sem_cfg_obs['domain']['xmin']))
xmax_bg = float(bg_cfg.get('xmax', sem_cfg_obs['domain']['xmax']))
zmin_bg = float(bg_cfg.get('zmin', sem_cfg_obs['domain']['zmin']))
zmax_bg = float(bg_cfg.get('zmax', sem_cfg_obs['domain']['zmax']))
nx_bg   = int(bg_cfg.get('nx', vp_grid.shape[1]))
nz_bg   = int(bg_cfg.get('nz', vp_grid.shape[0]))
xg = np.linspace(xmin_bg, xmax_bg, nx_bg, dtype=np.float64)
zg = np.linspace(zmin_bg, zmax_bg, nz_bg, dtype=np.float64)

interp_method = str(bg_cfg.get('interp', 'linear'))
v_background = build_background_from_grid(
    nodes_xy=global_coords,
    xg=xg, zg=zg, vp_grid=vp_grid,
    method=interp_method,
    clip=True
)

# True model: vp_smooth.npy already includes the CO2 anomaly.
# Do NOT inject a second B-spline anomaly here.
velocity_model = v_background
# Plot the velocity model
plt.figure(figsize=(10, 8))
x_coords = global_coords[:, 0]
z_coords = global_coords[:, 1]
scatter = plt.scatter(x_coords, z_coords, c=velocity_model, cmap='turbo', s=5, marker='s', linewidths=0)
plt.colorbar(scatter, label='Velocity (m/s)')
# Plot source positions
source_positions = sem_config['source']['positions']
for i, src_pos in enumerate(source_positions):
    plt.plot(src_pos[0], src_pos[1], 'k*', markersize=15,
             label='Sources' if i == 0 else "", markeredgecolor='red', markeredgewidth=1.0)
# Plot receiver positions
receiver_positions = np.asarray(sem_config['receivers']['positions'])
plt.plot(receiver_positions[:, 0], receiver_positions[:, 1], 'b^', markersize=8,
         label='Receivers', markeredgecolor='black', markeredgewidth=0.5)
xmin = sem_config['domain']['xmin']; xmax = sem_config['domain']['xmax']
zmin = sem_config['domain']['zmin']; zmax = sem_config['domain']['zmax']
plt.xlim(xmin, xmax); plt.ylim(zmin, zmax)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()  # top shallower like MCMC
plt.title('True Model')
plt.xlabel('X (m)'); plt.ylabel('Z (m)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('velocity_model_true.png', dpi=150)
plt.close()
# -------------------- EXTRA CHECK PLOT: Initial model (vp_pre + initial B-spline anomaly) --------------------
# Initial boundary = ctrl_pts_init_base (K2 nodes) with ZERO offsets. Inside vp is fixed to 3500 m/s.
if DEBUG_PLOT_INIT:
    try:
        # Build background from vp_pre (inversion background)
        bg_cfg_i = sem_cfg_inv['velocity'].get('background_2d', sem_cfg_inv['velocity'].get('background', {}))
        vp_path_i = bg_cfg_i.get('vp_path', VP_PRE_PATH)
        # Align again (idempotent if already aligned path)
        vp_grid_i, vp_path_i, _ainfo_i = load_vp_grid_with_alignment(
            vp_path_i,
            ref_vp_path=VP_SMOOTH_PATH,
            save_aligned_path=None,
            heuristic_threshold=50.0,
            verbose=DEBUG_ALIGN_VP,
        )
        # ---- optional sanity check: does vp increase with depth? (bottom should usually be faster) ----
        if DEBUG_VP_CHECK:
            print(f"[CHECK vp_pre] top mean={vp_grid_i[0,:].mean():.2f}, bottom mean={vp_grid_i[-1,:].mean():.2f}")
            print(f"[CHECK vp_pre] top min/max=({vp_grid_i[0,:].min():.2f},{vp_grid_i[0,:].max():.2f}), bottom min/max=({vp_grid_i[-1,:].min():.2f},{vp_grid_i[-1,:].max():.2f})")
    

        xmin_bg_i = float(bg_cfg_i.get('xmin', sem_cfg_inv['domain']['xmin']))
        xmax_bg_i = float(bg_cfg_i.get('xmax', sem_cfg_inv['domain']['xmax']))
        zmin_bg_i = float(bg_cfg_i.get('zmin', sem_cfg_inv['domain']['zmin']))
        zmax_bg_i = float(bg_cfg_i.get('zmax', sem_cfg_inv['domain']['zmax']))
        nx_bg_i   = int(bg_cfg_i.get('nx', vp_grid_i.shape[1]))
        nz_bg_i   = int(bg_cfg_i.get('nz', vp_grid_i.shape[0]))
        xg_i = np.linspace(xmin_bg_i, xmax_bg_i, nx_bg_i, dtype=np.float64)
        zg_i = np.linspace(zmin_bg_i, zmax_bg_i, nz_bg_i, dtype=np.float64)

        interp_i = str(bg_cfg_i.get('interp', 'linear'))
        v_background_i = build_background_from_grid(
            nodes_xy=global_coords,
            xg=xg_i, zg=zg_i, vp_grid=vp_grid_i,
            method=interp_i,
            clip=True
        )

        # Initial boundary and curve
        ctrl_init = np.asarray(ctrl_pts_init_base, dtype=np.float64).reshape(-1, 2)
        _, _, curve_points_init = build_closed_bspline(ctrl_init, n_samples=800)

        v_inside_i = float(sem_cfg_inv['velocity'].get('anomaly', {}).get('v_inside', 3500.0))
        tau_i = float(sem_cfg_inv['velocity'].get('anomaly', {}).get('tau', sem_cfg_inv['velocity'].get('tau', 10.0)))
        spls_i = int(sem_cfg_inv['velocity'].get('anomaly', {}).get('spline_samples', sem_cfg_inv['velocity'].get('spline_samples', 400)))
        blend_i = str(sem_cfg_inv['velocity'].get('anomaly', {}).get('blend', 'smooth'))

        velocity_model_init, _, _ = build_velocity_2d_background_with_anomaly(
            nodes_xy=global_coords,
            v_background=v_background_i,
            ctrl6_xy=ctrl_init,
            v_inside=v_inside_i,
            tau=tau_i,
            samples=spls_i,
            newton_steps=7,
            blend=blend_i,
        )

        plt.figure(figsize=(10, 8))
        x_coords = global_coords[:, 0]
        z_coords = global_coords[:, 1]
        scatter = plt.scatter(x_coords, z_coords, c=velocity_model_init, cmap='turbo', s=5, marker='s', linewidths=0)
        plt.colorbar(scatter, label='Velocity (m/s)')

        # Boundary curve + nodes
        plt.plot(curve_points_init[:, 0], curve_points_init[:, 1], 'k-', lw=2.2, label='Initial boundary (curve)')
        ctrl_init_closed = np.vstack([ctrl_init, ctrl_init[0]])
        plt.plot(ctrl_init_closed[:, 0], ctrl_init_closed[:, 1], 'ko--',
                 markersize=6, label='Initial B-spline nodes', markeredgecolor='white', markeredgewidth=0.6)

        # Sources / receivers (same geometry)
        source_positions = sem_cfg_inv['source']['positions']
        for i, src_pos in enumerate(source_positions):
            plt.plot(src_pos[0], src_pos[1], 'k*', markersize=15,
                     label='Sources' if i == 0 else "", markeredgecolor='red', markeredgewidth=1.0)
        receiver_positions = np.asarray(sem_cfg_inv['receivers']['positions'])
        plt.plot(receiver_positions[:, 0], receiver_positions[:, 1], 'b^', markersize=8,
                 label='Receivers', markeredgecolor='black', markeredgewidth=0.5)

        xmin = sem_cfg_inv['domain']['xmin']; xmax = sem_cfg_inv['domain']['xmax']
        zmin = sem_cfg_inv['domain']['zmin']; zmax = sem_cfg_inv['domain']['zmax']
        plt.xlim(xmin, xmax); plt.ylim(zmin, zmax)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.title('Initial model: vp_pre background + B-spline anomaly (vp_inside=3500)')
        plt.xlabel('X (m)'); plt.ylabel('Z (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('velocity_model_initial.png', dpi=150)
        plt.close()

        print("[OK] Saved initial-model check figure: velocity_model_initial.png")
    except Exception as _e_plot_init:
        print(f"[WARN] Failed to create initial-model plot: {_e_plot_init}")
else:
    print('[SKIP] DEBUG_PLOT_INIT=0; skipping initial-model check plot.')
# ----------------------------------------------------------------------------------------------------------
# ====================================== step 3: SEM forward simulation (for obs) ====================================

def run_simulation(cfg_in, ctrl_params, noise_std):
    """
    Forward-only SEM simulation for generating observations / quick checks (no backprop).
    - cfg_in: SEMSimulation config (dict)
    - ctrl_params: (K,2) anomaly perturbations/offsets, or None (no anomaly injection here)
    - noise_std: scalar noise std (in same unit as data), or 0/None for noise-free
    """
    cfg0 = copy.deepcopy(cfg_in)

    # Optionally inject anomaly control points into cfg0 (used for inversion background tests).
    if ctrl_params is not None:
        if hasattr(ctrl_params, 'detach'):
            ctrl_params_np = ctrl_params.detach().cpu().numpy().reshape(-1, 2)
        else:
            ctrl_params_np = np.asarray(ctrl_params).reshape(-1, 2)
        cfg0['velocity']['anomaly']['perturbations'] = ctrl_params_np.tolist()
        cfg0['velocity']['perturbations'] = ctrl_params_np.tolist()
    else:
        # No anomaly injection for observations.
        # IMPORTANT: sem_waveform/core.py expects a flat array that can be reshaped to (-1,2).
        # Keep the perturbations already present in cfg_in (typically zeros of length 2*K2).
        if 'anomaly' in cfg0.get('velocity', {}):
            cfg0['velocity']['anomaly'].setdefault('perturbations', [])
        cfg0['velocity'].setdefault('perturbations', [])

    waveforms_list = []
    dt = None
    nt = None
    source_positions = cfg0['source']['positions']

    for k_src, pos in enumerate(source_positions):
        sx, sz = map(float, pos)
        cfg_k = copy.deepcopy(cfg0)
        cfg_k['source'] = cfg_k['source'].copy()
        cfg_k['source']['position'] = [sx, sz]  # single source for this run
        t_build0 = time.perf_counter()
        print(f"[SEM-SETUP] Building SEMSimulation for source {k_src+1}/{len(source_positions)} at (x={sx:.1f}, z={sz:.1f}) ...", flush=True)
        sim = SEMSimulation(cfg_k)
        print(f"[SEM-SETUP] SEMSimulation built in {time.perf_counter()-t_build0:.2f}s", flush=True)
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
        noisy_data = clean_data + torch.normal(mean=torch.zeros_like(clean_data), std=float(noise_std))
    return noisy_data, clean_data, dt, nt

# ======================= Observations cache (shared across stages) =======================
_obs_tag = Path(VP_SMOOTH_PATH).stem
_noisy_path = os.path.join(OBS_CACHE_DIR, f"noisy_data_{_obs_tag}.npy")
_clean_path = os.path.join(OBS_CACHE_DIR, f"clean_data_{_obs_tag}.npy")
_meta_path  = os.path.join(OBS_CACHE_DIR, f"obs_meta_{_obs_tag}.json")

# Optionally force regeneration of observations (delete cache) so obs_noise_std controls the added noise.
if FORCE_REGEN_OBS_CACHE and OBS_CACHE_DIR:
    try:
        for _p in (_noisy_path, _clean_path, _meta_path):
            if os.path.exists(_p):
                os.remove(_p)
        print(f"[OBS_CACHE] FORCE_REGEN_OBS_CACHE=1 -> deleted cached obs files for tag={_obs_tag}", flush=True)
    except Exception as _e_del:
        print(f"[OBS_CACHE][WARN] failed to delete cache files: {_e_del}", flush=True)
time1 = time.perf_counter()
if os.path.exists(_noisy_path) and os.path.exists(_clean_path) and os.path.exists(_meta_path):
    print(f"Loading cached observations from: {OBS_CACHE_DIR}")
    noisy_obs = torch.from_numpy(np.load(_noisy_path)).double()
    clean_obs = torch.from_numpy(np.load(_clean_path)).double()
    with open(_meta_path, "r") as _f:
        _meta = json.load(_f)
    obs_dt = float(_meta["obs_dt"])
    obs_nt = int(_meta["obs_nt"])
else:
    print("Generating observed data with SEM simulator (this may take some seconds)...")
    noisy_obs, clean_obs, actual_dt, actual_nt = run_simulation(sem_cfg_obs, None, noise_std=obs_noise_std)
    # Save observation time-grid parameters
    obs_dt = actual_dt
    obs_nt = actual_nt
    np.save(_noisy_path, noisy_obs.numpy())
    np.save(_clean_path, clean_obs.numpy())
    with open(_meta_path, "w") as _f:
        json.dump({"obs_dt": float(obs_dt), "obs_nt": int(obs_nt)}, _f, indent=2)
    print(f"SEM observations cached. dt={obs_dt:.6f}, nt={obs_nt}")
# Unify naming for downstream plotting/training (cache branch defines obs_dt/obs_nt only)
actual_dt = obs_dt
actual_nt = obs_nt

# --- Common observation aliases used by the training and pruning blocks ---
# y_obs: observed receiver data (nt, n_traces)
# noise_std: assumed observation noise standard deviation (scalar)
y_obs = noisy_obs
noise_std = obs_noise_std

time2 = time.perf_counter()
print(f"Step 2: SEM sim setup time: {time2 - time1:.2f}s")
time_axis = np.arange(actual_nt) * actual_dt
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

# ============================ Apply misfit window/decimation to observations (once) ============================
# If MISFIT_WINDOW['tmin'] is None, pick a conservative first-break-based tmin once from noisy observations.
if MISFIT_WINDOW.get("enabled", False):
    # Work in numpy for speed; keep torch dtype double afterwards
    noisy_np = noisy_obs.detach().cpu().numpy()
    clean_np = clean_obs.detach().cpu().numpy()
    if MISFIT_WINDOW.get("tmin", None) is None:
        MISFIT_WINDOW["tmin"] = _pick_first_break_tmin_np(
            noisy_np, obs_dt, float(sem_config['source'].get('frequency', 20.0)),
            float(MISFIT_WINDOW.get("pick_noise_sec", 0.01)),
            float(MISFIT_WINDOW.get("pick_k_sigma", 8.0)),
            float(MISFIT_WINDOW.get("pick_buffer_cycles", 2.0)),
            time_axis=0,
        )
        print(f"[MISFIT_WINDOW] auto tmin picked = {MISFIT_WINDOW['tmin']:.4f} s", flush=True)

    # If tmax is not explicitly set, allow specifying window_len so that tmax := tmin + window_len.
    # This is resolved ONCE here and then used consistently for both observations and SEM synthetics.
    if MISFIT_WINDOW.get("tmax", None) is None:
        wl = MISFIT_WINDOW.get("window_len", None)
        if wl is not None:
            MISFIT_WINDOW["tmax"] = float(MISFIT_WINDOW["tmin"]) + float(wl)
            print(f"[MISFIT_WINDOW] tmax set by window_len: tmax = tmin + {float(wl):.4f}s -> {MISFIT_WINDOW['tmax']:.4f} s", flush=True)


    # Resolve tmax from window_len even when tmin is provided manually.
    if MISFIT_WINDOW.get("tmax", None) is None:
        wl = MISFIT_WINDOW.get("window_len", None)
        if wl is not None and MISFIT_WINDOW.get("tmin", None) is not None:
            MISFIT_WINDOW["tmax"] = float(MISFIT_WINDOW["tmin"]) + float(wl)
            print(f"[MISFIT_WINDOW] tmax set by window_len: tmax = tmin + {float(wl):.4f}s -> {MISFIT_WINDOW['tmax']:.4f} s", flush=True)
    noisy_win, obs_dt_win, obs_nt_win, _i0, _i1 = _apply_window_and_decim_np(
        noisy_np, obs_dt,
        MISFIT_WINDOW.get("tmin", None),
        MISFIT_WINDOW.get("tmax", None),
        int(MISFIT_WINDOW.get("decim", 1)),
        time_axis=0,
    )
    clean_win, _, _, _, _ = _apply_window_and_decim_np(
        clean_np, obs_dt,
        MISFIT_WINDOW.get("tmin", None),
        MISFIT_WINDOW.get("tmax", None),
        int(MISFIT_WINDOW.get("decim", 1)),
        time_axis=0,
    )
    noisy_obs = torch.from_numpy(noisy_win).double()
    clean_obs = torch.from_numpy(clean_win).double()
    print(f"[MISFIT_WINDOW] obs crop: it=[{_i0}:{_i1}] (dt={obs_dt:.6g}s -> {obs_dt_win:.6g}s), nt={obs_nt} -> {obs_nt_win}", flush=True)
    obs_dt = float(obs_dt_win)
    obs_nt = int(obs_nt_win)

    
    # ============================ Experiment A: normalize waveform amplitude ============================
    # Goal: make waveform amplitudes O(1) to avoid tiny-scale numerics, while keeping SNR unchanged.
    # We compute a single global scale factor on the *windowed/decimated* observations and apply it to:
    #   - noisy_obs, clean_obs, y_obs (data)
    #   - noise_std (assumed noise level)
    #   - SEM source amplitude in BOTH obs/inversion configs (so synthetics match scaled data)
    # ---- Observation amplitude normalization removed (use raw amplitudes) ----
noise_std = float(obs_noise_std)

# FINAL_OBS_ALIAS_BLOCK: ensure downstream always uses the post-window, post-scaling observations and noise_std
y_obs = noisy_obs

# (y_obs already set to noisy_obs above)

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
        clean_blocks = []
        dt_sim = None
        nt_sim = None
        # Parallel helper for a single source (kept inside forward so outer structure stays unchanged)
        def _run_one_source(k_pos):
            k, pos = k_pos
            start_col = k * nrec
            end_col = (k + 1) * nrec
            y_obs_k = y_obs_np[:, start_col:end_col]
            cfg_k = copy.deepcopy(cfg_base)
            # Each SEMSimulation is configured for a single active source
            cfg_k['source']['position'] = [float(pos[0]), float(pos[1])]
            sim_k = get_or_make_sim(cfg_k)
            if not hasattr(sim_k, "run_forward_and_adjoint"):
                raise NotImplementedError(
                    "SEMSimulation.run_forward_and_adjoint(...) is not implemented. "
                    "Please implement this method in sem_waveform/core.py so that it returns: "
                    "{'loglik': float, 'grad_wrt_ctrl': (DIM_Z,), 'clean_data': (nt,nrec), 'dt': float, 'nt': int}"
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
                out_k["clean_data"],
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
        for loglik_k, grad_k, clean_k, dt_k, nt_k in results:
            total_loglik += loglik_k
            grad_wrt_ctrl_accum += grad_k
            clean_blocks.append(clean_k)
            if dt_sim is None:
                dt_sim = dt_k
                nt_sim = nt_k
            else:
                if (abs(dt_sim - dt_k) > 1e-12) or (nt_sim != nt_k):
                    raise ValueError("Inconsistent dt/nt between source runs")
        # Combine clean data blocks if needed (not currently used downstream)
        if len(clean_blocks) > 1:
            clean_data_all = np.concatenate(clean_blocks, axis=1)
        else:
            clean_data_all = clean_blocks[0]
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
    """
    Wrapper that automatically handles time-axis alignment.
    """
    # Ensure y_obs is generated under the current configuration
    result = SEMLikelihoodAdjointFn.apply(z, y_obs, noise_std, sem_cfg_inv, obs_dt, obs_nt)
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
# True PRQ (RQS) Coupling
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
    Rational-Quadratic Spline (RQS)
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
    True monotonic RQS Coupling (NSF style)，tails='linear' [-B, B]。
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
# ================== NormalizingFlow (uses ActNorm + PRQ Coupling) ==================
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
            # Swap in PRQ coupling layer (NSF-style), learned from Gravity_coupling_NSF.py
            coupling = PiecewiseRationalQuadraticCoupling(
                mask=mask,
                transform_net_create_fn=create_net,
                num_bins=K_bins,
                tails="linear",
                tail_bound=FIXED_SIGMA_STAGE1 * 1.5
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
    """Compute the ELBO for the fixed-sigma two-stage workflow.

    Current behavior:
      - Stage-1 uses a fixed Gaussian prior std FIXED_SIGMA_STAGE1 on control-point offsets.
      - Stage-2 uses a fixed Gaussian prior std FIXED_SIGMA_STAGE2 on control-point offsets.

    ELBO:
        E_q [ log p(y|z) + log p(z) + log p_fuse(z) - log q(z) ]
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
            (K2,), float(FIXED_SIGMA_STAGE1), device=z_samples.device, dtype=z_samples.dtype
        )
    elif int(stage) == 2:
        sigma_k = torch.full(
            (K2,), float(FIXED_SIGMA_STAGE2), device=z_samples.device, dtype=z_samples.dtype
        )
    else:
        raise ValueError(f"Unsupported stage={stage}. Only Stage-1 and Stage-2 are supported.")

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
# Default workflow: fixed K2 = K_MAX in Stage-1, then prune/reduce the model for Stage-2.
# This script now uses only the fixed-sigma two-stage workflow.
prior_mean = torch.zeros(DIM_Z, dtype=torch.float64)
# Flow base distribution: standard normal in R^{DIM_Z}
base_mean = torch.zeros(DIM_Z, dtype=torch.float64)
base_cov = torch.eye(DIM_Z, dtype=torch.float64)
model = NormalizingFlow(dim=DIM_Z, num_flows=num_flows, base_mean=base_mean, base_cov=base_cov)
# Fixed prior widths are defined below for Stage-1 and Stage-2.
def _inv_softplus(x: float) -> float:
    # stable inverse softplus for initialization
    # for large x, softplus(rho) ~ rho, so inv_softplus(x) ~ x
    if x > 20.0:
        return float(x)
    return float(math.log(math.expm1(x)))
FIXED_SIGMA_STAGE1 = float(os.environ.get("FIXED_SIGMA_STAGE1", "300.0"))
FIXED_SIGMA_STAGE2 = float(os.environ.get("FIXED_SIGMA_STAGE2", "300.0"))
if STAGE_INT in (1, 2):
    _fixed = FIXED_SIGMA_STAGE1 if (STAGE_INT == 1) else FIXED_SIGMA_STAGE2
    print(f"[STAGE-{STAGE_INT}] Fixed sigma = {_fixed:g} (fixed-sigma two-stage workflow)")

optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
elbo_history = []
log_likelihood_history = []
log_prior_z_history = []
log_q_history = []
log_p_fuse_history = []
gradient_history = []
def plot_posterior_samples(
    epoch,
    post_samples_np,
    *,
    outdir: str | None = None,
    vp_bg: np.ndarray | None = None,
    bg_extent: tuple[float, float, float, float] | None = None,
    zoom: tuple[float, float, float, float] | None = None,  # (xmin,xmax,zmin,zmax)
    nplot: int = 200,
    dpi: int = 200,
):
    """Plot VI geometry snapshots.

    - Overlays B-spline nodes/curve (initial/prior/posterior mean) on top of an optional 2D background velocity.
    - Saves to: <outdir>/posterior_boundary_epoch_XXXX.png
    - If zoom is provided, it zooms to (xmin,xmax,zmin,zmax) in meters.
    """
    # ---- output path ----
    save_dir = "." if outdir is None else outdir
    # Save directly into outdir (no intermediate_posteriors_real folder)
    save_path = os.path.join(save_dir, f"posterior_boundary_epoch_{epoch:04d}.png")

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # ---- background (true 2D model etc.) ----
    if vp_bg is not None and bg_extent is not None:
        # bg_extent = (xmin, xmax, zmin, zmax)
        xmin, xmax, zmin, zmax = bg_extent
        # Use (left,right,bottom,top) = (xmin,xmax,zmax,zmin) so shallow is at the top without invert_yaxis()
        # Match True Model colorbar via fixed vmin/vmax/cmap (can override via env)
        vp_vmin = float(os.environ.get("VP_PLOT_VMIN", "2800"))
        vp_vmax = float(os.environ.get("VP_PLOT_VMAX", "5300"))
        vp_cmap = os.environ.get("VP_PLOT_CMAP", "turbo")
        ax.imshow(
            vp_bg,
            extent=(xmin, xmax, zmax, zmin),
            origin="upper",
            aspect="auto",
            alpha=0.85,
            cmap=vp_cmap,
            vmin=vp_vmin,
            vmax=vp_vmax,
        )
        # colorbar
        cbar = plt.colorbar(ax.images[-1], ax=ax, shrink=0.9)
        cbar.set_label("Velocity (m/s)")

    # ---- initial geometry (the base control points) ----
    ctrl_init_closed = np.vstack([ctrl_pts_init_base, ctrl_pts_init_base[0:1]])
    _, _, curve_init = build_closed_bspline(ctrl_pts_init_base)
    ax.plot(ctrl_init_closed[:, 0], ctrl_init_closed[:, 1], "ro--", markersize=7, label="_nolegend_")
    ax.plot(curve_init[:, 0], curve_init[:, 1], "r-", lw=2.0, label="_nolegend_")

    # ---- prior mean boundary in z-space ----
    prior_mean_np = prior_mean.detach().cpu().numpy().reshape(-1, 2)  # (K2, 2)
    ctrl_pts_prior = ctrl_pts_init_base + prior_mean_np
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0:1]])
    _, _, curve_prior = build_closed_bspline(ctrl_pts_prior)
    ax.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], "ko--", markersize=6, alpha=0.8, label="Prior Mean Points")
    ax.plot(curve_prior[:, 0], curve_prior[:, 1], "k-", lw=1.8, alpha=0.8, label="Prior Mean Boundary")

    # ---- posterior sample boundaries ----
    post_samples_np = np.asarray(post_samples_np)
    if post_samples_np.size > 0:
        nplot_eff = min(int(nplot), int(post_samples_np.shape[0]))
        for i in range(nplot_eff):
            offset_i = post_samples_np[i].reshape(-1, 2)  # (K2, 2)
            ctrl_pts_i = ctrl_pts_init_base + offset_i
            _, _, curve_i = build_closed_bspline(ctrl_pts_i)
            ax.plot(curve_i[:, 0], curve_i[:, 1], "b--", lw=0.5, alpha=0.03)

        # ---- posterior mean boundary ----
        mean_sample = post_samples_np.mean(axis=0).reshape(-1, 2)
        mean_ctrl_pts = ctrl_pts_init_base + mean_sample
        mean_ctrl_closed = np.vstack([mean_ctrl_pts, mean_ctrl_pts[0:1]])
        _, _, curve_mean = build_closed_bspline(mean_ctrl_pts)
        ax.plot(mean_ctrl_closed[:, 0], mean_ctrl_closed[:, 1], "bo--", markersize=7, alpha=0.95, label="Posterior Mean Points")
        ax.plot(curve_mean[:, 0], curve_mean[:, 1], "b-", lw=2.2, alpha=0.95, label="Posterior Mean Boundary")
    else:
        mean_ctrl_pts = None

    # ---- zoom to research window ----
    if zoom is not None:
        zxmin, zxmax, zzmin, zzmax = zoom
        ax.set_xlim(zxmin, zxmax)
        # Keep shallow at top: set ylim(zmax,zmin)
        ax.set_ylim(zzmax, zzmin)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"VI Geometry Snapshot @ Epoch {epoch}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    return save_path, mean_ctrl_pts
# ================================= Step 7: Training =====================
print("Starting training...")
outdir = f"stage{STAGE}_K_{K2:02d}"
os.makedirs(outdir, exist_ok=True)
# --------------------------------------------------------------------------------------
# Snapshot background for plots (overlay B-spline geometry on the true/2D background model)
# --------------------------------------------------------------------------------------
VP_BG_FOR_PLOTS = None
VP_BG_EXTENT = None
try:
    # Use vp_smooth (true background used for observations) as plotting backdrop
    _vp_bg, _, _ = load_vp_grid_with_alignment(
        VP_SMOOTH_PATH,
        ref_vp_path=VP_SMOOTH_PATH,
        save_aligned_path=None,
        heuristic_threshold=50.0,
        verbose=False,
    )
    VP_BG_FOR_PLOTS = _vp_bg
    # Use the observation (true) domain for plotting extent
    VP_BG_EXTENT = (
        float(sem_cfg_obs['domain']['xmin']),
        float(sem_cfg_obs['domain']['xmax']),
        float(sem_cfg_obs['domain']['zmin']),
        float(sem_cfg_obs['domain']['zmax']),
    )  # (xmin,xmax,zmin,zmax)
except Exception as _e:
    print(f"[WARN] Could not load VP_SMOOTH_PATH for snapshot background: {_e}", flush=True)
# ======================= Stage-2: load base control points (prior mean geometry) =======================
# Design:
#   - Stage-2 prior mean = pruned Stage-1 posterior-mean control points
#   - BASE_CTRL_PATH is the preferred source; otherwise try common Stage-1 outputs
if STAGE_INT == 2:
    _base_path = BASE_CTRL_PATH
    if _base_path == "":
        _candidates = [
            os.path.join(f"stage1_K_{K_MAX:02d}", "stage1_mean_ctrl_pts.npy"),
            os.path.join(f"stage1_K_{K_MAX:02d}", "kept_ctrl_pts.npy"),
            os.path.join(f"stage1_K_{K_MAX:02d}", "merged_ctrl_pts.npy"),
        ]
        _base_path = next((p for p in _candidates if os.path.exists(p)), "")
    if (not _base_path) or (not os.path.exists(_base_path)):
        raise FileNotFoundError(
            f"[STAGE-2] BASE_CTRL_PATH not found: {_base_path}. "
            f"Set BASE_CTRL_PATH to Stage-1's stage1_mean_ctrl_pts.npy (preferred) "
            f"or to kept_ctrl_pts.npy / merged_ctrl_pts.npy from Stage-1."
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
        model, noisy_obs, noise_std,
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
    if (epoch % PRINT_EVERY == 0) or (epoch == num_epochs - 1):
        now = time.perf_counter()
        dt_print = now - last_print_time
        last_print_time = now
        print(f"[TRAIN] epoch {epoch:4d}/{num_epochs} | ELBO={elbo.item():.3f} | loglike={log_likelihood.item():.3f} | "
              f"logp(z)={log_p_z.item():.3f} | logq={log_q_z.item():.3f} | grad_norm={total_norm.item():.2e} | Δt={dt_print:6.2f}s")
    
    # quick boundary snapshots (save every PLOT_EVERY epochs)
    if (epoch % PLOT_EVERY == 0) or (epoch == num_epochs - 1):
        with torch.no_grad():
            post_samples, _ = model.forward(n_samples=5000)  # keep 5000 for accurate snapshot (user request)
            snap_path, mean_ctrl_pts = plot_posterior_samples(
                epoch,
                post_samples.cpu().numpy(),
                outdir=outdir,
                vp_bg=VP_BG_FOR_PLOTS,
                bg_extent=VP_BG_EXTENT,
                zoom=ZOOM_WINDOW,
                nplot=200,
                dpi=200,
            )

        # Save posterior-mean control points for this snapshot
        if mean_ctrl_pts is not None:
            mean_path = os.path.join(outdir, f"stage{STAGE}_mean_ctrl_pts_epoch{epoch:04d}.npy")
            np.save(mean_path, mean_ctrl_pts.astype(np.float32))
        print(f"[SNAPSHOT] epoch={epoch:4d} -> {snap_path}", flush=True)
# ---- posterior samples for final analysis ----
posterior_draw = 5000
with torch.no_grad():
    post_samples, _ = model.forward(n_samples=posterior_draw)
    post_samples_np = post_samples.cpu().numpy()
# ---- automatic merging based on posterior-mean control points ----
mean_offset = post_samples_np.mean(axis=0).reshape(-1, 2)  # (K2,2)
mean_ctrl_pts = ctrl_pts_init_base + mean_offset
# [INFO] merge_control_points / AUTO-REPORT disabled in two-stage sensitivity-pruning run.
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
        E_log_p_z=np.asarray(log_prior_z_history, dtype=np.float64),
        neg_E_log_q_z=np.asarray([-v for v in log_q_history], dtype=np.float64),
        # Optional regularization term used by this script: fused/TV contribution
        E_log_p_fuse=np.asarray(log_p_fuse_history, dtype=np.float64),
    )
    print(f"[OUTPUT] Saved raw ELBO-component data: {_elbo_npz_path}")
except Exception as _e:
    print(f"[WARN] Failed to save raw ELBO-component data: {_e}")
print(f"[OUTPUT] Saved outputs in folder: {outdir}")
# ================================= Step 8: Utilities to render requested figures  ====================================
def _save_nf_posterior_distributions(posterior_samples, true_params_flat=None, filename=os.path.join(outdir, 'nf_posterior_distributions.png')):
    # Allow running without any 'true' B-spline parameters
    if true_params_flat is None:
        true_params_flat = np.array([], dtype=float)

    """Draw histograms of each parameter with prior mean, true value (if available), and posterior mean.
    Works for arbitrary DIM_Z = posterior_samples.shape[1]. If true_params_flat has
    fewer entries than DIM_Z (e.g. K2 != K2), only the first len(true_params_flat)
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
    plt.savefig(filename, dpi=150)
    plt.close()
def _save_nf_boundary_comparison(posterior_samples, filename=os.path.join(outdir, 'nf_boundary_comparison.png')):
    """Draw true/prior/posterior-mean boundaries + a cloud of sample boundaries and nodes.
    TRUE boundary uses K2 control points (ctrl_pts_init_base).
    Prior / posterior boundaries use K2 control points (ctrl_pts_init_base) and
    NF offsets living in DIM_Z = 2 * K2.
    """
    posterior_samples = np.asarray(posterior_samples)
    total_samples, dim = posterior_samples.shape
    # Posterior mean (in z-space, DIM_Z)
    mean_offset = np.mean(posterior_samples, axis=0).reshape(-1, 2)  # (K2, 2)
    ctrl_pts_mean = ctrl_pts_init_base + mean_offset  # (K2, 2)
    _, _, curve_points_mean = build_closed_bspline(ctrl_pts_mean)
    # Base boundary (initial K2 control points)
    ctrl_closed_true = np.vstack([ctrl_pts_init_base, ctrl_pts_init_base[0]])
    _, _, curve_points_true_local = build_closed_bspline(ctrl_pts_init_base)
    # PRIOR boundary (zero offset in z-space, i.e. pure ctrl_pts_init_base)
    prior_offset = np.zeros((K2, 2))
    ctrl_pts_prior = ctrl_pts_init_base + prior_offset
    ctrl_closed_prior = np.vstack([ctrl_pts_prior, ctrl_pts_prior[0]])
    _, _, curve_points_prior = build_closed_bspline(ctrl_pts_prior)
    # Prepare figure
    plt.figure(figsize=(12, 10))
    # Base geometry (red; legend hidden)
    plt.plot(curve_points_true_local[:, 0], curve_points_true_local[:, 1], 'r-', lw=3, label='_nolegend_')
    plt.plot(ctrl_closed_true[:, 0], ctrl_closed_true[:, 1], 'ro--', markersize=8, alpha=0.9, label='_nolegend_')
    # Prior (black, K2 nodes)
    plt.plot(curve_points_prior[:, 0], curve_points_prior[:, 1], 'k--', lw=2, label='Prior mean boundary')
    plt.plot(ctrl_closed_prior[:, 0], ctrl_closed_prior[:, 1], 'ko--', markersize=6, alpha=0.7,
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
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
# ================================= Step 9: Final posterior distribution  ====================================
print("Training finished. Generating final posterior samples...")
with torch.no_grad():
    post_samples, _ = model.forward(n_samples=5000)
    post_samples_np = post_samples.cpu().numpy()
    final_snap_path, final_mean_ctrl = plot_posterior_samples(
        num_epochs,
        post_samples_np,
        outdir=outdir,
        vp_bg=VP_BG_FOR_PLOTS,
        bg_extent=VP_BG_EXTENT,
        zoom=ZOOM_WINDOW,
        nplot=200,
        dpi=250,
    )
    print(f"[FINAL SNAPSHOT] saved: {final_snap_path}")
    # Save as numpy file
    post_npy_path = os.path.join(outdir, "posterior_samples.npy")
    np.save(post_npy_path, post_samples_np)
    print(f"Posterior samples saved: {post_npy_path} | shape: {post_samples_np.shape}")
    # Save as model
    # === generate 'nf_posterior_distributions.png' and 'nf_boundary_comparison.png'
    try:
        _save_nf_posterior_distributions(
            posterior_samples=post_samples_np,
            true_params_flat=None,
            filename=os.path.join(outdir, 'nf_posterior_distributions.png')
        )
        print("Saved 'nf_posterior_distributions.png'")
    except Exception:
        pass
    try:
        _save_nf_boundary_comparison(
            posterior_samples=post_samples_np,
            filename=os.path.join(outdir, 'nf_boundary_comparison.png')
        )
        print("Saved 'nf_boundary_comparison.png'")
    except Exception as e:
        print("Failed to save 'nf_boundary_comparison.png':", e)
# Plot ELBO history (retain all original history curves in one figure)
plt.figure(figsize=(10, 6))
plt.plot(elbo_history, lw=2, label='ELBO')
plt.plot(log_likelihood_history, lw=2, label='E[log p(y|z)]')
plt.plot(log_prior_z_history, lw=2, label='E[log p(z)]')
plt.plot([-v for v in log_q_history], lw=2, label='-E[log q(z)]')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('ELBO and components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'elbo_history_real.png'), dpi=150)
plt.close()
# Plot gradient history
plot_gradient_history(gradient_history)
# ---------------- Save posterior-mean control points for each stage ----------------
# ======================= Auto launch Stage-2 =======================
# Stage-1 -> Stage-2 handoff: rerun this script with STAGE_INT=2 using the pruned control points.
# NOTE: AUTO_RUN_STAGE2 is already defined near the top with .strip(); do NOT redefine it here.

def _run_next(_env):
    try:
        _subprocess.run([sys.executable, __file__], env=_env, check=True)
    except Exception as _e:
        print(f"[AUTO] Failed to launch Stage-2: {_e}")

if (STAGE_INT == 1) and AUTO_RUN_STAGE2:
    # ======================= Stage-1 -> Stage-2 posterior-variance pruning =======================
    # ... (comments kept conceptually same; implementation is indentation-safe) ...

    # --------- User-tunable knobs (put alpha up front) ----------###########################################################################
    PRUNE_ALPHA = float(os.environ.get("PRUNE_ALPHA", "1.5"))   # threshold multiplier
    MIN_KEEP = int(os.environ.get("PRUNE_MIN_KEEP", "6"))       # safety valve
    MAX_DROP_FRAC = float(os.environ.get("PRUNE_MAX_DROP_FRAC", "0.70"))  # safety: do not drop >70% by default

    stage1_dir = outdir
    epoch_candidates = sorted(glob.glob(os.path.join(stage1_dir, "stage1_mean_ctrl_pts_epoch*.npy")))
    mean_ctrl_path = epoch_candidates[-1] if len(epoch_candidates) else os.path.join(stage1_dir, "stage1_mean_ctrl_pts.npy")
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

        # Primary keep: retain nodes whose sensitivity score is within quantile threshold
        keep = (score <= thr_score)
        keep_idx = np.where(keep)[0].tolist()


# ---------------- Occam / evidence quick-check ----------------
# Goal: refine the score-threshold keep-set before Stage-2 by comparing expected log-joint
# (log-likelihood + log-prior).
#
# STRICT-DELETE mode:
#   - We really delete a node: K -> K-1 (not "set dx,dz to 0").
#   - We rebuild the closed control-point set and re-run forward modeling.
#   - Greedy update: once a drop is accepted, the reduced model becomes the new baseline.
#   - Deterministic expectation: use {mean, mean±1σ, mean±2σ} representative samples.
#   - Default OCCAM_TOL=0: only drop when the expected log-joint does not get worse.
#
# Note:
#   - This is a refinement of the initial score-threshold pruning.
#   - It does not recover nodes removed by the first thresholding step.

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
    # score <= PRUNE_ALPHA * quantile(score, PRUNE_Q). We defensively verify those inputs exist.
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
                keep = np.ones(_K_local, dtype=bool)
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
                _occam_log(f"[PRUNE][SCORE] PRUNE_ALPHA={PRUNE_ALPHA:g} | PRUNE_Q={PRUNE_Q:.2f} | score_q={score_q:.6g} | thr={thr_score:.6g}")
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

            # Prior std for z dims: use the same fixed Stage-1 sigma as training.
            sigma_z = float(FIXED_SIGMA_STAGE1)
            if sigma_z <= 0:
                raise ValueError("sigma_z must be positive")

            # ---- No-grad SEM log-likelihood for an arbitrary ctrl-point set (K can change) ----
            # This bypasses SEMLikelihoodAdjointFn (which assumes fixed K and uses ctrl_pts_init_base + z_off).
            def _sem_loglik_no_grad_from_ctrl(abs_ctrl_pts: np.ndarray) -> float:
                abs_ctrl_pts = np.asarray(abs_ctrl_pts, dtype=np.float32)
                cfg_base = copy.deepcopy(sem_config)
                y_obs_np = y_obs.detach().cpu().numpy()
                noise_var = float(noise_std) ** 2 + EPS

                src_positions = cfg_base['source'].get('positions', None)
                if not src_positions:
                    raise ValueError("Occam strict-delete: 'source.positions' missing/empty in sem_config")
                nrec = len(cfg_base['receivers']['positions'])
                nsrc = len(src_positions)
                if y_obs_np.shape[1] != nsrc * nrec:
                    raise ValueError(
                        f"Occam strict-delete: expected {nsrc*nrec} traces ({nsrc}*{nrec}), got {y_obs_np.shape[1]}"
                    )

                # serial loop is fine here (Occam is already expensive); keep it deterministic and stable
                total_ll = 0.0
                for k, pos in enumerate(src_positions):
                    start_col = k * nrec
                    end_col = (k + 1) * nrec
                    y_obs_k = y_obs_np[:, start_col:end_col]
                    cfg_k = copy.deepcopy(cfg_base)
                    cfg_k['source']['position'] = [float(pos[0]), float(pos[1])]
                    sim_k = get_or_make_sim(cfg_k)
                    out_k = sim_k.run_forward_and_adjoint({
                        'bspline_ctrl': abs_ctrl_pts,
                        'y_obs': y_obs_k,
                        'obs_dt': float(obs_dt),
                        'obs_nt': int(obs_nt),
                        'noise_std': float(np.sqrt(noise_var))
                    })
                    total_ll += float(out_k["loglik"])
                return float(total_ll)

            # ---- Helper: build abs ctrl pts for an active index set and a z-offset vector ----
            def _abs_ctrl_from_z(active_idx: list[int], z_vec: np.ndarray) -> np.ndarray:
                z_off = np.asarray(z_vec, dtype=np.float32).reshape(K, 2)  # (K,2) in original Stage-1 indexing
                idx = np.asarray(active_idx, dtype=int)
                base = np.asarray(ctrl_pts_init_base, dtype=np.float32)[idx, :]  # (K_active,2)
                off  = z_off[idx, :]                                             # (K_active,2)
                return base + off

            # ---- Prior on active dims only (Gaussian with sigma_z) ----
            def _logprior_active(z_vec: np.ndarray, active_idx: list[int]) -> float:
                z_off = np.asarray(z_vec, dtype=np.float32).reshape(K, 2)
                idx = np.asarray(active_idx, dtype=int)
                zz = z_off[idx, :].reshape(-1)  # (2*K_active,)
                # log N(0, sigma_z^2)
                return float(np.sum(-0.5 * (zz / sigma_z) ** 2 - np.log(sigma_z) - 0.5 * np.log(2.0 * np.pi)))

            # ---- Deterministic representative batch in z-space ----
            def _det_z_batch() -> list[np.ndarray]:
                zs = []
                for ksig in OCCAM_KSIGS:
                    zs.append(z_mean + float(ksig) * z_std)
                return zs

            det_zs = _det_z_batch()

            # Expected log-joint for a given active set
            def _expected_logjoint_active(active_idx: list[int]) -> float:
                vals = []
                for z_vec in det_zs:
                    abs_ctrl = _abs_ctrl_from_z(active_idx, z_vec)
                    ll = _sem_loglik_no_grad_from_ctrl(abs_ctrl)
                    lp = _logprior_active(z_vec, active_idx)
                    vals.append(ll + lp)
                return float(np.mean(vals))

            # ---- Initialize active set from score-threshold pruning ----
            # (keep_idx was computed above from score <= PRUNE_ALPHA * quantile(score, PRUNE_Q))
            keep_idx = list(keep_idx)  # ensure list
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
                            ll = _sem_loglik_no_grad_from_ctrl(abs_ctrl)
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
            print("[PRUNE] Stage-1 anti-artifact pruning (normal-std + optional curvature + Occam)")
            print(f"[PRUNE] K_stage1={K} | PRUNE_ALPHA={PRUNE_ALPHA:.3g} | PRUNE_Q={PRUNE_Q:.2f} | score_q={score_q:.4g} | thr={thr_score:.4g}")
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
