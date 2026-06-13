# Clean single-stage fixed-K Normalizing Flow script with SEM-based simulator core
# - Uses the SEM wave simulator for observations and ELBO computation.
# - Uses ActNorm + Piecewise Rational Quadratic (PRQ) coupling for invertibility.
# - Single-stage fixed-K diagnostic run.
# Xingdaj@mun.ca
# 2026.6.12
# =============================================================================
# 1. Imports and fixed global plotting / runtime style
# =============================================================================
import os
import sys
import time
import copy
import math
import json
import random
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
from scipy.interpolate import BSpline

# Font/style for all generated figures.
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
matplotlib.rcParams['axes.titlesize'] = 22
matplotlib.rcParams['axes.labelsize'] = 19
matplotlib.rcParams['xtick.labelsize'] = 17
matplotlib.rcParams['ytick.labelsize'] = 17
matplotlib.rcParams['legend.fontsize'] = 13

# CPU threading: avoid oversubscription when multiple sources are run in parallel.
try:
    torch.set_num_threads(8)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

torch.set_default_dtype(torch.float64)

# SEM package import. Keep the script directory on sys.path for local sem_waveform/.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sem_waveform.core import SEMSimulation  # must provide run_forward_and_adjoint


# =============================================================================
# 2. User-editable runtime parameters
# =============================================================================
# In normal use, modify only this block.  All parameters can also be overridden
# by environment variables with the same names.

# --- Reproducibility ---
SEED = int(os.environ.get("SEED", "42"))

# --- Model files ---
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR / "model1"))).expanduser()

VP_SMOOTH_PATH = os.environ.get("VP_SMOOTH_PATH", str(MODEL_DIR / "vp_true_pml.npy"))
VP_META_PATH = os.environ.get("VP_META_PATH", str(MODEL_DIR / "vp_true_pml.json"))
VP_PRE_PATH = os.environ.get("VP_PRE_PATH", str(MODEL_DIR / "vp_initial_pml.npy"))
VP_PRE_META_PATH = os.environ.get("VP_PRE_META_PATH", str(MODEL_DIR / "vp_initial_pml.json"))
VP_BACKGROUND_PATH = os.environ.get("VP_BACKGROUND_PATH", str(MODEL_DIR / "vp_background_pml.npy"))
VP_BACKGROUND_META_PATH = os.environ.get("VP_BACKGROUND_META_PATH", str(MODEL_DIR / "vp_background_pml.json"))

# --- Fixed-K single-stage NF / prior settings ---
FIXED_SIGMA = float(os.environ.get("FIXED_SIGMA", "300.0"))
FUSE_LAMBDA_EFF = float(os.environ.get("FUSE_LAMBDA_EFF", "0.00"))
FUSE_EPS = 1e-3

# --- SEM acquisition and simulation settings ---
# Domain bounds default to the model metadata. Set the following environment
# variables only when you need to override the metadata bounds.
DOMAIN_XMIN_OVERRIDE = os.environ.get("DOMAIN_XMIN", None)
DOMAIN_XMAX_OVERRIDE = os.environ.get("DOMAIN_XMAX", None)
DOMAIN_ZMIN_OVERRIDE = os.environ.get("DOMAIN_ZMIN", None)
DOMAIN_ZMAX_OVERRIDE = os.environ.get("DOMAIN_ZMAX", None)
NELEM_X = int(os.environ.get("NELEM_X", "30"))
NELEM_Z = int(os.environ.get("NELEM_Z", "30"))
TOTAL_TIME = float(os.environ.get("TOTAL_TIME", "1.5"))
DT = float(os.environ.get("DT", "0.80e-4"))
POLYNOMIAL_ORDER = int(os.environ.get("POLYNOMIAL_ORDER", "5"))
PML_THICKNESS = float(os.environ.get("PML_THICKNESS", "300.0"))
ADJ_HISTORY_DTYPE = os.environ.get("ADJ_HISTORY_DTYPE", "float32")
VP_INTERP = os.environ.get("VP_INTERP", "linear")

SOURCE_POSITIONS = [
    [100.0, 20.0],
    [500.0, 20.0],
    [900.0, 20.0],
    [1300.0, 20.0],
    [1700.0, 20.0],
]
SOURCE_FREQUENCY = float(os.environ.get("SOURCE_FREQUENCY", "15.0"))
SOURCE_AMPLITUDE = float(os.environ.get("SOURCE_AMPLITUDE", "1.0e4"))
RECEIVER_XMIN = int(os.environ.get("RECEIVER_XMIN", "0"))
RECEIVER_XMAX = int(os.environ.get("RECEIVER_XMAX", "2000"))
RECEIVER_DX = int(os.environ.get("RECEIVER_DX", "20"))
RECEIVER_Z = float(os.environ.get("RECEIVER_Z", "20.0"))

MISFIT_WINDOW = {
    "enabled": False,
    "tmin": 0.0,
    "tmax": None,
    "window_len": None,
    "decim": 1,
    "pick_noise_sec": 0.01,
    "pick_k_sigma": 8.0,
    "pick_buffer_cycles": 2.0,
}

# --- Velocity/anomaly settings ---
VEL_VMIN = float(os.environ.get("VEL_VMIN", "340.0"))
VEL_VMAX = float(os.environ.get("VEL_VMAX", "3200.0"))
VEL_TAU = float(os.environ.get("VEL_TAU", "10.0"))
VEL_SPLINE_SAMPLES = int(os.environ.get("VEL_SPLINE_SAMPLES", "400"))
ANOMALY_ENABLED = True
ANOMALY_V_INSIDE_OVERRIDE = os.environ.get("ANOMALY_V_INSIDE", None)
ANOMALY_TAU_OVERRIDE = os.environ.get("ANOMALY_TAU", None)
ANOMALY_SPLINE_SAMPLES = int(os.environ.get("ANOMALY_SPLINE_SAMPLES", "400"))
ANOMALY_BLEND = os.environ.get("ANOMALY_BLEND", "smooth")

# --- Observation cache / outputs ---
OBS_CACHE_DIR = os.environ.get("OBS_CACHE_DIR", "obs_cache").strip() or "obs_cache"
FORCE_REGEN_OBS_CACHE = (os.environ.get("FORCE_REGEN_OBS_CACHE", "1").strip() == "1")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "sem_output")
SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", str(10**9)))
POSTERIOR_OUTDIR = os.environ.get("POSTERIOR_OUTDIR", "posterior_model")

# --- Figure outputs ---
MODEL_PLOT_VMIN = float(os.environ.get("MODEL_PLOT_VMIN", "1800.0"))
MODEL_PLOT_VMAX = float(os.environ.get("MODEL_PLOT_VMAX", "3200.0"))
MODEL_PLOT_CMAP = os.environ.get("MODEL_PLOT_CMAP", "turbo")
DEBUG_PLOT_INIT = (os.environ.get("DEBUG_PLOT_INIT", "1").strip() == "1")
SAVE_TRACE_COMPARISON = (os.environ.get("SAVE_TRACE_COMPARISON", "1").strip() == "1")
TRACE_COMPARISON_DIR = os.environ.get("TRACE_COMPARISON_DIR", "source_waveforms")
TRACE_COMPARISON_MAX_TRACES = int(os.environ.get("TRACE_COMPARISON_MAX_TRACES", "20"))
PLOT_MAX_POSTERIOR_CURVES = int(os.environ.get("PLOT_MAX_POSTERIOR_CURVES", "80"))
PLOT_SNAPSHOT_SAMPLES = int(os.environ.get("PLOT_SNAPSHOT_SAMPLES", "10000"))
FINAL_POST_SAMPLES = int(os.environ.get("FINAL_POST_SAMPLES", "10000"))
ZOOM_WINDOW = (
    float(os.environ.get("ZOOM_XMIN", "0.0")),
    float(os.environ.get("ZOOM_XMAX", "2000.0")),
    float(os.environ.get("ZOOM_ZMIN", "0.0")),
    float(os.environ.get("ZOOM_ZMAX", "1000.0")),
)

# --- Optimization settings ---
EPS64 = float(os.environ.get("EPS", "1e-30"))
EPS = EPS64
num_epochs = int(os.environ.get("NUM_EPOCHS", "100"))
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1"))
PLOT_EVERY = int(os.environ.get("PLOT_EVERY", "5"))
min_elbo_samples = int(os.environ.get("MIN_ELBO_SAMPLES", "4"))
max_elbo_samples = int(os.environ.get("MAX_ELBO_SAMPLES", "8"))
learning_rate = float(os.environ.get("LEARNING_RATE", "5e-4"))
num_flows = int(os.environ.get("NUM_FLOWS", "16"))
K_bins = int(os.environ.get("K_BINS", "12"))
# RQS coupling tail bound in physical control-point offset units (m).
# Previously this used undefined TSTD_DEFAULT; for the current fixed-K script,
# FIXED_SIGMA is the intended physical offset scale.
FLOW_TAIL_BOUND = float(os.environ.get("FLOW_TAIL_BOUND", str(float(FIXED_SIGMA) * 1.5)))
obs_noise_std = float(os.environ.get("OBS_NOISE_STD", "1.0e-4"))
max_grad_norm = float(os.environ.get("MAX_GRAD_NORM", "100.0"))
clip_gradient = (os.environ.get("CLIP_GRADIENT", "1").strip() == "1")


# =============================================================================
# 3. Helper functions and classes
# =============================================================================

_SIM_CACHE = {}
_JSON_CACHE = {}
_NPY_CACHE = {}
_MODEL_PLOT_CMAP_CACHE = None
_PLOT_BACKGROUND_CACHE = None

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

    # Determinism flags (best-effort; some ops/devices may still be non-deterministic)
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

def _sim_key_from_cfg(cfg):
    """Build a cache key that supports both layered and model1/background_2d configs."""
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
        pos_single = tuple(src_positions[0]) if src_positions else (0.0, 0.0)

    key_core = (
        float(dom['xmin']), float(dom['xmax']), float(dom['zmin']), float(dom['zmax']),
        int(dom['nelem_x']), int(dom['nelem_z']),
        int(meth.get('polynomial_order', 0)), float(meth.get('pml_thickness', 0.0)),
        float(tm['dt']), float(tm['total_time']),
        pos_single, src_positions, rec_positions,
        float(src.get('frequency', 0.0)), float(src.get('amplitude', 1.0)),
        str(meth.get('adj_history_dtype', '')),
    )

    bg = vel.get('background_2d', vel.get('background', {})) or {}
    if bg:
        bg_meta = (
            str(bg.get('vp_path', '')),
            float(bg.get('xmin', dom['xmin'])), float(bg.get('xmax', dom['xmax'])), int(bg.get('nx', 0)),
            float(bg.get('zmin', dom['zmin'])), float(bg.get('zmax', dom['zmax'])), int(bg.get('nz', 0)),
            str(bg.get('interp', 'linear')),
        )
    else:
        layers = vel.get('layers', {}) or {}
        bg_meta = (
            tuple(float(v) for v in layers.get('velocities', [])),
            tuple(float(v) for v in layers.get('interfaces_z', [])),
        )

    an = vel.get('anomaly', {}) or {}

    def _arr_sig(obj):
        if obj is None:
            return None
        arr = np.asarray(obj, dtype=np.float64).reshape(-1)
        return tuple(np.round(arr, 10).tolist())

    mw = meth.get('misfit_window', {}) or {}
    mw_meta = (
        bool(mw.get('enabled', False)),
        None if mw.get('tmin', None) is None else float(mw.get('tmin')),
        None if mw.get('tmax', None) is None else float(mw.get('tmax')),
        None if mw.get('window_len', None) is None else float(mw.get('window_len')),
        int(mw.get('decim', 1)),
    )
    an_meta = (
        bool(an.get('enabled', True)),
        float(an.get('v_inside', vel.get('vmin', 0.0))),
        float(an.get('tau', vel.get('tau', 0.0))),
        int(an.get('spline_samples', vel.get('spline_samples', 0))),
        str(an.get('blend', 'smooth')),
        bool(meth.get('VERIFY_PROJECTION', False)),
        str(an.get('boundary_to_grid_method', '')),
        _arr_sig(vel.get('control_points', an.get('control_points', None))),
        _arr_sig(vel.get('perturbations', an.get('perturbations', None))),
    )
    return key_core + bg_meta + mw_meta + an_meta
def get_or_make_sim(cfg):
    key = _sim_key_from_cfg(cfg)
    sim = _SIM_CACHE.get(key)
    if sim is None:
        sim = SEMSimulation(cfg)
        _SIM_CACHE[key] = sim
    return sim

def _read_json_if_exists(json_path: str) -> dict:
    json_path = str(json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON metadata not found: {json_path}")
    st = os.stat(json_path)
    key = (json_path, st.st_mtime_ns, st.st_size)
    if key not in _JSON_CACHE:
        with open(json_path, "r", encoding="utf-8") as f:
            _JSON_CACHE[key] = json.load(f)
    return _JSON_CACHE[key]

def _load_npy_cached(npy_path: str, *, dtype=np.float64):
    key = (str(npy_path), str(np.dtype(dtype)))
    if key not in _NPY_CACHE:
        _NPY_CACHE[key] = np.load(str(npy_path)).astype(dtype, copy=False)
    return _NPY_CACHE[key]

def _load_control_points_from_json(json_path: str, *, label: str):
    meta = _read_json_if_exists(json_path)
    anomaly = meta.get("anomaly", {}) or {}
    pts = anomaly.get("initial_control_points", None)
    if pts is None:
        pts = anomaly.get("control_points", None)
    if pts is None:
        raise KeyError(f"{label}: no anomaly.initial_control_points or anomaly.control_points in {json_path}")
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError(f"{label}: need at least 3 control points for quadratic B-spline, got {pts.shape[0]}")
    print(f"[OK] Loaded {label} control points from {os.path.basename(json_path)}: K={pts.shape[0]}")
    return pts

def _read_anomaly_value(meta_path: str, default: float = 2000.0) -> float:
    meta = _read_json_if_exists(meta_path)
    anomaly = meta.get("anomaly", {}) or {}
    for key in ("vp_inside_m_per_s", "v_inside", "v_inside_m_per_s", "velocity_inside", "vp_inside"):
        if key in anomaly:
            return float(anomaly[key])
    return float(default)

def _read_anomaly_tau_from_meta(meta_path: str, default: float = 10.0) -> float:
    meta = _read_json_if_exists(meta_path)
    anomaly = meta.get("anomaly", {}) or {}
    # Prefer the actual signed-distance sigmoid tau used by the SEM model.
    for key in ("tau_m", "interface_tau_m", "tau", "sigmoid_tau_m"):
        if anomaly.get(key, None) is not None:
            return float(anomaly[key])
    grid = meta.get("grid", {}) or {}
    smoothing = meta.get("smoothing", {}) or {}
    vals = []
    if smoothing.get("sigma_x_grid_cells", None) is not None and grid.get("dx", None) is not None:
        vals.append(float(smoothing["sigma_x_grid_cells"]) * float(grid["dx"]))
    if smoothing.get("sigma_z_grid_cells", None) is not None and grid.get("dz", None) is not None:
        vals.append(float(smoothing["sigma_z_grid_cells"]) * float(grid["dz"]))
    return float(np.mean(vals)) if vals else float(default)

def _read_grid_meta(meta_path: str):
    meta = _read_json_if_exists(meta_path)
    grid = meta.get("grid", meta) or {}
    required = ("xmin", "xmax", "zmin", "zmax", "nx", "nz")
    missing = [k for k in required if k not in grid]
    if missing:
        raise KeyError(f"Missing grid metadata keys {missing} in {meta_path}")
    return grid

def _extract_model_signature(meta_path: str, label: str) -> dict:
    meta = _read_json_if_exists(meta_path)
    grid = meta.get("grid", {}) or {}
    smoothing = meta.get("smoothing", {}) or {}
    layers = meta.get("background_layers", {}) or {}
    return {
        "label": label,
        "grid_bounds": tuple(float(grid[k]) for k in ("xmin", "xmax", "zmin", "zmax")),
        "grid_shape": (int(grid.get("nx")), int(grid.get("nz"))),
        "grid_spacing": (float(grid.get("dx")), float(grid.get("dz"))),
        "interfaces": tuple(float(v) for v in layers.get("interfaces_z_m", [])),
        "velocities": tuple(float(v) for v in layers.get("velocities_m_per_s", [])),
        "smooth_applied": bool(smoothing.get("applied", False)),
        "smooth_method": str(smoothing.get("method", "")),
        "smooth_sigma": (
            float(smoothing.get("sigma_x_grid_cells", np.nan)),
            float(smoothing.get("sigma_z_grid_cells", np.nan)),
        ),
    }

def _check_model_generation_consistency() -> None:
    sig_true = _extract_model_signature(VP_META_PATH, "true")
    sig_init = _extract_model_signature(VP_PRE_META_PATH, "initial")
    sig_bg = _extract_model_signature(VP_BACKGROUND_META_PATH, "background")

    # All three files must share grid and layered-background definition.
    for sig in (sig_init, sig_bg):
        for ksig in ("grid_bounds", "grid_shape", "grid_spacing", "interfaces", "velocities"):
            a, b = sig_true[ksig], sig[ksig]
            if isinstance(a, tuple) and any(isinstance(x, float) for x in a):
                if len(a) != len(b) or not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True):
                    raise ValueError(f"model1 {ksig} mismatch: true={a}, {sig['label']}={b}")
            elif a != b:
                raise ValueError(f"model1 {ksig} mismatch: true={a}, {sig['label']}={b}")

    # True and initial should use the same anomaly smoothing/mapping metadata.
    for ksig in ("smooth_applied", "smooth_method", "smooth_sigma"):
        a, b = sig_true[ksig], sig_init[ksig]
        if isinstance(a, tuple) and any(isinstance(x, float) for x in a):
            if len(a) != len(b) or not np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True):
                raise ValueError(f"model1 {ksig} mismatch: true={a}, initial={b}")
        elif a != b:
            raise ValueError(f"model1 {ksig} mismatch: true={a}, initial={b}")

    print("[CHECK] model1 grid/layer metadata are consistent; true/initial anomaly mapping metadata are consistent.")

def _get_model_plot_cmap():
    """Return the truncated velocity colormap used by overview figures."""
    global _MODEL_PLOT_CMAP_CACHE
    if _MODEL_PLOT_CMAP_CACHE is None:
        base = matplotlib.colormaps.get_cmap(MODEL_PLOT_CMAP)
        colors = base(np.linspace(0.10, 1.0, 256))
        _MODEL_PLOT_CMAP_CACHE = LinearSegmentedColormap.from_list(
            f"trunc_{MODEL_PLOT_CMAP}", colors
        )
    return _MODEL_PLOT_CMAP_CACHE

def build_closed_bspline(ctrl_pts_base, num_samples=800):
    pts = np.asarray(ctrl_pts_base, dtype=float).reshape(-1, 2)
    if pts.shape[0] < k + 1:
        raise ValueError(f"At least {k + 1} control points are required for degree {k}; got {pts.shape[0]}.")
    ctrl_ext = np.vstack([pts, pts[:k]])
    knots = np.arange(0.0, len(ctrl_ext) + k + 1, dtype=float)
    spline = BSpline(knots, ctrl_ext, k, extrapolate=False)
    t_curve = np.linspace(float(k), float(len(pts) + k), int(num_samples), endpoint=False)
    curve_points = np.asarray(spline(t_curve), dtype=float)
    curve_points = np.vstack([curve_points, curve_points[0]])
    return knots, t_curve, curve_points



def _plot_meta_from_json(json_path: str, vp_grid: np.ndarray):
    """Return full-grid metadata, physical plotting bounds, and color scale."""
    meta = _read_json_if_exists(json_path)
    grid = meta.get("grid", {}) if isinstance(meta, dict) else {}
    if not grid:
        grid = {
            "xmin": DOMAIN_XMIN, "xmax": DOMAIN_XMAX,
            "zmin": DOMAIN_ZMIN, "zmax": DOMAIN_ZMAX,
            "nx": vp_grid.shape[1], "nz": vp_grid.shape[0],
        }
    physical = meta.get("physical_subdomain", None) if isinstance(meta, dict) else None
    if not physical:
        # Hide the PML/air padding by default.
        physical = {"xmin": 0.0, "xmax": 2000.0, "zmin": 0.0, "zmax": 1000.0}
    smoothing = meta.get("smoothing", {}) if isinstance(meta, dict) else {}
    vmin = float(smoothing.get("plot_vmin_m_per_s", MODEL_PLOT_VMIN))
    vmax = float(smoothing.get("plot_vmax_m_per_s", MODEL_PLOT_VMAX))
    return grid, physical, vmin, vmax


def _crop_vp_to_physical_domain(vp_grid: np.ndarray, grid: dict, physical: dict):
    """Crop vp_grid from full PML grid to the physical domain for plotting only."""
    nx = int(grid.get("nx", vp_grid.shape[1]))
    nz = int(grid.get("nz", vp_grid.shape[0]))
    xmin = float(grid.get("xmin", DOMAIN_XMIN))
    xmax = float(grid.get("xmax", DOMAIN_XMAX))
    zmin = float(grid.get("zmin", DOMAIN_ZMIN))
    zmax = float(grid.get("zmax", DOMAIN_ZMAX))
    x = np.linspace(xmin, xmax, nx, dtype=float)
    z = np.linspace(zmin, zmax, nz, dtype=float)

    pxmin = float(physical.get("xmin", 0.0))
    pxmax = float(physical.get("xmax", 2000.0))
    pzmin = float(physical.get("zmin", 0.0))
    pzmax = float(physical.get("zmax", 1000.0))

    ix = (x >= pxmin - 1e-9) & (x <= pxmax + 1e-9)
    iz = (z >= pzmin - 1e-9) & (z <= pzmax + 1e-9)
    if not np.any(ix) or not np.any(iz):
        print("[WARN] Physical-domain crop failed; plotting full grid instead.")
        return vp_grid, xmin, xmax, zmin, zmax
    return vp_grid[np.ix_(iz, ix)], float(x[ix][0]), float(x[ix][-1]), float(z[iz][0]), float(z[iz][-1])


def plot_velocity_overview_from_grid(
    vp_grid: np.ndarray,
    meta_path: str,
    title: str,
    save_path: str,
    ctrl_pts=None,
    boundary_label="B-spline boundary",
    nodes_label="B-spline nodes",
):
    """Plot velocity model without PML and without SEM mesh/grid lines."""
    grid, physical, vmin, vmax = _plot_meta_from_json(meta_path, vp_grid)
    vp_plot, x0, x1, z0, z1 = _crop_vp_to_physical_domain(vp_grid, grid, physical)

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    im = ax.imshow(
        vp_plot,
        extent=(x0, x1, z1, z0),
        origin="upper",
        cmap=_get_model_plot_cmap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    if ctrl_pts is not None:
        ctrl_pts = np.asarray(ctrl_pts, dtype=np.float64).reshape(-1, 2)
        if ctrl_pts.shape[0] >= 4:
            _, _, curve = build_closed_bspline(ctrl_pts, num_samples=800)
            ax.plot(curve[:, 0], curve[:, 1], "k-", lw=2.2, label=boundary_label, zorder=5)
            closed = np.vstack([ctrl_pts, ctrl_pts[0]])
            ax.plot(
                closed[:, 0], closed[:, 1], linestyle="--", color="black", lw=1.6,
                marker="o", ms=6.8, mfc="black", mec="black", mew=0.9,
                label=nodes_label, zorder=6,
            )

    rec = np.asarray(sem_config["receivers"]["positions"], dtype=float)
    if rec.size:
        ax.plot(rec[:, 0], rec[:, 1], linestyle="None", marker="^", ms=7.6,
                mfc="yellow", mec="black", mew=0.70, label="Receivers", zorder=7)
    src = np.asarray(sem_config["source"]["positions"], dtype=float)
    if src.size:
        ax.plot(src[:, 0], src[:, 1], linestyle="None", marker="*", ms=17,
                mfc="red", mec="white", mew=1.05, label="Sources", zorder=9)

    ax.set_xlim(x0, x1)
    ax.set_ylim(z1, z0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.1, length=6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.0%", pad=0.14)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity (m/s)", fontsize=19)
    cbar.set_ticks(np.linspace(vmin, vmax, 8))
    cbar.ax.tick_params(labelsize=17, width=1.0, length=4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _save_all_source_waveform_figures(clean_obs, noisy_obs, actual_dt, actual_nt, sem_cfg_obs):
    """Save one clean/noisy waveform PNG for each source into source_waveforms/."""
    if not SAVE_TRACE_COMPARISON:
        print("[SKIP] SAVE_TRACE_COMPARISON=0; skipping source waveform figures.", flush=True)
        return

    def _as_numpy(arr):
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    clean_np = _as_numpy(clean_obs)
    noisy_np = _as_numpy(noisy_obs)
    time_axis = np.arange(int(actual_nt), dtype=float) * float(actual_dt)
    n_total_traces = int(noisy_np.shape[1])
    nsrc_plot = len(sem_cfg_obs.get("source", {}).get("positions", []))
    nrec_plot = len(sem_cfg_obs.get("receivers", {}).get("positions", []))
    rec_positions_plot = np.asarray(sem_cfg_obs.get("receivers", {}).get("positions", []), dtype=float)

    os.makedirs(TRACE_COMPARISON_DIR, exist_ok=True)

    def _save_trace_comparison_figure(save_path, trace_idx_show, *, rec_idx_show=None,
                                      src_idx=None, title_prefix=""):
        n_panels = len(trace_idx_show)
        fig, axes = plt.subplots(n_panels, 1, figsize=(15, 2.2 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        for ip, tr_idx in enumerate(trace_idx_show):
            ax = axes[ip]
            tr_idx = int(tr_idx)
            ax.plot(time_axis, clean_np[:, tr_idx], "k-", lw=1.5, label="Clean")
            ax.plot(time_axis, noisy_np[:, tr_idx], "r-", lw=1.0, alpha=0.7, label="Noisy")
            if rec_idx_show is not None and src_idx is not None and rec_positions_plot.size:
                ridx = int(rec_idx_show[ip])
                rx, rz = rec_positions_plot[ridx]
                ax.set_title(
                    f"{title_prefix}Source {src_idx + 1}/{nsrc_plot}, Receiver {ridx + 1}/{nrec_plot}, "
                    f"x={float(rx):.1f} m, z={float(rz):.1f} m"
                )
            else:
                ax.set_title(f"{title_prefix}Trace {tr_idx + 1}/{n_total_traces}")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            if ip == 0:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time(s)")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    if nsrc_plot > 0 and nrec_plot > 0 and n_total_traces == nsrc_plot * nrec_plot:
        n_show = min(int(TRACE_COMPARISON_MAX_TRACES), nrec_plot)
        rec_idx_show = np.unique(np.linspace(0, nrec_plot - 1, n_show, dtype=int))
        for src_idx_all in range(nsrc_plot):
            trace_idx_show_all = src_idx_all * nrec_plot + rec_idx_show
            src_pos_all = sem_cfg_obs["source"]["positions"][src_idx_all]
            save_path_all = os.path.join(TRACE_COMPARISON_DIR, f"source_{src_idx_all + 1:02d}_waveforms.png")
            _save_trace_comparison_figure(
                save_path_all,
                trace_idx_show_all,
                rec_idx_show=rec_idx_show,
                src_idx=src_idx_all,
            )
            print(
                f"[TRACE_PLOT] Saved {save_path_all} for source {src_idx_all + 1}/{nsrc_plot} "
                f"at (x={float(src_pos_all[0]):.1f}, z={float(src_pos_all[1]):.1f}); "
                f"{len(rec_idx_show)} receivers spanning x={rec_positions_plot[rec_idx_show[0],0]:.1f}"
                f"–{rec_positions_plot[rec_idx_show[-1],0]:.1f} m.",
                flush=True,
            )
        print(
            f"[TRACE_PLOT] All source waveform figures are in: {TRACE_COMPARISON_DIR}/. "
            "real_noise_comparison.png is not generated.",
            flush=True,
        )
    else:
        n_show = min(int(TRACE_COMPARISON_MAX_TRACES), n_total_traces)
        trace_idx_show = np.unique(np.linspace(0, n_total_traces - 1, n_show, dtype=int))
        save_path = os.path.join(TRACE_COMPARISON_DIR, "global_evenly_spaced_waveforms.png")
        _save_trace_comparison_figure(save_path, trace_idx_show)
        print(
            f"[TRACE_PLOT][WARN] Unexpected trace layout: n_total={n_total_traces}, "
            f"nsrc={nsrc_plot}, nrec={nrec_plot}. Saved fallback figure to {TRACE_COMPARISON_DIR}/. "
            "real_noise_comparison.png is not generated.",
            flush=True,
        )


# ---------------- Output helpers ----------------
def build_plot_background():
    """Load/crop the external initial model background for process snapshots."""
    global _PLOT_BACKGROUND_CACHE
    if _PLOT_BACKGROUND_CACHE is not None:
        return _PLOT_BACKGROUND_CACHE
    try:
        vp_bg = _load_npy_cached(VP_PRE_PATH, dtype=np.float64)
        grid, physical, vmin, vmax = _plot_meta_from_json(VP_PRE_META_PATH, vp_bg)
        vp_plot, x0, x1, z0, z1 = _crop_vp_to_physical_domain(vp_bg, grid, physical)
        vp_extent = (float(x0), float(x1), float(z0), float(z1))
        _PLOT_BACKGROUND_CACHE = (vp_plot, vp_extent)
        return _PLOT_BACKGROUND_CACHE
    except Exception as e:
        print(f"[WARN] Could not load initial-model plotting background: {e}")
        return None, None


def plot_structure_snapshot(ctrl_base, posterior_samples, save_path, *, title,
                            vp_bg=None, bg_extent=None, zoom=None,
                            show_posterior=True, show_true_boundary=True):
    """Snapshot: velocity background + base + posterior mean + true boundary."""
    ctrl_base = np.asarray(ctrl_base, dtype=float).reshape(-1, 2)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    if vp_bg is not None and bg_extent is not None:
        xmin, xmax, zmin, zmax = bg_extent
        im = ax.imshow(
            vp_bg, extent=(xmin, xmax, zmax, zmin), origin="upper", aspect="auto",
            cmap=_get_model_plot_cmap(), vmin=MODEL_PLOT_VMIN, vmax=MODEL_PLOT_VMAX,
            interpolation="nearest",
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.0%", pad=0.14)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Velocity (m/s)", fontsize=19)
        cbar.set_ticks(np.linspace(MODEL_PLOT_VMIN, MODEL_PLOT_VMAX, 8))
        cbar.ax.tick_params(labelsize=17, width=1.0, length=4)

    ctrl_closed = np.vstack([ctrl_base, ctrl_base[0:1]])
    _, _, curve_base = build_closed_bspline(ctrl_base, num_samples=800)
    if show_posterior:
        node_color = "0.25"; node_mfc = "0.25"; boundary_color = "0.20"
        nodes_label = "Base nodes"; boundary_label = "Base boundary"; base_zorder = 5
    else:
        node_color = "black"; node_mfc = "black"; boundary_color = "black"
        nodes_label = "Initial control polygon nodes"; boundary_label = "Initial periodic closed boundary"; base_zorder = 6
    ax.plot(ctrl_closed[:, 0], ctrl_closed[:, 1], linestyle="--", color=node_color, lw=1.6,
            marker="o", ms=6.8, mfc=node_mfc, mec="black", mew=0.9, alpha=0.95,
            label=nodes_label, zorder=base_zorder + 1)
    ax.plot(curve_base[:, 0], curve_base[:, 1], color=boundary_color,
            lw=2.2 if not show_posterior else 1.8, alpha=0.98,
            label=boundary_label, zorder=base_zorder)

    mean_ctrl = ctrl_base.copy()
    if show_posterior and posterior_samples.size > 0:
        nplot = min(PLOT_MAX_POSTERIOR_CURVES, posterior_samples.shape[0])
        for i in range(nplot):
            ctrl_i = ctrl_base + posterior_samples[i].reshape(-1, 2)
            _, _, curve_i = build_closed_bspline(ctrl_i, num_samples=400)
            ax.plot(curve_i[:, 0], curve_i[:, 1], color="blue", ls="--", lw=0.45, alpha=0.025, zorder=3)
        mean_off = posterior_samples.mean(axis=0).reshape(-1, 2)
        mean_ctrl = ctrl_base + mean_off
        mean_closed = np.vstack([mean_ctrl, mean_ctrl[0:1]])
        _, _, curve_mean = build_closed_bspline(mean_ctrl, num_samples=800)
        ax.plot(mean_closed[:, 0], mean_closed[:, 1], linestyle="--", color="blue", lw=1.6,
                marker="o", ms=7.0, mfc="blue", mec="black", mew=0.8, alpha=0.95,
                label="Posterior mean nodes", zorder=8)
        ax.plot(curve_mean[:, 0], curve_mean[:, 1], color="blue", lw=2.2, alpha=0.98,
                label="Posterior mean boundary", zorder=7)

    if show_true_boundary:
        _, _, curve_true = build_closed_bspline(ctrl_pts_true, num_samples=800)
        ax.plot(curve_true[:, 0], curve_true[:, 1], color="red", lw=2.8, alpha=0.98,
                label="True anomaly boundary", zorder=10)

    if zoom is not None:
        zxmin, zxmax, zzmin, zzmax = zoom
        ax.set_xlim(zxmin, zxmax); ax.set_ylim(zzmax, zzmin)
    elif bg_extent is not None:
        xmin, xmax, zmin, zmax = bg_extent
        ax.set_xlim(xmin, xmax); ax.set_ylim(zmax, zmin)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.1, length=6)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)
    return mean_ctrl


def plot_boundary_only_comparison(ctrl_compare, save_path, *, title, compare_label="Updated", zoom=None, auto_zoom=True):
    """Boundary-only comparison: true vs initial/updated."""
    ctrl_compare = np.asarray(ctrl_compare, dtype=float).reshape(-1, 2)
    _, _, curve_compare = build_closed_bspline(ctrl_compare, num_samples=1000)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    all_xy = [ctrl_compare, curve_compare]

    true_ctrl = np.asarray(ctrl_pts_true, dtype=float).reshape(-1, 2)
    true_closed = np.vstack([true_ctrl, true_ctrl[0:1]])
    _, _, curve_true = build_closed_bspline(true_ctrl, num_samples=1000)
    ax.plot(curve_true[:, 0], curve_true[:, 1], color="red", lw=2.8, alpha=0.98,
            label="True boundary", zorder=6)
    ax.plot(true_closed[:, 0], true_closed[:, 1], linestyle="--", color="red", lw=1.8,
            marker="o", ms=7.5, mfc="red", mec="black", mew=0.9, alpha=0.95,
            label="True nodes", zorder=7)
    all_xy.extend([true_ctrl, curve_true])

    compare_closed = np.vstack([ctrl_compare, ctrl_compare[0:1]])
    ax.plot(curve_compare[:, 0], curve_compare[:, 1], color="blue", lw=2.6, alpha=0.98,
            label=f"{compare_label} boundary", zorder=8)
    ax.plot(compare_closed[:, 0], compare_closed[:, 1], linestyle="--", color="blue", lw=1.8,
            marker="o", ms=7.5, mfc="blue", mec="black", mew=0.9, alpha=0.95,
            label=f"{compare_label} nodes", zorder=9)

    if zoom is not None:
        zxmin, zxmax, zzmin, zzmax = zoom
        ax.set_xlim(zxmin, zxmax); ax.set_ylim(zzmax, zzmin)
    elif auto_zoom and all_xy:
        xy = np.vstack(all_xy)
        xmin, xmax = float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))
        zmin, zmax = float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1]))
        dx = max(xmax - xmin, 1.0); dz = max(zmax - zmin, 1.0)
        pad = 0.22 * max(dx, dz)
        ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(zmax + pad, zmin - pad)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22, lw=0.8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.1, length=6)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout(); fig.savefig(save_path, dpi=260); plt.close(fig)


def plot_final_boundary_summary(initial_ctrl, posterior_base_ctrl, posterior_samples, save_path, *, title,
                                vp_bg=None, bg_extent=None, zoom=None):
    """Final summary with base, posterior mean/samples, and true boundary."""
    initial_ctrl = np.asarray(initial_ctrl, dtype=float).reshape(-1, 2)
    posterior_base_ctrl = np.asarray(posterior_base_ctrl, dtype=float).reshape(-1, 2)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    K_post = int(posterior_base_ctrl.shape[0])

    fig, ax = plt.subplots(figsize=(10.8, 6.3))
    if vp_bg is not None and bg_extent is not None:
        xmin, xmax, zmin, zmax = bg_extent
        im = ax.imshow(vp_bg, extent=(xmin, xmax, zmax, zmin), origin="upper", aspect="auto",
                       cmap=_get_model_plot_cmap(), vmin=MODEL_PLOT_VMIN, vmax=MODEL_PLOT_VMAX,
                       interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.0%", pad=0.14)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Velocity (m/s)", fontsize=19)
        cbar.set_ticks(np.linspace(MODEL_PLOT_VMIN, MODEL_PLOT_VMAX, 8))
        cbar.ax.tick_params(labelsize=17, width=1.0, length=4)

    base_closed = np.vstack([initial_ctrl, initial_ctrl[0:1]])
    _, _, base_curve = build_closed_bspline(initial_ctrl, num_samples=800)
    ax.plot(base_closed[:, 0], base_closed[:, 1], linestyle="--", color="0.25", lw=1.6,
            marker="o", ms=6.8, mfc="0.25", mec="black", mew=0.9, alpha=0.95,
            label="Base nodes", zorder=6)
    ax.plot(base_curve[:, 0], base_curve[:, 1], color="0.20", lw=1.8, alpha=0.98,
            label="Base boundary", zorder=5)

    if posterior_samples.ndim == 2 and posterior_samples.shape[1] == 2 * K_post and posterior_samples.size > 0:
        nplot = min(PLOT_MAX_POSTERIOR_CURVES, posterior_samples.shape[0])
        for i in range(nplot):
            ctrl_i = posterior_base_ctrl + posterior_samples[i].reshape(K_post, 2)
            _, _, curve_i = build_closed_bspline(ctrl_i, num_samples=400)
            ax.plot(curve_i[:, 0], curve_i[:, 1], color="blue", ls="--", lw=0.45, alpha=0.025, zorder=3)
        posterior_mean_ctrl = posterior_base_ctrl + posterior_samples.mean(axis=0).reshape(K_post, 2)
    else:
        posterior_mean_ctrl = posterior_base_ctrl.copy()

    post_closed = np.vstack([posterior_mean_ctrl, posterior_mean_ctrl[0:1]])
    _, _, post_curve = build_closed_bspline(posterior_mean_ctrl, num_samples=800)
    ax.plot(post_closed[:, 0], post_closed[:, 1], linestyle="--", color="blue", lw=1.6,
            marker="o", ms=7.0, mfc="blue", mec="black", mew=0.8, alpha=0.95,
            label="Posterior mean nodes", zorder=8)
    ax.plot(post_curve[:, 0], post_curve[:, 1], color="blue", lw=2.2, alpha=0.98,
            label="Posterior mean boundary", zorder=7)

    _, _, curve_true = build_closed_bspline(ctrl_pts_true, num_samples=800)
    ax.plot(curve_true[:, 0], curve_true[:, 1], color="red", lw=2.8, alpha=0.98,
            label="True anomaly boundary", zorder=10)

    if zoom is not None:
        zxmin, zxmax, zzmin, zzmax = zoom
        ax.set_xlim(zxmin, zxmax); ax.set_ylim(zzmax, zzmin)
    elif bg_extent is not None:
        xmin, xmax, zmin, zmax = bg_extent
        ax.set_xlim(xmin, xmax); ax.set_ylim(zmax, zmin)
    else:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    ax.tick_params(axis="both", which="major", labelsize=18, width=1.1, length=6)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout(); fig.savefig(save_path, dpi=220); plt.close(fig)
    return posterior_mean_ctrl


def save_global_history(records, save_dir):
    """Save global_iteration_history.csv."""
    if not records:
        return
    import csv
    os.makedirs(save_dir, exist_ok=True)
    keys = sorted({k for r in records for k in r.keys()})
    with open(os.path.join(save_dir, "global_iteration_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def save_elbo_component_plot(histories, save_path, title="ELBO and components"):
    """Save ELBO-component history plot."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))
    for key, label in [
        ("elbo", "ELBO"),
        ("log_likelihood", "E[log p(y|z)]"),
        ("log_p_z", "E[log p(z|sigma)]"),
        ("log_p_fuse", "E[log p_fuse]"),
        ("neg_log_q", "-E[log q(z)]"),
    ]:
        vals = histories.get(key, [])
        if len(vals) > 0:
            plt.plot(vals, "o-", lw=2, ms=4, label=label)
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=180); plt.close()

def _apply_vp_meta(cfg: dict, vp_path: str, meta_path: str, label: str):
    grid = _read_grid_meta(meta_path)
    bg = cfg['velocity'].setdefault('background_2d', {})
    bg['vp_path'] = str(vp_path)
    bg['nx'] = int(grid['nx']); bg['nz'] = int(grid['nz'])
    for key in ('xmin', 'xmax', 'zmin', 'zmax'):
        val = float(grid[key])
        cfg['domain'][key] = val
        bg[key] = val
    bg['interp'] = VP_INTERP
    print(f"[OK] Loaded {label} meta: {os.path.basename(meta_path)}; domain X=[{cfg['domain']['xmin']},{cfg['domain']['xmax']}], Z=[{cfg['domain']['zmin']},{cfg['domain']['zmax']}], grid=({bg['nx']},{bg['nz']})")

def build_base_sem_config(ctrl_points):
    zero_perturb = np.zeros((np.asarray(ctrl_points).reshape(-1, 2).shape[0], 2), dtype=float).tolist()
    return {
        'domain': {'xmin': DOMAIN_XMIN, 'xmax': DOMAIN_XMAX, 'zmin': DOMAIN_ZMIN, 'zmax': DOMAIN_ZMAX,
                   'nelem_x': NELEM_X, 'nelem_z': NELEM_Z},
        'time': {'total_time': TOTAL_TIME, 'dt': DT},
        'source': {'positions': copy.deepcopy(SOURCE_POSITIONS), 'frequency': SOURCE_FREQUENCY, 'amplitude': SOURCE_AMPLITUDE},
        'receivers': {'positions': copy.deepcopy(RECEIVER_POSITIONS)},
        'method': {'polynomial_order': POLYNOMIAL_ORDER, 'pml_thickness': PML_THICKNESS,
                   'adj_history_dtype': ADJ_HISTORY_DTYPE,
                   'misfit_window': MISFIT_WINDOW},
        'velocity': {
            'background_2d': {'vp_path': VP_BACKGROUND_PATH, 'xmin': DOMAIN_XMIN, 'xmax': DOMAIN_XMAX,
                              'nx': int(_grid_true['nx']), 'zmin': DOMAIN_ZMIN, 'zmax': DOMAIN_ZMAX,
                              'nz': int(_grid_true['nz']), 'interp': VP_INTERP},
            'vmin': VEL_VMIN, 'vmax': VEL_VMAX, 'tau': VEL_TAU, 'spline_samples': VEL_SPLINE_SAMPLES,
            'control_points': np.asarray(ctrl_points, dtype=float).tolist(),
            'perturbations': zero_perturb,
            'anomaly': {
                'enabled': ANOMALY_ENABLED,
                'control_points': np.asarray(ctrl_points, dtype=float).tolist(),
                'perturbations': zero_perturb,
                'v_inside': ANOMALY_V_INSIDE,
                'tau': ANOMALY_TAU,
                'spline_samples': ANOMALY_SPLINE_SAMPLES,
                'blend': ANOMALY_BLEND,
                'boundary_to_grid_method': 'periodic_bspline_signed_distance_sigmoid',
            },
        },
        'output': {'save_wavefield': False, 'save_seismograms': True, 'visualize': False,
                   'output_dir': OUTPUT_DIR, 'snapshot_interval': SNAPSHOT_INTERVAL},
    }

def _make_obs_and_inv_configs(init_ctrl_points, true_ctrl_points=None):
    """
    Build observation and inversion SEM configs using the same forward model.

    IMPORTANT:
    - Observations are now generated as:
          vp_background_pml + true B-spline anomaly
      with the same signed-distance/sigmoid mapping used in inversion.
    - We no longer generate observations by directly loading vp_true_pml.npy as a
      finished velocity grid while disabling the anomaly.  That old route mixed
      two different model parameterizations and could make the displayed model
      differ from the model used by the SEM forward calculation.
    """
    if true_ctrl_points is None:
        true_ctrl_points = init_ctrl_points

    obs_cfg = build_base_sem_config(true_ctrl_points)
    inv_cfg = build_base_sem_config(init_ctrl_points)

    _apply_vp_meta(obs_cfg, VP_BACKGROUND_PATH, VP_BACKGROUND_META_PATH, "obs vp_background_pml")
    _apply_vp_meta(inv_cfg, VP_BACKGROUND_PATH, VP_BACKGROUND_META_PATH, "inv vp_background_pml")

    # Observation: true absolute control points + zero perturbation.
    zero_true = np.zeros((np.asarray(true_ctrl_points).reshape(-1, 2).shape[0], 2), dtype=float).tolist()
    obs_cfg['velocity']['control_points'] = np.asarray(true_ctrl_points, dtype=float).tolist()
    obs_cfg['velocity']['perturbations'] = zero_true
    obs_cfg['velocity']['anomaly']['enabled'] = True
    obs_cfg['velocity']['anomaly']['control_points'] = np.asarray(true_ctrl_points, dtype=float).tolist()
    obs_cfg['velocity']['anomaly']['perturbations'] = zero_true
    obs_cfg['velocity']['background_2d']['vp_path'] = VP_BACKGROUND_PATH

    # Inversion: initial absolute control points + NF offsets.
    inv_cfg['velocity']['background_2d']['vp_path'] = VP_BACKGROUND_PATH
    inv_cfg['velocity']['anomaly']['enabled'] = True
    return obs_cfg, inv_cfg


def run_simulation(cfg_in, ctrl_params=None, noise_std=0.0, visualize=False):
    """Forward-only SEM simulation for observations/checks; no backpropagation."""
    cfg0 = copy.deepcopy(cfg_in)
    if ctrl_params is not None:
        if hasattr(ctrl_params, 'detach'):
            ctrl_params_np = ctrl_params.detach().cpu().numpy().reshape(-1, 2)
        else:
            ctrl_params_np = np.asarray(ctrl_params, dtype=float).reshape(-1, 2)
        cfg0['velocity'].setdefault('anomaly', {})['perturbations'] = ctrl_params_np.tolist()
        cfg0['velocity']['perturbations'] = ctrl_params_np.tolist()

    waveforms_list = []
    dt = None
    nt = None
    source_positions = cfg0['source']['positions']
    for _, pos in enumerate(source_positions):
        sx, sz = map(float, pos)
        cfg_k = copy.deepcopy(cfg0)
        cfg_k['source'] = cfg_k['source'].copy()
        cfg_k['source']['position'] = [sx, sz]
        # Forward-only checks/observation generation should not reuse a cached
        # SEMSimulation with stale control-point offsets.
        sim = SEMSimulation(cfg_k)
        results = sim.run()
        wf = results['receiver_data']
        if dt is None:
            dt = float(results['dt']); nt = int(results['nt'])
        waveforms_list.append(wf)
    data_all = waveforms_list[0] if len(waveforms_list) == 1 else np.concatenate(waveforms_list, axis=1)
    clean_data = torch.from_numpy(data_all).double()
    if noise_std is None or float(noise_std) == 0.0:
        noisy_data = clean_data.clone()
    else:
        noisy_data = clean_data + torch.normal(mean=torch.zeros_like(clean_data), std=float(noise_std))
    return noisy_data, clean_data, dt, nt

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
                tail_bound=float(FLOW_TAIL_BOUND)
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
                 current_epoch=None, total_epochs=None,
                 min_samples=8, max_samples=16):
    """Compute the single-stage ELBO with a fixed Gaussian offset prior.

    ELBO = E_q[log p(y|z) + log p(z; FIXED_SIGMA) + log p_fuse(z) - log q(z)].
    """
    # 1) number of Monte Carlo samples (annealed between min and max)
    if (current_epoch is not None) and (total_epochs is not None) and (total_epochs > 1):
        t = current_epoch / (total_epochs - 1)
        n_samples = int(round((1 - t) * min_samples + t * max_samples))
    else:
        n_samples = int(min_samples)
    # 2) q(z)
    z_samples, _ = model.forward(n_samples=n_samples)                # (S, DIM_Z)
    S = z_samples.shape[0]
    DZ = z_samples.shape[1]
    z_phys = z_samples
    log_q_z = model.log_prob(z_samples).unsqueeze(1)                 # (S, 1)
    # 3) fused (TV / fusion) prior on adjacent control-point offsets (cyclic)
    if float(FUSE_LAMBDA_EFF) > 0.0:
        z_pairs = z_samples.view(n_samples, K2, 2)                   # (S, K2, 2)
        idx_next = (torch.arange(K2, device=z_pairs.device) + 1) % K2
        diffs = z_pairs[:, idx_next, :] - z_pairs
        mags = torch.sqrt(torch.sum(diffs * diffs, dim=2) + (float(FUSE_EPS) ** 2))  # (S,K2)
        fuse_pen = torch.sum(mags, dim=1, keepdim=True)              # (S,1)
        log_p_fuse = -float(FUSE_LAMBDA_EFF) * fuse_pen
    else:
        # log_p_z not defined yet here; use log_q_z shape as reference
        log_p_fuse = torch.zeros_like(log_q_z)
    # 4) fixed Gaussian prior on physical control-point offsets.
    sigma_k = torch.full(
        (K2,),
        float(FIXED_SIGMA),
        device=z_samples.device,
        dtype=z_samples.dtype,
    )
    # Prior p(z): fixed prior in physical space (meters); shrinkage disabled.
    z_for_prior = z_phys
    sigma_vec = sigma_k.repeat_interleave(2)
    prior_dist = torch.distributions.Normal(loc=torch.zeros_like(sigma_vec), scale=sigma_vec)
    log_p_z = torch.sum(prior_dist.log_prob(z_for_prior), dim=1)
    # 5) log p(y|z) using SEM adjoint (multi-source handled in SEMLikelihoodAdjointFn)
    log_p_y_given_z = []
    for i in range(n_samples):
        z_i = z_samples[i]
        ll_i = sem_loglik_with_adjoint(z_i, y_obs, noise_std, obs_dt, obs_nt)
        log_p_y_given_z.append(ll_i)
    log_p_y_given_z = torch.stack(log_p_y_given_z, dim=0).unsqueeze(1)  # (S,1)
    # 6) ELBO
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
def _gradient_series(gradient_history, key, default=0.0):
    return [float(gh.get(key, default)) for gh in gradient_history]


def plot_gradient_history(gradient_history, filename='gradient_history_real.png'):
    """Plot gradient history before and after clipping."""
    if not gradient_history:
        return
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    epochs = np.arange(1, len(gradient_history) + 1)
    pre_total = _gradient_series(gradient_history, 'total_norm_pre_clip')
    post_total = _gradient_series(gradient_history, 'total_norm_post_clip')
    pre_max = _gradient_series(gradient_history, 'max_grad_pre_clip')
    post_max = _gradient_series(gradient_history, 'max_grad_post_clip')
    pre_mean = _gradient_series(gradient_history, 'mean_grad_pre_clip')
    post_mean = _gradient_series(gradient_history, 'mean_grad_post_clip')
    clip_scale = _gradient_series(gradient_history, 'clip_scale', 1.0)
    clipped_flag = _gradient_series(gradient_history, 'gradient_clipped', 0.0)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, pre_total, 'o-', lw=1.8, ms=4, label='before clipping')
    plt.plot(epochs, post_total, 'o-', lw=1.8, ms=4, label='after clipping')
    plt.title('Total Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    plt.subplot(2, 3, 2)
    plt.plot(epochs, pre_max, 'o-', lw=1.8, ms=4, label='before clipping')
    plt.plot(epochs, post_max, 'o-', lw=1.8, ms=4, label='after clipping')
    plt.title('Max Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    plt.subplot(2, 3, 3)
    plt.plot(epochs, pre_mean, 'o-', lw=1.8, ms=4, label='before clipping')
    plt.plot(epochs, post_mean, 'o-', lw=1.8, ms=4, label='after clipping')
    plt.title('Mean Gradient Value')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    plt.subplot(2, 3, 4)
    plt.plot(epochs, clipped_flag, 'o-', lw=1.8, ms=4)
    plt.title('Gradient Clipping Events')
    plt.xlabel('Epoch')
    plt.ylabel('Clipped (1=Yes)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(epochs, clip_scale, 'o-', lw=1.8, ms=4)
    plt.title('Applied Clipping Scale')
    plt.xlabel('Epoch')
    plt.ylabel('Scale')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 6)
    clipped_ratio = _gradient_series(gradient_history, 'clipped_ratio')
    plt.plot(epochs, clipped_ratio, 'o-', lw=1.8, ms=4)
    plt.title('Gradient Clipping Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Clipped Ratio')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def collect_grad_summary(params):
    """Return total norm, max absolute gradient, and mean absolute gradient."""
    if not params:
        return 0.0, 0.0, 0.0
    total = float(torch.norm(torch.stack([torch.norm(p.grad.detach()).cpu() for p in params])).cpu())
    max_abs = max(float(p.grad.detach().abs().max().cpu()) for p in params)
    mean_abs = float(torch.stack([p.grad.detach().abs().mean().cpu() for p in params]).mean())
    return total, max_abs, mean_abs


# =============================================================================
# 4. Initialization, observations, training, and output
# =============================================================================
set_global_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else (
        "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
    )
)
print(f"[DEVICE] {device}")
print("Using SEMSimulation from:", SEMSimulation.__module__, SEMSimulation.__qualname__)

# Validate required model files.
for _p, _label in [
    (VP_SMOOTH_PATH, "true grid"), (VP_META_PATH, "true meta"),
    (VP_PRE_PATH, "initial grid"), (VP_PRE_META_PATH, "initial meta"),
    (VP_BACKGROUND_PATH, "clean background grid"), (VP_BACKGROUND_META_PATH, "clean background meta"),
]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"Required model1 {_label} not found: {_p}")

# Derived constants that depend on metadata or the fixed configuration above.
os.makedirs(OBS_CACHE_DIR, exist_ok=True)

ctrl_pts_init_base = _load_control_points_from_json(VP_PRE_META_PATH, label="EXTERNAL initial")
TRUE_CTRL_PTS = _load_control_points_from_json(VP_META_PATH, label="EXTERNAL true")
ctrl_pts_true = TRUE_CTRL_PTS.copy()
K2 = int(ctrl_pts_init_base.shape[0])
DIM_Z = 2 * K2
print(f"[MODEL] Fixed NF dimension from current base model: K2={K2}, DIM_Z={DIM_Z}")

_grid_true = _read_grid_meta(VP_META_PATH)
DOMAIN_XMIN = float(DOMAIN_XMIN_OVERRIDE) if DOMAIN_XMIN_OVERRIDE is not None else float(_grid_true["xmin"])
DOMAIN_XMAX = float(DOMAIN_XMAX_OVERRIDE) if DOMAIN_XMAX_OVERRIDE is not None else float(_grid_true["xmax"])
DOMAIN_ZMIN = float(DOMAIN_ZMIN_OVERRIDE) if DOMAIN_ZMIN_OVERRIDE is not None else float(_grid_true["zmin"])
DOMAIN_ZMAX = float(DOMAIN_ZMAX_OVERRIDE) if DOMAIN_ZMAX_OVERRIDE is not None else float(_grid_true["zmax"])
RECEIVER_POSITIONS = [[float(x), RECEIVER_Z] for x in range(RECEIVER_XMIN, RECEIVER_XMAX + 1, RECEIVER_DX)]

TRUE_ANOMALY_V_INSIDE = _read_anomaly_value(VP_META_PATH, 2000.0)
ANOMALY_V_INSIDE = float(ANOMALY_V_INSIDE_OVERRIDE) if ANOMALY_V_INSIDE_OVERRIDE is not None else float(TRUE_ANOMALY_V_INSIDE)
TRUE_ANOMALY_TAU = _read_anomaly_tau_from_meta(VP_META_PATH, 10.0)
ANOMALY_TAU = float(ANOMALY_TAU_OVERRIDE) if ANOMALY_TAU_OVERRIDE is not None else float(TRUE_ANOMALY_TAU)

print(
    f"[ACQ] sources={len(SOURCE_POSITIONS)}, receivers={len(RECEIVER_POSITIONS)} "
    f"(dx={RECEIVER_DX} m, z={RECEIVER_Z} m), f0={SOURCE_FREQUENCY} Hz, "
    f"total_time={TOTAL_TIME}s, dt={DT:g}s"
)
print(f"[TRAIN] PRINT_EVERY={PRINT_EVERY}; training diagnostics are printed every epoch when PRINT_EVERY=1.")

_noise_var_raw = float(obs_noise_std) ** 2
_noise_var_eff = _noise_var_raw + float(EPS)
_eps_ratio = float(EPS) / max(_noise_var_raw, 1e-300)
print(
    f"[NUMERICS] OBS_NOISE_STD={obs_noise_std:.3e}; "
    f"noise_std**2={_noise_var_raw:.3e}; EPS={EPS:.1e}; "
    f"EPS/noise_var_raw={_eps_ratio:.3e}; effective_noise_std={math.sqrt(_noise_var_eff):.3e}"
)
if _eps_ratio > 1.0e-3:
    print("[NUMERICS][WARN] EPS is not negligible relative to OBS_NOISE_STD**2; it will inflate the likelihood variance.")

# B-spline setup.
time0 = time.perf_counter()
k = 2
knots_true, t_curve_true, curve_points_true = build_closed_bspline(ctrl_pts_true, num_samples=800)
knots = knots_true
t_curve = t_curve_true
curve_points = curve_points_true
# Process figures are saved under POSTERIOR_OUTDIR/fixedK##_diagnostic/.
time1 = time.perf_counter()
print(f" Step 1: model1 B-spline setup time: {time1 - time0:.2f}s")

# SEM configuration and model overview figures.
_check_model_generation_consistency()
sem_cfg_obs, sem_cfg_inv = _make_obs_and_inv_configs(ctrl_pts_init_base, ctrl_pts_true)
sem_config = sem_cfg_inv

try:
    _vp_true = _load_npy_cached(VP_SMOOTH_PATH)
    plot_velocity_overview_from_grid(
        vp_grid=_vp_true,
        meta_path=VP_META_PATH,
        title='True Model',
        save_path='velocity_model_true.png',
        ctrl_pts=TRUE_CTRL_PTS,
        boundary_label='True periodic closed boundary',
        nodes_label='True control polygon nodes',
    )
    print('[OK] Saved true-model check figure: velocity_model_true.png')
except Exception as _e:
    print(f"[WARN] Failed to save velocity_model_true.png: {_e}")

if DEBUG_PLOT_INIT:
    try:
        _vp_initial = _load_npy_cached(VP_PRE_PATH)
        plot_velocity_overview_from_grid(
            vp_grid=_vp_initial,
            meta_path=VP_PRE_META_PATH,
            title='Initial',
            save_path='velocity_model_initial.png',
            ctrl_pts=ctrl_pts_init_base,
            boundary_label='Initial periodic closed boundary',
            nodes_label='Initial control polygon nodes',
        )
        print('[OK] Saved initial-model check figure: velocity_model_initial.png')
    except Exception as _e_plot_init:
        print(f"[WARN] Failed to create initial-model plot: {_e_plot_init}")
else:
    print('[SKIP] DEBUG_PLOT_INIT=0; skipping initial-model check plot.')

# Observations cache.  The tag includes the actual acquisition/model controls so
# stale observations are not reused after changing geometry, source, noise, PML,
# or the true control points.
def _stable_json_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

_obs_tag = (
    f"semtrue_{Path(VP_BACKGROUND_PATH).stem}"
    + f"_{len(SOURCE_POSITIONS)}s_{len(RECEIVER_POSITIONS)}r"
    + f"_T{TOTAL_TIME:g}_dt{DT:g}"
    + f"_f{SOURCE_FREQUENCY:g}_a{SOURCE_AMPLITUDE:g}"
    + f"_noise{obs_noise_std:g}"
    + "_" + _stable_json_hash({
        "sources": SOURCE_POSITIONS,
        "receivers": RECEIVER_POSITIONS,
        "pml": PML_THICKNESS,
        "npol": POLYNOMIAL_ORDER,
        "nelem": [NELEM_X, NELEM_Z],
        "tau": ANOMALY_TAU,
        "v_inside": ANOMALY_V_INSIDE,
        "blend": ANOMALY_BLEND,
        "true_ctrl": np.asarray(ctrl_pts_true, dtype=float).round(8).tolist(),
    })
)
_noisy_path = os.path.join(OBS_CACHE_DIR, f"noisy_data_{_obs_tag}.npy")
_clean_path = os.path.join(OBS_CACHE_DIR, f"clean_data_{_obs_tag}.npy")
_meta_path = os.path.join(OBS_CACHE_DIR, f"obs_meta_{_obs_tag}.json")
if FORCE_REGEN_OBS_CACHE:
    for _p in (_noisy_path, _clean_path, _meta_path):
        if os.path.exists(_p):
            os.remove(_p)
    print(f"[OBS_CACHE] FORCE_REGEN_OBS_CACHE=1 -> regenerated observations for tag={_obs_tag}")

time1 = time.perf_counter()
if os.path.exists(_noisy_path) and os.path.exists(_clean_path) and os.path.exists(_meta_path):
    print(f"Loading cached observations from: {OBS_CACHE_DIR}")
    noisy_obs = torch.from_numpy(np.load(_noisy_path)).double()
    clean_obs = torch.from_numpy(np.load(_clean_path)).double()
    with open(_meta_path, "r") as _f:
        _meta = json.load(_f)
    obs_dt = float(_meta["obs_dt"]); obs_nt = int(_meta["obs_nt"])
else:
    print("Generating observed data with SAME SEM mapping as inversion: vp_background_pml + true B-spline anomaly...")
    noisy_obs, clean_obs, actual_dt, actual_nt = run_simulation(sem_cfg_obs, None, noise_std=obs_noise_std)
    obs_dt = actual_dt; obs_nt = actual_nt
    np.save(_noisy_path, noisy_obs.numpy()); np.save(_clean_path, clean_obs.numpy())
    with open(_meta_path, "w") as _f:
        json.dump({"obs_dt": float(obs_dt), "obs_nt": int(obs_nt)}, _f, indent=2)
    print(f"SEM observations cached. dt={obs_dt:.6f}, nt={obs_nt}")
actual_dt = obs_dt
actual_nt = obs_nt
y_obs = noisy_obs
noise_std = obs_noise_std
time2 = time.perf_counter()
print(f"Step 2: SEM sim setup time: {time2 - time1:.2f}s")

_save_all_source_waveform_figures(clean_obs, noisy_obs, actual_dt, actual_nt, sem_cfg_obs)
time3 = time.perf_counter()
print(f"Step 3: generated observations. Time: {time3 - time2:.2f}s")

# Training setup.
base_mean = torch.zeros(DIM_Z, dtype=torch.float64)
base_cov = torch.eye(DIM_Z, dtype=torch.float64)
model = NormalizingFlow(dim=DIM_Z, num_flows=num_flows, base_mean=base_mean, base_cov=base_cov)
print(f"[SINGLE-STAGE] Fixed prior sigma = {FIXED_SIGMA:g} m.")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
outdir = POSTERIOR_OUTDIR
diagnostic_dir = os.path.join(outdir, f"fixedK{K2:02d}_diagnostic")
os.makedirs(outdir, exist_ok=True)
os.makedirs(diagnostic_dir, exist_ok=True)
np.save(os.path.join(outdir, "fixedK_true_ctrl_pts.npy"), np.asarray(ctrl_pts_true, dtype=np.float64))
np.save(os.path.join(outdir, "fixedK_initial_ctrl_pts.npy"), np.asarray(ctrl_pts_init_base, dtype=np.float64))
with open(os.path.join(outdir, "fixedK_diagnostic_config.json"), "w") as _f_cfg:
    json.dump({
        "mode": "fixed_K_single_stage_RQS_Coupling30_forward_NF",
        "K": int(K2),
        "epochs": int(num_epochs),
        "PRINT_EVERY": 1,
        "PLOT_EVERY": int(PLOT_EVERY),
        "SAVE_ITERATIONS": "0, PLOT_EVERY, 2*PLOT_EVERY, ..., and final NUM_EPOCHS even if not divisible",
        "OBS_NOISE_STD": float(obs_noise_std),
        "TOTAL_TIME": float(TOTAL_TIME),
        "MISFIT_WINDOW_ENABLED": bool(MISFIT_WINDOW.get("enabled", False)),
        "EPS": float(EPS),
        "MODEL_MAPPING": "vp_background_pml + periodic B-spline signed-distance sigmoid anomaly",
        "OBSERVATION_MODEL": "same SEM anomaly mapping as inversion; not direct vp_true_pml grid",
    }, _f_cfg, indent=2)
print(f"[OUTPUT] Process outputs will be saved in: {diagnostic_dir}")
elbo_history = []
log_likelihood_history = []
log_prior_z_history = []
log_q_history = []
log_p_fuse_history = []
gradient_history = []

# Single-stage fixed-K NF training.
print("Starting single-stage fixed-K NF training...")
vp_bg, vp_extent = build_plot_background()
with torch.no_grad():
    init_mean_ctrl = plot_structure_snapshot(
        ctrl_pts_init_base, np.empty((0, 2 * K2)),
        os.path.join(diagnostic_dir, "snapshot_iteration_0000.png"),
        title=f"fixedK{K2}_diagnostic: iteration 0",
        vp_bg=vp_bg, bg_extent=vp_extent, zoom=ZOOM_WINDOW,
        show_posterior=False, show_true_boundary=False,
    )
    np.save(os.path.join(diagnostic_dir, "base_ctrl_pts_initial.npy"), np.asarray(ctrl_pts_init_base, dtype=np.float64))
    np.save(os.path.join(diagnostic_dir, "mean_ctrl_pts_iteration_0000.npy"), np.asarray(init_mean_ctrl, dtype=np.float64))
    plot_boundary_only_comparison(
        ctrl_pts_init_base, os.path.join(diagnostic_dir, "boundary_only_iteration_0000.png"),
        title=f"fixedK{K2}_diagnostic: boundary-only iteration 0",
        compare_label="Initial", auto_zoom=True,
    )

last_print_time = time.perf_counter()
for epoch in range(1, num_epochs + 1):
    sem_config['method']['VERIFY_PROJECTION'] = False
    optimizer.zero_grad()
    elbo, log_likelihood, log_p_z, log_p_fuse, log_q_z = compute_elbo(
        model, noisy_obs, obs_noise_std,
        current_epoch=epoch, total_epochs=num_epochs,
        min_samples=min_elbo_samples, max_samples=max_elbo_samples
    )
    loss = -elbo
    loss.backward()

    # Gradient clipping + diagnostics before/after clipping.
    params = [p for p in model.parameters() if getattr(p, "grad", None) is not None]
    _total_norm_pre, _grad_abs_max_pre, _grad_mean_pre = collect_grad_summary(params)
    _grad_was_clipped = bool(clip_gradient and params and (_total_norm_pre > max_grad_norm))
    _clip_scale = 1.0
    if _grad_was_clipped:
        _clip_scale = float(max_grad_norm / (_total_norm_pre + 1e-12))
        for p in params:
            p.grad.detach().mul_(_clip_scale)

    _total_norm_post, _grad_abs_max_post, _grad_mean_post = collect_grad_summary(params)
    if params:
        _param_norm = float(torch.norm(torch.stack([torch.norm(p.detach()).cpu() for p in params])))
    else:
        _param_norm = 0.0

    optimizer.step()
    gradient_history.append({
        "total_norm": float(_total_norm_post),
        "max_grad": float(_grad_abs_max_post),
        "mean_grad": float(_grad_mean_post),
        "total_norm_pre_clip": float(_total_norm_pre),
        "max_grad_pre_clip": float(_grad_abs_max_pre),
        "mean_grad_pre_clip": float(_grad_mean_pre),
        "total_norm_post_clip": float(_total_norm_post),
        "max_grad_post_clip": float(_grad_abs_max_post),
        "mean_grad_post_clip": float(_grad_mean_post),
        "clip_scale": float(_clip_scale),
        "gradient_clipped": float(1.0 if _grad_was_clipped else 0.0),
        "param_norm": float(_param_norm),
        "learning_rate": float(learning_rate),
        "clipped_ratio": float(1.0 if _grad_was_clipped else 0.0),
        "gradient_exploded": bool(_grad_was_clipped),
    })

    elbo_history.append(float(elbo.detach().cpu()))
    log_likelihood_history.append(float(log_likelihood.detach().cpu()))
    log_prior_z_history.append(float(log_p_z.detach().cpu()))
    log_p_fuse_history.append(float(log_p_fuse.detach().cpu()))
    log_q_history.append(float(log_q_z.detach().cpu()))

    now = time.perf_counter()
    dt_print = now - last_print_time
    last_print_time = now
    print(f"[TRAIN] epoch {epoch:4d}/{num_epochs} | ELBO={elbo.item():.3f} | loglike={log_likelihood.item():.3f} | "
          f"logp(z|σ={FIXED_SIGMA:g}m)={log_p_z.item():.3f} | logq={log_q_z.item():.3f} | "
          f"grad_norm_pre={_total_norm_pre:.2e} | grad_norm_post={_total_norm_post:.2e} | Δt={dt_print:6.2f}s", flush=True)

    iter_done = epoch
    if (iter_done % PLOT_EVERY) == 0 or (iter_done == num_epochs):
        with torch.no_grad():
            snap_samples, _ = model.forward(n_samples=min(PLOT_SNAPSHOT_SAMPLES, FINAL_POST_SAMPLES))
            mean_ctrl_epoch = plot_structure_snapshot(
                ctrl_pts_init_base, snap_samples.cpu().numpy(),
                os.path.join(diagnostic_dir, f"snapshot_iteration_{iter_done:04d}.png"),
                title=f"fixedK{K2}_diagnostic: iteration {iter_done}",
                vp_bg=vp_bg, bg_extent=vp_extent, zoom=ZOOM_WINDOW,
            )
            np.save(os.path.join(diagnostic_dir, f"mean_ctrl_pts_iteration_{iter_done:04d}.npy"), mean_ctrl_epoch)
            plot_boundary_only_comparison(
                mean_ctrl_epoch, os.path.join(diagnostic_dir, f"boundary_only_iteration_{iter_done:04d}.png"),
                title=f"fixedK{K2}_diagnostic: boundary-only iteration {iter_done}",
                compare_label="Updated", auto_zoom=True,
            )

# Final posterior samples and diagnostics.
posterior_draw = FINAL_POST_SAMPLES
with torch.no_grad():
    post_samples, _ = model.forward(n_samples=posterior_draw)
    post_samples_np = post_samples.cpu().numpy()
print("[DONE] Training finished; saving final fixed-K diagnostics.")

mean_ctrl_final = plot_structure_snapshot(
    ctrl_pts_init_base, post_samples_np, os.path.join(diagnostic_dir, "snapshot_final.png"),
    title=f"fixedK{K2}_diagnostic: final", vp_bg=vp_bg, bg_extent=vp_extent, zoom=ZOOM_WINDOW,
)
plot_boundary_only_comparison(
    mean_ctrl_final, os.path.join(diagnostic_dir, "boundary_only_final.png"),
    title=f"fixedK{K2}_diagnostic: boundary-only final", compare_label="Updated", auto_zoom=True,
)
np.save(os.path.join(diagnostic_dir, "posterior_samples.npy"), post_samples_np)
np.save(os.path.join(diagnostic_dir, "mean_ctrl_pts_final.npy"), mean_ctrl_final)
np.save(os.path.join(outdir, "final_mean_ctrl_pts.npy"), mean_ctrl_final)
np.save(os.path.join(outdir, "final_posterior_samples.npy"), post_samples_np)
plot_final_boundary_summary(
    ctrl_pts_init_base, ctrl_pts_init_base, post_samples_np,
    os.path.join(outdir, "final_boundary_summary.png"),
    title="Fixed-K diagnostic inversion result", vp_bg=vp_bg, bg_extent=vp_extent, zoom=ZOOM_WINDOW,
)
plot_boundary_only_comparison(
    mean_ctrl_final, os.path.join(outdir, "final_boundary_only_true_vs_updated.png"),
    title="Fixed-K diagnostic: True vs Updated", compare_label="Updated", auto_zoom=True,
)

global_records = []
for i in range(len(elbo_history)):
    global_records.append({
        "global_epoch": int(i),
        "phase": "fixedK_diagnostic",
        "K": int(K2),
        "elbo": float(elbo_history[i]),
        "log_likelihood": float(log_likelihood_history[i]),
        "log_p_z": float(log_prior_z_history[i]),
        "log_p_fuse": float(log_p_fuse_history[i]),
        "log_q": float(log_q_history[i]),
        "neg_log_q": float(-log_q_history[i]),
    })
save_global_history(global_records, outdir)
with open(os.path.join(outdir, "run_summary.json"), "w") as _f_log:
    json.dump({
        "mode": "fixed_K_single_stage_RQS_Coupling30_forward_NF",
        "K": int(K2),
        "epochs_run": int(len(elbo_history)),
        "fixed_prior_sigma_m": float(FIXED_SIGMA),
        "noise_std": float(obs_noise_std),
    }, _f_log, indent=2)

try:
    _elbo_npz_path = os.path.join(diagnostic_dir, "elbo_components_raw.npz")
    np.savez_compressed(
        _elbo_npz_path,
        stage=1,
        num_epochs=int(num_epochs),
        epoch=np.arange(len(elbo_history), dtype=np.int32),
        elbo=np.asarray(elbo_history, dtype=np.float64),
        E_log_p_y_given_z=np.asarray(log_likelihood_history, dtype=np.float64),
        E_log_p_z_given_sigma=np.asarray(log_prior_z_history, dtype=np.float64),
        neg_E_log_q_z=np.asarray([-v for v in log_q_history], dtype=np.float64),
        E_log_p_fuse=np.asarray(log_p_fuse_history, dtype=np.float64),
    )
    print(f"[OUTPUT] Saved raw ELBO-component data: {_elbo_npz_path}")
except Exception as _e:
    print(f"[WARN] Failed to save raw ELBO-component data: {_e}")

_histories_for_plot = {
    "elbo": elbo_history,
    "log_likelihood": log_likelihood_history,
    "log_p_z": log_prior_z_history,
    "log_p_fuse": log_p_fuse_history,
    "log_q": log_q_history,
    "neg_log_q": [-v for v in log_q_history],
}
save_elbo_component_plot(_histories_for_plot, os.path.join(diagnostic_dir, "elbo_history_real.png"), title="fixedK diagnostic: ELBO and components")
plot_gradient_history(gradient_history, filename=os.path.join(diagnostic_dir, "gradient_history_real.png"))

component_specs = [("elbo", "ELBO"), ("log_likelihood", "log likelihood"), ("log_p_z", "log prior z"), ("log_q", "log q")]
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
x_iter = np.arange(1, len(global_records) + 1)
for ax, (key, label) in zip(axes, component_specs):
    vals = np.asarray([r[key] for r in global_records], dtype=float)
    ax.plot(x_iter, vals, "o-", lw=2, ms=4, label=label)
    ax.set_ylabel(label); ax.grid(True, alpha=0.3); ax.legend(loc="best")
axes[-1].set_xlabel("Iteration")
fig.suptitle("Fixed-K diagnostic: ELBO components", fontsize=20)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(outdir, "fixedK_elbo_components_global.png"), dpi=180)
plt.close(fig)

print(f"[OUTPUT] Saved outputs in folder: {outdir}")
print(f"[DONE] Completed fixed-K single-stage NF run. Outputs in: {outdir}")
sys.exit(0)
