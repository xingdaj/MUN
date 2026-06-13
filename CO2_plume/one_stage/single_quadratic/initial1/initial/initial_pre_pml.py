#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the INITIAL PML model with the SAME boundary-to-velocity mapping used by
RQS_NF.py / sem_waveform.velocity:

    vp = vp_background + (vp_inside - vp_background) * sigmoid(-signed_distance/tau)

The closed quadratic B-spline convention is also identical:
    ctrl_ext = [P0, ..., P{K-1}, P0, P1]
    knots    = 0, 1, 2, ..., len(ctrl_ext)+degree
    valid parameter interval = [degree, K+degree)
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from sem_waveform.velocity import build_velocity_2d_background_with_anomaly, closed_bspline_curve_points

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 22
mpl.rcParams["axes.titlesize"] = 22
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 16

XMIN = float(os.environ.get("DOMAIN_XMIN", "-300.0"))
XMAX = float(os.environ.get("DOMAIN_XMAX", "2300.0"))
ZMIN = float(os.environ.get("DOMAIN_ZMIN", "-300.0"))
ZMAX = float(os.environ.get("DOMAIN_ZMAX", "1300.0"))
DX = float(os.environ.get("MODEL_DX", "10.0"))
DZ = float(os.environ.get("MODEL_DZ", "10.0"))
PHYS_XMIN = float(os.environ.get("PHYS_XMIN", "0.0"))
PHYS_XMAX = float(os.environ.get("PHYS_XMAX", "2000.0"))
PHYS_ZMIN = float(os.environ.get("PHYS_ZMIN", "0.0"))
PHYS_ZMAX = float(os.environ.get("PHYS_ZMAX", "1000.0"))

INTERFACES_Z = np.array([0.0, 200.0, 500.0, 700.0], dtype=float)
VEL_LAYERS = np.array([1800.0, 2400.0, 2900.0, 3200.0], dtype=float)
V_AIR = float(os.environ.get("V_AIR", "340.0"))

K_SPLINE = int(os.environ.get("K_SPLINE", "2"))
N_CTRL = int(os.environ.get("N_CTRL", "6"))
ANOMALY_CENTER = np.array([float(os.environ.get("ANOMALY_CENTER_X", "850.0")),
                           float(os.environ.get("ANOMALY_CENTER_Z", "350.0"))], dtype=float)
ANOMALY_RX = float(os.environ.get("ANOMALY_RX", "250.0"))
ANOMALY_RZ = float(os.environ.get("ANOMALY_RZ", "70.0"))
ANOMALY_ROTATION_DEG = float(os.environ.get("ANOMALY_ROTATION_DEG", "0.0"))
ANOMALY_VP = float(os.environ.get("ANOMALY_VP", "2000.0"))
ANOMALY_TAU = float(os.environ.get("ANOMALY_TAU", "10.0"))
BOUNDARY_CURVE_SAMPLES = int(os.environ.get("ANOMALY_SPLINE_SAMPLES", "800"))
RANDOM_SEED = int(os.environ.get("INITIAL_RANDOM_SEED", "20260530"))
NODE_PERTURB_MEAN = float(os.environ.get("NODE_PERTURB_MEAN", "0.0"))
NODE_PERTURB_STD = float(os.environ.get("NODE_PERTURB_STD", "50.0"))

PLOT_VMIN = float(os.environ.get("PLOT_VMIN", "1800.0"))
PLOT_VMAX = float(os.environ.get("PLOT_VMAX", "3200.0"))
PLOT_CMAP = os.environ.get("PLOT_CMAP", "turbo")


def resolve_workdir(workdir=None):
    return os.path.abspath(workdir) if workdir is not None else os.path.dirname(os.path.abspath(__file__))


def make_ellipse_control_points(center, num_points, rx, rz, rotation_deg=0.0):
    center = np.asarray(center, dtype=float)
    ang = np.linspace(0.0, 2.0 * np.pi, int(num_points), endpoint=False)
    pts = np.stack([rx * np.cos(ang), rz * np.sin(ang)], axis=1)
    th = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=float)
    return pts @ rot.T + center[None, :]


def build_layered_vp(Z):
    vp_bg = np.empty_like(Z, dtype=float)
    vp_bg[Z < 0.0] = V_AIR
    vp_bg[(Z >= 0.0) & (Z < INTERFACES_Z[1])] = VEL_LAYERS[0]
    vp_bg[(Z >= INTERFACES_Z[1]) & (Z < INTERFACES_Z[2])] = VEL_LAYERS[1]
    vp_bg[(Z >= INTERFACES_Z[2]) & (Z < INTERFACES_Z[3])] = VEL_LAYERS[2]
    vp_bg[Z >= INTERFACES_Z[3]] = VEL_LAYERS[3]
    return vp_bg


def grid_arrays():
    x = np.arange(XMIN, XMAX + 0.5 * DX, DX, dtype=float)
    z = np.arange(ZMIN, ZMAX + 0.5 * DZ, DZ, dtype=float)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z


def physical_masks(x, z):
    ix_phys = (x >= PHYS_XMIN - 1.0e-9) & (x <= PHYS_XMAX + 1.0e-9)
    iz_phys = (z >= PHYS_ZMIN - 1.0e-9) & (z <= PHYS_ZMAX + 1.0e-9)
    return ix_phys, iz_phys


def common_grid_json(x, z):
    return {"xmin": float(x[0]), "xmax": float(x[-1]), "zmin": float(z[0]), "zmax": float(z[-1]),
            "dx": float(DX), "dz": float(DZ), "nx": int(x.size), "nz": int(z.size)}


def common_background_json():
    return {"interfaces_z_m": INTERFACES_Z.tolist(), "velocities_m_per_s": VEL_LAYERS.tolist(),
            "vp_for_z_lt_0_m_per_s": float(V_AIR)}


def common_smoothing_json():
    return {"applied": True, "method": "signed-distance sigmoid; same as sem_waveform.velocity",
            "tau_m": float(ANOMALY_TAU), "sigma_x_grid_cells": float(ANOMALY_TAU / DX),
            "sigma_z_grid_cells": float(ANOMALY_TAU / DZ), "sigma_x_m": float(ANOMALY_TAU),
            "sigma_z_m": float(ANOMALY_TAU), "tau_mean_m": float(ANOMALY_TAU),
            "plot_vmin_m_per_s": float(PLOT_VMIN), "plot_vmax_m_per_s": float(PLOT_VMAX),
            "note": "This is not post-hoc Gaussian smoothing; it is the exact sigmoid interface used by inversion."}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_vp_models():
    x, z, X, Z = grid_arrays()
    vp_background = build_layered_vp(Z)
    true_ctrl_pts = make_ellipse_control_points(ANOMALY_CENTER, N_CTRL, ANOMALY_RX, ANOMALY_RZ, ANOMALY_ROTATION_DEG)
    rng = np.random.default_rng(RANDOM_SEED)
    perturb = rng.normal(loc=NODE_PERTURB_MEAN, scale=NODE_PERTURB_STD, size=true_ctrl_pts.shape)
    ctrl_pts = true_ctrl_pts + perturb
    curve_points = closed_bspline_curve_points(ctrl_pts, samples=BOUNDARY_CURVE_SAMPLES, degree=K_SPLINE, close=True)
    nodes_xy = np.column_stack([X.ravel(), Z.ravel()])

    vp_raw_flat, sd_flat, _ = build_velocity_2d_background_with_anomaly(
        nodes_xy, vp_background.ravel(), ctrl_pts, ANOMALY_VP, ANOMALY_TAU,
        samples=BOUNDARY_CURVE_SAMPLES, newton_steps=7, blend="replace")
    vp_smooth_flat, sd_flat, _ = build_velocity_2d_background_with_anomaly(
        nodes_xy, vp_background.ravel(), ctrl_pts, ANOMALY_VP, ANOMALY_TAU,
        samples=BOUNDARY_CURVE_SAMPLES, newton_steps=7, blend="smooth")

    vp_raw = vp_raw_flat.reshape(X.shape)
    vp_smooth = vp_smooth_flat.reshape(X.shape)
    signed_dist = sd_flat.reshape(X.shape)
    air_mask = Z < 0.0
    vp_raw[air_mask] = V_AIR
    vp_smooth[air_mask] = V_AIR
    smoothing_band = (np.abs(signed_dist) <= 3.0 * ANOMALY_TAU) & (~air_mask)
    anomaly_mask = signed_dist < 0.0
    ix_phys, iz_phys = physical_masks(x, z)
    return {"x_full": x, "z_full": z, "x_phys": x[ix_phys], "z_phys": z[iz_phys],
            "vp_raw_full": vp_raw, "vp_smooth_full": vp_smooth,
            "vp_raw_phys": vp_raw[np.ix_(iz_phys, ix_phys)],
            "vp_smooth_phys": vp_smooth[np.ix_(iz_phys, ix_phys)],
            "control_points": ctrl_pts, "true_control_points": true_ctrl_pts, "node_perturbations": perturb, "bspline_curve": curve_points,
            "anomaly_mask_full": anomaly_mask, "smoothing_band_full": smoothing_band,
            "signed_distance_full": signed_dist}


def anomaly_json(result):
    return {"enabled": True, "type": "closed periodic quadratic B-spline signed-distance sigmoid",
            "degree": int(K_SPLINE), "num_control_points": int(N_CTRL),
            "center_xy_m": ANOMALY_CENTER.tolist(), "rx_m": float(ANOMALY_RX), "rz_m": float(ANOMALY_RZ),
            "rotation_deg": float(ANOMALY_ROTATION_DEG), "vp_inside_m_per_s": float(ANOMALY_VP),
            "tau_m": float(ANOMALY_TAU), "blend": "smooth",
            "boundary_to_grid_method": "periodic_bspline_signed_distance_sigmoid",
            "control_points": result["control_points"].tolist(),
            "initial_control_points": result["control_points"].tolist(),
            "true_reference_control_points": result.get("true_control_points", result["control_points"]).tolist(),
            "node_perturbations_m": result.get("node_perturbations", np.zeros_like(result["control_points"])).tolist(),
            "initial_random_seed": int(RANDOM_SEED),
            "node_perturb_mean_m": float(NODE_PERTURB_MEAN),
            "node_perturb_std_m": float(NODE_PERTURB_STD),
            "boundary_curve_samples": int(BOUNDARY_CURVE_SAMPLES)}


def plot_common(ax, x, z, vp, title, curve_points=None, ctrl_points=None):
    im = ax.pcolormesh(x, z, vp, cmap=PLOT_CMAP, shading="nearest", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
    handles = []
    if curve_points is not None and ctrl_points is not None:
        ax.plot(curve_points[:, 0], curve_points[:, 1], "k-", lw=2.6)
        ctrl_closed = np.vstack([ctrl_points, ctrl_points[0]])
        ax.plot(ctrl_closed[:, 0], ctrl_closed[:, 1], "r--", lw=2.0, alpha=0.95)
        ax.scatter(ctrl_points[:, 0], ctrl_points[:, 1], s=75, facecolor="red", edgecolor="black", linewidth=1.2, zorder=8)
        handles = [Line2D([0], [0], color="black", lw=2.6, label="B-spline boundary"),
                   Line2D([0], [0], color="red", lw=2.0, ls="--", label="Control polygon"),
                   Line2D([0], [0], marker="o", ls="None", markersize=8, markerfacecolor="red", markeredgecolor="black", label="Nodes")]
    ax.set_xlim(PHYS_XMIN, PHYS_XMAX); ax.set_ylim(PHYS_ZMIN, PHYS_ZMAX); ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    if handles: ax.legend(handles=handles, loc="upper right", frameon=True)
    return im


def save_model_plot(path, x, z, vp, title, curve_points=None, ctrl_points=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = plot_common(ax, x, z, vp, title, curve_points, ctrl_points)
    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.08)
    plt.colorbar(im, cax=cax).set_label("Vp (m/s)")
    fig.tight_layout(); fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)


def save_outputs(result, workdir):
    os.makedirs(workdir, exist_ok=True)
    np.save(os.path.join(workdir, "vp_initial.npy"), result["vp_smooth_phys"])
    np.save(os.path.join(workdir, "vp_initial_pml.npy"), result["vp_smooth_full"])
    ix_phys, iz_phys = physical_masks(result["x_full"], result["z_full"])
    base = {"array_layout": "model[nz, nx], z increases downward, x increases to the right",
            "background_layers": common_background_json(), "anomaly": anomaly_json(result),
            "smoothing": common_smoothing_json()}
    obj_phys = dict(base, model_name="vp_initial", description="Initial Vp model generated by the same signed-distance sigmoid mapping as SEM inversion.",
                    array_file="vp_initial.npy", grid=common_grid_json(result["x_phys"], result["z_phys"]))
    obj_full = dict(base, model_name="vp_initial_pml", description="Initial Vp model on full PML grid, generated by the same signed-distance sigmoid mapping as SEM inversion.",
                    array_file="vp_initial_pml.npy", grid=common_grid_json(result["x_full"], result["z_full"]),
                    physical_subdomain={"xmin": PHYS_XMIN, "xmax": PHYS_XMAX, "zmin": PHYS_ZMIN, "zmax": PHYS_ZMAX})
    save_json(os.path.join(workdir, "vp_initial.json"), obj_phys)
    save_json(os.path.join(workdir, "vp_initial_pml.json"), obj_full)


def plot_models(result, workdir):
    save_model_plot(os.path.join(workdir, "vp_model_before_smoothing.png"), result["x_phys"], result["z_phys"], result["vp_raw_phys"], "Vp True Model: Hard Inside/Outside", result["bspline_curve"], result["control_points"])
    save_model_plot(os.path.join(workdir, "vp_model_after_smoothing.png"), result["x_phys"], result["z_phys"], result["vp_smooth_phys"], "Vp True Model: SEM Sigmoid Mapping", result["bspline_curve"], result["control_points"])


def main(workdir=None):
    workdir = resolve_workdir(workdir)
    result = build_vp_models()
    save_outputs(result, workdir)
    plot_models(result, workdir)
    print("[OK] Saved initial-model outputs to", workdir)
    print(f"[PARAM] mapping=periodic_bspline_signed_distance_sigmoid, tau={ANOMALY_TAU:g} m, samples={BOUNDARY_CURVE_SAMPLES}")


if __name__ == "__main__":
    main(None)
