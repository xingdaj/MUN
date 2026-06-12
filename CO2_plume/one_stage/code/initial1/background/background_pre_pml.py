#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the clean layered BACKGROUND PML model used by SEM inversion."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
PLOT_VMIN = float(os.environ.get("PLOT_VMIN", "1800.0"))
PLOT_VMAX = float(os.environ.get("PLOT_VMAX", "3200.0"))
PLOT_CMAP = os.environ.get("PLOT_CMAP", "turbo")


def resolve_workdir(workdir=None):
    return os.path.abspath(workdir) if workdir is not None else os.path.dirname(os.path.abspath(__file__))


def grid_arrays():
    x = np.arange(XMIN, XMAX + 0.5 * DX, DX, dtype=float)
    z = np.arange(ZMIN, ZMAX + 0.5 * DZ, DZ, dtype=float)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z


def build_layered_vp(Z):
    vp_bg = np.empty_like(Z, dtype=float)
    vp_bg[Z < 0.0] = V_AIR
    vp_bg[(Z >= 0.0) & (Z < INTERFACES_Z[1])] = VEL_LAYERS[0]
    vp_bg[(Z >= INTERFACES_Z[1]) & (Z < INTERFACES_Z[2])] = VEL_LAYERS[1]
    vp_bg[(Z >= INTERFACES_Z[2]) & (Z < INTERFACES_Z[3])] = VEL_LAYERS[2]
    vp_bg[Z >= INTERFACES_Z[3]] = VEL_LAYERS[3]
    return vp_bg


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
    return {"applied": False, "method": "none; clean layered background", "plot_vmin_m_per_s": float(PLOT_VMIN),
            "plot_vmax_m_per_s": float(PLOT_VMAX)}


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_vp_models():
    x, z, X, Z = grid_arrays()
    vp_bg = build_layered_vp(Z)
    ix_phys, iz_phys = physical_masks(x, z)
    return {"x_full": x, "z_full": z, "x_phys": x[ix_phys], "z_phys": z[iz_phys],
            "vp_full": vp_bg, "vp_phys": vp_bg[np.ix_(iz_phys, ix_phys)]}


def plot_model(path, x, z, vp, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(x, z, vp, cmap=PLOT_CMAP, shading="nearest", vmin=PLOT_VMIN, vmax=PLOT_VMAX)
    ax.set_xlim(PHYS_XMIN, PHYS_XMAX); ax.set_ylim(PHYS_ZMIN, PHYS_ZMAX); ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box"); ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_title(title)
    cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.08)
    plt.colorbar(im, cax=cax).set_label("Vp (m/s)")
    fig.tight_layout(); fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)


def save_outputs(result, workdir):
    os.makedirs(workdir, exist_ok=True)
    np.save(os.path.join(workdir, "vp_background.npy"), result["vp_phys"])
    np.save(os.path.join(workdir, "vp_background_pml.npy"), result["vp_full"])
    base = {"array_layout": "model[nz, nx], z increases downward, x increases to the right",
            "background_layers": common_background_json(), "smoothing": common_smoothing_json(),
            "anomaly": {"enabled": False, "boundary_to_grid_method": "none"}}
    obj_phys = dict(base, model_name="vp_background", description="Clean layered background on physical domain.",
                    array_file="vp_background.npy", grid=common_grid_json(result["x_phys"], result["z_phys"]))
    obj_full = dict(base, model_name="vp_background_pml", description="Clean layered background on full PML domain.",
                    array_file="vp_background_pml.npy", grid=common_grid_json(result["x_full"], result["z_full"]),
                    physical_subdomain={"xmin": PHYS_XMIN, "xmax": PHYS_XMAX, "zmin": PHYS_ZMIN, "zmax": PHYS_ZMAX})
    save_json(os.path.join(workdir, "vp_background.json"), obj_phys)
    save_json(os.path.join(workdir, "vp_background_pml.json"), obj_full)


def main(workdir=None):
    workdir = resolve_workdir(workdir)
    result = build_vp_models()
    save_outputs(result, workdir)
    plot_model(os.path.join(workdir, "vp_background_before_smoothing.png"), result["x_phys"], result["z_phys"], result["vp_phys"], "Vp Background Model")
    plot_model(os.path.join(workdir, "vp_background_after_smoothing.png"), result["x_phys"], result["z_phys"], result["vp_phys"], "Vp Background Model")
    print("[OK] Saved background-model outputs to", workdir)


if __name__ == "__main__":
    main(None)
