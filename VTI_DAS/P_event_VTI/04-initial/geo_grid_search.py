#!/usr/bin/env python3
"""Estimate initial source locations by coarse grid search.

This replaces the old geo_add.py workflow that perturbed the true source
locations directly.  The initial source positions are obtained by fitting the
observed travel-time data with the perturbed initial VTI model.  The grid can be
coarse because the following MCMC stage refines the source locations.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import importlib.util
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TRUE_GEOMETRY = Path("../01-input/output/geometry.dat")
DEFAULT_INIT_VEL = Path("../04-initial/output/vel.dat")
DEFAULT_OBS = Path("../04-initial/output/nobs.dat")
DEFAULT_OUTPUT = Path("../04-initial/output/geo.dat")
DEFAULT_FIG = Path("../04-initial/output/figures/geo_grid_search.png")
DEFAULT_CSV = Path("../04-initial/output/geo_grid_search.csv")
DEFAULT_MCMC_MODULE = Path("../05-inversion/vti_joint_mcmc_dram.py")


def load_mcmc_module(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find MCMC/forward module: {path}")
    spec = importlib.util.spec_from_file_location("vti_mcmc_for_grid_search", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def numeric_tokens(path: Path) -> list[float]:
    vals: list[float] = []
    for tok in path.read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_geometry(path: Path):
    vals = numeric_tokens(path)
    if not vals:
        raise ValueError(f"Cannot read geometry from {path}")
    idx = 0
    ns = int(vals[idx]); idx += 1
    sx = np.zeros(ns, dtype=float)
    sz = np.zeros(ns, dtype=float)
    for i in range(ns):
        sx[i] = vals[idx]
        sz[i] = vals[idx + 1]
        idx += 2
    nr = int(vals[idx]); idx += 1
    rx = np.zeros(nr, dtype=float)
    rz = np.zeros(nr, dtype=float)
    for j in range(nr):
        rx[j] = vals[idx]
        rz[j] = vals[idx + 1]
        idx += 2
    return sx, sz, rx, rz


def write_geometry(path: Path, sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("----src(x,z)----\n")
        f.write(f"{len(sx):d}\n")
        for x, z in zip(sx, sz):
            f.write(f"{float(x):.12f}\t{float(z):.12f}\n")
        f.write("----rev(x,z)----\n")
        f.write(f"{len(rx):d}\n")
        for x, z in zip(rx, rz):
            f.write(f"{float(x):.12f}\t{float(z):.12f}\n")


def grid_values(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    n = int(math.floor((vmax - vmin) / step + 0.5))
    vals = vmin + step * np.arange(n + 1, dtype=float)
    vals = vals[vals <= vmax + 1e-9]
    if vals.size == 0:
        raise ValueError(f"empty grid for range [{vmin}, {vmax}] and step {step}")
    return vals


def source_misfit_from_tp(tp_row: np.ndarray, isrc: int, obs, like, objective_type: str) -> float:
    if objective_type == "absolute":
        r = tp_row - obs.tp[isrc, :]
        return float(np.sum((r / like.sigma_abs) ** 2))
    if objective_type in {"diff-p-adjacent", "diff-p-reference"}:
        pred_d = like.diff_matrix @ tp_row
        r = pred_d - like.obs_d[isrc, :]
        return float(r @ like.cov_inv @ r)
    raise ValueError(f"Unsupported objective_type={objective_type!r}; use absolute, diff-p-adjacent, or diff-p-reference")


def plot_result(path: Path, sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray,
                true_sx: np.ndarray | None, true_sz: np.ndarray | None, x_grid: np.ndarray, z_grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(rx, rz, "ks", markerfacecolor="r", markersize=3.5, label="DAS")
    if true_sx is not None and true_sz is not None and len(true_sx) == len(sx):
        ax.plot(true_sx, true_sz, "k+", markersize=7, label="True source")
    ax.plot(sx, sz, "bo", markerfacecolor="none", markersize=6, label="Grid-search initial")
    ax.set_xlim(min(np.min(x_grid), np.min(rx)) - 50.0, max(np.max(x_grid), np.max(rx)) + 50.0)
    ax.set_ylim(max(np.max(z_grid), np.max(rz)) + 80.0, 0.0)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Initial source locations from coarse grid search")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Coarse grid-search initialization for source locations")
    p.add_argument("--receiver-geometry", type=Path, default=DEFAULT_TRUE_GEOMETRY,
                   help="Geometry file used only to read receiver coordinates and event count")
    p.add_argument("--init-vel", type=Path, default=DEFAULT_INIT_VEL)
    p.add_argument("--obs-file", type=Path, default=DEFAULT_OBS)
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--csv-file", type=Path, default=DEFAULT_CSV)
    p.add_argument("--fig-file", type=Path, default=DEFAULT_FIG)
    p.add_argument("--mcmc-module", type=Path, default=DEFAULT_MCMC_MODULE)
    p.add_argument("--x-min", type=float, default=350.0)
    p.add_argument("--x-max", type=float, default=1050.0)
    p.add_argument("--z-min", type=float, default=350.0)
    p.add_argument("--z-max", type=float, default=850.0)
    p.add_argument("--dx", type=float, default=25.0)
    p.add_argument("--dz", type=float, default=25.0)
    p.add_argument("--refine", action="store_true", help="Run a second local grid with half step around the coarse best point")
    p.add_argument("--objective", default="diff-p-adjacent", choices=["absolute", "diff-p-adjacent", "diff-p-reference"])
    p.add_argument("--sigma-mode", default="absolute", choices=["absolute", "objective-iid"])
    p.add_argument("--qx-stop", type=float, default=1e-6)
    p.add_argument("--qx-max-iter", type=int, default=20)
    args = p.parse_args(argv)

    mod = load_mcmc_module(args.mcmc_module)
    true_sx, true_sz, rx, rz = read_geometry(args.receiver_geometry)
    model = mod.read_model(args.init_vel)
    obs = mod.read_observed(args.obs_file)
    like = mod.build_likelihood_config(obs, args.objective, "P", sigma_mode=args.sigma_mode)

    obs_ns, obs_nr = obs.tp.shape
    if obs_ns != len(true_sx):
        raise ValueError(f"Observation ns={obs_ns} differs from geometry ns={len(true_sx)}")
    if obs_nr != len(rx):
        raise ValueError(f"Observation nr={obs_nr} differs from geometry nr={len(rx)}")

    x_grid = grid_values(args.x_min, args.x_max, args.dx)
    z_grid = grid_values(args.z_min, args.z_max, args.dz)

    est_sx = np.zeros(obs_ns, dtype=float)
    est_sz = np.zeros(obs_ns, dtype=float)
    best_mis = np.full(obs_ns, np.inf, dtype=float)

    pmax_cache = mod._wave_pmax_cache(model, {"P"})

    print(f"[GRID] model={args.init_vel}")
    print(f"[GRID] obs={args.obs_file}")
    print(f"[GRID] x=[{args.x_min:g},{args.x_max:g}] dx={args.dx:g}; z=[{args.z_min:g},{args.z_max:g}] dz={args.dz:g}")
    print(f"[GRID] candidates per source = {len(x_grid) * len(z_grid)}; ns={obs_ns}; nr={obs_nr}")

    for isrc in range(obs_ns):
        best = (np.inf, np.nan, np.nan)
        for x in x_grid:
            for z in z_grid:
                geo_one = mod.Geometry(
                    sx=np.asarray([x], dtype=float),
                    sz=np.asarray([z], dtype=float),
                    rx=np.asarray(rx, dtype=float),
                    rz=np.asarray(rz, dtype=float),
                )
                try:
                    tp_row, _, _ = mod.forward_direct_source(
                        geo_one, model, 0,
                        stop=args.qx_stop, max_iter=args.qx_max_iter,
                        needed_waves=("P",), pmax_cache=pmax_cache,
                    )
                    mis = source_misfit_from_tp(tp_row, isrc, obs, like, args.objective)
                except Exception:
                    mis = np.inf
                if mis < best[0]:
                    best = (float(mis), float(x), float(z))

        if args.refine and np.isfinite(best[0]):
            hx = args.dx / 2.0
            hz = args.dz / 2.0
            x_local = grid_values(max(args.x_min, best[1] - args.dx), min(args.x_max, best[1] + args.dx), hx)
            z_local = grid_values(max(args.z_min, best[2] - args.dz), min(args.z_max, best[2] + args.dz), hz)
            for x in x_local:
                for z in z_local:
                    geo_one = mod.Geometry(np.asarray([x]), np.asarray([z]), np.asarray(rx), np.asarray(rz))
                    try:
                        tp_row, _, _ = mod.forward_direct_source(
                            geo_one, model, 0,
                            stop=args.qx_stop, max_iter=args.qx_max_iter,
                            needed_waves=("P",), pmax_cache=pmax_cache,
                        )
                        mis = source_misfit_from_tp(tp_row, isrc, obs, like, args.objective)
                    except Exception:
                        mis = np.inf
                    if mis < best[0]:
                        best = (float(mis), float(x), float(z))

        best_mis[isrc], est_sx[isrc], est_sz[isrc] = best
        print(f"[GRID] source {isrc+1:02d}/{obs_ns}: x={est_sx[isrc]:.2f}, z={est_sz[isrc]:.2f}, mis={best_mis[isrc]:.6g}")

    write_geometry(args.output_file, est_sx, est_sz, rx, rz)

    args.csv_file.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "sx_grid", "sz_grid", "misfit", "true_sx_for_check", "true_sz_for_check"])
        for i in range(obs_ns):
            w.writerow([i, est_sx[i], est_sz[i], best_mis[i], true_sx[i], true_sz[i]])

    plot_result(args.fig_file, est_sx, est_sz, rx, rz, true_sx, true_sz, x_grid, z_grid)
    print(f"[GRID] wrote {args.output_file}")
    print(f"[GRID] wrote {args.csv_file}")
    print(f"[GRID] wrote {args.fig_file}")


if __name__ == "__main__":
    main()
