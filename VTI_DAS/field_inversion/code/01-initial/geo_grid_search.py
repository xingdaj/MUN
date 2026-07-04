#!/usr/bin/env python3
"""Estimate initial field-event locations by grid search.

The source positions are obtained by fitting the observed field P travel-time
curves with the initial layered VTI model.  The input geometry.dat contains
only the event count and receiver coordinates; no source placeholders are used.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import concurrent.futures as cf
import csv
import importlib.util
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_TRUE_GEOMETRY = Path("../01-initial/output/geometry.dat")
DEFAULT_INIT_VEL = Path("../01-initial/output/vel.dat")
DEFAULT_OBS = Path("../01-initial/output/nobs.dat")
DEFAULT_OUTPUT = Path("../01-initial/output/geo.dat")
DEFAULT_FIG = Path("../01-initial/output/figures/geo_grid_search.pdf")
DEFAULT_CSV = Path("../01-initial/output/geo_grid_search.csv")
DEFAULT_MCMC_MODULE = Path("../02-inversion/vti_joint_mcmc_dram.py")


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


def _numeric_values(line: str) -> list[float]:
    vals: list[float] = []
    for tok in line.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_geometry(path: Path):
    """Read either receiver-only geometry.dat or full geo.dat.

    Receiver-only format before grid search:
        ns
        nr
        receiver_id rx rz
        ...

    Full format after grid search:
        ns
        sx sz
        ...
        nr
        rx rz
        ...
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    state = None
    ns: int | None = None
    nr: int | None = None
    rx_rows: list[tuple[float, float]] = []
    sx_rows: list[tuple[float, float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "src" in low or "number of events" in low:
            state = "src"
            continue
        if "rev" in low or "receiver" in low and "event" not in low:
            state = "rev"
            continue
        vals = _numeric_values(line)
        if not vals:
            continue
        if state == "src":
            if ns is None:
                ns = int(vals[0])
            elif len(vals) >= 2:
                # Full geo.dat may contain source coordinates.  geometry.dat does not.
                sx_rows.append((float(vals[0]), float(vals[1])))
        elif state == "rev":
            if nr is None:
                nr = int(vals[0])
            elif len(vals) >= 3:
                # receiver_id rx rz
                rx_rows.append((float(vals[-2]), float(vals[-1])))
            elif len(vals) >= 2:
                # rx rz
                rx_rows.append((float(vals[0]), float(vals[1])))
        else:
            # Fallback for very compact numeric files is intentionally not used,
            # because it can confuse ns/nr with receiver IDs.
            continue

    if ns is None or nr is None:
        raise ValueError(f"Cannot read ns/nr from geometry file {path}")
    if len(rx_rows) != nr:
        raise ValueError(f"Geometry file {path} says nr={nr}, but {len(rx_rows)} receiver rows were read")

    sx = np.full(ns, np.nan, dtype=float)
    sz = np.full(ns, np.nan, dtype=float)
    if len(sx_rows) == ns:
        sx[:] = [v[0] for v in sx_rows]
        sz[:] = [v[1] for v in sx_rows]
    rx = np.asarray([v[0] for v in rx_rows], dtype=float)
    rz = np.asarray([v[1] for v in rx_rows], dtype=float)
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
    """One-source grid-search misfit with fast exact differencing.

    For the current field setting (diff-p-adjacent + objective-iid), this avoids
    the dense A @ t and r @ C^{-1} @ r operations for every grid candidate.
    """
    if objective_type == "absolute":
        r = tp_row - obs.tp[isrc, :]
        return float(np.sum((r / like.sigma_abs) ** 2))
    if objective_type == "diff-p-adjacent":
        pred_d = tp_row[:-1] - tp_row[1:]
        r = pred_d - like.obs_d[isrc, :]
        if getattr(like, "sigma_mode", "absolute") == "objective-iid":
            return float(np.sum((r / like.sigma_abs) ** 2))
        return float(r @ like.cov_inv @ r)
    if objective_type == "diff-p-reference":
        pred_d = tp_row[1:] - tp_row[0]
        r = pred_d - like.obs_d[isrc, :]
        if getattr(like, "sigma_mode", "absolute") == "objective-iid":
            return float(np.sum((r / like.sigma_abs) ** 2))
        return float(r @ like.cov_inv @ r)
    raise ValueError(f"Unsupported objective_type={objective_type!r}; use absolute, diff-p-adjacent, or diff-p-reference")


def plot_result(path: Path, sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray,
                x_grid: np.ndarray, z_grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(rx, rz, "-", color="black", linewidth=1.0, alpha=0.65, label="DAS receiver order")
    ax.scatter(rx, rz, s=12, color="black", label="DAS receivers")
    ax.scatter(sx, sz, s=28, marker="o", facecolors="none", edgecolors="red", linewidths=1.2, label="Grid-search initial events")
    ax.set_xlim(min(np.min(x_grid), np.min(rx)) - 50.0, max(np.max(x_grid), np.max(rx)) + 50.0)
    ax.set_ylim(max(np.max(z_grid), np.max(rz), 2000.0) + 50.0, min(np.min(z_grid), np.min(rz), 200.0) - 50.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Initial field-event locations from grid search")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Worker globals for parallel source-level grid search.  Each worker process
# loads the forward/MCMC module and input data once, then evaluates one or more
# source events independently.
_GRID_MOD = None
_GRID_MODEL = None
_GRID_OBS = None
_GRID_LIKE = None
_GRID_RX = None
_GRID_RZ = None
_GRID_OBJECTIVE = None
_GRID_QX_STOP = None
_GRID_QX_MAX_ITER = None
_GRID_PMAX_CACHE = None


def _init_grid_worker(mcmc_module_path: str, init_vel_path: str, obs_file_path: str,
                      objective_type: str, sigma_mode: str,
                      rx: np.ndarray, rz: np.ndarray,
                      qx_stop: float, qx_max_iter: int) -> None:
    global _GRID_MOD, _GRID_MODEL, _GRID_OBS, _GRID_LIKE, _GRID_RX, _GRID_RZ
    global _GRID_OBJECTIVE, _GRID_QX_STOP, _GRID_QX_MAX_ITER, _GRID_PMAX_CACHE
    _GRID_MOD = load_mcmc_module(Path(mcmc_module_path))
    _GRID_MODEL = _GRID_MOD.read_model(Path(init_vel_path))
    _GRID_OBS = _GRID_MOD.read_observed(Path(obs_file_path))
    _GRID_LIKE = _GRID_MOD.build_likelihood_config(_GRID_OBS, objective_type, "P", sigma_mode=sigma_mode)
    _GRID_RX = np.asarray(rx, dtype=float)
    _GRID_RZ = np.asarray(rz, dtype=float)
    _GRID_OBJECTIVE = objective_type
    _GRID_QX_STOP = float(qx_stop)
    _GRID_QX_MAX_ITER = int(qx_max_iter)
    _GRID_PMAX_CACHE = _GRID_MOD._wave_pmax_cache(_GRID_MODEL, {"P"})


def _evaluate_source_grid(isrc: int, x_grid: np.ndarray, z_grid: np.ndarray,
                          refine: bool, x_min: float, x_max: float,
                          z_min: float, z_max: float,
                          dx: float, dz: float) -> tuple[int, float, float, float]:
    if _GRID_MOD is None:
        raise RuntimeError("grid-search worker was not initialized")

    def eval_one(x: float, z: float) -> float:
        geo_one = _GRID_MOD.Geometry(
            sx=np.asarray([x], dtype=float),
            sz=np.asarray([z], dtype=float),
            rx=_GRID_RX,
            rz=_GRID_RZ,
        )
        try:
            tp_row, _, _ = _GRID_MOD.forward_direct_source(
                geo_one, _GRID_MODEL, 0,
                stop=_GRID_QX_STOP, max_iter=_GRID_QX_MAX_ITER,
                needed_waves=("P",), pmax_cache=_GRID_PMAX_CACHE,
            )
            return source_misfit_from_tp(tp_row, isrc, _GRID_OBS, _GRID_LIKE, _GRID_OBJECTIVE)
        except Exception:
            return float("inf")

    best = (float("inf"), float("nan"), float("nan"))
    for x in x_grid:
        for z in z_grid:
            mis = eval_one(float(x), float(z))
            if mis < best[0]:
                best = (float(mis), float(x), float(z))

    if refine and np.isfinite(best[0]):
        hx = dx / 2.0
        hz = dz / 2.0
        x_local = grid_values(max(x_min, best[1] - dx), min(x_max, best[1] + dx), hx)
        z_local = grid_values(max(z_min, best[2] - dz), min(z_max, best[2] + dz), hz)
        for x in x_local:
            for z in z_local:
                mis = eval_one(float(x), float(z))
                if mis < best[0]:
                    best = (float(mis), float(x), float(z))

    return int(isrc), float(best[0]), float(best[1]), float(best[2])


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
    p.add_argument("--x-min", type=float, default=800.0)
    p.add_argument("--x-max", type=float, default=1600.0)
    p.add_argument("--z-min", type=float, default=1600.0)
    p.add_argument("--z-max", type=float, default=2000.0)
    p.add_argument("--dx", type=float, default=10.0)
    p.add_argument("--dz", type=float, default=10.0)
    p.add_argument("--refine", action="store_true", help="Run a second local grid with half step around the coarse best point")
    p.add_argument("--objective", default="diff-p-adjacent", choices=["absolute", "diff-p-adjacent", "diff-p-reference"])
    p.add_argument("--sigma-mode", default="absolute", choices=["absolute", "objective-iid"])
    p.add_argument("--qx-stop", type=float, default=1e-6)
    p.add_argument("--qx-max-iter", type=int, default=20)
    p.add_argument("--forward-workers", type=int, default=1,
                   help="Number of parallel workers for source-level grid search. Use 1 for serial; use 0 for all available CPU cores capped by number of events.")
    args = p.parse_args(argv)

    mod = load_mcmc_module(args.mcmc_module)
    event_count_sx, event_count_sz, rx, rz = read_geometry(args.receiver_geometry)
    model = mod.read_model(args.init_vel)
    obs = mod.read_observed(args.obs_file)
    like = mod.build_likelihood_config(obs, args.objective, "P", sigma_mode=args.sigma_mode)

    obs_ns, obs_nr = obs.tp.shape
    if obs_ns != len(event_count_sx):
        raise ValueError(f"Observation ns={obs_ns} differs from geometry ns={len(event_count_sx)}")
    if obs_nr != len(rx):
        raise ValueError(f"Observation nr={obs_nr} differs from geometry nr={len(rx)}")

    x_grid = grid_values(args.x_min, args.x_max, args.dx)
    z_grid = grid_values(args.z_min, args.z_max, args.dz)

    est_sx = np.zeros(obs_ns, dtype=float)
    est_sz = np.zeros(obs_ns, dtype=float)
    best_mis = np.full(obs_ns, np.inf, dtype=float)

    pmax_cache = mod._wave_pmax_cache(model, {"P"})

    forward_workers = int(args.forward_workers)
    if forward_workers <= 0:
        forward_workers = max(1, min(obs_ns, (os.cpu_count() or 1)))
    forward_workers = max(1, min(forward_workers, obs_ns))

    print(f"[GRID] model={args.init_vel}")
    print(f"[GRID] obs={args.obs_file}")
    print(f"[GRID] x=[{args.x_min:g},{args.x_max:g}] dx={args.dx:g}; z=[{args.z_min:g},{args.z_max:g}] dz={args.dz:g}")
    print(f"[GRID] candidates per source = {len(x_grid) * len(z_grid)}; ns={obs_ns}; nr={obs_nr}")
    print(f"[GRID] source-level parallelism: workers={forward_workers}")

    if forward_workers == 1:
        _init_grid_worker(
            str(args.mcmc_module), str(args.init_vel), str(args.obs_file),
            args.objective, args.sigma_mode, rx, rz, args.qx_stop, args.qx_max_iter,
        )
        for isrc in range(obs_ns):
            isrc_out, mis, sx_i, sz_i = _evaluate_source_grid(
                isrc, x_grid, z_grid, args.refine,
                args.x_min, args.x_max, args.z_min, args.z_max, args.dx, args.dz,
            )
            best_mis[isrc_out], est_sx[isrc_out], est_sz[isrc_out] = mis, sx_i, sz_i
            print(f"[GRID] source {isrc_out+1:02d}/{obs_ns}: x={sx_i:.2f}, z={sz_i:.2f}, mis={mis:.6g}")
    else:
        with cf.ProcessPoolExecutor(
            max_workers=forward_workers,
            initializer=_init_grid_worker,
            initargs=(
                str(args.mcmc_module), str(args.init_vel), str(args.obs_file),
                args.objective, args.sigma_mode, rx, rz, args.qx_stop, args.qx_max_iter,
            ),
        ) as executor:
            futures = [
                executor.submit(
                    _evaluate_source_grid,
                    isrc, x_grid, z_grid, args.refine,
                    args.x_min, args.x_max, args.z_min, args.z_max, args.dx, args.dz,
                )
                for isrc in range(obs_ns)
            ]
            for fut in cf.as_completed(futures):
                isrc_out, mis, sx_i, sz_i = fut.result()
                best_mis[isrc_out], est_sx[isrc_out], est_sz[isrc_out] = mis, sx_i, sz_i
                print(f"[GRID] source {isrc_out+1:02d}/{obs_ns}: x={sx_i:.2f}, z={sz_i:.2f}, mis={mis:.6g}")

    write_geometry(args.output_file, est_sx, est_sz, rx, rz)

    args.csv_file.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_index_0based", "sx_grid", "sz_grid", "misfit"])
        for i in range(obs_ns):
            w.writerow([i, est_sx[i], est_sz[i], best_mis[i]])

    plot_result(args.fig_file, est_sx, est_sz, rx, rz, x_grid, z_grid)
    print(f"[GRID] wrote {args.output_file}")
    print(f"[GRID] wrote {args.csv_file}")
    print(f"[GRID] wrote {args.fig_file}")


if __name__ == "__main__":
    main()
