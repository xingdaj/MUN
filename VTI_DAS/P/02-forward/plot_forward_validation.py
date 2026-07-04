#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot and validate diagnostic outputs from vti_direct_mcmc_ready.py.

Purpose
-------
Read the files written in ../03-output or ../03-ouput:
  - qx.dat
  - ttime.dat
  - diagnostics.dat
  - layer_contributions.dat
  - iteration_detailed.dat
  - input_summary.dat

Then generate figures to check whether the forward calculation is internally correct:
  1. qx maps for qP/qSV/qSH
  2. travel-time maps for qP/qSV/qSH
  3. offset residual and dx-sum residual histograms
  4. Newton iteration count histograms
  5. travel-time versus receiver depth for selected sources
  6. qx versus receiver depth for selected sources
  7. layer contribution checks: sum(layer_dx)-H and sum(layer_dt)-ttime
  8. convergence histories for selected non-horizontal pairs

This script intentionally uses only numpy and matplotlib, not pandas.

Usage
-----
Put this script in 02-forward and run:

    python plot_forward_validation.py

or specify output directory:

    python plot_forward_validation.py --output-dir ../03-ouput

The generated figures will be saved to:

    ../03-ouput/validation_plots/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


WAVE_NAMES = {1: "qP", 2: "qSV", 3: "qSH"}
WAVE_COLS = {"qP": 4, "qSV": 5, "qSH": 6}


def auto_find_output_dir(user_dir: str | None = None) -> Path:
    """Find output directory robustly.

    Your current forward code uses '../03-ouput' with a typo.
    This script also supports '../03-output'.
    """
    if user_dir is not None:
        p = Path(user_dir)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified output directory does not exist: {p}")

    candidates = [
        Path("../03-ouput"),
        Path("../03-output"),
        Path("03-ouput"),
        Path("03-output"),
        Path("."),
    ]
    required = ["qx.dat", "ttime.dat", "diagnostics.dat"]
    for p in candidates:
        if all((p / name).is_file() for name in required):
            return p

    raise FileNotFoundError(
        "Cannot find output directory. Expected qx.dat, ttime.dat, diagnostics.dat "
        "in ../03-ouput or ../03-output."
    )


def read_matrix_file(path: Path):
    """Read qx.dat or ttime.dat.

    Format:
      first line: ns nr
      following rows: sx sz rx rz qP qSV qSH
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().split()
    if len(header) < 2:
        raise ValueError(f"Bad header in {path}")
    ns, nr = int(float(header[0])), int(float(header[1]))

    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    expected = ns * nr
    if data.shape[0] != expected:
        raise ValueError(f"{path.name}: expected {expected} rows from ns*nr, got {data.shape[0]}")
    if data.shape[1] < 7:
        raise ValueError(f"{path.name}: expected at least 7 columns, got {data.shape[1]}")

    return ns, nr, data


def read_diagnostics(path: Path):
    """Read diagnostics.dat.

    Columns:
      isrc ircv wave_id H qx px ttime offset_error niter converged dx_sum dx_minus_H
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 12:
        raise ValueError(f"diagnostics.dat should have 12 columns, got {data.shape[1]}")
    return data


def read_layer_contributions(path: Path):
    """Read layer_contributions.dat.

    Columns:
      isrc ircv wave_id layer Z layer_dx layer_dt pz Vz g0 g1
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 11:
        raise ValueError(f"layer_contributions.dat should have 11 columns, got {data.shape[1]}")
    return data


def read_iteration_detailed(path: Path):
    """Read iteration_detailed.dat.

    Columns:
      isrc ircv wave_id iter qx px Xcal f df d2f disc chosen_qx chosen_f
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 13:
        raise ValueError(f"iteration_detailed.dat should have 13 columns, got {data.shape[1]}")
    return data


def finite_check(name: str, arr: np.ndarray, summary: list[str]):
    nbad = int(np.sum(~np.isfinite(arr)))
    if nbad == 0:
        summary.append(f"[OK] {name}: all values finite")
    else:
        summary.append(f"[BAD] {name}: {nbad} non-finite values found")


def reshape_wave(data: np.ndarray, ns: int, nr: int, wave_name: str):
    col = WAVE_COLS[wave_name]
    return data[:, col].reshape(ns, nr)


def plot_matrix(mat: np.ndarray, title: str, xlabel: str, ylabel: str, cbar_label: str, out_path: Path):
    plt.figure(figsize=(9, 5))
    im = plt.imshow(mat, aspect="auto", origin="lower")
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 60, logy: bool = False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    plt.figure(figsize=(7, 4.5))
    plt.hist(values, bins=bins)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_by_receiver_depth(data: np.ndarray, ns: int, nr: int, value_col: int, ylabel: str,
                           title_prefix: str, out_path: Path, selected_sources: list[int]):
    """Plot value versus receiver depth for selected source indices."""
    plt.figure(figsize=(8, 5))
    for isrc in selected_sources:
        if isrc < 0 or isrc >= ns:
            continue
        block = data[isrc * nr:(isrc + 1) * nr, :]
        rz = block[:, 3]
        vals = block[:, value_col]
        order = np.argsort(rz)
        sx, sz = block[0, 0], block[0, 1]
        plt.plot(rz[order], vals[order], marker=".", linewidth=1, label=f"src {isrc}: x={sx:g}, z={sz:g}")
    plt.xlabel("Receiver depth z (m)")
    plt.ylabel(ylabel)
    plt.title(title_prefix)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_iteration_examples(iter_data: np.ndarray, diag: np.ndarray, out_dir: Path, max_examples: int = 9):
    """Plot convergence histories for selected non-horizontal examples."""
    # Choose examples with niter > 0 and a mix of waves.
    examples = []
    for wave_id in (1, 2, 3):
        sub = diag[(diag[:, 2].astype(int) == wave_id) & (diag[:, 8] > 0)]
        if sub.size == 0:
            continue
        # Pick low, middle, high H examples.
        order = np.argsort(sub[:, 3])
        pick_idx = sorted(set([0, len(order) // 2, len(order) - 1]))
        for idx in pick_idx:
            row = sub[order[idx]]
            examples.append((int(row[0]), int(row[1]), int(row[2])))
    examples = examples[:max_examples]

    for isrc, ircv, wave_id in examples:
        sub = iter_data[
            (iter_data[:, 0].astype(int) == isrc)
            & (iter_data[:, 1].astype(int) == ircv)
            & (iter_data[:, 2].astype(int) == wave_id)
        ]
        if sub.size == 0:
            continue
        order = np.argsort(sub[:, 3])
        sub = sub[order]
        it = sub[:, 3]
        f = sub[:, 7]
        qx = sub[:, 4]

        plt.figure(figsize=(7, 4.5))
        plt.plot(it, np.abs(f), marker="o")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("|offset error| (m)")
        plt.title(f"Convergence: {WAVE_NAMES[wave_id]}, source={isrc}, receiver={ircv}")
        plt.tight_layout()
        plt.savefig(out_dir / f"convergence_{WAVE_NAMES[wave_id]}_src{isrc:02d}_rec{ircv:03d}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(it, qx, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("qx")
        plt.title(f"qx history: {WAVE_NAMES[wave_id]}, source={isrc}, receiver={ircv}")
        plt.tight_layout()
        plt.savefig(out_dir / f"qx_history_{WAVE_NAMES[wave_id]}_src{isrc:02d}_rec{ircv:03d}.png", dpi=300)
        plt.close()


def layer_internal_consistency(layer: np.ndarray, diag: np.ndarray, summary: list[str], out_dir: Path):
    """Check layer sums against diagnostics.

    layer rows:
      0 isrc, 1 ircv, 2 wave_id, 3 layer, 4 Z, 5 layer_dx, 6 layer_dt

    diag rows:
      0 isrc, 1 ircv, 2 wave_id, 3 H, 6 ttime, 10 dx_sum, 11 dx_minus_H
    """
    keys = diag[:, 0].astype(int) * 10_000_000 + diag[:, 1].astype(int) * 10_000 + diag[:, 2].astype(int)
    diag_map = {int(k): row for k, row in zip(keys, diag)}

    residual_dx = []
    residual_dt = []
    n_paths = 0

    unique_keys = np.unique(layer[:, 0].astype(int) * 10_000_000 + layer[:, 1].astype(int) * 10_000 + layer[:, 2].astype(int))
    for key in unique_keys:
        sub = layer[
            (layer[:, 0].astype(int) * 10_000_000 + layer[:, 1].astype(int) * 10_000 + layer[:, 2].astype(int)) == key
        ]
        drow = diag_map.get(int(key))
        if drow is None:
            continue
        n_paths += 1
        sum_dx = float(np.nansum(sub[:, 5]))
        sum_dt = float(np.nansum(sub[:, 6]))
        H = float(drow[3])
        ttime = float(drow[6])
        residual_dx.append(sum_dx - H)
        residual_dt.append(sum_dt - ttime)

    residual_dx = np.asarray(residual_dx)
    residual_dt = np.asarray(residual_dt)

    summary.append(f"[CHECK] layer paths checked: {n_paths}")
    summary.append(f"[CHECK] max|sum(layer_dx)-H| = {np.nanmax(np.abs(residual_dx)):.12e} m")
    summary.append(f"[CHECK] max|sum(layer_dt)-ttime| = {np.nanmax(np.abs(residual_dt)):.12e} s")

    plot_hist(residual_dx, "Layer check: sum(layer_dx)-H", "Residual (m)",
              out_dir / "hist_layer_sum_dx_minus_H.png", bins=80, logy=True)
    plot_hist(residual_dt, "Layer check: sum(layer_dt)-ttime", "Residual (s)",
              out_dir / "hist_layer_sum_dt_minus_ttime.png", bins=80, logy=True)


def make_summary(ns: int, nr: int, qx: np.ndarray, tt: np.ndarray, diag: np.ndarray,
                 layer: np.ndarray | None, summary_path: Path):
    summary = []
    summary.append("Forward validation summary")
    summary.append("=" * 80)
    summary.append(f"ns = {ns}, nr = {nr}, total pairs = {ns * nr}")
    summary.append("")

    finite_check("qx.dat", qx, summary)
    finite_check("ttime.dat", tt, summary)
    finite_check("diagnostics.dat", diag, summary)
    if layer is not None:
        finite_check("layer_contributions.dat", layer, summary)
    summary.append("")

    for wave_id, wave_name in WAVE_NAMES.items():
        sub = diag[diag[:, 2].astype(int) == wave_id]
        if sub.size == 0:
            summary.append(f"[BAD] {wave_name}: no diagnostics rows")
            continue
        nbad = int(np.sum(sub[:, 9].astype(int) == 0))
        max_err = float(np.nanmax(np.abs(sub[:, 7])))
        max_dx_err = float(np.nanmax(np.abs(sub[:, 11])))
        min_tt = float(np.nanmin(sub[:, 6]))
        max_tt = float(np.nanmax(sub[:, 6]))
        min_iter = int(np.nanmin(sub[:, 8]))
        max_iter = int(np.nanmax(sub[:, 8]))
        n_horizontal = int(np.sum(sub[:, 8] == 0))
        summary.append(
            f"[CHECK] {wave_name}: nonconverged={nbad}, "
            f"max|offset_error|={max_err:.12e} m, "
            f"max|sum_dx-H|={max_dx_err:.12e} m, "
            f"ttime_range=({min_tt:.12e}, {max_tt:.12e}) s, "
            f"niter_range=({min_iter}, {max_iter}), "
            f"horizontal_or_direct_special_cases(niter=0)={n_horizontal}"
        )

    summary.append("")
    # Physical sanity checks.
    for wave_name in ("qP", "qSV", "qSH"):
        mat_tt = reshape_wave(tt, ns, nr, wave_name)
        mat_qx = reshape_wave(qx, ns, nr, wave_name)
        n_neg_tt = int(np.sum(mat_tt < 0))
        n_neg_qx = int(np.sum(mat_qx < 0))
        summary.append(f"[SANITY] {wave_name}: negative ttime count={n_neg_tt}, negative qx count={n_neg_qx}")

    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot and validate VTI direct-forward diagnostic output files.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory containing qx.dat, ttime.dat, diagnostics.dat. "
                             "Default: auto-detect ../03-ouput or ../03-output.")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for validation plots. Default: output_dir/validation_plots")
    parser.add_argument("--show", action="store_true", help="Show figures interactively instead of only saving.")
    args = parser.parse_args()

    output_dir = auto_find_output_dir(args.output_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir / "validation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    qx_ns, qx_nr, qx_data = read_matrix_file(output_dir / "qx.dat")
    tt_ns, tt_nr, tt_data = read_matrix_file(output_dir / "ttime.dat")
    if (qx_ns, qx_nr) != (tt_ns, tt_nr):
        raise ValueError(f"qx.dat ns/nr={(qx_ns, qx_nr)} but ttime.dat ns/nr={(tt_ns, tt_nr)}")

    ns, nr = qx_ns, qx_nr
    diag = read_diagnostics(output_dir / "diagnostics.dat")

    layer = None
    layer_path = output_dir / "layer_contributions.dat"
    if layer_path.is_file():
        layer = read_layer_contributions(layer_path)

    iter_data = None
    iter_path = output_dir / "iteration_detailed.dat"
    if iter_path.is_file():
        iter_data = read_iteration_detailed(iter_path)

    # Basic matrix plots.
    for wave_name in ("qP", "qSV", "qSH"):
        qx_mat = reshape_wave(qx_data, ns, nr, wave_name)
        tt_mat = reshape_wave(tt_data, ns, nr, wave_name)

        plot_matrix(qx_mat, f"{wave_name} qx map", "Receiver index", "Source index", "qx",
                    plot_dir / f"map_qx_{wave_name}.png")
        plot_matrix(tt_mat, f"{wave_name} travel-time map", "Receiver index", "Source index", "Travel time (s)",
                    plot_dir / f"map_ttime_{wave_name}.png")

    # Diagnostics histograms.
    for wave_id, wave_name in WAVE_NAMES.items():
        sub = diag[diag[:, 2].astype(int) == wave_id]
        plot_hist(sub[:, 7], f"{wave_name}: offset error", "Xcal - H (m)",
                  plot_dir / f"hist_offset_error_{wave_name}.png", bins=80, logy=True)
        plot_hist(sub[:, 11], f"{wave_name}: sum(layer_dx)-H", "dx_sum - H (m)",
                  plot_dir / f"hist_dx_minus_H_{wave_name}.png", bins=80, logy=True)
        plot_hist(sub[:, 8], f"{wave_name}: iteration count", "Number of iterations",
                  plot_dir / f"hist_niter_{wave_name}.png", bins=np.arange(-0.5, np.nanmax(sub[:, 8]) + 1.5, 1), logy=False)

    # Curves versus receiver depth for selected sources.
    selected_sources = sorted(set([0, ns // 2, ns - 1]))
    for wave_name in ("qP", "qSV", "qSH"):
        plot_by_receiver_depth(tt_data, ns, nr, WAVE_COLS[wave_name],
                               f"{wave_name} travel time (s)",
                               f"{wave_name} travel time versus receiver depth",
                               plot_dir / f"curves_ttime_vs_rz_{wave_name}.png",
                               selected_sources)
        plot_by_receiver_depth(qx_data, ns, nr, WAVE_COLS[wave_name],
                               f"{wave_name} qx",
                               f"{wave_name} qx versus receiver depth",
                               plot_dir / f"curves_qx_vs_rz_{wave_name}.png",
                               selected_sources)

    summary = make_summary(ns, nr, qx_data, tt_data, diag, layer, plot_dir / "forward_validation_summary.txt")

    if layer is not None:
        layer_internal_consistency(layer, diag, summary, plot_dir)
        # Rewrite summary with layer checks included.
        (plot_dir / "forward_validation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    if iter_data is not None:
        plot_iteration_examples(iter_data, diag, plot_dir)

    print("\n".join(summary))
    print("")
    print(f"[DONE] Figures and summary saved to: {plot_dir.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
