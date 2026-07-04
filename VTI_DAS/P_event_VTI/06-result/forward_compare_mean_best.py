#!/usr/bin/env python3
"""Forward-data comparison for posterior mean and best MCMC models.

This script is designed for the test6/test5 VTI DAS workflow.  It reads the
MCMC output chain.npz, reconstructs

    1. posterior mean model/source parameters, and
    2. best-sample model/source parameters,

then runs the same forward solver used by vti_joint_mcmc_dram.py and compares
predicted qP travel times with the noisy observations in 04-initial/output/nobs.dat.

For the station-pair P-wave objective, it also compares the transformed
receiver-pair differential data, using the same likelihood covariance as MCMC.

Outputs are written by default to:
    06-result/output/forward_compare/

Main outputs:
    forward_compare_summary.csv
    forward_compare_by_source.csv
    mean_predicted_ttime.dat
    best_predicted_ttime.dat
    mean_forward_comparison.npz
    best_forward_comparison.npz
    *_absolute_qP_obs_vs_pred.png
    *_diff_qP_obs_vs_pred.png
    *_source_abs_qP_rmse.png
    *_source_diff_qP_rmse.png
    arrival_plots/absolute_qP_all_sources_true_noise_mean_best.png
    arrival_plots/diff_qP_all_sources_true_noise_mean_best.png
    arrival_plots/absolute_by_source/source_XXX_absolute_qP_true_noise_mean_best.png
    arrival_plots/diff_by_source/source_XXX_diff_qP_true_noise_mean_best.png

The arrival-time comparison figures follow noise_add.py style:
    horizontal axis = time, vertical axis = receiver/station-pair index,
    and the vertical axis is inverted.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import importlib.util
import math
import sys
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None


def infer_project_root(user_root: Path | None = None) -> Path:
    """Infer project root containing 01-input, 04-initial, 05-inversion, 06-result."""
    if user_root is not None:
        return Path(user_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for base in (cwd, here):
        candidates.extend([base, base.parent, base.parent.parent])

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if all((c / d).is_dir() for d in ["01-input", "04-initial", "05-inversion", "06-result"]):
            return c

    if cwd.name == "06-result":
        return cwd.parent
    if here.name == "06-result":
        return here.parent
    return cwd


def import_mcmc_module(root: Path):
    """Import ../05-inversion/vti_joint_mcmc_dram.py as a module."""
    module_path = root / "05-inversion" / "vti_joint_mcmc_dram.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find MCMC module: {module_path}")
    spec = importlib.util.spec_from_file_location("vti_joint_mcmc_dram_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_string(pack: dict[str, Any], key: str, default: str) -> str:
    """Read a string-like scalar from np.load output robustly."""
    if key not in pack:
        return default
    value = np.asarray(pack[key])
    try:
        return str(value.item())
    except Exception:
        return str(value)


def selected_waves_from_text(text: str) -> tuple[str, ...]:
    tokens = [t.strip().upper() for t in text.replace(";", ",").split(",") if t.strip()]
    aliases = {"QP": "P", "P": "P", "QSV": "SV", "SV": "SV", "QSH": "SH", "SH": "SH"}
    waves: list[str] = []
    for t in tokens:
        if t in aliases and aliases[t] not in waves:
            waves.append(aliases[t])
    return tuple(waves) if waves else ("P",)


def finite_metrics(residual: np.ndarray) -> dict[str, float]:
    """Basic residual metrics on finite values."""
    r = np.asarray(residual, dtype=float)
    mask = np.isfinite(r)
    if not np.any(mask):
        return {"n": 0.0, "mean": math.nan, "rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
    x = r[mask]
    return {
        "n": float(x.size),
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "mae": float(np.mean(np.abs(x))),
        "max_abs": float(np.max(np.abs(x))),
    }


def write_predicted_ttime(path: Path, geo, pred: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Write predicted travel times in a simple ttime-like table."""
    tp, tsv, tsh = pred
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# ns nr\n")
        f.write(f"{geo.ns:d} {geo.nr:d}\n")
        f.write("# sx sz rx rz tp tsv tsh\n")
        for isrc in range(geo.ns):
            for ir in range(geo.nr):
                f.write(
                    f"{geo.sx[isrc]:.12f}\t{geo.sz[isrc]:.12f}\t"
                    f"{geo.rx[ir]:.12f}\t{geo.rz[ir]:.12f}\t"
                    f"{tp[isrc, ir]:.16e}\t{tsv[isrc, ir]:.16e}\t{tsh[isrc, ir]:.16e}\n"
                )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scatter_plot(path: Path, obs: np.ndarray, pred: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    if plt is None:
        return
    obs1 = np.asarray(obs, dtype=float).ravel()
    pred1 = np.asarray(pred, dtype=float).ravel()
    mask = np.isfinite(obs1) & np.isfinite(pred1)
    if not np.any(mask):
        return
    x = obs1[mask]
    y = pred1[mask]
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = 0.03 * max(hi - lo, 1.0e-12)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(
        x, y,
        marker=".", linestyle="none", markersize=2.5, alpha=0.45,
        label="Predicted vs observed",
    )
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        linestyle="--", linewidth=1.0, label="1:1 reference",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def source_metric_plot(path: Path, source_ids: np.ndarray, values: np.ndarray, title: str, ylabel: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(source_ids, values, marker="o", linewidth=1.0, label=ylabel)
    ax.set_xlabel("Source index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)




def read_ttime_table(path: Path, ns: int, nr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a ttime-like table and return tp/tsv/tsh arrays with shape (ns, nr).

    Expected numerical rows are either:
        sx sz rx rz tp tsv tsh
    or any longer row whose last three columns are tp/tsv/tsh.
    Comment/header lines are ignored.  A bare ``ns nr`` line is also ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot find ttime file: {path}")

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            vals: list[float] = []
            ok = True
            for part in parts:
                try:
                    vals.append(float(part))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            # skip the ns nr header line
            if len(vals) == 2 and int(vals[0]) == ns and int(vals[1]) == nr:
                continue
            if len(vals) >= 7:
                rows.append(vals)

    expected = ns * nr
    if len(rows) < expected:
        raise ValueError(f"{path} contains only {len(rows)} travel-time rows; expected at least {expected}")
    rows = rows[:expected]

    arr = np.asarray(rows, dtype=float)
    tp = arr[:, -3].reshape(ns, nr)
    tsv = arr[:, -2].reshape(ns, nr)
    tsh = arr[:, -1].reshape(ns, nr)
    return tp, tsv, tsh


def _noise_style_for_label(name: str) -> dict[str, Any]:
    """Line/marker style matching the visual convention used by noise_add.py."""
    key = name.lower()
    if "true" in key:
        return {"color": "k", "marker": "o", "linestyle": "-", "label": name}
    if "noise" in key or "obs" in key:
        return {"color": "k", "marker": "*", "linestyle": "-", "label": name}
    if "mean" in key:
        return {"color": "r", "marker": "o", "linestyle": "-", "label": name}
    if "best" in key:
        return {"color": "b", "marker": "o", "linestyle": "-", "label": name}
    return {"marker": ".", "linestyle": "-", "label": name}


def estimate_constant_time_shift(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Estimate a constant origin-time shift that best aligns predicted to reference.

    For a fixed source and model, the differential-time likelihood does not
    constrain the absolute origin time.  When plotting absolute arrivals, use

        t_pred_aligned = t_pred + t0

    with t0 chosen by least squares against the noiseless true arrival curve.
    This is simply the mean finite residual reference - predicted.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    pred = np.asarray(predicted, dtype=float).ravel()
    mask = np.isfinite(ref) & np.isfinite(pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(ref[mask] - pred[mask]))


def label_with_t0(label: str, t0: float | None) -> str:
    """Append estimated origin-time correction to a plot label."""
    if t0 is None or not np.isfinite(t0):
        return label
    return f"{label} (t0={t0:+.4f} s)"


def curve_plot(path: Path, receiver_or_pair_index: np.ndarray, series: list[tuple[str, np.ndarray]],
               title: str, vertical_label: str, horizontal_label: str) -> None:
    """Plot arrival-time curves in the same orientation as noise_add.py.

    noise_add.py uses:
        ax.plot(time, receiver_index, ...)
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Receiver")
        ax.invert_yaxis()

    Therefore ``receiver_or_pair_index`` is plotted on the vertical axis and
    each series value, e.g. qP arrival time, is plotted on the horizontal axis.
    """
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    yy = np.asarray(receiver_or_pair_index, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, xx in series:
        xx = np.asarray(xx, dtype=float)
        mask = np.isfinite(xx) & np.isfinite(yy)
        if np.any(mask):
            style = _noise_style_for_label(name)
            ax.plot(xx[mask], yy[mask], markersize=3.0, linewidth=1.0, **style)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(vertical_label)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def panel_arrival_plot(
    path: Path,
    receiver_or_pair_index: np.ndarray,
    source_series: list[list[tuple[str, np.ndarray]]],
    title: str,
    vertical_label: str,
    horizontal_label: str,
    ncols: int = 4,
) -> None:
    """Plot all sources as small panels using noise_add.py orientation."""
    if plt is None:
        return
    ns = len(source_series)
    if ns == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    yy = np.asarray(receiver_or_pair_index, dtype=float)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(ns / ncols))
    fig_w = 3.4 * ncols
    fig_h = 2.8 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    handles = None
    labels = None
    for isrc, series in enumerate(source_series):
        ax = axes[isrc // ncols][isrc % ncols]
        for name, xx in series:
            xx = np.asarray(xx, dtype=float)
            mask = np.isfinite(xx) & np.isfinite(yy)
            if np.any(mask):
                style = _noise_style_for_label(name)
                ax.plot(xx[mask], yy[mask], markersize=2.0, linewidth=0.9, **style)
        ax.grid(True)
        ax.invert_yaxis()
        ax.set_title(f"Source {isrc + 1}", fontsize=9)
        ax.tick_params(labelsize=8)
        if isrc // ncols == nrows - 1:
            ax.set_xlabel(horizontal_label, fontsize=8)
        if isrc % ncols == 0:
            ax.set_ylabel(vertical_label, fontsize=8)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    for j in range(ns, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9)
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=600)
    plt.close(fig)


def write_all_arrival_comparison_plots(
    outdir: Path,
    geo,
    like,
    obs,
    true_pred: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    model_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    mcmc=None,
) -> None:
    """Generate true/noisy/mean/best arrival-time comparison plots.

    The noisy curve is the observed nobs.dat qP data.  The true curve is read
    from the noiseless forward output 03-output/ttime.dat when available.
    """
    if plt is None:
        print("[WARN] matplotlib is unavailable; skip arrival comparison plots")
        return

    plot_dir = outdir / "arrival_plots"
    abs_dir = plot_dir / "absolute_by_source"
    diff_dir = plot_dir / "diff_by_source"
    plot_dir.mkdir(parents=True, exist_ok=True)
    abs_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    # noise_add.py plots travel time on the horizontal axis and receiver number on the vertical axis.
    # Use 1-based receiver index here so the figures match the original noise figures exactly.
    receiver_index = np.arange(1, int(geo.nr) + 1, dtype=float)
    receiver_ylabel = "Receiver"

    # For absolute-time plots only, align posterior mean/best curves to the
    # noiseless true qP arrivals with one constant source-time correction per
    # source and per model.  This does NOT change the saved forward predictions
    # or the MCMC/differential-time residuals; it is only a visual correction for
    # the origin-time ambiguity introduced by station-pair differential data.
    origin_shift_rows: list[dict[str, Any]] = []
    source_abs_series: list[list[tuple[str, np.ndarray]]] = []
    for isrc in range(geo.ns):
        series: list[tuple[str, np.ndarray]] = []
        true_qp = true_pred[0][isrc, :] if true_pred is not None else None
        if true_qp is not None:
            series.append(("true/noiseless", true_qp))
        series.append(("noisy obs", obs.tp[isrc, :]))
        for label in ("mean", "best"):
            if label in model_preds:
                raw_qp = model_preds[label][0][isrc, :]
                if true_qp is not None:
                    t0 = estimate_constant_time_shift(true_qp, raw_qp)
                    plot_qp = raw_qp + t0
                else:
                    t0 = float("nan")
                    plot_qp = raw_qp
                origin_shift_rows.append({
                    "source": int(isrc),
                    "model": label,
                    "estimated_t0_s": t0,
                    "estimated_t0_ms": 1000.0 * t0 if np.isfinite(t0) else float("nan"),
                })
                series.append((label_with_t0(label, t0), plot_qp))
        source_abs_series.append(series)
        curve_plot(
            abs_dir / f"source_{isrc:03d}_absolute_qP_true_noise_mean_best.png",
            receiver_index,
            series,
            f"Source {isrc + 1}: absolute qP arrival times",
            receiver_ylabel,
            "Time(s)",
        )

    if origin_shift_rows:
        write_csv(
            plot_dir / "absolute_qP_estimated_origin_time_shifts.csv",
            origin_shift_rows,
            ["source", "model", "estimated_t0_s", "estimated_t0_ms"],
        )

    panel_arrival_plot(
        plot_dir / "absolute_qP_all_sources_true_noise_mean_best.png",
        receiver_index,
        source_abs_series,
        "Absolute qP arrival-time comparison: true/noisy/mean/best",
        receiver_ylabel,
        "Time(s)",
        ncols=4,
    )

    # Differential station-pair arrival-time comparison, using exactly the same diff matrix as the 05-inversion likelihood.
    if like.diff_matrix is not None and like.obs_d is not None and like.obs_d.size > 0:
        pair_index = np.arange(1, like.obs_d.shape[1] + 1, dtype=float)
        if mcmc is not None and hasattr(mcmc, "apply_difference"):
            diff_func = lambda arr: mcmc.apply_difference(arr, like.diff_matrix)
        else:
            diff_func = lambda arr: np.asarray([like.diff_matrix @ arr[isrc, :] for isrc in range(arr.shape[0])])
        true_diff_all = diff_func(true_pred[0]) if true_pred is not None else None
        model_diff_all = {label: diff_func(pred[0]) for label, pred in model_preds.items()}
        source_diff_series: list[list[tuple[str, np.ndarray]]] = []
        for isrc in range(geo.ns):
            series = []
            if true_diff_all is not None:
                series.append(("true/noiseless", true_diff_all[isrc, :]))
            series.append(("noisy obs", like.obs_d[isrc, :]))
            for label in ("mean", "best"):
                if label in model_preds:
                    series.append((label, model_diff_all[label][isrc, :]))
            source_diff_series.append(series)
            curve_plot(
                diff_dir / f"source_{isrc:03d}_diff_qP_true_noise_mean_best.png",
                pair_index,
                series,
                f"Source {isrc + 1}: station-pair differential qP arrivals ({like.objective_type})",
                "Adjacent station-pair index" if like.objective_type == "diff-p-adjacent" else "Reference-pair index",
                "Differential Time(s)",
            )

        panel_arrival_plot(
            plot_dir / "diff_qP_all_sources_true_noise_mean_best.png",
            pair_index,
            source_diff_series,
            f"Station-pair differential qP comparison: true/noisy/mean/best ({like.objective_type})",
            "Adjacent station-pair index" if like.objective_type == "diff-p-adjacent" else "Reference-pair index",
            "Differential Time(s)",
            ncols=4,
        )

def compare_one_model(label: str, theta: np.ndarray, args, root: Path, mcmc, geo0, model0, obs, like, outdir: Path,
                      qx_stop: float, qx_max_iter: int, invert_depths: bool, fix_last_layer: bool,
                      forward_waves: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Forward one parameter vector and compare predicted/observed data."""
    geo, model = mcmc.unpack_params(theta, geo0, model0, invert_depths, fix_last_layer)
    pred = mcmc.forward_direct(geo, model, stop=qx_stop, max_iter=qx_max_iter, needed_waves=forward_waves)
    tp, tsv, tsh = pred

    # MCMC objective misfit, using the same transformed data and covariance.
    objective_misfit = float(mcmc.misfit_from_times(pred, obs, like))

    abs_p_res = tp - obs.tp
    abs_m = finite_metrics(abs_p_res)

    if like.objective_type in {"diff-p-adjacent", "diff-p-reference"} and like.diff_matrix is not None and like.obs_d is not None:
        pred_d = mcmc.apply_difference(tp, like.diff_matrix)
        diff_res = pred_d - like.obs_d
        diff_m = finite_metrics(diff_res)
        diff_qform_by_source = np.einsum("si,ij,sj->s", diff_res, like.cov_inv, diff_res) if like.cov_inv is not None else np.full(geo.ns, np.nan)
    else:
        pred_d = np.full((geo.ns, 0), np.nan)
        diff_res = np.full((geo.ns, 0), np.nan)
        diff_m = {"n": 0.0, "mean": math.nan, "rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
        diff_qform_by_source = np.full(geo.ns, np.nan)

    summary = {
        "model": label,
        "objective_type": like.objective_type,
        "sigma_mode": like.sigma_mode,
        "sigma_abs": float(obs.sigma),
        "objective_misfit": objective_misfit,
        "absolute_qP_n": int(abs_m["n"]),
        "absolute_qP_mean_residual": abs_m["mean"],
        "absolute_qP_rmse": abs_m["rmse"],
        "absolute_qP_mae": abs_m["mae"],
        "absolute_qP_max_abs": abs_m["max_abs"],
        "diff_qP_n": int(diff_m["n"]),
        "diff_qP_mean_residual": diff_m["mean"],
        "diff_qP_rmse": diff_m["rmse"],
        "diff_qP_mae": diff_m["mae"],
        "diff_qP_max_abs": diff_m["max_abs"],
        "ns": geo.ns,
        "nr": geo.nr,
    }

    by_source_rows: list[dict[str, Any]] = []
    abs_rmse_per_source = []
    diff_rmse_per_source = []
    for isrc in range(geo.ns):
        am = finite_metrics(abs_p_res[isrc, :])
        dm = finite_metrics(diff_res[isrc, :]) if diff_res.shape[1] > 0 else {"rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
        abs_rmse_per_source.append(am["rmse"])
        diff_rmse_per_source.append(dm["rmse"])
        by_source_rows.append({
            "model": label,
            "source": isrc,
            "sx": float(geo.sx[isrc]),
            "sz": float(geo.sz[isrc]),
            "absolute_qP_rmse": am["rmse"],
            "absolute_qP_mae": am["mae"],
            "absolute_qP_max_abs": am["max_abs"],
            "diff_qP_rmse": dm["rmse"],
            "diff_qP_mae": dm["mae"],
            "diff_qP_max_abs": dm["max_abs"],
            "diff_qP_qform": float(diff_qform_by_source[isrc]) if isrc < len(diff_qform_by_source) else math.nan,
        })

    write_predicted_ttime(outdir / f"{label}_predicted_ttime.dat", geo, pred)
    mcmc.write_model_summary(outdir / f"{label}_model_summary.dat", model, geo, objective_misfit)
    np.savez(
        outdir / f"{label}_forward_comparison.npz",
        sx=geo.sx, sz=geo.sz, rx=geo.rx, rz=geo.rz,
        pred_tp=tp, pred_tsv=tsv, pred_tsh=tsh,
        obs_tp=obs.tp, obs_tsv=obs.tsv, obs_tsh=obs.tsh,
        abs_p_residual=abs_p_res,
        pred_diff_qp=pred_d,
        obs_diff_qp=like.obs_d if like.obs_d is not None else np.asarray([]),
        diff_qp_residual=diff_res,
        objective_misfit=objective_misfit,
    )

    scatter_plot(
        outdir / f"{label}_absolute_qP_obs_vs_pred.png",
        obs.tp, tp,
        f"{label}: absolute qP observed vs predicted",
        "Observed qP time", "Predicted qP time",
    )
    if diff_res.shape[1] > 0:
        scatter_plot(
            outdir / f"{label}_diff_qP_obs_vs_pred.png",
            like.obs_d, pred_d,
            f"{label}: station-pair qP observed vs predicted",
            "Observed differential qP", "Predicted differential qP",
        )
    source_ids = np.arange(geo.ns)
    source_metric_plot(
        outdir / f"{label}_source_abs_qP_rmse.png",
        source_ids, np.asarray(abs_rmse_per_source, dtype=float),
        f"{label}: absolute qP RMSE by source", "RMSE",
    )
    if diff_res.shape[1] > 0:
        source_metric_plot(
            outdir / f"{label}_source_diff_qP_rmse.png",
            source_ids, np.asarray(diff_rmse_per_source, dtype=float),
            f"{label}: differential qP RMSE by source", "RMSE",
        )

    print(
        f"[{label.upper()}] objective_misfit={objective_misfit:.6g}, "
        f"abs_qP_RMSE={summary['absolute_qP_rmse']:.6g}, "
        f"diff_qP_RMSE={summary['diff_qP_rmse']:.6g}"
    )
    return summary, by_source_rows, pred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare posterior mean/best forward predictions with observations.")
    parser.add_argument("--root", type=Path, default=None, help="Project root containing 01-input, 04-initial, 05-inversion, 06-result")
    parser.add_argument("--input-dir", type=Path, default=None, help="Initial inversion input dir; default=root/04-initial/output")
    parser.add_argument("--result-dir", type=Path, default=None, help="MCMC output dir containing chain.npz; default=root/06-result/output")
    parser.add_argument("--output-dir", type=Path, default=None, help="Comparison output dir; default=result-dir/forward_compare")
    parser.add_argument("--control-file", type=Path, default=None, help="control.dat for qx stop/max_iter; default=root/01-input/output/control.dat")
    parser.add_argument("--true-ttime-file", type=Path, default=None, help="Noiseless true ttime.dat; default=root/03-output/ttime.dat")
    parser.add_argument("--plot-arrival-comparisons", dest="plot_arrival_comparisons", action="store_true", default=True, help="Write true/noisy/mean/best arrival-time comparison plots")
    parser.add_argument("--no-plot-arrival-comparisons", dest="plot_arrival_comparisons", action="store_false")
    parser.add_argument("--qx-stop", type=float, default=None, help="Override qx solver stop tolerance")
    parser.add_argument("--qx-max-iter", type=int, default=None, help="Override qx solver max iterations")
    parser.add_argument("--objective", choices=["from-chain", "diff-p-adjacent", "diff-p-reference", "absolute"], default="from-chain")
    parser.add_argument("--sigma-mode", choices=["from-chain", "absolute", "objective-iid"], default="from-chain")
    parser.add_argument("--forward-waves", default="P", help="Waves to forward: P, P,SV,SH, etc. Default P because current objective uses qP station-pair data.")
    parser.add_argument("--invert-depths", dest="invert_depths", action="store_true", default=True)
    parser.add_argument("--no-invert-depths", dest="invert_depths", action="store_false")
    parser.add_argument("--fix-last-layer", action="store_true", default=False)
    args = parser.parse_args(argv)

    root = infer_project_root(args.root)
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir is not None else root / "04-initial" / "output"
    result_dir = args.result_dir.expanduser().resolve() if args.result_dir is not None else root / "06-result" / "output"
    outdir = args.output_dir.expanduser().resolve() if args.output_dir is not None else result_dir / "forward_compare"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] project root = {root}")
    print(f"[PATH] input_dir    = {input_dir}")
    print(f"[PATH] result_dir   = {result_dir}")
    print(f"[PATH] output_dir   = {outdir}")

    mcmc = import_mcmc_module(root)

    chain_path = result_dir / "chain.npz"
    if not chain_path.exists():
        raise FileNotFoundError(f"Cannot find chain file: {chain_path}")
    pack = dict(np.load(chain_path, allow_pickle=True))

    geo0 = mcmc.read_geometry(input_dir / "geo.dat")
    model0 = mcmc.read_model(input_dir / "vel.dat")
    obs = mcmc.read_observed(input_dir / "nobs.dat")

    true_ttime_file = args.true_ttime_file.expanduser().resolve() if args.true_ttime_file is not None else root / "03-output" / "ttime.dat"
    true_pred = None
    if true_ttime_file.exists():
        try:
            true_pred = read_ttime_table(true_ttime_file, geo0.ns, geo0.nr)
            print(f"[TRUE] loaded noiseless true ttime from {true_ttime_file}")
        except Exception as exc:
            print(f"[WARN] failed to read true ttime file {true_ttime_file}: {exc}")
    else:
        print(f"[WARN] true ttime file not found: {true_ttime_file}")

    objective = scalar_string(pack, "objective_type", "diff-p-adjacent") if args.objective == "from-chain" else args.objective
    sigma_mode = scalar_string(pack, "sigma_mode", "absolute") if args.sigma_mode == "from-chain" else args.sigma_mode
    like = mcmc.build_likelihood_config(obs, objective_type=objective, use_waves="P", sigma_mode=sigma_mode)

    control_file = args.control_file.expanduser().resolve() if args.control_file is not None else root / "01-input" / "output" / "control.dat"
    if args.qx_stop is not None and args.qx_max_iter is not None:
        qx_stop = float(args.qx_stop)
        qx_max_iter = int(args.qx_max_iter)
    elif control_file.exists():
        qx_stop, qx_max_iter = mcmc.read_forward_control(control_file)
        if args.qx_stop is not None:
            qx_stop = float(args.qx_stop)
        if args.qx_max_iter is not None:
            qx_max_iter = int(args.qx_max_iter)
    else:
        qx_stop = float(args.qx_stop) if args.qx_stop is not None else float(mcmc.DEFAULT_STOP)
        qx_max_iter = int(args.qx_max_iter) if args.qx_max_iter is not None else int(mcmc.DEFAULT_MAX_ITER)

    forward_waves = selected_waves_from_text(args.forward_waves)
    # The comparison always needs P because the station-pair objective is qP.
    if "P" not in forward_waves:
        forward_waves = ("P",) + forward_waves

    if "mean_theta" not in pack:
        raise KeyError("chain.npz does not contain mean_theta")
    mean_theta = np.asarray(pack["mean_theta"], dtype=float)

    if "best_theta" in pack:
        best_theta = np.asarray(pack["best_theta"], dtype=float)
    else:
        chain = np.asarray(pack["chain"], dtype=float)
        misfit = np.asarray(pack["misfit"], dtype=float)
        best_theta = chain[int(np.argmin(misfit))]

    print(f"[CONFIG] objective={objective}, sigma_mode={sigma_mode}, sigma={obs.sigma:.6g}")
    print(f"[CONFIG] qx_stop={qx_stop:.6g}, qx_max_iter={qx_max_iter}, forward_waves={','.join(forward_waves)}")

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    model_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, theta in [("mean", mean_theta), ("best", best_theta)]:
        summary, by_source, pred = compare_one_model(
            label=label,
            theta=theta,
            args=args,
            root=root,
            mcmc=mcmc,
            geo0=geo0,
            model0=model0,
            obs=obs,
            like=like,
            outdir=outdir,
            qx_stop=qx_stop,
            qx_max_iter=qx_max_iter,
            invert_depths=args.invert_depths,
            fix_last_layer=args.fix_last_layer,
            forward_waves=forward_waves,
        )
        summary_rows.append(summary)
        source_rows.extend(by_source)
        model_preds[label] = pred

    if args.plot_arrival_comparisons:
        write_all_arrival_comparison_plots(
            outdir=outdir,
            geo=geo0,
            like=like,
            obs=obs,
            true_pred=true_pred,
            model_preds=model_preds,
            mcmc=mcmc,
        )

    summary_fields = [
        "model", "objective_type", "sigma_mode", "sigma_abs", "objective_misfit",
        "absolute_qP_n", "absolute_qP_mean_residual", "absolute_qP_rmse", "absolute_qP_mae", "absolute_qP_max_abs",
        "diff_qP_n", "diff_qP_mean_residual", "diff_qP_rmse", "diff_qP_mae", "diff_qP_max_abs",
        "ns", "nr",
    ]
    source_fields = [
        "model", "source", "sx", "sz",
        "absolute_qP_rmse", "absolute_qP_mae", "absolute_qP_max_abs",
        "diff_qP_rmse", "diff_qP_mae", "diff_qP_max_abs", "diff_qP_qform",
    ]
    write_csv(outdir / "forward_compare_summary.csv", summary_rows, summary_fields)
    write_csv(outdir / "forward_compare_by_source.csv", source_rows, source_fields)

    print(f"[DONE] comparison files saved to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
