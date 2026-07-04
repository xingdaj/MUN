#!/usr/bin/env python3
"""Forward-data comparison for posterior mean and best MCMC models.

This script is designed for the test6/test5 VTI DAS workflow.  It reads the
MCMC output chain.npz, reconstructs

    1. posterior mean model/source parameters, and
    2. best-sample model/source parameters,

then runs the same forward solver used by vti_joint_mcmc_dram.py and compares
predicted qP/qSV/qSH travel times with the noisy observations in 04-initial/output/nobs.dat.

For station-pair differential objectives, it also compares the transformed
receiver-pair differential data for every selected phase, using the same
differencing matrix and likelihood covariance as MCMC.

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
    *_diff_qP/qSV/qSH_obs_vs_pred.png
    *_source_abs_qP_rmse.png
    *_source_diff_qP/qSV/qSH_rmse.png
    arrival_plots/absolute_qP/qSV/qSH_all_sources_true_noise_mean_best.png
    arrival_plots/diff_qP/qSV/qSH_all_sources_true_noise_mean_best.png
    arrival_plots/absolute_estimated_origin_times_by_source.csv
    arrival_plots/absolute_by_source/source_XXX_absolute_qP_qSV_qSH_true_noise_mean_best_1x3.png
    arrival_plots/diff_by_source/source_XXX_diff_qP_qSV_qSH_true_noise_mean_best_1x3.png

The arrival-time comparison figures follow noise_add.py style:
    horizontal axis = time/differential time, vertical axis = receiver/station-pair index,
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
    fig.savefig(path, dpi=300)
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
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _wave_arrays(pred: tuple[np.ndarray, np.ndarray, np.ndarray], obs, wave: str) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted and observed arrays for one wave label."""
    wave = wave.upper()
    if wave == "P":
        return pred[0], obs.tp
    if wave == "SV":
        return pred[1], obs.tsv
    if wave == "SH":
        return pred[2], obs.tsh
    raise ValueError(f"Unknown wave label: {wave}")


def _wave_label(wave: str) -> str:
    return {"P": "qP", "SV": "qSV", "SH": "qSH"}.get(wave.upper(), wave)


def _wave_pred(pred: tuple[np.ndarray, np.ndarray, np.ndarray], wave: str) -> np.ndarray:
    wave = wave.upper()
    if wave == "P":
        return pred[0]
    if wave == "SV":
        return pred[1]
    if wave == "SH":
        return pred[2]
    raise ValueError(f"Unknown wave label: {wave}")


def _wave_obs(obs, wave: str) -> np.ndarray:
    wave = wave.upper()
    if wave == "P":
        return obs.tp
    if wave == "SV":
        return obs.tsv
    if wave == "SH":
        return obs.tsh
    raise ValueError(f"Unknown wave label: {wave}")


def _obs_diff_3d(like) -> np.ndarray | None:
    if getattr(like, "obs_d", None) is None:
        return None
    arr = np.asarray(like.obs_d, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr



def _available_plot_waves() -> tuple[str, str, str]:
    """Waves to show in diagnostic arrival-time plots."""
    return ("P", "SV", "SH")


def _wave_reference_array(
    true_pred: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    obs,
    wave: str,
) -> np.ndarray:
    """Use noiseless true arrivals as reference when available; otherwise use observations."""
    if true_pred is not None:
        return _wave_pred(true_pred, wave)
    return _wave_obs(obs, wave)


def estimate_source_origin_time_multiwave(
    reference_pred: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    obs,
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    waves: tuple[str, ...],
    isrc: int,
) -> tuple[float, float, int]:
    """Estimate one source-origin time shift from all requested phases.

    For one source and one predicted model, the best constant origin-time shift
    is the least-squares solution of

        reference_time ~= predicted_travel_time + t0

    using all finite receiver samples from qP/qSV/qSH together.  Therefore each
    source has exactly one t0 for each model (mean or best), not one t0 per phase.

    Returns
    -------
    t0 : float
        Best common origin-time shift in seconds.
    rmse_after_shift : float
        RMSE against the reference after applying the common shift.
    n_values : int
        Number of finite phase/receiver samples used.
    """
    residual_chunks: list[np.ndarray] = []
    for wave in waves:
        ref = _wave_reference_array(reference_pred, obs, wave)[isrc, :]
        pred = _wave_pred(predicted, wave)[isrc, :]
        mask = np.isfinite(ref) & np.isfinite(pred)
        if np.any(mask):
            residual_chunks.append(ref[mask] - pred[mask])
    if not residual_chunks:
        return float("nan"), float("nan"), 0
    residual = np.concatenate(residual_chunks)
    t0 = float(np.mean(residual))
    rmse_after_shift = float(np.sqrt(np.mean((residual - t0) ** 2)))
    return t0, rmse_after_shift, int(residual.size)


def multiphase_source_plot(
    path: Path,
    receiver_or_pair_index: np.ndarray,
    phase_series: list[tuple[str, list[tuple[str, np.ndarray]]]],
    title: str,
    vertical_label: str,
    horizontal_label: str,
) -> None:
    """Plot qP/qSV/qSH for one source as a 1 x N phase panel.

    This keeps the noise_add.py orientation: time is horizontal and receiver or
    station-pair index is vertical.  When N=3, the output is the requested 1 x 3
    qP/qSV/qSH figure for one source.
    """
    if plt is None:
        return
    if not phase_series:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    yy = np.asarray(receiver_or_pair_index, dtype=float)
    nphase = len(phase_series)
    fig_w = max(5.0, 4.8 * nphase)
    fig, axes = plt.subplots(1, nphase, figsize=(fig_w, 4.8), sharey=True, squeeze=False)
    handles = None
    labels = None
    for iphase, (phase_label, series) in enumerate(phase_series):
        ax = axes[0][iphase]
        for name, xx in series:
            xx = np.asarray(xx, dtype=float)
            mask = np.isfinite(xx) & np.isfinite(yy)
            if np.any(mask):
                style = _noise_style_for_label(name)
                ax.plot(xx[mask], yy[mask], markersize=3.0, linewidth=1.0, **style)
        ax.grid(True)
        ax.set_title(phase_label, fontsize=11)
        ax.set_xlabel(horizontal_label)
        if iphase == 0:
            ax.set_ylabel(vertical_label)
        ax.invert_yaxis()
        ax.tick_params(labelsize=10)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9)
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300)
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
    """Generate qP/qSV/qSH absolute-arrival and station-pair differential plots.

    Absolute plots use one common origin-time shift for each source and model.
    The shift is estimated by least squares from qP/qSV/qSH together, then the
    same t0 is applied to all three phases for that source.  Differential plots
    are unchanged by t0 and therefore remain the direct station-pair quantities.
    """
    if plt is None:
        print("[WARN] matplotlib is unavailable; skip arrival comparison plots")
        return

    plot_dir = outdir / "arrival_plots"
    abs_dir = plot_dir / "absolute_by_source"
    abs_resid_dir = plot_dir / "absolute_residual_by_source"
    diff_dir = plot_dir / "diff_by_source"
    diff_resid_dir = plot_dir / "diff_residual_by_source"
    for d in (plot_dir, abs_dir, abs_resid_dir, diff_dir, diff_resid_dir):
        d.mkdir(parents=True, exist_ok=True)

    receiver_index = np.arange(1, int(geo.nr) + 1, dtype=float)
    receiver_ylabel = "Receiver"
    plot_waves = _available_plot_waves()
    phase_tag = "_".join(_wave_label(w) for w in plot_waves)
    reference_name = "true/noiseless" if true_pred is not None else "noisy obs"

    # 1) Estimate one origin time per source and model using all three phases.
    origin_times: dict[str, list[float]] = {label: [] for label in model_preds}
    origin_shift_rows: list[dict[str, Any]] = []
    for label, pred in model_preds.items():
        for isrc in range(geo.ns):
            t0, rmse_after_shift, n_values = estimate_source_origin_time_multiwave(
                reference_pred=true_pred,
                obs=obs,
                predicted=pred,
                waves=plot_waves,
                isrc=isrc,
            )
            origin_times[label].append(t0)
            origin_shift_rows.append({
                "source": int(isrc),
                "source_1based": int(isrc + 1),
                "model": label,
                "reference": reference_name,
                "waves": ",".join(_wave_label(w) for w in plot_waves),
                "estimated_origin_time_s": t0,
                "estimated_origin_time_ms": 1000.0 * t0 if np.isfinite(t0) else float("nan"),
                "rmse_after_origin_shift_s": rmse_after_shift,
                "rmse_after_origin_shift_ms": 1000.0 * rmse_after_shift if np.isfinite(rmse_after_shift) else float("nan"),
                "n_values": int(n_values),
            })

    if origin_shift_rows:
        write_csv(
            plot_dir / "absolute_estimated_origin_times_by_source.csv",
            origin_shift_rows,
            [
                "source", "source_1based", "model", "reference", "waves",
                "estimated_origin_time_s", "estimated_origin_time_ms",
                "rmse_after_origin_shift_s", "rmse_after_origin_shift_ms", "n_values",
            ],
        )

    # 2) Absolute arrival-time plots.  Each source now has a 1 x 3 qP/qSV/qSH panel.
    wave_abs_panels: dict[str, list[list[tuple[str, np.ndarray]]]] = {w: [] for w in plot_waves}
    wave_abs_resid_panels: dict[str, list[list[tuple[str, np.ndarray]]]] = {w: [] for w in plot_waves}

    for isrc in range(geo.ns):
        source_abs_phase_series: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        source_abs_resid_phase_series: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

        for wave in plot_waves:
            wlabel = _wave_label(wave)
            obs_arr = _wave_obs(obs, wave)
            true_arr = _wave_pred(true_pred, wave) if true_pred is not None else None

            abs_series_source: list[tuple[str, np.ndarray]] = []
            abs_resid_series_source: list[tuple[str, np.ndarray]] = []
            abs_series_panel: list[tuple[str, np.ndarray]] = []
            abs_resid_series_panel: list[tuple[str, np.ndarray]] = []

            if true_arr is not None:
                abs_series_source.append(("true/noiseless", true_arr[isrc, :]))
                abs_resid_series_source.append(("true-noisy", true_arr[isrc, :] - obs_arr[isrc, :]))
                abs_series_panel.append(("true/noiseless", true_arr[isrc, :]))
                abs_resid_series_panel.append(("true-noisy", true_arr[isrc, :] - obs_arr[isrc, :]))
            abs_series_source.append(("noisy obs", obs_arr[isrc, :]))
            abs_series_panel.append(("noisy obs", obs_arr[isrc, :]))

            for label in ("mean", "best"):
                if label not in model_preds:
                    continue
                pred_arr = _wave_pred(model_preds[label], wave)
                t0 = origin_times.get(label, [float("nan")] * geo.ns)[isrc]
                plot_arr = pred_arr[isrc, :] + t0 if np.isfinite(t0) else pred_arr[isrc, :]
                abs_series_source.append((label_with_t0(label, t0), plot_arr))
                abs_resid_series_source.append((label_with_t0(label, t0), plot_arr - obs_arr[isrc, :]))
                # For all-source summary panels, keep labels compact; source-specific
                # numeric t0 values are shown in the 1 x 3 source figures and CSV.
                abs_series_panel.append((f"{label}+t0", plot_arr))
                abs_resid_series_panel.append((f"{label}+t0", plot_arr - obs_arr[isrc, :]))

            source_abs_phase_series.append((wlabel, abs_series_source))
            source_abs_resid_phase_series.append((wlabel, abs_resid_series_source))
            wave_abs_panels[wave].append(abs_series_panel)
            wave_abs_resid_panels[wave].append(abs_resid_series_panel)

        multiphase_source_plot(
            abs_dir / f"source_{isrc:03d}_absolute_{phase_tag}_true_noise_mean_best_1x3.png",
            receiver_index,
            source_abs_phase_series,
            f"Source {isrc + 1}: absolute arrivals with one three-phase origin time",
            receiver_ylabel,
            "Time(s)",
        )
        multiphase_source_plot(
            abs_resid_dir / f"source_{isrc:03d}_absolute_residual_{phase_tag}_mean_best_minus_obs_1x3.png",
            receiver_index,
            source_abs_resid_phase_series,
            f"Source {isrc + 1}: absolute residuals after one three-phase origin-time shift",
            receiver_ylabel,
            "Predicted - observed time(s)",
        )

    for wave in plot_waves:
        wlabel = _wave_label(wave)
        panel_arrival_plot(
            plot_dir / f"absolute_{wlabel}_all_sources_true_noise_mean_best.png",
            receiver_index,
            wave_abs_panels[wave],
            f"Absolute {wlabel} arrival-time comparison: true/noisy/mean/best+t0",
            receiver_ylabel,
            "Time(s)",
            ncols=4,
        )
        panel_arrival_plot(
            plot_dir / f"absolute_residual_{wlabel}_all_sources_mean_best_minus_obs.png",
            receiver_index,
            wave_abs_resid_panels[wave],
            f"Absolute {wlabel} residuals after origin-time shift: predicted - observed",
            receiver_ylabel,
            "Predicted - observed time(s)",
            ncols=4,
        )

    # 3) Station-pair differential plots.  t0 cancels out, so no shift is applied.
    if like.diff_matrix is None:
        return

    pair_index = np.arange(1, int(like.diff_matrix.shape[0]) + 1, dtype=float)
    if mcmc is not None and hasattr(mcmc, "apply_difference"):
        diff_func = lambda arr: mcmc.apply_difference(arr, like.diff_matrix)
    else:
        diff_func = lambda arr: np.asarray([like.diff_matrix @ arr[isrc, :] for isrc in range(arr.shape[0])])

    objective_waves = tuple(getattr(like, "waves", ("P",))) or ("P",)
    obs_diff = _obs_diff_3d(like)

    for wave in plot_waves:
        wlabel = _wave_label(wave)
        true_arr = _wave_pred(true_pred, wave) if true_pred is not None else None
        true_diff_all = diff_func(true_arr) if true_arr is not None else None
        # Use the likelihood data vector for objective phases; otherwise compute
        # diagnostic qSV/qSH station-pair differences from the observed absolute arrivals.
        if obs_diff is not None and wave in objective_waves:
            iw_obj = objective_waves.index(wave)
            obs_diff_w = obs_diff[iw_obj] if iw_obj < obs_diff.shape[0] else diff_func(_wave_obs(obs, wave))
        else:
            obs_diff_w = diff_func(_wave_obs(obs, wave))
        model_diff_all = {label: diff_func(_wave_pred(pred, wave)) for label, pred in model_preds.items()}

        source_diff_panels: list[list[tuple[str, np.ndarray]]] = []
        source_diff_resid_panels: list[list[tuple[str, np.ndarray]]] = []
        source_diff_phase_series_by_source: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        source_diff_resid_phase_series_by_source: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

        for isrc in range(geo.ns):
            diff_series: list[tuple[str, np.ndarray]] = []
            diff_resid_series: list[tuple[str, np.ndarray]] = []
            if true_diff_all is not None:
                diff_series.append(("true/noiseless", true_diff_all[isrc, :]))
                diff_resid_series.append(("true-noisy", true_diff_all[isrc, :] - obs_diff_w[isrc, :]))
            diff_series.append(("noisy obs", obs_diff_w[isrc, :]))
            for label in ("mean", "best"):
                if label in model_diff_all:
                    diff_series.append((label, model_diff_all[label][isrc, :]))
                    diff_resid_series.append((label, model_diff_all[label][isrc, :] - obs_diff_w[isrc, :]))

            source_diff_panels.append(diff_series)
            source_diff_resid_panels.append(diff_resid_series)

        panel_arrival_plot(
            plot_dir / f"diff_{wlabel}_all_sources_true_noise_mean_best.png",
            pair_index,
            source_diff_panels,
            f"Station-pair differential {wlabel} comparison: true/noisy/mean/best",
            "Station-pair",
            "Differential time(s)",
            ncols=4,
        )
        panel_arrival_plot(
            plot_dir / f"diff_residual_{wlabel}_all_sources_mean_best_minus_obs.png",
            pair_index,
            source_diff_resid_panels,
            f"Station-pair differential {wlabel} residuals: predicted - observed",
            "Station-pair",
            "Predicted - observed differential time(s)",
            ncols=4,
        )

    # 4) Per-source 1 x 3 differential figures.
    all_true_diff = {w: diff_func(_wave_pred(true_pred, w)) for w in plot_waves} if true_pred is not None else {}
    all_obs_diff: dict[str, np.ndarray] = {}
    for wave in plot_waves:
        if obs_diff is not None and wave in objective_waves:
            iw_obj = objective_waves.index(wave)
            all_obs_diff[wave] = obs_diff[iw_obj] if iw_obj < obs_diff.shape[0] else diff_func(_wave_obs(obs, wave))
        else:
            all_obs_diff[wave] = diff_func(_wave_obs(obs, wave))
    all_model_diff = {
        label: {wave: diff_func(_wave_pred(pred, wave)) for wave in plot_waves}
        for label, pred in model_preds.items()
    }

    for isrc in range(geo.ns):
        source_diff_phase_series: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        source_diff_resid_phase_series: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        for wave in plot_waves:
            wlabel = _wave_label(wave)
            obs_diff_w = all_obs_diff[wave]
            diff_series: list[tuple[str, np.ndarray]] = []
            diff_resid_series: list[tuple[str, np.ndarray]] = []
            if wave in all_true_diff:
                diff_series.append(("true/noiseless", all_true_diff[wave][isrc, :]))
                diff_resid_series.append(("true-noisy", all_true_diff[wave][isrc, :] - obs_diff_w[isrc, :]))
            diff_series.append(("noisy obs", obs_diff_w[isrc, :]))
            for label in ("mean", "best"):
                if label in all_model_diff:
                    diff_series.append((label, all_model_diff[label][wave][isrc, :]))
                    diff_resid_series.append((label, all_model_diff[label][wave][isrc, :] - obs_diff_w[isrc, :]))
            source_diff_phase_series.append((wlabel, diff_series))
            source_diff_resid_phase_series.append((wlabel, diff_resid_series))

        multiphase_source_plot(
            diff_dir / f"source_{isrc:03d}_diff_{phase_tag}_true_noise_mean_best_1x3.png",
            pair_index,
            source_diff_phase_series,
            f"Source {isrc + 1}: station-pair differential arrivals",
            "Station-pair",
            "Differential time(s)",
        )
        multiphase_source_plot(
            diff_resid_dir / f"source_{isrc:03d}_diff_residual_{phase_tag}_mean_best_minus_obs_1x3.png",
            pair_index,
            source_diff_resid_phase_series,
            f"Source {isrc + 1}: station-pair differential residuals",
            "Station-pair",
            "Predicted - observed differential time(s)",
        )

def compare_one_model(label: str, theta: np.ndarray, args, root: Path, mcmc, geo0, model0, obs, like, outdir: Path,
                      qx_stop: float, qx_max_iter: int, invert_depths: bool, fix_last_layer: bool,
                      invert_sources: bool, forward_waves: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Forward one parameter vector and compare predicted/observed data."""
    geo, model = mcmc.unpack_params(theta, geo0, model0, invert_depths, fix_last_layer, invert_sources)
    pred = mcmc.forward_direct(geo, model, stop=qx_stop, max_iter=qx_max_iter, needed_waves=forward_waves)
    tp, tsv, tsh = pred

    # MCMC objective misfit, using the same transformed data and covariance.
    objective_misfit = float(mcmc.misfit_from_times(pred, obs, like))

    waves_all = ("P", "SV", "SH")
    abs_res: dict[str, np.ndarray] = {w: _wave_pred(pred, w) - _wave_obs(obs, w) for w in waves_all}
    abs_metrics = {w: finite_metrics(abs_res[w]) for w in waves_all}

    diff_pred: dict[str, np.ndarray] = {}
    diff_res: dict[str, np.ndarray] = {}
    diff_metrics: dict[str, dict[str, float]] = {}
    diff_qform_by_source = np.full(geo.ns, np.nan)
    obs_diff = _obs_diff_3d(like)
    if like.objective_type in {"diff-p-adjacent", "diff-p-reference"} and like.diff_matrix is not None and obs_diff is not None:
        qform_sum = np.zeros(geo.ns, dtype=float)
        for iw, wave in enumerate(tuple(getattr(like, "waves", ("P",))) or ("P",)):
            if iw >= obs_diff.shape[0]:
                continue
            pd = mcmc.apply_difference(_wave_pred(pred, wave), like.diff_matrix)
            rd = pd - obs_diff[iw]
            diff_pred[wave] = pd
            diff_res[wave] = rd
            diff_metrics[wave] = finite_metrics(rd)
            if like.cov_inv is not None:
                qform_sum += np.einsum("si,ij,sj->s", rd, like.cov_inv, rd)
        diff_qform_by_source = qform_sum
    for w in waves_all:
        diff_metrics.setdefault(w, {"n": 0.0, "mean": math.nan, "rmse": math.nan, "mae": math.nan, "max_abs": math.nan})

    summary = {
        "model": label,
        "objective_type": like.objective_type,
        "sigma_mode": like.sigma_mode,
        "sigma_abs": float(obs.sigma),
        "objective_misfit": objective_misfit,
        "absolute_qP_n": int(abs_metrics["P"]["n"]),
        "absolute_qP_mean_residual": abs_metrics["P"]["mean"],
        "absolute_qP_rmse": abs_metrics["P"]["rmse"],
        "absolute_qP_mae": abs_metrics["P"]["mae"],
        "absolute_qP_max_abs": abs_metrics["P"]["max_abs"],
        "absolute_qSV_n": int(abs_metrics["SV"]["n"]),
        "absolute_qSV_mean_residual": abs_metrics["SV"]["mean"],
        "absolute_qSV_rmse": abs_metrics["SV"]["rmse"],
        "absolute_qSV_mae": abs_metrics["SV"]["mae"],
        "absolute_qSV_max_abs": abs_metrics["SV"]["max_abs"],
        "absolute_qSH_n": int(abs_metrics["SH"]["n"]),
        "absolute_qSH_mean_residual": abs_metrics["SH"]["mean"],
        "absolute_qSH_rmse": abs_metrics["SH"]["rmse"],
        "absolute_qSH_mae": abs_metrics["SH"]["mae"],
        "absolute_qSH_max_abs": abs_metrics["SH"]["max_abs"],
        "diff_qP_n": int(diff_metrics["P"]["n"]),
        "diff_qP_mean_residual": diff_metrics["P"]["mean"],
        "diff_qP_rmse": diff_metrics["P"]["rmse"],
        "diff_qP_mae": diff_metrics["P"]["mae"],
        "diff_qP_max_abs": diff_metrics["P"]["max_abs"],
        "diff_qSV_n": int(diff_metrics["SV"]["n"]),
        "diff_qSV_mean_residual": diff_metrics["SV"]["mean"],
        "diff_qSV_rmse": diff_metrics["SV"]["rmse"],
        "diff_qSV_mae": diff_metrics["SV"]["mae"],
        "diff_qSV_max_abs": diff_metrics["SV"]["max_abs"],
        "diff_qSH_n": int(diff_metrics["SH"]["n"]),
        "diff_qSH_mean_residual": diff_metrics["SH"]["mean"],
        "diff_qSH_rmse": diff_metrics["SH"]["rmse"],
        "diff_qSH_mae": diff_metrics["SH"]["mae"],
        "diff_qSH_max_abs": diff_metrics["SH"]["max_abs"],
        "ns": geo.ns,
        "nr": geo.nr,
    }

    by_source_rows: list[dict[str, Any]] = []
    abs_rmse_per_source: dict[str, list[float]] = {w: [] for w in waves_all}
    diff_rmse_per_source: dict[str, list[float]] = {w: [] for w in waves_all}
    for isrc in range(geo.ns):
        am = {w: finite_metrics(abs_res[w][isrc, :]) for w in waves_all}
        dm: dict[str, dict[str, float]] = {}
        for w in waves_all:
            if w in diff_res:
                dm[w] = finite_metrics(diff_res[w][isrc, :])
            else:
                dm[w] = {"rmse": math.nan, "mae": math.nan, "max_abs": math.nan}
            abs_rmse_per_source[w].append(am[w]["rmse"])
            diff_rmse_per_source[w].append(dm[w]["rmse"])

        by_source_rows.append({
            "model": label,
            "source": isrc,
            "sx": float(geo.sx[isrc]),
            "sz": float(geo.sz[isrc]),
            "absolute_qP_rmse": am["P"]["rmse"],
            "absolute_qP_mae": am["P"]["mae"],
            "absolute_qP_max_abs": am["P"]["max_abs"],
            "absolute_qSV_rmse": am["SV"]["rmse"],
            "absolute_qSV_mae": am["SV"]["mae"],
            "absolute_qSV_max_abs": am["SV"]["max_abs"],
            "absolute_qSH_rmse": am["SH"]["rmse"],
            "absolute_qSH_mae": am["SH"]["mae"],
            "absolute_qSH_max_abs": am["SH"]["max_abs"],
            "diff_qP_rmse": dm["P"]["rmse"],
            "diff_qP_mae": dm["P"]["mae"],
            "diff_qP_max_abs": dm["P"]["max_abs"],
            "diff_qSV_rmse": dm["SV"]["rmse"],
            "diff_qSV_mae": dm["SV"]["mae"],
            "diff_qSV_max_abs": dm["SV"]["max_abs"],
            "diff_qSH_rmse": dm["SH"]["rmse"],
            "diff_qSH_mae": dm["SH"]["mae"],
            "diff_qSH_max_abs": dm["SH"]["max_abs"],
            "diff_objective_qform": float(diff_qform_by_source[isrc]) if isrc < len(diff_qform_by_source) else math.nan,
        })

    write_predicted_ttime(outdir / f"{label}_predicted_ttime.dat", geo, pred)
    mcmc.write_model_summary(outdir / f"{label}_model_summary.dat", model, geo, objective_misfit)

    savez_kwargs = dict(
        sx=geo.sx, sz=geo.sz, rx=geo.rx, rz=geo.rz,
        pred_tp=tp, pred_tsv=tsv, pred_tsh=tsh,
        obs_tp=obs.tp, obs_tsv=obs.tsv, obs_tsh=obs.tsh,
        abs_p_residual=abs_res["P"],
        abs_sv_residual=abs_res["SV"],
        abs_sh_residual=abs_res["SH"],
        objective_misfit=objective_misfit,
    )
    for w, suffix in [("P", "qp"), ("SV", "qsv"), ("SH", "qsh")]:
        savez_kwargs[f"pred_diff_{suffix}"] = diff_pred.get(w, np.asarray([]))
        if obs_diff is not None and w in tuple(getattr(like, "waves", ())):
            iw = tuple(getattr(like, "waves", ())).index(w)
            savez_kwargs[f"obs_diff_{suffix}"] = obs_diff[iw]
        else:
            savez_kwargs[f"obs_diff_{suffix}"] = np.asarray([])
        savez_kwargs[f"diff_{suffix}_residual"] = diff_res.get(w, np.asarray([]))
    np.savez(outdir / f"{label}_forward_comparison.npz", **savez_kwargs)

    # Absolute observed-vs-predicted scatter plots.
    for w, obs_arr, pred_arr in [("P", obs.tp, tp), ("SV", obs.tsv, tsv), ("SH", obs.tsh, tsh)]:
        wlabel = _wave_label(w)
        scatter_plot(
            outdir / f"{label}_absolute_{wlabel}_obs_vs_pred.png",
            obs_arr, pred_arr,
            f"{label}: absolute {wlabel} observed vs predicted",
            f"Observed {wlabel} time", f"Predicted {wlabel} time",
        )

    # Differential observed-vs-predicted scatter plots for selected objective phases.
    obs_diff = _obs_diff_3d(like)
    if obs_diff is not None:
        for iw, w in enumerate(tuple(getattr(like, "waves", ("P",))) or ("P",)):
            if w not in diff_pred or iw >= obs_diff.shape[0]:
                continue
            wlabel = _wave_label(w)
            scatter_plot(
                outdir / f"{label}_diff_{wlabel}_obs_vs_pred.png",
                obs_diff[iw], diff_pred[w],
                f"{label}: station-pair differential {wlabel} observed vs predicted",
                f"Observed differential {wlabel}", f"Predicted differential {wlabel}",
            )

    source_ids = np.arange(geo.ns)
    for w in waves_all:
        wlabel = _wave_label(w)
        source_metric_plot(
            outdir / f"{label}_source_abs_{wlabel}_rmse.png",
            source_ids, np.asarray(abs_rmse_per_source[w], dtype=float),
            f"{label}: absolute {wlabel} RMSE by source", "RMSE",
        )
        if w in diff_pred:
            source_metric_plot(
                outdir / f"{label}_source_diff_{wlabel}_rmse.png",
                source_ids, np.asarray(diff_rmse_per_source[w], dtype=float),
                f"{label}: differential {wlabel} RMSE by source", "RMSE",
            )

    print(
        f"[{label.upper()}] objective_misfit={objective_misfit:.6g}, "
        f"diff_qP_RMSE={summary['diff_qP_rmse']:.6g}, "
        f"diff_qSV_RMSE={summary['diff_qSV_rmse']:.6g}, "
        f"diff_qSH_RMSE={summary['diff_qSH_rmse']:.6g}, "
        f"abs_qP_RMSE={summary['absolute_qP_rmse']:.6g}"
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
    parser.add_argument("--forward-waves", default="from-chain", help="Waves to forward/use in comparison: P, P,SV,SH, etc.; default reads use_waves from chain.npz when available.")
    parser.add_argument("--invert-depths", dest="invert_depths", action="store_true", default=True)
    parser.add_argument("--no-invert-depths", dest="invert_depths", action="store_false")
    parser.add_argument("--fix-last-layer", action="store_true", default=False)
    parser.add_argument("--invert-sources", dest="invert_sources", action="store_true", default=None)
    parser.add_argument("--no-invert-sources", dest="invert_sources", action="store_false")
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
    forward_waves_text = scalar_string(pack, "use_waves", "P") if args.forward_waves == "from-chain" else args.forward_waves
    like = mcmc.build_likelihood_config(obs, objective_type=objective, use_waves=forward_waves_text, sigma_mode=sigma_mode)

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

    forward_waves = selected_waves_from_text(forward_waves_text)
    # Make sure all waves required by the likelihood and by the qP/qSV/qSH
    # diagnostic plots are forwarded.  This keeps the objective unchanged when
    # the chain uses qP only, but still lets the comparison script draw all
    # three phase arrivals and estimate the three-phase source origin time.
    for wave in tuple(getattr(like, "waves", ())) + _available_plot_waves():
        if wave not in forward_waves:
            forward_waves = forward_waves + (wave,)

    if "mean_theta" not in pack:
        raise KeyError("chain.npz does not contain mean_theta")
    mean_theta = np.asarray(pack["mean_theta"], dtype=float)

    if "best_theta" in pack:
        best_theta = np.asarray(pack["best_theta"], dtype=float)
    else:
        chain = np.asarray(pack["chain"], dtype=float)
        misfit = np.asarray(pack["misfit"], dtype=float)
        best_theta = chain[int(np.argmin(misfit))]

    if args.invert_sources is None:
        invert_sources = bool(np.asarray(pack.get("invert_sources", np.asarray(True))).item())
    else:
        invert_sources = bool(args.invert_sources)

    print(f"[CONFIG] objective={objective}, sigma_mode={sigma_mode}, sigma={obs.sigma:.6g}, invert_sources={invert_sources}")
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
            invert_sources=invert_sources,
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
        "absolute_qSV_n", "absolute_qSV_mean_residual", "absolute_qSV_rmse", "absolute_qSV_mae", "absolute_qSV_max_abs",
        "absolute_qSH_n", "absolute_qSH_mean_residual", "absolute_qSH_rmse", "absolute_qSH_mae", "absolute_qSH_max_abs",
        "diff_qP_n", "diff_qP_mean_residual", "diff_qP_rmse", "diff_qP_mae", "diff_qP_max_abs",
        "diff_qSV_n", "diff_qSV_mean_residual", "diff_qSV_rmse", "diff_qSV_mae", "diff_qSV_max_abs",
        "diff_qSH_n", "diff_qSH_mean_residual", "diff_qSH_rmse", "diff_qSH_mae", "diff_qSH_max_abs",
        "ns", "nr",
    ]
    source_fields = [
        "model", "source", "sx", "sz",
        "absolute_qP_rmse", "absolute_qP_mae", "absolute_qP_max_abs",
        "absolute_qSV_rmse", "absolute_qSV_mae", "absolute_qSV_max_abs",
        "absolute_qSH_rmse", "absolute_qSH_mae", "absolute_qSH_max_abs",
        "diff_qP_rmse", "diff_qP_mae", "diff_qP_max_abs",
        "diff_qSV_rmse", "diff_qSV_mae", "diff_qSV_max_abs",
        "diff_qSH_rmse", "diff_qSH_mae", "diff_qSH_max_abs",
        "diff_objective_qform",
    ]
    write_csv(outdir / "forward_compare_summary.csv", summary_rows, summary_fields)
    write_csv(outdir / "forward_compare_by_source.csv", source_rows, source_fields)

    print(f"[DONE] comparison files saved to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
