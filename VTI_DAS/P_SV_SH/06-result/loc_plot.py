#!/usr/bin/env python3
"""Plot event locations and posterior distributions from chain.npz.

Outputs
-------
location_figures/
    all_events.png
        True / Initial / Mean / Best event locations.
    event_###.png
        One True / Initial / Mean / Best figure per event.
    posterior_event_###.png
        Posterior sx-sz cloud for each event, plus true/initial/mean/best markers.
    posterior_event_###_marginals.png
        Marginal posterior histograms of sx and sz, plus True/Initial/Best/Mean lines.
    location_errors.csv
        True / Initial / Mean / Best location errors.
    location_posterior_summary.csv
        Posterior mean, std, quantiles, and distance-error quantiles for every event.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from vti_plot_utils import read_geometry_dat, read_summary_dat


# =============================================================================
# Plot colors: keep these consistent in all figures.
# =============================================================================
C_POST = 'tab:blue'       # posterior samples / histogram
C_TRUE = 'black'          # true location; black avoids confusion with posterior blue
C_INIT = 'tab:orange'     # initial location
C_BEST = 'tab:green'      # best location
C_MEAN = 'tab:red'        # posterior mean location
C_ELLIPSE = 'black'


# =============================================================================
# Path utilities
# =============================================================================
def infer_project_root(user_root: Path | None = None) -> Path:
    """Infer project root robustly.

    The project root is the folder containing:
      01-input/
      04-initial/
      06-result/

    This lets the script run from either the project root or from inside 06-result.
    """
    if user_root is not None:
        return Path(user_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve().parent

    candidates = []
    for base in (cwd, here):
        candidates.extend([base, base.parent, base.parent.parent])

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for c in unique_candidates:
        if (c / '01-input').is_dir() and (c / '04-initial').is_dir() and (c / '06-result').is_dir():
            return c

    if cwd.name == '06-result':
        return cwd.parent
    if here.name == '06-result':
        return here.parent
    return cwd


def first_existing_path(candidates: list[Path], label: str) -> Path:
    """Return the first existing path; otherwise return the first candidate and warn."""
    for path in candidates:
        if path.exists():
            return path
    warnings.warn(
        f'No existing {label} file found. Tried:\n  ' +
        '\n  '.join(str(p) for p in candidates) +
        f'\nUsing first candidate anyway: {candidates[0]}'
    )
    return candidates[0]


# =============================================================================
# Chain loading and summaries
# =============================================================================
def _clean_param_names(raw_names: np.ndarray) -> list[str]:
    names = []
    for x in raw_names:
        if isinstance(x, bytes):
            names.append(x.decode('utf-8', errors='ignore'))
        else:
            names.append(str(x))
    return names


def load_chain(result_dir: Path, burnin: int | None) -> tuple[np.ndarray | None, list[str], int]:
    path = result_dir / 'chain.npz'
    if not path.exists():
        warnings.warn(f'chain.npz not found: {path}. Posterior plots will be skipped.')
        return None, [], 0

    data = np.load(path, allow_pickle=True)
    if 'chain' not in data or 'param_names' not in data:
        warnings.warn(f'{path} does not contain both chain and param_names. Posterior plots will be skipped.')
        return None, [], 0

    chain = np.asarray(data['chain'], dtype=float)
    names = _clean_param_names(data['param_names'])
    if chain.ndim != 2 or chain.shape[1] != len(names):
        warnings.warn('chain shape and param_names length are inconsistent. Posterior plots will be skipped.')
        return None, [], 0

    if burnin is None and 'burnin_eff' in data:
        used_burnin = int(np.asarray(data['burnin_eff']).item())
    elif burnin is None and 'burnin' in data:
        used_burnin = int(np.asarray(data['burnin']).item())
    else:
        used_burnin = int(max(0, burnin))

    if used_burnin >= chain.shape[0]:
        warnings.warn(f'burnin={used_burnin} >= chain length={chain.shape[0]}; using the full chain instead.')
        used_burnin = 0

    return chain[used_burnin:], names, used_burnin


def column_by_name(names: list[str], target: str) -> int | None:
    try:
        return names.index(target)
    except ValueError:
        return None


def quantile_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f'{prefix}_mean': np.nan,
            f'{prefix}_std': np.nan,
            f'{prefix}_q05': np.nan,
            f'{prefix}_q16': np.nan,
            f'{prefix}_q50': np.nan,
            f'{prefix}_q84': np.nan,
            f'{prefix}_q95': np.nan,
        }

    q05, q16, q50, q84, q95 = np.quantile(values, [0.05, 0.16, 0.50, 0.84, 0.95])
    return {
        f'{prefix}_mean': float(np.mean(values)),
        f'{prefix}_std': float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        f'{prefix}_q05': float(q05),
        f'{prefix}_q16': float(q16),
        f'{prefix}_q50': float(q50),
        f'{prefix}_q84': float(q84),
        f'{prefix}_q95': float(q95),
    }


def get_event_samples(samples: np.ndarray, names: list[str], event: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    jx = column_by_name(names, f'sx[{event}]')
    jz = column_by_name(names, f'sz[{event}]')
    sx = samples[:, jx] if jx is not None else None
    sz = samples[:, jz] if jz is not None else None
    return sx, sz


# =============================================================================
# Plot helpers
# =============================================================================
def add_cov_ellipse(ax, sx: np.ndarray, sz: np.ndarray, nstd: float = 2.0) -> None:
    """Add a covariance ellipse. nstd=2 is a visual uncertainty ellipse, not an exact 95% contour."""
    good = np.isfinite(sx) & np.isfinite(sz)
    sx = sx[good]
    sz = sz[good]
    if sx.size < 3:
        return

    cov = np.cov(np.vstack([sx, sz]))
    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * nstd * np.sqrt(vals)
    ell = Ellipse(
        (np.mean(sx), np.mean(sz)),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=C_ELLIPSE,
        linewidth=1.5,
        linestyle='--',
        label=f'{nstd:g}σ ellipse',
    )
    ax.add_patch(ell)


def save_posterior_summary(samples: np.ndarray | None, names: list[str], true_geo, outdir: Path, n: int) -> None:
    if samples is None:
        return

    rows = []
    for i in range(n):
        sx, sz = get_event_samples(samples, names, i)
        if sx is None and sz is None:
            continue

        row = {'event': i}
        if sx is not None:
            row.update(quantile_summary(sx, 'sx'))
        if sz is not None:
            row.update(quantile_summary(sz, 'sz'))
        if sx is not None and sz is not None:
            err = np.hypot(sx - true_geo.sx[i], sz - true_geo.sz[i])
            row.update(quantile_summary(err, 'distance_error_m'))
            row['sx_sz_corr'] = float(np.corrcoef(sx, sz)[0, 1]) if sx.size > 1 else np.nan
        rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(outdir / 'location_posterior_summary.csv', index=False)


def add_marginal_lines(ax, q16: float, q84: float,
                       true_value: float, init_value: float, best_value: float, mean_value: float) -> None:
    """Add 16/84% and True/Initial/Best/Mean vertical lines to a marginal posterior axis."""
    ax.axvline(q16, color=C_POST, linestyle='--', linewidth=1.2, label='16/84%')
    ax.axvline(q84, color=C_POST, linestyle='--', linewidth=1.2)

    ax.axvline(true_value, color=C_TRUE, linestyle='-', linewidth=2.0, label='True')
    ax.axvline(init_value, color=C_INIT, linestyle='-', linewidth=2.0, label='Initial')
    ax.axvline(best_value, color=C_BEST, linestyle='-', linewidth=2.0, label='Best')
    ax.axvline(mean_value, color=C_MEAN, linestyle='-', linewidth=2.0, label='Mean')




def location_error_m(x: float, z: float, true_x: float, true_z: float) -> float:
    """Euclidean location error relative to the true event position."""
    return float(np.hypot(x - true_x, z - true_z))


def format_error_label(name: str, error_m: float) -> str:
    """Legend label with location error in meters."""
    if np.isfinite(error_m):
        return f'{name} (Δ={error_m:.1f} m)'
    return f'{name} (Δ=nan m)'


def add_error_text_box(ax, best_err: float, mean_err: float, loc: str = 'upper left') -> None:
    """Add a compact text box summarizing Best/Mean errors relative to True."""
    text = f'Best–True Δ = {best_err:.1f} m\nMean–True Δ = {mean_err:.1f} m'
    anchor = {
        'upper left': (0.02, 0.98, 'left', 'top'),
        'upper right': (0.98, 0.98, 'right', 'top'),
        'lower left': (0.02, 0.02, 'left', 'bottom'),
        'lower right': (0.98, 0.02, 'right', 'bottom'),
    }.get(loc, (0.02, 0.98, 'left', 'top'))
    x, y, ha, va = anchor
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='0.7'),
    )


def plot_event_posterior(samples: np.ndarray, names: list[str], i: int, true_geo, init_geo, best_geo, mean_geo,
                         outdir: Path, max_scatter: int, xlim, ylim) -> None:
    sx, sz = get_event_samples(samples, names, i)
    if sx is None or sz is None:
        return

    good = np.isfinite(sx) & np.isfinite(sz)
    sx = sx[good]
    sz = sz[good]
    if sx.size == 0:
        return

    best_err = location_error_m(best_geo.sx[i], best_geo.sz[i], true_geo.sx[i], true_geo.sz[i])
    mean_err = location_error_m(mean_geo.sx[i], mean_geo.sz[i], true_geo.sx[i], true_geo.sz[i])

    if sx.size > max_scatter:
        idx = np.linspace(0, sx.size - 1, max_scatter).astype(int)
        px = sx[idx]
        pz = sz[idx]
    else:
        px, pz = sx, sz

    # -------------------------------------------------------------------------
    # 2-D posterior source-location cloud
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(px, pz, s=8, alpha=0.25, color=C_POST, label='Posterior samples')
    add_cov_ellipse(ax, sx, sz, nstd=2.0)
    ax.plot(true_geo.sx[i], true_geo.sz[i], 'o', color=C_TRUE, markersize=9, label='True')
    ax.plot(init_geo.sx[i], init_geo.sz[i], '*', color=C_INIT, markersize=10, label='Initial')
    ax.plot(best_geo.sx[i], best_geo.sz[i], 's', color=C_BEST, markersize=8, label=format_error_label('Best', best_err))
    ax.plot(mean_geo.sx[i], mean_geo.sz[i], 'v', color=C_MEAN, markersize=8, label=format_error_label('Mean', mean_err))
    add_error_text_box(ax, best_err, mean_err, loc='lower right')
    ax.set_xlabel('Distance sx (m)')
    ax.set_ylabel('Depth sz (m)')
    ax.set_title(f'Posterior event location: event {i}')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.25)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(outdir / f'posterior_event_{i:03d}.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Marginal posterior histograms: no Median line; add True/Initial/Best/Mean.
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # sx marginal
    axes[0].hist(sx, bins=40, density=True, alpha=0.75, color=C_POST)
    qx16, qx84 = np.quantile(sx, [0.16, 0.84])
    add_marginal_lines(
        axes[0],
        qx16,
        qx84,
        true_geo.sx[i],
        init_geo.sx[i],
        best_geo.sx[i],
        mean_geo.sx[i],
    )
    axes[0].set_xlabel('sx (m)')
    axes[0].set_ylabel('Posterior density')
    axes[0].set_title(f'sx[{i}]')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    # sz marginal
    axes[1].hist(sz, bins=40, density=True, alpha=0.75, color=C_POST)
    qz16, qz84 = np.quantile(sz, [0.16, 0.84])
    add_marginal_lines(
        axes[1],
        qz16,
        qz84,
        true_geo.sz[i],
        init_geo.sz[i],
        best_geo.sz[i],
        mean_geo.sz[i],
    )
    axes[1].set_xlabel('sz (m)')
    axes[1].set_ylabel('Posterior density')
    axes[1].set_title(f'sz[{i}]')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(f'Marginal posterior distributions: event {i}', y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f'posterior_event_{i:03d}_marginals.png', dpi=300)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description='Plot source locations and posterior distributions')
    p.add_argument('--root', type=Path, default=None, help='Project root containing 01-input, 04-initial, 06-result')
    p.add_argument('--true-geo', type=Path, default=None)
    p.add_argument('--init-geo', type=Path, default=None)
    p.add_argument('--result-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--xlim', nargs=2, type=float, default=None)
    p.add_argument('--ylim', nargs=2, type=float, default=None)
    p.add_argument('--label-events', action='store_true')
    p.add_argument('--burnin', type=int, default=None, help='Discard this many MCMC samples; default reads burnin_eff from chain.npz')
    p.add_argument('--max-scatter', type=int, default=3000, help='Maximum posterior samples shown in each sx-sz scatter plot')
    args = p.parse_args(argv)

    root = infer_project_root(args.root)
    result_dir = args.result_dir.expanduser().resolve() if args.result_dir is not None else root / '06-result/output'
    outdir = args.output_dir.expanduser().resolve() if args.output_dir is not None else result_dir / 'location_figures'
    outdir.mkdir(parents=True, exist_ok=True)

    # Robust default paths. This also avoids failure when the true geometry is named geo.dat.
    if args.true_geo is not None:
        true_geo_path = args.true_geo.expanduser().resolve()
    else:
        true_geo_path = first_existing_path(
            [
                root / '01-input/output/geometry.dat',
                root / '01-input/output/geo.dat',
                root / '01-input/geometry.dat',
                root / '01-input/geo.dat',
            ],
            'true geometry',
        )

    if args.init_geo is not None:
        init_geo_path = args.init_geo.expanduser().resolve()
    else:
        init_geo_path = first_existing_path(
            [
                root / '04-initial/output/geo.dat',
                root / '04-initial/output/geometry.dat',
                root / '04-initial/geo.dat',
                root / '04-initial/geometry.dat',
            ],
            'initial geometry',
        )

    print(f'[PATH] project root = {root}')
    print(f'[PATH] result_dir   = {result_dir}')
    print(f'[PATH] true_geo     = {true_geo_path}')
    print(f'[PATH] init_geo     = {init_geo_path}')

    true_geo = read_geometry_dat(true_geo_path, with_receivers=False)
    init_geo = read_geometry_dat(init_geo_path, with_receivers=True)
    best_mis, _, best_geo = read_summary_dat(result_dir / 'best.dat')
    mean_mis, _, mean_geo = read_summary_dat(result_dir / 'mean.dat')

    n = min(len(true_geo.sx), len(init_geo.sx), len(best_geo.sx), len(mean_geo.sx))
    rows = []
    for i in range(n):
        rows.append({
            'event': i,
            'true_sx': true_geo.sx[i],
            'true_sz': true_geo.sz[i],
            'init_sx': init_geo.sx[i],
            'init_sz': init_geo.sz[i],
            'best_sx': best_geo.sx[i],
            'best_sz': best_geo.sz[i],
            'mean_sx': mean_geo.sx[i],
            'mean_sz': mean_geo.sz[i],
            'init_error_m': float(np.hypot(init_geo.sx[i] - true_geo.sx[i], init_geo.sz[i] - true_geo.sz[i])),
            'best_error_m': float(np.hypot(best_geo.sx[i] - true_geo.sx[i], best_geo.sz[i] - true_geo.sz[i])),
            'mean_error_m': float(np.hypot(mean_geo.sx[i] - true_geo.sx[i], mean_geo.sz[i] - true_geo.sz[i])),
        })
    pd.DataFrame(rows).to_csv(outdir / 'location_errors.csv', index=False)

    # -------------------------------------------------------------------------
    # All events in one figure
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(true_geo.sx[:n], true_geo.sz[:n], 'o', color=C_TRUE, linestyle='none', markersize=8, label='True')
    ax.plot(init_geo.sx[:n], init_geo.sz[:n], '*', color=C_INIT, linestyle='none', markersize=9, label='Initial')
    best_errors = np.hypot(best_geo.sx[:n] - true_geo.sx[:n], best_geo.sz[:n] - true_geo.sz[:n])
    mean_errors = np.hypot(mean_geo.sx[:n] - true_geo.sx[:n], mean_geo.sz[:n] - true_geo.sz[:n])
    ax.plot(best_geo.sx[:n], best_geo.sz[:n], 's', color=C_BEST, linestyle='none', markersize=7, label=f'Best (misfit={best_mis:.3g}, mean Δ={np.nanmean(best_errors):.1f} m)')
    ax.plot(mean_geo.sx[:n], mean_geo.sz[:n], 'v', color=C_MEAN, linestyle='none', markersize=7, label=f'Mean (misfit={mean_mis:.3g}, mean Δ={np.nanmean(mean_errors):.1f} m)')
    if args.label_events:
        for i in range(n):
            ax.text(true_geo.sx[i], true_geo.sz[i], str(i), fontsize=8)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.25)
    if args.xlim:
        ax.set_xlim(args.xlim)
    if args.ylim:
        ax.set_ylim(args.ylim)
    fig.tight_layout()
    fig.savefig(outdir / 'all_events.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # One event per figure
    # -------------------------------------------------------------------------
    for i in range(n):
        fig, ax = plt.subplots(figsize=(5, 5))
        best_err = location_error_m(best_geo.sx[i], best_geo.sz[i], true_geo.sx[i], true_geo.sz[i])
        mean_err = location_error_m(mean_geo.sx[i], mean_geo.sz[i], true_geo.sx[i], true_geo.sz[i])
        ax.plot(true_geo.sx[i], true_geo.sz[i], 'o', color=C_TRUE, markersize=9, label='True')
        ax.plot(init_geo.sx[i], init_geo.sz[i], '*', color=C_INIT, markersize=10, label='Initial')
        ax.plot(best_geo.sx[i], best_geo.sz[i], 's', color=C_BEST, markersize=8, label=format_error_label('Best', best_err))
        ax.plot(mean_geo.sx[i], mean_geo.sz[i], 'v', color=C_MEAN, markersize=8, label=format_error_label('Mean', mean_err))
        add_error_text_box(ax, best_err, mean_err, loc='upper left')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.25)
        if args.xlim:
            ax.set_xlim(args.xlim)
        if args.ylim:
            ax.set_ylim(args.ylim)
        fig.tight_layout()
        fig.savefig(outdir / f'event_{i:03d}.png', dpi=300)
        plt.close(fig)

    samples, names, used_burnin = load_chain(result_dir, args.burnin)
    if samples is not None:
        print(f'[POSTERIOR] chain samples used = {samples.shape[0]} after burnin = {used_burnin}')
        save_posterior_summary(samples, names, true_geo, outdir, n)
        for i in range(n):
            plot_event_posterior(
                samples,
                names,
                i,
                true_geo,
                init_geo,
                best_geo,
                mean_geo,
                outdir,
                args.max_scatter,
                args.xlim,
                args.ylim,
            )

    print(f'Saved location figures, posterior figures, and CSV summaries to {outdir}')


if __name__ == '__main__':
    main()
