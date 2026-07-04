#!/usr/bin/env python3
"""Plot VTI step profiles and posterior distributions from chain.npz.

Outputs
-------
velocity_figures/
    P.png, S.png, E.png, G.png, D.png
        True / Initial / Mean / Best step profiles, same as the original script.
    posterior_<param>_hist.png
        Layer-by-layer marginal posterior histograms after burn-in.
    posterior_<param>_interval.png
        Layerwise posterior median and 68%/90% intervals plotted versus depth.
    posterior_depth_hist.png
        Marginal posterior histograms of inverted layer-interface depths, if present.
    velocity_model_comparison.csv
        True / Initial / Mean / Best model values.
    velocity_posterior_summary.csv
        Posterior mean, std, and quantiles for model parameters and depths.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vti_plot_utils import read_model_dat, read_summary_dat, step_profile, find_existing


PARAMS = [
    ('alpha', 'P wave alpha (m/s)', 'P.png'),
    ('beta', 'S wave beta (m/s)', 'S.png'),
    ('epsilon', 'Epsilon', 'E.png'),
    ('gamma', 'Gamma', 'G.png'),
    ('delta', 'Delta', 'D.png'),
]

MODEL_COLORS = {
    'True': 'black',
    'Initial': 'tab:orange',
    'Mean': 'tab:green',
    'Best': 'tab:red',
}

POSTERIOR_INTERVAL_COLOR = 'tab:blue'
POSTERIOR_HIST_COLOR = 'tab:blue'


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


def _clean_param_names(raw_names: np.ndarray) -> list[str]:
    names = []
    for x in raw_names:
        if isinstance(x, bytes):
            names.append(x.decode('utf-8', errors='ignore'))
        else:
            names.append(str(x))
    return names


def load_chain(result_dir: Path, burnin: int | None) -> tuple[np.ndarray | None, list[str], int]:
    """Load posterior samples and parameter names from chain.npz."""
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

    if burnin is None and "burnin_eff" in data:
        used_burnin = int(np.asarray(data["burnin_eff"]).item())
    elif burnin is None and "burnin" in data:
        used_burnin = int(np.asarray(data["burnin"]).item())
    else:
        used_burnin = int(max(0, burnin))
    if used_burnin >= chain.shape[0]:
        warnings.warn(f'burnin={used_burnin} >= chain length={chain.shape[0]}; using the full chain instead.')
        used_burnin = 0
    samples = chain[used_burnin:]
    return samples, names, used_burnin


def column_by_name(names: list[str], target: str) -> int | None:
    try:
        return names.index(target)
    except ValueError:
        return None


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'mean': np.nan, 'std': np.nan, 'q05': np.nan, 'q16': np.nan, 'q50': np.nan, 'q84': np.nan, 'q95': np.nan}
    q05, q16, q50, q84, q95 = np.quantile(values, [0.05, 0.16, 0.50, 0.84, 0.95])
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        'q05': float(q05), 'q16': float(q16), 'q50': float(q50), 'q84': float(q84), 'q95': float(q95),
    }


def extract_param_matrix(samples: np.ndarray, names: list[str], par: str, nlayer: int) -> np.ndarray:
    """Return samples x nlayer array for a VTI parameter; missing columns become NaN."""
    out = np.full((samples.shape[0], nlayer), np.nan, dtype=float)
    for k in range(nlayer):
        j = column_by_name(names, f'{par}[{k}]')
        if j is not None:
            out[:, k] = samples[:, j]
    return out


def extract_depth_matrix(samples: np.ndarray, names: list[str], init_dep: np.ndarray, nlayer: int) -> np.ndarray:
    """Return samples x nlayer depth array.

    dep[0] is normally fixed. Any depth not present in the chain is filled by the
    initial model depth, so both fixed-depth and inverted-depth runs can be plotted.
    """
    dep = np.tile(np.asarray(init_dep[:nlayer], dtype=float), (samples.shape[0], 1))
    for k in range(nlayer):
        j = column_by_name(names, f'dep[{k}]')
        if j is not None:
            dep[:, k] = samples[:, j]
    return dep


def plot_param_histograms(
    samples: np.ndarray,
    names: list[str],
    nlayer: int,
    par: str,
    title: str,
    outdir: Path,
    bins: int,
    models: list[tuple[str, object]] | None = None,
    best_model_label: str = 'Best',
) -> None:
    """Plot layerwise posterior histograms with True / Initial / Mean / Best markers.

    Median marker is intentionally removed. The posterior mean is computed from
    the chain samples and plotted as ``Mean``. ``True``, ``Initial`` and ``Best``
    are read from the corresponding model files.
    """
    arr = extract_param_matrix(samples, names, par, nlayer)
    valid_layers = [k for k in range(nlayer) if np.any(np.isfinite(arr[:, k]))]
    if not valid_layers:
        return

    model_lookup = dict(models or [])

    ncols = min(3, len(valid_layers))
    nrows = int(math.ceil(len(valid_layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for ax, k in zip(axes.ravel(), valid_layers):
        ax.set_visible(True)
        vals = arr[:, k]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        ax.hist(vals, bins=bins, density=True, alpha=0.75, color=POSTERIOR_HIST_COLOR)

        # 16% and 84% posterior interval, kept for uncertainty reference.
        q16, q84 = np.quantile(vals, [0.16, 0.84])
        ax.axvline(q16, linestyle='--', linewidth=1.0, color=POSTERIOR_INTERVAL_COLOR, label='16/84%' if k == valid_layers[0] else None)
        ax.axvline(q84, linestyle='--', linewidth=1.0, color=POSTERIOR_INTERVAL_COLOR)

        # Reference/model markers. The first valid layer owns the legend labels.
        label_once = (k == valid_layers[0])
        for label in ('True', 'Initial'):
            model = model_lookup.get(label)
            if model is None:
                continue
            values = getattr(model, par, None)
            if values is not None and k < len(values) and np.isfinite(values[k]):
                ax.axvline(values[k], linestyle='-', linewidth=1.8, color=MODEL_COLORS.get(label), label=label if label_once else None)

        mean_val = float(np.mean(vals))
        if np.isfinite(mean_val):
            ax.axvline(mean_val, linestyle='-', linewidth=1.8, color=MODEL_COLORS.get('Mean'), label='Mean' if label_once else None)

        best_model = model_lookup.get(best_model_label)
        if best_model is not None:
            values = getattr(best_model, par, None)
            if values is not None and k < len(values) and np.isfinite(values[k]):
                ax.axvline(values[k], linestyle='-', linewidth=2.0, color=MODEL_COLORS.get('Best'), label='Best' if label_once else None)

        ax.set_title(f'{par}[{k}]')
        ax.set_xlabel(title)
        ax.set_ylabel('Posterior density')
        ax.grid(True, alpha=0.25)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'Posterior distribution: {title}', y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / f'posterior_{par}_hist.png', dpi=300)
    plt.close(fig)


def plot_param_interval_profile(samples: np.ndarray, names: list[str], init_dep: np.ndarray, nlayer: int,
                                par: str, title: str, outdir: Path) -> None:
    arr = extract_param_matrix(samples, names, par, nlayer)
    dep = extract_depth_matrix(samples, names, init_dep, nlayer)

    mean = np.nanmean(arr, axis=0)
    q16 = np.nanquantile(arr, 0.16, axis=0)
    q84 = np.nanquantile(arr, 0.84, axis=0)
    q05 = np.nanquantile(arr, 0.05, axis=0)
    q95 = np.nanquantile(arr, 0.95, axis=0)
    zmean = np.nanmean(dep, axis=0)

    valid = np.isfinite(mean) & np.isfinite(zmean)
    if not np.any(valid):
        return

    x68_low = np.abs(mean[valid] - q16[valid])
    x68_high = np.abs(q84[valid] - mean[valid])
    x90_low = np.abs(mean[valid] - q05[valid])
    x90_high = np.abs(q95[valid] - mean[valid])

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.errorbar(mean[valid], zmean[valid], xerr=[x68_low, x68_high],
                fmt='o-', capsize=3, label='mean and 68% interval')
    ax.errorbar(mean[valid], zmean[valid], xerr=[x90_low, x90_high],
                fmt='none', capsize=2, alpha=0.45, label='90% interval')
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.set_ylabel('Mean layer-top depth (m)')
    ax.set_title(f'Posterior interval profile: {title}')
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f'posterior_{par}_interval.png', dpi=300)
    plt.close(fig)


def plot_depth_histograms(samples: np.ndarray, names: list[str], nlayer: int, outdir: Path, bins: int) -> None:
    depth_cols = [(k, column_by_name(names, f'dep[{k}]')) for k in range(nlayer)]
    depth_cols = [(k, j) for k, j in depth_cols if j is not None]
    if not depth_cols:
        return

    ncols = min(3, len(depth_cols))
    nrows = int(math.ceil(len(depth_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (k, j) in zip(axes.ravel(), depth_cols):
        ax.set_visible(True)
        vals = samples[:, j]
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, density=True, alpha=0.75, color=POSTERIOR_HIST_COLOR)
        q = np.quantile(vals, [0.16, 0.50, 0.84])
        ax.axvline(q[1], linestyle='-', linewidth=1.6)
        ax.axvline(q[0], linestyle='--', linewidth=1.0)
        ax.axvline(q[2], linestyle='--', linewidth=1.0)
        ax.set_title(f'dep[{k}]')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Posterior density')
        ax.grid(True, alpha=0.25)
    fig.suptitle('Posterior distribution: layer-interface depths', y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / 'posterior_depth_hist.png', dpi=300)
    plt.close(fig)


def save_posterior_summary(samples: np.ndarray | None, names: list[str], init_dep: np.ndarray, nlayer: int, outdir: Path) -> None:
    if samples is None:
        return
    rows = []
    for k in range(nlayer):
        j = column_by_name(names, f'dep[{k}]')
        if j is not None:
            row = {'parameter': 'dep', 'layer': k, 'chain_name': f'dep[{k}]'}
            row.update(quantile_summary(samples[:, j]))
            rows.append(row)
    for par, _, _ in PARAMS:
        for k in range(nlayer):
            j = column_by_name(names, f'{par}[{k}]')
            if j is None:
                continue
            row = {'parameter': par, 'layer': k, 'chain_name': f'{par}[{k}]'}
            row.update(quantile_summary(samples[:, j]))
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / 'velocity_posterior_summary.csv', index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description='Plot VTI parameter profiles and posterior distributions')
    p.add_argument('--root', type=Path, default=None, help='Project root containing 01-input, 04-initial, 06-result')
    p.add_argument('--true-vel', type=Path, default=None, help='Default uses 01-input/output/vel.dat. Pass vel1.dat explicitly only for Test 4 diagnostics.')
    p.add_argument('--plot-last-row', action='store_true', help='Also plot the final model row. By default it is treated as the bottom boundary/fixed half-space row and is omitted from layerwise posterior plots.')
    p.add_argument('--init-vel', type=Path, default=None)
    p.add_argument('--result-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--burnin', type=int, default=None, help='Discard this many MCMC samples; default reads burnin_eff from chain.npz')
    p.add_argument('--bins', type=int, default=40, help='Histogram bins for posterior plots')
    args = p.parse_args(argv)

    root = infer_project_root(args.root)
    result_dir = (args.result_dir.expanduser().resolve() if args.result_dir is not None else root / '06-result/output')
    outdir = (args.output_dir.expanduser().resolve() if args.output_dir is not None else result_dir / 'velocity_figures')
    outdir.mkdir(parents=True, exist_ok=True)

    true_vel_path = args.true_vel.expanduser().resolve() if args.true_vel is not None else find_existing(root / '01-input/output/vel.dat', root / '01-input/output/vel1.dat')
    init_vel_path = args.init_vel.expanduser().resolve() if args.init_vel is not None else root / '04-initial/output/vel.dat'

    print(f'[PATH] project root = {root}')
    print(f'[PATH] result_dir   = {result_dir}')
    print(f'[PATH] true_vel     = {true_vel_path}')
    print(f'[PATH] init_vel     = {init_vel_path}')

    m_true = read_model_dat(true_vel_path)
    m_init = read_model_dat(init_vel_path)
    _, m_mean, _ = read_summary_dat(result_dir / 'mean.dat')
    _, m_best, _ = read_summary_dat(result_dir / 'best.dat')
    models = [('True', m_true), ('Initial', m_init), ('Mean', m_mean), ('Best', m_best)]

    rows = []
    n_all = min(len(m_true.dep), len(m_init.dep), len(m_mean.dep), len(m_best.dep))
    # direct.py writes depth nodes [top, interfaces..., bottom].  The final row
    # is the bottom boundary/fixed half-space marker and is not an independently
    # interpreted layer in the synthetic tests.  Omit it by default.
    n = n_all if args.plot_last_row else max(1, n_all - 1)
    for k in range(n):
        row = {'layer': k, 'depth': m_true.dep[k]}
        for label, model in models:
            row[f'{label}_dep'] = model.dep[k] if k < len(model.dep) else np.nan
            for par, _, _ in PARAMS:
                row[f'{label}_{par}'] = getattr(model, par)[k] if k < len(getattr(model, par)) else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / 'velocity_model_comparison.csv', index=False)

    # Original profile figures: True / Initial / Mean / Best.
    for par, title, fname in PARAMS:
        fig, ax = plt.subplots(figsize=(5, 6))
        for label, model in models:
            x, z = step_profile(model, par)
            ax.plot(x, z, marker='o', linewidth=1.8, color=MODEL_COLORS.get(label), label=label)
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=300)
        plt.close(fig)

    # New posterior figures from chain.npz.
    samples, names, used_burnin = load_chain(result_dir, args.burnin)
    if samples is not None:
        print(f'[POSTERIOR] chain samples used = {samples.shape[0]} after burnin = {used_burnin}')
        save_posterior_summary(samples, names, m_init.dep, n, outdir)
        plot_depth_histograms(samples, names, n, outdir, args.bins)
        for par, title, _ in PARAMS:
            plot_param_histograms(samples, names, n, par, title, outdir, args.bins, models=models)
            plot_param_interval_profile(samples, names, m_init.dep, n, par, title, outdir)

    print(f'Saved velocity profiles, posterior figures, and CSV summaries to {outdir}')


if __name__ == '__main__':
    main()
