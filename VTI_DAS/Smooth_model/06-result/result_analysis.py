#!/usr/bin/env python3
"""Analyze and plot vti_joint_mcmc_dram2.py output.

This is the Python replacement of result_analysis.m.  It reads ../06-result/chain.npz
written by vti_joint_mcmc_dram2.py, compares posterior samples with the true model
and initial model, and writes diagnostic figures and CSV summaries.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vti_plot_utils import read_model_dat, read_geometry_dat, load_chain, chain_column, find_existing

PARAMS = ['alpha', 'beta', 'epsilon', 'gamma', 'delta']


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

    # Fallback for the common case:
    # user runs the script while the terminal is inside 06-result.
    if cwd.name == '06-result':
        return cwd.parent
    if here.name == '06-result':
        return here.parent
    return cwd

def _vline(ax, x, label=None, **kwargs):
    if x is not None and np.isfinite(x):
        ax.axvline(float(x), label=label, **kwargs)

def main(argv=None):
    p = argparse.ArgumentParser(description='Posterior analysis for VTI DRAM chain.npz')
    p.add_argument('--root', type=Path, default=None, help='Project root containing 01-input, 04-initial, 06-result')
    p.add_argument('--true-vel', type=Path, default=None)
    p.add_argument('--true-geo', type=Path, default=None)
    p.add_argument('--init-vel', type=Path, default=None)
    p.add_argument('--init-geo', type=Path, default=None)
    p.add_argument('--result-dir', type=Path, default=None)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--burnin-frac', type=float, default=None, help='Fallback burn-in fraction if chain.npz has no burnin_eff')
    p.add_argument('--burnin', type=int, default=None, help='Override burn-in sample count')
    p.add_argument('--hist-bins', type=int, default=30)
    p.add_argument('--plot-layer-hist', action='store_true', help='Draw layer_*.png histograms; default is disabled')
    p.add_argument('--max-events', type=int, default=0, help='Maximum event histograms to export; default 0 means do not draw event_*.png')
    args = p.parse_args(argv)

    root = infer_project_root(args.root)
    result_dir = (args.result_dir.expanduser().resolve() if args.result_dir is not None else root / '06-result/output')
    outdir = (args.output_dir.expanduser().resolve() if args.output_dir is not None else result_dir / 'analysis_figures')
    outdir.mkdir(parents=True, exist_ok=True)

    true_vel = args.true_vel.expanduser().resolve() if args.true_vel is not None else find_existing(root/'01-input/output/vel.dat', root/'01-input/output/vel1.dat')
    true_geo = args.true_geo.expanduser().resolve() if args.true_geo is not None else root/'01-input/output/geometry.dat'
    init_vel = args.init_vel.expanduser().resolve() if args.init_vel is not None else root/'04-initial/output/vel.dat'
    init_geo = args.init_geo.expanduser().resolve() if args.init_geo is not None else root/'04-initial/output/geo.dat'

    print(f'[PATH] project root = {root}')
    print(f'[PATH] result_dir   = {result_dir}')
    print(f'[PATH] true_vel     = {true_vel}')
    print(f'[PATH] init_vel     = {init_vel}')

    m_true = read_model_dat(true_vel)
    m_init = read_model_dat(init_vel)
    g_true = read_geometry_dat(true_geo, with_receivers=False)
    g_init = read_geometry_dat(init_geo, with_receivers=True)
    pack = load_chain(result_dir)
    chain = pack['chain']
    misfit = pack['misfit']
    accepted = pack.get('accepted')
    accepted_stage = pack.get('accepted_stage')
    names = pack.get('param_names')
    niter = len(misfit)
    if args.burnin is not None:
        burnin = int(max(0, args.burnin))
    elif "burnin_eff" in pack:
        burnin = int(np.asarray(pack["burnin_eff"]).item())
    elif "burnin" in pack:
        burnin = int(np.asarray(pack["burnin"]).item())
    else:
        frac = 0.2 if args.burnin_frac is None else float(args.burnin_frac)
        burnin = int(round(niter * frac))
    if burnin >= niter:
        print(f"[WARN] burnin={burnin} >= niter={niter}; using the full chain instead")
        burnin = 0
    samples = chain[burnin:]
    print(f"[POSTERIOR] chain samples used = {samples.shape[0]} after burnin = {burnin}")

    # Misfit and acceptance diagnostics.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, niter+1), misfit, marker='.', linewidth=0.8)
    ax.axvline(burnin + 1, linestyle='--', label='burn-in')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Misfit')
    ax.legend(); fig.tight_layout(); fig.savefig(outdir/'misfit_trace.png', dpi=300); plt.close(fig)

    if accepted is not None:
        win = max(1, min(500, niter // 20 if niter >= 20 else 1))
        mov = np.convolve(accepted.astype(float), np.ones(win)/win, mode='valid')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(win, win+len(mov)), mov, linewidth=1.0)
        ax.set_xlabel('Iteration'); ax.set_ylabel(f'Moving acceptance rate, window={win}')
        fig.tight_layout(); fig.savefig(outdir/'acceptance_trace.png', dpi=300); plt.close(fig)

    rows = []
    if names is not None:
        for j, name in enumerate(names):
            vals = samples[:, j]
            rows.append({'param': name, 'initial': chain[0, j], 'mean': np.mean(vals), 'std': np.std(vals, ddof=1),
                         'p025': np.quantile(vals, 0.025), 'p50': np.quantile(vals, 0.5), 'p975': np.quantile(vals, 0.975)})
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir/'posterior_summary.csv', index=False)

    # Layer-wise histograms: one figure per layer, five parameters.
    # Disabled by default because these layer_*.png figures are already generated elsewhere.
    nlayer = len(m_true.dep)
    if args.plot_layer_hist:
        for k in range(nlayer):
            fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
            for ax, par in zip(axes, PARAMS):
                col = chain_column(pack, f'{par}[{k}]')
                if col is None:
                    ax.set_visible(False); continue
                vals = col[burnin:]
                ax.hist(vals, bins=args.hist_bins, density=True, alpha=0.75)
                _vline(ax, getattr(m_true, par)[k] if k < len(getattr(m_true, par)) else None, 'True', linewidth=1.8)
                _vline(ax, col[0], 'Initial', linestyle='--', linewidth=1.4)
                _vline(ax, np.mean(vals), 'Mean', linestyle=':', linewidth=1.8)
                ax.set_title(par); ax.set_xlabel(par)
            axes[0].set_ylabel('Density')
            handles, labels = axes[0].get_legend_handles_labels()
            if handles: fig.legend(handles, labels, loc='upper right')
            fig.suptitle(f'Layer {k}')
            fig.tight_layout(); fig.savefig(outdir/f'layer_{k:02d}_hist.png', dpi=300); plt.close(fig)

    # Location histograms and posterior clouds.
    # event_*.png figures are skipped by default because --max-events defaults to 0.
    ns = min(len(g_true.sx), args.max_events)
    loc_rows = []
    for i in range(ns):
        sx = chain_column(pack, f'sx[{i}]'); sz = chain_column(pack, f'sz[{i}]')
        if sx is None or sz is None: continue
        sxs = sx[burnin:]; szs = sz[burnin:]
        loc_rows.append({'event': i, 'sx_true': g_true.sx[i], 'sz_true': g_true.sz[i],
                         'sx_init': sx[0], 'sz_init': sz[0],
                         'sx_mean': np.mean(sxs), 'sz_mean': np.mean(szs),
                         'sx_std': np.std(sxs, ddof=1), 'sz_std': np.std(szs, ddof=1),
                         'error_mean_m': float(np.hypot(np.mean(sxs)-g_true.sx[i], np.mean(szs)-g_true.sz[i]))})
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].hist(sxs, bins=args.hist_bins, density=True, alpha=0.75)
        _vline(axes[0], g_true.sx[i], 'True', linewidth=1.8); _vline(axes[0], sx[0], 'Initial', linestyle='--'); _vline(axes[0], np.mean(sxs), 'Mean', linestyle=':')
        axes[0].set_xlabel('sx'); axes[0].set_ylabel('Density')
        axes[1].hist(szs, bins=args.hist_bins, density=True, alpha=0.75)
        _vline(axes[1], g_true.sz[i], 'True', linewidth=1.8); _vline(axes[1], sz[0], 'Initial', linestyle='--'); _vline(axes[1], np.mean(szs), 'Mean', linestyle=':')
        axes[1].set_xlabel('sz')
        axes[2].plot(sxs, szs, marker='.', linestyle='none', markersize=2, alpha=0.4, label='Samples')
        axes[2].plot(g_true.sx[i], g_true.sz[i], marker='o', linestyle='none', label='True')
        axes[2].plot(np.mean(sxs), np.mean(szs), marker='x', linestyle='none', label='Mean')
        axes[2].invert_yaxis(); axes[2].set_xlabel('sx'); axes[2].set_ylabel('sz'); axes[2].legend()
        fig.suptitle(f'Event {i}')
        fig.tight_layout(); fig.savefig(outdir/f'event_{i:03d}_posterior.png', dpi=300); plt.close(fig)
    pd.DataFrame(loc_rows).to_csv(outdir/'location_summary.csv', index=False)

    # Correlation matrix across sx/sz only.
    if names is not None:
        loc_idx = [j for j, nm in enumerate(names) if nm.startswith('sx[') or nm.startswith('sz[')]
        if len(loc_idx) >= 2:
            corr = np.corrcoef(samples[:, loc_idx], rowvar=False)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(corr, vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, label='Correlation')
            ax.set_title('sx/sz posterior correlation')
            fig.tight_layout(); fig.savefig(outdir/'location_correlation.png', dpi=300); plt.close(fig)
            np.savetxt(outdir/'location_correlation.csv', corr, delimiter=',')

    report = {
        'n_iterations': int(niter), 'burnin': int(burnin),
        'initial_misfit': float(misfit[0]), 'final_misfit': float(misfit[-1]),
        'best_misfit': float(np.min(misfit)), 'best_iteration_1based': int(np.argmin(misfit) + 1),
        'total_acceptance_percent': float(100*np.mean(accepted)) if accepted is not None else None,
        'stage1_acceptance_percent': float(100*np.mean(accepted_stage == 1)) if accepted_stage is not None else None,
        'stage2_acceptance_percent': float(100*np.mean(accepted_stage == 2)) if accepted_stage is not None else None,
        'analysis_output': str(outdir),
    }
    (outdir/'analysis_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
