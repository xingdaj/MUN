#!/usr/bin/env python3
"""Field-data workflow for the VTI DAS MCMC code.

Current clean directory layout:

    01-initial/   prepare field receiver/P-time/velocity files, run grid search,
                  write prior/prop, and draw input-check figures
    02-inversion/ joint VTI + event-location DRAM inversion
    03-output/    posterior/result figures after inversion

Default behavior stops after 01-initial so the input files and figures can be
checked before starting the expensive inversion.  Set RUN_INVERSION=True below
after the initial check is confirmed.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time


# =============================================================================
# 0. Field-data controls: normally only edit this block
# =============================================================================
MASTER_SEED = 0

FIELD_VELOCITY_CSV = "../layered_velocity_model.csv"
FIELD_TIMES_CSV = "../event_receiver_times.csv"
FIELD_EVENT_IDS = "all"

# 2-D field inversion: project the field geometry onto the horizontal Y-depth plane.
# The MCMC code still calls this horizontal coordinate X internally.
FIELD_RECEIVER_X_COL = "y_m"
FIELD_RECEIVER_Z_COL = "depth_m"
FIELD_RECEIVER_SORT_COL = "receiver_id"
FIELD_RENUMBER_RECEIVERS = True
FIELD_RECEIVER_STRIDE = 1

FIELD_EPSILON = 0.0
FIELD_GAMMA = 0.0
FIELD_DELTA = 0.0

QX_STOP = 1.0e-6
QX_MAX_ITER = 20
# Fixed objective noise standard deviation. It is written once to nobs.dat/prop.dat
# and is not treated as an inversion parameter.
OBS_NOISE_STD = 0.0020  ###################

# Source initialization and source-position uniform prior.
INIT_GRID_X_MIN = 800.0
INIT_GRID_X_MAX = 1600.0
INIT_GRID_Z_MIN = 1600.0
INIT_GRID_Z_MAX = 2000.0
INIT_GRID_DX = 10.0 ###################################
INIT_GRID_DZ = 10.0 ###################################
INIT_GRID_REFINE = False  # keep final grid spacing at 10 m

PRIOR_SOURCE_X_MIN = 800.0
PRIOR_SOURCE_X_MAX = 1600.0
PRIOR_SOURCE_Z_MIN = 1600.0
PRIOR_SOURCE_Z_MAX = 2000.0

# Uniform VTI parameter priors.
PRIOR_ALPHA_MIN = 1500.0
PRIOR_ALPHA_MAX = 7000.0
PRIOR_BETA_MIN = 1000.0
PRIOR_BETA_MAX = 5000.0
PRIOR_EPSILON_MIN = 0.0
PRIOR_EPSILON_MAX = 0.5
PRIOR_GAMMA_MIN = 0.0
PRIOR_GAMMA_MAX = 0.5
PRIOR_DELTA_MIN = 0.0
PRIOR_DELTA_MAX = 0.5

# 100 m field velocity model: 18 intervals -> 19 depth nodes after appending bottom.
# dep[0]=200 and dep[-1]=2000 are fixed in the MCMC code; internal interfaces are updated.
PRIOR_NMIN = 19
PRIOR_NMAX = 19
PRIOR_SOURCE_MARGIN = 0.0
PRIOR_DEPTH_MARGIN = 0.0
PRIOR_ALPHA_MARGIN = 0.0
PRIOR_BETA_MARGIN = 0.0
PRIOR_ANISO_MARGIN = 0.0

# Interface max deviation around the initial velocity model.  Velocity max_dev=0
# means true box-uniform priors are used instead of initial ± deviation.
PRIOR_DEPTH_MAX_DEV = 20.0
PRIOR_ALPHA_MAX_DEV = 0.0
PRIOR_BETA_MAX_DEV = 0.0
PRIOR_EPSILON_MAX_DEV = 0.0
PRIOR_GAMMA_MAX_DEV = 0.0
PRIOR_DELTA_MAX_DEV = 0.0
PRIOR_SOURCE_X_MAX_DEV = 0.0
PRIOR_SOURCE_Z_MAX_DEV = 0.0
PROP_DEPTH_STD = 10.0
PROP_ALPHA_STD = 100.0
PROP_BETA_STD = 50.0
PROP_EPSILON_STD = 0.005
PROP_GAMMA_STD = 0.005
PROP_DELTA_STD = 0.005
PROP_SOURCE_X_STD = 50.0
PROP_SOURCE_Z_STD = 50.0

# Stage control.  Keep False while checking 01-initial/output/input_check/*.pdf.
# First check field inputs without running the expensive source grid search.
# Set RUN_GRID_SEARCH=True after input_check/*.pdf is confirmed.
RUN_GRID_SEARCH = True
RUN_INVERSION = True
RUN_VELOCITY_PLOTS = True
RUN_FORWARD_COMPARE = True
RUN_EVENT_POSTERIOR_PLOTS = True

# Event posterior plotting controls.
EVENT_POSTERIOR_MAX_SAMPLES = 2500
EVENT_POSTERIOR_DPI = 600

# MCMC controls.
FIX_VELOCITY = False
INVERT_DEPTHS = True
FIX_LAST_LAYER = False
MCMC_ITERATIONS = 300000 ##########
MCMC_BURNIN = 100000  #############
MCMC_PRINT_EVERY = 100
MCMC_SEED = MASTER_SEED + 100
MCMC_ADAPT_START = 2000
MCMC_ADAPT_INTERVAL = 500
MCMC_ADAPT_STOP = 0
MCMC_DR_SCALE = 0.2
MCMC_UPDATE_ORDER = "random"
# Parallel workers used only for full all-event forward modelling when a velocity/depth
# parameter is proposed. Source-location proposals still use one-event forward modelling.
# Set to 1 for serial; set to 0 to use all available CPU cores capped by event count.
MCMC_FORWARD_WORKERS = 8
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_USE_WAVES = "P"
MCMC_SIGMA_MODE = "objective-iid"
PLOT_HIST_BINS_VEL = 40


def infer_project_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = []
    for base in (cwd, here):
        candidates.extend([base, base.parent, base.parent.parent])
    for c in candidates:
        if all((c / d).is_dir() for d in ["01-initial", "02-inversion", "03-output"]):
            return c
    return cwd


def run_py(folder: Path, script_name: str, args: list[str | Path | int | float] | None = None) -> None:
    script = folder / script_name
    if not script.exists():
        raise FileNotFoundError(f"Cannot find script: {script}")
    cmd = [sys.executable, script.name] + [str(a) for a in (args or [])]
    print(f"\n[RUN] cd {folder}")
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=folder, check=True)


def unlink_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        print(f"[CLEAN][WARN] failed to remove {path}: {exc}")


def main() -> int:
    t0 = time.perf_counter()
    root = infer_project_root()
    print(f"[PATH] project root = {root}")

    initial_dir = root / "01-initial" / "output"
    result_dir = root / "03-output" / "output"

    # Step 1: field input preparation from the supplied CSV files.
    run_py(root / "01-initial", "field_prepare.py", [
        "--velocity-csv", (root / FIELD_VELOCITY_CSV).resolve(),
        "--times-csv", (root / FIELD_TIMES_CSV).resolve(),
        "--output-dir", initial_dir,
        "--sigma", OBS_NOISE_STD,
        "--qx-stop", QX_STOP,
        "--qx-max-iter", QX_MAX_ITER,
        "--event-ids", FIELD_EVENT_IDS,
        "--receiver-x-col", FIELD_RECEIVER_X_COL,
        "--receiver-z-col", FIELD_RECEIVER_Z_COL,
        "--receiver-sort-col", FIELD_RECEIVER_SORT_COL,
        "--receiver-stride", FIELD_RECEIVER_STRIDE,
        "--renumber-receivers", str(FIELD_RENUMBER_RECEIVERS),
        "--epsilon", FIELD_EPSILON,
        "--gamma", FIELD_GAMMA,
        "--delta", FIELD_DELTA,
    ])

    # Remove stale grid-search/source files from previous runs.  The input check
    # below must reflect only geometry.dat, not an old geo.dat.
    unlink_if_exists(initial_dir / "geo.dat")
    unlink_if_exists(initial_dir / "geo_grid_search.csv")
    unlink_if_exists(initial_dir / "figures" / "geo_grid_search.png")
    unlink_if_exists(initial_dir / "figures" / "geo_grid_search.pdf")

    # Step 2: prior/proposal files. This uses receiver-only geometry.dat here;
    # explicit source bounds control the event prior before grid search.
    run_py(root / "01-initial", "prior_prop.py", [
        "--true-geo", initial_dir / "geometry.dat",
        "--init-geo", initial_dir / "geometry.dat",
        "--true-vel", initial_dir / "vel.dat",
        "--init-vel", initial_dir / "vel.dat",
        "--prior-file", initial_dir / "prior.dat",
        "--prop-file", initial_dir / "prop.dat",
        "--nmin", PRIOR_NMIN,
        "--nmax", PRIOR_NMAX,
        "--source-margin", PRIOR_SOURCE_MARGIN,
        "--source-x-min", PRIOR_SOURCE_X_MIN,
        "--source-x-max", PRIOR_SOURCE_X_MAX,
        "--source-z-min", PRIOR_SOURCE_Z_MIN,
        "--source-z-max", PRIOR_SOURCE_Z_MAX,
        "--depth-margin", PRIOR_DEPTH_MARGIN,
        "--alpha-margin", PRIOR_ALPHA_MARGIN,
        "--beta-margin", PRIOR_BETA_MARGIN,
        "--aniso-margin", PRIOR_ANISO_MARGIN,
        "--alpha-min", PRIOR_ALPHA_MIN,
        "--alpha-max", PRIOR_ALPHA_MAX,
        "--beta-min", PRIOR_BETA_MIN,
        "--beta-max", PRIOR_BETA_MAX,
        "--epsilon-min", PRIOR_EPSILON_MIN,
        "--epsilon-max", PRIOR_EPSILON_MAX,
        "--gamma-min", PRIOR_GAMMA_MIN,
        "--gamma-max", PRIOR_GAMMA_MAX,
        "--delta-min", PRIOR_DELTA_MIN,
        "--delta-max", PRIOR_DELTA_MAX,
        "--ddep", PRIOR_DEPTH_MAX_DEV,
        "--da", PRIOR_ALPHA_MAX_DEV,
        "--db", PRIOR_BETA_MAX_DEV,
        "--de", PRIOR_EPSILON_MAX_DEV,
        "--dg", PRIOR_GAMMA_MAX_DEV,
        "--dd", PRIOR_DELTA_MAX_DEV,
        "--dh", PRIOR_SOURCE_X_MAX_DEV,
        "--dz", PRIOR_SOURCE_Z_MAX_DEV,
        "--noise-std", OBS_NOISE_STD,
        "--depstd", PROP_DEPTH_STD,
        "--astd", PROP_ALPHA_STD,
        "--bstd", PROP_BETA_STD,
        "--estd", PROP_EPSILON_STD,
        "--gstd", PROP_GAMMA_STD,
        "--dstd", PROP_DELTA_STD,
        "--hstd", PROP_SOURCE_X_STD,
        "--zstd", PROP_SOURCE_Z_STD,
    ])

    # Step 3: draw input checks before source grid search.
    run_py(root / "01-initial", "plot_initial_check.py", [
        "--initial-dir", initial_dir,
        "--geometry-file", initial_dir / "geometry.dat",
        "--mcmc-module", root / "02-inversion" / "vti_joint_mcmc_dram.py",
        "--objective", MCMC_OBJECTIVE,
        "--sigma-mode", MCMC_SIGMA_MODE,
    ])

    if not RUN_GRID_SEARCH:
        print("\n[STOP] Input-check PDFs are ready. Grid search has not been run.")
        print(f"[CHECK] Review: {initial_dir / 'input_check'}")
        print("[NEXT] Set RUN_GRID_SEARCH=True after checking inputs.")
        print(f"[DONE] Initial-stage elapsed time: {time.perf_counter() - t0:.2f} s")
        return 0

    # Step 4: source initial locations from field P-arrival curves.
    grid_args = [
        "--receiver-geometry", initial_dir / "geometry.dat",
        "--init-vel", initial_dir / "vel.dat",
        "--obs-file", initial_dir / "nobs.dat",
        "--output-file", initial_dir / "geo.dat",
        "--csv-file", initial_dir / "geo_grid_search.csv",
        "--fig-file", initial_dir / "figures" / "geo_grid_search.pdf",
        "--mcmc-module", root / "02-inversion" / "vti_joint_mcmc_dram.py",
        "--x-min", INIT_GRID_X_MIN,
        "--x-max", INIT_GRID_X_MAX,
        "--z-min", INIT_GRID_Z_MIN,
        "--z-max", INIT_GRID_Z_MAX,
        "--dx", INIT_GRID_DX,
        "--dz", INIT_GRID_DZ,
        "--objective", MCMC_OBJECTIVE,
        "--sigma-mode", MCMC_SIGMA_MODE,
        "--qx-stop", QX_STOP,
        "--qx-max-iter", QX_MAX_ITER,
        "--forward-workers", MCMC_FORWARD_WORKERS,
    ]
    if INIT_GRID_REFINE:
        grid_args.append("--refine")
    run_py(root / "01-initial", "geo_grid_search.py", grid_args)

    if not RUN_INVERSION:
        print("\n[STOP] Grid-search initial locations are ready.")
        print(f"[CHECK] Review: {initial_dir / 'figures' / 'geo_grid_search.pdf'}")
        print("[NEXT] Set RUN_INVERSION=True after checking grid-search results.")
        print(f"[DONE] Initial-stage elapsed time: {time.perf_counter() - t0:.2f} s")
        return 0

    # Step 5: MCMC inversion.
    mcmc_args = [
        "--input-dir", initial_dir,
        "--output-dir", result_dir,
        "--iterations", MCMC_ITERATIONS,
        "--burnin", MCMC_BURNIN,
        "--seed", MCMC_SEED,
        "--use-waves", MCMC_USE_WAVES,
        "--objective", MCMC_OBJECTIVE,
        "--sigma-mode", MCMC_SIGMA_MODE,
        "--qx-stop", QX_STOP,
        "--qx-max-iter", QX_MAX_ITER,
        "--adapt-start", MCMC_ADAPT_START,
        "--adapt-interval", MCMC_ADAPT_INTERVAL,
        "--adapt-stop", MCMC_ADAPT_STOP,
        "--print-every", MCMC_PRINT_EVERY,
        "--dr-scale", MCMC_DR_SCALE,
        "--update-order", MCMC_UPDATE_ORDER,
        "--forward-workers", MCMC_FORWARD_WORKERS,
    ]
    mcmc_args.append("--invert-depths" if INVERT_DEPTHS else "--no-invert-depths")
    if FIX_LAST_LAYER:
        mcmc_args.append("--fix-last-layer")
    if FIX_VELOCITY:
        mcmc_args.append("--fix-velocity")
    run_py(root / "02-inversion", "vti_joint_mcmc_dram.py", mcmc_args)

    # Step 6: field diagnostics/results.
    if RUN_VELOCITY_PLOTS:
        run_py(root / "03-output", "vel_plot.py", [
            "--root", root,
            "--result-dir", result_dir,
            "--init-vel", initial_dir / "vel.dat",
            "--true-vel", initial_dir / "vel.dat",
            "--bins", PLOT_HIST_BINS_VEL,
            "--plot-last-row",
        ])
    if RUN_FORWARD_COMPARE:
        compare_args = [
            "--root", root,
            "--input-dir", initial_dir,
            "--result-dir", result_dir,
            "--true-ttime-file", initial_dir / "ttime.dat",
            "--control-file", initial_dir / "control.dat",
            "--qx-stop", QX_STOP,
            "--qx-max-iter", QX_MAX_ITER,
            "--objective", MCMC_OBJECTIVE,
            "--sigma-mode", MCMC_SIGMA_MODE,
            "--forward-waves", MCMC_USE_WAVES,
        ]
        compare_args.append("--invert-depths" if INVERT_DEPTHS else "--no-invert-depths")
        if FIX_LAST_LAYER:
            compare_args.append("--fix-last-layer")
        if FIX_VELOCITY:
            compare_args.append("--fix-velocity")
        run_py(root / "03-output", "forward_compare_mean_best.py", compare_args)

    if RUN_EVENT_POSTERIOR_PLOTS:
        run_py(root / "03-output", "event_posterior_plot_region.py", [
            "--root", root,
            "--input-dir", initial_dir,
            "--result-dir", result_dir,
            "--max-samples", EVENT_POSTERIOR_MAX_SAMPLES,
            "--dpi", EVENT_POSTERIOR_DPI,
            "--no-initial",
        ])

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] Total elapsed time: {elapsed:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
