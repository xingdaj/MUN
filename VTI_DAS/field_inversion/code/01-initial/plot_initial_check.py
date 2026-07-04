#!/usr/bin/env python3
"""Draw initial field-input diagnostic figures before starting inversion."""
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import sys


def load_mcmc_module(path: Path):
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find MCMC module: {path}")
    spec = importlib.util.spec_from_file_location("vti_mcmc_for_initial_check", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot initial velocity/geometry/P-time checks")
    p.add_argument("--initial-dir", type=Path, default=Path("../01-initial/output"))
    p.add_argument("--geometry-file", type=Path, default=None,
                   help="Geometry file to check. Use geometry.dat before grid search or geo.dat after grid search.")
    p.add_argument("--mcmc-module", type=Path, default=Path("../02-inversion/vti_joint_mcmc_dram.py"))
    p.add_argument("--objective", default="diff-p-adjacent")
    p.add_argument("--sigma-mode", default="objective-iid")
    args = p.parse_args(argv)

    initial_dir = args.initial_dir.resolve()
    mod = load_mcmc_module(args.mcmc_module)
    geometry_file = args.geometry_file if args.geometry_file is not None else initial_dir / "geo.dat"
    geo = mod.read_geometry(geometry_file)
    model = mod.read_model(initial_dir / "vel.dat")
    obs = mod.read_observed(initial_dir / "nobs.dat")
    prior = mod.read_prior(initial_dir / "prior.dat")
    like = mod.build_likelihood_config(obs, args.objective, "P", sigma_mode=args.sigma_mode)
    mod.validate_observed_geometry(geo, obs, tol=1e-8, check_sources=False)
    mod.plot_input_diagnostics(initial_dir, geo, model, obs, like, prior)
    print(f"[PLOT] geometry checked from: {geometry_file}")
    print(f"[PLOT] initial check figures: {initial_dir / 'input_check'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
