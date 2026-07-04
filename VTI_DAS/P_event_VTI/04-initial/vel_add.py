#!/usr/bin/env python3
"""Create an initial VTI layer model by perturbing the true model."""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

DEFAULT_INPUT_FILE = Path("../01-input/output/vel.dat")
DEFAULT_OUTPUT_FILE = Path("../04-initial/output/vel.dat")


def read_velocity(path: Path):
    numbers: list[float] = []
    for tok in path.read_text(encoding="utf-8").replace(",", " ").split():
        try:
            numbers.append(float(tok))
        except ValueError:
            pass
    if not numbers:
        raise ValueError(f"No numeric values found in {path}")
    nlayer = int(numbers[0])
    arr = np.asarray(numbers[1:], dtype=float).reshape(nlayer, 6)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]


def write_velocity(path: Path, dep, alpha0, beta0, epsilon, gamma, delta):
    path.parent.mkdir(parents=True, exist_ok=True)
    nlayer = len(dep)
    with path.open("w", encoding="utf-8") as f:
        f.write("----nlayer----\n")
        f.write(f"{nlayer:d}\n")
        f.write("-----dep, alpha0, beta0, epsilon, gamma, delta ----\n")
        for row in zip(dep, alpha0, beta0, epsilon, gamma, delta):
            f.write("\t".join(f"{v:.12f}" for v in row) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate initial VTI model")
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--seed", type=int, default=0, help="Use -1 for non-repeatable random state")
    p.add_argument("--dstd", type=float, default=20.0)
    p.add_argument("--astd", type=float, default=500.0)
    p.add_argument("--bstd", type=float, default=300.0)
    p.add_argument("--estd", type=float, default=0.2)
    p.add_argument("--gstd", type=float, default=0.2)
    p.add_argument("--delta-std", type=float, default=0.2)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(None if int(args.seed) < 0 else int(args.seed))

    dep, alpha0, beta0, epsilon, gamma, delta = read_velocity(args.input_file)
    nlayer = len(dep)

    ddep = dep.copy()
    aalpha0 = alpha0.copy()
    bbeta0 = beta0.copy()
    eepsilon = epsilon.copy()
    ggamma = gamma.copy()
    ddelta = delta.copy()

    # MATLAB loop: for i = 1:nlayer-1. The last boundary/model row is unchanged.
    npert = nlayer - 1
    ddep[:npert] = dep[:npert] + float(args.dstd) * (rng.random(npert) - 0.5) * 2.0
    aalpha0[:npert] = alpha0[:npert] + float(args.astd) * (rng.random(npert) - 0.5) * 2.0
    bbeta0[:npert] = beta0[:npert] + float(args.bstd) * (rng.random(npert) - 0.5) * 2.0
    eepsilon[:npert] = epsilon[:npert] + float(args.estd) * (rng.random(npert) - 0.5) * 2.0
    ggamma[:npert] = gamma[:npert] + float(args.gstd) * (rng.random(npert) - 0.5) * 2.0
    ddelta[:npert] = delta[:npert] + float(args.delta_std) * (rng.random(npert) - 0.5) * 2.0

    # Keep the first depth boundary at the surface.
    ddep[0] = dep[0]

    write_velocity(args.output_file, ddep, aalpha0, bbeta0, eepsilon, ggamma, ddelta)
    print(f"[VEL_INIT] input={args.input_file}")
    print("[VEL_INIT] perturb half-widths: "
          f"dep={args.dstd:g}, alpha={args.astd:g}, beta={args.bstd:g}, "
          f"epsilon={args.estd:g}, gamma={args.gstd:g}, delta={args.delta_std:g}, seed={args.seed}")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
