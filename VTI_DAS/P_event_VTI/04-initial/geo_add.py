#!/usr/bin/env python3
"""Create an initial source geometry by perturbing the true source locations."""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

DEFAULT_INPUT_FILE = Path("../01-input/output/geometry.dat")
DEFAULT_OUTPUT_FILE = Path("../04-initial/output/geo.dat")


def numeric_tokens(path: Path) -> list[float]:
    vals: list[float] = []
    for tok in path.read_text(encoding="utf-8").replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_geometry(path: Path):
    vals = numeric_tokens(path)
    idx = 0
    ns = int(vals[idx]); idx += 1
    sx = np.zeros(ns, dtype=float)
    sz = np.zeros(ns, dtype=float)
    for i in range(ns):
        sx[i] = vals[idx]
        sz[i] = vals[idx + 1]
        idx += 2

    mr = int(vals[idx]); idx += 1
    rx = np.zeros(mr, dtype=float)
    rz = np.zeros(mr, dtype=float)
    for i in range(mr):
        rx[i] = vals[idx]
        rz[i] = vals[idx + 1]
        idx += 2
    return sx, sz, rx, rz


def write_geometry(path: Path, sx, sz, rx, rz):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("----src(x,z)----\n")
        f.write(f"{len(sx):d}\n")
        for x, z in zip(sx, sz):
            f.write(f"{x:.12f}\t{z:.12f}\n")
        f.write("----rev(x,z)----\n")
        f.write(f"{len(rx):d}\n")
        for x, z in zip(rx, rz):
            f.write(f"{x:.12f}\t{z:.12f}\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate initial source geometry")
    p.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    p.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    p.add_argument("--seed", type=int, default=0, help="Use -1 for non-repeatable random state")
    p.add_argument("--xstd", type=float, default=100.0, help="Uniform perturbation half-width for sx")
    p.add_argument("--zstd", type=float, default=100.0, help="Uniform perturbation half-width for sz")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(None if int(args.seed) < 0 else int(args.seed))
    sx, sz, rx, rz = read_geometry(args.input_file)
    ns = len(sx)

    snx = sx + float(args.xstd) * (rng.random(ns) - 0.5) * 2.0
    snz = sz + float(args.zstd) * (rng.random(ns) - 0.5) * 2.0

    write_geometry(args.output_file, snx, snz, rx, rz)
    print(f"[GEO_INIT] input={args.input_file}")
    print(f"[GEO_INIT] xstd={args.xstd:g}, zstd={args.zstd:g}, seed={args.seed}")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
