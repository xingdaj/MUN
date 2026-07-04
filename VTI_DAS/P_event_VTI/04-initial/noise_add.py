#!/usr/bin/env python3
"""Add Gaussian noise to forward travel times and write nobs.dat."""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_OBS_FILE = Path("../03-output/ttime.dat")
DEFAULT_NOISE_FILE = Path("../04-initial/output/nobs.dat")
DEFAULT_FIG_DIR = Path("../04-initial/output/figures")


def read_ttime(path: Path):
    """Read vti_direct.py ttime.dat.

    Format:
        ns nr
        sx sz rx rz tp tsv tsh
        ... repeated ns*nr rows ...
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().split()
        if len(first) < 2:
            raise ValueError("The first line of ttime.dat must contain ns and nr")
        ns, nr = int(first[0]), int(first[1])
        data = np.loadtxt(f)

    expected = ns * nr
    data = np.atleast_2d(data)
    if data.shape[0] != expected or data.shape[1] < 7:
        raise ValueError(f"Expected {expected} rows and at least 7 columns in {path}, got {data.shape}")

    sx = data[:, 0].reshape(ns, nr)[:, 0]
    sz = data[:, 1].reshape(ns, nr)[:, 0]
    rx = data[:, 2].reshape(ns, nr)[0, :]
    rz = data[:, 3].reshape(ns, nr)[0, :]
    tp = data[:, 4].reshape(ns, nr)
    tsv = data[:, 5].reshape(ns, nr)
    tsh = data[:, 6].reshape(ns, nr)
    return sx, sz, rx, rz, tp, tsv, tsh


def write_noisy_obs(path: Path, sigma: float, sx, sz, rx, rz, tp, tsv, tsh):
    """Write nobs.dat in the layout expected by vti_joint_mcmc_dram.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, nr = tp.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("--------------- noise standard deviation(s) ---------------------\n")
        f.write(f"{sigma:.12e}\n")
        f.write("----------------------- ns,nr-------------------------------\n")
        f.write(f"{ns:d}\t{nr:d}\n")
        f.write("-------------------sx,sz,rx,rz,tp,tsv,tsh---------------------------------\n")
        for i in range(ns):
            for j in range(nr):
                f.write(
                    f"{sx[i]:.12f}\t{sz[i]:.12f}\t{rx[j]:.12f}\t{rz[j]:.12f}\t"
                    f"{tp[i, j]:.12e}\t{tsv[i, j]:.12e}\t{tsh[i, j]:.12e}\n"
                )


def plot_noise(fig_dir: Path, tp, tsv, tsh, tpn, tsvn, tshn):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ns, nr = tp.shape
    receiver_index = np.arange(1, nr + 1)
    for i in range(ns):
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        ax.plot(tp[i, :], receiver_index, "ko-", label="True-P")
        ax.plot(tsv[i, :], receiver_index, "ro-", label="True-SV")
        ax.plot(tsh[i, :], receiver_index, "bo-", label="True-SH")
        ax.plot(tpn[i, :], receiver_index, "k*-", label="Noise-P")
        ax.plot(tsvn[i, :], receiver_index, "r*-", label="Noise-SV")
        ax.plot(tshn[i, :], receiver_index, "b*-", label="Noise-SH")
        ax.grid(True)
        ax.legend(loc="upper right")
        ax.set_xlabel("Time(s)")
        ax.set_ylabel("Receiver")
        ax.invert_yaxis()
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{i + 1}noise.png", dpi=600)
        plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Add Gaussian noise to ttime.dat")
    p.add_argument("--obs-file", type=Path, default=DEFAULT_OBS_FILE)
    p.add_argument("--output-file", type=Path, default=DEFAULT_NOISE_FILE)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="Use -1 for non-repeatable random state")
    p.add_argument("--noise-std", type=float, default=0.00100)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(None if int(args.seed) < 0 else int(args.seed))
    sx, sz, rx, rz, tp, tsv, tsh = read_ttime(args.obs_file)
    sigma = float(args.noise_std)

    tpn = tp + rng.standard_normal(tp.shape) * sigma
    tsvn = tsv + rng.standard_normal(tsv.shape) * sigma
    tshn = tsh + rng.standard_normal(tsh.shape) * sigma

    if not args.no_figures:
        plot_noise(args.fig_dir, tp, tsv, tsh, tpn, tsvn, tshn)
        print(f"Wrote noise figures to {args.fig_dir}")
    write_noisy_obs(args.output_file, sigma, sx, sz, rx, rz, tpn, tsvn, tshn)
    print(f"[NOISE] input={args.obs_file}")
    print(f"[NOISE] std={sigma:.12e}, seed={args.seed}")
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
