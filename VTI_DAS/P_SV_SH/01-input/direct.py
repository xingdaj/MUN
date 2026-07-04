#!/usr/bin/env python3
"""
Generate synthetic geometry and VTI velocity models for DAS microseismic tests.

This script generates:
  - control.dat
  - geometry.dat
  - vel.dat        : baseline layered VTI model for Tests 1--3
  - vel1.dat       : fine-layer smooth complex truth model for Test 4
  - geo.png
  - vel.png        : baseline layered model plot
  - vel1.png       : complex truth model plot
  - vel_compare.png: baseline vs complex model comparison

Design logic:
  - Tests 1--3 use vel.dat as both the truth and inversion parameterization.
  - Test 4 uses vel1.dat as the truth model, but inversion can still use vel.dat
    as an effective simplified layered VTI model.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User settings
# -----------------------------
OUT_DIR = Path("./output")
SEED: int | None = 0  # use -1 in command line for non-repeatable random perturbations

CONFILE = OUT_DIR / "control.dat"
GEOFILE = OUT_DIR / "geometry.dat"
VELFILE = OUT_DIR / "vel.dat"
VELFILE1 = OUT_DIR / "vel1.dat"

# Output precision: keep enough decimals for geometry/model/control files.
FLOAT_FMT = ".12f"
SMALL_FLOAT_FMT = ".12e"


# -----------------------------
# Helper functions
# -----------------------------
def make_step_profile(depth: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create a stair-step profile for plotting layer-wise constants.

    The input uses depth nodes and values at those nodes. For a model with
    depth=[0, d1, d2, ..., zmax], the plotted constant layers are
    [0,d1], [d1,d2], ..., using value[:-1].
    """
    depth = np.asarray(depth, dtype=float)
    value = np.asarray(value, dtype=float)
    if depth.ndim != 1 or value.ndim != 1 or len(depth) != len(value):
        raise ValueError("depth and value must be 1D arrays with the same length")
    if len(depth) < 2:
        raise ValueError("at least two depth nodes are required")

    nlayer = len(depth)
    nout = nlayer * 2 - 2
    depth_plot = np.zeros(nout, dtype=float)
    value_plot = np.zeros(nout, dtype=float)

    depth_plot[0] = depth[0]
    k = 1
    for i in range(1, nlayer - 1):
        depth_plot[k] = depth[i]
        k += 1
        depth_plot[k] = depth[i]
        k += 1
    depth_plot[-1] = depth[-1]

    k = 0
    for i in range(nlayer - 1):
        value_plot[k] = value[i]
        k += 1
        value_plot[k] = value[i]
        k += 1

    return depth_plot, value_plot


def write_geometry(path: Path, sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("----src(x,z)----\n")
        f.write(f"{len(sx)}\n")
        for x, z in zip(sx, sz):
            f.write(f"{x:{FLOAT_FMT}}\t{z:{FLOAT_FMT}}\n")

        f.write("----rev(x,z)----\n")
        f.write(f"{len(rx)}\n")
        for x, z in zip(rx, rz):
            f.write(f"{x:{FLOAT_FMT}}\t{z:{FLOAT_FMT}}\n")


def write_velocity(
    path: Path,
    dep: np.ndarray,
    alpha0: np.ndarray,
    beta0: np.ndarray,
    epsilon: np.ndarray,
    gamma: np.ndarray,
    delta: np.ndarray,
) -> None:
    arrays = [dep, alpha0, beta0, epsilon, gamma, delta]
    n = len(dep)
    if any(len(a) != n for a in arrays):
        raise ValueError("all velocity arrays must have the same length as dep")

    with path.open("w", encoding="utf-8") as f:
        f.write("----nlayer----\n")
        f.write(f"{n}\n")
        f.write("-----dep, alpha0, beta0, epsilon, gamma, delta ----\n")
        for row in zip(dep, alpha0, beta0, epsilon, gamma, delta):
            f.write("\t".join(f"{v:{FLOAT_FMT}}" for v in row) + "\n")


def write_control(path: Path, stop: float = 1e-6, max_iter: int = 20) -> None:
    # Direct-wave only. Keep id=1 and ir=1 to match the original file format.
    direct_id = 1
    ir = 1
    with path.open("w", encoding="utf-8") as f:
        f.write("----stop criterion----\n")
        f.write(f"{stop:{SMALL_FLOAT_FMT}}\n")
        f.write("------ maximum number of iteration ---------\n")
        f.write(f"{max_iter:d}\n")
        f.write("-------------direct/reflect-----------\n")
        f.write(f"{direct_id:d}\n")
        f.write(f"{ir:d}\n")


def smooth_complex_model(
    dep_nodes: np.ndarray,
    alpha_base: np.ndarray,
    beta_base: np.ndarray,
    eps_base: np.ndarray,
    gam_base: np.ndarray,
    del_base: np.ndarray,
    dz: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth fine-layer model for Test 4.

    The complex model is intentionally not a random white-noise model. It uses
    smooth intra-layer trends and long-wavelength sinusoidal perturbations so
    that it remains geologically plausible and numerically stable for ray tracing.
    """
    zmin, zmax = float(dep_nodes[0]), float(dep_nodes[-1])
    depi = np.arange(zmin, zmax + 0.5 * dz, dz, dtype=float)
    n = len(depi)

    alphai = np.zeros(n, dtype=float)
    betai = np.zeros(n, dtype=float)
    epsiloni = np.zeros(n, dtype=float)
    gammai = np.zeros(n, dtype=float)
    deltai = np.zeros(n, dtype=float)

    # Amplitudes are modest: enough to create parameterization error, but not
    # so large that the model becomes unrealistic or unstable.
    amp_alpha = np.array([90.0, 80.0, 70.0, 120.0])
    amp_beta = np.array([55.0, 50.0, 45.0, 75.0])
    amp_eps = np.array([0.020, 0.025, 0.020, 0.030])
    amp_gam = np.array([0.020, 0.020, 0.018, 0.025])
    amp_del = np.array([0.025, 0.030, 0.020, 0.028])

    trend_alpha = np.array([-70.0, 60.0, -50.0, 100.0])
    trend_beta = np.array([-45.0, 35.0, -30.0, 60.0])
    trend_eps = np.array([-0.015, 0.020, -0.015, 0.020])
    trend_gam = np.array([0.015, -0.012, 0.012, -0.015])
    trend_del = np.array([-0.020, 0.018, -0.012, 0.015])

    for i in range(len(dep_nodes) - 1):
        z0 = dep_nodes[i]
        z1 = dep_nodes[i + 1]
        if i == len(dep_nodes) - 2:
            mask = (depi >= z0) & (depi <= z1)
        else:
            mask = (depi >= z0) & (depi < z1)
        z = depi[mask]
        xi = (z - z0) / max(z1 - z0, dz)

        # Two smooth wavelengths within each layer. Phases differ by parameter
        # to avoid all parameters varying in exactly the same way.
        s1 = np.sin(2.0 * np.pi * xi + 0.35 * i)
        s2 = 0.45 * np.sin(4.0 * np.pi * xi + 0.70 + 0.20 * i)
        smooth = s1 + s2

        alphai[mask] = alpha_base[i] + trend_alpha[i] * (xi - 0.5) + amp_alpha[i] * smooth
        betai[mask] = beta_base[i] + trend_beta[i] * (xi - 0.5) + amp_beta[i] * (0.85 * s1 + 0.35 * s2)
        epsiloni[mask] = eps_base[i] + trend_eps[i] * (xi - 0.5) + amp_eps[i] * (0.80 * s1 - 0.25 * s2)
        gammai[mask] = gam_base[i] + trend_gam[i] * (xi - 0.5) + amp_gam[i] * (0.65 * s1 + 0.30 * s2)
        deltai[mask] = del_base[i] + trend_del[i] * (xi - 0.5) + amp_del[i] * (0.75 * s1 - 0.20 * s2)

    # Keep anisotropic parameters within conservative physical ranges.
    epsiloni = np.clip(epsiloni, 0.08, 0.42)
    gammai = np.clip(gammai, 0.08, 0.42)
    deltai = np.clip(deltai, 0.05, 0.42)

    return depi, alphai, betai, epsiloni, gammai, deltai


def plot_geometry(out_path: Path, sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray, dep: np.ndarray) -> None:
    xline = np.array([0.0, 1200.0])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(sx, sz, "kp", markerfacecolor="r", markersize=8, label="Source")
    ax.plot(rx, rz, "ks", markerfacecolor="r", markersize=4, label="DAS")
    for d in dep[:-1]:
        ax.plot(xline, [d, d], "k-", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Geometry")
    ax.set_xlim(-60.0, 1250.0)
    ax.set_ylim(1000.0, 0.0)
    ax.legend(loc="upper right")
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_velocity(out_path: Path, dep: np.ndarray, alpha0: np.ndarray, beta0: np.ndarray,
                  epsilon: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
                  title: str = "Velocity") -> None:
    z, a = make_step_profile(dep, alpha0)
    _, b = make_step_profile(dep, beta0)
    _, e = make_step_profile(dep, epsilon)
    _, g = make_step_profile(dep, gamma)
    _, d = make_step_profile(dep, delta)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8))

    ax = axes[0]
    ax.plot(a, z, "k-", linewidth=2.0, label=r"$\alpha_0$")
    ax.plot(b, z, "k--", linewidth=2.0, label=r"$\beta_0$")
    ax.set_xlim(0.0, max(np.max(alpha0), np.max(beta0)) * 1.1)
    ax.set_ylim(dep[-1], dep[0])
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.tick_params(labelsize=10)

    ax = axes[1]
    ax.plot(e, z, "k-", linewidth=2.0, label=r"$\epsilon$")
    ax.plot(g, z, "k--", linewidth=2.0, label=r"$\gamma$")
    ax.plot(d, z, "k:", linewidth=2.0, label=r"$\delta$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(dep[-1], dep[0])
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_complex_velocity(out_path: Path, depi: np.ndarray, alphai: np.ndarray, betai: np.ndarray,
                          epsiloni: np.ndarray, gammai: np.ndarray, deltai: np.ndarray,
                          title: str = "Complex truth model") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8))

    ax = axes[0]
    ax.plot(alphai, depi, "k-", linewidth=1.2, label=r"$\alpha_0$")
    ax.plot(betai, depi, "k--", linewidth=1.2, label=r"$\beta_0$")
    ax.set_xlim(0.0, max(np.max(alphai), np.max(betai)) * 1.1)
    ax.set_ylim(depi[-1], depi[0])
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.tick_params(labelsize=10)

    ax = axes[1]
    ax.plot(epsiloni, depi, "k-", linewidth=1.2, label=r"$\epsilon$")
    ax.plot(gammai, depi, "k--", linewidth=1.2, label=r"$\gamma$")
    ax.plot(deltai, depi, "k:", linewidth=1.2, label=r"$\delta$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(depi[-1], depi[0])
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_velocity_compare(out_path: Path, dep: np.ndarray, alpha0: np.ndarray, beta0: np.ndarray,
                          epsilon: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
                          depi: np.ndarray, alphai: np.ndarray, betai: np.ndarray,
                          epsiloni: np.ndarray, gammai: np.ndarray, deltai: np.ndarray) -> None:
    z, a = make_step_profile(dep, alpha0)
    _, b = make_step_profile(dep, beta0)
    _, e = make_step_profile(dep, epsilon)
    _, g = make_step_profile(dep, gamma)
    _, d = make_step_profile(dep, delta)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.8))

    ax = axes[0]
    ax.plot(alphai, depi, "k-", linewidth=1.0, alpha=0.65, label=r"complex $\alpha_0$")
    ax.plot(betai, depi, "k--", linewidth=1.0, alpha=0.65, label=r"complex $\beta_0$")
    ax.plot(a, z, "k-", linewidth=2.4, label=r"layered $\alpha_0$")
    ax.plot(b, z, "k--", linewidth=2.4, label=r"layered $\beta_0$")
    ax.set_xlim(0.0, max(np.max(alphai), np.max(alpha0)) * 1.1)
    ax.set_ylim(dep[-1], dep[0])
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_title("Baseline vs complex")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    ax.plot(epsiloni, depi, "k-", linewidth=1.0, alpha=0.65, label=r"complex $\epsilon$")
    ax.plot(gammai, depi, "k--", linewidth=1.0, alpha=0.65, label=r"complex $\gamma$")
    ax.plot(deltai, depi, "k:", linewidth=1.0, alpha=0.65, label=r"complex $\delta$")
    ax.plot(e, z, "k-", linewidth=2.4, label=r"layered $\epsilon$")
    ax.plot(g, z, "k--", linewidth=2.4, label=r"layered $\gamma$")
    ax.plot(d, z, "k:", linewidth=2.4, label=r"layered $\delta$")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(dep[-1], dep[0])
    ax.set_ylabel("Depth (m)")
    ax.set_title("Baseline vs complex")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


# -----------------------------
# Main script
# -----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate DAS synthetic geometry and VTI models")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR)
    p.add_argument(
        "--seed",
        type=int,
        default=SEED if SEED is not None else -1,
        help="Random seed. Use -1 for non-repeatable random state. Current geometry is deterministic.",
    )
    p.add_argument("--qx-stop", type=float, default=1e-6)
    p.add_argument("--qx-max-iter", type=int, default=20)
    p.add_argument("--complex-dz", type=float, default=0.25, help="Depth interval for vel1.dat")
    return p.parse_args(argv)


def main(argv=None) -> None:
    global OUT_DIR, CONFILE, GEOFILE, VELFILE, VELFILE1
    args = parse_args(argv)
    OUT_DIR = Path(args.output_dir)
    CONFILE = OUT_DIR / "control.dat"
    GEOFILE = OUT_DIR / "geometry.dat"
    VELFILE = OUT_DIR / "vel.dat"
    VELFILE1 = OUT_DIR / "vel1.dat"

    seed = None if int(args.seed) < 0 else int(args.seed)
    if seed is not None:
        np.random.seed(seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Geometry
    # -----------------------------
    # Use 25 events to give Test 1 enough constraints for VTI-parameter recovery.
    # The event cloud is nearly regular for easy error statistics, but includes
    # small deterministic perturbations so that it is not a perfectly artificial grid.
    sx_grid = np.array([500.0, 600.0, 700.0, 800.0, 900.0], dtype=float)
    sz_grid = np.array([440.0, 530.0, 610.0, 710.0, 780.0], dtype=float)
    sx_mesh, sz_mesh = np.meshgrid(sx_grid, sz_grid)
    sx = sx_mesh.ravel()
    sz = sz_mesh.ravel()

    sx += np.array(
        [0.0, -12.0, 8.0, -6.0, 10.0,
         -8.0, 10.0, -5.0, 12.0, -10.0,
         6.0, -9.0, 0.0, 9.0, -6.0,
         -10.0, 5.0, -12.0, 7.0, 11.0,
         8.0, -6.0, 10.0, -8.0, 0.0],
        dtype=float,
    )
    sz += np.array(
        [6.0, -5.0, 3.0, -4.0, 5.0,
         -4.0, 6.0, -5.0, 4.0, -3.0,
         5.0, -4.0, 0.0, 4.0, -5.0,
         -5.0, 3.0, 6.0, -4.0, 5.0,
         4.0, -6.0, 5.0, -3.0, 0.0],
        dtype=float,
    )

    # L-shaped DAS array: vertical part + horizontal part. This can be naturally
    # split into H+V, horizontal-only, and vertical-only geometries for Test 2.
    mr1 = 65
    rx1 = np.full(mr1, 200.0, dtype=float)
    rz1 = 0.0 + 10.0 * np.arange(mr1, dtype=float)  # 0--640 m

    mr2 = 100
    rx2 = 200.0 + 10.0 * np.arange(mr2, dtype=float)  # 200--1190 m
    rz2 = np.full(mr2, 650.0, dtype=float)

    rx = np.concatenate([rx1, rx2])
    rz = np.concatenate([rz1, rz2])

    # -----------------------------
    # Baseline layered VTI velocity model for Tests 1--3
    # -----------------------------
    # Four effective VTI layers with interfaces at 300, 400, and 500 m.
    # The final row at 1000 m carries the same properties as the bottom layer
    # to avoid creating an artificial parameter jump at the model bottom.
    dep = np.array([0.0, 300.0, 400.0, 500.0, 1000.0], dtype=float)
    alpha0 = np.array([5600.0, 5200.0, 4800.0, 5050.0, 5050.0], dtype=float)
    beta0 = np.array([3200.0, 3000.0, 2800.0, 2950.0, 2950.0], dtype=float)
    epsilon = np.array([0.30, 0.25, 0.20, 0.27, 0.27], dtype=float)
    gamma = np.array([0.28, 0.30, 0.24, 0.29, 0.29], dtype=float)
    delta = np.array([0.24, 0.30, 0.16, 0.22, 0.22], dtype=float)

    # -----------------------------
    # Fine-layer smooth complex truth model for Test 4
    # -----------------------------
    depi, alphai, betai, epsiloni, gammai, deltai = smooth_complex_model(
        dep_nodes=dep,
        alpha_base=alpha0,
        beta_base=beta0,
        eps_base=epsilon,
        gam_base=gamma,
        del_base=delta,
        dz=float(args.complex_dz),
    )

    # -----------------------------
    # Write files
    # -----------------------------
    write_geometry(GEOFILE, sx, sz, rx, rz)
    write_velocity(VELFILE, dep, alpha0, beta0, epsilon, gamma, delta)
    write_velocity(VELFILE1, depi, alphai, betai, epsiloni, gammai, deltai)
    write_control(CONFILE, stop=float(args.qx_stop), max_iter=int(args.qx_max_iter))

    # -----------------------------
    # Plot geometry and velocity models
    # -----------------------------
    plot_geometry(OUT_DIR / "geo.png", sx, sz, rx, rz, dep)
    plot_velocity(OUT_DIR / "vel.png", dep, alpha0, beta0, epsilon, gamma, delta, title="Baseline layered model")
    plot_complex_velocity(OUT_DIR / "vel1.png", depi, alphai, betai, epsiloni, gammai, deltai, title="Complex truth model")
    plot_velocity_compare(
        OUT_DIR / "vel_compare.png",
        dep, alpha0, beta0, epsilon, gamma, delta,
        depi, alphai, betai, epsiloni, gammai, deltai,
    )

    print("Done. Files written:")
    for p in [
        CONFILE,
        GEOFILE,
        VELFILE,
        VELFILE1,
        OUT_DIR / "geo.png",
        OUT_DIR / "vel.png",
        OUT_DIR / "vel1.png",
        OUT_DIR / "vel_compare.png",
    ]:
        print(f"  {p}")
    print(f"Number of sources: {len(sx)}")
    print(f"Number of receivers: {len(rx)}")
    print(f"Number of fine layers in vel1.dat: {len(depi)}")


if __name__ == "__main__":
    main()
