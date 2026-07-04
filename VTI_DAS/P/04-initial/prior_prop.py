#!/usr/bin/env python3
"""Write prior.dat and prop.dat using parameters controlled by shell_simple.py.

The key difference from the old version is that prior bounds are derived from the
true and initial geometry/model files, plus configurable margins.  Therefore, if
the synthetic model domain or true VTI parameters change, the prior file changes
consistently instead of keeping stale hard-coded limits.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

DEFAULT_PRIOR_FILE = Path("../04-initial/output/prior.dat")
DEFAULT_PROP_FILE = Path("../04-initial/output/prop.dat")


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
    sx = np.zeros(ns); sz = np.zeros(ns)
    for i in range(ns):
        sx[i] = vals[idx]; sz[i] = vals[idx + 1]; idx += 2
    nr = int(vals[idx]); idx += 1
    rx = np.zeros(nr); rz = np.zeros(nr)
    for j in range(nr):
        rx[j] = vals[idx]; rz[j] = vals[idx + 1]; idx += 2
    return sx, sz, rx, rz


def read_velocity(path: Path):
    vals = numeric_tokens(path)
    if not vals:
        raise ValueError(f"No numeric values found in {path}")
    nlayer = int(vals[0])
    arr = np.asarray(vals[1:], dtype=float).reshape(nlayer, 6)
    return arr


def _minmax_with_margin(values, margin: float, lower_clip: float | None = None, upper_clip: float | None = None):
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values) - margin)
    hi = float(np.nanmax(values) + margin)
    if lower_clip is not None:
        lo = max(lo, float(lower_clip))
    if upper_clip is not None:
        hi = min(hi, float(upper_clip))
    if not lo < hi:
        raise ValueError(f"Invalid bounds after applying margin: {lo}, {hi}")
    return lo, hi


def write_prior(path: Path, *, nmin: int, nmax: int, true_geo: Path, init_geo: Path, true_vel: Path, init_vel: Path,
                source_margin: float, depth_margin: float, alpha_margin: float, beta_margin: float, aniso_margin: float,
                ddep: float, da: float, db: float, de: float, dg: float, dd: float, dh: float, dz: float,
                noise_max: float, noise_step: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    tsx, tsz, trx, trz = read_geometry(true_geo)
    isx, isz, irx, irz = read_geometry(init_geo)
    tvel = read_velocity(true_vel)
    ivel = read_velocity(init_vel)
    vel_all = np.vstack([tvel, ivel])

    # Depth and source bounds follow the generated model/geometry.
    depmin, depmax = _minmax_with_margin(vel_all[:, 0], depth_margin, lower_clip=0.0)
    hmin, hmax = _minmax_with_margin(np.r_[tsx, isx, trx, irx], source_margin)
    zmin, zmax = _minmax_with_margin(np.r_[tsz, isz, trz, irz, vel_all[:, 0]], source_margin, lower_clip=0.0)

    # VTI physical bounds follow true+initial values plus margins.
    amin, amax = _minmax_with_margin(vel_all[:, 1], alpha_margin, lower_clip=1.0)
    bmin, bmax = _minmax_with_margin(vel_all[:, 2], beta_margin, lower_clip=1.0)
    emin, emax = _minmax_with_margin(vel_all[:, 3], aniso_margin, lower_clip=0.0)
    gmin, gmax = _minmax_with_margin(vel_all[:, 4], aniso_margin, lower_clip=0.0)
    deltamin, deltamax = _minmax_with_margin(vel_all[:, 5], aniso_margin, lower_clip=0.0)

    nsmin, nsmax, dn = 0.0, float(noise_max), float(noise_step)

    with path.open("w", encoding="utf-8") as f:
        f.write("------nmin, nmax----------\n")
        f.write(f"{int(nmin):d}\t{int(nmax):d}\n")
        f.write("------ depth min, max, ddep----------------\n")
        f.write(f"{depmin:.12f}\t{depmax:.12f}\t{float(ddep):.12f}\n")
        f.write("------alpha min, max, da------------\n")
        f.write(f"{amin:.12f}\t{amax:.12f}\t{float(da):.12f}\n")
        f.write("------beta min, max, db------------\n")
        f.write(f"{bmin:.12f}\t{bmax:.12f}\t{float(db):.12f}\n")
        f.write("------epsilon min, max, de------------\n")
        f.write(f"{emin:.12f}\t{emax:.12f}\t{float(de):.12f}\n")
        f.write("------gamma min, max, dg------------\n")
        f.write(f"{gmin:.12f}\t{gmax:.12f}\t{float(dg):.12f}\n")
        f.write("------delta min, max, dd------------\n")
        f.write(f"{deltamin:.12f}\t{deltamax:.12f}\t{float(dd):.12f}\n")
        f.write("------hon min, max, dh------------\n")
        f.write(f"{hmin:.12f}\t{hmax:.12f}\t{float(dh):.12f}\n")
        f.write("------ver min, max, dz------------\n")
        f.write(f"{zmin:.12f}\t{zmax:.12f}\t{float(dz):.12f}\n")
        f.write("------noise variance min, max, dtp------\n")
        f.write(f"{nsmin:.12e}\t{nsmax:.12e}\t{dn:.12e}\n")

    print("[PRIOR] bounds generated from true+initial geometry/model files")
    print(f"[PRIOR] source x=({hmin:g}, {hmax:g}), z=({zmin:g}, {zmax:g})")
    print(f"[PRIOR] depth=({depmin:g}, {depmax:g}), alpha=({amin:g}, {amax:g}), beta=({bmin:g}, {bmax:g})")
    print(f"[PRIOR] epsilon=({emin:g}, {emax:g}), gamma=({gmin:g}, {gmax:g}), delta=({deltamin:g}, {deltamax:g})")


def write_prop(path: Path, *, noise_std: float, depstd: float, astd: float, bstd: float,
               estd: float, gstd: float, dstd: float, hstd: float, zstd: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("-------noise std----------\n")
        f.write(f"{float(noise_std):.12e}\n")
        f.write("------ depth std----------\n")
        f.write(f"{float(depstd):.12f}\n")
        f.write("------alpha std------------\n")
        f.write(f"{float(astd):.12f}\n")
        f.write("------beta std------------\n")
        f.write(f"{float(bstd):.12f}\n")
        f.write("------epsilon std------------\n")
        f.write(f"{float(estd):.12f}\n")
        f.write("------gamma std------------\n")
        f.write(f"{float(gstd):.12f}\n")
        f.write("------delta std------------\n")
        f.write(f"{float(dstd):.12f}\n")
        f.write("------------hon std------------\n")
        f.write(f"{float(hstd):.12f}\n")
        f.write("------ver std------------\n")
        f.write(f"{float(zstd):.12f}\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate prior.dat and prop.dat")
    p.add_argument("--prior-file", type=Path, default=DEFAULT_PRIOR_FILE)
    p.add_argument("--prop-file", type=Path, default=DEFAULT_PROP_FILE)
    p.add_argument("--true-geo", type=Path, default=Path("../01-input/output/geometry.dat"))
    p.add_argument("--init-geo", type=Path, default=Path("../04-initial/output/geo.dat"))
    p.add_argument("--true-vel", type=Path, default=Path("../01-input/output/vel.dat"))
    p.add_argument("--init-vel", type=Path, default=Path("../04-initial/output/vel.dat"))
    p.add_argument("--nmin", type=int, default=2)
    p.add_argument("--nmax", type=int, default=20)
    p.add_argument("--source-margin", type=float, default=100.0)
    p.add_argument("--depth-margin", type=float, default=0.0)
    p.add_argument("--alpha-margin", type=float, default=500.0)
    p.add_argument("--beta-margin", type=float, default=300.0)
    p.add_argument("--aniso-margin", type=float, default=0.10)
    p.add_argument("--ddep", type=float, default=40.0)
    p.add_argument("--da", type=float, default=2500.0)
    p.add_argument("--db", type=float, default=1500.0)
    p.add_argument("--de", type=float, default=0.4)
    p.add_argument("--dg", type=float, default=0.4)
    p.add_argument("--dd", type=float, default=0.4)
    p.add_argument("--dh", type=float, default=200.0)
    p.add_argument("--dz", type=float, default=200.0)
    p.add_argument("--noise-std", type=float, default=0.001)
    p.add_argument("--noise-max", type=float, default=0.01)
    p.add_argument("--noise-step", type=float, default=0.0001)
    p.add_argument("--depstd", type=float, default=10.0)
    p.add_argument("--astd", type=float, default=100.0)
    p.add_argument("--bstd", type=float, default=50.0)
    p.add_argument("--estd", type=float, default=0.005)
    p.add_argument("--gstd", type=float, default=0.005)
    p.add_argument("--dstd", type=float, default=0.005)
    p.add_argument("--hstd", type=float, default=25.0)
    p.add_argument("--zstd", type=float, default=25.0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    write_prior(
        args.prior_file, nmin=args.nmin, nmax=args.nmax,
        true_geo=args.true_geo, init_geo=args.init_geo, true_vel=args.true_vel, init_vel=args.init_vel,
        source_margin=args.source_margin, depth_margin=args.depth_margin,
        alpha_margin=args.alpha_margin, beta_margin=args.beta_margin, aniso_margin=args.aniso_margin,
        ddep=args.ddep, da=args.da, db=args.db, de=args.de, dg=args.dg, dd=args.dd, dh=args.dh, dz=args.dz,
        noise_max=args.noise_max, noise_step=args.noise_step,
    )
    write_prop(
        args.prop_file, noise_std=args.noise_std, depstd=args.depstd, astd=args.astd, bstd=args.bstd,
        estd=args.estd, gstd=args.gstd, dstd=args.dstd, hstd=args.hstd, zstd=args.zstd,
    )
    print(f"Wrote {args.prior_file}")
    print(f"Wrote {args.prop_file}")


if __name__ == "__main__":
    main()
