#!/usr/bin/env python3
"""Write prior.dat and prop.dat for field VTI DAS inversion.

The field workflow uses explicit uniform prior bounds when they are provided
from shell_field.py.  The older margin-based behavior is still available as a
fallback, but the active field run sets fixed source and VTI bounds.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

DEFAULT_PRIOR_FILE = Path("../01-initial/output/prior.dat")
DEFAULT_PROP_FILE = Path("../01-initial/output/prop.dat")


def numeric_tokens(path: Path) -> list[float]:
    vals: list[float] = []
    for tok in path.read_text(encoding="utf-8").replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def _numeric_values(line: str) -> list[float]:
    vals: list[float] = []
    for tok in line.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_geometry(path: Path):
    """Read full geo.dat or receiver-only geometry.dat.

    geometry.dat intentionally has no source coordinates before grid search:
        ns
        nr
        receiver_id rx rz

    geo.dat after grid search has source coordinates:
        ns
        sx sz
        ...
        nr
        rx rz
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    state = None
    ns = None
    nr = None
    sx_rows: list[tuple[float, float]] = []
    rx_rows: list[tuple[float, float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "src" in low or "number of events" in low:
            state = "src"
            continue
        if "rev" in low or ("receiver" in low and "event" not in low):
            state = "rev"
            continue

        vals = _numeric_values(line)
        if not vals:
            continue
        if state == "src":
            if ns is None:
                ns = int(vals[0])
            elif len(vals) >= 2:
                sx_rows.append((float(vals[0]), float(vals[1])))
        elif state == "rev":
            if nr is None:
                nr = int(vals[0])
            elif len(vals) >= 3:
                rx_rows.append((float(vals[-2]), float(vals[-1])))
            elif len(vals) >= 2:
                rx_rows.append((float(vals[0]), float(vals[1])))

    if ns is None or nr is None:
        # Fallback for old compact numeric geo.dat files.
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

    if len(rx_rows) != nr:
        raise ValueError(f"Geometry file {path} says nr={nr}, but {len(rx_rows)} receiver rows were read")

    sx = np.full(ns, np.nan, dtype=float)
    sz = np.full(ns, np.nan, dtype=float)
    if len(sx_rows) == ns:
        sx[:] = [v[0] for v in sx_rows]
        sz[:] = [v[1] for v in sx_rows]
    rx = np.asarray([v[0] for v in rx_rows], dtype=float)
    rz = np.asarray([v[1] for v in rx_rows], dtype=float)
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
                source_x_min: float | None = None, source_x_max: float | None = None,
                source_z_min: float | None = None, source_z_max: float | None = None,
                alpha_min: float | None = None, alpha_max: float | None = None,
                beta_min: float | None = None, beta_max: float | None = None,
                epsilon_min: float | None = None, epsilon_max: float | None = None,
                gamma_min: float | None = None, gamma_max: float | None = None,
                delta_min: float | None = None, delta_max: float | None = None):
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
    if source_x_min is not None:
        hmin = float(source_x_min)
    if source_x_max is not None:
        hmax = float(source_x_max)
    if source_z_min is not None:
        zmin = float(source_z_min)
    if source_z_max is not None:
        zmax = float(source_z_max)
    if not hmin < hmax:
        raise ValueError(f"Invalid explicit source x bounds: {hmin}, {hmax}")
    if not zmin < zmax:
        raise ValueError(f"Invalid explicit source z bounds: {zmin}, {zmax}")

    # VTI physical bounds can either follow true+initial values plus margins
    # or be set explicitly for field-data uniform priors.  When both explicit
    # bounds are supplied, skip the data-derived range so a constant initial
    # anisotropy model (epsilon/gamma/delta = 0) is valid.
    def explicit_or_range(values, margin, lower_clip, lo_exp, hi_exp, name):
        if lo_exp is not None and hi_exp is not None:
            lo, hi = float(lo_exp), float(hi_exp)
        else:
            lo, hi = _minmax_with_margin(values, margin, lower_clip=lower_clip)
            if lo_exp is not None:
                lo = float(lo_exp)
            if hi_exp is not None:
                hi = float(hi_exp)
        if not lo < hi:
            raise ValueError(f"Invalid explicit {name} bounds: {lo}, {hi}")
        return lo, hi

    amin, amax = explicit_or_range(vel_all[:, 1], alpha_margin, 1.0, alpha_min, alpha_max, "alpha")
    bmin, bmax = explicit_or_range(vel_all[:, 2], beta_margin, 1.0, beta_min, beta_max, "beta")
    emin, emax = explicit_or_range(vel_all[:, 3], aniso_margin, 0.0, epsilon_min, epsilon_max, "epsilon")
    gmin, gmax = explicit_or_range(vel_all[:, 4], aniso_margin, 0.0, gamma_min, gamma_max, "gamma")
    deltamin, deltamax = explicit_or_range(vel_all[:, 5], aniso_margin, 0.0, delta_min, delta_max, "delta")
    for name, lo, hi in [
        ("alpha", amin, amax), ("beta", bmin, bmax), ("epsilon", emin, emax),
        ("gamma", gmin, gmax), ("delta", deltamin, deltamax),
    ]:
        if not lo < hi:
            raise ValueError(f"Invalid explicit {name} bounds: {lo}, {hi}")

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

    print("[PRIOR] bounds generated from true+initial geometry/model files")
    print(f"[PRIOR] source x=({hmin:g}, {hmax:g}), z=({zmin:g}, {zmax:g})")
    print(f"[PRIOR] depth=({depmin:g}, {depmax:g}), alpha=({amin:g}, {amax:g}), beta=({bmin:g}, {bmax:g})")
    print(f"[PRIOR] epsilon=({emin:g}, {emax:g}), gamma=({gmin:g}, {gmax:g}), delta=({deltamin:g}, {deltamax:g})")


def write_prop(path: Path, *, noise_std: float, depstd: float, astd: float, bstd: float,
               estd: float, gstd: float, dstd: float, hstd: float, zstd: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("-------fixed objective noise std, not inverted----------\n")
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
    p.add_argument("--true-geo", type=Path, default=Path("../01-initial/output/geo.dat"))
    p.add_argument("--init-geo", type=Path, default=Path("../01-initial/output/geo.dat"))
    p.add_argument("--true-vel", type=Path, default=Path("../01-initial/output/vel.dat"))
    p.add_argument("--init-vel", type=Path, default=Path("../01-initial/output/vel.dat"))
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
    p.add_argument("--source-x-min", type=float, default=None, help="Optional explicit source x prior minimum")
    p.add_argument("--source-x-max", type=float, default=None, help="Optional explicit source x prior maximum")
    p.add_argument("--source-z-min", type=float, default=None, help="Optional explicit source z prior minimum")
    p.add_argument("--source-z-max", type=float, default=None, help="Optional explicit source z prior maximum")
    p.add_argument("--alpha-min", type=float, default=None, help="Optional explicit alpha/Vp prior minimum")
    p.add_argument("--alpha-max", type=float, default=None, help="Optional explicit alpha/Vp prior maximum")
    p.add_argument("--beta-min", type=float, default=None, help="Optional explicit beta/Vs prior minimum")
    p.add_argument("--beta-max", type=float, default=None, help="Optional explicit beta/Vs prior maximum")
    p.add_argument("--epsilon-min", type=float, default=None, help="Optional explicit epsilon prior minimum")
    p.add_argument("--epsilon-max", type=float, default=None, help="Optional explicit epsilon prior maximum")
    p.add_argument("--gamma-min", type=float, default=None, help="Optional explicit gamma prior minimum")
    p.add_argument("--gamma-max", type=float, default=None, help="Optional explicit gamma prior maximum")
    p.add_argument("--delta-min", type=float, default=None, help="Optional explicit delta prior minimum")
    p.add_argument("--delta-max", type=float, default=None, help="Optional explicit delta prior maximum")
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
        source_x_min=args.source_x_min, source_x_max=args.source_x_max,
        source_z_min=args.source_z_min, source_z_max=args.source_z_max,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        beta_min=args.beta_min, beta_max=args.beta_max,
        epsilon_min=args.epsilon_min, epsilon_max=args.epsilon_max,
        gamma_min=args.gamma_min, gamma_max=args.gamma_max,
        delta_min=args.delta_min, delta_max=args.delta_max,
    )
    write_prop(
        args.prop_file, noise_std=args.noise_std, depstd=args.depstd, astd=args.astd, bstd=args.bstd,
        estd=args.estd, gstd=args.gstd, dstd=args.dstd, hstd=args.hstd, zstd=args.zstd,
    )
    print(f"Wrote {args.prior_file}")
    print(f"Wrote {args.prop_file}")


if __name__ == "__main__":
    main()
