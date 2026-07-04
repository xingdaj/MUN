#!/usr/bin/env python3
"""Plot field-event posterior location distributions from MCMC chain.npz.

This script is designed for the field-only workflow.  It reads posterior
samples of sx[i] and sz[i] from 03-output/output/chain.npz, reads posterior
mean event locations from mean.dat, and generates one posterior-location PDF
for each event.

Outputs
-------
event_posterior/
    Event_001_posterior.pdf, Event_002_posterior.pdf, ...
    field_event_posterior_map.pdf
    field_event_posterior_summary.csv

For field data there are no known true event locations.  Therefore the figures
show:
    Posterior samples, posterior mean, and 95% covariance ellipse.
    The field map and event panels are cropped to the research region.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import math
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

try:
    from vti_plot_utils import read_summary_dat
except Exception:  # pragma: no cover - fallback for unusual run locations
    read_summary_dat = None


# Research-region plotting window.
# Only this prior/research area is shown in both the field map and each event panel.
PLOT_X_MIN = 800.0
PLOT_X_MAX = 1600.0
PLOT_Z_MIN = 1600.0
PLOT_Z_MAX = 2000.0
PLOT_X_TICKS = np.arange(PLOT_X_MIN, PLOT_X_MAX + 0.1, 200.0)
PLOT_Z_TICKS = np.arange(PLOT_Z_MIN, PLOT_Z_MAX + 0.1, 100.0)


@dataclass
class Geometry:
    sx: np.ndarray
    sz: np.ndarray
    rx: np.ndarray | None = None
    rz: np.ndarray | None = None

    @property
    def ns(self) -> int:
        return int(len(self.sx))


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def infer_project_root(user_root: Path | None = None) -> Path:
    """Infer the code0 project root containing 01-initial/02-inversion/03-output."""
    if user_root is not None:
        return Path(user_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    here = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for base in (cwd, here):
        candidates.extend([base, base.parent, base.parent.parent])

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if all((c / d).is_dir() for d in ("01-initial", "02-inversion", "03-output")):
            return c
    return cwd


def _numeric_values(line: str) -> list[float]:
    vals: list[float] = []
    for tok in re.split(r"[\s,]+", line.strip()):
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_field_geometry(path: str | Path) -> Geometry:
    """Read either receiver-only geometry.dat or full geo.dat.

    Receiver-only geometry.dat format:
        number of events
        ns
        receivers
        nr
        receiver_id rx rz

    Full geo.dat format after grid search:
        src
        ns
        sx sz
        ...
        receiver
        nr
        rx rz
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    state: str | None = None
    ns: int | None = None
    nr: int | None = None
    sx_rows: list[tuple[float, float]] = []
    rx_rows: list[tuple[float, float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "src" in low or "source" in low or "number of events" in low:
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
                ns = int(round(vals[0]))
            elif len(vals) >= 2:
                # Full geo.dat may contain source coordinates. geometry.dat does not.
                sx_rows.append((float(vals[-2]), float(vals[-1])))
        elif state == "rev":
            if nr is None:
                nr = int(round(vals[0]))
            elif len(vals) >= 3:
                # receiver_id rx rz
                rx_rows.append((float(vals[-2]), float(vals[-1])))
            elif len(vals) >= 2:
                rx_rows.append((float(vals[0]), float(vals[1])))

    if ns is None:
        raise ValueError(f"Cannot read event count from geometry file: {path}")
    sx = np.full(ns, np.nan, dtype=float)
    sz = np.full(ns, np.nan, dtype=float)
    if len(sx_rows) == ns:
        sx[:] = [v[0] for v in sx_rows]
        sz[:] = [v[1] for v in sx_rows]

    rx = rz = None
    if nr is not None and len(rx_rows) == nr:
        rx = np.asarray([v[0] for v in rx_rows], dtype=float)
        rz = np.asarray([v[1] for v in rx_rows], dtype=float)
    return Geometry(sx=sx, sz=sz, rx=rx, rz=rz)


def fallback_read_summary_dat(path: str | Path) -> tuple[float | None, Geometry]:
    """Read only the source-location block from mean.dat/best.dat."""
    rows: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        vals = _numeric_values(line)
        if vals:
            rows.append(vals)
    if len(rows) < 2:
        raise ValueError(f"Cannot read summary file: {path}")
    r = 0
    misfit = float(rows[r][0]); r += 1
    nlayer = int(round(rows[r][0])); r += 1
    r += nlayer
    if r >= len(rows):
        raise ValueError(f"{path}: missing source-location block")
    ns = int(round(rows[r][0])); r += 1
    sx, sz = [], []
    for _ in range(ns):
        row = rows[r]; r += 1
        if len(row) >= 3:
            sx.append(float(row[-2])); sz.append(float(row[-1]))
        elif len(row) >= 2:
            sx.append(float(row[0])); sz.append(float(row[1]))
        else:
            raise ValueError(f"{path}: invalid source row {row}")
    return misfit, Geometry(np.asarray(sx, dtype=float), np.asarray(sz, dtype=float))


def read_mean_geometry(path: str | Path) -> Geometry:
    if read_summary_dat is not None:
        _, _, geo = read_summary_dat(Path(path))
        return Geometry(np.asarray(geo.sx, dtype=float), np.asarray(geo.sz, dtype=float))
    _, geo = fallback_read_summary_dat(path)
    return geo


def clean_param_names(raw_names) -> list[str]:
    out: list[str] = []
    for x in raw_names:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="ignore"))
        else:
            out.append(str(x))
    return out


def default_param_names(npar: int, ns: int | None = None) -> list[str]:
    """Fallback names; normally chain.npz contains param_names."""
    if ns is not None and npar >= 2 * ns:
        other = npar - 2 * ns
        return [f"p[{i}]" for i in range(other)] + [f"sx[{i}]" for i in range(ns)] + [f"sz[{i}]" for i in range(ns)]
    return [f"p[{i}]" for i in range(npar)]


def load_chain(result_dir: Path, burnin: int | None = None, ns_hint: int | None = None) -> tuple[np.ndarray | None, list[str], int]:
    path = result_dir / "chain.npz"
    if not path.exists():
        warnings.warn(f"chain.npz not found: {path}; event posterior plots are skipped")
        return None, [], 0

    data = np.load(path, allow_pickle=True)
    chain = None
    for key in ("chain", "samples", "arr_0"):
        if key in data:
            chain = np.asarray(data[key], dtype=float)
            break
    if chain is None:
        warnings.warn(f"{path} has no chain/samples array; event posterior plots are skipped")
        return None, [], 0
    if chain.ndim == 3:
        chain = chain.reshape(-1, chain.shape[-1])
    if chain.ndim != 2:
        warnings.warn(f"Invalid chain shape in {path}: {chain.shape}; event posterior plots are skipped")
        return None, [], 0

    if "param_names" in data:
        names = clean_param_names(data["param_names"])
    elif "names" in data:
        names = clean_param_names(data["names"])
    elif "columns" in data:
        names = clean_param_names(data["columns"])
    else:
        names = default_param_names(chain.shape[1], ns_hint)

    if len(names) != chain.shape[1]:
        warnings.warn(f"Name count {len(names)} does not match chain dimension {chain.shape[1]} in {path}; using fallback names")
        names = default_param_names(chain.shape[1], ns_hint)

    if burnin is None and "burnin_eff" in data:
        used_burnin = int(np.asarray(data["burnin_eff"]).item())
    elif burnin is None and "burnin" in data:
        used_burnin = int(np.asarray(data["burnin"]).item())
    else:
        used_burnin = int(max(0, burnin or 0))

    if used_burnin >= chain.shape[0]:
        warnings.warn(f"burnin={used_burnin} >= chain length={chain.shape[0]}; using full chain")
        used_burnin = 0
    return chain[used_burnin:], names, used_burnin


def column_index(names: list[str], target: str) -> int | None:
    try:
        return names.index(target)
    except ValueError:
        return None


def event_samples(samples: np.ndarray | None, names: list[str], event_index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if samples is None:
        return None, None
    jx = column_index(names, f"sx[{event_index}]")
    jz = column_index(names, f"sz[{event_index}]")
    if jx is None or jz is None:
        return None, None
    sx = np.asarray(samples[:, jx], dtype=float)
    sz = np.asarray(samples[:, jz], dtype=float)
    return sx, sz


def add_cov_ellipse(ax, sx: np.ndarray, sz: np.ndarray, nstd: float = 2.44774683068, **kwargs) -> None:
    sx = np.asarray(sx, dtype=float)
    sz = np.asarray(sz, dtype=float)
    good = np.isfinite(sx) & np.isfinite(sz)
    sx = sx[good]
    sz = sz[good]
    if sx.size < 3:
        return
    cov = np.cov(np.vstack([sx, sz]))
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * nstd * np.sqrt(vals)
    ax.add_patch(Ellipse((np.mean(sx), np.mean(sz)), width=width, height=height,
                         angle=angle, fill=False, **kwargs))


def nice_step(span: float) -> float:
    if span <= 25:
        return 5.0
    if span <= 60:
        return 10.0
    if span <= 120:
        return 20.0
    if span <= 250:
        return 50.0
    return 100.0


def rounded_axis_bounds(lo: float, hi: float, inverted: bool = False) -> tuple[float, float, np.ndarray]:
    a = min(lo, hi)
    b = max(lo, hi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a, b = 0.0, 1.0
    step = nice_step(b - a)
    aa = step * math.floor(a / step)
    bb = step * math.ceil(b / step)
    ticks = np.arange(aa, bb + 0.5 * step, step)
    if inverted:
        return bb, aa, ticks
    return aa, bb, ticks


def event_label(event_index: int, event_ids: list[str] | None = None) -> str:
    if event_ids is not None and event_index < len(event_ids):
        return f"Event {event_ids[event_index]}"
    return f"Event {event_index + 1}"


def read_event_id_map(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("event_id")
            if val is not None and str(val).strip():
                out.append(str(val).strip())
    return out or None


def finite_xy(sx: np.ndarray, sz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx = np.asarray(sx, dtype=float)
    sz = np.asarray(sz, dtype=float)
    good = np.isfinite(sx) & np.isfinite(sz)
    return sx[good], sz[good]


def region_mask_xy(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Mask finite points inside the fixed research/prior region."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    return (
        np.isfinite(x) & np.isfinite(z)
        & (x >= PLOT_X_MIN) & (x <= PLOT_X_MAX)
        & (z >= PLOT_Z_MIN) & (z <= PLOT_Z_MAX)
    )


def plot_all_event_map(mean_geo: Geometry, init_geo: Geometry | None, receiver_geo: Geometry | None,
                       out_pdf: Path, dpi: int, event_ids: list[str] | None = None) -> None:
    """Plot only the fixed research region with DAS and posterior-mean events.

    Initial/grid-search locations are intentionally not drawn.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    handles: list = []
    if receiver_geo is not None and receiver_geo.rx is not None and receiver_geo.rz is not None:
        das_mask = region_mask_xy(receiver_geo.rx, receiver_geo.rz)
        if np.any(das_mask):
            das, = ax.plot(receiver_geo.rx[das_mask], receiver_geo.rz[das_mask],
                           linestyle="None", marker="s", markersize=2.4,
                           markerfacecolor="red", markeredgecolor="black", markeredgewidth=0.25,
                           label="DAS")
            handles.append(das)

    mean_mask = region_mask_xy(mean_geo.sx, mean_geo.sz)
    if np.any(mean_mask):
        mean_h, = ax.plot(mean_geo.sx[mean_mask], mean_geo.sz[mean_mask],
                          linestyle="None", marker="o", markersize=6.4,
                          markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.4,
                          label="Posterior mean")
        handles.append(mean_h)

        for i in np.where(mean_mask)[0]:
            label = event_ids[i] if event_ids is not None and i < len(event_ids) else str(i + 1)
            ax.text(mean_geo.sx[i], mean_geo.sz[i], label, fontsize=6.2,
                    ha="center", va="center", color="black")

    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Z_MAX, PLOT_Z_MIN)
    ax.set_xticks(PLOT_X_TICKS)
    ax.set_yticks(PLOT_Z_TICKS)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)
    ax.grid(color="0.92", linewidth=0.6)
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_one_event(event_index: int, sx: np.ndarray, sz: np.ndarray, mean_geo: Geometry,
                   init_geo: Geometry | None, out_pdf: Path, dpi: int, max_samples: int,
                   event_ids: list[str] | None = None, rng: np.random.Generator | None = None) -> dict[str, float | int | str]:
    """Plot one event posterior in the fixed research region.

    Initial/grid-search locations are intentionally not drawn.  Summary
    statistics are still computed from all finite posterior samples; only the
    displayed points are clipped to the research window.
    """
    rng = rng or np.random.default_rng(20260628)
    sx, sz = finite_xy(sx, sz)
    if sx.size == 0:
        raise ValueError(f"No finite posterior samples for event {event_index + 1}")

    if sx.size > max_samples:
        idx = rng.choice(sx.size, size=max_samples, replace=False)
        sxp, szp = sx[idx], sz[idx]
    else:
        sxp, szp = sx, sz

    # Only show posterior samples inside the research/prior region.
    show_mask = region_mask_xy(sxp, szp)
    sxp_show = sxp[show_mask]
    szp_show = szp[show_mask]

    mean_x = float(mean_geo.sx[event_index])
    mean_z = float(mean_geo.sz[event_index])
    post_mean_x = float(np.mean(sx))
    post_mean_z = float(np.mean(sz))
    mean_from_file_diff = float(np.hypot(mean_x - post_mean_x, mean_z - post_mean_z))

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    post_handle, = ax.plot(sxp_show, szp_show, linestyle="None", marker=".", markersize=1.4,
                           color="0.55", alpha=0.40, label="Posterior")
    add_cov_ellipse(ax, sx, sz, edgecolor="black", linestyle="--", linewidth=1.05)
    ellipse_handle = Line2D([], [], color="black", linestyle="--", linewidth=1.05, label="95% ellipse")

    mean_handle, = ax.plot(mean_x, mean_z, marker="o", linestyle="None", markersize=7.2,
                           markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.4,
                           label="Posterior mean", zorder=5)

    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Z_MAX, PLOT_Z_MIN)
    ax.set_xticks(PLOT_X_TICKS)
    ax.set_yticks(PLOT_Z_TICKS)
    ax.set_aspect("equal", adjustable="box")

    qx025, qx16, qx50, qx84, qx975 = np.quantile(sx, [0.025, 0.16, 0.50, 0.84, 0.975])
    qz025, qz16, qz50, qz84, qz975 = np.quantile(sz, [0.025, 0.16, 0.50, 0.84, 0.975])
    std_x = float(np.std(sx, ddof=1)) if sx.size > 1 else 0.0
    std_z = float(np.std(sz, ddof=1)) if sz.size > 1 else 0.0
    corr = float(np.corrcoef(sx, sz)[0, 1]) if sx.size > 1 and std_x > 0 and std_z > 0 else np.nan
    ci_label = f"Std x/z = {std_x:.1f}/{std_z:.1f} m"
    dummy = Line2D([], [], linestyle="None", color="none", label=ci_label)

    ax.set_title(event_label(event_index, event_ids), fontsize=14, pad=3)
    ax.set_xlabel("Distance (m)", fontsize=14)
    ax.set_ylabel("Depth (m)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10, pad=2)
    ax.grid(color="0.92", linewidth=0.6)

    handles = [post_handle, ellipse_handle, mean_handle, dummy]
    labels = [h.get_label() for h in handles[:-1]] + [ci_label]
    ax.legend(handles, labels, loc="upper right", frameon=True, fontsize=8.0,
              borderpad=0.25, handlelength=1.5, handletextpad=0.45)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    eid = event_ids[event_index] if event_ids is not None and event_index < len(event_ids) else str(event_index + 1)
    return {
        "source_index_0based": event_index,
        "source_index_1based": event_index + 1,
        "event_id": eid,
        "n_samples": int(sx.size),
        "n_samples_shown_in_region": int(sxp_show.size),
        "mean_file_x_m": mean_x,
        "mean_file_z_m": mean_z,
        "posterior_mean_x_m": post_mean_x,
        "posterior_mean_z_m": post_mean_z,
        "mean_file_vs_chain_mean_diff_m": mean_from_file_diff,
        "std_x_m": std_x,
        "std_z_m": std_z,
        "corr_xz": corr,
        "q025_x_m": float(qx025),
        "q16_x_m": float(qx16),
        "q50_x_m": float(qx50),
        "q84_x_m": float(qx84),
        "q975_x_m": float(qx975),
        "q025_z_m": float(qz025),
        "q16_z_m": float(qz16),
        "q50_z_m": float(qz50),
        "q84_z_m": float(qz84),
        "q975_z_m": float(qz975),
    }


def write_summary_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_event_posteriors(samples: np.ndarray, names: list[str], mean_geo: Geometry,
                          init_geo: Geometry | None, receiver_geo: Geometry | None,
                          out_dir: Path, dpi: int, max_samples: int,
                          event_ids: list[str] | None = None) -> list[dict[str, float | int | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260628)
    rows: list[dict[str, float | int | str]] = []

    for ev in range(mean_geo.ns):
        sx, sz = event_samples(samples, names, ev)
        if sx is None or sz is None:
            warnings.warn(f"Cannot find sx[{ev}] and sz[{ev}] in chain; skip event {ev + 1}")
            continue
        label_id = event_ids[ev] if event_ids is not None and ev < len(event_ids) else f"{ev + 1:03d}"
        safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(label_id))
        out_pdf = out_dir / f"Event_{ev + 1:03d}_id_{safe_id}_posterior.pdf"
        rows.append(plot_one_event(ev, sx, sz, mean_geo, init_geo, out_pdf, dpi, max_samples, event_ids, rng))

    plot_all_event_map(mean_geo, init_geo, receiver_geo, out_dir / "field_event_posterior_map.pdf", dpi, event_ids)
    write_summary_csv(out_dir / "field_event_posterior_summary.csv", rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot field-event posterior location distributions.")
    parser.add_argument("--root", type=Path, default=None,
                        help="Project root containing 01-initial, 02-inversion, and 03-output. Default: inferred.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Initial output directory. Default: ROOT/01-initial/output.")
    parser.add_argument("--result-dir", type=Path, default=None,
                        help="MCMC result directory containing chain.npz and mean.dat. Default: ROOT/03-output/output.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory. Default: RESULT_DIR/event_posterior.")
    parser.add_argument("--burnin", type=int, default=None,
                        help="Override burn-in. By default use burnin_eff stored in chain.npz.")
    parser.add_argument("--max-samples", type=int, default=2500,
                        help="Maximum posterior samples drawn in each event panel.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-initial", action="store_true",
                        help="Deprecated compatibility option. Initial locations are not drawn in this version.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_style()

    root = infer_project_root(args.root)
    input_dir = (args.input_dir or (root / "01-initial" / "output")).expanduser().resolve()
    result_dir = (args.result_dir or (root / "03-output" / "output")).expanduser().resolve()
    output_dir = (args.output_dir or (result_dir / "event_posterior")).expanduser().resolve()

    mean_path = result_dir / "mean.dat"
    if not mean_path.exists():
        raise FileNotFoundError(f"Cannot find posterior mean file: {mean_path}")
    mean_geo = read_mean_geometry(mean_path)

    receiver_geo = None
    geom_path = input_dir / "geometry.dat"
    if geom_path.exists():
        receiver_geo = read_field_geometry(geom_path)

    # Initial/grid-search locations are intentionally not loaded or plotted.
    init_geo = None

    event_ids = read_event_id_map(input_dir / "field_event_id_map.csv")
    samples, names, used_burnin = load_chain(result_dir, burnin=args.burnin, ns_hint=mean_geo.ns)
    if samples is None:
        return 1

    rows = plot_event_posteriors(
        samples=samples,
        names=names,
        mean_geo=mean_geo,
        init_geo=init_geo,
        receiver_geo=receiver_geo,
        out_dir=output_dir,
        dpi=args.dpi,
        max_samples=args.max_samples,
        event_ids=event_ids,
    )

    print(f"[EVENT_POSTERIOR] burnin used = {used_burnin}; posterior samples = {samples.shape[0]}")
    print(f"[EVENT_POSTERIOR] event panels written = {len(rows)}")
    print(f"[EVENT_POSTERIOR] output directory = {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
