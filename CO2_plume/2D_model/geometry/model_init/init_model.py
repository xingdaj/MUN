import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


# =========================
# Section property table
# =========================
SECTION_INFO = {
    1: dict(rho=2890, vp=5300, vs=3180, name="Rome Formation"),
    2: dict(rho=2710, vp=4500, vs=2650, name="Martinsburg Formation"),
    3: dict(rho=2400, vp=2800, vs=1650, name="Forkenobs and Brallier"),
    4: dict(rho=2660, vp=4000, vs=2400, name="Oriskany Sandstone"),
    5: dict(rho=2890, vp=5300, vs=3180, name="Rome Formation"),
    6: dict(rho=2795, vp=4800, vs=3000, name="Pulsak Thrust Sheet"),

    # --- MOD #1: make Section 7 identical to Section 4 (rho/vp/vs same), keep name as Injected CO2 ---
    7: dict(rho=2660, vp=4000, vs=2400, name="Injected CO2"),

    8: dict(rho=2675, vp=4100, vs=2460, name="Mississipian Age Formation"),
}


def build_property_field(reg_map, key: str):
    out = np.full(reg_map.shape, np.nan, dtype=float)
    for sec, info in SECTION_INFO.items():
        out[reg_map == sec] = float(info[key])
    return out


# -------------------- IO helpers --------------------
def resolve_workdir(workdir=None):
    return os.path.abspath(workdir) if workdir is not None else os.getcwd()


def locate_ascii_dir(workdir):
    cand = [
        os.path.join(workdir, "ascii"),
        os.path.join(workdir, "ASCII"),
        workdir,
    ]
    for d in cand:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "curve_1.txt")):
            return d
    raise FileNotFoundError(
        "Cannot locate curve files. Expected curve_1.txt under one of:\n"
        + "\n".join(cand)
    )


def read_curve_as_function(curve_path: str):
    """
    Read curve file robustly.

    Common formats:
      A) x  z  something
      B) x  y(=0)  z   (very common)
    We auto-detect: if column-2 has near-zero variation and col-3 exists, use col-3 as z.
    """
    arr = np.loadtxt(curve_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Curve file must have at least 2 columns. Got: {arr.shape}")

    x = arr[:, 0].astype(float)
    z_candidate = arr[:, 1].astype(float)

    # If col-2 is (almost) constant and a 3rd column exists, treat col-3 as depth z
    if arr.shape[1] >= 3:
        z3 = arr[:, 2].astype(float)
        if np.nanmax(z_candidate) - np.nanmin(z_candidate) < 1e-6 and (np.nanmax(z3) - np.nanmin(z3) > 1e-6):
            z = z3
        else:
            z = z_candidate
    else:
        z = z_candidate

    # Sort by x
    idx = np.argsort(x)
    x, z = x[idx], z[idx]

    return x, z


def interp_to_grid(xg, x, z):
    idx = np.argsort(x)
    x2, z2 = x[idx], z[idx]
    return np.interp(xg, x2, z2, left=np.nan, right=np.nan)


# -------------------- raster polyline draw --------------------
def draw_polyline_to_mask(mask, x, z, xmin, xmax, zmin, zmax, nx, nz):
    """
    Rasterize polyline into mask.
    """
    if xmax == xmin or zmax == zmin:
        return

    ix = ((x - xmin) / (xmax - xmin) * (nx - 1))
    iz = ((z - zmin) / (zmax - zmin) * (nz - 1))

    # remove NaN/inf
    ok = np.isfinite(ix) & np.isfinite(iz)
    if np.sum(ok) < 2:
        return
    ix = ix[ok].astype(int)
    iz = iz[ok].astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iz = np.clip(iz, 0, nz - 1)

    for k in range(len(ix) - 1):
        x0, z0 = ix[k], iz[k]
        x1, z1 = ix[k + 1], iz[k + 1]
        dx = abs(x1 - x0)
        dz = abs(z1 - z0)
        n = max(dx, dz, 1)
        xs = np.linspace(x0, x1, n + 1).astype(int)
        zs = np.linspace(z0, z1, n + 1).astype(int)
        mask[zs, xs] = True


def dilate_4(mask, iters=1):
    m = mask.copy()
    for _ in range(iters):
        up = np.zeros_like(m); up[1:, :] = m[:-1, :]
        dn = np.zeros_like(m); dn[:-1, :] = m[1:, :]
        lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
        rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
        m = m | up | dn | lf | rt
    return m


# -------------------- connected components --------------------
def label_components(free_mask):
    nz, nx = free_mask.shape
    labels = np.zeros((nz, nx), dtype=np.int32)
    cur = 0

    for r in range(nz):
        for c in range(nx):
            if (not free_mask[r, c]) or labels[r, c] != 0:
                continue
            cur += 1
            q = deque([(r, c)])
            labels[r, c] = cur
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < nz and 0 <= c2 < nx:
                        if free_mask[r2, c2] and labels[r2, c2] == 0:
                            labels[r2, c2] = cur
                            q.append((r2, c2))
    return labels, cur


def keep_largest_k(labels, nlab, k=8, min_area_frac=0.001):
    if nlab == 0:
        return labels * 0, []

    areas = np.zeros(nlab + 1, dtype=np.int64)
    for i in range(1, nlab + 1):
        areas[i] = np.sum(labels == i)

    tot = labels.size
    keep = [i for i in range(1, nlab + 1) if areas[i] >= int(min_area_frac * tot)]
    keep.sort(key=lambda i: areas[i], reverse=True)
    keep = keep[:k]

    out = labels.copy()
    mask_keep = np.isin(out, keep)
    out[~mask_keep] = 0
    return out, keep



# =========================
# Interface smoothing utilities (for adjoint/gradient stability)
# =========================
def _distance_to_mask_4(mask: np.ndarray) -> np.ndarray:
    """Approximate 4-neighbor distance-to-mask (in grid steps) using BFS."""
    nz, nx = mask.shape
    dist = np.full((nz, nx), -1, dtype=np.int32)
    q = deque()

    ys, xs = np.where(mask)
    for r, c in zip(ys.tolist(), xs.tolist()):
        dist[r, c] = 0
        q.append((r, c))

    if not q:
        return np.full((nz, nx), 10**9, dtype=np.int32)

    while q:
        r, c = q.popleft()
        d0 = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < nz and 0 <= c2 < nx and dist[r2, c2] < 0:
                dist[r2, c2] = d0 + 1
                q.append((r2, c2))
    return dist


def fill_nan_nearest_4(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs by nearest (4-neighbor) valid value using multi-source BFS."""
    a = np.asarray(arr, dtype=float)
    if np.all(np.isfinite(a)):
        return a

    nz, nx = a.shape
    out = a.copy()
    visited = np.isfinite(out)
    q = deque()

    ys, xs = np.where(visited)
    for r, c in zip(ys.tolist(), xs.tolist()):
        q.append((r, c))

    if not q:
        raise ValueError("All values are NaN; cannot fill.")

    while q:
        r, c = q.popleft()
        v = out[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < nz and 0 <= c2 < nx and (not visited[r2, c2]):
                out[r2, c2] = v
                visited[r2, c2] = True
                q.append((r2, c2))

    return out


def smooth_near_interfaces(prop: np.ndarray,
                           boundary_mask: np.ndarray,
                           dx: float,
                           dz: float,
                           tau_m: float = 40.0,
                           n_iter: int = 120,
                           band_m: float | None = None) -> np.ndarray:
    """Smooth a piecewise-constant property ONLY in a band around interfaces."""
    prop0 = np.asarray(prop, dtype=float)
    if not np.all(np.isfinite(prop0)):
        raise ValueError("Property contains NaN/Inf. Call fill_nan_nearest_4() first.")

    out = prop0.copy()

    if band_m is None:
        band_m = 3.0 * float(tau_m)

    step_m = float(min(abs(dx), abs(dz)))
    dist_steps = _distance_to_mask_4(boundary_mask)
    dist_m = dist_steps.astype(float) * step_m

    band = dist_m <= float(band_m)
    if not np.any(band):
        return out

    tau = float(max(tau_m, 1e-6))
    alpha = np.exp(-(dist_m / tau) ** 2)

    for _ in range(int(n_iter)):
        # edge-replicated neighbors (no wrap)
        up = np.vstack([out[0:1, :], out[:-1, :]])
        dn = np.vstack([out[1:, :], out[-1:, :]])
        lf = np.hstack([out[:, 0:1], out[:, :-1]])
        rt = np.hstack([out[:, 1:], out[:, -1:]])
        avg4 = 0.25 * (up + dn + lf + rt)
        out[band] = (1.0 - alpha[band]) * prop0[band] + alpha[band] * avg4[band]

    return out

# -------------------- robust label (anti-overlap) --------------------
def place_value_labels(fig, ax, reg_map, xg, Z_plot, values_by_section,
                       fontsize=20, min_sep_px=95, n_iter=120,
                       fmt="{:.0f}", prefix="", suffix=""):
    nsec = int(reg_map.max())
    if nsec <= 0:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    txt_effect = [pe.withStroke(linewidth=5, foreground="white")]

    pts_data = []
    texts = []

    for sec in range(1, nsec + 1):
        ys, xs = np.where(reg_map == sec)
        if len(xs) == 0:
            continue
        ix = int(np.median(xs))
        iz = int(np.median(ys))

        pts_data.append([float(xg[ix]), float(Z_plot[iz, ix])])

        v = values_by_section.get(sec, np.nan)
        if np.isnan(v):
            texts.append(f"Sec {sec}")
        else:
            texts.append(f"{prefix}{fmt.format(float(v))}{suffix}")

    if len(pts_data) == 0:
        return

    pts_data = np.array(pts_data, dtype=float)
    trans = ax.transData
    inv = trans.inverted()
    pts_px = trans.transform(pts_data)

    n = pts_px.shape[0]
    for _ in range(n_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts_px[j, 0] - pts_px[i, 0]
                dy = pts_px[j, 1] - pts_px[i, 1]
                dist = (dx * dx + dy * dy) ** 0.5 + 1e-12
                if dist < min_sep_px:
                    push = 0.5 * (min_sep_px - dist)
                    ux, uy = dx / dist, dy / dist
                    pts_px[i, 0] -= push * ux
                    pts_px[i, 1] -= push * uy
                    pts_px[j, 0] += push * ux
                    pts_px[j, 1] += push * uy
                    moved = True
        if not moved:
            break

    bbox = ax.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    pad = 12
    pts_px[:, 0] = np.clip(pts_px[:, 0], x0 + pad, x1 - pad)
    pts_px[:, 1] = np.clip(pts_px[:, 1], y0 + pad, y1 - pad)

    pts_new = inv.transform(pts_px)
    for i in range(n):
        x, y = pts_new[i]
        ax.text(
            x, y, texts[i],
            color="black",
            fontsize=fontsize,
            ha="center", va="center",
            path_effects=txt_effect,
            zorder=60,
            clip_on=True,
        )


# ==========================================================
# PML padding MUST FOLLOW "smooth" ARRAY DIRECTION:
# smooth 的“地表(0m)”在数组底部，因此 top-pad 要加在 axis=0 的末尾
# ==========================================================
def pad_for_pml_like_smooth(arr: np.ndarray, pad_x: int, pad_z_top: int) -> np.ndarray:
    """
    arr: (nz, nx)
    pad_x: pad on left/right (axis=1)
    pad_z_top: pad on TOP (z<0) following smooth orientation => append rows at bottom (axis=0 end)
    mode='edge' to replicate boundary values
    """
    if pad_x < 0 or pad_z_top < 0:
        raise ValueError("pad sizes must be non-negative")
    out = arr
    if pad_z_top > 0:
        out = np.pad(out, ((0, pad_z_top), (0, 0)), mode="edge")  # <-- KEY CHANGE
    if pad_x > 0:
        out = np.pad(out, ((0, 0), (pad_x, pad_x)), mode="edge")
    return out


def plot_one_pml(prop_pml, title, unit_label, outname,
                 xmin_pml, xmax_pml, zmin_pml, zmax_pml,
                 xmin0, xmax0, zmin0, zmax0,
                 pad_m, workdir, cmap="jet"):
    """
    PML 图也按 smooth 的数组方向来配 z：
      - 数组第 0 行对应深部(≈4000)
      - 数组最后几行对应浅部(≈0)以及 PML 顶部(-300)
    因此：zg_pml 必须从 zmax_pml -> zmin_pml（递减）
    """
    nzp, nxp = prop_pml.shape
    xg_pml = np.linspace(xmin_pml, xmax_pml, nxp)

    # <-- KEY CHANGE: z axis follows smooth array direction
    zg_pml = np.linspace(zmax_pml, zmin_pml, nzp)  # 4000 ... -300
    Xp, Zp = np.meshgrid(xg_pml, zg_pml)

    fig, ax = plt.subplots(figsize=(14, 9))
    m = ax.pcolormesh(Xp, Zp, prop_pml, shading="auto", cmap=cmap)

    # Shade PML zones (left/right/top)
    ax.axvspan(xmin_pml, xmin0, alpha=0.18, color="gray", lw=0)
    ax.axvspan(xmax0, xmax_pml, alpha=0.18, color="gray", lw=0)
    ax.axhspan(zmin_pml, zmin0, alpha=0.18, color="gray", lw=0)

    # Original-domain box
    ax.plot([xmin0, xmax0, xmax0, xmin0, xmin0],
            [zmin0, zmin0, zmax0, zmax0, zmin0],
            "w--", lw=2.4, alpha=0.95)

    ax.text(xmin_pml + 0.02 * (xmax_pml - xmin_pml),
            zmin0 + 0.02 * (zmax0 - zmin0),
            f"PML pad = {pad_m:.0f} m (left/right/top)",
            color="white", fontsize=16,
            ha="left", va="top",
            path_effects=[pe.withStroke(linewidth=4, foreground="black")])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)

    # Keep the same visual convention: 0 at top, depth increasing downward
    ax.set_ylim(zmax0, zmin_pml)

    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(unit_label, fontsize=18)
    cb.ax.tick_params(labelsize=16)

    plt.tight_layout()
    outpath = os.path.join(workdir, outname)
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", outpath)


# -------------------- main --------------------


# ==========================================================
# NEW: crop study-area data from the PRE-PML model and plot it
# ==========================================================
def crop_and_plot_study_area_from_pml(
    prop_pml_save,
    xmin_pml, xmax_pml, zmin_pml, zmax_pml,
    x1, x2, z1, z2,
    workdir,
    npy_name="vp_study_area.npy",
    meta_name="vp_study_area_meta.json",
    fig_name="property_vp_study_area.png",
    title="Vp in study area",
    unit_label="Vp (m/s)",
    cmap="jet"
):
    """
    prop_pml_save:
        saved array orientation, same as vp_pre_pml.npy in this script.
        Row 0 corresponds to z=zmax_pml (deep), row end corresponds to z=zmin_pml (shallow/top).

    crop window:
        X in [x1, x2], Z in [z1, z2]
    """

    nz, nx = prop_pml_save.shape

    xg = np.linspace(xmin_pml, xmax_pml, nx)
    zg = np.linspace(zmax_pml, zmin_pml, nz)   # match vp_pre_pml saved orientation

    ix = np.where((xg >= x1) & (xg <= x2))[0]
    iz = np.where((zg >= z1) & (zg <= z2))[0]

    if len(ix) == 0 or len(iz) == 0:
        raise ValueError(
            f"No grid points found inside study area: "
            f"X[{x1},{x2}], Z[{z1},{z2}]"
        )

    prop_crop = prop_pml_save[np.ix_(iz, ix)]
    x_crop = xg[ix]
    z_crop = zg[iz]

    np.save(os.path.join(workdir, npy_name), prop_crop)

    meta = {
        "source_array": "vp_pre_pml.npy",
        "crop_window": {
            "xmin": float(x1),
            "xmax": float(x2),
            "zmin": float(z1),
            "zmax": float(z2)
        },
        "xmin": float(x_crop[0]),
        "xmax": float(x_crop[-1]),
        "zmin": float(np.min(z_crop)),
        "zmax": float(np.max(z_crop)),
        "nx": int(len(x_crop)),
        "nz": int(len(z_crop)),
        "dx": float(xg[1] - xg[0]) if len(xg) > 1 else None,
        "dz": float(abs(zg[1] - zg[0])) if len(zg) > 1 else None,
        "array_layout": "cropped_array[nz, nx], saved in the same vertical orientation as vp_pre_pml.npy"
    }
    with open(os.path.join(workdir, meta_name), "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Saved:", os.path.join(workdir, npy_name))
    print("[OK] Saved:", os.path.join(workdir, meta_name))

    Xc, Zc = np.meshgrid(x_crop, z_crop)

    fig, ax = plt.subplots(figsize=(10, 7))
    m = ax.pcolormesh(Xc, Zc, prop_crop, shading="auto", cmap=cmap)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(title)
    ax.set_xlim(x1, x2)
    ax.set_ylim(z2, z1)
    ax.set_aspect('equal', adjustable='box')

    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(unit_label, fontsize=18)
    cb.ax.tick_params(labelsize=16)

    plt.tight_layout()
    outpath = os.path.join(workdir, fig_name)
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", outpath)


def main(workdir=None):
    workdir = resolve_workdir(workdir)
    ascii_dir = locate_ascii_dir(workdir)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    # ----- read curves -----
    curves = {}
    for i in range(1, 8):
        p = os.path.join(ascii_dir, f"curve_{i}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing curve file: {p}")
        curves[i] = read_curve_as_function(p)

    xs_all = np.concatenate([curves[i][0] for i in range(1, 8)])
    zs_all = np.concatenate([curves[i][1] for i in range(1, 8)])

    xmin, xmax = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    zmin, zmax = float(np.nanmin(zs_all)), float(np.nanmax(zs_all))

    if not np.isfinite(zmin) or not np.isfinite(zmax) or abs(zmax - zmin) < 1e-6:
        raise ValueError(
            "Your curve Z values appear constant or invalid.\n"
            "Likely your curve files are (x,y,z) and z is in 3rd column.\n"
            "This script already auto-detects that, so please check curve_i.txt columns.\n"
            f"Detected zmin={zmin}, zmax={zmax} from: {ascii_dir}"
        )

    nx, nz = 2000, 800
    xg = np.linspace(xmin, xmax, nx)
    zg = np.linspace(zmin, zmax, nz)
    dz_grid = float(abs(zg[1] - zg[0]))

    X, Z = np.meshgrid(xg, zg)

    # curves on x-grid
    zc = {i: interp_to_grid(xg, curves[i][0], curves[i][1]) for i in range(1, 8)}

    best_reg = None
    best_tol = None

    # ----- build boundary mask & region labels with adaptive tolerance -----
    for tol in [1.5 * dz_grid, 2.0 * dz_grid, 2.5 * dz_grid, 3.0 * dz_grid, 3.5 * dz_grid, 4.0 * dz_grid]:
        boundary = np.zeros((nz, nx), dtype=bool)

        for i in range(1, 8):
            zi = zc[i]
            defined = np.isfinite(zi)
            if not np.any(defined):
                continue
            boundary[:, defined] |= (np.abs(Z[:, defined] - zi[None, defined]) <= tol)

        for i in range(1, 8):
            x, z = curves[i]
            draw_polyline_to_mask(boundary, x, z, xmin, xmax, zmin, zmax, nx, nz)

        boundary = dilate_4(boundary, iters=2)
        free = ~boundary

        labels, nlab = label_components(free)
        labels2, kept = keep_largest_k(labels, nlab, k=8, min_area_frac=0.001)

        kept = kept[:8]
        region_map = {old: new for new, old in enumerate(kept, start=1)}
        reg = np.zeros_like(labels2, dtype=np.int32)
        for old, new in region_map.items():
            reg[labels2 == old] = new

        nreg = int(reg.max())
        print(f"[TRY] tol={tol/dz_grid:.1f}*dz -> regions={nreg}")

        if nreg == 8:
            best_reg, best_tol = reg, tol
            break

        if best_reg is None or nreg > int(best_reg.max()):
            best_reg, best_tol = reg, tol

    print(f"[INFO] Final regions={int(best_reg.max())}, tol={best_tol/dz_grid:.1f}*dz")

    # Rebuild a boundary mask at the selected tolerance (best_tol).
    boundary_best = np.zeros((nz, nx), dtype=bool)
    for i in range(1, 8):
        zi = zc[i]
        defined = np.isfinite(zi)
        if not np.any(defined):
            continue
        boundary_best[:, defined] |= (np.abs(Z[:, defined] - zi[None, defined]) <= best_tol)
    for i in range(1, 8):
        x, z = curves[i]
        draw_polyline_to_mask(boundary_best, x, z, xmin, xmax, zmin, zmax, nx, nz)
    boundary_best = dilate_4(boundary_best, iters=2)

    # ----- vertical stretch to 4000 m for plotting -----
    TARGET_MAX_DEPTH = 4000.0
    depth_range = float(zmax - zmin)

    D = Z - zmin
    d7 = zc[7] - zmin
    d7_safe = np.where(np.isfinite(d7), d7, depth_range)
    D7 = d7_safe[None, :]

    denom = (depth_range - d7_safe)
    denom = np.where(denom <= 0, 1.0, denom)
    s = (TARGET_MAX_DEPTH - d7_safe) / denom
    s = np.where(s <= 0, 1.0, s)
    S = s[None, :]

    D_stretch = np.where(D <= D7, D, D7 + (D - D7) * S)
    Z_plot = TARGET_MAX_DEPTH - D_stretch

    # ----- build properties (piecewise-constant by region) -----
    rho = build_property_field(best_reg, "rho")
    vp  = build_property_field(best_reg, "vp")
    vs  = build_property_field(best_reg, "vs")

    # Fill NaNs created by the boundary corridor (reg==0)
    rho = fill_nan_nearest_4(rho)
    vp  = fill_nan_nearest_4(vp)
    vs  = fill_nan_nearest_4(vs)

    # Smooth interfaces
    TAU_LAYER = float(os.environ.get("TAU_LAYER", "40.0"))
    SMOOTH_ITERS = int(os.environ.get("SMOOTH_ITERS", "120"))
    dx_grid = float(abs(xg[1] - xg[0]))
    dz_grid2 = float(abs(zg[1] - zg[0]))
    vp_smooth  = smooth_near_interfaces(vp,  boundary_best, dx_grid, dz_grid2, tau_m=TAU_LAYER, n_iter=SMOOTH_ITERS)
    vs_smooth  = smooth_near_interfaces(vs,  boundary_best, dx_grid, dz_grid2, tau_m=TAU_LAYER, n_iter=SMOOTH_ITERS)
    rho_smooth = smooth_near_interfaces(rho, boundary_best, dx_grid, dz_grid2, tau_m=TAU_LAYER, n_iter=SMOOTH_ITERS)

    # =========================
    # MOD #2: Save "pre" outputs
    # =========================
    # Keep piecewise reference
    np.save(os.path.join(workdir, "vp_piecewise.npy"), vp)
    np.save(os.path.join(workdir, "vs_piecewise.npy"), vs)
    np.save(os.path.join(workdir, "rho_piecewise.npy"), rho)

    # Save smoothed fields as *_pre.npy (downstream background/pre model)
    np.save(os.path.join(workdir, "vp_pre.npy"),  vp_smooth)
    np.save(os.path.join(workdir, "vs_pre.npy"),  vs_smooth)
    np.save(os.path.join(workdir, "rho_pre.npy"), rho_smooth)

    # --- Save coordinate metadata for vp_pre.npy ---
    vp_meta = {
        "xmin": float(xmin), "xmax": float(xmax), "nx": int(nx),
        "zmin": float(zmin), "zmax": float(zmax), "nz": int(nz),
        "array_layout": "vp_pre[nz, nx] with z increasing from zmin to zmax, x increasing from xmin to xmax",
        "note": "Use these bounds to build xg=np.linspace(xmin,xmax,nx) and zg=np.linspace(zmin,zmax,nz) when projecting onto another mesh.",
    }
    meta_path = os.path.join(workdir, "vp_pre_meta.json")
    with open(meta_path, "w") as f:
        json.dump(vp_meta, f, indent=2)
    print(f"[OK] Saved meta: {meta_path}")

    np.save(os.path.join(workdir, "reg_map.npy"), best_reg)
    print(f"[OK] Saved numpy fields (TAU_LAYER={TAU_LAYER} m, SMOOTH_ITERS={SMOOTH_ITERS}): "
          "vp_pre.npy / vs_pre.npy / rho_pre.npy")


    # =========================
    # PML (fixed to match smooth orientation for BOTH plot & save)
    # =========================
    PAD_M = float(os.environ.get("PML_PAD_M", "300.0"))
    pad_x = int(np.round(PAD_M / dx_grid))
    pad_z_top = int(np.round(PAD_M / dz_grid2))

    vp_pre_pml  = pad_for_pml_like_smooth(vp_smooth,  pad_x=pad_x, pad_z_top=pad_z_top)
    vs_pre_pml  = pad_for_pml_like_smooth(vs_smooth,  pad_x=pad_x, pad_z_top=pad_z_top)
    rho_pre_pml = pad_for_pml_like_smooth(rho_smooth, pad_x=pad_x, pad_z_top=pad_z_top)

    # coordinates follow smooth display domain: Z=[0,4000] and padded top Z=[-300,0]
    xmin0, xmax0 = float(xmin), float(xmax)
    zmin0, zmax0 = 0.0, float(TARGET_MAX_DEPTH)

    xmin_pml = float(xmin0 - PAD_M)
    xmax_pml = float(xmax0 + PAD_M)
    zmin_pml = float(zmin0 - PAD_M)
    zmax_pml = float(zmax0)

    nx_pml = int(vp_pre_pml.shape[1])
    nz_pml = int(vp_pre_pml.shape[0])

    np.save(os.path.join(workdir, "vp_pre_pml.npy"), vp_pre_pml)
    np.save(os.path.join(workdir, "vs_pre_pml.npy"), vs_pre_pml)
    np.save(os.path.join(workdir, "rho_pre_pml.npy"), rho_pre_pml)

    vp_pml_meta = {
        "xmin": xmin_pml, "xmax": xmax_pml, "nx": nx_pml,
        "zmin": zmin_pml, "zmax": zmax_pml, "nz": nz_pml,
        "dx": float(dx_grid), "dz": float(dz_grid2),
        "pad_m": float(PAD_M),
        "pad_x_cells": int(pad_x),
        "pad_z_top_cells": int(pad_z_top),
        "padding_sides": {"left": True, "right": True, "top": True, "bottom": False},
        "padding_mode": "edge (replicate boundary values)",
        "array_layout": "vp_pre_pml[nz_pml, nx_pml] with SAME vertical orientation as vp_smooth",
        "note": "Both plotting and saved arrays follow the SAME vertical orientation as smooth."
    }
    meta_pml_path = os.path.join(workdir, "vp_pre_pml_meta.json")
    with open(meta_pml_path, "w") as f:
        json.dump(vp_pml_meta, f, indent=2)
    print(f"[OK] Saved PML-padded models (PAD_M={PAD_M} m): vp_pre_pml.npy / vs_pre_pml.npy / rho_pre_pml.npy")
    print(f"[OK] Saved meta: {meta_pml_path}")

    # PML images
    plot_one_pml(
        vp_pre_pml,
        "Vp (pre/smoothed, PML-padded)",
        "Vp (m/s)",
        "property_vp_pre_pml.png",
        xmin_pml, xmax_pml, zmin_pml, zmax_pml,
        xmin0, xmax0, zmin0, zmax0,
        PAD_M, workdir
    )
    plot_one_pml(
        vs_pre_pml,
        "Vs (pre/smoothed, PML-padded)",
        "Vs (m/s)",
        "property_vs_pre_pml.png",
        xmin_pml, xmax_pml, zmin_pml, zmax_pml,
        xmin0, xmax0, zmin0, zmax0,
        PAD_M, workdir
    )
    plot_one_pml(
        rho_pre_pml,
        "Density (pre/smoothed, PML-padded)",
        "Density (kg/m$^3$)",
        "property_density_pre_pml.png",
        xmin_pml, xmax_pml, zmin_pml, zmax_pml,
        xmin0, xmax0, zmin0, zmax0,
        PAD_M, workdir
    )


    # ==========================================================
    # NEW: save and plot study area from PRE-PML model
    # Research area: X[00,6500], Z[1000,3000]
    # ==========================================================
    crop_and_plot_study_area_from_pml(
        prop_pml_save=vp_pre_pml,
        xmin_pml=xmin_pml,
        xmax_pml=xmax_pml,
        zmin_pml=zmin_pml,
        zmax_pml=zmax_pml,
        x1=0.0,
        x2=6500.0,
        z1=1000.0,
        z2=3200.0,
        workdir=workdir,
        npy_name="vp_pre_pml_study_area.npy",
        meta_name="vp_pre_pml_study_area_meta.json",
        fig_name="property_vp_pre_pml_study_area.png",
        title="Vp (study area: X[1000,5000], Z[1000,3000])",
        unit_label="Vp (m/s)",
        cmap="jet"
    )

    def plot_one(prop, title, unit_label, outname, key):
        fig, ax = plt.subplots(figsize=(14, 9))
        m = ax.pcolormesh(X, Z_plot, prop, shading="auto", cmap="jet")

        # boundaries
        for i in range(1, 8):
            x, z = curves[i]
            d = z - zmin
            d7_i = np.interp(x, xg, d7_safe)
            denom_i = (depth_range - d7_i)
            denom_i = np.where(denom_i <= 0, 1.0, denom_i)
            s_i = (TARGET_MAX_DEPTH - d7_i) / denom_i
            s_i = np.where(s_i <= 0, 1.0, s_i)
            d_stretch = np.where(d <= d7_i, d, d7_i + (d - d7_i) * s_i)
            ax.plot(x, TARGET_MAX_DEPTH - d_stretch, "k-", lw=5.0)

        # receivers & sources
        rx_x = np.arange(300.0, 6000.0 + 1e-9, 100.0)
        rx_z = np.zeros_like(rx_x)
        src_x = np.array([1000.0, 3000.0, 5000.0], dtype=float)
        src_z = np.array([10.0, 10.0, 10.0], dtype=float)

        ax.plot(rx_x, rx_z, marker="v", linestyle="None",
                markersize=16.0, markeredgewidth=0.8, color="green",
                zorder=80, clip_on=False)
        ax.plot(src_x, src_z, marker="*", linestyle="None",
                markersize=18.0, markeredgewidth=1.2, color="red",
                zorder=90, clip_on=False)

        legend_handles = [
            Line2D([0], [0], marker="*", color="red", linestyle="None", markersize=14, label="Source"),
            Line2D([0], [0], marker="v", color="green", linestyle="None", markersize=12, label="Receiver"),
        ]
        ax.legend(handles=legend_handles, loc="upper left",
                  frameon=True, framealpha=0.85, facecolor="white",
                  edgecolor="black", fontsize=14)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(title)

        labels = np.arange(0.0, TARGET_MAX_DEPTH + 500.0, 500.0)
        ax.set_yticks(TARGET_MAX_DEPTH - labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.0f}"))
        ax.set_ylim(TARGET_MAX_DEPTH, 0.0)

        cb = fig.colorbar(m, ax=ax, pad=0.02)
        cb.set_label(unit_label, fontsize=18)
        cb.ax.tick_params(labelsize=16)

        values_by_section = {sec: float(SECTION_INFO[sec][key]) for sec in SECTION_INFO.keys()}
        place_value_labels(
            fig, ax, best_reg, xg, Z_plot,
            values_by_section=values_by_section,
            fontsize=22, min_sep_px=95, n_iter=140,
            fmt="{:.0f}", prefix="", suffix=""
        )

        plt.tight_layout()
        outpath = os.path.join(workdir, outname)
        fig.savefig(outpath, dpi=600)
        plt.close(fig)
        print("[OK] Saved:", outpath)

    # Smoothed (for inversion / "pre")
    plot_one(rho_smooth, "Density (pre/smoothed)", "Density (kg/m$^3$)", "property_density_pre.png", key="rho")
    plot_one(vp_smooth,  "Vp (pre/smoothed)", "Vp (m/s)", "property_vp_pre.png", key="vp")
    plot_one(vs_smooth,  "Vs (pre/smoothed)", "Vs (m/s)", "property_vs_pre.png", key="vs")

    # Piecewise-constant (reference)
    plot_one(rho, "Density (piecewise)", "Density (kg/m$^3$)", "property_density_piecewise.png", key="rho")
    plot_one(vp,  "Vp (piecewise)", "Vp (m/s)", "property_vp_piecewise.png", key="vp")
    plot_one(vs,  "Vs (piecewise)", "Vs (m/s)", "property_vs_piecewise.png", key="vs")


if __name__ == "__main__":
    main(None)