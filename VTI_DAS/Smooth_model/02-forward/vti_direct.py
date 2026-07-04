#!/usr/bin/env python3
"""
Direct-wave qx two-point ray tracing in layered VTI media, with strict checks.

Compatible with the current direct.py generator: default input directory is ../01-input/output and default output directory is ../03-output and all numeric outputs use high precision.

This diagnostic/MCMC-ready version is designed for synthetic-test validation before MCMC:
  1. no silent clipping of non-physical VTI slowness surfaces;
  2. final qx is re-evaluated after the last Newton update;
  3. qx convergence is checked for every source-receiver-wave pair;
  4. diagnostics.dat records qx error, niter, sum(layer_dx)-H, and status;
  5. input_summary.dat/input_geometry_echo.dat/input_velocity_echo.dat echo all input parameters;
  6. layer_contributions.dat and iteration_detailed.dat record intermediate quantities;
  7. same-depth horizontal paths are handled by a direct horizontal-velocity branch.

This version does NOT generate nobs.dat. It only checks the deterministic forward run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import numpy as np

WAVES = {1: "qP", 2: "qSV", 3: "qSH"}
DEFAULT_STOP = 1.0e-6
DEFAULT_MAX_ITER = 20

ROUND_TOL = 1.0e-10
MIN_POSITIVE = 1.0e-300
# Numerical safeguards for near-critical rays. When pz^2 approaches zero,
# analytical second/third derivatives become singular. The solver then falls
# back to bracketed bisection/Newton instead of evaluating unsafe derivatives.
CRITICAL_G0 = 1.0e-24
QX_UPPER_LIMIT = 1.0e8
BRACKET_EXPAND_ITERS = 80
ROBUST_SOLVE_MIN_ITERS = 80
OUT_FLOAT_FMT = ".16e"  # high precision for all forward outputs


@dataclass
class VTIModel:
    dep: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    epsilon: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray

    @property
    def nlayer(self) -> int:
        return len(self.dep)


def _to_float_tokens(path: str | Path) -> list[float]:
    vals: list[float] = []
    for tok in Path(path).read_text(encoding="utf-8", errors="ignore").replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def read_geometry(path: str | Path):
    vals = _to_float_tokens(path)
    if len(vals) < 2:
        raise ValueError(f"Cannot read geometry from {path}")
    i = 0
    ns = int(vals[i]); i += 1
    sx, sz = [], []
    for _ in range(ns):
        sx.append(vals[i]); sz.append(vals[i + 1]); i += 2
    nr = int(vals[i]); i += 1
    rx, rz = [], []
    for _ in range(nr):
        rx.append(vals[i]); rz.append(vals[i + 1]); i += 2
    return np.asarray(sx, dtype=float), np.asarray(sz, dtype=float), np.asarray(rx, dtype=float), np.asarray(rz, dtype=float)


def validate_model(model: VTIModel) -> None:
    if model.nlayer < 1:
        raise ValueError("model must have at least one layer")
    if not np.all(np.isfinite(model.dep)) or not np.all(np.diff(model.dep) > 0.0):
        raise ValueError("Layer depths dep must be finite and strictly increasing")
    if np.any(~np.isfinite(model.alpha)) or np.any(model.alpha <= 0.0):
        raise ValueError("alpha must be finite and positive")
    if np.any(~np.isfinite(model.beta)) or np.any(model.beta <= 0.0):
        raise ValueError("beta must be finite and positive")
    if np.any(~np.isfinite(model.epsilon)) or np.any(~np.isfinite(model.gamma)) or np.any(~np.isfinite(model.delta)):
        raise ValueError("epsilon/gamma/delta must be finite")
    if np.any(1.0 + 2.0 * model.epsilon <= 0.0):
        raise ValueError("qP pmax is invalid because 1 + 2*epsilon <= 0")
    if np.any(1.0 + 2.0 * model.gamma <= 0.0):
        raise ValueError("qSH pmax is invalid because 1 + 2*gamma <= 0")


def read_velocity(path: str | Path) -> VTIModel:
    vals = _to_float_tokens(path)
    if len(vals) < 7:
        raise ValueError(f"Cannot read VTI model from {path}")
    nlayer = int(vals[0])
    arr = np.asarray(vals[1:1 + 6 * nlayer], dtype=float).reshape(nlayer, 6)
    model = VTIModel(
        dep=arr[:, 0].copy(), alpha=arr[:, 1].copy(), beta=arr[:, 2].copy(),
        epsilon=arr[:, 3].copy(), gamma=arr[:, 4].copy(), delta=arr[:, 5].copy(),
    )
    validate_model(model)
    return model


def read_control(path: str | Path, stop_override: float | None = None, max_iter_override: int | None = None):
    """Read control.dat robustly for debugging.

    The original MATLAB folders may contain different control.dat conventions.
    Some files begin with flags such as 0/1 before the numerical tolerance.
    For this diagnostic forward code, we need a strictly positive offset
    tolerance and a positive max-iteration count.

    Priority:
      1) command-line overrides, if provided;
      2) first two values if they look like [positive stop, positive max_iter];
      3) fallback: first positive value < 1 as stop, first positive integer as max_iter;
      4) final defaults DEFAULT_STOP and DEFAULT_MAX_ITER.
    """
    vals = _to_float_tokens(path)

    raw_stop = float(vals[0]) if len(vals) >= 1 else float('nan')
    raw_max_iter = int(vals[1]) if len(vals) >= 2 and float(vals[1]).is_integer() else None

    if stop_override is not None:
        stop = float(stop_override)
        stop_source = "command-line --stop"
    elif len(vals) >= 1 and vals[0] > 0.0 and math.isfinite(vals[0]):
        stop = float(vals[0])
        stop_source = "control.dat value[0]"
    else:
        # Search for a tolerance-looking positive number. This handles
        # old control.dat layouts with leading flags such as 0/1.
        candidates = [float(v) for v in vals if math.isfinite(float(v)) and 0.0 < float(v) < 1.0]
        if candidates:
            stop = candidates[0]
            stop_source = "control.dat first positive value < 1"
        else:
            stop = DEFAULT_STOP
            stop_source = "DEFAULT_STOP because control.dat stop was not positive"

    if max_iter_override is not None:
        max_iter = int(max_iter_override)
        max_iter_source = "command-line --max-iter"
    elif len(vals) >= 2 and vals[1] > 0.0 and float(vals[1]).is_integer():
        max_iter = int(vals[1])
        max_iter_source = "control.dat value[1]"
    else:
        int_candidates = [int(v) for v in vals if math.isfinite(float(v)) and float(v).is_integer() and int(v) > 0]
        if int_candidates:
            max_iter = int_candidates[0]
            max_iter_source = "control.dat first positive integer"
        else:
            max_iter = DEFAULT_MAX_ITER
            max_iter_source = "DEFAULT_MAX_ITER"

    if stop <= 0.0 or not math.isfinite(stop):
        raise ValueError(f"Invalid stop tolerance after parsing control.dat: {stop}; raw first value={raw_stop}")
    if max_iter <= 0:
        raise ValueError(f"Invalid max_iter after parsing control.dat: {max_iter}; raw second value={raw_max_iter}")

    if len(vals) == 0:
        print(f"[WARN] control.dat is empty or unreadable; using stop={stop:g}, max_iter={max_iter}", flush=True)
    elif not (len(vals) >= 2 and vals[0] > 0.0 and vals[1] > 0.0):
        print(
            f"[WARN] control.dat first values do not look like [positive stop, positive max_iter]. "
            f"raw_values={vals[:10]}; using stop={stop:g} ({stop_source}), "
            f"max_iter={max_iter} ({max_iter_source}).",
            flush=True,
        )

    return stop, max_iter, vals, stop_source, max_iter_source


def direct_layer_thickness(z1: float, z2: float, model: VTIModel) -> np.ndarray:
    top_z, bot_z = sorted((float(z1), float(z2)))
    out = np.zeros(model.nlayer, dtype=float)
    for k in range(model.nlayer):
        layer_top = model.dep[k]
        layer_bot = model.dep[k + 1] if k + 1 < model.nlayer else bot_z
        out[k] = max(0.0, min(bot_z, layer_bot) - max(top_z, layer_top))
    return out


def layer_index_at_depth(z: float, model: VTIModel) -> int:
    """Return the layer containing depth z.

    Convention: dep[k] <= z < dep[k+1]. If z is exactly on an
    interface, the lower layer is used. This matches np.searchsorted(...,
    side="right") and avoids assigning an interface point to the layer
    above it.
    """
    z = float(z)
    if not math.isfinite(z):
        raise ValueError(f"non-finite depth: {z}")
    k = int(np.searchsorted(model.dep, z, side="right") - 1)
    return max(0, min(k, model.nlayer - 1))


def horizontal_velocity_for_wave(ip: int, k: int, model: VTIModel) -> float:
    """Horizontal phase/group velocity for a same-depth direct horizontal path.

    For a homogeneous VTI layer, pure horizontal propagation has the limiting
    horizontal velocities used by pmax_for_wave():
      qP  : alpha * sqrt(1 + 2*epsilon)
      qSV : beta
      qSH : beta * sqrt(1 + 2*gamma)
    This branch is only used when total vertical path thickness is zero.
    """
    if ip == 1:
        v = model.alpha[k] * math.sqrt(1.0 + 2.0 * model.epsilon[k])
    elif ip == 2:
        v = model.beta[k]
    elif ip == 3:
        v = model.beta[k] * math.sqrt(1.0 + 2.0 * model.gamma[k])
    else:
        raise ValueError("ip must be 1(qP), 2(qSV), or 3(qSH)")
    if not math.isfinite(v) or v <= 0.0:
        raise FloatingPointError(f"invalid horizontal velocity for {WAVES[ip]} in layer {k}: {v}")
    return float(v)


def horizontal_direct_result(ip: int, H: float, z: float, model: VTIModel, max_iter: int) -> dict:
    """Return a result row for a same-depth horizontal direct path.

    The qx formulation parameterizes px = qx*pmin/sqrt(1+qx^2), so a
    perfectly horizontal path corresponds to the limit qx -> infinity and
    px -> pmax of the current layer. To keep output finite and easy to read,
    qx is written as 1e30 for H > 0. The physically important values here are
    px and ttime.
    """
    k = layer_index_at_depth(z, model)
    v_h = horizontal_velocity_for_wave(ip, k, model)
    px = 1.0 / v_h if H > 0.0 else 0.0
    qx = 1.0e30 if H > 0.0 else 0.0
    ttime = float(H) / v_h if H > 0.0 else 0.0

    Z = np.zeros(model.nlayer, dtype=float)
    layer_dx = np.zeros(model.nlayer, dtype=float)
    layer_dt = np.zeros(model.nlayer, dtype=float)
    # Use zeros for inactive layers instead of NaN.
    # This keeps layer_contributions.dat fully finite for validation/MCMC.
    # For horizontal paths, only the source/receiver layer has physical values;
    # all other layers remain 0.0 and should be interpreted as inactive.
    layer_pz = np.zeros(model.nlayer)
    layer_vz = np.zeros(model.nlayer)
    layer_g0 = np.zeros(model.nlayer)
    layer_g1 = np.zeros(model.nlayer)

    layer_dx[k] = float(H)
    layer_dt[k] = ttime
    layer_pz[k] = 0.0
    layer_vz[k] = 0.0
    layer_g0[k] = 0.0
    # For a pure horizontal limiting path, g1 is not used by the forward output.
    # Keep it finite for downstream validation/MCMC.
    layer_g1[k] = 0.0

    history = [(float(qx), 0.0)]
    if len(history) < max_iter:
        history.extend([(0.0, 0.0)] * (max_iter - len(history)))
    elif len(history) > max_iter:
        history = history[:max_iter]

    history_detail = [{
        "qx": float(qx), "px": float(px), "Xcal": float(H), "f": 0.0,
        "df": 0.0, "d2f": 0.0, "disc": 0.0,
        "chosen_qx": float(qx), "chosen_f": 0.0,
    }]

    return {
        "wave": WAVES[ip], "qx": qx, "px": px, "ttime": ttime,
        "error": 0.0, "niter": 0, "converged": True,
        "history": history, "history_detail": history_detail,
        "layer_dx": layer_dx, "layer_dt": layer_dt,
        "layer_pz": layer_pz, "layer_vz": layer_vz,
        "layer_g0": layer_g0, "layer_g1": layer_g1,
        "Z": Z.copy(),
        "H": float(H), "dx_sum": float(H), "dx_minus_H": 0.0,
        "horizontal_path": True, "horizontal_layer": k,
    }


def pmax_for_wave(ip: int, model: VTIModel) -> np.ndarray:
    if ip == 1:
        pmax = 1.0 / (model.alpha * np.sqrt(1.0 + 2.0 * model.epsilon))
    elif ip == 2:
        pmax = 1.0 / model.beta
    elif ip == 3:
        pmax = 1.0 / (model.beta * np.sqrt(1.0 + 2.0 * model.gamma))
    else:
        raise ValueError("ip must be 1(qP), 2(qSV), or 3(qSH)")
    if np.any(~np.isfinite(pmax)) or np.any(pmax <= 0.0):
        raise FloatingPointError(f"invalid pmax for wave {WAVES[ip]}")
    return pmax


def q_to_p(qx: float, pmin: float) -> float:
    if not math.isfinite(qx):
        raise FloatingPointError("non-finite qx")
    qx = max(float(qx), 0.0)
    return qx * pmin / math.sqrt(1.0 + qx * qx)


def _strict_sqrt_arg(value: float, scale: float, name: str) -> float:
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is non-finite")
    scale = max(float(scale), MIN_POSITIVE)
    if value < -ROUND_TOL * scale:
        raise FloatingPointError(f"{name} is negative: {value:.16e}, scale={scale:.16e}")
    return max(value, 0.0)


def g_derivatives(ip: int, px: float, a: float, b: float, eps: float, delta: float, gamma: float):
    """Return g, g1, g2, g3 where pz^2 = g(px)."""
    if a <= 0.0 or b <= 0.0 or not all(math.isfinite(v) for v in (px, a, b, eps, delta, gamma)):
        raise FloatingPointError("non-finite or non-positive VTI parameters")

    if ip == 3:
        A = 1.0 + 2.0 * gamma
        if A <= 0.0:
            raise FloatingPointError("qSH invalid because 1 + 2*gamma <= 0")
        g0 = 1.0 / (b * b) - A * px * px
        scale = max(abs(1.0 / (b * b)), abs(A * px * px), MIN_POSITIVE)
        g0 = _strict_sqrt_arg(g0, scale, "qSH g0=pz^2")
        return g0, -2.0 * A * px, -2.0 * A, 0.0

    A = 1.0 + 2.0 * eps
    if A <= 0.0:
        raise FloatingPointError("qP/qSV invalid because 1 + 2*epsilon <= 0")

    K = 1.0 + delta + (eps - delta) * a * a / (b * b)
    B = 1.0 / (a * a) + 1.0 / (b * b) - 2.0 * K * px * px
    B1 = -4.0 * K * px
    B2 = -4.0 * K
    B3 = 0.0

    C = (A * px * px - 1.0 / (a * a)) * (px * px - 1.0 / (b * b))
    C1 = 4.0 * A * px**3 - 2.0 * (A / (b * b) + 1.0 / (a * a)) * px
    C2 = 12.0 * A * px * px - 2.0 * (A / (b * b) + 1.0 / (a * a))
    C3 = 24.0 * A * px

    U_raw = B * B - 4.0 * C
    U_scale = max(B * B, abs(4.0 * C), MIN_POSITIVE)
    U = _strict_sqrt_arg(U_raw, U_scale, "Christoffel discriminant U=B^2-4C")
    r = math.sqrt(max(U, MIN_POSITIVE))

    U1 = 2.0 * B * B1 - 4.0 * C1
    U2 = 2.0 * (B1 * B1 + B * B2) - 4.0 * C2
    U3 = 2.0 * (3.0 * B1 * B2 + B * B3) - 4.0 * C3

    r1 = 0.5 * U1 / r
    r2 = 0.5 * U2 / r - 0.25 * U1 * U1 / r**3
    r3 = 0.5 * U3 / r - 0.75 * U1 * U2 / r**3 + 0.375 * U1**3 / r**5

    sign = -1.0 if ip == 1 else 1.0
    g0 = 0.5 * (B + sign * r)
    g1 = 0.5 * (B1 + sign * r1)
    g2 = 0.5 * (B2 + sign * r2)
    g3 = 0.5 * (B3 + sign * r3)

    g_scale = max(abs(B), abs(r), MIN_POSITIVE)
    g0 = _strict_sqrt_arg(g0, g_scale, f"{WAVES[ip]} g0=pz^2")
    return g0, g1, g2, g3


def offset_value(ip: int, qx: float, H: float, Z: np.ndarray, model: VTIModel, pmin: float):
    """Return f(qx)=Xcal-H and px using only first slowness derivatives.

    This is the safe path used by the bracketed solver. It does not evaluate
    G2/G3, so it remains well behaved near critical rays where pz^2 is close
    to zero and higher derivatives are singular.
    """
    px = q_to_p(qx, pmin)
    active = Z > 0.0
    Xcal = 0.0
    for k in range(model.nlayer):
        if not active[k]:
            continue
        g0, g1, _, _ = g_derivatives(
            ip, px,
            model.alpha[k], model.beta[k], model.epsilon[k], model.delta[k], model.gamma[k],
        )
        root_g0 = math.sqrt(max(g0, MIN_POSITIVE))
        G1 = 0.5 * g1 / root_g0
        Xcal += -float(Z[k]) * G1
    f = float(Xcal) - float(H)
    if not all(math.isfinite(v) for v in (f, px, Xcal)):
        raise FloatingPointError("non-finite offset value")
    return f, px, float(Xcal)


def G_derivatives(ip: int, px: float, model: VTIModel, active: np.ndarray | None = None):
    """Return G derivatives only for layers crossed by the direct ray.

    If an active layer is at or extremely close to the critical slowness
    surface, G2/G3 are mathematically singular. In that case this function
    raises FloatingPointError, and solve_qx() uses the bracketed fallback.
    This avoids RuntimeWarning messages and avoids propagating inf/nan values
    into MCMC likelihood calculations.
    """
    G1 = np.zeros(model.nlayer)
    G2 = np.zeros(model.nlayer)
    G3 = np.zeros(model.nlayer)

    if active is None:
        active = np.ones(model.nlayer, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool)
        if active.shape != (model.nlayer,):
            raise ValueError(f"active mask shape must be ({model.nlayer},), got {active.shape}")

    for k in range(model.nlayer):
        if not active[k]:
            continue

        g0, g1, g2, g3 = g_derivatives(
            ip, px,
            model.alpha[k], model.beta[k], model.epsilon[k], model.delta[k], model.gamma[k],
        )
        if g0 <= CRITICAL_G0:
            raise FloatingPointError(
                f"near-critical {WAVES[ip]} derivative is unsafe in layer {k}: pz^2={g0:.16e}"
            )
        root_g0 = math.sqrt(g0)
        root2 = root_g0 * root_g0
        root3 = root2 * root_g0
        root5 = root3 * root2
        G1[k] = 0.5 * g1 / root_g0
        G2[k] = -0.25 * g1 * g1 / root3 + 0.5 * g2 / root_g0
        G3[k] = 0.375 * g1**3 / root5 - 0.75 * g1 * g2 / root3 + 0.5 * g3 / root_g0
        if not all(math.isfinite(v) for v in (G1[k], G2[k], G3[k])):
            raise FloatingPointError(f"non-finite {WAVES[ip]} G derivatives in layer {k}")
    return G1, G2, G3


def offset_and_derivatives(ip: int, qx: float, H: float, Z: np.ndarray, model: VTIModel, pmin: float):
    px = q_to_p(qx, pmin)
    active = Z > 0.0
    G1, G2, G3 = G_derivatives(ip, px, model, active)
    Xcal = -float(np.sum(Z * G1))
    f = Xcal - H

    dpdq = pmin * (1.0 + qx * qx)**(-1.5)
    d2pdq2 = -3.0 * qx * pmin * (1.0 + qx * qx)**(-2.5)
    term = G3 * dpdq * dpdq + G2 * d2pdq2
    if not np.all(np.isfinite(term[active])):
        raise FloatingPointError("non-finite second-derivative term")
    dfdq = -float(np.sum(Z * G2)) * dpdq
    d2fdq2 = -float(np.sum(Z * term))
    if not all(math.isfinite(v) for v in (f, dfdq, d2fdq2, px, Xcal)):
        raise FloatingPointError("non-finite offset equation value/derivative")
    return f, dfdq, d2fdq2, px, Xcal


def _safe_offset_for_trial(ip: int, qx: float, H: float, Z: np.ndarray, model: VTIModel, pmin: float):
    """Safe f evaluation for candidate testing."""
    if not math.isfinite(qx) or qx < 0.0:
        raise FloatingPointError("invalid trial qx")
    return offset_value(ip, qx, H, Z, model, pmin)


def solve_qx(ip: int, H: float, Z: np.ndarray, model: VTIModel, stop: float, max_iter: int):
    active = Z > 0.0
    history: list[tuple[float, float]] = []
    history_detail: list[dict[str, float]] = []
    if not np.any(active):
        raise FloatingPointError("zero vertical path length: source/receiver may be in same depth interval")
    if H == 0.0:
        zero_detail = {
            "qx": 0.0, "px": 0.0, "Xcal": 0.0, "f": 0.0,
            "df": 0.0, "d2f": 0.0, "disc": 0.0, "chosen_qx": 0.0, "chosen_f": 0.0,
        }
        return 0.0, 0.0, 0.0, 0, [(0.0, 0.0)] * max_iter, True, [zero_detail]

    pmax = pmax_for_wave(ip, model)
    pmin = float(np.min(pmax[active]))
    if not math.isfinite(pmin) or pmin <= 0.0:
        raise FloatingPointError("invalid pmin")

    # Bracket the root using f-only evaluations. qx=0 gives Xcal=0, so f=-H.
    q_low = 0.0
    f_low, px_low, x_low = offset_value(ip, q_low, H, Z, model, pmin)
    if abs(f_low) <= stop:
        zero_detail = {
            "qx": 0.0, "px": px_low, "Xcal": x_low, "f": f_low,
            "df": math.nan, "d2f": math.nan, "disc": math.nan,
            "chosen_qx": 0.0, "chosen_f": f_low,
        }
        return 0.0, px_low, f_low, 0, [(0.0, f_low)] * max_iter, True, [zero_detail]

    q_high = 1.0
    f_high = math.nan
    px_high = math.nan
    x_high = math.nan
    for _ in range(BRACKET_EXPAND_ITERS):
        try:
            f_high, px_high, x_high = offset_value(ip, q_high, H, Z, model, pmin)
        except FloatingPointError:
            q_high *= 0.5
            continue
        if f_high >= 0.0:
            break
        q_high *= 2.0
        if q_high > QX_UPPER_LIMIT:
            break
    if not math.isfinite(f_high) or f_high < 0.0:
        raise FloatingPointError(
            f"Cannot bracket {WAVES[ip]} qx root: f_low={f_low:.6e}, f_high={f_high:.6e}, q_high={q_high:.6e}"
        )

    qx = min(max(1.0, q_low), q_high)
    it_done = 0
    converged = False
    max_internal_iter = max(int(max_iter), ROBUST_SOLVE_MIN_ITERS)

    for it in range(1, max_internal_iter + 1):
        # Prefer full derivatives when safe; otherwise use f-only evaluation.
        derivative_ok = False
        df = math.nan
        d2f = math.nan
        disc = math.nan
        try:
            f, df, d2f, px, Xcal = offset_and_derivatives(ip, qx, H, Z, model, pmin)
            derivative_ok = True
        except FloatingPointError:
            f, px, Xcal = offset_value(ip, qx, H, Z, model, pmin)

        it_done = it
        if len(history) < max_iter:
            history.append((float(qx), float(f)))

        if abs(f) <= stop:
            converged = True
            if len(history_detail) < max_iter:
                history_detail.append({
                    "qx": float(qx), "px": float(px), "Xcal": float(Xcal), "f": float(f),
                    "df": float(df), "d2f": float(d2f), "disc": float(disc),
                    "chosen_qx": float(qx), "chosen_f": float(f),
                })
            break

        # Maintain a valid bracket.
        if f < 0.0:
            q_low = qx
            f_low = f
        else:
            q_high = qx
            f_high = f

        candidates = []
        if derivative_ok:
            disc = df * df - 2.0 * d2f * f
            if abs(d2f) > 1e-30 and math.isfinite(disc) and disc >= 0.0:
                root = math.sqrt(disc)
                candidates.extend([qx + (-df + root) / d2f, qx + (-df - root) / d2f])
            if abs(df) > 1e-30 and math.isfinite(df):
                candidates.append(qx - f / df)

        # Always include the bracket midpoint as a safe fallback.
        candidates.append(0.5 * (q_low + q_high))

        best_q = None
        best_f = None
        best_err = float("inf")
        for qtrial in candidates:
            if not math.isfinite(qtrial):
                continue
            if qtrial <= q_low or qtrial >= q_high:
                continue
            try:
                ftrial, _, _ = _safe_offset_for_trial(ip, qtrial, H, Z, model, pmin)
            except FloatingPointError:
                continue
            err = abs(ftrial)
            if err < best_err:
                best_q = qtrial
                best_f = ftrial
                best_err = err

        if best_q is None:
            best_q = 0.5 * (q_low + q_high)
            best_f, _, _ = offset_value(ip, best_q, H, Z, model, pmin)

        if len(history_detail) < max_iter:
            history_detail.append({
                "qx": float(qx), "px": float(px), "Xcal": float(Xcal), "f": float(f),
                "df": float(df), "d2f": float(d2f), "disc": float(disc),
                "chosen_qx": float(best_q), "chosen_f": float(best_f),
            })
        qx = best_q

    # Critical fix: recompute f and px for the final qx after the last update.
    f_final, px_final, Xcal_final = offset_value(ip, qx, H, Z, model, pmin)
    if abs(f_final) <= stop:
        converged = True
    if len(history) == 0 or history[-1][0] != float(qx):
        history.append((float(qx), float(f_final)))
    if not history_detail or history_detail[-1]["qx"] != float(qx):
        history_detail.append({
            "qx": float(qx), "px": float(px_final), "Xcal": float(Xcal_final), "f": float(f_final),
            "df": math.nan, "d2f": math.nan, "disc": math.nan,
            "chosen_qx": float(qx), "chosen_f": float(f_final),
        })

    if len(history) < max_iter:
        history.extend([(0.0, 0.0)] * (max_iter - len(history)))
    elif len(history) > max_iter:
        history = history[:max_iter - 1] + [(float(qx), float(f_final))]

    if len(history_detail) > max_iter:
        history_detail = history_detail[:max_iter - 1] + [{
            "qx": float(qx), "px": float(px_final), "Xcal": float(Xcal_final), "f": float(f_final),
            "df": math.nan, "d2f": math.nan, "disc": math.nan,
            "chosen_qx": float(qx), "chosen_f": float(f_final),
        }]

    return qx, px_final, f_final, min(it_done, max_internal_iter), history, converged, history_detail

def travel_time_and_layer_dx(ip: int, px: float, Z: np.ndarray, model: VTIModel):
    dt = np.zeros(model.nlayer)
    dx = np.zeros(model.nlayer)
    # Use zeros for inactive layers instead of NaN.
    # This keeps layer_contributions.dat fully finite for downstream validation/MCMC.
    # Inactive layers are already identified by Z[k] == 0.0, so pz/Vz/g0/g1=0.0
    # should be interpreted as "not used", not as a physical value.
    pz_arr = np.zeros(model.nlayer)
    vz_arr = np.zeros(model.nlayer)
    g0_arr = np.zeros(model.nlayer)
    g1_arr = np.zeros(model.nlayer)
    for k in range(model.nlayer):
        if Z[k] <= 0.0:
            continue
        g0, g1, _, _ = g_derivatives(
            ip, px,
            model.alpha[k], model.beta[k], model.epsilon[k], model.delta[k], model.gamma[k],
        )
        pz = math.sqrt(max(g0, MIN_POSITIVE))
        dS_dpx = -g1
        dS_dpz = 2.0 * pz
        denom = px * dS_dpx + pz * dS_dpz
        if not math.isfinite(denom) or abs(denom) < 1e-30:
            raise FloatingPointError("zero or non-finite group-velocity denominator")
        Vz = dS_dpz / denom
        if not math.isfinite(Vz) or Vz <= 0.0:
            raise FloatingPointError(f"invalid vertical group velocity Vz={Vz}")
        dt[k] = Z[k] / Vz
        dx[k] = -Z[k] * (0.5 * g1 / pz)
        pz_arr[k] = pz
        vz_arr[k] = Vz
        g0_arr[k] = g0
        g1_arr[k] = g1
    total = float(np.sum(dt))
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("invalid travel time")
    return total, dx, dt, pz_arr, vz_arr, g0_arr, g1_arr

def trace_direct_pair(sx: float, sz: float, rx: float, rz: float, model: VTIModel, stop: float, max_iter: int):
    H = abs(float(rx) - float(sx))
    Z = direct_layer_thickness(sz, rz, model)
    rows = []

    # Special case: same-depth horizontal path. The qx root formulation uses
    # vertical layer thicknesses and cannot solve a path with sum(Z)=0. In this
    # case, compute the direct travel time from the horizontal velocity of the
    # layer containing the source/receiver depth.
    if not np.any(Z > 0.0):
        z_mid = 0.5 * (float(sz) + float(rz))
        for ip in (1, 2, 3):
            rows.append(horizontal_direct_result(ip, H, z_mid, model, max_iter))
        return rows

    for ip in (1, 2, 3):
        qx, px, err, niter, history, converged, history_detail = solve_qx(ip, H, Z, model, stop, max_iter)
        ttime, layer_dx, layer_dt, layer_pz, layer_vz, layer_g0, layer_g1 = travel_time_and_layer_dx(ip, px, Z, model)
        dx_sum = float(np.sum(layer_dx))
        rows.append({
            "wave": WAVES[ip], "qx": qx, "px": px, "ttime": ttime,
            "error": err, "niter": niter, "converged": converged,
            "history": history, "history_detail": history_detail,
            "layer_dx": layer_dx, "layer_dt": layer_dt,
            "layer_pz": layer_pz, "layer_vz": layer_vz,
            "layer_g0": layer_g0, "layer_g1": layer_g1,
            "Z": Z.copy(),
            "H": H, "dx_sum": dx_sum, "dx_minus_H": dx_sum - H,
            "horizontal_path": False, "horizontal_layer": -1,
        })
    return rows

def write_input_echo_files(output_dir: Path, sx: np.ndarray, sz: np.ndarray,
                           rx: np.ndarray, rz: np.ndarray,
                           model: VTIModel, stop: float, max_iter: int,
                           control_values: list[float] | None = None,
                           stop_source: str = "", max_iter_source: str = "") -> None:
    """Write exactly what the code read from geometry.dat/vel.dat/control.dat."""
    with (output_dir / "input_summary.dat").open("w", encoding="utf-8") as f:
        f.write("# Echo of input files read by vti_direct_forward_debug.py\n")
        f.write(f"ns\t{len(sx)}\n")
        f.write(f"nr\t{len(rx)}\n")
        f.write(f"nlayer\t{model.nlayer}\n")
        f.write(f"control_stop\t{stop:.16e}\n")
        f.write(f"control_max_iter\t{max_iter:d}\n")
        f.write(f"control_stop_source\t{stop_source}\n")
        f.write(f"control_max_iter_source\t{max_iter_source}\n")
        if control_values is not None:
            f.write("control_raw_numeric_values\t" + "\t".join(f"{v:.16e}" for v in control_values) + "\n")
        f.write(f"source_x_minmax\t{np.min(sx):.16e}\t{np.max(sx):.16e}\n")
        f.write(f"source_z_minmax\t{np.min(sz):.16e}\t{np.max(sz):.16e}\n")
        f.write(f"receiver_x_minmax\t{np.min(rx):.16e}\t{np.max(rx):.16e}\n")
        f.write(f"receiver_z_minmax\t{np.min(rz):.16e}\t{np.max(rz):.16e}\n")
        f.write("# model columns: k dep alpha beta epsilon gamma delta pmax_qP pmax_qSV pmax_qSH\n")
        pmax_p = pmax_for_wave(1, model)
        pmax_sv = pmax_for_wave(2, model)
        pmax_sh = pmax_for_wave(3, model)
        for k in range(model.nlayer):
            f.write(
                f"layer\t{k:d}\t{model.dep[k]:.16e}\t{model.alpha[k]:.16e}\t{model.beta[k]:.16e}\t"
                f"{model.epsilon[k]:.16e}\t{model.gamma[k]:.16e}\t{model.delta[k]:.16e}\t"
                f"{pmax_p[k]:.16e}\t{pmax_sv[k]:.16e}\t{pmax_sh[k]:.16e}\n"
            )

    with (output_dir / "input_geometry_echo.dat").open("w", encoding="utf-8") as f:
        f.write("# Sources read from geometry.dat\n")
        f.write("# isrc sx sz\n")
        for i in range(len(sx)):
            f.write(f"{i:d}\t{sx[i]:.16e}\t{sz[i]:.16e}\n")
        f.write("# Receivers read from geometry.dat\n")
        f.write("# ircv rx rz\n")
        for i in range(len(rx)):
            f.write(f"{i:d}\t{rx[i]:.16e}\t{rz[i]:.16e}\n")

    with (output_dir / "input_velocity_echo.dat").open("w", encoding="utf-8") as f:
        f.write("# VTI model read from vel.dat\n")
        f.write("# k dep alpha beta epsilon gamma delta\n")
        for k in range(model.nlayer):
            f.write(
                f"{k:d}\t{model.dep[k]:.16e}\t{model.alpha[k]:.16e}\t{model.beta[k]:.16e}\t"
                f"{model.epsilon[k]:.16e}\t{model.gamma[k]:.16e}\t{model.delta[k]:.16e}\n"
            )


def print_input_summary(sx: np.ndarray, sz: np.ndarray, rx: np.ndarray, rz: np.ndarray,
                        model: VTIModel, stop: float, max_iter: int,
                        control_values: list[float] | None = None,
                        stop_source: str = "", max_iter_source: str = "") -> None:
    print("[INPUT] geometry/control/model loaded", flush=True)
    print(f"[INPUT] ns={len(sx)}, nr={len(rx)}, nlayer={model.nlayer}", flush=True)
    print(f"[INPUT] control: stop={stop:.12e} m ({stop_source}), max_iter={max_iter} ({max_iter_source})", flush=True)
    if control_values is not None:
        print(f"[INPUT] control raw numeric values first 10 = {control_values[:10]}", flush=True)
    print(
        f"[INPUT] source x range=({np.min(sx):.6g}, {np.max(sx):.6g}), "
        f"z range=({np.min(sz):.6g}, {np.max(sz):.6g})",
        flush=True,
    )
    print(
        f"[INPUT] receiver x range=({np.min(rx):.6g}, {np.max(rx):.6g}), "
        f"z range=({np.min(rz):.6g}, {np.max(rz):.6g})",
        flush=True,
    )
    print("[INPUT] first layers: k dep alpha beta epsilon gamma delta", flush=True)
    for k in range(min(model.nlayer, 8)):
        print(
            f"[INPUT] layer {k:02d}: dep={model.dep[k]:.6g}, alpha={model.alpha[k]:.6g}, "
            f"beta={model.beta[k]:.6g}, eps={model.epsilon[k]:.6g}, "
            f"gamma={model.gamma[k]:.6g}, delta={model.delta[k]:.6g}",
            flush=True,
        )
    if model.nlayer > 8:
        print(f"[INPUT] ... {model.nlayer - 8} more layers written to input_velocity_echo.dat", flush=True)


def run_from_files(input_dir: str | Path = "../01-input/output", output_dir: str | Path = "../03-output",
                   fail_on_nonconvergence: bool = True, verbose_pairs: int = 20,
                   stop_override: float | None = None, max_iter_override: int | None = None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = ["geometry.dat", "vel.dat", "control.dat"]
    missing = [str(input_dir / name) for name in required_files if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s): " + ", ".join(missing) +
            f". Current working directory is: {Path.cwd()}"
        )

    print(f"[PATH] input_dir  = {input_dir}", flush=True)
    print(f"[PATH] output_dir = {output_dir}", flush=True)

    sx, sz, rx, rz = read_geometry(input_dir / "geometry.dat")
    model = read_velocity(input_dir / "vel.dat")
    stop, max_iter, control_values, stop_source, max_iter_source = read_control(
        input_dir / "control.dat", stop_override=stop_override, max_iter_override=max_iter_override
    )

    print_input_summary(sx, sz, rx, rz, model, stop, max_iter, control_values, stop_source, max_iter_source)
    write_input_echo_files(output_dir, sx, sz, rx, rz, model, stop, max_iter, control_values, stop_source, max_iter_source)

    qx_rows = []
    tt_rows = []
    all_results = []
    diagnostic_rows = []
    layer_rows = []
    iteration_detail_rows = []
    nonconv = []

    total_pairs = len(sx) * len(rx)
    pair_count = 0

    for isrc in range(len(sx)):
        print(f"[FORWARD] source {isrc + 1}/{len(sx)}: sx={sx[isrc]:.12e}, sz={sz[isrc]:.12e}", flush=True)
        for ircv in range(len(rx)):
            pair_count += 1
            if pair_count <= verbose_pairs or pair_count == total_pairs:
                print(
                    f"[PAIR] {pair_count}/{total_pairs}: isrc={isrc}, ircv={ircv}, "
                    f"rx={rx[ircv]:.12e}, rz={rz[ircv]:.12e}, H={abs(rx[ircv] - sx[isrc]):.12e}",
                    flush=True,
                )
            results = trace_direct_pair(sx[isrc], sz[isrc], rx[ircv], rz[ircv], model, stop, max_iter)
            all_results.append((isrc, ircv, results))
            qx_rows.append([sx[isrc], sz[isrc], rx[ircv], rz[ircv]] + [r["qx"] for r in results])
            tt_rows.append([sx[isrc], sz[isrc], rx[ircv], rz[ircv]] + [r["ttime"] for r in results])
            for ip, r in enumerate(results, start=1):
                diagnostic_rows.append([
                    isrc, ircv, ip, r["H"], r["qx"], r["px"], r["ttime"],
                    r["error"], r["niter"], int(r["converged"]), r["dx_sum"], r["dx_minus_H"],
                ])
                if pair_count <= verbose_pairs:
                    print(
                        f"[PAIR][{WAVES[ip]}] qx={r['qx']:.9e}, px={r['px']:.9e}, "
                        f"ttime={r['ttime']:.12e}, err={r['error']:.12e}, "
                        f"niter={r['niter']}, converged={int(r['converged'])}",
                        flush=True,
                    )
                if not r["converged"]:
                    nonconv.append((isrc, ircv, WAVES[ip], r["error"], r["niter"]))

                Z = r["Z"]
                for k in range(model.nlayer):
                    layer_rows.append([
                        isrc, ircv, ip, k, Z[k], r["layer_dx"][k], r["layer_dt"][k],
                        r["layer_pz"][k], r["layer_vz"][k], r["layer_g0"][k], r["layer_g1"][k],
                    ])
                for it, item in enumerate(r["history_detail"], start=1):
                    iteration_detail_rows.append([
                        isrc, ircv, ip, it,
                        item["qx"], item["px"], item["Xcal"], item["f"],
                        item["df"], item["d2f"], item["disc"], item["chosen_qx"], item["chosen_f"],
                    ])

    def write_table(path: Path, rows: list[list[float]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{len(sx):d}\t{len(rx):d}\n")
            for row in rows:
                f.write("\t".join(f"{float(v):{OUT_FLOAT_FMT}}" for v in row) + "\n")

    write_table(output_dir / "qx.dat", qx_rows)
    write_table(output_dir / "ttime.dat", tt_rows)

    with (output_dir / "iteration.dat").open("w", encoding="utf-8") as f:
        f.write(f"{len(sx):d}\t{len(rx):d}\n")
        f.write(f"{max_iter:d}\n")
        for isrc, ircv, results in all_results:
            f.write(f"{sx[isrc]:{OUT_FLOAT_FMT}}\t{sz[isrc]:{OUT_FLOAT_FMT}}\t{rx[ircv]:{OUT_FLOAT_FMT}}\t{rz[ircv]:{OUT_FLOAT_FMT}}\n")
            for ip, r in enumerate(results, start=1):
                f.write(f"{ip:d}\n")
                for q, err in r["history"][:max_iter]:
                    f.write(f"{q:{OUT_FLOAT_FMT}}\t{err:{OUT_FLOAT_FMT}}\n")

    with (output_dir / "diagnostics.dat").open("w", encoding="utf-8") as f:
        f.write("# isrc ircv wave_id H qx px ttime offset_error niter converged dx_sum dx_minus_H\n")
        for row in diagnostic_rows:
            f.write("\t".join(
                f"{int(v):d}" if i in (0, 1, 2, 8, 9) else f"{float(v):{OUT_FLOAT_FMT}}"
                for i, v in enumerate(row)
            ) + "\n")

    with (output_dir / "layer_contributions.dat").open("w", encoding="utf-8") as f:
        f.write("# isrc ircv wave_id layer Z layer_dx layer_dt pz Vz g0 g1\n")
        for row in layer_rows:
            f.write("\t".join(
                f"{int(v):d}" if i in (0, 1, 2, 3) else f"{float(v):{OUT_FLOAT_FMT}}"
                for i, v in enumerate(row)
            ) + "\n")

    with (output_dir / "iteration_detailed.dat").open("w", encoding="utf-8") as f:
        f.write("# isrc ircv wave_id iter qx px Xcal f df d2f disc chosen_qx chosen_f\n")
        for row in iteration_detail_rows:
            f.write("\t".join(
                f"{int(v):d}" if i in (0, 1, 2, 3) else f"{float(v):{OUT_FLOAT_FMT}}"
                for i, v in enumerate(row)
            ) + "\n")

    diag = np.asarray(diagnostic_rows, dtype=float)
    print(f"[OK] forward completed: ns={len(sx)}, nr={len(rx)}, nlayer={model.nlayer}", flush=True)
    print(f"[OUTPUT] wrote: qx.dat, ttime.dat, iteration.dat, diagnostics.dat", flush=True)
    print(f"[OUTPUT] wrote: input_summary.dat, input_geometry_echo.dat, input_velocity_echo.dat", flush=True)
    print(f"[OUTPUT] wrote: layer_contributions.dat, iteration_detailed.dat", flush=True)
    print(f"[CHECK] stop={stop:g}, max_iter={max_iter}", flush=True)
    for ip in (1, 2, 3):
        sub = diag[diag[:, 2] == ip]
        max_err = float(np.max(np.abs(sub[:, 7])))
        max_dx_err = float(np.max(np.abs(sub[:, 11])))
        nbad = int(np.sum(sub[:, 9] == 0))
        print(
            f"[CHECK] {WAVES[ip]}: nonconverged={nbad}, "
            f"max|offset_error|={max_err:.12e} m, max|sum_dx-H|={max_dx_err:.12e} m, "
            f"ttime_range=({np.min(sub[:, 6]):.12e}, {np.max(sub[:, 6]):.12e}) s, "
            f"niter_range=({int(np.min(sub[:, 8]))}, {int(np.max(sub[:, 8]))})",
            flush=True,
        )

    if nonconv:
        msg = "\n".join(
            f"source={i}, receiver={j}, wave={w}, err={e:.12e}, niter={n}"
            for i, j, w, e, n in nonconv[:20]
        )
        print("[WARN] Non-converged qx solves found. First cases:\n" + msg, flush=True)
        if fail_on_nonconvergence:
            raise RuntimeError("At least one qx solve did not converge. See diagnostics.dat.")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Verbose diagnostic direct-wave qx forward modelling only")
    p.add_argument("--input-dir", type=Path, default=Path("../01-input/output"), help="directory containing control.dat, geometry.dat, vel.dat generated by direct.py")
    p.add_argument("--output-dir", type=Path, default=Path("../03-output"), help="directory for qx.dat, ttime.dat and diagnostic files")
    p.add_argument("--allow-nonconvergence", action="store_true", help="do not raise an error when some qx solves fail")
    p.add_argument("--verbose-pairs", type=int, default=20,
                   help="print detailed qx/ttime results for the first N source-receiver pairs; all details are always written to files")
    p.add_argument("--stop", type=float, default=None,
                   help="override control.dat stop tolerance, e.g. --stop 1e-8")
    p.add_argument("--max-iter", type=int, default=None,
                   help="override control.dat max Newton iterations, e.g. --max-iter 50")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_from_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fail_on_nonconvergence=not args.allow_nonconvergence,
        verbose_pairs=args.verbose_pairs,
        stop_override=args.stop,
        max_iter_override=args.max_iter,
    )
