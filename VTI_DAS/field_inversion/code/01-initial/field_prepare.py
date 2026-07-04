#!/usr/bin/env python3
"""Prepare field DAS P-arrival data for the field-only VTI DAS workflow.

This script writes all initial/check files into one directory, normally

    01-initial/output/

No synthetic source placeholders are written.  Before grid search, geometry.dat
contains only the event count and receiver geometry.  After grid search,
geo.dat contains the initial source coordinates used by the inversion.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
from typing import Iterable

import numpy as np

FLOAT_FMT = ".12f"
SMALL_FLOAT_FMT = ".12e"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    return rows


def _as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except KeyError as exc:
        raise KeyError(f"Missing required column {key!r}. Available columns: {list(row)}") from exc
    except Exception as exc:
        raise ValueError(f"Cannot parse column {key!r} value {row.get(key)!r} as float") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in column {key!r}: {value!r}")
    return value


def _as_int(row: dict[str, str], key: str) -> int:
    return int(round(_as_float(row, key)))


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_event_ids(text: str | None, available: Iterable[int]) -> list[int]:
    avail = sorted(set(int(x) for x in available))
    if text is None or str(text).strip() == "" or str(text).strip().lower() in {"all", "*"}:
        return avail
    out: list[int] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    missing = sorted(set(out) - set(avail))
    if missing:
        raise ValueError(f"Requested event IDs not found in data: {missing[:20]}")
    wanted = set(out)
    return [x for x in avail if x in wanted]


def write_control(path: Path, stop: float, max_iter: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("----stop criterion----\n")
        f.write(f"{float(stop):{SMALL_FLOAT_FMT}}\n")
        f.write("------ maximum number of iteration ---------\n")
        f.write(f"{int(max_iter):d}\n")
        f.write("-------------direct/reflect-----------\n")
        f.write("1\n1\n")


def write_receiver_geometry(path: Path, ns: int, receiver_ids: np.ndarray, rx: np.ndarray, rz: np.ndarray) -> None:
    """Write receiver-only geometry used before grid search.

    The source section stores only the number of events.  No placeholder source
    coordinates are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("----number of events----\n")
        f.write(f"{int(ns):d}\n")
        f.write("----rev(receiver_id,x,z)----\n")
        f.write(f"{len(rx):d}\n")
        for rid, x, z in zip(receiver_ids, rx, rz):
            f.write(f"{int(rid):d}\t{float(x):{FLOAT_FMT}}\t{float(z):{FLOAT_FMT}}\n")


def write_velocity(path: Path, dep: np.ndarray, vp: np.ndarray, vs: np.ndarray,
                   epsilon: np.ndarray, gamma: np.ndarray, delta: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(dep)
    if not all(len(a) == n for a in (vp, vs, epsilon, gamma, delta)):
        raise ValueError("Velocity arrays must all have the same length")
    with path.open("w", encoding="utf-8") as f:
        f.write("----nlayer----\n")
        f.write(f"{n:d}\n")
        f.write("-----dep, alpha0, beta0, epsilon, gamma, delta ----\n")
        for row in zip(dep, vp, vs, epsilon, gamma, delta):
            f.write("\t".join(f"{float(v):{FLOAT_FMT}}" for v in row) + "\n")


def write_nobs_compact(path: Path, sigma: float, event_ids: list[int], receiver_ids: np.ndarray,
                       rx: np.ndarray, rz: np.ndarray, tp: np.ndarray) -> None:
    """Write observed P-only field data without source coordinates.

    Numeric layout after sigma and ns/nr is five columns per row:
        event_id receiver_id rx rz tp
    vti_joint_mcmc_dram.py supports this compact field format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, nr = tp.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("--------------- assumed data standard deviation ---------------------\n")
        f.write(f"{float(sigma):.12e}\n")
        f.write("----------------------- ns,nr -------------------------------\n")
        f.write(f"{ns:d}\t{nr:d}\n")
        f.write("-------------------event_id,receiver_id,rx,rz,tp---------------------------------\n")
        for i, eid in enumerate(event_ids):
            for j, rid in enumerate(receiver_ids):
                f.write(
                    f"{int(eid):d}\t{int(rid):d}\t"
                    f"{float(rx[j]):.12f}\t{float(rz[j]):.12f}\t"
                    f"{float(tp[i, j]):.12e}\n"
                )


def write_field_ttime(path: Path, event_ids: list[int], receiver_ids: np.ndarray,
                      rx: np.ndarray, rz: np.ndarray, tp: np.ndarray) -> None:
    """Write a human-readable field P-time table without source coordinates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ns, nr = tp.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("----field P arrival times: ns,nr----\n")
        f.write(f"{ns:d}\t{nr:d}\n")
        f.write("----event_id, receiver_id, rx, rz, tp----\n")
        for i, eid in enumerate(event_ids):
            for j, rid in enumerate(receiver_ids):
                f.write(
                    f"{int(eid):d}\t{int(rid):d}\t"
                    f"{float(rx[j]):.12f}\t{float(rz[j]):.12f}\t"
                    f"{float(tp[i, j]):.12e}\n"
                )


def prepare_velocity(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    rows = _read_csv_dicts(args.velocity_csv)
    rows = sorted(rows, key=lambda r: _as_float(r, args.velocity_top_col))
    top = np.asarray([_as_float(r, args.velocity_top_col) for r in rows], dtype=float)
    bottom = np.asarray([_as_float(r, args.velocity_bottom_col) for r in rows], dtype=float)
    vp = np.asarray([_as_float(r, args.vp_col) for r in rows], dtype=float)
    vs = np.asarray([_as_float(r, args.vs_col) for r in rows], dtype=float)
    if not np.all(np.diff(top) > 0.0):
        raise ValueError("Velocity top depths are not strictly increasing")
    if np.any(bottom <= top):
        raise ValueError("Velocity bottom depths must be deeper than top depths")

    dep = top.copy()
    if args.add_bottom_halfspace:
        dep = np.r_[dep, bottom[-1]]
        vp = np.r_[vp, vp[-1]]
        vs = np.r_[vs, vs[-1]]

    epsilon = np.full(dep.shape, float(args.epsilon), dtype=float)
    gamma = np.full(dep.shape, float(args.gamma), dtype=float)
    delta = np.full(dep.shape, float(args.delta), dtype=float)
    meta = {
        "n_input_velocity_rows": len(rows),
        "n_model_depth_nodes_written": int(len(dep)),
        "model_top_depth_m": float(dep[0]),
        "model_last_depth_node_m": float(dep[-1]),
        "added_bottom_halfspace": bool(args.add_bottom_halfspace),
        "epsilon_default": float(args.epsilon),
        "gamma_default": float(args.gamma),
        "delta_default": float(args.delta),
    }
    return dep, vp, vs, epsilon, gamma, delta, meta


def prepare_times(args) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, list[dict[str, object]], list[dict[str, object]]]:
    rows = _read_csv_dicts(args.times_csv)
    all_event_ids = [_as_int(r, args.event_col) for r in rows]
    event_ids = parse_event_ids(args.event_ids, all_event_ids)
    event_set = set(event_ids)
    rows = [r for r in rows if _as_int(r, args.event_col) in event_set]
    if not rows:
        raise ValueError("No travel-time rows remain after event filtering")

    by_event: dict[int, list[dict[str, str]]] = {eid: [] for eid in event_ids}
    for r in rows:
        by_event[_as_int(r, args.event_col)].append(r)

    for eid in event_ids:
        by_event[eid].sort(key=lambda r: (_as_float(r, args.receiver_sort_col), _as_int(r, args.receiver_id_col)))
        if args.receiver_stride > 1:
            by_event[eid] = by_event[eid][::args.receiver_stride]

    first = by_event[event_ids[0]]
    original_receiver_ids = [_as_int(r, args.receiver_id_col) for r in first]
    if len(original_receiver_ids) < 2:
        raise ValueError("Need at least two receivers for differential P arrivals")
    if len(set(original_receiver_ids)) != len(original_receiver_ids):
        raise ValueError("Duplicate receiver IDs in the first selected event")

    if args.renumber_receivers:
        receiver_ids = np.arange(1, len(original_receiver_ids) + 1, dtype=int)
    else:
        receiver_ids = np.asarray(original_receiver_ids, dtype=int)

    rx = np.asarray([_as_float(r, args.receiver_x_col) for r in first], dtype=float)
    rz = np.asarray([_as_float(r, args.receiver_z_col) for r in first], dtype=float)
    tp = np.zeros((len(event_ids), len(receiver_ids)), dtype=float)

    for i, eid in enumerate(event_ids):
        recs = by_event[eid]
        ids = [_as_int(r, args.receiver_id_col) for r in recs]
        if ids != original_receiver_ids:
            raise ValueError(
                f"Receiver list/order for event {eid} differs from event {event_ids[0]}. "
                "Use a consistent receiver_sort_col/receiver_stride or filter the CSV first."
            )
        tp[i, :] = [_as_float(r, args.p_time_col) for r in recs]
        if not np.all(np.isfinite(tp[i, :])):
            raise ValueError(f"Non-finite P times for event {eid}")

    receiver_rows = []
    for j, r in enumerate(first):
        receiver_rows.append({
            "receiver_index_0based": j,
            "receiver_index_1based": j + 1,
            "receiver_id": int(receiver_ids[j]),
            "original_receiver_id": _as_int(r, args.receiver_id_col),
            "receiver_sort_value": _as_float(r, args.receiver_sort_col),
            "rx_model_m": float(rx[j]),
            "rz_model_m": float(rz[j]),
            "trace_0based": r.get("trace_0based", ""),
            "channel_1based": r.get("channel_1based", ""),
            "x_m": r.get("x_m", ""),
            "y_m": r.get("y_m", ""),
            "depth_m": r.get("depth_m", ""),
            "distance_along_fiber_m": r.get("distance_along_fiber_m", ""),
        })

    event_rows = [{"source_index_0based": i, "source_index_1based": i + 1, "event_id": eid} for i, eid in enumerate(event_ids)]

    meta = {
        "selected_event_ids": event_ids,
        "ns": int(len(event_ids)),
        "nr": int(len(receiver_ids)),
        "receiver_x_col_for_2d_model": args.receiver_x_col,
        "receiver_z_col_for_2d_model": args.receiver_z_col,
        "receiver_sort_col": args.receiver_sort_col,
        "renumber_receivers_to_1based_order": bool(args.renumber_receivers),
        "receiver_stride": int(args.receiver_stride),
        "original_receiver_id_first": int(original_receiver_ids[0]),
        "original_receiver_id_last": int(original_receiver_ids[-1]),
        "model_receiver_id_first": int(receiver_ids[0]),
        "model_receiver_id_last": int(receiver_ids[-1]),
        "receiver_x_range_m": [float(np.min(rx)), float(np.max(rx))],
        "receiver_z_range_m": [float(np.min(rz)), float(np.max(rz))],
        "p_time_range_s": [float(np.min(tp)), float(np.max(tp))],
    }
    return event_ids, receiver_ids, rx, rz, tp, meta, receiver_rows, event_rows


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepare field DAS P-time CSV data for VTI DAS MCMC")
    p.add_argument("--velocity-csv", type=Path, required=True)
    p.add_argument("--times-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("../01-initial/output"))
    p.add_argument("--sigma", type=float, default=0.005, help="Assumed P-pick/differential-time uncertainty")
    p.add_argument("--qx-stop", type=float, default=1e-6)
    p.add_argument("--qx-max-iter", type=int, default=20)

    p.add_argument("--velocity-top-col", default="top_depth_m")
    p.add_argument("--velocity-bottom-col", default="bottom_depth_m")
    p.add_argument("--vp-col", default="vp_mps")
    p.add_argument("--vs-col", default="vs_mps")
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--add-bottom-halfspace", action="store_true", default=True)
    p.add_argument("--no-add-bottom-halfspace", dest="add_bottom_halfspace", action="store_false")

    p.add_argument("--event-ids", default="all")
    p.add_argument("--event-col", default="event_id")
    p.add_argument("--receiver-id-col", default="receiver_id")
    p.add_argument("--receiver-sort-col", default="receiver_id")
    p.add_argument("--renumber-receivers", type=str2bool, default=True)
    p.add_argument("--receiver-stride", type=int, default=1)
    p.add_argument("--receiver-x-col", default="x_m")
    p.add_argument("--receiver-z-col", default="depth_m")
    p.add_argument("--p-time-col", default="newP_time_s")
    args = p.parse_args(argv)

    if args.receiver_stride < 1:
        raise ValueError("--receiver-stride must be >= 1")
    if args.sigma <= 0.0 or not math.isfinite(args.sigma):
        raise ValueError("--sigma must be finite and positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    dep, vp, vs, eps, gam, delt, vel_meta = prepare_velocity(args)
    event_ids, receiver_ids, rx, rz, tp, time_meta, receiver_rows, event_rows = prepare_times(args)

    write_control(out / "control.dat", args.qx_stop, args.qx_max_iter)
    write_receiver_geometry(out / "geometry.dat", len(event_ids), receiver_ids, rx, rz)
    write_velocity(out / "vel.dat", dep, vp, vs, eps, gam, delt)
    write_nobs_compact(out / "nobs.dat", args.sigma, event_ids, receiver_ids, rx, rz, tp)
    write_field_ttime(out / "ttime.dat", event_ids, receiver_ids, rx, rz, tp)
    write_field_ttime(out / "field_p_times.dat", event_ids, receiver_ids, rx, rz, tp)
    write_csv_rows(out / "field_event_id_map.csv", event_rows)
    write_csv_rows(out / "field_receivers_used.csv", receiver_rows)

    meta = {
        "velocity_csv": str(args.velocity_csv),
        "times_csv": str(args.times_csv),
        "velocity": vel_meta,
        "travel_times": time_meta,
        "files_written": {
            "control": str(out / "control.dat"),
            "receiver_only_geometry": str(out / "geometry.dat"),
            "initial_velocity": str(out / "vel.dat"),
            "observed_p_data": str(out / "nobs.dat"),
            "field_p_times_readable": str(out / "field_p_times.dat"),
            "receiver_map": str(out / "field_receivers_used.csv"),
            "event_map": str(out / "field_event_id_map.csv"),
        },
        "note": "The solver is 2-D. The selected horizontal CSV column is used as model X; geometry.dat intentionally contains no source placeholders. geo.dat is produced by grid search.",
    }
    meta_path = out / "field_prepare_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[FIELD] events={len(event_ids)}, receivers={len(rx)}, observations={tp.size}")
    if receiver_rows:
        print(
            "[FIELD] receiver order: "
            f"model receiver_id {receiver_rows[0]['receiver_id']}..{receiver_rows[-1]['receiver_id']} "
            f"mapped from original_receiver_id {receiver_rows[0]['original_receiver_id']}..{receiver_rows[-1]['original_receiver_id']}"
        )
    print(f"[FIELD] receiver x from {args.receiver_x_col}: {np.min(rx):.3f} to {np.max(rx):.3f} m")
    print(f"[FIELD] receiver z from {args.receiver_z_col}: {np.min(rz):.3f} to {np.max(rz):.3f} m")
    print(f"[FIELD] P-time range: {np.min(tp):.6f} to {np.max(tp):.6f} s")
    print(f"[FIELD] velocity nodes written: {len(dep)}; dep {dep[0]:.1f} to {dep[-1]:.1f} m")
    print(f"[FIELD] wrote metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
