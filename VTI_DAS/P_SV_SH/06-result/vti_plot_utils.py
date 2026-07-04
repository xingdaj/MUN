#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

@dataclass
class VTIModel:
    dep: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    epsilon: np.ndarray
    gamma: np.ndarray
    delta: np.ndarray

@dataclass
class Geometry:
    sx: np.ndarray
    sz: np.ndarray
    rx: np.ndarray | None = None
    rz: np.ndarray | None = None


def float_tokens(path: str | Path) -> list[float]:
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    vals: list[float] = []
    for token in re.split(r'[\s,]+', text):
        if not token or token.startswith('#'):
            continue
        try:
            vals.append(float(token))
        except ValueError:
            pass
    return vals


def read_model_dat(path: str | Path) -> VTIModel:
    vals = float_tokens(path)
    if len(vals) < 7:
        raise ValueError(f'Cannot read VTI model from {path}')
    nlayer = int(vals[0])
    rest = vals[1:]
    # Supports both formats:
    #   nlayer; dep alpha beta epsilon gamma delta
    #   nlayer; index dep alpha beta epsilon gamma delta
    if len(rest) >= 7 * nlayer:
        arr7 = np.asarray(rest[:7*nlayer], dtype=float).reshape(nlayer, 7)
        first_col = arr7[:, 0]
        if np.allclose(first_col, np.arange(1, nlayer + 1)) or np.allclose(first_col, np.arange(nlayer)):
            arr = arr7[:, 1:7]
        else:
            arr = np.asarray(rest[:6*nlayer], dtype=float).reshape(nlayer, 6)
    else:
        arr = np.asarray(rest[:6*nlayer], dtype=float).reshape(nlayer, 6)
    return VTIModel(dep=arr[:,0].copy(), alpha=arr[:,1].copy(), beta=arr[:,2].copy(),
                    epsilon=arr[:,3].copy(), gamma=arr[:,4].copy(), delta=arr[:,5].copy())


def _numeric_rows(path: str | Path) -> list[list[float]]:
    """Read a whitespace/comma separated numeric file, ignoring comments and blank lines.

    This preserves row boundaries, which is important for best.dat/mean.dat.
    """
    rows: list[list[float]] = []
    for raw in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        vals: list[float] = []
        for token in re.split(r'[\s,]+', line):
            if not token:
                continue
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if vals:
            rows.append(vals)
    return rows


def read_summary_dat(path: str | Path) -> tuple[float | None, VTIModel, Geometry]:
    """Read mean.dat/best.dat written by vti_joint_mcmc_dram2.py.

    Robust expected format:
      # misfit
      value
      # nlayer
      n
      # dep alpha beta epsilon gamma delta
      n rows, each with either:
        dep alpha beta epsilon gamma delta
      or:
        index dep alpha beta epsilon gamma delta
      # ns
      ns
      # sx sz
      ns rows, each with either:
        sx sz
      or:
        index sx sz

    The previous token-based reader could mistake dep=0 as an index column and
    shift the whole file. This line-based reader avoids that problem.
    """
    rows = _numeric_rows(path)
    if len(rows) < 2:
        raise ValueError(f'Cannot read summary from {path}')

    r = 0
    misfit = float(rows[r][0]); r += 1
    nlayer = int(round(rows[r][0])); r += 1

    model_rows: list[list[float]] = []
    for k in range(nlayer):
        if r >= len(rows):
            raise ValueError(f'{path}: expected {nlayer} model rows, got {k}')
        row = rows[r]; r += 1
        if len(row) >= 7:
            # Treat the first column as an index only when the row has 7+ columns.
            vals6 = row[1:7]
        elif len(row) >= 6:
            vals6 = row[:6]
        else:
            raise ValueError(f'{path}: model row {k} has too few numeric columns: {row}')
        model_rows.append(vals6)

    if r >= len(rows):
        raise ValueError(f'{path}: missing ns/source-location block after model rows')
    ns = int(round(rows[r][0])); r += 1

    sx, sz = [], []
    for k in range(ns):
        if r >= len(rows):
            raise ValueError(f'{path}: expected {ns} source rows, got {k}')
        row = rows[r]; r += 1
        if len(row) >= 3:
            # Optional event index + sx + sz.
            sx.append(row[1]); sz.append(row[2])
        elif len(row) >= 2:
            sx.append(row[0]); sz.append(row[1])
        else:
            raise ValueError(f'{path}: source row {k} has too few numeric columns: {row}')

    arr = np.asarray(model_rows, dtype=float)
    model = VTIModel(dep=arr[:,0].copy(), alpha=arr[:,1].copy(), beta=arr[:,2].copy(),
                     epsilon=arr[:,3].copy(), gamma=arr[:,4].copy(), delta=arr[:,5].copy())
    geo = Geometry(np.asarray(sx, dtype=float), np.asarray(sz, dtype=float))
    return misfit, model, geo


def read_geometry_dat(path: str | Path, with_receivers: bool = True) -> Geometry:
    vals = float_tokens(path)
    if not vals:
        raise ValueError(f'Cannot read geometry from {path}')
    i = 0
    ns = int(vals[i]); i += 1
    sx = np.zeros(ns); sz = np.zeros(ns)
    for k in range(ns):
        sx[k] = vals[i]; sz[k] = vals[i+1]; i += 2
    rx = rz = None
    if with_receivers and i < len(vals):
        nr = int(vals[i]); i += 1
        rx = np.zeros(nr); rz = np.zeros(nr)
        for k in range(nr):
            rx[k] = vals[i]; rz[k] = vals[i+1]; i += 2
    return Geometry(sx=sx, sz=sz, rx=rx, rz=rz)


def find_existing(*paths: str | Path) -> Path:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    raise FileNotFoundError('None of these paths exist: ' + ', '.join(map(str, paths)))


def step_profile(model: VTIModel, attr: str) -> tuple[np.ndarray, np.ndarray]:
    dep = np.asarray(model.dep, dtype=float)
    val = np.asarray(getattr(model, attr), dtype=float)
    n = len(dep)
    if n < 2:
        return val, dep
    z = [dep[0]]
    x = []
    for k in range(n - 1):
        x.extend([val[k], val[k]])
        if k == 0:
            z = [dep[k], dep[k+1]]
        else:
            z.extend([dep[k], dep[k+1]])
    # Correct duplicate pattern length to nlayer*2-2.
    z = []
    x = []
    for k in range(n - 1):
        z.extend([dep[k], dep[k+1]])
        x.extend([val[k], val[k]])
    return np.asarray(x), np.asarray(z)


def load_chain(result_dir: str | Path):
    path = Path(result_dir) / 'chain.npz'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run vti_joint_mcmc_dram2.py first or point --result-dir to the folder containing chain.npz')
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if 'param_names' in out:
        out['param_names'] = np.asarray(out['param_names']).astype(str)
    return out


def chain_column(chain_pack: dict, name: str) -> np.ndarray | None:
    names = chain_pack.get('param_names')
    if names is None:
        return None
    idx = np.where(names == name)[0]
    if len(idx) == 0:
        return None
    return chain_pack['chain'][:, int(idx[0])]
