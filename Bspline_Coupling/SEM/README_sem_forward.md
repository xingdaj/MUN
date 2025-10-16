# SEM Forward Simulation (Spectral Element Method) — README

> ****  
> `sem_forward.py` is the *main entry point*. It builds a mesh, constructs a B‑spline–defined two‑region velocity model, sets up a source and circular array of receivers, applies CFL‑safe time‑stepping with PML absorbing boundaries, and saves seismograms and figures to `sem_output/`.

---

## 1) Project structure & roles

```
sem_forward.py          # Main script: config, run, save, and plotting
sem_waveform/
  core.py               # SEMSimulation: end‑to‑end forward engine (NumPy/Torch)
  mesh.py               # Global GLL mesh and connectivity + nearest‑node helper
  operators.py          # SEM operators (Gx, Gz, S_interp, w2D), mass/stiffness, K@u
  receivers.py          # Receiver geometry + sparse interpolation (NumPy/Torch)
  sources.py            # Ricker + DC removal + time gating utilities
  boundary.py           # PML construction and damping (NumPy/Torch)
  velocity.py           # B‑spline interface → signed distance → nodal velocities
  utils.py              # GLL utilities, differentiation, CFL checks & dt adjust
  visualization.py      # Velocity model plotting and optional GIF helpers
```

---

## 2) End‑to‑end pipeline

1. **Define configuration** in `sem_forward.py` under `config` (domain, time, source, receivers, method, velocity, output).  
2. **Instantiate** `SEMSimulation(config)` and call `sim.run()`.  
3. **Outputs**: receiver seismograms (`.npz`), a montage of traces (`sem_seismograms.png`), and a velocity‐model figure with the “true” boundary (`velocity_model_true.png`). Optional wavefield snapshots/GIFs when enabled.  

Key artifacts are written to `config['output']['output_dir']` (default `sem_output/`).

---

## 3) Configuration reference (what each section controls)

- **domain**: spatial extents and element counts (`xmin/xmax/zmin/zmax`, `nelem_x/nelem_z`).  
- **time**: `total_time` and a *candidate* `dt`; `dt` will be adjusted for CFL stability at runtime.  
- **source**: `(x,z)` position, peak frequency `frequency`, and `amplitude`.  
- **receivers**: number and circular `radius` around the source.  
- **method**: SEM polynomial order `polynomial_order` and `pml_thickness` (meters).  
- **velocity**: two constants (`inside_velocity`, `outside_velocity`) and a **closed cubic B‑spline** interface from 6 control points (`control_points`) plus optional `perturbations` (a “true” offset); `tau` controls sigmoid softness of the interface; `spline_samples` controls SDF refinement.  
- **output**: directory, booleans for saving wavefield / seismograms / live visualization, and optional snapshot cadence.

> **Tip**: All of the above are consumed by `SEMSimulation._initialize_components()` and re‑used in `run()` after the velocity is updated from the B‑spline control points.

---

## 4) Numerics & implementation highlights

### 4.1 Mesh (GLL grid)
- Builds a structured quadrilateral mesh of GLL points of order `p` with global connectivity.
- Provides `find_closest_node(x,z, …)` for locating the source and for sanity warnings when off‑node.

### 4.2 SEM operators & matrices
- Precomputes **gradient** operators `Gx, Gz`, a **node→quadrature interpolation** `S_interp`, and 2‑D **weights** `w2D` for each element.
- Assembles a **lumped mass** diagonal (NumPy or Torch) and either assembles `K` once (Torch sparse COO) or applies `K@u` **matrix‑free** per step (NumPy).

### 4.3 Velocity via B‑spline interface
- A *closed* cubic B‑spline is constructed from 6 control points by duplicating the first point three times and using open knot vectors.
- A **signed distance field** to the curve is computed with Newton refinement starting from uniformly sampled parameters; the sign is chosen by a point‑in‑polygon test.
- Velocities are blended from `inside_velocity`/`outside_velocity` using a smooth sigmoid controlled by `tau`.

### 4.4 CFL‑safe time step
- The candidate `dt` from config is **checked and possibly reduced** according to a SEM CFL bound using mesh size, polynomial order, and the maximum velocity. The loop runs with the *final* `dt`, and dependent pieces (source wavelet, PML decay) are rebuilt accordingly.

### 4.5 Absorbing boundaries (PML)
- A cubic ramp PML is built on all four sides with target reflection `R` and thickness `pml_thickness`.  
- Decay factors are precomputed once and applied **in‑place** each step (NumPy mask or Torch indexed variant).

### 4.6 Time stepping & receivers
- Central‑difference update: `u_{t+1} = 2u_t − u_{t−1} + dt^2 * M^{-1} ( −Ku_t + f_t )` with a collocated point source (scaled by local mass).  
- Receiver traces are collected each step via a **sparse interpolation matrix** (`R_sparse`/`R_torch`).

---

## 5) What gets saved

- **Seismograms**: `sem_output/sem_forward_results.npz` with keys:
  - `receiver_data` (shape `nt × nrec`), `dt` (scalar), `nt` (scalar), `receiver_coords` (`nrec × 2`).
- **Figures**:
  - `sem_seismograms.png` — tiled receiver waveforms.
  - `velocity_model_true.png` — “true” B‑spline boundary over colored nodal velocities.
  - `sem_velocity_model.png` — snapshot of the model used by the simulation (when requested).
- **Wavefield** (optional): snapshots in ROI and an animated GIF if enabled.

---

## 6) Quick start

1. Install Python 3.10+ and packages: `numpy`, `scipy`, `matplotlib`, `torch`, `imageio`.  
2. Adjust `config` in `sem_forward.py` if needed.  
3. Run:
   ```bash
   python sem_forward.py
   ```
4. Inspect `sem_output/` for `.npz` and figures.

---

## 7) Troubleshooting & tips

- **CFL warnings**: If you see stability warnings, the code will internally reduce `dt`. You can also lower `dt` in the config or reduce `polynomial_order` / increase element sizes.
- **PML strength**: `pml_thickness` controls attenuation; ensure it’s large enough for your domain and dominant wavelength. The code requires `c_max` (here set from the configured `outside_velocity`) when building PML.
- **Source off‑node warning**: “Closest node is X m away…” just means the exact source coordinate isn’t a mesh node; it’s informational unless the offset is large.
- **NumPy vs Torch path**: `SEMSimulation.run(use_torch=True)` executes a Torch time stepping loop with a sparse COO global stiffness matrix and Torch‑native receiver interpolation. Set `use_torch=False` to use the NumPy matrix‑free path.

---

## 8) Extending

- **Change geometry**: Modify `control_points` and `perturbations` to move the boundary; adjust `tau` for sharper/softer transitions.
- **Different sources**: Add new wavelets in `sources.py` or change `t0`, `gate_time`, `frequency`.
- **More/other receivers**: Adjust `num_receivers` and `radius`, or implement custom layouts in `receivers.py`.
- **Performance**: Increase snapshot interval, disable visualization for speed, or run the Torch path (which can be ported to GPU with minor changes).
- **Adjoint/gradients**: The code base already contains kernels used by gradient calculations; see `core.py` functions for accumulating sensitivities.

---

## 9) Key APIs (by file)

- **`sem_forward.py`**: defines `config`, runs `SEMSimulation(config).run()`, saves/plots artifacts.
- **`core.py`** (`SEMSimulation`):
  - `_initialize_components()`: mesh → initial velocity → CFL dt → source/receivers/operators/PML.
  - `run(ctrl_offsets_flat=None, use_torch=True)`: rebuilds velocity from control points, re‑checks CFL/PML, and dispatches to NumPy or Torch stepping.
  - `_time_stepping_loop_numpy()` / `_time_stepping_loop_torch()`: main loops; apply PML, record receivers, take snapshots, update viz.
- **`mesh.py`**: `create_global_mesh(...)`, `find_closest_node(...)`.
- **`operators.py`**: `precompute_sem_operators(...)`, `assemble_mass_matrix_lumped_*`, `assemble_stiffness_matrix_*`, and `stiffness_matrix_vector_product(_torch)`.
- **`velocity.py`**: `_closed_bspline_from_ctrl(...)`, `signed_distance_to_spline_newton(...)`, `build_velocity_on_sem_nodes(...)`.
- **`receivers.py`**: `setup_receivers(...)` builds both SciPy CSR and Torch sparse matrices for interpolation; `interpolate_to_receivers_*`.
- **`sources.py`**: `ricker_wavelet(...)`, `create_source_wavelet(...)` (DC‑removed and time‑gated).
- **`boundary.py`**: `setup_pml(...)` and fast in‑place `apply_pml_*` helpers.
- **`utils.py`**: GLL points/weights, barycentric differentiation, CFL checks and `adjust_time_step_for_stability(...)`.
- **`visualization.py`**: `plot_sem_velocity_model(...)` and helpers to persist wavefield/GIF.

---

## 10) Repro checklist

- Set domain, time, method (order, PML), velocity (inside/outside, B‑spline), source, receivers, output.  
-  Run `sem_forward.py`.  
-  Verify `dt` log and “CFL OK” message; inspect `sem_output/*.png` and `sem_forward_results.npz`.  
-  If integrating with inference, reuse the same `config` so forward calls and gradient checks match.
