# 2D CO₂ plume boundary inversion with normalizing flows and SEM wave simulation

## Overview

This repository implements a **two-stage Bayesian inversion workflow** for probabilistic delineation of a 2D CO₂ plume boundary from waveform data. The inversion combines:

- a **closed cubic B-spline** parameterization for the plume boundary,
- a **normalizing flow** posterior model for flexible variational inference,
- and a **spectral element method (SEM)** forward/adjoint wave simulator for waveform-based likelihood evaluation.

The main script is:

```bash
inversion/RQS42_coupling_flow_2D_model.py
```

The current implementation is designed for a **fixed-sigma two-stage workflow**:

1. **Stage-1** performs inversion with the full set of control points.
2. A **posterior-variance-based pruning + Occam-style refinement** removes weakly supported control points.
3. **Stage-2** restarts inversion in the reduced parameter space using the pruned Stage-1 mean boundary as the new prior mean geometry.

This code is intended for research on **probabilistic CO₂ plume boundary inversion**, especially when the target is to recover both the **mean boundary geometry** and its **posterior uncertainty**.

---

## Repository structure

```text
2D_model/
├── geometry/
│   ├── model_init/
│   └── model_true/
└── inversion/
    ├── RQS42_coupling_flow_2D_model.py   # main inversion script
    ├── model1/
    │   ├── vp_init.npy                   # initial/background velocity model
    │   ├── vp_init_aligned.npy           # aligned background model (generated/used for checks)
    │   ├── vp_true.npy                   # reference/true velocity model used to generate observations
    │   └── vp_meta.json                  # grid/domain metadata
    └── sem_waveform/
        ├── __init__.py
        ├── boundary.py                   # PML and boundary damping
        ├── core.py                       # SEMSimulation core
        ├── mesh.py                       # global mesh construction
        ├── operators.py                  # SEM operators
        ├── receivers.py                  # receiver interpolation setup
        ├── sources.py                    # source wavelet functions
        ├── utils.py                      # numerical utilities
        ├── velocity.py                   # background/anomaly velocity construction
        └── visualization.py              # plotting and output helpers
```

---

## Main idea of the method

### 1. Boundary parameterization

The plume boundary is represented by a **closed cubic B-spline** defined by `K` control points:

- Stage-1 starts from `K_MAX = 20` control points.
- The latent inversion variable has dimension `DIM_Z = 2K`, because each control point has two coordinates `(x, z)`.
- The initial geometry is an ellipse centered at

```text
co2_center = [2800.0, 2250.0]
```

with configurable horizontal and vertical radii.

### 2. Bayesian posterior approximation

The posterior over boundary parameters is approximated by a **normalizing flow**:

- base distribution: multivariate Gaussian in latent space,
- transform stack: ActNorm + piecewise rational quadratic coupling transforms,
- objective: **ELBO** using waveform likelihood and Gaussian prior on control-point offsets.

### 3. Waveform likelihood

For each posterior sample, the code:

- maps latent variables to boundary control-point offsets,
- builds the corresponding velocity model,
- runs the SEM forward simulation,
- compares synthetic and observed traces,
- accumulates the likelihood contribution inside the ELBO.

### 4. Two-stage complexity control

After Stage-1:

- posterior samples are saved,
- per-node posterior variability is analyzed,
- weakly supported control points are pruned,
- an **Occam-style log-joint check** is applied,
- a smaller Stage-2 problem is launched automatically when enabled.

This design keeps Stage-1 flexible, then improves stability and conditioning in Stage-2.

---

## Input data and model files

The default model directory is:

```bash
inversion/model1/
```

### Files

#### `vp_true.npy`
2D reference velocity model used to generate the observed waveform data.

#### `vp_init.npy`
Initial/background velocity model used during inversion.

#### `vp_init_aligned.npy`
Aligned version of the background model. This is useful for consistency checks and plotting.

#### `vp_meta.json`
Metadata describing the velocity grid and physical domain. In the current package, it contains:

- `nx = 1998`
- `nz = 442`
- `xmin = 2.4736601007787726`
- `xmax = 6497.526339899222`
- `zmin = 1000.4634994206258`
- `zmax = 3197.798377752028`
- `dx ≈ 3.2524 m`
- `dz ≈ 4.9826 m`

Array layout is:

```text
cropped_array[nz, nx], row-0=zmin, row-end=zmax, col-0=xmin, col-end=xmax
```

So the first dimension corresponds to depth and the second to horizontal position.

---

## Dependencies

The code imports and uses the following main Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `imageio`

Standard-library modules such as `json`, `os`, `sys`, `time`, `glob`, `pathlib`, and `subprocess` are also used.

A typical installation is:

```bash
pip install numpy scipy matplotlib torch imageio
```

### Notes

- The script uses `matplotlib.use("Agg")`, so it can run in a non-interactive environment.
- The code is written to run on CPU and can also use PyTorch-compatible accelerators when available.
- The local package `sem_waveform/` must remain in the same directory as the main script unless the import path is changed.

---

## How to run

Enter the inversion directory first:

```bash
cd inversion
```

### Basic run

```bash
python RQS42_coupling_flow_2D_model.py
```

By default, this starts **Stage-1**. If `AUTO_RUN_STAGE2=1`, the script will:

1. finish Stage-1,
2. save Stage-1 outputs,
3. prune weak control points,
4. generate `stage1_pruned_ctrl_pts.npy`,
5. launch **Stage-2** automatically.

### Explicit Stage-1 run

```bash
STAGE=1 python RQS42_coupling_flow_2D_model.py
```

### Explicit Stage-2 run

If you want to run Stage-2 manually, you must provide the pruned control points and matching `K2`:

```bash
STAGE=2 \
K2_OVERRIDE=<number_of_pruned_points> \
BASE_CTRL_PATH=stage1_K_20/stage1_pruned_ctrl_pts.npy \
python RQS42_coupling_flow_2D_model.py
```

Replace `<number_of_pruned_points>` with the number of rows in `stage1_pruned_ctrl_pts.npy`.

---

## Important runtime parameters

The script is controlled mainly through environment variables.

### Stage and workflow control

- `STAGE`: `1` or `2`
- `AUTO_RUN_STAGE2`: whether Stage-2 should be launched automatically after Stage-1
- `K2_OVERRIDE`: override the number of control points for Stage-2
- `BASE_CTRL_PATH`: path to the Stage-2 base control points
- `OBS_CACHE_DIR`: directory for cached observed data
- `FORCE_REGEN_OBS_CACHE`: whether to regenerate cached observations

### Geometry and prior settings

- `K_MAX=20`: maximum number of Stage-1 control points
- `ELLIPSE_RX_INIT`: initial ellipse half-width in x
- `ELLIPSE_RZ_INIT`: initial ellipse half-width in z
- `ELLIPSE_ROT_INIT_DEG`: initial ellipse rotation angle
- `FIXED_SIGMA_STAGE1`: prior standard deviation for Stage-1 control-point offsets
- `FIXED_SIGMA_STAGE2`: prior standard deviation for Stage-2 control-point offsets

### SEM domain and acquisition settings

- `DOMAIN_XMIN`, `DOMAIN_XMAX`
- `DOMAIN_ZMIN`, `DOMAIN_ZMAX`
- `NELEM_X`, `NELEM_Z`
- `TOTAL_TIME`
- `DT`
- `POLYNOMIAL_ORDER`
- `PML_THICKNESS`
- `SOURCE_FREQUENCY`
- `SOURCE_AMPLITUDE`
- `RECEIVER_XMIN`, `RECEIVER_XMAX`, `RECEIVER_DX`, `RECEIVER_Z`

### Training settings

- `NUM_EPOCHS_STAGE1`
- `NUM_EPOCHS_STAGE2`
- `LEARNING_RATE`
- `NUM_FLOWS`
- `K_BINS`
- `MIN_ELBO_SAMPLES`
- `MAX_ELBO_SAMPLES`
- `OBS_NOISE_STD`
- `MAX_GRAD_NORM`
- `CLIP_GRADIENT`
- `PLOT_EVERY`
- `PRINT_EVERY`

### Pruning settings

Used after Stage-1 when auto-pruning is enabled:

- `PRUNE_ALPHA`
- `PRUNE_MIN_KEEP`
- `PRUNE_MAX_DROP_FRAC`

---

## Default acquisition and simulation setup

The current default settings in the script are approximately:

### Sources
Six sources at depth `z = 1310 m`:

- `(500, 1310)`
- `(1500, 1310)`
- `(2500, 1310)`
- `(3500, 1310)`
- `(4500, 1310)`
- `(5500, 1310)`

### Receivers
Receivers are placed from `x = 500 m` to `x = 6000 m` with spacing `50 m` at depth:

- `z = 1300 m`

### SEM domain
Default inversion domain parameters:

- `x: 0 – 6500 m`
- `z: 1000 – 3200 m`
- `nelem_x = 46`
- `nelem_z = 25`
- `dt = 9.0e-5 s`
- `total_time = 1.8 s`
- `polynomial_order = 5`
- `PML thickness = 250 m`

The script also reads `vp_meta.json` and synchronizes the physical domain accordingly, which is important for consistency between the SEM domain and the velocity arrays.

---

## Misfit windowing

The code includes a waveform windowing setup called `MISFIT_WINDOW`, which is applied consistently to observations and synthetics. Default settings include:

- `enabled = True`
- `tmin = 0.2`
- `window_len = 1.5`
- `tmax = None` (computed from `tmin + window_len`)
- `decim = 1`

This is useful for reducing cost and focusing the inversion on the part of the signal with the strongest information content.

---

## Typical outputs

Each stage writes results into its own folder:

```text
stage1_K_20/
stage2_K_XX/
```

where `XX` is the Stage-2 control-point count after pruning.

### Important output files

#### `posterior_samples.npy`
Posterior samples of the control-point parameters.

#### `stage1_mean_ctrl_pts_epochXXXX.npy` / `stage2_mean_ctrl_pts_epochXXXX.npy`
Posterior mean control points saved during training.

#### `stage1_pruned_ctrl_pts.npy`
Pruned control points generated after Stage-1 and used to initialize Stage-2.

#### `stage1_elbo_components_raw.npz` / `stage2_elbo_components_raw.npz`
Raw ELBO-component histories for later analysis and plotting.

#### `elbo_history_real.png`
ELBO history plot.

#### `nf_boundary_comparison.png`
Boundary comparison figure showing posterior boundary results.

#### `nf_posterior_distributions.png`
Posterior distribution diagnostics.

#### `occam_prune_log.txt`
Text log of the Occam-style pruning decisions.

In addition, the script may generate diagnostic model images such as:

- `velocity_model_true.png`
- `velocity_model_initial.png`
- `real_noise_comparison.png`

---

## Stage-1 and Stage-2 logic

### Stage-1

Stage-1 solves the inversion with the full control-point set.

Its role is to:

- capture the main plume geometry with high flexibility,
- explore posterior variability,
- identify which control points are strongly or weakly supported by waveform data.

### Pruning between stages

After Stage-1, the script examines posterior samples and computes node-level variability measures. Weakly supported nodes are first screened by score-based pruning, then refined by an **Occam-style expected log-joint test**.

The final kept control points are saved to:

```bash
stage1_K_20/stage1_pruned_ctrl_pts.npy
```

### Stage-2

Stage-2 uses the pruned Stage-1 mean control points as the new base geometry and restarts inversion in a lower-dimensional space.

Its role is to:

- remove unstable or redundant degrees of freedom,
- improve posterior conditioning,
- produce a cleaner and more stable boundary estimate,
- reduce spurious posterior variability.

---

## What this code is good for

This repository is especially useful for:

- probabilistic plume-boundary inversion,
- uncertainty quantification of CO₂ plume extent,
- waveform-based geometric inversion,
- testing reduced-order boundary parameterizations,
- comparing full-DOF and pruned two-stage inversion strategies.

---

## Practical notes

1. **Keep paths consistent.** The main script expects `sem_waveform/` and `model1/` to be located relative to the script.
2. **Do not remove `vp_meta.json`.** It is used to align the SEM domain and velocity grid correctly.
3. **Stage-2 depends on Stage-1 outputs.** If you disable automatic Stage-2 launching, you must manually provide both `BASE_CTRL_PATH` and `K2_OVERRIDE`.
4. **Observation caching matters.** The script stores and reuses synthetic observations through `OBS_CACHE_DIR`, which can save substantial runtime.
5. **This is a research codebase.** Some parameters are exposed directly near the top of the script for easier experimentation.

---

## Minimal example

Run the default two-stage workflow:

```bash
cd inversion
AUTO_RUN_STAGE2=1 python RQS42_coupling_flow_2D_model.py
```

Run Stage-1 only:

```bash
cd inversion
STAGE=1 AUTO_RUN_STAGE2=0 python RQS42_coupling_flow_2D_model.py
```

Run Stage-2 manually after Stage-1:

```bash
cd inversion
STAGE=2 \
K2_OVERRIDE=12 \
BASE_CTRL_PATH=stage1_K_20/stage1_pruned_ctrl_pts.npy \
python RQS42_coupling_flow_2D_model.py
```

---

## Disclaimer

This repository is a research implementation. It is intended for method development, numerical experiments, and reproducible inversion tests rather than production deployment. Users are encouraged to verify configuration choices, domain alignment, and inversion settings before drawing scientific conclusions.

---

## Contact

xingdaj@mun.ca

