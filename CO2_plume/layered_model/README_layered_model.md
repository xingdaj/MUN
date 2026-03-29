# Probabilistic CO₂ Plume-Boundary Delineation with SEM and Normalizing Flows

This repository contains a **two-stage Bayesian full-waveform inversion (FWI)** workflow for **direct CO₂ plume-boundary delineation** in a layered acoustic model.

Instead of inverting for a full pixel-based velocity field, the plume geometry is represented by a **closed cubic B-spline boundary**. The posterior distribution of the boundary is inferred using a **coupling-flow-based normalizing flow**, while waveform simulation and adjoint gradients are computed with a **spectral element method (SEM)** backend.

The code is designed to:

- infer the **posterior distribution** of a plume boundary from seismic waveform data,
- quantify **geometric uncertainty** along the inferred boundary,
- reduce redundant degrees of freedom using **score-threshold pruning + Occam evidence check**, and
- retrain a reduced model in **Stage-2** for a more stable and compact posterior representation.

---

## Repository structure

```text
layered_model/
├── geometry/
│   ├── geo.py
│   └── geo_smooth.py
└── inversion/
    └── RQS_Coupling_layered_model.py
```

### `inversion/RQS_Coupling_layered_model.py`
Main inversion script.

It includes:

- synthetic true-model construction,
- SEM forward simulation for observed data generation,
- normalizing-flow variational inference,
- Stage-1 training,
- pruning and Occam evidence check,
- Stage-2 retraining,
- posterior visualization and output saving.

### `geometry/geo.py`
Utility script to generate and visualize a **layered velocity model with a hard-boundary anomaly**.

### `geometry/geo_smooth.py`
Utility script to generate and visualize a **layered velocity model with a smooth-boundary anomaly**.

---

## Method overview

### Parameterization

The plume boundary is parameterized by a **closed cubic B-spline**.

- The **true model** uses `K1 = 6` control points.
- The **inversion model** uses an over-parameterized boundary with up to `K_MAX = 20` control points.
- Each control point is represented through 2D offsets relative to an initial elliptical prior geometry.

### Stage-1

Stage-1 performs variational inference in the full parameter space.

- A coupling-flow normalizing flow is trained to approximate the posterior.
- SEM forward and adjoint calculations provide waveform-based likelihood information.
- Posterior samples, ELBO histories, and boundary snapshots are saved.

### Pruning and Occam check

After Stage-1:

1. nodes are screened using a **score-threshold rule** based on posterior variability,
2. remaining candidates are tested with an **Occam evidence check**, and
3. only nodes whose removal does **not reduce** the expected log-joint are pruned.

This produces a reduced control-point set for Stage-2.

### Stage-2

Stage-2 retrains the inversion using the pruned geometry.

This stage typically produces:

- a smoother posterior mean boundary,
- reduced posterior spread,
- a more compact parameterization, and
- improved stability in weakly constrained regions.

---

## Main dependencies

The script relies on standard scientific Python packages plus a local SEM backend.

### Python packages

Typical required packages include:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`

### Local module dependency

The inversion script imports:

```python
from sem_waveform.core import SEMSimulation
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_velocity_layered_with_anomaly
```

So the repository must also provide a compatible local package named **`sem_waveform`**.
That package is **not included in the current folder snapshot**, so you need to place it in the Python path or in the same project tree before running the inversion.

---

## How to run

### 1. Geometry-only visualization

Generate a layered model with a hard anomaly boundary:

```bash
python geometry/geo.py
```

Generate a layered model with a smooth anomaly boundary:

```bash
python geometry/geo_smooth.py
```

### 2. Run Stage-1 only

```bash
STAGE=1 AUTO_RUN_STAGE2=0 python inversion/RQS_Coupling_layered_model.py
```

### 3. Run Stage-1 and automatically launch Stage-2

```bash
STAGE=1 AUTO_RUN_STAGE2=1 python inversion/RQS_Coupling_layered_model.py
```

### 4. Run Stage-2 manually

```bash
STAGE=2 BASE_CTRL_PATH=stage1_K_20/stage1_pruned_ctrl_pts.npy python inversion/RQS_Coupling_layered_model.py
```

`BASE_CTRL_PATH` should point to the control-point file used to initialize Stage-2.
In this script, Stage-2 preferentially uses:

- `stage1_mean_ctrl_pts.npy`,
- `stage1_pruned_ctrl_pts.npy`,
- or other legacy Stage-1 control-point files if available.

---

## Important environment variables

The script is configured primarily through environment variables.
Below are the most important ones.

### Stage control

```bash
STAGE=1
AUTO_RUN_STAGE2=1
BASE_CTRL_PATH=...
```

- `STAGE`: choose `1` or `2`
- `AUTO_RUN_STAGE2`: whether to auto-launch Stage-2 after Stage-1
- `BASE_CTRL_PATH`: control-point file used to initialize Stage-2

### Prior / geometry

```bash
TSTD_DEFAULT=50.0
FIXED_SIGMA_STAGE1=50.0
FIXED_SIGMA_STAGE2=20.0
K2_OVERRIDE=20
```

- `TSTD_DEFAULT`: standard deviation used to generate synthetic true offsets
- `FIXED_SIGMA_STAGE1`: fixed Gaussian prior scale for Stage-1
- `FIXED_SIGMA_STAGE2`: fixed Gaussian prior scale for Stage-2
- `K2_OVERRIDE`: override the inversion control-point count

### Training

```bash
NUM_EPOCHS_STAGE1=200
NUM_EPOCHS_STAGE2=100
LEARNING_RATE=5e-4
NUM_FLOWS=16
PRINT_EVERY=1
PLOT_EVERY=10
```

### SEM / acquisition

```bash
DOMAIN_XMIN=-300
DOMAIN_XMAX=2000
DOMAIN_ZMIN=-300
DOMAIN_ZMAX=1000
NELEM_X=30
NELEM_Z=30
TOTAL_TIME=1.2
DT=0.80e-4
POLYNOMIAL_ORDER=5
SOURCE_FREQUENCY=20.0
```

Receiver and source locations are also defined near the top of the script.

---

## Typical outputs

The inversion script writes figures, cached observations, posterior samples, and Stage-specific diagnostics.

### Common figures

- `Bspline_model_structure_real.png`
- `velocity_model_true.png`
- `real_noise_comparison.png`

### Observation cache

Stored in:

```text
obs_cache/
```

Typical files:

- `noisy_data_real.npy`
- `clean_data_real.npy`
- `obs_meta.json`

### Stage output directory

The script writes Stage-specific outputs into a folder such as:

```text
stage1_K_20/
stage2_K_xx/
```

Typical files include:

- `posterior_samples.npy`
- `stage1_mean_ctrl_pts.npy` or `stage2_mean_ctrl_pts.npy`
- `stage1_mean_offset.npy` or `stage2_mean_offset.npy`
- `elbo_components.png`
- `nf_posterior_distributions.png`
- `nf_boundary_comparison.png`
- boundary snapshots saved every `PLOT_EVERY` epochs
- `occam_prune_log.txt`
- `stage1_pruned_ctrl_pts.npy`

---

## Notes on reproducibility

- Random seeds are controlled through `SEED`.
- CPU threading can be controlled through `CPU_NUM_THREADS`.
- The script uses cached observations when the cache signature matches the current configuration.
- If the acquisition setup, model geometry, or noise settings change, observations are regenerated automatically.

---

## Notes for GitHub users

If you plan to share this code publicly, it is a good idea to add the following files in addition to this README:

- `requirements.txt` or `environment.yml`
- a short description of the missing `sem_waveform` dependency
- one example figure in the repository root
- a `.gitignore` file for large outputs such as `obs_cache/`, `stage1_K_*/`, and `stage2_K_*/`

A minimal `.gitignore` could include:

```gitignore
__pycache__/
*.pyc
.DS_Store
obs_cache/
stage1_K_*/
stage2_K_*/
*.npy
*.npz
```

---

## Citation / acknowledgment

If you use this code in academic work, please cite the corresponding paper or preprint describing the method.

Suggested description:

> Two-stage probabilistic CO₂ plume-boundary delineation using spectral-element full-waveform inversion and coupling-flow-based variational inference.

---

## Contact

Contact email in the source code:

**xingdaj@mun.ca**
