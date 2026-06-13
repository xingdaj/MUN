# Fixed-K Normalizing Flow Inversion with SEM Waveform Forward Modelling

This repository contains a cleaned, reproducible version of the fixed-K closed B-spline anomaly inversion code.
It uses a single-stage Rational Quadratic Spline Normalizing Flow (RQS-NF) and a local SEM waveform simulator.

xingdaj@mun.ca  
2026.6.12

## Repository layout

```text
RQS_NF.py                 Main inversion script
shell_run.py              One-click runner: regenerate PML models, clear obs_cache, run RQS_NF.py
sem_waveform/             Local SEM forward/adjoint simulation package
initial1/background/      Background PML model generator
initial1/true/            True PML model generator
initial1/initial/         Initial PML model generator
model1/                   Current PML model files used by RQS_NF.py
```

## Recommended run

```bash
python shell_run.py
```

`shell_run.py` does three things in order:

1. regenerates the background, true, and initial PML models using the same boundary-to-grid mapping;
2. copies only the PML `.npy/.json` files required by the inversion into `model1/`;
3. removes `obs_cache/` and runs `RQS_NF.py` with `FORCE_REGEN_OBS_CACHE=1`.

## Detailed description of `RQS_NF.py`

`RQS_NF.py` is the main fixed-K inversion program. It treats the anomaly boundary as a closed periodic B-spline curve and uses a Rational Quadratic Spline Normalizing Flow to approximate the posterior distribution of the B-spline control-point offsets. The fixed-K setting means that the number of B-spline nodes is read from the initial model metadata and is not changed during the run.

### Inversion unknowns

The unknown vector is

```text
z = [dx_1, dz_1, dx_2, dz_2, ..., dx_K, dz_K]
```

where `K` is the number of B-spline control points in `vp_initial_pml.json`.

- `ctrl_pts_init_base` is the fixed initial/base control polygon.
- A sampled vector `z` gives a physical offset in metres.
- The current boundary used by SEM is `ctrl_pts_init_base + z.reshape(K, 2)`.
- The NF dimension is therefore `DIM_Z = 2 * K`.
- The closed B-spline is currently built with cubic degree `k = 3`.

### Forward model used in both observation generation and inversion

The script deliberately uses the same model construction for synthetic observations and inversion:

```text
velocity model = vp_background_pml + periodic B-spline signed-distance sigmoid anomaly
```

This is important because it avoids the inconsistency of generating observations from a finished `vp_true_pml.npy` grid while inverting with a boundary-to-grid anomaly mapping. Before running, `RQS_NF.py` checks that the true, initial, and background model metadata are consistent in grid bounds, grid shape, grid spacing, layer interfaces, layer velocities, and smoothing/mapping settings.

The SEM configuration is built by `build_base_sem_config()` and `_make_obs_and_inv_configs()`. For each source, the local `sem_waveform.core.SEMSimulation` simulator is called. The likelihood uses `run_forward_and_adjoint()` so that SEM adjoint gradients with respect to the B-spline control points can be passed back into PyTorch.

### Observation cache

Observed data are stored in `obs_cache/`. The cache file name includes the source/receiver geometry, time sampling, source frequency, source amplitude, noise level, PML thickness, polynomial order, mesh size, anomaly parameters, and true control points. This reduces the chance of accidentally reusing stale data after changing acquisition or model parameters.

Set

```bash
FORCE_REGEN_OBS_CACHE=1
```

to force observation regeneration.

### Normalizing-flow architecture

The posterior approximation is a single-stage RQS-NF:

```text
base Gaussian -> [ActNorm -> RQS coupling -> permutation] x NUM_FLOWS -> z samples
```

The main components are:

- `ActNorm`: initializes a per-dimension shift and scale from the first sampled batch and contributes the corresponding log-determinant.
- `FixedPermutation`: applies a fixed random permutation between coupling layers to improve dimension mixing.
- `PiecewiseRationalQuadraticCoupling`: implements a monotonic rational-quadratic spline coupling transform with linear tails over `[-FLOW_TAIL_BOUND, FLOW_TAIL_BOUND]`.
- `NormalizingFlow`: stacks `NUM_FLOWS` ActNorm and RQS coupling layers and provides both sampling (`forward`) and density evaluation (`log_prob`).

Each RQS coupling layer uses a small fully connected conditioner network:

```text
Linear(in_features, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, out_features)
```

The number of spline bins is controlled by `K_BINS`.

### ELBO objective

The optimized objective is the single-stage fixed-K ELBO:

```text
ELBO = E_q[ log p(y | z) + log p(z) + log p_fuse(z) - log q(z) ]
```

where:

- `log p(y | z)` is the SEM waveform likelihood computed using the multi-source forward/adjoint calculation;
- `log p(z)` is a fixed zero-mean Gaussian prior on physical control-point offsets, with standard deviation `FIXED_SIGMA`;
- `log p_fuse(z)` is an optional cyclic adjacent-node fusion penalty controlled by `FUSE_LAMBDA_EFF`;
- `log q(z)` is the NF posterior density.

By default, `FUSE_LAMBDA_EFF=0.0`, so the fusion penalty is disabled. The likelihood variance is controlled by `OBS_NOISE_STD`, with a small numerical `EPS` added to the variance.

### Training loop

The training loop performs the following steps:

1. load and validate `model1/` files;
2. load initial and true B-spline control points from JSON metadata;
3. build the SEM observation and inversion configurations;
4. generate or load cached noisy/clean observations;
5. initialize the RQS-NF posterior model;
6. maximize the ELBO using Adam;
7. monitor ELBO components and gradients;
8. save posterior boundary snapshots every `PLOT_EVERY` epochs;
9. draw final posterior samples and save diagnostic plots/data.

The number of Monte Carlo samples used in the ELBO is annealed from `MIN_ELBO_SAMPLES` to `MAX_ELBO_SAMPLES` over the training run.

### Gradient handling

Gradients from the SEM adjoint enter the PyTorch graph through `SEMLikelihoodAdjointFn`. During backpropagation, the script records gradient diagnostics before and after clipping. If `CLIP_GRADIENT=1`, gradients are rescaled when the total gradient norm exceeds `MAX_GRAD_NORM`.

### Main diagnostic outputs from `RQS_NF.py`

At runtime, the script writes both model-check figures and posterior diagnostics, including:

```text
velocity_model_true.png
velocity_model_initial.png
source_waveforms/source_XX_waveforms.png
posterior_model/fixedK_true_ctrl_pts.npy
posterior_model/fixedK_initial_ctrl_pts.npy
posterior_model/fixedK_diagnostic_config.json
posterior_model/fixedK##_diagnostic/snapshot_iteration_XXXX.png
posterior_model/fixedK##_diagnostic/boundary_only_iteration_XXXX.png
posterior_model/fixedK##_diagnostic/mean_ctrl_pts_iteration_XXXX.npy
posterior_model/fixedK##_diagnostic/posterior_samples.npy
posterior_model/fixedK##_diagnostic/mean_ctrl_pts_final.npy
posterior_model/fixedK##_diagnostic/elbo_history_real.png
posterior_model/fixedK##_diagnostic/elbo_components_raw.npz
posterior_model/fixedK##_diagnostic/gradient_history_real.png
posterior_model/final_mean_ctrl_pts.npy
posterior_model/final_posterior_samples.npy
posterior_model/final_boundary_summary.png
posterior_model/final_boundary_only_true_vs_updated.png
posterior_model/global_iteration_history.csv
posterior_model/run_summary.json
posterior_model/fixedK_elbo_components_global.png
```

The most useful quick-check figures are usually:

- `velocity_model_true.png`: confirms the true model and acquisition geometry;
- `velocity_model_initial.png`: confirms the initial/base model;
- `posterior_model/fixedK##_diagnostic/snapshot_iteration_XXXX.png`: shows posterior evolution;
- `posterior_model/final_boundary_summary.png`: summarizes the final posterior mean, posterior samples, and true boundary;
- `posterior_model/fixedK_elbo_components_global.png`: shows the ELBO-component evolution;
- `posterior_model/fixedK##_diagnostic/gradient_history_real.png`: checks gradient stability and clipping.

## Main parameters

Most parameters are near the top of `RQS_NF.py` and can also be overridden with environment variables.

### Model and prior

- `MODEL_DIR`: directory containing the PML model files, default `model1/`.
- `VP_SMOOTH_PATH`: true PML model grid, default `model1/vp_true_pml.npy`.
- `VP_META_PATH`: true PML metadata, default `model1/vp_true_pml.json`.
- `VP_PRE_PATH`: initial PML model grid, default `model1/vp_initial_pml.npy`.
- `VP_PRE_META_PATH`: initial PML metadata, default `model1/vp_initial_pml.json`.
- `VP_BACKGROUND_PATH`: background PML model grid, default `model1/vp_background_pml.npy`.
- `VP_BACKGROUND_META_PATH`: background PML metadata, default `model1/vp_background_pml.json`.
- `FIXED_SIGMA`: Gaussian prior scale for control-point offsets, default `300.0` m.
- `FUSE_LAMBDA_EFF`: optional adjacent-node fusion penalty weight, default `0.0`.

### SEM acquisition and simulation

- `NELEM_X`, `NELEM_Z`: SEM element numbers in x and z, default `30`, `30`.
- `TOTAL_TIME`: total simulation time, default `1.5` s.
- `DT`: time step, default `0.80e-4` s.
- `POLYNOMIAL_ORDER`: SEM polynomial order, default `5`.
- `PML_THICKNESS`: PML thickness, default `200.0` m.
- `SOURCE_FREQUENCY`: Ricker/source frequency, default `15.0` Hz.
- `SOURCE_AMPLITUDE`: source amplitude, default `1.0e4`.
- `RECEIVER_XMIN`, `RECEIVER_XMAX`, `RECEIVER_DX`, `RECEIVER_Z`: receiver-line controls, default `0`, `2000`, `20`, `20.0`.

### NF and optimization

- `NUM_EPOCHS`: total training iterations, default `150`.
- `LEARNING_RATE`: Adam learning rate, default `5e-4`.
- `NUM_FLOWS`: number of ActNorm + RQS coupling blocks, default `16`.
- `K_BINS`: number of RQS bins per transformed dimension, default `12`.
- `FLOW_TAIL_BOUND`: RQS coupling tail bound in metres. By default it is `1.5 * FIXED_SIGMA`.
- `MIN_ELBO_SAMPLES`: initial number of Monte Carlo samples per ELBO estimate, default `4`.
- `MAX_ELBO_SAMPLES`: final number of Monte Carlo samples per ELBO estimate, default `8`.
- `OBS_NOISE_STD`: observation noise standard deviation used by the likelihood, default `1.0e-4`.
- `EPS`: small numerical value added to the likelihood variance, default `1e-30`.
- `MAX_GRAD_NORM`: gradient clipping threshold, default `100.0`.
- `CLIP_GRADIENT`: enable gradient clipping when set to `1`, default `1`.

### Plotting and output

- `PLOT_EVERY`: save posterior/boundary snapshots every N epochs, default `5`.
- `PLOT_SNAPSHOT_SAMPLES`: posterior samples used for intermediate plots, default `10000`.
- `FINAL_POST_SAMPLES`: final posterior samples saved after training, default `10000`.
- `POSTERIOR_OUTDIR`: posterior output directory, default `posterior_model`.
- `SAVE_TRACE_COMPARISON`: save clean/noisy waveform comparison figures, default `1`.
- `TRACE_COMPARISON_DIR`: waveform figure directory, default `source_waveforms`.

Example:

```bash
NUM_EPOCHS=200 OBS_NOISE_STD=1e-4 SOURCE_AMPLITUDE=1e4 python shell_run.py
```

## Notes on consistency

The true, initial, and background models are generated with consistent grid, PML, layer, and signed-distance sigmoid mapping metadata. `RQS_NF.py` checks this metadata before running. Observation data are generated from `vp_background_pml + true B-spline anomaly`, which is the same model construction seen by the inversion.

For clean tests, it is recommended to regenerate the observation cache after changing any of the following:

- source/receiver geometry;
- source frequency or amplitude;
- `TOTAL_TIME` or `DT`;
- `OBS_NOISE_STD`;
- PML thickness or SEM polynomial order;
- true or initial B-spline control points;
- anomaly velocity, anomaly smoothing, or boundary-to-grid mapping.

Using `shell_run.py` is the safest route because it regenerates the model files, clears `obs_cache/`, and then runs the inversion.

## Generated outputs

Typical runtime outputs are intentionally not tracked in the clean archive:

- `obs_cache/`
- `sem_output/`
- `posterior_model/`
- `source_waveforms/`
- `fixedK_diagnostics/`

These folders are regenerated during execution.
