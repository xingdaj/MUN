# Smooth-Model VTI DAS Microseismic Joint-Inversion Example

This folder contains a complete Python workflow for a synthetic VTI DAS microseismic inversion test with **model-parameterization mismatch**. The forward observations are generated from a fine-layer smooth/complex VTI truth model, while the inversion uses a simplified layered effective VTI model. The purpose is to test whether qP differential-arrival data from an L-shaped DAS array can still provide robust event locations when the true subsurface structure is smoother and more complex than the inversion parameterization.

## 1. Main objective

The workflow performs the following experiment:

1. Generate a synthetic L-shaped DAS geometry and 25 microseismic events.
2. Generate two VTI velocity models:
   - `vel.dat`: simplified layered VTI model used as the inversion parameterization.
   - `vel1.dat`: fine-layer smooth/complex VTI model used only as the forward truth.
3. Compute synthetic direct-wave travel times from `vel1.dat`.
4. Add Gaussian picking noise to the synthetic qP arrivals.
5. Estimate initial event locations by qP grid search.
6. Build prior bounds and proposal scales.
7. Run DRAM MCMC to jointly invert event locations, layered VTI parameters, and layer-interface depths.
8. Analyze posterior samples and compare the posterior mean/best forward predictions with the noisy observations.

This example is therefore designed to evaluate how well a simplified layered VTI model can act as an effective kinematic model for event location when the actual medium is smooth and more complex.

## 2. Folder structure

```text
Smooth_model/
├── shell_simple.py
├── 01-input/
│   └── direct.py
├── 02-forward/
│   ├── vti_direct.py
│   └── plot_forward_validation.py
├── 04-initial/
│   ├── vel_add.py
│   ├── noise_add.py
│   ├── geo_grid_search.py
│   ├── geo_add.py
│   └── prior_prop.py
├── 05-inversion/
│   └── vti_joint_mcmc_dram.py
└── 06-result/
    ├── result_analysis.py
    ├── loc_plot.py
    ├── vel_plot.py
    ├── forward_compare_mean_best.py
    └── vti_plot_utils.py
```

The recommended entry point is:

```bash
python shell_simple.py
```

` shell_simple.py` runs all major steps in the correct order and passes consistent parameters to the downstream scripts.

## 3. Required Python packages

The code mainly uses standard Python libraries plus:

```bash
numpy
matplotlib
pandas
```

A typical installation command is:

```bash
pip install numpy matplotlib pandas
```

The full MCMC run can be computationally expensive. For a quick code check, reduce `MCMC_ITERATIONS` and `MCMC_BURNIN` in `shell_simple.py` before running the full experiment.

## 4. End-to-end workflow

### Step 1: generate geometry and VTI models

Script:

```text
01-input/direct.py
```

Main outputs:

```text
01-input/output/control.dat
01-input/output/geometry.dat
01-input/output/vel.dat
01-input/output/vel1.dat
01-input/output/geo.png
01-input/output/vel.png
01-input/output/vel1.png
01-input/output/vel_compare.png
```

Important model distinction:

- `vel.dat` is the simplified layered VTI model. It defines the effective inversion parameterization.
- `vel1.dat` is the fine-layer smooth/complex VTI model. It is used only to synthesize the observations.

In this workflow, `shell_simple.py` creates a separate temporary forward-truth input directory:

```text
01-input/output_forward_truth/
```

Inside this directory, `vel1.dat` is copied as `vel.dat` only for forward modelling. This avoids overwriting the original layered `01-input/output/vel.dat` used by the inversion.

### Step 2: compute forward travel times

Script:

```text
02-forward/vti_direct.py
```

Main outputs:

```text
03-output/qx.dat
03-output/ttime.dat
03-output/diagnostics.dat
03-output/layer_contributions.dat
03-output/iteration.dat
03-output/iteration_detailed.dat
03-output/input_summary.dat
03-output/input_geometry_echo.dat
03-output/input_velocity_echo.dat
```

The forward solver computes direct-wave qP, qSV, and qSH travel times in layered VTI media. For each source-receiver pair, it solves the horizontal ray parameter `qx` from the horizontal-offset equation. After `qx` is obtained, the layerwise travel-time contributions are summed to obtain the total arrival time.

The deterministic forward run is strict by design. It records convergence information, offset residuals, and layer contributions so that numerical problems are not silently hidden.

### Step 3: build initial model and noisy observations

Scripts:

```text
04-initial/vel_add.py
04-initial/noise_add.py
04-initial/geo_grid_search.py
04-initial/prior_prop.py
```

Main outputs:

```text
04-initial/output/vel.dat
04-initial/output/nobs.dat
04-initial/output/geo.dat
04-initial/output/prior.dat
04-initial/output/prop.dat
```

The initialization procedure has four parts:

1. `vel_add.py` perturbs the layered model `01-input/output/vel.dat` to generate the initial inversion model.
2. `noise_add.py` adds Gaussian picking noise to the synthetic arrivals from `03-output/ttime.dat`.
3. `geo_grid_search.py` estimates the initial event locations by qP grid search using the noisy observations and the perturbed initial VTI model.
4. `prior_prop.py` writes the prior bounds and scalar proposal standard deviations used by the MCMC sampler.

The grid-search initialization is important because event locations are inverted. The source locations in `geo.dat` are not simply random perturbations of the true sources; they are estimated from the observed qP arrivals.

### Step 4: run MCMC joint inversion

Script:

```text
05-inversion/vti_joint_mcmc_dram.py
```

Main outputs:

```text
06-result/output/chain.npz
06-result/output/mean.dat
06-result/output/best.dat
06-result/output/misfit.png
```

The inversion uses a component-wise delayed-rejection adaptive Metropolis method. Each MCMC iteration updates one scalar parameter. If the first proposal is rejected, a smaller second-stage proposal is attempted. Proposal standard deviations can be adapted during the early part of the chain and are then fixed for posterior sampling.

The inverted unknowns include:

- event locations: `sx`, `sz` for all events;
- layer-interface depths: `dep[1]`, `dep[2]`, ... when `INVERT_DEPTHS = True`;
- VTI parameters per layer: `alpha`, `beta`, `epsilon`, `gamma`, and `delta`;
- optionally, the bottom half-space can be fixed with `FIX_LAST_LAYER = True`.

The default objective in this workflow is qP adjacent-channel differential arrival time:

```text
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_USE_WAVES = "P"
```

This means the likelihood is built from qP station-pair differential arrivals rather than raw absolute arrival times. The adjacent-channel differencing follows the DAS moveout-based formulation used in the manuscript workflow.

### Step 5: posterior analysis and plotting

Scripts:

```text
06-result/result_analysis.py
06-result/loc_plot.py
06-result/vel_plot.py
```

Main outputs include:

```text
06-result/output/analysis/posterior_summary.csv
06-result/output/analysis/location_summary.csv
06-result/output/analysis/analysis_report.json
06-result/output/location_plots/all_events.png
06-result/output/location_plots/event_*.png
06-result/output/location_plots/location_errors.csv
06-result/output/velocity_plots/velocity_model_comparison.csv
06-result/output/velocity_plots/velocity_posterior_summary.csv
06-result/output/velocity_plots/posterior_*_hist.png
06-result/output/velocity_plots/posterior_*_interval.png
```

These scripts read `chain.npz`, discard burn-in samples, and compute posterior mean values, posterior uncertainty summaries, event-location errors, and VTI-parameter posterior distributions.

### Step 6: forward validation using posterior mean and best models

Script:

```text
06-result/forward_compare_mean_best.py
```

Main outputs:

```text
06-result/output/forward_compare/mean_predicted_ttime.dat
06-result/output/forward_compare/best_predicted_ttime.dat
06-result/output/forward_compare/mean_model_summary.dat
06-result/output/forward_compare/best_model_summary.dat
06-result/output/forward_compare/forward_compare_summary.csv
06-result/output/forward_compare/forward_compare_by_source.csv
06-result/output/forward_compare/*.png
```

This step recomputes forward travel times using the posterior mean and best-fitting models. It compares these predictions with the noisy observations and the noiseless truth. The comparison helps check whether the simplified layered posterior model can reproduce the main qP moveout generated by the smooth/complex truth model.

## 5. Key parameters in `shell_simple.py`

Most experiment-level parameters are controlled in the first block of `shell_simple.py`. In normal use, edit this file rather than changing the same parameter in multiple downstream scripts.

### Random seeds

```python
MASTER_SEED = 0
MCMC_SEED = MASTER_SEED + 100
```

These values control reproducibility. The downstream scripts receive deterministic seed offsets for velocity perturbation, noise generation, and MCMC sampling.

### qx solver controls

```python
QX_STOP = 1.0e-6
QX_MAX_ITER = 20
```

`QX_STOP` is the stopping tolerance for solving the horizontal ray parameter. `QX_MAX_ITER` is the maximum number of Newton or bracketed iterations. These values are written to `control.dat` and also passed directly to the forward and inversion scripts.

### Observation noise

```python
OBS_NOISE_STD = 0.001000
```

This is the standard deviation of Gaussian picking noise added to the synthetic absolute qP arrivals. With `sigma-mode = "absolute"`, the differential-arrival covariance is constructed consistently from this absolute-arrival noise level.

### Initial grid search for event locations

```python
INIT_GRID_X_MIN = 350.0
INIT_GRID_X_MAX = 1050.0
INIT_GRID_Z_MIN = 350.0
INIT_GRID_Z_MAX = 850.0
INIT_GRID_DX = 25.0
INIT_GRID_DZ = 25.0
INIT_GRID_REFINE = True
```

These parameters define the initial qP grid-search region and spacing. The optional refinement step performs a second local grid search around the best coarse-grid solution.

### Initial VTI-model perturbation

```python
INIT_DEP_STD = 20.0
INIT_ALPHA_STD = 500.0
INIT_BETA_STD = 300.0
INIT_EPSILON_STD = 0.2
INIT_GAMMA_STD = 0.2
INIT_DELTA_STD = 0.2
```

These half widths control the random perturbation applied to the layered starting model. They define how far the initial model can deviate from the simplified layered reference model.

### Prior bounds

```python
PRIOR_NMIN = 2
PRIOR_NMAX = 20
PRIOR_SOURCE_MARGIN = 100.0
PRIOR_DEPTH_MARGIN = 0.0
PRIOR_ALPHA_MARGIN = 500.0
PRIOR_BETA_MARGIN = 300.0
PRIOR_ANISO_MARGIN = 0.10
```

The prior bounds are generated from the true and initial files with additional margins. The prior file stores admissible ranges for layer depths, VTI parameters, source coordinates, and noise level.

The maximum deviations relative to the initial model are controlled by:

```python
PRIOR_DEPTH_MAX_DEV = 2.0 * INIT_DEP_STD
PRIOR_ALPHA_MAX_DEV = 5.0 * INIT_ALPHA_STD
PRIOR_BETA_MAX_DEV = 5.0 * INIT_BETA_STD
PRIOR_EPSILON_MAX_DEV = 2.0 * INIT_EPSILON_STD
PRIOR_GAMMA_MAX_DEV = 2.0 * INIT_GAMMA_STD
PRIOR_DELTA_MAX_DEV = 2.0 * INIT_DELTA_STD
PRIOR_SOURCE_X_MAX_DEV = 2.0 * INIT_SOURCE_X_STD
PRIOR_SOURCE_Z_MAX_DEV = 2.0 * INIT_SOURCE_Z_STD
```

These values control the allowed MCMC exploration range around the initial model.

### Proposal standard deviations

```python
PROP_DEPTH_STD = 10.0
PROP_ALPHA_STD = 100.0
PROP_BETA_STD = 50.0
PROP_EPSILON_STD = 0.005
PROP_GAMMA_STD = 0.005
PROP_DELTA_STD = 0.005
PROP_SOURCE_X_STD = 25.0
PROP_SOURCE_Z_STD = 25.0
```

These values define the initial scalar proposal widths for the component-wise MCMC updates. The adaptive stage can rescale them, but they remain the baseline step sizes.

### MCMC controls

```python
MCMC_ITERATIONS = 100000
MCMC_BURNIN = 20000
MCMC_PRINT_EVERY = 100
MCMC_ADAPT_START = 5000
MCMC_ADAPT_INTERVAL = 1000
MCMC_ADAPT_STOP = 0
MCMC_DR_SCALE = 0.2
MCMC_UPDATE_ORDER = "cyclic"
```

Important meanings:

- `MCMC_ITERATIONS`: total MCMC iterations.
- `MCMC_BURNIN`: number of early samples discarded in posterior summaries.
- `MCMC_ADAPT_START`: first iteration at which proposal adaptation can begin.
- `MCMC_ADAPT_INTERVAL`: interval between proposal-scale adaptation steps.
- `MCMC_ADAPT_STOP = 0`: adaptation stops at burn-in inside the MCMC code.
- `MCMC_DR_SCALE`: second-stage delayed-rejection proposal scale relative to the first-stage proposal.
- `MCMC_UPDATE_ORDER`: `"cyclic"` updates parameters in order; `"random"` selects a random scalar parameter each iteration.

### Objective and wave selection

```python
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_USE_WAVES = "P"
MCMC_SIGMA_MODE = "absolute"
```

The default inversion uses only qP adjacent-channel differential arrivals. This reduces sensitivity to origin-time errors and emphasizes DAS moveout information. The code can also support `absolute` and `diff-p-reference` objectives, but this smooth-model test is configured for qP adjacent-channel data.

### Depth and bottom-layer treatment

```python
INVERT_DEPTHS = True
FIX_LAST_LAYER = True
```

`INVERT_DEPTHS = True` allows layer-interface depths to be updated during MCMC. `FIX_LAST_LAYER = True` keeps the bottom half-space parameters fixed, which is useful when the final row is mainly a boundary/half-space representation rather than a fully independent layer.

## 6. Important data-file formats

### `geometry.dat`

Typical structure:

```text
ns
sx sz
...
nr
rx rz
...
```

where `ns` is the number of events and `nr` is the number of DAS channels.

### `vel.dat` and `vel1.dat`

Typical structure:

```text
nlayer
dep alpha beta epsilon gamma delta
...
```

Definitions:

- `dep`: depth node or layer boundary depth;
- `alpha`: vertical qP velocity, often denoted `alpha0`;
- `beta`: vertical qS velocity, often denoted `beta0`;
- `epsilon`, `gamma`, `delta`: Thomsen anisotropy parameters.

### `ttime.dat`

Typical structure:

```text
ns nr
sx sz rx rz qP qSV qSH
...
```

The first line gives the number of sources and receivers. Each following row stores source coordinates, receiver coordinates, and the three direct-wave arrival times.

### `nobs.dat`

This file stores noisy observed arrivals generated from `ttime.dat`. It is the observation file used by the inversion.

### `prior.dat`

This file stores prior lower/upper bounds and maximum proposal limits for depth, velocity, anisotropy, source coordinates, and noise.

### `prop.dat`

This file stores scalar proposal standard deviations for noise, depth, VTI parameters, and source coordinates.

### `chain.npz`

This NumPy archive stores the full MCMC chain and metadata, including:

```text
chain
misfit
accepted
accepted_stage
updated_index
mean_theta
std_theta
best_theta
burnin
burnin_eff
objective_type
sigma_mode
param_names
```

The plotting and analysis scripts read this file directly.

## 7. Forward-modelling principle

The forward solver uses direct-wave ray tracing in layered VTI media. For each event-channel pair and wave mode, the code solves for the horizontal slowness/ray parameter that satisfies the horizontal offset between the source and receiver. Once the horizontal ray parameter is found, each crossed layer contributes a horizontal distance and a travel time. The total arrival time is obtained by summing the layer contributions.

The solver includes several safeguards:

- strict validation of VTI parameter ranges;
- high-precision output for forward files;
- fallback handling near critical slowness values;
- convergence diagnostics for every source-receiver-wave pair;
- layer-contribution checks using `sum(layer_dx) - H` and `sum(layer_dt) - ttime`.

These checks are useful because unstable qx solutions can strongly affect MCMC likelihood evaluation.

## 8. Bayesian inversion principle

The MCMC script samples the posterior distribution of event locations and VTI parameters. The posterior is proportional to the product of the likelihood and the prior.

The likelihood compares observed and predicted arrival-time data. In the default setting, the data vector contains adjacent-channel qP differential arrivals. If the original absolute arrival errors are independent with standard deviation `OBS_NOISE_STD`, the adjacent-channel differential data have a non-diagonal covariance structure. The code accounts for this covariance through the selected objective and sigma mode.

The prior restricts physically reasonable model ranges and keeps the chain within predefined bounds. The proposal distribution is component-wise Gaussian. Delayed rejection improves acceptance by trying a smaller second proposal when the first proposal is rejected. Adaptation adjusts proposal scales during the early part of the run.

The outputs `mean.dat` and `best.dat` summarize two useful models:

- `mean.dat`: posterior mean model after burn-in;
- `best.dat`: model with the minimum sampled misfit.

## 9. Recommended quick test

To verify that the workflow runs without committing to a long MCMC run, edit `shell_simple.py`:

```python
MCMC_ITERATIONS = 2000
MCMC_BURNIN = 500
MCMC_PRINT_EVERY = 100
```

Then run:

```bash
python shell_simple.py
```

This quick test is only for checking the workflow and file generation. It should not be used for final posterior interpretation.

## 10. Recommended production run

For a more stable posterior estimate, use the default or larger MCMC settings:

```python
MCMC_ITERATIONS = 100000
MCMC_BURNIN = 20000
```

For manuscript-level results, inspect:

```text
06-result/output/misfit.png
06-result/output/analysis/posterior_summary.csv
06-result/output/location_plots/location_errors.csv
06-result/output/velocity_plots/velocity_posterior_summary.csv
06-result/output/forward_compare/forward_compare_summary.csv
```

Also check event-wise posterior plots and forward-comparison plots before drawing conclusions.

## 11. Notes and cautions

1. `vel1.dat` is the smooth/complex truth model. Do not replace `01-input/output/vel.dat` with `vel1.dat` unless the intention is to remove the parameterization-mismatch test.
2. The inversion is not expected to recover the exact smooth truth model because it only uses a simplified layered parameterization.
3. Event locations may still be accurate if the effective layered VTI model reproduces the dominant qP moveout.
4. qP-only data cannot fully resolve all VTI parameters. Some parameters may remain weakly constrained or correlated.
5. Increasing the number of MCMC iterations generally improves posterior stability but increases runtime.
6. The diagnostic forward files can be large because they store convergence and layer-contribution information for many source-receiver pairs.

## 12. Typical command summary

Run the complete workflow:

```bash
python shell_simple.py
```

Run only the input generator:

```bash
cd 01-input
python direct.py
```

Run only the forward solver:

```bash
cd 02-forward
python vti_direct.py --input-dir ../01-input/output --output-dir ../03-output
```

Run forward validation plots:

```bash
cd 02-forward
python plot_forward_validation.py --output-dir ../03-output
```

Run only the MCMC inversion after initialization files exist:

```bash
cd 05-inversion
python vti_joint_mcmc_dram.py --input-dir ../04-initial/output --output-dir ../06-result/output --iterations 100000 --burnin 20000 --objective diff-p-adjacent --use-waves P --sigma-mode absolute --invert-depths --fix-last-layer
```

Run result analysis after MCMC:

```bash
cd 06-result
python result_analysis.py --root .. --result-dir output
python loc_plot.py --root .. --result-dir output
python vel_plot.py --root .. --result-dir output
python forward_compare_mean_best.py --root .. --result-dir output --objective diff-p-adjacent --sigma-mode absolute --forward-waves P --invert-depths --fix-last-layer
```
