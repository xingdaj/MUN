# P_SV_SH: fixed-event qP/qSV/qSH three-phase VTI Bayesian inversion

## Purpose of this package

This folder is designed for three-phase qP/qSV/qSH VTI parameter inversion with fixed microseismic event locations. Compared with the `P` folder, this experiment includes qP, qSV, and qSH adjacent-channel DAS differential traveltimes in the objective function. It is therefore better suited for testing whether multicomponent phase information can recover the full layered VTI parameter set.

The main scientific question is whether three-phase moveout can jointly constrain `alpha0`, `beta0`, `epsilon`, `gamma`, `delta`, and interface depths when event locations are known. In general, qSV and qSH arrivals provide more direct sensitivity to `beta0` and `gamma` than qP-only data.

## Key settings in this folder

```python
MCMC_USE_WAVES = "P,SV,SH"
MCMC_OBJECTIVE = "diff-p-adjacent"
INVERT_SOURCES = False
INVERT_DEPTHS = True
FIX_LAST_LAYER = True
```

These settings mean:

- qP, qSV, and qSH traveltimes are all used;
- adjacent-channel differential traveltimes are constructed for each phase;
- the same station-pair differencing matrix is used for qP, qSV, and qSH;
- event locations are fixed at the true synthetic geometry;
- interface depths and active-layer VTI parameters are inverted;
- the last half-space row of the velocity model is fixed.

In the default workflow, `04-initial/output/geo.dat` is copied directly from `01-input/output/geometry.dat`. This is therefore a fixed-event VTI calibration test, not an event-location test.


## Forward modelling and inversion principle

### 1. Layered VTI model

The velocity model is stored in `vel.dat`. Each row is a depth node or layer boundary and has six columns:

```text
dep, alpha0, beta0, epsilon, gamma, delta
```

where:

- `dep` is depth in metres;
- `alpha0` is the vertical qP velocity in m/s;
- `beta0` is the vertical qS velocity in m/s;
- `epsilon`, `gamma`, and `delta` are the Thomsen VTI anisotropy parameters.

The default synthetic model contains four effective layers. The main interfaces are at 300 m, 400 m, and 500 m. The last row is placed at 1000 m and repeats the bottom-layer parameters. It mainly acts as the bottom boundary or half-space representation. When `FIX_LAST_LAYER=True`, this last row is not updated by the inversion.

### 2. DAS geometry

`01-input/direct.py` generates 25 synthetic microseismic events and an L-shaped DAS array.

- The events are arranged approximately as a 5 by 5 pattern. Their x coordinates are about 500-900 m, and their z coordinates are about 440-780 m, with small deterministic perturbations.
- The vertical DAS segment is located at x = 200 m, z = 0-640 m, with 10 m channel spacing and 65 channels.
- The horizontal DAS segment is located at z = 650 m, x = 200-1190 m, with 10 m channel spacing and 100 channels.
- The total number of DAS channels is 165.

The geometry files `geometry.dat` and `geo.dat` first store source coordinates `sx, sz`, followed by receiver/DAS-channel coordinates `rx, rz`.

### 3. qx direct-wave ray tracing

`02-forward/vti_direct.py` and `05-inversion/vti_joint_mcmc_dram.py` use the same qx direct-wave forward modelling core. For each event-channel pair, the code first computes the thickness `Z[k]` crossed in each VTI layer and then solves for the horizontal slowness parameter.

In each VTI layer, the qP, qSV, and qSH slowness surfaces are obtained from the Christoffel equation and are written in the form:

```text
pz^2 = g(px)
```

where `px` is the horizontal slowness and `pz` is the vertical slowness. Given the horizontal offset `H`, the code solves the horizontal-offset equation:

```text
sum_k layer_dx[k] = H
```

with

```text
layer_dx[k] = -Z[k] * 0.5 * g1 / pz
```

where `g1` is the first derivative of `g(px)` with respect to `px`. After `px` is found, the code computes the vertical group velocity `Vz` in each layer and sums the layer traveltimes:

```text
t = sum_k Z[k] / Vz[k]
```

For numerical stability, the qx solver combines bracketed Newton steps with bisection. If the solution is close to the critical slowness, if derivatives become unstable, or if the Newton step fails, the solver falls back to a more robust bracketed update.

### 4. Adjacent-channel differential traveltime data

The default objective is:

```python
MCMC_OBJECTIVE = "diff-p-adjacent"
```

This means that the likelihood uses adjacent-channel traveltime differences instead of absolute traveltimes. This reduces the influence of common terms such as the event origin time. When `MCMC_SIGMA_MODE="absolute"`, the code does not treat the differenced residuals as independent noise. Instead, it propagates absolute picking uncertainty into the adjacent-difference covariance:

```text
C_d = sigma^2 A A^T
```

where `A` is the adjacent-channel differencing matrix and `sigma` is set by `OBS_NOISE_STD`. If `MCMC_SIGMA_MODE` is changed to `"objective-iid"`, the differenced data are treated as independent observations. That option is only appropriate when independent noise has already been added directly to the differenced data.

### 5. Bayesian DRAM MCMC

`05-inversion/vti_joint_mcmc_dram.py` performs fixed-dimension Bayesian inversion using a delayed-rejection adaptive Metropolis strategy. The inverted variables can include interface depths, VTI parameters, and, for the joint-location case, event coordinates. Each MCMC step updates one scalar parameter.

The basic workflow is:

1. Read the initial geometry and velocity model from `04-initial/output/geo.dat` and `04-initial/output/vel.dat`.
2. Read uniform prior bounds and maximum perturbation limits from `prior.dat`.
3. Read proposal standard deviations from `prop.dat`.
4. Propose a first-stage Gaussian perturbation for one scalar parameter.
5. If the first-stage proposal is rejected, propose a smaller delayed-rejection move using `MCMC_DR_SCALE`.
6. Accept or reject the proposal using the Metropolis rule based on the change in misfit.
7. After burn-in, compute the posterior mean, posterior standard deviation, and best-misfit sample.

The fixed normalization constants and log determinant of the covariance are omitted from the misfit because the data covariance is fixed in the current setting and therefore does not affect the Metropolis acceptance ratio.


## Three-phase differential objective

For each phase, the code computes adjacent-channel differential residuals. When all three phases are selected, the total misfit is the sum of the three phase-specific misfits:

```text
Phi = Phi_qP + Phi_qSV + Phi_qSH
```

Each phase uses the same covariance structure for adjacent-channel differences:

```text
C_d = sigma^2 A A^T
```

This assumes that picking errors are independent between phases, while adjacent differences within the same phase are correlated because neighbouring difference pairs share DAS channels.


## Directory structure

```text
01-input/
  direct.py                      Generate true geometry, true layered VTI model, and control parameters
  output/                        Generated geometry.dat, vel.dat, vel1.dat, control.dat, and model figures

02-forward/
  vti_direct.py                  Compute qP/qSV/qSH direct-wave traveltimes from the true model
  plot_forward_validation.py     Check and visualize forward-modelling results

03-output/                       Automatically generated by 02-forward
  ttime.dat                      Noise-free qP/qSV/qSH traveltimes
  qx.dat                         qx/px solutions
  diagnostics.dat                qx convergence, residual, and iteration diagnostics
  layer_contributions.dat        Layer-by-layer traveltime and horizontal-offset contributions

04-initial/
  vel_add.py                     Add perturbations to the true VTI parameters and interface depths
  noise_add.py                   Add Gaussian picking noise to forward traveltimes
  geo_grid_search.py             Estimate initial event locations using qP traveltime grid search
  geo_add.py                     Legacy script for generating perturbed initial event locations
  prior_prop.py                  Generate prior.dat and prop.dat
  output/                        Generated vel.dat, geo.dat, nobs.dat, prior.dat, and prop.dat

05-inversion/
  vti_joint_mcmc_dram.py         Main Bayesian DRAM MCMC inversion program

06-result/
  result_analysis.py             Posterior statistics, misfit, and acceptance-rate analysis
  loc_plot.py                    Event-location posterior plots and location-error summaries
  vel_plot.py                    VTI posterior profiles and histograms
  forward_compare_mean_best.py   Compare observed data with forward predictions from posterior mean and best models
  vti_plot_utils.py              Plotting and file-reading utilities
```



## How to run

Run the workflow from the current folder:

```bash
python shell_simple.py
```

`shell_simple.py` executes the main scripts in order:

```text
1. 01-input/direct.py
2. 02-forward/vti_direct.py
3. 04-initial/vel_add.py
4. 04-initial/noise_add.py
5. 04-initial/geo_grid_search.py or direct copy of true event locations, depending on the experiment
6. 04-initial/prior_prop.py
7. 05-inversion/vti_joint_mcmc_dram.py
8. 06-result/result_analysis.py
9. 06-result/loc_plot.py
10. 06-result/vel_plot.py
11. 06-result/forward_compare_mean_best.py
```

The recommended way to change the experiment is to modify the unified parameter block near the top of `shell_simple.py`. Avoid changing the same parameter independently in several sub-scripts unless a specific debugging test requires it.


## Current three-phase inversion variables

With the default configuration, the MCMC parameter vector contains:

```text
1. dep[1], dep[2], dep[3], ...              Interface depths; dep[0] is fixed
2. alpha0[k]                                Active-layer vertical qP velocity
3. beta0[k]                                 Active-layer vertical qS velocity
4. epsilon[k]
5. gamma[k]
6. delta[k]
```

Because `INVERT_SOURCES=False`, event coordinates are not included in the MCMC parameter vector. Because `FIX_LAST_LAYER=True`, the bottom half-space row is also not updated.


## Important parameters in `shell_simple.py`

### Random seed and qx solver

```python
MASTER_SEED = 0
QX_STOP = 1.0e-6
QX_MAX_ITER = 20
```

- `MASTER_SEED` controls repeatability for geometry perturbations, initial-model perturbations, noise generation, and MCMC randomness.
- `QX_STOP` is the convergence tolerance for the qx horizontal-offset equation.
- `QX_MAX_ITER` is the maximum iteration count recorded in output diagnostics. The internal solver can still use robust fallback steps when needed.

### Noise level

```python
OBS_NOISE_STD = 0.001000
```

This is the standard deviation of absolute traveltime picking noise in seconds. `noise_add.py` uses it to add noise to `ttime.dat`, and `prior_prop.py` writes the same value to `prop.dat`. To change the noise level, change this value in `shell_simple.py`.

### Initial velocity-model perturbations

```python
INIT_DEP_STD = 20.0
INIT_ALPHA_STD = 500.0
INIT_BETA_STD = 300.0
INIT_EPSILON_STD = 0.2
INIT_GAMMA_STD = 0.2
INIT_DELTA_STD = 0.2
```

These parameters control the half-widths of the uniform perturbations added by `vel_add.py` to the true layered model. They determine how far the initial model is from the true model.

### Prior width

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

These values define the maximum allowed movement of each parameter relative to the initial model. If many samples hit the prior boundary, the corresponding prior may be too narrow. If the posterior is unrealistically broad, the prior or proposal scale may need to be reduced.

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

These are the initial standard deviations for first-stage MCMC proposals. If the acceptance rate is too low, the proposal step may be too large. If the chain moves slowly, the proposal step may be too small. After `MCMC_ADAPT_START`, the code adapts proposal sizes using empirical chain statistics.

### MCMC control parameters

```python
MCMC_ITERATIONS = 100000
MCMC_BURNIN = 20000
MCMC_PRINT_EVERY = 100
MCMC_ADAPT_START = 5000
MCMC_ADAPT_INTERVAL = 1000
MCMC_ADAPT_STOP = 0
MCMC_DR_SCALE = 0.2
MCMC_UPDATE_ORDER = "cyclic"
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_SIGMA_MODE = "absolute"
INVERT_DEPTHS = True
FIX_LAST_LAYER = True
```

- `MCMC_ITERATIONS` is the total number of MCMC iterations.
- `MCMC_BURNIN` is the number of early samples discarded before computing posterior statistics.
- `MCMC_ADAPT_START` defines when adaptive proposal tuning starts.
- `MCMC_ADAPT_INTERVAL` defines how often proposal scales are updated.
- `MCMC_ADAPT_STOP=0` means adaptation stops at the burn-in point inside the MCMC code.
- `MCMC_DR_SCALE=0.2` means the delayed-rejection proposal standard deviation is 20% of the first-stage proposal standard deviation.
- `MCMC_UPDATE_ORDER="cyclic"` updates parameters in a fixed cyclic order. Changing it to `"random"` makes the code randomly choose one parameter at each step.
- `INVERT_DEPTHS=True` enables interface-depth inversion.
- `FIX_LAST_LAYER=True` keeps the bottom half-space row fixed.


## Notes specific to the three-phase experiment

1. Using qP/qSV/qSH substantially increases the forward-modelling cost per MCMC step.
2. If `MCMC_USE_WAVES` is changed to `"P"`, this folder becomes a qP-only fixed-event inversion.
3. Three-phase data provide a more complete constraint on VTI parameters, but this relies on reliable qSV and qSH arrival picking.
4. For field DAS data where qSV and qSH arrivals are difficult to pick consistently, this package is better treated as a synthetic calibration test rather than the default field-inversion setup.
5. Event locations are fixed in this package. Use `P_event_VTI` when posterior event-location uncertainty is the target.


## Main outputs

After the workflow finishes, the most important inversion outputs are in `06-result/output/`:

```text
chain.npz       Full MCMC chain, misfit values, acceptance flags, parameter names, posterior mean, and best sample
mean.dat        Posterior mean model and event locations after burn-in
best.dat        Model and event locations from the minimum-misfit sample
misfit.png      MCMC misfit curve versus iteration
```

Post-processing outputs include:

```text
06-result/output/analysis_figures/
  posterior_summary.csv
  location_summary.csv
  analysis_report.json
  misfit_trace.png
  acceptance_trace.png

06-result/output/location_figures/
  all_events.png
  event_XXX.png
  posterior_event_XXX.png
  location_errors.csv
  location_posterior_summary.csv

06-result/output/velocity_figures/
  velocity_model_comparison.csv
  velocity_posterior_summary.csv
  posterior_*_hist.png
  posterior_*_interval.png

06-result/output/forward_compare/
  forward_compare_summary.csv
  forward_compare_by_source.csv
  mean_predicted_ttime.dat
  best_predicted_ttime.dat
  mean_forward_comparison.npz
  best_forward_comparison.npz
  arrival_plots/
```

For a quick check of the inversion result, start with `chain.npz`, `mean.dat`, `best.dat`, `velocity_posterior_summary.csv`, and `location_errors.csv`.


## Recommended modifications

- To change the phase combination, modify `MCMC_USE_WAVES`, for example `"P,SV"` or `"P"`.
- To change the noise level, modify `OBS_NOISE_STD`.
- To change the MCMC length, modify `MCMC_ITERATIONS` and `MCMC_BURNIN`.
- To enable or disable interface-depth inversion, modify `INVERT_DEPTHS`.
- To change the true model or DAS geometry, edit `01-input/direct.py`.
- To tune priors and proposals, modify the `PRIOR_*` and `PROP_*_STD` parameters.
