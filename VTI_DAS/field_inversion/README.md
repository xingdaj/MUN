# Field DAS qP Travel-Time Inversion in a Layered VTI Model

This package contains a field-data workflow for Bayesian microseismic event location and effective VTI model estimation using distributed acoustic sensing (DAS) P-wave arrival times. The workflow prepares field DAS P-arrival data, estimates initial event locations by grid search, runs a joint delayed-rejection adaptive Metropolis (DRAM) inversion, and generates posterior diagnostic figures.

The current configuration is designed for field qP-only inversion. The inversion jointly updates event locations, layered VTI velocity parameters, and internal interface depths. The objective function uses adjacent-channel differential P arrival times, which reduces sensitivity to unknown origin-time shifts and emphasizes the dense moveout information recorded by the DAS array.

## Package layout

```text
field_inversion/
├── README.md
├── layered_velocity_model.csv
├── event_receiver_times.csv
├── event_polarity_directions.csv
└── code/
    ├── shell_field.py
    ├── 01-initial/
    │   ├── field_prepare.py
    │   ├── geo_grid_search.py
    │   ├── prior_prop.py
    │   └── plot_initial_check.py
    ├── 02-inversion/
    │   ├── vti_joint_mcmc_dram.py
    │   └── vti_joint_mcmc_dram.py.bak
    └── 03-output/
        ├── event_posterior_plot_region.py
        ├── forward_compare_mean_best.py
        ├── vel_plot.py
        └── vti_plot_utils.py
```

The main driver is:

```text
code/shell_field.py
```

This script controls the whole workflow and passes the parameter settings to the preparation, inversion, and plotting scripts.

## Input data

### `layered_velocity_model.csv`

This file stores the initial field velocity model. It currently contains 18 depth intervals from 200 m to 2000 m, with 100 m interval thickness. The important columns are:

| Column | Meaning |
|---|---|
| `top_depth_m` | Top depth of each layer interval |
| `bottom_depth_m` | Bottom depth of each layer interval |
| `mid_depth_m` | Mid-depth of each interval, used for checking and plotting |
| `vp_mps` | Vertical P-wave velocity or initial alpha value in m/s |
| `vs_mps` | Vertical S-wave velocity or initial beta value in m/s |
| `vp_source` | Source label for the P velocity value |
| `vs_source` | Source label for the S velocity value |

During preparation, `field_prepare.py` converts this interval model into the `vel.dat` format used by the inversion code. A bottom depth node is appended by default, so the inversion model has 19 depth nodes. The first and last depth nodes are fixed by the MCMC code, while the internal interfaces can be updated if `INVERT_DEPTHS = True`.

### `event_receiver_times.csv`

This file stores the picked field P-wave arrival times. The current file contains 41 events and 488 DAS receiver/channel samples. The important columns are:

| Column | Meaning |
|---|---|
| `event_id` | Event index |
| `receiver_id` | Receiver/channel index after DAS subsampling |
| `trace_0based` | Original zero-based trace index |
| `channel_1based` | Original one-based DAS channel index |
| `distance_along_fiber_m` | Distance along the DAS fiber |
| `station_name` | Receiver name |
| `x_m` | Field X coordinate |
| `y_m` | Field Y coordinate |
| `z_or_elev_m` | Elevation or vertical coordinate in the original field table |
| `depth_m` | Receiver depth |
| `newP_time_s` | Picked P-wave arrival time in seconds |

The current workflow performs a 2-D projected inversion in the horizontal-coordinate/depth plane. In `shell_field.py`, the field `y_m` coordinate is used as the internal horizontal coordinate, and `depth_m` is used as the vertical coordinate:

```python
FIELD_RECEIVER_X_COL = "y_m"
FIELD_RECEIVER_Z_COL = "depth_m"
```

The MCMC script still calls the horizontal coordinate `x` internally. In this field setting, that internal `x` coordinate corresponds to the selected field coordinate `y_m`.

### `event_polarity_directions.csv`

This file stores event polarity-direction information, including orientation, angular spread, and reliability labels. It is included as auxiliary event information. The main inversion driver does not directly use this file.

## Main workflow

The complete workflow is controlled by:

```bash
cd field_inversion/code
python shell_field.py
```

The driver performs the following steps.

### Step 1: Prepare field input files

Script:

```text
01-initial/field_prepare.py
```

This script reads `layered_velocity_model.csv` and `event_receiver_times.csv`, then writes field input files into:

```text
code/01-initial/output/
```

Main generated files include:

| File | Purpose |
|---|---|
| `geometry.dat` | Receiver-only geometry before grid search; it stores the event count and receiver coordinates without source placeholders |
| `vel.dat` | Initial layered VTI model in the inversion format |
| `nobs.dat` | Observed field P arrival times and assumed data standard deviation |
| `ttime.dat` | Human-readable field P-time table |
| `control.dat` | qx solver tolerance and maximum iteration settings |
| `field_prepare_summary.json` | Summary of selected events, receivers, model size, and preparation settings |

The field observations are written in compact P-only format. The row layout in `nobs.dat` is:

```text
event_id receiver_id rx rz tp
```

where `tp` is the observed P-wave arrival time.

### Step 2: Write prior and proposal files

Script:

```text
01-initial/prior_prop.py
```

This script writes:

```text
code/01-initial/output/prior.dat
code/01-initial/output/prop.dat
```

`prior.dat` defines uniform prior bounds for source positions, interface depths, and VTI parameters. `prop.dat` defines the scalar proposal standard deviations used by the component-wise DRAM sampler.

### Step 3: Draw input-check figures

Script:

```text
01-initial/plot_initial_check.py
```

This script creates input-check figures before the expensive inversion starts. These figures are written to:

```text
code/01-initial/output/input_check/
```

Use these plots to verify the receiver geometry, selected arrival-time curves, velocity model, and transformed objective data before running the grid search and MCMC inversion.

### Step 4: Estimate initial event locations by grid search

Script:

```text
01-initial/geo_grid_search.py
```

The grid search estimates one initial source location for each field event by fitting the observed P-arrival moveout with the initial layered model. The current search range is:

```python
INIT_GRID_X_MIN = 800.0
INIT_GRID_X_MAX = 1600.0
INIT_GRID_Z_MIN = 1600.0
INIT_GRID_Z_MAX = 2000.0
INIT_GRID_DX = 10.0
INIT_GRID_DZ = 10.0
```

The generated initial event geometry is written to:

```text
code/01-initial/output/geo.dat
```

The grid-search summary and figure are written to:

```text
code/01-initial/output/geo_grid_search.csv
code/01-initial/output/figures/geo_grid_search.pdf
```

### Step 5: Run joint DRAM MCMC inversion

Script:

```text
02-inversion/vti_joint_mcmc_dram.py
```

The inversion reads `geo.dat`, `vel.dat`, `nobs.dat`, `prior.dat`, and `prop.dat` from `code/01-initial/output/`. It writes results to:

```text
code/03-output/output/
```

The current inversion updates:

1. Event locations: source horizontal coordinate and depth for each event.
2. Layered VTI parameters: `alpha`, `beta`, `epsilon`, `gamma`, and `delta` for each layer/node.
3. Internal layer-interface depths, if `INVERT_DEPTHS = True`.

The first and last depth nodes are fixed. The internal interfaces are allowed to move within the prior limits.

### Step 6: Generate posterior and forward-comparison figures

Scripts:

```text
03-output/vel_plot.py
03-output/forward_compare_mean_best.py
03-output/event_posterior_plot_region.py
```

These scripts generate velocity-posterior plots, forward-predicted P-arrival comparisons, and event-location posterior figures. The main outputs are written under:

```text
code/03-output/output/
```

Typical output subdirectories include:

```text
velocity_figures/
forward_compare/
event_posterior_region/
```

## Main parameter settings in `shell_field.py`

Most user-editable settings are collected near the top of `code/shell_field.py`.

### Field-data selection and geometry projection

```python
FIELD_VELOCITY_CSV = "../layered_velocity_model.csv"
FIELD_TIMES_CSV = "../event_receiver_times.csv"
FIELD_EVENT_IDS = "all"
FIELD_RECEIVER_X_COL = "y_m"
FIELD_RECEIVER_Z_COL = "depth_m"
FIELD_RECEIVER_SORT_COL = "receiver_id"
FIELD_RENUMBER_RECEIVERS = True
FIELD_RECEIVER_STRIDE = 1
```

Meaning:

| Parameter | Meaning |
|---|---|
| `FIELD_EVENT_IDS` | Events used in the inversion. Use `"all"` for all events, or strings such as `"1,2,5"` or `"1-20"` for selected events. |
| `FIELD_RECEIVER_X_COL` | CSV column used as the internal horizontal coordinate. The current field inversion uses `y_m`. |
| `FIELD_RECEIVER_Z_COL` | CSV column used as the depth coordinate. |
| `FIELD_RECEIVER_SORT_COL` | Column used to sort receivers before constructing the arrival-time vector. |
| `FIELD_RENUMBER_RECEIVERS` | If `True`, selected receivers are renumbered from 1 to N. |
| `FIELD_RECEIVER_STRIDE` | Receiver decimation factor. `1` keeps all selected receivers. `2` keeps every second receiver, and so on. |

### Initial anisotropy values

```python
FIELD_EPSILON = 0.0
FIELD_GAMMA = 0.0
FIELD_DELTA = 0.0
```

The field CSV provides P and S velocities. The initial anisotropy parameters are assigned constant values by `field_prepare.py`. The inversion can then update these parameters within their prior ranges.

### qx forward solver settings

```python
QX_STOP = 1.0e-6
QX_MAX_ITER = 20
```

The forward solver uses the direct-wave horizontal ray-parameter, or qx, formulation. For each source-receiver pair and each trial VTI model, the code solves for the horizontal ray parameter that matches the source-receiver horizontal offset. It then computes the layerwise travel time along the ray path.

`QX_STOP` controls the convergence tolerance. `QX_MAX_ITER` controls the maximum number of qx iterations.

### Data-noise setting

```python
OBS_NOISE_STD = 0.0020
```

This is the assumed standard deviation of the objective data in seconds. It is written to `nobs.dat` and `prop.dat`. It is fixed during inversion and is not treated as an unknown parameter.

Because the current objective is `diff-p-adjacent`, this value is used for the adjacent-channel P-time difference residuals under `MCMC_SIGMA_MODE = "objective-iid"`.

### Initial source grid search

```python
INIT_GRID_X_MIN = 800.0
INIT_GRID_X_MAX = 1600.0
INIT_GRID_Z_MIN = 1600.0
INIT_GRID_Z_MAX = 2000.0
INIT_GRID_DX = 10.0
INIT_GRID_DZ = 10.0
INIT_GRID_REFINE = False
```

These parameters control the source-location grid search. The grid-search output provides initial event locations for the MCMC inversion.

### Source-location prior

```python
PRIOR_SOURCE_X_MIN = 800.0
PRIOR_SOURCE_X_MAX = 1600.0
PRIOR_SOURCE_Z_MIN = 1600.0
PRIOR_SOURCE_Z_MAX = 2000.0
```

These values define the uniform source-location prior. They should normally cover the expected field event region and be consistent with the grid-search range.

### VTI parameter priors

```python
PRIOR_ALPHA_MIN = 1500.0
PRIOR_ALPHA_MAX = 7000.0
PRIOR_BETA_MIN = 1000.0
PRIOR_BETA_MAX = 5000.0
PRIOR_EPSILON_MIN = 0.0
PRIOR_EPSILON_MAX = 0.5
PRIOR_GAMMA_MIN = 0.0
PRIOR_GAMMA_MAX = 0.5
PRIOR_DELTA_MIN = 0.0
PRIOR_DELTA_MAX = 0.5
```

The inversion uses uniform priors for the five VTI parameters:

| Parameter | Meaning |
|---|---|
| `alpha` | Vertical P-wave velocity, equivalent to `Vp0` |
| `beta` | Vertical S-wave velocity, equivalent to `Vs0` |
| `epsilon` | Thomsen epsilon parameter |
| `gamma` | Thomsen gamma parameter |
| `delta` | Thomsen delta parameter |

The current field inversion uses qP observations only. Therefore, some VTI parameters may remain weakly resolved, especially the parameters that are not strongly expressed in qP moveout for the current geometry and phase availability.

### Layer-depth prior and proposal settings

```python
PRIOR_NMIN = 19
PRIOR_NMAX = 19
PRIOR_DEPTH_MAX_DEV = 20.0
PROP_DEPTH_STD = 10.0
```

`PRIOR_NMIN` and `PRIOR_NMAX` fix the model size to 19 depth nodes. `PRIOR_DEPTH_MAX_DEV = 20.0` allows each internal interface to move within plus or minus 20 m around the initial model, subject to the full prior constraints. `PROP_DEPTH_STD` defines the initial proposal standard deviation for an interface-depth update.

### Velocity and anisotropy proposal settings

```python
PROP_ALPHA_STD = 100.0
PROP_BETA_STD = 50.0
PROP_EPSILON_STD = 0.005
PROP_GAMMA_STD = 0.005
PROP_DELTA_STD = 0.005
```

These values define the initial scalar proposal standard deviations for the DRAM sampler. The code can adapt these values during burn-in according to the adaptation settings.

### Source proposal settings

```python
PROP_SOURCE_X_STD = 50.0
PROP_SOURCE_Z_STD = 50.0
```

These values control the initial proposal width for event horizontal location and depth.

### Stage-control flags

```python
RUN_GRID_SEARCH = True
RUN_INVERSION = True
RUN_VELOCITY_PLOTS = True
RUN_FORWARD_COMPARE = True
RUN_EVENT_POSTERIOR_PLOTS = True
```

Recommended safe workflow:

1. Set `RUN_GRID_SEARCH = False` and `RUN_INVERSION = False` to generate and check only the initial input files and figures.
2. After the input-check figures are confirmed, set `RUN_GRID_SEARCH = True` and keep `RUN_INVERSION = False` to inspect initial source locations.
3. After grid-search results are confirmed, set `RUN_INVERSION = True` to run the full MCMC inversion.
4. Keep the plotting flags enabled after the MCMC output is available.

The uploaded configuration has all major stages enabled.

### MCMC settings

```python
FIX_VELOCITY = False
INVERT_DEPTHS = True
FIX_LAST_LAYER = False
MCMC_ITERATIONS = 300000
MCMC_BURNIN = 100000
MCMC_PRINT_EVERY = 100
MCMC_ADAPT_START = 2000
MCMC_ADAPT_INTERVAL = 500
MCMC_ADAPT_STOP = 0
MCMC_DR_SCALE = 0.2
MCMC_UPDATE_ORDER = "random"
MCMC_FORWARD_WORKERS = 8
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_USE_WAVES = "P"
MCMC_SIGMA_MODE = "objective-iid"
```

Important options:

| Parameter | Meaning |
|---|---|
| `FIX_VELOCITY` | If `True`, velocity parameters are fixed and only source/depth options allowed by other flags are updated. |
| `INVERT_DEPTHS` | If `True`, internal layer-interface depths are updated. |
| `FIX_LAST_LAYER` | If `True`, the last half-space VTI parameters are fixed. |
| `MCMC_ITERATIONS` | Total number of MCMC iterations. |
| `MCMC_BURNIN` | Number of samples discarded before posterior summary statistics are computed. |
| `MCMC_DR_SCALE` | Delayed-rejection proposal scale. The second proposal uses `MCMC_DR_SCALE` times the first proposal standard deviation. |
| `MCMC_UPDATE_ORDER` | `random` means one scalar parameter is selected randomly at each iteration. |
| `MCMC_FORWARD_WORKERS` | Number of parallel workers for full all-event forward modeling when a velocity or depth parameter is proposed. |
| `MCMC_OBJECTIVE` | Current objective type. The field workflow uses adjacent-channel differential P arrivals. |
| `MCMC_USE_WAVES` | Wave phases used by the inversion. The current field workflow uses `P`. |
| `MCMC_SIGMA_MODE` | `objective-iid` treats objective residuals as independent with the specified standard deviation. |

## Inversion principle

### 1. Forward modeling in layered VTI media

The forward model computes direct qP travel times through a layered VTI model. For a given source, receiver, and layered VTI model, the solver:

1. Determines the layer path between the source depth and receiver depth.
2. Solves the horizontal ray parameter `qx` so that the predicted horizontal offset matches the observed source-receiver offset.
3. Uses the VTI slowness relation to compute layerwise travel-time contributions.
4. Sums the layerwise contributions to obtain the predicted qP arrival time.

The same forward solver is used in grid search, MCMC likelihood evaluation, and posterior mean/best forward comparison. This keeps the initialization and inversion internally consistent.

### 2. Adjacent-channel differential P-time objective

For each event, the observed absolute P arrival vector is first converted to adjacent-channel differential arrivals:

```text
d_i = t_i - t_{i+1}
```

where `t_i` and `t_{i+1}` are P arrival times at adjacent selected DAS channels. The model prediction is transformed in the same way. The likelihood compares observed and predicted differential arrivals.

This objective is useful for field DAS microseismic data because it reduces sensitivity to event origin-time uncertainty and focuses the inversion on dense spatial moveout along the DAS fiber.

### 3. Bayesian parameterization

The unknown parameter vector contains:

```text
source locations + internal interface depths + layered VTI parameters
```

For 41 field events, the source part contains one horizontal coordinate and one depth coordinate for each event. The velocity part contains the five VTI parameters for the layered model. If depth inversion is enabled, the internal interface depths are also included.

The prior is box-uniform within the bounds written to `prior.dat`. Proposals are initialized from the standard deviations in `prop.dat`.

### 4. Component-wise DRAM sampling

The MCMC code uses a component-wise delayed-rejection adaptive Metropolis strategy:

1. At each iteration, one scalar parameter is selected.
2. A first proposal is generated using the current proposal standard deviation.
3. If the first proposal is rejected, a smaller delayed-rejection proposal is attempted.
4. Proposal scales can be adapted during burn-in using the empirical chain variability.
5. After burn-in, posterior mean, posterior standard deviation, best sample, and chain arrays are saved.

This scalar update strategy is computationally efficient for field inversion because a source-location proposal only requires forward recomputation for that event, while a velocity or depth proposal requires recomputation for all events.

## Output files

After a successful full run, the main MCMC output directory is:

```text
code/03-output/output/
```

Typical files include:

| File | Purpose |
|---|---|
| `chain.npz` | Full MCMC chain and metadata used by all plotting scripts |
| `mean.dat` | Posterior mean model and event locations |
| `best.dat` | Best-sample model and event locations |
| `acceptance_summary.csv` | Acceptance and proposal information, if generated by the MCMC script |
| `misfit.png` or `misfit.pdf` | Misfit history, if plotting is available |

The exact set of output files depends on plotting availability and selected workflow flags.

### Velocity posterior outputs

Generated by:

```text
03-output/vel_plot.py
```

Typical output directory:

```text
code/03-output/output/velocity_figures/
```

Typical outputs:

| File | Purpose |
|---|---|
| `P.png` | P-wave velocity comparison |
| `S.png` | S-wave velocity comparison |
| `E.png` | Epsilon comparison |
| `G.png` | Gamma comparison |
| `D.png` | Delta comparison |
| `posterior_<param>_hist.png` | Marginal posterior histograms |
| `posterior_<param>_interval.png` | Layerwise posterior intervals |
| `velocity_model_comparison.csv` | Initial, mean, and best model values |
| `velocity_posterior_summary.csv` | Posterior statistics and quantiles |

### Forward-comparison outputs

Generated by:

```text
03-output/forward_compare_mean_best.py
```

Typical output directory:

```text
code/03-output/output/forward_compare/
```

Typical outputs:

| File | Purpose |
|---|---|
| `forward_compare_summary.csv` | Overall absolute and differential P-time residual statistics |
| `forward_compare_by_source.csv` | Event-by-event residual statistics |
| `mean_predicted_ttime.dat` | P arrivals predicted by the posterior mean model |
| `best_predicted_ttime.dat` | P arrivals predicted by the best sample |
| `mean_forward_comparison.npz` | Array data for posterior mean forward comparison |
| `best_forward_comparison.npz` | Array data for best-sample forward comparison |
| `arrival_plots/` | Arrival-time comparison figures |

### Event-location posterior outputs

Generated by:

```text
03-output/event_posterior_plot_region.py
```

Typical output directory:

```text
code/03-output/output/event_posterior_region/
```

The plotting region is controlled inside `event_posterior_plot_region.py`:

```python
PLOT_X_MIN = 800.0
PLOT_X_MAX = 1600.0
PLOT_Z_MIN = 1600.0
PLOT_Z_MAX = 2000.0
```

The plotting script draws posterior samples, posterior mean locations, and location uncertainty summaries for each event within the selected research region.

## Running selected parts manually

Although `shell_field.py` is the recommended entry point, each script can also be run manually. For example:

```bash
cd field_inversion/code/01-initial
python field_prepare.py \
  --velocity-csv ../../layered_velocity_model.csv \
  --times-csv ../../event_receiver_times.csv \
  --output-dir output \
  --sigma 0.002 \
  --event-ids all \
  --receiver-x-col y_m \
  --receiver-z-col depth_m
```

A full manual workflow must preserve the same file dependencies:

1. `field_prepare.py` must run before `prior_prop.py` and `plot_initial_check.py`.
2. `geo_grid_search.py` must run before the MCMC inversion because it creates `geo.dat` with initial event locations.
3. `vti_joint_mcmc_dram.py` must finish before `vel_plot.py`, `forward_compare_mean_best.py`, and `event_posterior_plot_region.py` are run.

## Practical notes

- The full field inversion is computationally expensive because each velocity or interface-depth proposal requires forward modeling for all events.
- Increase `MCMC_FORWARD_WORKERS` only if the machine has enough CPU cores and memory.
- Use `FIELD_EVENT_IDS` to test a smaller event subset before running all 41 events.
- Use `FIELD_RECEIVER_STRIDE` to reduce the number of DAS receivers during debugging.
- Use `RUN_GRID_SEARCH = False` and `RUN_INVERSION = False` for the first check run.
- Keep `OBS_NOISE_STD` fixed unless there is a clear reason to change the assumed picking uncertainty.
- For reproducibility, keep `MASTER_SEED` and `MCMC_SEED` fixed when comparing different parameter settings.

## Minimal quick-start checklist

1. Open `code/shell_field.py`.
2. Check the input CSV paths:

   ```python
   FIELD_VELOCITY_CSV = "../layered_velocity_model.csv"
   FIELD_TIMES_CSV = "../event_receiver_times.csv"
   ```

3. Confirm the coordinate projection:

   ```python
   FIELD_RECEIVER_X_COL = "y_m"
   FIELD_RECEIVER_Z_COL = "depth_m"
   ```

4. For a first check-only run, set:

   ```python
   RUN_GRID_SEARCH = False
   RUN_INVERSION = False
   ```

5. Run:

   ```bash
   cd field_inversion/code
   python shell_field.py
   ```

6. Inspect:

   ```text
   code/01-initial/output/input_check/
   ```

7. Enable grid search and inspect:

   ```text
   code/01-initial/output/figures/geo_grid_search.pdf
   ```

8. Enable inversion only after the field inputs and grid-search initial locations are confirmed.

## Citation and reproducibility notes

This package is intended to support field DAS microseismic location experiments in layered VTI media. When using the results in a manuscript or supplementary repository, report the following settings clearly:

- Selected event IDs.
- Receiver coordinate projection and receiver stride.
- Assumed objective noise standard deviation.
- Source prior range.
- VTI parameter prior ranges.
- Whether internal interface depths were inverted.
- MCMC iteration number, burn-in length, proposal adaptation settings, and random seed.
- Objective type, which is `diff-p-adjacent` in the current configuration.
