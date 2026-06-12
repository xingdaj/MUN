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

## Main parameters

Most parameters are near the top of `RQS_NF.py` and can also be overridden with environment variables:

- `NUM_EPOCHS`: total training iterations, default `150`.
- `OBS_NOISE_STD`: observation noise standard deviation used by the likelihood, default `1.0e-4`.
- `SOURCE_AMPLITUDE`: source amplitude, default `1.0e4`.
- `FIXED_SIGMA`: Gaussian prior scale for control-point offsets, default `200.0` m.
- `FLOW_TAIL_BOUND`: RQS coupling tail bound. By default it is `1.5 * FIXED_SIGMA`.
- `PLOT_EVERY`: save posterior/boundary snapshots every N epochs, default `5`.

Example:

```bash
NUM_EPOCHS=200 OBS_NOISE_STD=1e-4 SOURCE_AMPLITUDE=1e4 python shell_run.py
```

## Notes on consistency

The true, initial, and background models are generated with consistent grid, PML, layer, and signed-distance sigmoid mapping metadata. `RQS_NF.py` checks this metadata before running. Observation data are generated from `vp_background_pml + true B-spline anomaly`, which is the same model construction seen by the inversion.

## Generated outputs

Typical runtime outputs are intentionally not tracked in the clean archive:

- `obs_cache/`
- `sem_output/`
- `posterior_model/`
- `source_waveforms/`
- `fixedK_diagnostics/`

These folders are regenerated during execution.
