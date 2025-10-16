# MCMC — Bayesian Inference of a B-spline Velocity Boundary with SEM (DRAM MCMC)

> This README explains the theory and algorithmic structure of `MCMC.py`, which infers a **12-dimensional** B-spline boundary from seismic data by coupling a **Spectral Element Method (SEM)** forward model with a **Delayed Rejection Adaptive Metropolis** (DRAM) MCMC sampler.

---

## 1) Problem setup

We model a subsurface with two constant velocities separated by a **closed cubic B-spline** boundary. Let
\[
\mathbf z\in\mathbb{R}^{12}
\]
be the latent offsets applied to a base set of 6 control points \((x_i,z_i)_{i=1}^6\). From \(\mathbf z\) we build a signed-distance field and a soft interface that blends \(v_\mathrm{in}\) and \(v_\mathrm{out}\) to obtain nodal velocities on the SEM grid. The SEM forward model maps \(\mathbf z\mapsto \mathbf y_\text{syn}(\mathbf z)\) (receiver seismograms).

With observed waveforms \(\mathbf y\) and Gaussian noise \(\mathcal N(0,\sigma^2)\), the log-likelihood is
\[
\log p(\mathbf y\mid \mathbf z) = -\tfrac{1}{2\sigma^2}\,\|\mathbf y-\mathbf y_\text{syn}(\mathbf z)\|_2^2 + C,
\]
and the prior on \(\mathbf z\) is factorized Gaussian centered at 0 with standard deviation \(t_\text{std}\). We target
\[
p(\mathbf z\mid \mathbf y)\propto p(\mathbf y\mid \mathbf z)\,p(\mathbf z).
\]

---

## 2) Geometry: B-spline parameterization

- **6 control points** define a closed cubic B-spline; the first point is duplicated to enforce closure.
- The latent vector \(\mathbf z\) reshapes to \((6,2)\) and perturbs the base control points to define each candidate boundary.
- A signed distance field is computed; velocities are blended with a sigmoid (soft interface).

---

## 3) Physics: SEM forward model

- Rectangular domain, GLL polynomial order \(p\), PML thickness, total time \(T\), and CFL-controlled \(\Delta t\).
- Velocity field built from B-spline offsets \(\mathbf z\).
- Circular array of receivers; Ricker wavelet source.
- Observations \(\mathbf y\) are generated once at a “true” \(\mathbf z_\star\) and stored.

---

## 4) Log posterior with temperature scaling

\[
\log \pi(\mathbf z)=\log p(\mathbf z)+\log p(\mathbf y\mid \mathbf z;\,\sigma_\text{eff})
\]
where \(\sigma_\text{eff}=\sigma\times T\). The temperature \(T\) controls **posterior sharpness** and **exploration**.

---

## 5) MCMC algorithm: DRAM

### 5.1 Stage 0: Random-walk Metropolis
- Propose \(z'_j\sim \mathcal N(z_j, s_0^2)\) for a random coordinate \(j\).
- Accept with probability \(\min(1, \exp(\Delta \log \pi))\).

### 5.2 Delayed rejection
- On rejection, try a **smaller step** (scale factors e.g., 1.0, 0.25, 0.0625).
- Acceptance probability adjusted for two-stage proposal structure.
- Multiple stages increase overall acceptance without breaking detailed balance.

### 5.3 Adaptive Metropolis
- After burn-in, update covariance matrix from past samples.
- Proposal scale \( 2.38^2 / d \) heuristic.
- Adaptation every fixed number of iterations.

### 5.4 Chains and acceptance
- Single or multiple chains.
- Records per-stage acceptance, ETA, timing.

---

## 6) Likelihood evaluation

- Each `log_posterior(z)` call runs the SEM forward model.
- Gaussian likelihood vs. stored noisy observations with temperature-adjusted \(\sigma\).
- Physics remains fixed; only geometry varies.

---

## 7) Outputs and diagnostics

- **Trace plots** for all 12 parameters.
- **Posterior histograms** with true, initial, and posterior mean.
- **Posterior geometry** overlays (true vs. mean vs. initial).
- **RMS misfit curves** over iterations with noise reference.
- **Waveform comparisons** at posterior mean vs. data.
- Results saved as `dram_mcmc_results.npz`.

---

## 8) Practical guidance

- **Initialization:** Start near prior mean; offsets can be random or informed.
- **Temperature:** Affects mixing. Larger T → broader posterior.
- **Adaptation:** Schedule after burn-in for stability.
- **DR ladder:** Shrinking step sizes improves acceptance.
- **Cost:** Each iteration runs SEM; tune mesh/order for runtime control.

---

## 9) How to run

```bash
python MCMC.py
```

Artifacts include:
- `dram_mcmc_trace_plots.png`
- `dram_posterior_distributions.png`
- `dram_posterior_velocity_models.png`
- `dram_mcmc_misfit_curve.png`
- `dram_mean_posterior_waveform_comparison.png`
- `dram_mcmc_results.npz`

---

## 10) NF vs. MCMC

| Aspect                | Normalizing Flow (NF)                   | MCMC (DRAM)                          |
|------------------------|------------------------------------------|---------------------------------------|
| Posterior quality      | Approximate (may miss tails)            | Asymptotically exact                  |
| Cost per sample        | Low after training                      | High (full SEM call per step)        |
| Initialization         | Requires training                       | Simple                               |
| Expressiveness         | Limited by flow architecture            | Full posterior exploration           |
| Mixing & multi-modality| May struggle                            | Captures multi-modality if well-tuned|

---

## 11) Repro checklist

- Observations generated (clean & noisy).  
- DRAM parameters set (burn-in, adapt interval, DR scales).  
- SEM configuration correct.  
- Chains mix (check trace, histogram).  
- Acceptance rates ~20–40%.  
- RMS misfit stabilizes near noise level.
