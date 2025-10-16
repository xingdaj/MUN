# RQS_autoregressive — Variational Inference with **Autoregressive** RQS Normalizing Flows + SEM

This README explains the **theory and design** behind `RQS_autoregressive.py`: how an **autoregressive** normalizing flow built from **Rational–Quadratic Spline (RQS)** transforms (MADE-style) is coupled to a **Spectral Element Method (SEM)** forward simulator and adjoint to perform **Bayesian inversion** of a 12-D B-spline boundary.

---

## 1) Problem setup: geometry → waveforms → inverse inference

We infer a 12-dimensional latent vector  
\[
\mathbf z\in\mathbb R^{12}
\]
that encodes **offsets to 6 B-spline control points** \((x,z)\). These offsets deform a **closed cubic B-spline** interface separating two media with velocities \((v_\text{in},v_\text{out})\). Given a source and a circular array of receivers, the **SEM** forward model generates synthetic seismograms \(\mathbf y_\text{syn}(\mathbf z)\). With observed data \(\mathbf y\) and Gaussian noise \(\sigma\), the likelihood is
\[
\log p(\mathbf y\mid\mathbf z)=-\frac1{2\sigma^2}\|\mathbf y-\mathbf y_\text{syn}(\mathbf z)\|_2^2+\text{const},
\]
and with a factorized Gaussian prior on \(\mathbf z\) we maximize the **ELBO**
\[
\mathcal L(\phi)=\mathbb E_{q_\phi}\!\left[\log p(\mathbf y\mid\mathbf z)\right]+\mathbb E_{q_\phi}\!\left[\log p(\mathbf z)-\log q_\phi(\mathbf z)\right].
\]
(Observed data are generated once at “true” control-point offsets; see §3.)

---

## 2) Geometry: 12-DOF closed cubic B-spline boundary

- Six control points define a **closed cubic** B-spline (degree \(k=3\)), with the first control point repeated to enforce closure.  
- The latent vector \(\mathbf z\) reshapes to \((6,2)\) and perturbs a base template \(C_\text{base}\), i.e., \(C(\mathbf z)=C_\text{base}+\text{reshape}(\mathbf z,6,2)\).  
- A **signed-distance field** from SEM nodes to the curve is computed; a **soft interface** (sigmoid with width \(\tau\)) blends \(v_\text{in}\) and \(v_\text{out}\) into nodal velocities \(v(\mathbf x;\mathbf z)\).  
- The script draws and saves figures of the “true” geometry and velocity model for sanity checks.

---

## 3) Physics layer: SEM forward model & observation generation

- **Mesh:** structured quadrilateral GLL grid with user-specified polynomial order \(p\).  
- **Time stepping:** explicit central difference with **CFL-safe** \(\Delta t\) (adjusted internally by the SEM).  
- **Boundary:** PML all around.  
- **Acquisition:** circular receiver ring; Ricker source.  
- **Observations:** the script runs SEM at the **true** offsets to save **clean** and **noisy** traces (noise std `obs_noise_std`). Plots compare clean/noisy seismograms per receiver.

---

## 4) SEM adjoint via a custom autograd Function

To make \(\log p(\mathbf y\mid\mathbf z)\) **differentiable w.r.t.** \(\mathbf z\) without unrolling the PDE solver, the script defines `SEMLikelihoodAdjointFn`:

- **Forward:** calls `SEMSimulation.run_forward_and_adjoint({...})` with absolute control points and observed data; returns a scalar **log-likelihood** and an adjoint gradient \(\partial\log p/\partial\mathbf z\).  
- **Backward:** injects the stored adjoint gradient into the computational graph, propagating derivatives through the flow \(q_\phi\).  
This follows the standard **adjoint-state** approach and keeps training practical. (The SEM instance is **cached** by config to avoid re-initialization.)

---

## 5) Autoregressive RQS flow (NSF-AR)

### 5.1 Base distribution & change of variables
The flow maps \(\mathbf x\sim\mathcal N(\boldsymbol\mu_0,\boldsymbol\Sigma_0)\) through a stack of invertible transforms:
\[
\mathbf z=f_\phi(\mathbf x),\qquad
\log q_\phi(\mathbf z)=\log \mathcal N(\mathbf x;\boldsymbol\mu_0,\boldsymbol\Sigma_0)+\sum_\ell \log\left|\det\frac{\partial f_\ell}{\partial\text{input}}\right|.
\]
The base covariance is parameterized by a learned **Cholesky** factor for stability.

### 5.2 MADE-style autoregressive parameterization
- The network is a **masked MLP** (MADE) so that the parameters for dimension \(i\) depend only on the **prefix** \(x_{<i}\).  
- Masks are constructed once from degree assignments; outputs are **per-dimension** spline parameters.  
- A **Glow-style ActNorm** (data-dependent init) precedes each AR block for numerical hygiene and simple log-det.

### 5.3 1-D monotone RQS transform per dimension
For each scalar, the MLP predicts **bin widths/heights** and **knot derivatives**; these are normalized to valid partitions with **softmax** and positive derivatives with **softplus**. The transform:

- is **piecewise rational–quadratic** on a bounded interval \([-B,B]\) with **linear tails**,  
- has a **closed-form derivative** (tractable log-det),  
- is **invertible** via **Newton iterations** (monotonicity ensures convergence).  
The script includes explicit forward/inverse routines (`_rqs_1d`) and parameter normalization helpers, with a near-**identity initialization** for stable early training.

### 5.4 Full flow stack
Multiple blocks of **[ActNorm → AR-RQS]** are stacked; the composed Jacobian log-det is just the sum of per-layer contributions. Sampling (`forward(n)`) draws from the base and pushes forward; `log_prob(z)` inverts through the stack and evaluates the base density.

---

## 6) ELBO with SEM-adjoint likelihood

For an epoch-dependent number of Monte-Carlo samples \(n\) (linearly scheduled between `min_elbo_samples` and `max_elbo_samples`):

1. Draw \(\{\mathbf z^{(i)}\}_{i=1}^n\sim q_\phi\).  
2. Compute \(\log q_\phi(\mathbf z^{(i)})\) by inverting through the AR stack.  
3. Evaluate the **prior** \(\log p(\mathbf z^{(i)})\) (factorized Gaussian).  
4. Call the **SEM adjoint** to get \(\log p(\mathbf y\mid \mathbf z^{(i)})\) and gradients.  
5. Average \(\big(\log p(\mathbf y\mid\mathbf z^{(i)})+\log p(\mathbf z^{(i)})-\log q_\phi(\mathbf z^{(i)})\big)\) to form the ELBO.  
The script records ELBO and its components every epoch.

---

## 7) Optimization hygiene & gradient control

- **Precision:** `torch.float64` throughout.  
- **Gradient control:** monitor total/mean/max norms; apply **soft scaling** (clipping-by-scaling) when exceeding a threshold; log “explosion” events.  
- **Identity init:** final layer biases are set so that widths/heights are uniform and derivatives ≈ 1, making the initial transform close to identity.  
- **Tail bound:** chosen from base covariance or specified; keeps most mass inside spline support.

---

## 8) What the training loop does

- Toggles a `VERIFY_PROJECTION` flag periodically to sanity-check geometry projections.  
- Computes ELBO and backpropagates (adjoint gradient flows into flow params).  
- Scales gradients if needed; steps Adam optimizer.  
- Every few epochs, samples thousands from \(q_\phi\) and renders **posterior boundary overlays**.  
- At the end, saves **posterior samples**, **trained flow weights**, **ELBO curves**, **gradient history**, and summary figures (histograms and boundary overlays).

---

## 9) Outputs

- `posterior_samples.npy` — samples of \(\mathbf z\) from \(q_\phi\).  
- `trained_flow_model.pth` — learned AR-RQS flow weights.  
- `nf_posterior_distributions.png` — 12 histograms with prior mean, true value, and posterior mean.  
- `nf_boundary_comparison.png` — true/prior/posterior-mean boundaries + sample cloud.  
- `elbo_history_real.png`, `gradient_history.png` — training diagnostics.  
- Intermediate overlays in `intermediate_posteriors_real/`.

---

## 10) Practical guidance

- **Noise scale \(\sigma\):** match to SNR; too small makes the likelihood overly sharp and training unstable.  
- **Flow capacity:** start with fewer blocks/bins and moderate tail bound; scale up after ELBO stabilizes.  
- **Mesh/time step:** if CFL reduces \(\Delta t\) significantly, consider lower polynomial order or coarser mesh to keep iteration time reasonable.  
- **Identifiability:** circular acquisition can be ambiguous; posterior breadth isn’t always a bug—inspect overlays and marginals.

---

## 11) What’s different vs. `affine_Bspline.py` and `RQS_Bspline.py`?

- **Conditioning:** this script is **autoregressive** (MADE) rather than **coupling-based** (mask-split). Each dim is transformed conditioned on the prefix \(x_{<i}\), increasing expressiveness for complex posteriors.  
- **Spline layer:** uses the same **monotone RQS** idea but parameterized **per dimension** by an AR network; inversion uses Newton per dimension at evaluation time.  
- **Base covariance:** explicitly learned via a **Cholesky** factor, not fixed isotropic, improving fit when parameters are correlated.

---

## 12) Minimal “how to run”

1. Adjust **SEM configuration** (domain, time, source, receivers, velocity) and **flow hyperparameters** (number of flows, hidden size, number of bins, tail bound) at the top of the script.  
2. Run:
   ```bash
   python RQS_autoregressive.py
   ```
   (The script first generates observations, then trains the AR-RQS flow with adjoint-based likelihood gradients.)  
3. Inspect the saved `.npy`, `.pth`, and figure files listed in §9.

---

## 13) Repro checklist

- Observations generated and saved (clean & noisy).  
- Adjacent SEM–adjoint call returns finite log-likelihood and gradients.  
- ELBO improves and gradient norms remain finite (see `gradient_history.png`).  
- Posterior overlays contract toward the true boundary; histograms center near true offsets with credible spread.  
- Saved `posterior_samples.npy` reproduces figures and statistics on reload.
