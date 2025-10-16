# RQS_Coupling — Variational Inference with Piecewise Rational-Quadratic (RQS) Normalizing Flows + SEM

This README explains the **theory and design** behind `RQS_Coupling.py`: how it combines a **Rational-Quadratic Spline (RQS) normalizing flow** (ActNorm + PRQ coupling) with a **Spectral Element Method (SEM)** forward simulator and adjoint to perform **Bayesian inversion** of a 12‑D B‑spline boundary.

---

## 1) Problem setup (geometry → waveforms → inverse inference)

We infer a 12‑dimensional latent vector  
\[
\mathbf z \in \mathbb{R}^{12}
\]
that encodes **offsets to six B‑spline control points** (x/z per node). These offsets deform a **closed cubic B‑spline** interface separating two media with velocities \((v_{\text{in}}, v_{\text{out}})\). Given a source and a circular array of receivers, the **SEM** forward model generates synthetic seismograms \(\mathbf y_{\text{syn}}(\mathbf z)\). With observed data \(\mathbf y\), a Gaussian noise model with standard deviation \(\sigma\) defines
\[
\log p(\mathbf y \mid \mathbf z) \;=\; -\frac{1}{2\sigma^2} \big\| \mathbf y - \mathbf y_{\text{syn}}(\mathbf z) \big\|_2^2 + \text{const}.
\]

We place a factorized Gaussian prior on \(\mathbf z\) centered at 0 (the base control points). A **normalizing flow** \(q_\phi(\mathbf z)\) is trained by maximizing the **ELBO**:
\[
\mathcal L(\phi) \;=\; \mathbb E_{q_\phi}[\log p(\mathbf y\mid\mathbf z)] \;+\; \mathbb E_{q_\phi}[\log p(\mathbf z) - \log q_\phi(\mathbf z)].
\]

---

## 2) B‑spline boundary parameterization (12 DOF)

- Six control points define a **closed cubic B‑spline**; the first point is repeated to enforce closure.  
- The latent vector \(\mathbf z\) (shape 12) reshapes to \((6,2)\) and perturbs a fixed template \(C_{\text{base}}\):  
  \[
  C(\mathbf z) \;=\; C_{\text{base}} + \text{reshape}(\mathbf z,6,2).
  \]
- A **signed distance field** from SEM nodes to the curve is computed with Newton refinement; a **soft interface** (sigmoid with width \(\tau\)) blends \(v_{\text{in}}\) and \(v_{\text{out}}\) to form the nodal velocity \(v(\mathbf x;\mathbf z)\).

---

## 3) SEM forward & adjoint (physics layer)

- **Mesh:** structured quadrilateral GLL grid, order \(p\).  
- **Time stepping:** explicit central difference with **CFL‑safe** \(\Delta t\).  
- **Boundary:** PML on all sides.  
- **Acquisition:** circular receiver ring; source is a Ricker wavelet.  

The code uses a persistent SEM instance (memoized by the configuration) and exposes a **forward+adjoint** API that, for a given \(\mathbf z\), returns both \(\log p(\mathbf y\mid\mathbf z)\) and its gradient \(\partial \log p/\partial \mathbf z\). A small **custom autograd function** inserts this adjoint gradient directly into the backprop graph so that the flow parameters receive informative PDE‑consistent gradients without unrolling the solver.

---

## 4) Normalizing flow with **RQS coupling** (NSF‑style)

### 4.1 Base distribution and change of variables
The flow transforms a base Gaussian \(\mathcal N(\mu_0,\Sigma_0)\) via an invertible map \(f_\phi\) built from **ActNorm**, **RQS coupling layers**, and **fixed permutations**:
\[
\mathbf z = f_\phi(\mathbf x),\quad \mathbf x \sim \mathcal N(\mu_0,\Sigma_0),\quad
\log q_\phi(\mathbf z) = \log \mathcal N(\mathbf x;\mu_0,\Sigma_0) + \sum_{\ell} \log\left|\det \frac{\partial f_\ell}{\partial \text{input}}\right|.
\]

### 4.2 ActNorm (per‑dim affine)
ActNorm applies a per‑dimension affine transform \(y = (x + b)\odot e^{s}\) with data‑dependent initialization on the first batch. Its Jacobian log‑det is simply \(\sum s\).

### 4.3 Masked **Piecewise Rational‑Quadratic** (PRQ/RQS) coupling
We split the features into **identity** and **transform** subsets by a binary mask (alternated across layers and permuted between blocks). A small MLP predicts parameters of a **monotonic RQS** on each transformed dimension with \(K\) bins and **linear tails** on \([ -B, B ]\).

- **Bin parameters.** For each transformed scalar, the network outputs
  - unnormalized **widths** \(\tilde w\in\mathbb R^K\), **heights** \(\tilde h\in\mathbb R^K\), and **derivatives** \(\tilde d\in\mathbb R^{K+1}\);
  - they are normalized to valid partitions: \(w = \text{softmax}(\tilde w)\), \(h=\text{softmax}(\tilde h)\), then cumulated to \(\text{cumwidths}\) / \(\text{cumheights}\) on \([0,1]\); node derivatives are enforced positive via \(\text{softplus}(\tilde d)+d_{\min}\).
- **Forward map.** With \(x\in[-B,B]\), map to \(x_{\text{sc}}\in[0,1]\), find the bin index \(k\), and evaluate the **rational‑quadratic** \(s(t)\) inside the bin (Durkan et al.):
  \[
  s(t) = \frac{a t^2 + b t (1-t)}{a + (b+c-2a)\,t(1-t)}, \quad t\in[0,1],
  \]
  where \(a = h_k/w_k\), \(b=d_k\), \(c=d_{k+1}\). The output is \(y_{\text{sc}} = y_k + h_k\,s(t)\), then \(y\in[-B,B]\) by linear rescaling (tails are identity outside \([-B,B]\)).
- **Log‑det.** The derivative \(dy/dx\) has a **closed form**; the coupling log‑det is the sum of \(\log|dy/dx|\) across transformed dimensions.  
- **Inverse.** Given \(y\), we invert the spline by Newton iterations on \(t\) (monotonicity guarantees convergence), then recover \(x\).

This gives a **flexible, smooth, and strictly monotone** 1‑D bijection per feature—combined with masking/permutations, the joint transform is expressive and invertible.

---

## 5) ELBO with SEM‑adjoint likelihood

For a minibatch of \(n\) samples \( \{ \mathbf z^{(i)} \}_{i=1}^n \sim q_\phi \):
\[
\widehat{\mathcal L}
= \frac1n \sum_{i=1}^n \Big(
\underbrace{\log p(\mathbf y\mid\mathbf z^{(i)})}_{\text{SEM adjoint}}
+ \underbrace{\log p(\mathbf z^{(i)})}_{\text{Gaussian prior}}
- \underbrace{\log q_\phi(\mathbf z^{(i)})}_{\text{change of vars}}
\Big).
\]
The code **linearly schedules** \(n\) from a small to a larger value across epochs to reduce estimator variance as training proceeds.

---

## 6) Optimization hygiene & diagnostics

- **Precision:** double (`float64`) for stability.  
- **Gradients:** soft clipping when the global norm exceeds a threshold; record total/mean/max norms and “explosion” flags.  
- **Architecture:** \(L\) blocks of `[ActNorm → RQS coupling → Permutation]`; moderate MLPs (2×128 hidden).  
- **Hyperparameters:** number of flow layers, number of bins \(K\), tail bound \(B\), learning rate, and ELBO sample schedule.  
- **Plots/Artifacts:** ELBO components, gradient history, posterior samples (boundary overlays), 12 histograms with prior/true/posterior mean.

---

## 7) Practical advice

- **Likelihood scale \(\sigma\):** mismatched noise causes over/under‑confidence; tune to observed SNR.  
- **CFL:** if the solver reduces \(\Delta t\) substantially, consider lower polynomial order or coarser elements.  
- **Flow capacity:** start smaller (few blocks, smaller \(K\), moderate \(B\)) and increase once ELBO stabilizes.  
- **Identifiability:** circular acquisition can admit multiple shapes; broad posteriors may be expected—inspect the overlays and marginals.

---

## 8) Minimal “how to run”

1. Configure the SEM domain/time/source/receivers/velocity and training hyperparameters in `RQS_Bspline.py`.  
2. Run the script; it first **generates observations** with the SEM at a randomly sampled “true” geometry, then trains the RQS flow with adjoint‑based likelihood gradients.  
3. Inspect `posterior_samples.npy`, `trained_flow_model.pth`, and figures such as `nf_posterior_distributions.png`, `nf_boundary_comparison.png`, and `elbo_history_real.png`.

---

## 9) What’s different from `affine_Bspline.py`?

- **Coupling transform:** affine \(\to\) **Rational‑Quadratic Spline** with \(K\) bins, linear tails, and Newton‑based inverse.  
- **Parameters per dim:** affine predicts scale/shift; RQS predicts **\((3K+1)\)** parameters per transformed scalar (bin widths, heights, and node derivatives).  
- **Expressiveness:** RQS layers capture complex, non‑linear marginal shapes while preserving monotonicity and tractable Jacobians.

---

## 10) Repro checklist

- Verify SEM config and observation generation.  
- Confirm adjoint gradients are injected (ELBO improves, gradients finite).  
- Tune \(K\) and tail bound \(B\) for stability vs. flexibility.  
- Validate posterior overlays vs. the “true” boundary; re‑simulate at the posterior mean for sanity.
