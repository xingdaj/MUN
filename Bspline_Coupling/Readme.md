#### Introduction
* We have upload 5 files to infer the same posterior distribution problem -- One for forward code, and four for different posterior distribution solver
*-- 'SEM' is for forward calculation
*-- 'MCMC'is for MCMC sampler with  Delayed Rejection Adaptive Metropolis (DRAM)  
*-- 'Affine_Coupling' is the coupling flows with affine function (Normalizing Flows)
*-- 'RQS_Coupling' is the coupling flows with Rational-Quadratic Spline (RQS) function (Normalizing Flows)
*-- 'RQS_autoregressive' is the auroregressive flows with Rational-Quadratic Spline (RQS) function (Normalizing Flows)

#### Notice
* For normalizing flows, you'll meet one problem when 'min_elbo_samples = 1' : the gradient will be extreme large. That will result in the failed optimization.
The reason is that:
'During the first forward pass, each ActNorm layer calls initialize_parameters(x) with the current batch. When min_elbo_samples=1, that batch has shape (1, 12).
Your init code explicitly sets std=0 when batch_size==1, then clamps it to 1e-6, and sets logs = log(1/std) ≈ log(1e6) ≈ 13.8. That means ActNorm multiplies activations by ~1e6 and adds a huge log-det term.
Downstream, that blows up the gradient norms. With two or more samples, std is computed from data and stays sane, so gradients don’t explode.'
Of course we can solve that by changing 'clamps it to 1e-6'. However, I just want to show that more samples are better in one iteration, so that I leave the error here to remind the users.
     
