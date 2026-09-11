# Uncertainty coverage and a beam-width recovery control

## What is implemented

`solve_pointing(..., estimate_uncertainty=True)` computes local Gauss–Newton
marginal standard deviations from the final whitened residual Jacobian,
including all prior rows. SVD propagation avoids explicitly forming normal
equations. Numerically rank-deficient precision is rejected rather than
mistaking pseudoinverse zeros for certainty. Outputs include pointing,
flux and flux-ratio standard deviations and optionally beam log-width.

This is a local linearized Gaussian approximation, not exact posterior
sampling or a model-error covariance. Noise variance is supplied, not inflated
until residuals look acceptable. Frozen parameters have zero conditional
variance; that is not evidence that the corresponding physical quantity is
known. Flux-ratio uncertainties use the joint covariance, not independent
error propagation of two marginal flux errors.

`beam_log_width_prior=0.03` fits one global log-FWHM multiplier through the
tabulated beam callback. Both axes scale equally at all frequencies; squint
is fixed. Its prior is centred at the supplied beam with log-width sigma0.03,
a controlled assumption, not a measured MeerKAT uncertainty. The prior and
its correlations with pointing, gains and sky are included in propagation.

## Repeated-noise experiment

Hold truth seed71 fixed; vary noise seeds1–12, retaining the same six held-out
times as before. Gain/pointing trajectories and gain/flux priors are unchanged
across noise draws. Only the joint-flux, zero-common-pointing case is used in
coverage mode. Truth is fixed, not drawn from the prior.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --coverage --truth-seed 71 --seeds 1 2 3 4 5 6 7 8 9 10 11 12
```

Repeat with `--beam-width-error 0.01`, then with both
`--beam-width-error 0.01 --beam-log-width-prior 0.03`.

| Case | Mean point-coordinate 95% coverage | Off-axis flux-ratio 95% coverage | Mean held-out residual statistic |
| --- | --- | --- | --- |
| Correct fixed beam | 96.0% | 35/36 | 1.030 |
| True beam 1% wider, fixed fitted beam | 85.6% | 0/36 | 2.556 |
| Same width error, one fitted width parameter | 96.0% | 34/36 | 1.030 |

The inferred width multiplier spans1.009765–1.010264; truth is1.01. The fitted
width addresses the specific injected mismatch and restores predictive
adequacy here. Absolute flux intervals are broad due to the gain/flux scale
ambiguity, so flux-ratio coverage is essential to expose relative-flux bias.

Pointing coordinates are strongly correlated across time and antennas;
36 flux-ratio intervals are also correlated. These counts are descriptive,
**not independent binomial trials or a precise coverage guarantee**. Twelve
noise draws at one fixed truth are a pilot, not complete uncertainty validation.
Coverage is averaged over all evaluated times, including training times;
held-out prediction is scored separately.

## Stronger temporal holdout

`--joint-gains --held-out --holdout-mode contiguous` with the standard three
seeds withholds time indices9–14, all frequencies/baselines. This is a central
gap, not extrapolation. Joint fits with zero common pointing have held-out
residual statistics1.0108,1.0136,0.9897. The matched spline assumption remains.
The archive includes gain-only and nonzero-common controls as well.

[Individual results and provenance](results/uncertainty-coverage.json).

## Research interpretation

This establishes a baseline for uncertainty-aware model checking and a
successful **known-family** correction. Joint beam-width inference itself
is established prior art (see BIRO in the literature notes). No state-of-the-art
quality or performance advantage is claimed. Gates5 and7 have pilot results;
they are not complete. Gate6 still needs systematic common-mode/prior sweeps.

Next: wrong-family beam perturbations, wider truth/noise coverage campaigns,
and a training/validation/test separation for selecting parameter complexity.
The candidate must beat an ordinary joint beam fit, not just a deliberately
misspecified fixed-beam baseline.
