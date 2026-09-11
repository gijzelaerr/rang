# Robustness gates: the matched-model result is not sufficient

Objective: improve calibration/imaging quality at comparable cost, or match
quality with materially less compute than a relevant state-of-the-art method.
The following are **toy controls**, not that comparison. See the
[development roadmap](development-roadmap.md) for remaining gates.

## Reproduce

Run each with default seeds 7, 11, 19:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --beam-width-error 0.01
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --gain-per-channel
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --pointing-ripple-arcmin 0.1
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --missing-source-jy 0.02
```

[Archived individual results](results/robustness-stress.json) contain 96 fits,
including the fixed-sky and gain/sky-only controls and nonzero common pointing.
All converge. Every case retains the same held-out time split, sky, noise seed,
gain truth and priors as the earlier benchmark. No hyperparameters are tuned
on the held-out data. Runs use the approximate eight-antenna geometry.

## First results

Ranges over three seeds for joint pointing/gain/flux fits with zero physical
common offset. The residual statistic is squared complex residual divided
by twice the component noise variance, averaged over test visibilities.

| Perturbation | Relative pointing RMSE (arcsec) | Test residual statistic |
| --- | --- | --- |
| Previous matched model | 0.75–1.02 | 1.011–1.023 |
| True beam 1% wider than fitted beam | 1.09–1.46 | 2.38–2.57 |
| Independent per-channel fitted gains | 0.77–1.09 | 1.053–1.065 |
| Added fast pointing ripple | 4.31–4.32 | 1.366–1.384 |
| Omitted 20 mJy source | 12.57–12.88 | 160–165 |

The width perturbation changes both FWHM axes by 1%, keeping squint fixed.
It is a controlled stress amplitude, **not a measured beam-width uncertainty**.
Despite near-arcsecond pointing recovery, off-axis flux ratios relative to the
central source are biased by 0.53–0.76%. Removing the arbitrary global flux
scale does not remove this science error. Visibility prediction is worse by
roughly seven times than in the matched joint fit.

Per-channel gain fitting retains time splines, with independent amplitude and
phase coefficients at four frequencies and the same per-coefficient priors.
Truth remains achromatic: this isolates the cost of additional freedom, not
recovery of realistic spectral gain corruption. Automated noiseless tests
separately inject channel-dependent gain amplitudes and verify joint recovery
with a fixed central flux anchor. No frequency smoothness is implemented yet.

The ripple has 0.1 arcmin amplitude before subtracting the antenna mean,
90-minute period, and deterministic antenna/axis phase offsets. The solver
still has four six-hour-track knots. This exposes out-of-family trajectory
error without inferring a drift model from historical holography.

The omitted source has direction cosines (0.003,-0.004), flux 0.02 Jy at the
component reference frequency, and spectral index -0.7. Even `known_sky` in
this campaign means the **retained four-source sky** is correct, not that the
fifth source is modelled. No claim is made to recover the missing flux. Its
effect on calibration is a sky-model failure, not merely noisy pointing.

## Interpretation for novelty

The matched-model elevenfold prediction improvement does not establish a
robust scientific advantage. These controls expose distinct failure modes:
beam errors corrupt flux ratios, insufficient temporal flexibility corrupts
pointing, and omitted emission produces large calibration errors. Simply
increasing parameter count is not necessarily beneficial either.

A possible contribution is a computationally efficient, uncertainty-aware
choice of physical calibration parameters with checks for sky/beam mismatch.
That remains a hypothesis. It must outperform ordinary joint fitting at matched
quality/cost and then an established pointing/DDE implementation. The current
timings include compilation and concurrent processes; they are recorded for
provenance, **not usable as a performance comparison**.

Next within gates 1–4: antenna-dependent beam/squint perturbations, smooth
frequency-gain corruption, pointing jumps, extended and spectrally incorrect
sky models. Gates 5–7 require uncertainty coverage and independent validation
design; gates 8–10 need verified metadata, geometry and a suitable observation.
