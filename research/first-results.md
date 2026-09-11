# First results: signal preservation is not flux accuracy

2026-09-11. A reproducible simulation result for discussion, not a new calibration algorithm or a result on observed MeerKAT data.

## Main observation

A 2% primary-beam width error produces a roughly 71% overestimate of a faint extended source's flux, even though a complete-recalibration injection test reports a response of 99.75%. Adding one shared beam-width parameter resolves this bias in the controlled model. Our prototype adaptive protection does not improve on ordinary joint sky/pointing fitting.

Twenty paired random seeds (1–20), each with independent noise and pointing draws, give the following mean recovered flux densities. The true source flux is 20 mJy in every realization.

| Calibration model | Correct beam width: flux (mJy) | True width 2% larger: flux (mJy) | 2% case: injection response |
|---|---:|---:|---:|
| Fixed beam, no pointing fit | 19.050 | 32.217 | 100.000% |
| Pointing only, science source omitted during calibration | 12.129 | 20.948 | 75.465% |
| Joint sky + pointing, fixed width | 20.030 | 34.269 | 99.748% |
| Joint sky + pointing + shared width | 20.081 | 20.124 | 99.990% |
| Protected pointing modes, fixed width | 20.030 | 34.269 | 99.748% |
| Known pointing, fixed nominal width | 20.022 | 33.199 | 100.000% |

The apparently good pointing-only flux in the mismatched case is cancellation of errors: it still suppresses about a quarter of the injected target response. The known-pointing control is **not** a known-beam oracle when width is mismatched.

For the mismatched case, held-out visibility RMS improves from 11.706 mJy per real/imaginary component with joint fixed-width fitting to 9.976 mJy with width fitting; the injected noise standard deviation is 10 mJy. The held-out samples are the last six time samples and are not used to fit parameters or source amplitude. Finite-injection metrics, separately, use all samples after refitting on the training subset.

Means and sample standard deviations, seeds, settings and executable SHA-256 are preserved in [the campaign summary](results/beam-width-campaign.json). Scatter across these realizations is not a posterior uncertainty interval or a guarantee over other skies.

## Why the injection check misses the bias

Locally, let a recovered scalar flux behave as `f_hat = a f + b`, where `a` describes response to additional sky signal and `b` is a residual beam/bright-source contamination projected onto the science template. An injection-and-difference test measures `a`; a nearly constant `b` cancels. Thus `a ≈ 1` imposes no bound on `b`.

This is an interpretation of the experiment, not a proof that the nonlinear fit is affine. In the implemented test the response is measured in visibility residuals after bright-model subtraction, not directly as a derivative of the recovered flux. Both illustrate the limitation of differential checks. Orthogonal residual distortion is also recorded: a near-unit projection alone does not ensure preservation of morphology.

## What was actually implemented

Rust evaluates a scalar RIME with the full non-coplanar `w(n−1)` phase. The toy uses eight approximate published MeerKAT core horizontal positions, zero antenna heights, 24 time samples spanning six hours, and four frequencies from 0.95 to 1.55 GHz. This is not a verified three-dimensional MeerKAT survey model; see [geometry provenance](../data/README.md).

The power beam is Gaussian, with nominal FWHM `1.02 λ / 13.5 m`. Four bright sources have exactly known positions, fluxes and spectra. The faint component has known extended morphology, with different resolved quadrature grids for truth and fitting. Sixteen constant antenna-frame pointing parameters and its amplitude are fitted. The new model adds a shared frequency-independent fractional width correction, parameterized as `width = nominal_width × exp(q/100)`.

The solver is regularized Gauss–Newton with a profiled linear science amplitude and backtracking, not posterior sampling. Priors are 1 arcmin per pointing component, 1% log width, and 0.1 Jy for science amplitude around zero. The final amplitude is re-estimated without its penalty, conditional on the fitted instrument. Analytic pointing and width derivatives are checked against finite differences. The width fit has 17 physical parameters versus 16 for the fixed-width joint model.

## Prior work and limits

Joint beam, pointing and flux inference is established: Lochner et al.'s [BIRO paper](https://arxiv.org/abs/1501.05304), co-authored by Oleg Smirnov, jointly inferred source fluxes, pointing errors, beamwidth and noise in simulated data. This toy is not an implementation of BIRO, nor evidence of novelty over it. Our proposed adaptive selection must still show an advantage against matched joint-inference baselines; it has not done so here.

The successful width fit is deliberately a favorable case: the true error lies exactly in the added parameter family, and bright-source properties are perfect. It does not establish recovery with uncertain calibrator spectra, measured asymmetric beams, time-variable gains, polarization leakage, RFI, ionospheric structure or full imaging/deconvolution. The width prior has not yet been swept. Local protection budgets do not imply global nonlinear fidelity, and the width-fitting model is not itself protected by adaptive mode selection.

## Reproduce

```sh
python3 examples/toy.py --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --output outputs/campaign-matched
python3 examples/toy.py --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --beam-error 0.02 --output outputs/campaign-width
```

Each command builds the Rust binary and produces raw JSON plus PNG/PDF comparison plots. Add `--no-plot` if Matplotlib is unavailable. Generated artifacts are ignored by Git; the compact numerical summary above is retained in the repository.

## Useful next research question

When bright-source fluxes and spectra are uncertain, which combinations of beam and sky remain identifiable, and can observation diversity or externally measured beam priors constrain the resulting absolute flux bias? That is the next discriminating test—not further tuning to make a transfer-only metric look favorable. A meaningful new proposal would need to bound both signal distortion and plausible model-error contamination, and outperform ordinary joint inference at matched assumptions and computational cost.
