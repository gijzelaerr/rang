# Antenna beam errors can masquerade as pointing

In the current toy, unmodelled dish-to-dish beam-width differences produce **6–9 arcsec relative pointing errors**. One shared beam-width parameter does not repair them. Fitting seven relative antenna-width parameters restores approximately **0.8–1.1 arcsec**, close to the exact-known-beam control.

This strengthens the joint-model baseline; it is not a novelty claim. Joint pointing, beam and sky inference is established, including [BIRO (Lochner et al. 2015)](https://arxiv.org/abs/1501.05304). No comparison with an operational calibrator or measured imaging improvement is made here.

## Controlled comparison

Eight antennas, four frequencies, 24 times, four spline knots and scalar katbeam H response. The [chromatic-gain experiment](chromatic-gains.md) supplies linear-plus-quadratic log-gain truth and degree 2 fitted gains. Source fluxes and power-law indices are fitted with 5% and 0.1 priors. Relative pointing has exactly zero antenna mean; an explicit common spline has a 0.5 arcmin knot prior.

A separate deterministic random stream supplies antenna log-width errors with standard deviation 0.01 before subtracting their antenna mean. Each antenna's multiplier applies to both beam axes, at every frequency and time. These are synthetic width errors, not measured MeerKAT dish uncertainties. Demeaning fixes their geometric-mean multiplier to one; the fitter may still estimate a shared width separately.

Seeds 7, 11 and 19 are paired across all methods, each with physical common pointing zero or constant (+30, −30) arcsec in antenna coordinates. Six whole times are withheld. The table summarizes joint sky/pointing fits; known-sky, wrong-fixed-sky and no-pointing controls remain in the [full archive](results/antenna-beam-widths.json).

| Beam treatment | Relative pointing RMSE (arcsec) | Held-out normalized mean square |
| --- | ---: | ---: |
| Ignore antenna differences; fixed shared width | 6.30–9.26 | 2.602–3.982 |
| Ignore antenna differences; fit shared width | 6.33–9.19 | 2.602–3.989 |
| Exact known antenna widths (oracle) | 0.73–1.07 | 1.043–1.047 |
| Fit relative widths, prior sigma 0.03; fit shared width | 0.79–1.10 | 1.049–1.051 |
| Same, prior sigma 0.003 | 0.81–1.11 | 1.048–1.051 |
| Same, prior sigma 0.0003 | 3.17–4.36 | 1.426–1.722 |

The noise-only expectation of the normalized mean square is one. All 144 stress/prior fits converge, including the inadequate cases. The inferred-width model uses 260 total parameters versus 253 with only shared width. Relative log-width RMSE is 0.000233–0.000355 for the 0.03 prior, approximately 0.023–0.035% in fractional width. The overly tight prior forces substantial width error into pointing again.

This is an in-family recovery test. Antenna-dependent ellipticity, squint, sidelobes, polarization leakage, time-dependent beams and incomplete diffuse sky are not covered. Fewer parameters or better residuals against a wrong beam model do not establish superiority over state of the art.

## Separate relative and common uncertainties

Two additional 12-noise pilots hold truth seed 71 fixed, with zero physical common pointing. Both fit shared width; one omits relative antenna-width freedom and the other includes it with prior sigma 0.03. The same configurations were replayed to add separate relative/common uncertainty diagnostics; these replays are not extra independent trials.

Mean coordinate coverage of nominal 95% local Gauss–Newton intervals:

| Fitted beam model | Relative pointing | Common pointing | Total physical pointing |
| --- | ---: | ---: | ---: |
| Shared width only | 17.0% | 41.3% | 26.3% |
| Shared + relative antenna widths | 95.7% | 85.4% | 89.4% |

This distinction matters: the relative-pointing intervals behave much better after modelling antenna beam differences, while the weak shared mode remains less reliable in this small pilot. Mean median relative standard deviations are 0.868 arcsec without antenna-width freedom and 0.899 arcsec with it—the improvement is primarily removal of bias, not indiscriminately widening intervals. The joint model covers 94 of 96 antenna-width coordinates and 34 of 36 off-axis flux ratios; the shared-only model covers 28 of 36 flux ratios.

These coordinates are correlated across antennas and times; all runs share one truth realization. Do not interpret thousands of coordinates as independent coverage trials or infer a precise coverage guarantee. The intervals include priors, assume the fitted model, and are not a sampled nonlinear posterior. [Component-wise diagnostic replay](results/antenna-beam-coverage-components.json).

## Implementation and reproduction

`make_beam_predictor(..., antenna_log_width=values)` supplies known per-antenna corrections. `solve_pointing(..., beam_antenna_log_width_prior=0.03)` instead infers constant zero-mean log-width deviations in an orthonormal antenna-contrast basis. This prevents duplication of the shared width parameter. The prior applies to contrast coefficients, not independent antenna entries.

`beam_antenna_log_width` returns fitted deviations; known corrections in the predictor remain separate. Optional uncertainty includes `beam_antenna_log_width_std`, `relative_offset_std_arcmin` and `common_offset_std_arcmin`. Existing `offset_std_arcmin` still describes total physical pointing. Total variance is not generally the sum of relative and common variances because they can be correlated.

```sh
.venv/bin/python examples/relative_pointing.py --joint-gains --held-out \
  --common-prior-arcmin 0.5 --chromatic-gain-truth 1 \
  --chromatic-gain-curvature 0.5 --gain-frequency-degree 2 \
  --spectral-index-prior 0.1 --antenna-beam-log-width-std 0.01 \
  --beam-log-width-prior 0.03 --beam-antenna-log-width-prior 0.03
```

Remove the antenna-width prior for the shared-only comparison; remove both beam priors for the fixed-width comparison. For the oracle, remove both beam priors and add `--known-antenna-beam`. Replace the antenna prior by 0.003 or 0.0003 for sensitivity controls. Add `--coverage --truth-seed 71 --seeds 1 2 3 4 5 6 7 8 9 10 11 12` to reproduce each uncertainty pilot.

Uniform-width equivalence, derivative locality, finite differences, invalid inputs, combined shared/relative-width and common-pointing noiseless recovery, and uncertainty-output consistency are tested. The next discriminating experiment is independent model-adequacy selection with out-of-family beam errors and signal-recovery metrics, not another comparison against an intentionally incomplete model.
