# Spectral uncertainty and pointing-mode selection

## Spectra must participate in the pointing solution

We added independent Gaussian spectral-index priors to the joint smooth-pointing solver. Like flux priors, zero sigma fixes a source; `None` keeps every index fixed. The central source remains fixed in these experiments.

With 2% off-axis flux errors and spectral-index errors of ±0.1, twenty seeds at 1 mJy/component give:

| Fit | Pointing RMSE (arcsec) | Held-out RMS (mJy/component) |
|:--|--:|--:|
| Known correct sky | 2.402 | 1.066 |
| Incorrect fixed sky | 22.560 | 6.720 |
| Joint flux, incorrect fixed spectra | 68.511 | 2.394 |
| Joint flux and spectra | 4.132 | 1.068 |

These are means of realization-wise RMSEs. Lower prediction residuals can accompany substantially worse physical pointing estimates. The correct-spectrum assumption was essential to the earlier flux-only recovery result.

## Information-selected modes: a useful negative result

An experimental option, `minimum_mode_information`, evaluates the pointing and nuisance Jacobians at zero pointing and the supplied sky. It eliminates prior-normalized sky nuisance parameters, whitens by the pointing prior and retains generalized information modes above a threshold.

For whitened pointing Jacobian `P`, prior-normalized sky Jacobian `S`, and pointing-prior precision `R`,

```math
F=P^TP-P^TS(S^TS+I)^{-1}S^TP,
\qquad Fv=\eta Rv.
```

Modes with `η ≥ threshold` are retained; discarded directions are fixed at zero. The basis is frozen before fitting and never uses held-out data. This is prior-dependent information truncation, not the proposed dual-budget protection rule and not inherently a new method.

At 10 mJy noise and fixed smoothing, threshold 1 keeps 56 of 96 pointing parameters and reduces mean pointing RMSE from 19.271 to 18.135 arcsec. At 1 mJy it keeps all 96 modes and is equivalent to the full joint fit.

The decisive comparison gives **both** methods the same five smoothing candidates and separate training, validation and test times. Each method selects its own smoothing using validation residuals, refits on training plus validation, and is evaluated on untouched test times.

| Validation-tuned solver | Pointing RMSE | Test RMS | Mean retained pointing modes |
|:--|--:|--:|--:|
| Full joint inference | 18.501 arcsec | 10.402 mJy | 96 |
| Information-selected joint inference | 18.497 arcsec | 10.422 mJy | 44.8 |

There is **no meaningful pointing-accuracy gain** in this 20-seed comparison. Reduced dimensionality may be useful, but the current code still constructs full Jacobians before projection and no runtime advantage is claimed. A novelty claim based only on the fixed-smoothing comparison would be misleading.

Related work already studies calibration regularization and automated time-scale selection: [Sob et al. (2021)](https://academic.oup.com/mnras/article/504/2/1714/6211002), [Tasse (2014)](https://arxiv.org/abs/1403.6308). Information-based truncation and nuisance elimination are established mathematical techniques.

## Reproduce

```sh
.venv/bin/python examples/sky_uncertainty.py --spectral-error .1 --spectral-prior .2 --minimum-mode-information 1 --output outputs/spectral-20
.venv/bin/python examples/sky_uncertainty.py --spectral-error .1 --spectral-prior .2 --minimum-mode-information 1 --noise .01 --output outputs/spectral-noisy-20
.venv/bin/python examples/mode_selection.py --output outputs/mode-cv-20
```

The scripts default to seeds 1–20. The validation comparison uses smoothing `[0.0001, 0.001, 0.01, 0.1, 1]`, six spline knots, 5% flux priors and spectral-index sigma 0.2. Error signs, source layout and beam are fixed; these are not population-wide performance claims. [Saved summaries and paired validation outcomes](results/symmetry-and-spectral-campaigns.json).

The more focused physical investigation that followed is [the beam-symmetry ambiguity](beam-rotation.md).
