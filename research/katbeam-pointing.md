# Chromatic beam shape supplies pointing information in the toy

## A measurement-informed model, not calibrated telescope data

Rang now supports a differentiable adapter for
[SARAO's katbeam](https://github.com/ska-sa/katbeam). Its co-polar voltage model
uses a cosine-aperture taper with frequency-dependent widths and squints
derived from holography. The upstream description identifies it as a
simplified, averaged beam at 60° elevation, not a substitute for individual
antenna beams under all observing conditions.

We pin public PyPI **katbeam 0.1**, model `MKAT-AA-L-JIM-2020`. That release's
table spans **900–1650 MHz**, unlike the latest GitHub table. Rang reads its
parameters at runtime and records their hash; no upstream coefficient table
is copied into this repository. H and V below are separate scalar co-polar
experiments, not a joint full-polarization solve.

## Controlled information audit

The eight-antenna fixture has 24 times, four frequencies (950, 1150, 1350,
1550 MHz), all 28 baselines, four point sources and 1 mJy noise per real
visibility component. The full w phase is retained. Each case allows arbitrary
antenna/time/channel complex gains, differential pointing at each time, all
other common temporal modes, and independent source/channel amplitudes.
There is no pointing or gain prior. Beam parameters are fixed and exactly known.
The local audit is evaluated at zero pointing and unit gains.

| Model or ablation | H: local coordinate bounds (arcsec) | V: local coordinate bounds (arcsec) |
| --- | ---: | ---: |
| Native cosine profile, chromatic widths and squint | 7.958, 7.998 | 8.038, 8.132 |
| Remove squint | 7.864, 7.972 | 8.571, 8.466 |
| Freeze axis ratio; retain geometric-mean width versus frequency, remove squint | 112.118, 119.404 | 103.242, 109.687 |
| Gaussian profile with native chromatic widths, no squint | 8.403, 8.309 | 8.969, 8.745 |
| Gaussian profile with frozen axis ratio, no squint | Degenerate | Degenerate |
| Circular cosine profile, no squint | Degenerate | Degenerate |
| Native model, no beam rotation | Degenerate | Degenerate |

The frozen axis ratio is evaluated at 1284 MHz. The same geometric-mean width
curve is retained to isolate the changing aspect ratio rather than ordinary
beam narrowing. These modified models are mathematical controls, not
additional measured beam products.

Here **chromatic ellipticity supplies far more information than the departure
of a fixed-shape profile from a Gaussian**. This motivates testing how well
that frequency-dependent shape is known; it does not establish universal
dominance for other fields, tracks, beams or calibration models.

With independent 1% external log-flux uncertainties on all 16 source/channel
amplitudes, the native H bounds become 5.395 and 5.706 arcsec; V becomes 5.420
and 5.704 arcsec. The [information budget](information-budget.md) distinguishes
external assistance from standalone visibility information.

These are **local Fisher bounds, not measured nonlinear recovery errors**.
The nominal noise, four-source field, averaged beam and eight approximate
antenna positions do not constitute a MeerKAT sensitivity forecast. No real
visibility data have been fitted. Beam uncertainty and scalar-polarization
limitations remain particularly important.

## Why this does not contradict the Gaussian ambiguity

The exact [Gaussian transformation](gaussian-gauge.md) requires a physical,
frequency-independent pointing change consistent with the beam metric at
every channel. Ordinary scalar λ/D narrowing can be absorbed into the
source/channel amplitudes. Frequency-dependent aspect ratio combined with
rotation generally requires different compensating pointing trajectories at
different channels, which are not allowed in the mechanical pointing model.

The native Gaussian-profile control therefore lifts the null space, whereas
the Gaussian with a common aspect ratio remains degenerate. Suppressing
rotation removes the time diversity and restores a common-shift/sky ambiguity
even for the native profile at the zero-pointing linearization.

## Implementation and verification

`rangtoy.beams` supplies validated tables, a stable JAX cosine taper, direct
voltage prediction and a predictor callback accepted by the smooth pointing
solver and audit. The analytic/Rust path remains the default. Custom predictors
cannot silently be combined with analytic beam-shape flags.

The taper uses a sinc identity to remove its apparent pole and a near-origin
series to avoid differentiating a square root at zero. Tests cover the
half-power point, removable pole, signed sidelobes, central Hessian,
finite-difference derivatives and both polarizations against katbeam.
Release 0.1 rounds the radius normalization to eight decimal places; Rang
uses the higher-precision half-power normalization and comparisons allow that
small difference. Frequency extrapolation is refused by solver/audit validation;
direct JAX prediction outside the table returns NaN instead of clamping.

A tabulated Gaussian reproduces the original DFT at the reference frequencies.
A noiseless known-sky fit verifies the custom-predictor integration; it is not
the joint-gain recovery experiment.

```sh
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[pointing,beam,test]'
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/katbeam_observability.py
```

The experiment writes `outputs/katbeam-observability/results.json`.
[Archived output](results/katbeam-observability.json) includes all 14 cases,
model identity, parameter hashes and fixture provenance.
