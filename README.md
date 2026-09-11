# Rang

**Radio Astronomy Next Generation · MeerKAT calibration research**

Reproducible experiments in direction-dependent calibration: recovering faint emission while accounting for uncertain telescope response. Rust provides the numerical core; Python provides research interfaces and experiment orchestration.

[Pointing solutions](research/pointing-solutions.md) · [Results](research/first-results.md) · [Research direction](research/novelty.md) · [Experiment guide](examples/README.md)

## Current research lead: beam symmetry and pointing ambiguity

**Circularizing the beam can remove pointing information, not merely approximate its response.** We identified an exact shared-pointing/sky ambiguity for identical circular beams when source spectra are unconstrained. A rotating elliptical beam breaks that ambiguity in the controlled model.

The new spectrum-prior-free reference solver recovers the shared pointing mode at **1.86 arcsec RMS** while jointly fitting beam axial ratio. It declines the degenerate fixed-circle case. A wrong fixed ellipticity instead gives **52 arcsec** error despite small formal uncertainties.

[Derivation, 20-realization tests and limitations →](research/beam-rotation.md)

This is a concrete candidate contribution, **not a verified novelty claim**. Other pointing modes and direction-independent gains are held fixed in this experiment. General information-mode truncation did not improve pointing accuracy over a fairly tuned joint baseline; that [negative result is retained](research/spectral-pointing.md).

A separate local audit now allows independent antenna/time/channel gains and
zero-mean per-antenna pointing errors. Both shared modes survive with complete
eight-antenna baseline coverage, but disappear in the thinned fixture. This is
an identifiability check, not yet joint nonlinear recovery; the distinction
and numerical results are in the linked research note.

## Smooth pointing-error solutions

The current focus is recovering **smoothly time-varying, per-antenna pointing offsets** from a component-list sky model. A JAX reference path provides direct Fourier prediction, automatic derivatives and cubic-spline pointing fits alongside the Rust baseline.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[pointing,test]'
.venv/bin/python examples/pointing.py --output outputs/pointing
```

The example holds out interior time samples and saves fitted trajectories, truth and prediction metrics. The first solver assumes a fixed, correct sky and Gaussian beam. [Model, API and limitations →](research/pointing-solutions.md)

Initial control: **2.0 arcsec trajectory RMSE**, with held-out residuals reduced from **2.64 to 0.95 mJy/component** against 1 mJy injected noise. One seed, not a robustness claim.

<details>
<summary>View per-antenna pointing trajectories</summary>

![Simulated and recovered pointing offsets for eight antennas](research/figures/pointing-trajectories.svg)

</details>

## First result

### Sky uncertainty and pointing recovery

The [latest 20-seed experiment](research/sky-uncertainty.md) introduces 2% source-flux errors. Fixed-sky calibration gives **30.24 arcsec** pointing RMSE; joint flux/pointing inference gives **2.48 arcsec**, close to the **2.40 arcsec** correct-sky control. Overly tight flux priors leave **28.19 arcsec** error. The solver now supports per-component flux uncertainty; positions, spectra and beam shape remain fixed.

### Beam-width control

**Preserving an injected signal does not establish accurate source flux.** In a controlled 20-seed experiment, a 2% beam-width error produces a 71% flux overestimate despite a 99.75% injection response. Fitting beam width resolves the bias when the true error belongs to the fitted model family.

| Joint calibration model | Recovered flux | Injection response | Held-out residual RMS |
|:--|--:|--:|--:|
| Sky + pointing, fixed beam width | 34.27 mJy | 99.75% | 11.71 mJy |
| Sky + pointing + beam width | 20.12 mJy | 99.99% | 9.98 mJy |

True source flux: **20 mJy**. Noise: **10 mJy per visibility component**. Values are means across seeds 1–20, not uncertainty intervals. [Full results and controls →](research/first-results.md)

## Quick start

Requires Rust with edition-2024 support and Python 3.10+. The Rust core has no external crate dependencies.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -e '.[plot,test]'
.venv/bin/python examples/toy.py --seeds 7 11 19 --beam-error 0.02 --output outputs/demo
```

The script builds the optimized Rust executable and writes JSON results and PNG/PDF plots. Without plotting dependencies, run `python3 examples/toy.py --no-plot`. See the [experiment guide](examples/README.md) for the Python API and metric definitions.

## Research direction

The active physical lead is the [symmetry-aware pointing audit](research/beam-rotation.md). The broader decision-rule proposal below remains a separate, unproven extension.

We are investigating **calibration-mode selection with separate signal-distortion and model-error contamination budgets**. A differential response test can miss an additive flux bias; the proposed second budget addresses sensitivity to plausible model errors.

Joint sky/beam inference is established prior work. The current adaptive protection prototype has **not** improved on ordinary joint fitting. The dual-budget extension is a specified hypothesis, not an implemented or validated new algorithm. Its [prior-art comparison and acceptance criteria](research/novelty.md) define what must be demonstrated before claiming a contribution.

## Scientific status

The simulator uses a scalar RIME with the full non-coplanar phase, a Gaussian primary beam and eight approximate MeerKAT core antenna positions. It fits pointing offsets, a faint extended-source amplitude and optionally shared beam width.

This is a controlled research demonstrator, **not an operational MeerKAT calibrator**. Verified full-array geometry, measured beams, uncertain bright-source spectra and imaging-domain validation remain outstanding. [Assumptions and provenance →](data/README.md)

## Repository guide

| Location | Contents |
|:--|:--|
| `src/` | Visibility simulation, analytic derivatives and calibration solvers |
| `python/rangtoy/` | Python interface to the Rust executable |
| `examples/` | Reproducible campaigns and comparison plots |
| `tests/` | Python integration and scientific regression tests |
| `research/` | Results, mathematical proposals and primary-source references |
| `data/` | Geometry provenance and limitations |

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for local checks and research reporting standards. Retain negative results: a lower residual alone is not evidence of more faithful astronomy.
