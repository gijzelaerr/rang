# Rang

**Radio Astronomy Next Generation · MeerKAT calibration research**

Reproducible experiments in direction-dependent calibration: recovering faint emission while accounting for uncertain telescope response. Rust provides the numerical core; Python provides research interfaces and experiment orchestration.

[Pointing solutions](research/pointing-solutions.md) · [Results](research/first-results.md) · [Research direction](research/novelty.md) · [Experiment guide](examples/README.md)

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
