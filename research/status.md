# Research status — 11 September 2026

Rang is ready for scientific discussion as a controlled experimental platform.
It is not yet a demonstrated novel algorithm, an operational MeerKAT calibrator,
or evidence of improved real-data imaging.

## Implemented and demonstrated

- Rust simulation/calibration baseline and structured projection kernels.
- JAX component-list DFT with the full non-coplanar phase, automatic derivatives,
  smooth per-antenna pointing and exact zero-mean relative-pointing constraints.
- Joint flux, spectral-index, complex-gain and selected beam-parameter inference;
  optional common pointing has an explicit prior, separate from relative pointing.
- Reproducible held-out prediction, mismatch and local uncertainty experiments.

The strongest current comparison is [antenna beam-width inference](antenna-beam-widths.md):
relative pointing RMSE falls from 6.33–9.19 arcsec with a shared-width-only model
to 0.79–1.10 arcsec when seven relative widths are fitted, versus 0.73–1.07 arcsec
with exact known antenna widths. Three seeds and two common-pointing settings
use eight approximate antennas and synthetic in-family width errors.

A separate 12-noise, fixed-truth pilot improves nominal 95% relative-pointing
coordinate coverage from 17.0% to 95.7%. Common-pointing coverage reaches only
85.4%. Coordinates are correlated; this is not a calibrated posterior guarantee.
The archive contains 168 fit configurations plus a diagnostic replay of the same
24 coverage configurations, not 24 extra independent trials.

The [Gaussian gauge](gaussian-gauge.md) is an exact finite ambiguity under its
stated freedoms. Its literature novelty is unconfirmed. Restricted-model common
pointing recovery does not contradict it or establish general identifiability.

## What has not worked or remains unproven

- Earlier mode selection did not beat a fairly tuned joint solver in pointing
  accuracy. Joint beam/sky/pointing fitting itself has established precedent.
- Polynomial gains compress an in-family truth, but an insufficient order fails.
- Noise-level residuals and near-unit injection response can coexist with flux bias.
- Missing-source and out-of-family pointing controls expose substantial errors.
- Structured projection reduces one audit's cost; it is not an end-to-end
  calibrator advantage, and the measured Rust path did not beat grouped NumPy.
- No matched state-of-the-art pipeline comparison or deconvolved-image improvement
  has been demonstrated. Full-array geometry and realistic polarized beams remain
  outside the main toy. Public scan ingestion works, but metadata conventions
  block a trustworthy physical pointing comparison.

## Verification and reproducibility

Handoff verification passed on macOS ARM64 with Rust 1.93.1, Python 3.14.7,
JAX 0.11.1, NumPy 2.5.3 and SciPy 1.18.1:

- Release build, 12 Rust tests and 107 Python tests (no skips).
- Clippy, Rust formatting, Ruff lint/format, Python byte-compilation and whitespace checks.
- Both README examples: smooth pointing converged at 2.005 arcsec RMSE with
  held-out RMS 2.635 → 0.950 mJy/component; the seed-7 Rust toy completed.
- Seed-7 joint antenna-width campaign: all eight cases converged. The two
  joint-flux cases reproduce approximately 1.10 arcsec relative pointing RMSE.
  Convergence of the deliberately inadequate controls does not imply accuracy.
- All 20 compact JSON result archives parse; local file links in the handoff
  documents resolve.

The smoke test caught and fixed a JSON export regression: complex gains are now
stored as separate real/imaginary arrays. A new end-to-end example test covers
this output and the generated trajectory SVG. Runs are in ignored
`outputs/handoff-20260911-pointing` and `outputs/handoff-20260911-rust`;
the joint campaign uses the example's parameter-derived output directory.
Full archived research campaigns were retained, not all rerun for this checkpoint.
The Measurement Set container inspection was verified at the earlier checkpoint,
not rerun here; its unresolved physical metadata limitations still apply.

Commands and assumptions: [contributor checks](../CONTRIBUTING.md),
[experiment guide](../examples/README.md), [antenna-width reproduction](antenna-beam-widths.md).
Generated run artifacts remain in ignored `outputs/`; compact result archives
and their scientific limitations remain under `research/results/` and reports.

## Next decision and discussion questions

The [next-stage plan](development-roadmap.md) prioritizes one falsifiable
selection algorithm and matched quality/cost measurements. Separate budgets for
signal distortion and model-error contamination are still a proposed mechanism.
No time estimate guarantees a novel result.

Useful questions for scientific review:

1. Is selecting relative beam/pointing freedom a useful target beyond existing
   joint calibration and solution-interval selection?
2. Which measured MeerKAT antenna beam errors and science-field dataset provide
   a discriminating test, rather than another matched-model demonstration?
3. Which established solver and image/flux tolerances make the fairest comparison?

This repository is a discussion package; no message or external scientific
submission is sent by this checkpoint.
