# Rang research state

Updated 2026-09-11.

## Latest checkpoint

- Added shared log-beam-width fitting (17 physical parameters), analytic derivative test, and end-to-end width-mismatch recovery test. JSON keeps pointing arrays at 16 elements and reports fitted width separately.
- Twenty paired seeds 1–20: 2% width mismatch gives fixed-width joint flux 34.269 mJy versus width-fitting 20.124 mJy (truth 20). Corresponding injection response is 99.748% versus 99.990%; held-out RMS 11.706 versus 9.976 mJy. Matched-width flux 20.030 versus 20.081 mJy. Adaptive protection still gives no added benefit.
- Professor-facing discussion note: `research/first-results.md`; machine-readable campaign summary in `research/results/beam-width-campaign.json`. Raw runs and inspected plots in ignored `outputs/campaign-{matched,width}`.
- BIRO (Lochner et al. 2015, including Smirnov) directly precedes joint flux/pointing/beamwidth inference. Explicitly cited; no novelty claimed.
- Full release build, 9 Rust tests, 8 Python tests, Clippy, Ruff and both format checks passed after implementation.
- User authorized pushing and supplied `https://github.com/gijzelaerr/rang`. Both Git HTTPS and authenticated GitHub CLI report repository not found. Asked asynchronously for creation/visibility permission or access correction; do not create without that choice. Earlier approval classification issue was resolved enough to run the explicit supplied-URL checks; current problem is repository availability.

## Earlier checkpoints

- Request: review recent radio calibration/imaging and related fields, then identify a plausible new mathematical method, including non-coplanar geometry and 3D propagation.
- Repository initially contained no implementation or research documents; existing `.gitignore` was preserved.
- Completed a targeted primary-source literature investigation. Findings and a detailed candidate method are in `calibration-research-2026-09-11.md`.
- Current user scope: MeerKAT only; omit Starlink from the initial work. Active specification is `meerkat-scope.md`. The original report remains background and is marked superseded for experiment scope.
- User implementation conventions: Rust for computational kernels; Python for astronomer-facing interfaces and pipeline scripting. Commit solely under the user's identity without agent signatures or co-author trailers. Persisted in the root `AGENTS.md`.
- Initial working choice: MeerKAT L-band Stokes-I continuum, physical pointing and small beam corrections selected using sky-uncertainty-adjusted information and a local bound on science-signal distortion. L-band is an agent-selected starting point, not a user-imposed band restriction.
- Closest precedents: Albert tomography; BayesCal; resolve; Mouri Sardarabadi/Koopmans identifiability and suppression; Yatawatta/Leahy influence-based tuning; IDG-CAL 2026.
- Relevant latest result: Russeeawon et al. June 2026 clock/TEC initialization, including Smirnov, incorporated into QuartiCal.
- Status: runnable Rust/Python toy implemented. Rust has no external crate dependencies; Python invokes a JSON executable and plots with Matplotlib. No telescope downloads, commits or external publication.
- Additional close precedents: pointing selfcal, MeerKAT L-band beam measurements/holography, EIDOS and the December 2025 MeerKAT visibility-plane mosaicing framework.
- Toy: eight rounded published MeerKAT core positions, flat heights, Gaussian 13.5 m beam, 24 time samples, four L-band frequencies, full w phase, 16 constant antenna-frame pointing parameters, four known bright sources and one uncertain extended component. Geometry provenance/verification limitation is explicit in `data/README.md`.
- Initial matched-width result (seeds 7, 11, 19; true source 20 mJy): pointing-only recovers 12.05 mJy; joint/protected 19.79 mJy; known-pointing 19.89 mJy. All 16 modes survive the default budget; no evidence of added benefit from adaptive protection.
- Stress result: 2% beam-width mismatch biases joint/protected flux to 34.03 mJy while the finite injection response stays near one. Distinguish signal-response preservation from absolute accuracy.
- Strict budget (1e-6, seed 7) underfits and recovers 27.78 mJy instead of 20 mJy. A shifted source outside the fitted template shows 7.62% finite norm distortion despite a near-unit response projection. Both failure cases are documented.
- Controls: zero-error base fit exact; analytic derivatives, geometry, linear algebra and noiseless recovery tested. Coarse source quadrature initially biased even known-pointing recovery; resolved by adequate sampling and covered by a finer-grid convergence test.
- Main entry point: `python3 examples/toy.py --seeds 7 11 19 --output outputs/demo`. Artifacts in ignored `outputs/`; reproducible commands/results documented in `examples/README.md`.
- Verification: 8 Rust tests and 7 Python tests passed; release build, Clippy with warnings denied, Rust formatting and Python Ruff lint/format passed. PNG visually inspected. Final state is uncommitted per current workflow.
- Next research step: replace the idealized beam with measured beam information, include bright-source uncertainty and evaluate whether any adaptive selection benefit survives a matched joint-inference comparison. Acquire verified full-array coordinates before instrument-specific claims. Keep the w-term; introduce ionospheric complexity only as supported. No satellite dependency.
