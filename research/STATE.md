# Rang research state

Updated 2026-09-11.

## Latest checkpoint

- Gain-nuisance extension of the local symmetry audit: optional constant or per-time complex antenna gains, shared across frequency, projected jointly with free source/channel amplitudes. Both shared pointing modes survive ratio-1.1 rotating-beam audit; local coordinate bounds change from [2.068, 1.916] to [2.549, 2.405] arcsec with 384 gain columns. Not a nonlinear gain/pointing recovery result. Regression covers nested information loss and preservation of circular-beam ambiguity. Reproduction: `examples/observability.py` (seven cases). Public SARAO L-band metrics download returned HTTP 502; measured-beam validation remains pending.

- New concrete physical lead: exact sky-locked common-pointing/sky degeneracy for identical circular beams and unconstrained source/channel amplitudes. Measured pilot error energy is 99.9% in this two-dimensional shared trajectory. Implemented `rangtoy.observability`, local sky-prior-free audit, and nonlinear variable-projection solver; see `research/beam-rotation.md`. Generic mode truncation remains a negative accuracy result.
- Controlled 20-noise-realization shared-mode test: circular-beam fits refused; correct rotating ratio-1.1 beam recovers 1.787 arcsec RMS; wrong fixed ratio1.08 gives 51.929 arcsec; jointly fitting ratio from circular initialization restores 1.859 arcsec. Sixteen source/channel fluxes free, no spectral prior. Other pointing and DI gains fixed. Not a confirmed first-in-literature contribution or general telescope result.
- Exact finite gauge, local rank controls, nonlinear recovery, and shape recovery tests pass. Reports `research/beam-rotation.md`, `research/spectral-pointing.md`; combined summaries under `research/results/symmetry-and-spectral-campaigns.json`. JAX-only new paths preserve Rust baseline.

- Work toward a defensible contribution: optional spectral-index priors and prior-whitened, sky-marginalized pointing-mode truncation are implemented. Expanded `examples/sky_uncertainty.py`; new `examples/mode_selection.py` gives both full/selected solvers identical five-value smoothing grids with separate train/validation/test times.
- Spectral mismatch (alpha ±.1, flux ±2%, 20 seeds, 1 mJy noise): flux-only joint fit gives 68.51 arcsec pointing error despite improved residuals; jointly fitting spectra gives 4.132 arcsec. At 10 mJy, fixed-smoothing selection keeps 56/96 modes and improves 19.27→18.13 arcsec. Equal-budget validation tuning removes pointing advantage: full 18.5009 vs selected 18.4975 arcsec, selected mean 44.8/96 modes. Do not claim a new accuracy improvement.
- Current runs: `outputs/spectral-20`, `outputs/spectral-noisy-20`, `outputs/mode-cv-20`; three-seed physical decomposition in `outputs/common-mode-pilot` shows flux-only spectral-error failure is mostly array-common (68.79 arcsec common versus 4.79 differential). Investigating sky-locked/parallactic-frame ambiguity, not yet a validated novelty claim.
- Additional close precedents reviewed: Sob et al. 2021 solution interval selection, Tasse 2014 nonlinear Kalman calibration, Fisher/score compression and low-rank methods. Absence of exact search matches does not establish novelty.

- Added optional per-component Gaussian flux priors to smooth-pointing inference; zero sigma fixes a component, None preserves fixed-sky behavior. Returns fitted fluxes; prior-normalized additive corrections are differentiated jointly with spline coefficients.
- Three 20-seed campaigns (180 converged fits, including duplicate paired controls): 2% source-flux errors cause 30.237 arcsec fixed-sky pointing RMSE; 5% joint flux priors recover 2.475 arcsec vs 2.402 correct-sky control. Held-out RMS 3.682 → 1.067 mJy/component. Overconfident 0.01% priors leave 28.187 arcsec error. Matched-sky joint fit costs a small increase to 2.475 arcsec.
- Results and reproduction: `research/sky-uncertainty.md`, `research/results/sky-uncertainty.json`, `examples/sky_uncertainty.py`. Fixed source layout/sign pattern, exact central flux anchor and correct spectra/beam remain limitations. No new-algorithm claim.

- Converted all 18 display equations across four research documents to GitHub-supported fenced math blocks; equations unchanged. Documented the formatting convention in CONTRIBUTING.md.

- User/professor focus: pointing-error solutions; explicitly requested component-list conversion, DFT prediction, JAX automatic differentiation and smoothly time-varying per-antenna offsets. Implemented optional `rangtoy.pointing` with natural cubic spline trajectories, integrated-curvature regularization and SciPy least squares using JAX Jacobians.
- Public PyPI explicitly required by user; isolated `.venv` installed JAX/JAXlib 0.11.1, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1 and Ruff 0.16.7. Do not use the Spotify registry. Rules saved in AGENTS.md.
- Rust `--pointing-reference` exports deterministic prediction/analytic-derivative fixtures; JAX comparisons, finite differences, spline controls and noiseless recovery passed (7 new tests).
- Initial smooth-pointing example (seed 7, noise 1 mJy/component, six knots, smoothness .01): 2.005 arcsec trajectory RMSE, held-out residual RMS 2.635 → .950 mJy/component. Sinusoidal truth differs from spline fit; sky/beam remain exactly known. Full details in `research/pointing-solutions.md`.
- Saved smooth-pointing numerical summary and dependency-free SVG trajectory figure under `research/results` and `research/figures`. Full local check passed with 9 Rust and 15 Python tests, including JAX, plus Clippy and both format/lint checks.

- Public repository created and verified at `https://github.com/gijzelaerr/rang`; initial public commit `2587480`. Both author and committer use `gijsmolenaar@gmail.com`; this identity is configured locally and required by AGENTS.md.
- Presentation revised: result-led README, contributor guide, clear implementation status and linked reproducibility material. Current contribution assessment is `research/novelty.md`.
- Targeted prior-art review adds robust ellipsoidal beamforming and partially constrained CMB component separation as strong objections to broad novelty. The sharper candidate uses separate response-distortion and model-error sensitivity budgets for physical calibration selection; specified, not implemented or demonstrated.
- Next discriminating work: complete-estimator derivatives and uncertain bright-source/beam perturbations with matched joint-inference baselines. No novelty claim is warranted by current results.

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
- Current user scope: MeerKAT only. Active specification is `meerkat-scope.md`. The original report remains background and is marked superseded for experiment scope.
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
- Next research step: replace the idealized beam with measured beam information, include bright-source uncertainty and evaluate whether any adaptive selection benefit survives a matched joint-inference comparison. Acquire verified full-array coordinates before instrument-specific claims. Keep the w-term; introduce ionospheric complexity only as supported.
