# Beam symmetry determines which pointing modes can be measured

## Research lead

**A circular beam can make a coherent pointing trajectory indistinguishable from a different sky. Known beam asymmetry and rotation can break that ambiguity.** The candidate contribution is a symmetry-aware pointing workflow: separate these shared modes, audit their information without spectral priors, and decline unsupported estimates rather than interpreting regularization as a measurement.

This is a concrete mathematical and numerical result in Rang, not a verified first-in-literature claim. The estimator uses established DFT prediction, nuisance projection and variable projection. The proposed contribution is the physical observability criterion and its use in pointing calibration, not those mathematical primitives.

## How we found the mode

### Additional gain-nuisance check

The local audit now optionally projects out complex direction-independent
antenna gains together with the independent source/channel fluxes. Gains are
linearized at unity, with log-amplitude derivative `(1[p=a]+1[q=a]) V` and
phase derivative `i (1[p=a]-1[q=a]) V`. Redundant gauge columns are retained;
least-squares projection uses their column space, not an invertible normal matrix.

For the rotating ratio-1.1 toy at 1 mJy noise, the two common pointing modes
survive both tested gain models:

| Gain nuisance model | Gain columns | Local bounds on the two sky coordinates |
| --- | ---: | ---: |
| Fixed gains | 0 | 2.068, 1.916 arcsec |
| One complex gain per antenna for the whole track | 16 | 2.075, 1.922 arcsec |
| One complex gain per antenna per time | 384 | 2.549, 2.405 arcsec |

Both free-gain models share gains across frequency. These are conditional local
Fisher bounds, **not nonlinear joint gain/pointing recovery measurements**.
Other pointing trajectories remain fixed. Frequency-independent gains are a
substantial constraint; frequency-dependent calibration still needs testing.
The fixture retains every fifth baseline/time/channel row, so its coverage
must be revisited before interpreting a more flexible gain audit.
Reproduce all seven audit cases with
`.venv/bin/python examples/observability.py`; output is
`outputs/observability/results.json`.

An attempt to retrieve SARAO's public 27 MB L-band beam-metrics archive on
2026-09-11 returned HTTP 502. No measured metrics were incorporated. The axis
ratios above remain controlled assumptions, not fitted MeerKAT measurements.

In the three-seed spectral-error pilot, flux-only joint fitting produced about 69 arcsec trajectory error. Almost all of it was shared by the array: 68.8 arcsec common versus 4.8 arcsec differential error. Of the common error energy, 99.9% lay in the two-dimensional trajectory family

```math
\delta_p(t)=R_t a\qquad\text{for every antenna }p,
```

where `a` is a fixed two-component sky-frame displacement and `R_t` rotates sky coordinates into the antenna beam frame. This is a diagnostic computed using simulation truth, not an automatic test that real motion is spurious. A real common motion can occupy the same family.

## Exact ambiguity for identical circular beams

Assume identical scalar circular voltage beams, fixed source positions, known direction-independent gains, and only the shared trajectory above. Circular symmetry gives

```math
E_\nu(R_t s_j-R_t a)=E_\nu(s_j-a).
```

Thus beam attenuation for each source is independent of time and baseline. For any alternative displacement, changing each source's frequency-dependent flux by the corresponding power-beam ratio leaves every visibility unchanged, provided the relevant beam responses are nonzero. More baselines, longer tracking or the full w-term do not remove this particular ambiguity when source/channel fluxes are free.

For the implemented Gaussian voltage beam, with radians used in this equation,

```math
E_\nu(s)=\exp[-\kappa_\nu\|s\|^2],\qquad
\kappa_\nu=\frac{2\ln 2}{\mathrm{FWHM}_\nu^2},
```

the transformation from zero offset to `a` is explicitly

```math
I'_j(\nu)=I_j(\nu)
\exp[-4\kappa_\nu s_j^T a+2\kappa_\nu\|a\|^2].
```

Substitution into the component DFT gives exactly the original visibilities. A regression test verifies the finite transformation to numerical precision. This is not merely a first-order coincidence. A restrictive source-spectrum family can remove the degeneracy mathematically, but the inferred pointing then depends on that spectral assumption.

The general argument requires **identical circular beams and the specified common trajectory**. Antenna-dependent beams, additional direction-dependent effects, known source fluxes, pointing telemetry or other observations can change the identifiability question. It is not a claim that all pointing self-calibration is impossible.

## What breaks the ambiguity

For an elliptical Gaussian beam with antenna-frame quadratic form `Q`, rotation produces `R_tᵀ Q R_t` in sky coordinates. If `Q` is not proportional to the identity and the orientation varies, the beam-ratio perturbation is generally time-dependent and cannot be absorbed by time-independent source/channel fluxes.

The implementation first computes the two common-pointing derivatives `B` and the independent source/channel amplitude derivatives `A`. With noise whitening, it measures the singular values of

```math
B_\perp=(I-AA^+)B.
```

There is no sky prior in this projection. A circular beam gives zero residual derivative to approximately machine precision. An elliptical beam with a fixed orientation also gives zero in this experiment. A rotating ellipse has two nonzero common-pointing directions for this source layout and track.

| Assumed axial ratio | Beam rotation | Observable shared modes | Local per-axis CRLB at 1 mJy noise |
|:--|:--|--:|:--|
| 1.00 | Yes | 0 | Not identifiable |
| 1.02 | Yes | 2 | 9.88, 9.25 arcsec |
| 1.10 | Yes | 2 | 2.07, 1.92 arcsec |
| 1.20 | Yes | 2 | 1.09, 1.00 arcsec |
| 1.10 | No | 0 | Not identifiable |

These are local bounds at zero pointing, conditional on the stated model and fixed other parameters. The axial ratios are controlled assumptions, not measurements of MeerKAT's actual beam. Fixed-orientation controls preserve the visibility geometry but disable the beam rotation.

## Nonlinear recovery without source-spectrum priors

We then fit the two common-pointing coordinates, profiling out sixteen independent source/channel fluxes at every objective evaluation. There are four sources and four frequencies; the source/channel amplitudes deliberately depart from a power law. Neither spectral smoothing nor source-flux priors are used. All differential pointing and direction-independent gains are held fixed.

The true common sky-frame shift is `[0.3, -0.2]` arcmin. Twenty paired noise realizations at 1 mJy/component give:

| True beam ratio | Fitting assumption | Pointing outcome |
|:--|:--|:--|
| 1.00 | Fixed correct circle | All 20 estimates refused: exact ambiguity |
| 1.10 | Fixed correct ellipse | 1.787 arcsec RMSE |
| 1.10 | Fixed incorrect ratio 1.08 | 51.929 arcsec RMSE |
| 1.10 | Jointly fit ratio, initialized at 1.00 | 1.859 arcsec RMSE |

RMSE is over both coordinates and all twenty noise realizations. The truth displacement, component geometry and channel amplitudes are fixed across this experiment; it is not twenty independent physical fields. Joint shape/pointing fitting has three nonlinear parameters; the sixteen channel amplitudes remain profiled linear parameters.

The incorrect fixed ellipse reports local errors of only about 2.2–2.4 arcsec while being wrong by roughly 52 arcsec. Local rank and formal precision are **not model-accuracy certificates**. Jointly fitting axial ratio fixes the deliberately in-family error, but does not establish robustness to arbitrary beam-model mismatch.

The shared-mode solver can initialize shape fitting from a circle because it does not treat the initial shape audit as a veto when shape is free. It checks the profiled Jacobian rank at the fitted solution. That numerical check remains local and conditional; it does not certify uniqueness, accurate uncertainty coverage or sufficient information at a requested accuracy.

## Why it matters for MeerKAT

MeerKAT beam measurements establish pointing and beam-shape uncertainty as practical concerns, including frequency-dependent structure. The [SARAO beam documentation](https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/1481572357/The%2BMeerKAT%2Bprimary%2Bbeam) describes a vertically elongated beam. The inference here is that **circularizing a beam model may remove useful pointing information, not just approximate its response**. This needs testing with measured Jones beams and real flags, gains and antenna geometry. [L-band beam measurements](https://arxiv.org/abs/2202.02101).

## Prior art and novelty boundary

[Pointing self-calibration](https://arxiv.org/abs/1709.08681), [BIRO's joint instrumental/sky inference](https://arxiv.org/abs/1501.05304), and [RIME calibration ambiguities](https://arxiv.org/abs/1101.1765) are direct precedents. Known beam rotation and asymmetry are already used in direction-dependent modelling. Nothing here establishes that the symmetry argument or its practical consequences are absent from that literature.

The narrower candidate for a contribution is a **spectral-prior-free audit and solver for shared pointing modes**, accompanied by a quantitative demonstration of when beam diversity supports them and when the solver must decline to report them. The audit must be extended to simultaneous differential pointing, direction-independent gains, uncertainty in measured beams and real-data sampling before claiming a generally useful new MeerKAT method. The uncertainty budgets in the earlier proposal are not implemented by this rank test.

## Reproduce

```sh
.venv/bin/python examples/observability.py
.venv/bin/python examples/sky_locked.py
```

Raw outputs are in `outputs/observability` and `outputs/sky-locked`. The [saved campaign summaries](results/symmetry-and-spectral-campaigns.json) include these results and the unsuccessful general mode-selection experiment. Both scripts use the same approximate eight-antenna Rust fixture; no observed MeerKAT data is used.
