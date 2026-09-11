# Pointing calibration: what is the data actually constraining?

Gijs Molenaar · 11 September 2026 · Discussion draft

## The question

Following the suggestion to investigate pointing-error solutions, Rang now
has a Rust simulation baseline and a Python/JAX component-list DFT predictor
with differentiable, smoothly time-varying per-antenna pointing fits.
The interesting result so far is an identifiability problem, not a faster or
more accurate calibration algorithm.

**For Gaussian voltage beams, a smooth pointing change can be exchanged for
a sky-brightness tilt and antenna-gain changes without changing any complex
visibility. Ellipticity and parallactic rotation do not by themselves remove
this ambiguity when the common trajectory is sufficiently flexible.**

![Different pointing trajectories with identical predicted data](figures/gauge-orbit.svg)

## What is established in this repository

- An explicit finite transformation, not just a small Jacobian singular value.
  For beam metric `Q` and sky-to-antenna rotation `R(t)`, its pointing part is
  `h(t) = Q⁻¹R(t)a`. The corresponding flux and gain factors cancel in the RIME.
- A numerical example with nonzero differential pointing and complex gains:
  approximately 35 arcsec RMS pointing change, with prediction differences at
  floating-point roundoff (~10⁻¹⁵ Jy). A 21-point sweep preserves the noisy-data
  likelihood as well. The full w phase remains in the predictor.
- Controls distinguish restricted and flexible trajectories. Earlier good
  recovery in a two-mode model was conditional on excluding other common
  temporal modes; the note explicitly corrects that interpretation.
- A central source with perfectly known flux does not remove either mode.
  Central plus one off-axis anchor removes one; adding a non-collinear second
  off-axis anchor removes the other. Three collinear anchors still leave one.
- A synthetic non-Gaussian elliptical beam lifts the local degeneracy, but
  weakly: one tested shape coefficient gives bounds near 60 arcsec, not the
  ~1 arcsec obtained with the restricted common trajectory.

## Why this may matter

A spline or smoothness prior can return a stable, precise-looking solution
while choosing between likelihood-equivalent—or nearly equivalent—physical
pointing trajectories. The distinction between data information and prior
selection matters if these solutions are interpreted as telescope pointing
measurements, or if their sky fluxes are treated as independently calibrated.

The candidate next step is to quantify the information in measured
non-Gaussian and chromatic beam structure, rather than just ask whether an
optimizer converges. The audit should include gain, beam and sky uncertainty.
Known antenna-to-antenna Gaussian differences alone are not necessarily a
cure: the same algebra permits antenna-dependent shifts when those pointing
trajectories are free. This extension is derived, not yet simulated here.

## What is not established

This is not a first-in-literature claim. RIME ambiguities, joint inference and
pointing self-calibration are established work. The result does not contradict
pointing recovery against a known sky. Nor is it a statement that real MeerKAT
pointing is unmeasurable: the toy uses eight approximate antenna positions,
four sources, four channels and scalar analytic beams. No measured beam cube
or real visibility data has been fitted. The non-Gaussian uncertainty numbers
are local Fisher bounds, not empirical solver performance.

## Questions for review

1. Is this explicit Gaussian pointing/sky/gain transformation already described
   in the pointing-calibration literature, beyond general RIME gauge freedom?
2. Which real MeerKAT constraint is most likely to dominate the weak modes:
   chromatic beam shape, polarization, flux anchors, gain stability or pointing
   telemetry?
3. Would a gauge-aware diagnostic separating likelihood and prior information
   be useful alongside existing pointing solvers?

[Full derivation, assumptions and primary references](gaussian-gauge.md) ·
[Archived numerical results](results/gaussian-gauge.json) ·
[Reproduction script](../examples/gaussian_gauge.py)

Reproduce in the documented environment with
`OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/gaussian_gauge.py`.
