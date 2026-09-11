# A Gaussian pointing–sky–gain ambiguity

## Result and correction to the earlier interpretation

**Ellipticity and rotation alone do not determine general absolute pointing
when the sky and antenna gains are free.** The earlier two-mode experiment
restricted the common trajectory to `R(t) a`. Freeing its temporal complement
reveals a different, smooth two-dimensional ambiguity. The earlier numerical
results are reproducible conditional results, not evidence that a general
pointing solver can measure those modes without additional information.

In a finite test, pointing trajectories differing by 34.95 arcsec RMS per
coordinate produce identical visibilities to a maximum difference of
9.31 × 10⁻¹⁶ Jy. This holds with nonzero differential pointing, non-unit
complex gains, a rotating ratio-1.1 Gaussian beam and the full w-term.
It is an algebraic symmetry, not a local optimizer failure.

## Exact transformation

Let the identical scalar voltage beams be

```math
E_{pjt\nu}=\exp[-\kappa_\nu(R_t s_j-d_{pt})^T Q(R_t s_j-d_{pt})],
\qquad Q=\operatorname{diag}(r,r^{-1}).
```

Here `s` is the two-vector of sky direction cosines, `d` is pointing in radians,
`R` rotates into the antenna frame, and the known positive-definite beam metric
`Q` is constant across antennas and channels. The Gaussian-core scale sets
`κ = 2 ln(2) / FWHM²`. For any constant two-vector `a` in radians, define

```math
h_t=Q^{-1}R_t a,\qquad d'_{pt}=d_{pt}+h_t,
```

```math
I'_{j\nu}=I_{j\nu}\exp[-4\kappa_\nu s_j^T a],
```

```math
g'_{pt\nu}=g_{pt\nu}\exp[2\kappa_\nu d_{pt}^TQh_t+
\kappa_\nu h_t^TQh_t].
```

Expanding the beam quadratic gives

```math
\frac{E'_{pjt\nu}}{E_{pjt\nu}}=
\exp[2\kappa_\nu s_j^T a-2\kappa_\nu d_{pt}^TQh_t-
\kappa_\nu h_t^TQh_t].
```

Multiplication by the transformed gain leaves `exp(2κ sᵀa)` per antenna.
The two antennas' factors cancel the transformed source flux exactly, source
by source and baseline by baseline. Fourier phases, including `w(n−1)`, are
unchanged. More baselines or lower noise cannot distinguish this family under
these assumptions. The transformation composes and is inverted by replacing
`a` with `−a` at the transformed model; regression tests verify that inverse.

The gain transformation is source independent but generally depends on time,
antenna and frequency. Free source/channel amplitudes are essential to the
stated symmetry. Known spectra, gain constraints, a beam metric that varies
between antennas/channels, or independent pointing information can change it.
Smoothness alone does not exclude this family: `Q⁻¹R(t)a` is smooth whenever
the rotation is smooth. A particular low-order spline space may exclude the
exact family, but then any resulting constraint can depend on that choice.

At zero pointing the compensating gain change is second order in `a`.
Consequently, the local pointing/sky Jacobian has this null space even when
gains are fixed. A local null direction alone does not imply an exact finite
ambiguity with fixed gains; the finite proof above allows gains to change.

## What breaks it in the toy?

We added a controlled non-Gaussian voltage shape

```math
E=\exp(-z-\epsilon z^2),\qquad
z=\kappa_\nu(R_t s-d)^TQ(R_t s-d).
```

`ε` is a dimensionless shape coefficient, **not a percentage beam error**.
At nonzero `ε`, the nominal Gaussian FWHM is only a core scale. This is an
analytic stress test, not a measured MeerKAT beam or a claim about its shape.

The local audit uses all 2,688 complex rows of the eight-antenna toy, 1 mJy
noise per real component, free source/channel fluxes, 1,536 real gain columns,
336 differential-pointing columns and the 46-dimensional temporal complement
of the two target shared trajectories. All nuisance parameters have no prior.
The target coordinates are coefficients of `R(t)a`; after marginalizing the
complement these describe the corresponding projection of a general common
trajectory. The audit is evaluated at zero pointing and unit gains.

| Beam | Shape coefficient | Measurable target modes | Local coordinate bounds |
| --- | ---: | ---: | ---: |
| Circular | 0, 0.01 or 0.1 | 0 | Undefined |
| Rotating ratio-1.1 ellipse | 0 | 0 | Undefined |
| Rotating ratio-1.1 ellipse | 0.01 | 2 | 561, 586 arcsec |
| Rotating ratio-1.1 ellipse | 0.1 | 2 | 57.7, 60.0 arcsec |

The distinction is between mathematical rank and usable information. Even
when non-Gaussian structure lifts the null space, its information can be weak.
These are conditional local Fisher bounds, not measured estimator errors or
reliable large-displacement nonlinear uncertainty intervals. Beam shape is
known in this audit; fitting it could weaken the information further.

Applying the *Gaussian* finite transformation to a non-Gaussian beam changes
the prediction, including for the circular case with nonzero differential
pointing. That merely disproves this particular transformation there; it does
not establish global uniqueness. It does not contradict the circular local
null space at zero differential pointing.

## External flux anchors: a concrete control

With the Gaussian ellipse and all temporal/gain nuisances still free, fixing
the central source's flux in every channel does **not** remove either pointing
mode. The central direction has `s=0`, so its flux is unchanged by the gauge.
Adding one exactly known off-axis source removes one mode; adding a second
non-collinear off-axis source removes the other:

| Exactly known channel fluxes | Measurable target modes |
| --- | ---: |
| None | 0 |
| Central source only | 0 |
| Central + first off-axis source | 1 |
| Central + two non-collinear off-axis sources | 2 |

The last case gives idealized local bounds 0.090 and 0.186 arcsec, but assumes
**perfect external flux knowledge**. These numbers are not a practical accuracy
claim. Finite and potentially correlated flux uncertainties must be propagated.
The central anchor fixes the ordinary absolute flux-scale ambiguity as well;
the requirement should not be shortened to “any two known sources suffice.”

## Research significance and prior art

RIME calibration ambiguities are established; see
[Smirnov (2011), Section 1.5](https://arxiv.org/abs/1101.1765).
Joint sky/pointing/beam inference is also established in
[Lochner et al. (2015), BIRO](https://academic.oup.com/mnras/article/450/2/1308/981750).
The explicit transformation here is our derivation within that framework,
not a verified first-in-literature result or a new universal impossibility
theorem. It does not contradict pointing self-calibration with a known sky.

The useful research question is now: **which measured departures from a
Gaussian beam constrain the otherwise ambiguous common pointing modes,
after realistic gain, trajectory, beam and sky uncertainties are allowed?**
The candidate workflow should expose those modes and distinguish information
from observations, external anchors and trajectory priors. It should not
interpret a convenient gauge choice or regularization optimum as a measurement.

The full finite symmetry test is implemented. The non-Gaussian and anchor
results are local audits, not a full nonlinear joint solver. No real telescope
data have been calibrated; the 27 MB measured-metrics download previously
failed with HTTP 502. The toy uses eight approximate MeerKAT positions, not
the complete array or a verified three-dimensional antenna survey.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/gaussian_gauge.py
```

Seed 61 specifies the same sky, smooth pointing and complex gains for the
finite controls. Six beam cases and four anchor cases are saved to
`outputs/gaussian-gauge/results.json`; the
[archived output](results/gaussian-gauge.json) includes all numerical results.
The exact transformation is `rangtoy.gauge.gaussian_gauge`; the local audit
options are `common_time_variation`, `beam_quartic` and `fixed_flux_sources`.
