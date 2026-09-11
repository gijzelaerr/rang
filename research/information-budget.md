# Where does the pointing information come from?

## Result

The local audit now separates unconstrained visibility information from the
information available after imposing uncertain external source-flux constraints.
For the Gaussian toy, those external constraints produce finite pointing
uncertainties even though the likelihood alone still has the exact ambiguity.
This distinction is implemented explicitly, not inferred from convergence.

The experiment retains the eight-antenna full-baseline fixture, 1 mJy noise
per real visibility component, ratio-1.1 rotating beam, arbitrary complex gains
per antenna/time/channel, zero-mean differential pointing per time, and the
temporal complement of the two shared target trajectories. There is **no
pointing prior**. All 16 source/channel fluxes are uncertain, including the
central source. Audits are local to zero pointing and unit gains, with the
true beam shape treated as known.

## Finite, correlated flux constraints

The external prior is Gaussian in log flux. For small errors, a log-flux
standard deviation of 0.01 corresponds approximately to a 1% fractional
uncertainty. Covariance entries are ordered channel first, then source.
These are controlled assumptions, not measured catalogue uncertainties.

| External log-flux constraint | Gaussian: local coordinate σ | Non-Gaussian: local coordinate σ | Non-Gaussian data-only information fraction |
| --- | ---: | ---: | ---: |
| Independent 1% errors | 6.269, 6.731 arcsec | 5.907, 6.297 arcsec | 0.75–1.34% |
| Independent 5% errors | 31.341, 33.642 arcsec | 26.233, 27.829 arcsec | 15.96–25.34% |
| Independent 1% + common 5% flux-scale error | 6.269, 6.731 arcsec | 5.907, 6.297 arcsec | 0.75–1.34% |
| 1% marginal error, channel correlation 0.95 within each source | 4.066, 4.386 arcsec | 3.726, 3.978 arcsec | 0.30–0.54% |

The Gaussian data-only fractions are zero to numerical precision in every
case (below 2 × 10⁻²³). The non-Gaussian beam has the synthetic quartic
log-voltage coefficient 0.1 from [the ambiguity note](gaussian-gauge.md).
It is not a 10% beam error and is not a measured MeerKAT beam.

The fractions are two generalized information eigenvalues, **not percentages
of posterior variance, probabilities, or separate x/y error contributions**.
They quantify the information available without flux constraints relative to
the constrained information in the least/most supported combinations.

Two useful controls:

- Adding uncertainty common to *all* fluxes changes nothing here: unconstrained
  antenna gains already absorb the overall flux scale. Relative flux constraints
  determine the pointing information in this model.
- Positive channel correlation can tighten pointing constraints. It permits a
  shared source-amplitude error but restricts relative spectral errors; the
  Gaussian gauge's frequency-dependent brightness tilt is then penalized more
  strongly. This is additional spectral knowledge, not extra independent data.

## Definition of the information split

Let `B` be the two target pointing columns and `A` all nuisance columns in the
noise-whitened real visibility Jacobian. Let `L` map nuisance perturbations to
whitened external-prior residuals; its gain and pointing columns are zero.
The log-flux block is obtained from the Cholesky factor of its covariance.
Compute the nuisance response to a target perturbation using

```math
C_* = \mathop{\mathrm{argmin}}_C
\left\|B-AC\right\|_F^2+\left\|LC\right\|_F^2.
```

The implementation solves the stacked least-squares system and tolerates
redundant gain/flux gauge columns; it does not require an invertible nuisance
normal matrix. Define

```math
F_{\mathrm{data|external}}=(B-AC_*)^T(B-AC_*),\qquad
F_{\mathrm{external}}=(LC_*)^TLC_*,
```

```math
F_{\mathrm{total}}=F_{\mathrm{data|external}}+F_{\mathrm{external}}.
```

Separately, remove all nuisance directions **without** the external prior:

```math
B_0=(I-AA^+)B,\qquad F_0=B_0^TB_0.
```

`F_data|external` is not standalone likelihood information: its nuisance
response was selected using the external prior. A nonzero value of that term
does not contradict a zero `F_0`. The independently available fraction is
reported from

```math
F_0 v_i = f_i F_{\mathrm{total}}v_i.
```

The output retains all four information matrices, the fractions, constrained
rank and local covariance. Covariance and fractions are left undefined if
the constrained target remains rank deficient. Information has units of
inverse arcminutes squared; reported standard deviations are in arcseconds.

This is established Gaussian nuisance elimination applied to the explicit
pointing gauge. It is a diagnostic contribution in Rang, **not a new statistical
estimator or a verified first-in-literature method**. It does not yet propagate
beam uncertainty, nonlinear posterior structure, incorrect prior means, or
correlations between external flux estimates and the same visibility data.
Using a sky prior derived from these data as if independent would double-count
information. The external prior must come from genuinely independent evidence
or a joint model that includes its correlations with the data.

## Reproduction and next discriminating test

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/information_budget.py
```

The script writes all eight results and full covariance matrices to
`outputs/information-budget/results.json`. The
[compact archive](results/information-budget.json) retains the diagnostic
matrices and explicit covariance definitions. Tests cover an analytic scalar
degeneracy, additive information accounting, nuisance-coordinate invariance,
the unconstrained limit, covariance validation, common-scale invariance and
the full toy Gaussian/non-Gaussian distinction.

The next meaningful test is a measured beam model with this same nuisance
budget, followed by beam-model uncertainty and nonlinear recovery. This work
does not yet establish how much pointing information real MeerKAT data contain.
