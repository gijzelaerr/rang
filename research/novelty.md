# Contribution assessment and next experiment

Last reviewed: 11 September 2026.

## Assessment

The active application is now [smooth per-antenna pointing solutions](pointing-solutions.md), following the requested component-list / DFT / JAX workflow. The decision-rule hypothesis below remains unimplemented; the new spline solver is an established-method baseline for testing it.

**Novelty is not established.** The implemented beam/pointing/sky fit reproduces an established class of inference methods. The adaptive signal-protection prototype adds no measured benefit in the present experiment. Neither is presented as a new superior calibrator.

The research candidate is narrower: **select physical calibration modes and regularization using separate budgets for science response distortion and worst-case contamination from a specified beam/sky uncertainty set**. Report infeasibility when an observation cannot support both budgets at acceptable noise variance. This is a proposed decision rule, not yet an implemented extension.

## Closest prior work

| Prior work | Already established | Difference Rang would need to demonstrate |
|:--|:--|:--|
| [BIRO, Lochner et al. (2015)](https://arxiv.org/abs/1501.05304) | Joint source flux, pointing, beamwidth and noise inference | A decision criterion beyond adding beam parameters |
| [BayesCal, Sims et al. (2022)](https://arxiv.org/abs/2206.13554) | Calibration accounting for uncertain and missing sky emission | Useful uncertainty-set budgeting beyond marginalization |
| [Yatawatta (2019)](https://academic.oup.com/mnras/article/486/4/5646/5484901) | Calibration influence and input/output Jacobian spectra as distortion diagnostics | Joint response/contamination control in physical mode selection |
| [Mouri Sardarabadi & Koopmans (2019)](https://arxiv.org/abs/1902.02482) | Identifiability and estimability of direction-dependent calibration | Value beyond known null-space and information analysis |
| [Lorenz & Boyd (2005)](https://stanford.edu/~boyd/papers/rmvb.html) | Robust beamforming with ellipsoidal response uncertainty | A substantive extension to data-dependent nonlinear calibration |
| [Abylkairov et al. (2021)](https://arxiv.org/abs/2012.04032) | Signal-preserving CMB estimation with bounded foreground contamination | More than renaming constrained component separation |

This is a targeted primary-source review, not an exhaustive novelty clearance. Robust beamforming and constrained component separation are particularly strong objections to a broad novelty claim.

## Candidate mathematical criterion

Use real-stacked, noise-whitened visibilities. Around a fitted reference, write

\[
\delta y = A\delta x + B\delta\theta + Uz + n,
\qquad \|z\|_2\leq 1.
\]

`A` maps science perturbations, `B` maps physical calibration perturbations, and `U` contains scaled plausible unmodelled beam/sky perturbations. Its scales must come from independent knowledge or explicit stress assumptions, not the desired answer. The joint unit ball describes an ellipsoid, not independent unit bounds on every nuisance parameter.

For a chosen calibration subspace `E` and regularization `λ`, let `K(E,λ)` be the local derivative of the **complete reported science estimator** with respect to visibilities. Include calibration refitting, final flux estimation and every data-dependent step being claimed. Let `D` normalize science outputs using independently chosen tolerances. Consider

\[
\|D(KA-I)\|_2\leq\epsilon_s,
\qquad
\|DKU\|_2\leq\epsilon_b.
\]

The first limits science response distortion; the second limits model-error contamination. Science input units must be chosen explicitly so a unit perturbation ball has a specified meaning. For scalar reported flux `kᵀy`, the second expression before output normalization is `||Uᵀk||₂`: the exact worst-case linear contamination over the ellipsoid, by Cauchy–Schwarz. This is standard mathematics, not a new theorem.

A possible objective is minimum propagated noise variance `tr(D K Kᵀ Dᵀ)` subject to both budgets, a held-out prediction criterion and a declared cost limit. Searching physical subspaces is generally discrete, and the nonlinear estimator changes the derivative. This is not automatically a convex optimization problem.

`B δθ` describes fitted physical degrees of freedom; `Uz` describes residual errors not adequately covered by that family. Overlap must be represented as degeneracy, not independent evidence. Fix flux-scale gauge constraints before interpreting `KA−I`.

These derivative bounds **do not bound existing offset bias**, nonlinear remainder, noise realizations or errors absent from `U`. An absolute guarantee additionally needs a bound on reference error and nonlinear remainder. Call these local sensitivity bounds, not accuracy certificates.

## Connection to measured results

The [20-seed experiment](first-results.md) shows high differential response alongside a large absolute flux error. It motivates separate model-error sensitivity analysis; it does not prove the proposed rule works. Current code measures bright-subtracted residual response, not `K` for the complete science estimator. The latter remains to be implemented and checked against complete-refit finite differences.

## Required experiment

1. Introduce uncertain bright-source fluxes/spectra and beam-shape errors outside the fitted width family. Specify uncertainty scales independently. Retain matched controls and fixed seeds.
2. Implement complete-estimator derivatives, validate finite perturbations, and compare the worst-case linear direction with sampled nuisance directions.
3. Compare joint physical inference with tuned priors, signal-only selection, bias-only selection and the dual-budget candidate. Match observations, parameter families and tuning effort.
4. Choose budgets before held-out evaluation. Report flux bias, variance, response, norm distortion, held-out residuals, runtime and infeasibility. Separate errors inside and outside the uncertainty set.
5. Use measured beams and verified geometry before MeerKAT performance claims. Image-domain claims require an explicit imager and corresponding validation.

## Acceptance and rejection

A contribution requires reproducible improvement in the bias–variance–distortion tradeoff or useful, calibrated detection of infeasibility at comparable cost, plus a defensible distinction from prior work. Improvement only over a deliberately incomplete sky model is insufficient.

Reject or narrow the claim if matched joint inference reaches the same operating point, if the criterion is equivalent to an existing method without computational or practical benefit, or if uncertainty-set tuning merely encodes simulated truth. Failed candidates remain useful results; they should not be relabelled as novel.
