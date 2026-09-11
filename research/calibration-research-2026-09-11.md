# Calibration research and a candidate method

Research date: 11 September 2026.

Scope update: the user subsequently restricted the project to MeerKAT. The [active MeerKAT proposal](meerkat-scope.md) supersedes the atmospheric emphasis and generic low-frequency experiment below. This document remains the background literature and mathematical derivation; it does not define the current experiment.

## Recommendation and evidence status

Investigate **adaptive physical calibration with an explicit bound on science-signal distortion**. Represent the ionosphere with a small number of moving layers on curved Earth geometry; couple those layers to a full wide-field measurement operator; select combinations of calibration parameters according to their distinguishability from uncertain sky emission; and constrain their measured influence on designated science signals.

The scientific question is whether this selection improves the tradeoff between calibration residuals, recovered sky fidelity, and computation compared with well-tuned existing methods. It is a candidate algorithmic contribution, not an established novel method or a demonstrated improvement.

This is a targeted search of primary papers and author-maintained software, including 2025–2026 work and older close precedents. It is not an exhaustive bibliographic or priority search. Some sources were assessed through abstracts and relevant sections; the closest accessible methods were examined in more detail. No telescope dataset has been processed, no numerical benchmark has been run, and no performance figures below describe Rang results.

## 1. What the geometry actually requires

Two issues should be kept separate:

| Issue | Appropriate model | What is inferred? |
| --- | --- | --- |
| Non-coplanar baselines and a wide field on the celestial sphere | Three-component baseline vector and spherical sky, or an equivalent operator retaining the w-term | Normally the geometry is known; its effect must be computed accurately |
| A spatially extended ionosphere above a curved Earth | Rays through curved shells or a three-dimensional electron-density field | Only combinations supported by ray coverage, bandwidth, time coverage, and priors |

For a far-field sky, a convenient full-polarization model is

```math
V_{pq}(t,\nu)=\int_{\Omega}
J_p(\hat s,t,\nu)\,\mathcal B(\hat s,\nu)\,
J_q^H(\hat s,t,\nu)
e^{-2\pi i(\nu/c)b_{pq}(t)\cdot(\hat s-\hat s_0)}\,d\Omega
+N_{pq}.
```

Here J excludes the geometric phase explicitly written in the exponential, and the polarization basis transformations belong in the full operator. In tangent-plane coordinates the phase contains

```math
ul+vm+w(\sqrt{1-l^2-m^2}-1),\qquad
d\Omega=dl\,dm/\sqrt{1-l^2-m^2}.
```

Earth curvature contributes to array geometry, but the w-term is not exclusively an Earth-curvature effect. The astronomical unknown is generally brightness on a two-dimensional sphere, not a three-dimensional emitting volume. Frequency cubes and near-field imaging introduce different notions of depth. Smirnov's full-sky formalism supplies the starting point. Historically, the RIME originates in Hamaker, Bregman and Sault's work; Smirnov substantially developed and clarified it. [Smirnov 2011](https://arxiv.org/abs/1101.1764)

For near-field interference, trajectory-based joint subtraction and calibration already has a direct radio precedent: TABASCAL. It should be treated as a separate extension. [Finlay et al. 2023](https://academic.oup.com/mnras/article/524/3/3231/7219326)

## 2. Established methods and recent developments

The distinction between solving instrumental effects and applying them during imaging matters. DDFacet provides wide-field, wideband deconvolution accounting for direction-dependent effects; killMS supplies direction-dependent calibration solutions. Calling DDFacet alone a calibration algorithm obscures that division. [DDFacet paper](https://arxiv.org/abs/1712.02078), [killMS repository](https://github.com/saopicc/killMS)

| Work | Relevant contribution | Consequence for Rang |
| --- | --- | --- |
| Tasse / Smirnov & Tasse, 2014–2015 | Complex calibration derivatives and structured numerical solvers | Differentiating the RIME is established; automatic differentiation alone is not a novelty claim. [Wirtinger paper](https://arxiv.org/abs/1410.8706) |
| Tasse, 2014 | Nonlinear Kalman filtering with physical instrumental effects and dynamics | Temporal filtering of clock/ionosphere parameters is already radio work. [Paper](https://arxiv.org/abs/1403.6308) |
| Yatawatta, 2013 and 2015 | Riemannian calibration and distributed consensus calibration | Manifold optimization and frequency coupling are established. [Manifolds](https://arxiv.org/abs/1303.1029), [distributed calibration](https://academic.oup.com/mnras/article/449/4/4506/1203097) |
| Repetti et al., 2017 | Joint imaging and direction-dependent calibration with convergence results under assumptions | Joint estimation and sparse priors are existing foundations. [Paper](https://academic.oup.com/mnras/article/470/4/3981/3852300) |
| Albert et al., 2020; preprint 2019 | A physical Gaussian-process model relating electron density to differential TEC through ray integrals | Three-dimensional probabilistic ionospheric tomography is a particularly close precedent. [Paper](https://arxiv.org/abs/1910.04529) |
| Sims, Pober & Sievers, 2022: BayesCal | Analytic marginalization of missing/uncertain sky contributions in calibration | Sky-uncertainty marginalization is not new; its primary demonstration concerns absolute calibration of a redundantly calibrated array. [Formalism](https://academic.oup.com/mnras/article/517/1/910/6631568) |
| Roth et al., 2023: resolve | Joint Bayesian sky and direction-dependent gain inference using image-domain gridding, demonstrated on Cygnus A | Continuous correlated gains, joint inference, IDG, and approximate uncertainty maps already coexist. [Paper](https://arxiv.org/abs/2305.05489) |
| Kenyon et al., preprint December 2024: QuartiCal | Scalable calibration using Numba and Dask, with a MeerKAT demonstration | A practical integration/comparison target. [Paper](https://arxiv.org/abs/2412.10072) |
| Brackenhoff et al., July 2025 | Spectral regularization weighted by expected station response for difficult off-axis calibration | Compare against physically weighted regularization, not just unrestricted gains. [Paper](https://academic.oup.com/mnras/article/541/4/3993/8215205) |
| Hodgson & Johnston-Hollitt, January 2026: Pigi | GPU image-domain gridding for varying A-terms; spatial sampling limits accuracy | Efficient DDE application is available, but throughput does not establish calibration accuracy. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S221313372500085X) |
| van der Tol et al., February 2026: IDG-CAL | Direct fitting of continuous A-terms through IDG, using basis functions and stochastic optimization; LOFAR comparison | Continuous gain fields replacing discrete directional solutions are already proposed and demonstrated. Calibration and imaging remain separate steps in this formulation. [Preprint](https://arxiv.org/html/2602.06002v1) |
| Russeeawon, Kenyon, Bester, Edler, de Gasperin & Smirnov, June 2026 | Physically motivated regression initialization separates clock and TEC signatures; tested on difficult LOFAR LBA data and incorporated into QuartiCal | Especially relevant recent work from Rhodes/SARAO collaborators. Initialization and phase wrapping must be addressed before local identifiability analysis is useful. [Paper](https://academic.oup.com/mnras/article/549/4/stag1078/8704126) |

Learned imaging also continues to improve. The 2025 R2D2 study revisits training, architecture and stopping based on residual consistency; its stated setting is monochromatic intensity imaging with telescope-specific training. This is useful imaging research, but does not by itself solve unknown DDEs. A learned sky prior should be an optional later comparison with explicit tests of unusual morphology. [R2D2 paper](https://arxiv.org/abs/2503.02554)

## 3. What transfers from other fields

### MRI: joint estimation, subspaces and separable optimization

Parallel MRI measures samples of the Fourier transform of an image multiplied by each coil's sensitivity. ESPIRiT estimates sensitivity subspaces from correlations in calibration data. The useful analogy is that channel responses and object structure must be consistent with the same observations. [Uecker et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4142121/)

The analogy has a limit: radio visibilities are correlations of stochastic electric fields, with baseline response J_p B J_q^H. They are not individual MRI coil measurements, and usually do not provide MRI's fully sampled central calibration region. A direct transplant of ESPIRiT's k-space calibration matrix would need a new identifiability argument.

MRI water/fat and field-map estimation also exploits variables that are linear conditional on nonlinear field parameters. Variable projection eliminates the linear variables inside an outer nonlinear solve. This is a useful numerical pattern for sky coefficients conditional on calibration parameters. [Hernando et al. 2008](https://experts.illinois.edu/en/publications/joint-estimation-of-waterfat-images-and-field-inhomogeneity-map/)

Positivity and sparse sky penalties matter: with these constraints the inner image solve is generally an optimization problem, not a closed-form inverse. Variable projection, joint MAP fitting, and Bayesian marginalization must not be described as interchangeable.

### Adaptive optics: predictive atmospheric tomography

Adaptive optics uses spatially structured state models to predict moving turbulent layers. Sparse Kalman models can exploit approximately frozen flow. Transfer the physical model and computational structure to radio ray integrals, with process noise for departures from frozen flow. [Fraanje et al. 2010](https://doi.org/10.1364/JOSAA.27.00A235)

The radio filter precedent above rules out claiming that Kalman prediction itself is new. The opportunity is to adapt atmospheric model complexity to the information in each observation, while checking sky distortion.

### Ptychography: ambiguity analysis and depth limits

Blind ptychography jointly estimates an object and illumination, with nontrivial scaling and phase ambiguities. Its lesson is to identify the null space before attributing structure to the reconstructed object or instrument. [Fannjiang & Chen](https://arxiv.org/abs/1806.02674)

Multislice ptychographic tomography models propagation through depth and has experimental demonstrations. This suggests a possible later treatment of radio scintillation using propagation between layers. However, coherent illumination, scan overlap and controlled projection angles provide information unavailable to a typical radio observation. Extra layers do not automatically make atmospheric depth recoverable. [Multislice tomography experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC5795012/)

### Component separation: explicit signal-response constraints

CMB internal linear combination methods preserve selected spectral responses while suppressing contamination; partially constrained variants expose the bias–variance tradeoff. The transferable idea is an explicit signal-distortion budget. Radio calibration requires a local nonlinear generalization because the gains are estimated from the same data. [Abylkairov et al.](https://arxiv.org/abs/2012.04032)

## 4. Candidate method

### 4.1 A physical basis with realistic geometry

Start with scalar Stokes-I, phase-only direction-dependent calibration and separately anchored instrumental amplitude. Represent electron-density perturbations in a few thick curved layers. For a straight-ray, weak-refraction approximation,

```math
T_p(\hat s,t)=\int_{\mathrm{ray}(p,\hat s)} n_e(r,t)\,d\ell,
\qquad
\phi_p(\hat s,t,\nu)=2\pi\nu\tau_p(t)-K T_p(\hat s,t)/\nu.
```

K is the plasma phase conversion constant for the chosen TEC units. Differential TEC and delay determine baseline phases. Use station positions and rotating celestial directions in a consistent Earth-centred coordinate system to evaluate rays.

A discrete dynamics model is

```math
a_{t+1}=M(v_t)a_t+\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,Q_t),
```

where a contains layer coefficients and M transports them along the shells. Begin with fixed layer heights and externally chosen or conservatively estimated velocities. Test inference of heights and velocities only after demonstrating observable vertical or temporal structure. A flexible independent gain field can otherwise mask failures of the physical model.

Retain the exact spherical geometric phase in the reference calculation. For scale, replace direct summation with a verified wide-field operator supporting the necessary A-terms and derivatives. The physical atmosphere model and the numerical gridding model are different components.

### 4.2 Measure what calibration can distinguish from sky uncertainty

The following derivation is a local Gaussian model, not a global theorem. Stack real and imaginary visibility parts and whiten using their noise covariance. Linearize around a current sky and the predicted physical calibration state:

```math
r=A\,\delta x+B\,\delta\theta+\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
```

A contains uncertain sky-template responses; B contains physical calibration derivatives. Use centred priors with precision matrices Lambda_x and Lambda_theta. The local objective is

```math
\tfrac12\|r-A\delta x-B\delta\theta\|^2
+\tfrac12\delta x^T\Lambda_x\delta x
+\tfrac12\delta\theta^T\Lambda_\theta\delta\theta.
```

Eliminating delta x yields

```math
R=I-A(A^TA+\Lambda_x)^{-1}A^T,
\quad F=B^TRB,\quad F_0=B^TB,
```

and the calibration normal equation

```math
(F+\Lambda_\theta)\delta\theta=B^TRr.
```

Use linear solves, not explicit inverses. For rank-deficient unregularized cases use appropriate pseudoinverses. If relinearizing away from the prior centre, include the prior-gradient terms in the right-hand side.

F_0 treats the sky as known. F measures the local information left after allowing the specified sky uncertainty. With proper positive Gaussian sky covariance C_x, R=(I+A C_x A^T)^(-1). For fixed A and covariance, Gaussian profiling and marginalization give the same local calibration quadratic. When A or covariance depends on theta, the marginal likelihood also contains a log-determinant and its derivatives; ignoring these does not implement full BayesCal or Bayesian joint inference.

The elimination and uncertainty treatment have close precedents in BayesCal and joint inversion. Efficient block elimination in radio calibration also predates this proposal. [Block LDU calibration](https://arxiv.org/abs/1803.05707)

Remove true gauge freedoms before interpreting F. A common antenna phase is unobservable; flux scale requires an anchor when amplitudes and sky flux are jointly free. Astrometric and polarization ambiguities depend on the chosen model and available calibrators. A reference antenna alone does not resolve every degeneracy.

### 4.3 Select modes with information and distortion criteria

On the subspace where F_0 is positive definite, solve

```math
Fv_k=\eta_k F_0v_k.
```

For the local model, 0 <= eta_k <= 1. Small eta means that allowing sky uncertainty removes much of the apparent information for that calibration combination. This ratio alone is insufficient: a mode can have eta near one and still have almost no absolute information. Also compare its information to the physical prior, for example through v_k^T F v_k / (v_k^T Lambda_theta v_k) for a proper prior.

Choose retained columns E from these candidate modes. For fixed E, A, B and priors, define the calibration-induced subtraction response

```math
L_E=BE(E^TB^TRBE+\Lambda_E)^{-1}E^TB^TR,
\qquad \Lambda_E=E^T\Lambda_\theta E.
```

A small added visibility signal s then contributes (I-L_E)s to the residual after the calibration-induced model update in this local model. This describes model subtraction; calibrated images and inverse-gain-corrected data have additional response operators.

Let Q_s have orthonormal columns spanning whitened visibility responses of designated science perturbations. Propose selecting E and its regularization subject to

```math
\|L_E Q_s\|_2\leq\varepsilon.
```

This bounds the worst relative visibility distortion for perturbations within that specified subspace. It is stronger than checking that the current fitted correction has a small projection onto Q_s: the constraint concerns how the estimator responds to new signal.

Use a greedy addition/pruning rule or a small search over mode thresholds and shrinkage, selecting the best blocked validation likelihood among candidates satisfying the budget. Charge the selection cost to the runtime. Initially recompute modes once per major cycle or time block, not per visibility.

This is the proposed research contribution: combining physical mode selection with an explicit estimator-response budget in wide-field direction-dependent calibration. It remains possible that it is equivalent to an existing regularization or constrained-estimation method. The algorithm must demonstrate more than repackaging a Gaussian prior.

The local guarantee does not cover changes in the selected basis, fitted hyperparameters, nonlinear solver basin, or subsequent deconvolution. Validate the full pipeline by finite signal injections with complete refitting. Protecting all possible sky signals would generally leave no useful calibration information; Q_s must state the science scope, and results must also be tested outside it.

### 4.4 An analytic failure check

Take a unit sky response a and a unit calibration response b with correlation c=a^T b. With a completely uncertain sky coefficient,

```math
R=I-aa^T,\qquad F=1-c^2,\qquad F_0=1.
```

When c=0, sky uncertainty costs no calibration information. When c approaches one, the unregularized joint calibration variance scales as 1/(1-c^2). At c=1, the sky coefficient and gain perturbation cannot be separated from these data.

For r=a alpha+b beta, ordinary calibration treating the sky as known estimates beta+c alpha and absorbs part of the sky signal. Eliminating the uncertain sky coefficient removes this bias for |c|<1 but pays the variance penalty. With a prior or mode freezing it pays a bias penalty instead. This example is a mathematical sanity check, not a telescope experiment and not a novel result.

## 5. The closest novelty objections

| Proposed ingredient | Close precedent | What still needs to differ |
| --- | --- | --- |
| Physical ionospheric tomography | Albert et al. 2020 | Adaptive choice of observable combinations tied to science distortion |
| Sky uncertainty marginalized in calibration | BayesCal 2022; resolve 2023 | A useful scalable selection/response constraint in physical wide-field DDE calibration |
| Calibration identifiability | Mouri Sardarabadi & Koopmans 2019 | Turning diagnostics into a tested model-complexity rule. [Paper](https://arxiv.org/abs/1902.02482) |
| Quantifying signal suppression | Mouri Sardarabadi & Koopmans 2019 | Enforcing a specified transfer budget during model selection. [Paper](https://academic.oup.com/mnras/article/483/4/5480/5257855) |
| Adaptive regularization based on estimator influence | Yatawatta & Leahy 2021 | Explicit worst-case response constraints and physical modal selection instead of only a learned regularization policy. [Paper](https://academic.oup.com/mnras/article/505/2/2141/6276731) |
| Continuous A-terms solved through IDG | IDG-CAL 2026 | Physics and science-response constraints beyond continuous parameterization |

The suppression literature already shows why short-baseline exclusion and smooth gains involve tradeoffs. The reinforcement-learning paper already uses influence functions to choose calibration regularization. Both are mandatory comparisons before describing this as a new approach to preserving faint emission.

## 6. An experiment that can reject the proposal

### First controlled test

Use a synthetic low-frequency array with 24–32 stations, an approximately ten-degree field, several frequencies across a broad band, and Earth rotation. Treat these as manageable experimental settings, not a faithful LOFAR simulator. Use bright compact calibrators, faint compact objects, and extended emission with a range of angular scales and spectral structures.

Generate data with a direct spherical RIME and ray integrals through one or two thick moving layers. Include clock terms and thermal noise. Fit with a coarser, different atmospheric basis and a different sky representation to avoid an inverse crime. Add departures from frozen flow and imperfect beam information in subsequent runs. Start with correct known beam and fixed amplitude to isolate the calibration-selection mechanism.

Sweep calibrator density, thermal noise, field size, vertical separation, atmospheric evolution, bandwidth, and missing sky power. Include negligible ionosphere, correct simple-screen, severe wrapping, and deliberately unidentifiable configurations. A useful method must know when its extra complexity is unsupported.

### Comparisons and ablations

Run matched physical/gain models, sky priors, data weights, initialization and compute budgets where feasible:

1. Direction-independent calibration, as a basic control.
2. Smooth direction-dependent/facet calibration with tuned regularization.
3. Fixed-complexity physical layer calibration.
4. The same physical model with uncertain-sky joint fitting or marginalization, without adaptive selection.
5. Information-based mode selection without the distortion budget.
6. Mode selection plus the proposed distortion budget.
7. Oracle known gains and oracle complete sky, as diagnostic bounds.

The fourth comparison is decisive: if it matches the full proposal, the demonstrated benefit comes from known uncertainty modelling rather than the proposed selection rule. Next compare with available IDG-CAL and killMS/DDFacet or QuartiCal workflows on a shared dataset. A reimplementation inspired by a paper must be labelled as such, not reported as a run of that package.

### Metrics and validation

Measure recovered flux and spectra, astrometric error, extended-emission recovery by angular scale, predicted visibility likelihood on held-out data, residual source-correlated structure, gain/TEC errors modulo gauges, runtime and memory. For a 21-cm experiment, add a dedicated power-spectrum transfer measurement; continuum flux recovery is not a substitute.

Split time blocks and frequency groups with enough separation to expose interpolation errors, while preserving enough training coverage for an identifiable solve. Reserve final test blocks separate from model-selection blocks. Shared sky and atmosphere mean held-out visibilities are not fully independent science realizations; multiple independent simulated skies and atmospheres are also needed.

Inject science components into visibilities before calibration, using the forward corruption model. Refit every step on paired injected/uninjected data with matched noise, then measure their difference. Repeat with novel positions, morphologies, amplitudes and spectral shapes absent from the protected templates. Propagate the measurement through imaging as well as calibration.

Choose the tolerance before examining final results. For example, a one-percent local visibility-distortion budget is an experimental setting, not a universal astronomy requirement or a guaranteed one-percent image/power error. Report the measured response and uncertainty across seeds. Compare methods at matched residual fidelity and compute, including the cost of hyperparameter search.

Reject or narrow the proposal if selection adds no benefit over ordinary joint inference, protection only works for its training templates, underfitting leaves worse foreground residuals, gains remain prior-dominated, initialization failures dominate, or mode computation costs more than it saves. A cleaner-looking image or lower training residual alone is insufficient evidence.

### Scaling after a successful test

Use matrix-free A, B and adjoints, iterative inner solves, and partial eigensolvers. Do not form a visibility-by-visibility R or L_E matrix. Apply L_E only to template vectors or randomized probes. A small atmosphere coefficient space and infrequent basis updates are essential hypotheses for computational feasibility.

Only after the controlled test should the project add Measurement Set I/O, full Jones polarization with physically ordered terms, realistic beams, robust noise models, and a public real observation. Record the observation identifier, software versions, flags, weights, calibration anchors and full recipes when selecting that dataset. None has been selected or downloaded in this investigation.

## 7. Research decision

The most defensible first Rang study is about **how much calibration complexity the data can support without unacceptable science distortion**. MRI, adaptive optics, ptychography and component separation suggest useful mathematics, while radio's own recent literature supplies most building blocks and strong competing explanations.

The current deliverable is a falsifiable method specification and its prior-art map. A claim of novelty needs a deeper citation search around the six closest precedents; a claim of usefulness needs the controlled and real-data experiments above.
