# Zero-mean relative pointing: recovery and a mismatch control

Following Oleg's recommendation, the smooth solver can constrain the
unweighted antenna mean pointing to zero at every time. An orthonormal
antenna contrast basis removes two parameters per spline knot. It preserves
the isotropic offset prior and enforces the constraint between knots too.
This is an established identifiability convention, not a novelty claim.

## Reproduce

Install the optional JAX and pinned katbeam dependencies as described in the
experiment guide, then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --seeds 7 11 19
```

The [archived results](results/relative-pointing.json) include the fixture hash,
beam provenance, convergence and individual errors. This uses the existing
eight-antenna approximate MeerKAT toy, all 2688 complex visibility rows, four
spline knots over six hours, scalar katbeam H and 1 mJy noise per real/imaginary
component. All gains are known and unity; beam parameters are exact.
Truth is a seeded smooth relative trajectory with 0.3 arcmin knot scatter
before subtracting the antenna mean. The physical shared offset is either
zero or (+0.5, -0.5) arcmin. Cases share truth and noise within each seed.

## Measured results

Ranges across three seeds; errors are per-coordinate RMS over time/antennas.

| Physical common offset | Sky treatment | Relative pointing RMSE (arcsec) |
| --- | --- | --- |
| Zero | Known sky | 0.63–0.89 |
| Zero | Fixed sky with ±2% off-axis flux errors | 5.67–6.22 |
| Zero | Joint fluxes, 5% priors around the wrong input sky | 0.63–0.88 |
| (+30, -30) arcsec | Known sky | 4.25–4.65 |
| (+30, -30) arcsec | Wrong fixed sky | 4.50–5.16 |
| (+30, -30) arcsec | Joint fluxes | 3.48–3.98 |

All 18 fits converge. The fitted mean is below 3e-15 arcsec. With zero
physical mean and joint fluxes, the mean squared whitened real residual is
0.975–0.999. With the nonzero physical mean it is 35.6–35.7, and the clean
visibility discrepancy is about 8.32 mJy complex RMS. The constraint alone
therefore does not make a restricted forward model adequate.

These are in-sample, known-gain recovery controls, not held-out validation or
real-data performance. The shared physical offset has deliberately not been
absorbed into a transformed sky/gain model. Consequently, this negative
control does not contradict a gauge convention in a sufficiently flexible
joint calibration model. Next: fit gains jointly and measure both relative
pointing errors and residual mismatch. Do not interpret a zero fitted mean
as evidence of zero physical mean.

## Joint smooth gains follow-up

The solver now accepts `gain_prior_sigma=(0.1, 0.1)` to fit log-amplitude and
phase spline coefficients jointly with pointing and sky. Phase is referenced
to antenna zero. Gains are achromatic, use the same knots as pointing, and
have Gaussian knot priors (no extra curvature penalty). This is a restricted
gain model, not free time/channel calibration. The default remains fixed gains.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains
```

The [joint-gain results](results/joint-gain-pointing.json) inject smooth gain
log-amplitude scatter 0.02 and phase scatter 0.03 radians at the knots, then
fit with 0.1 priors in each coordinate. All 18 fits converge. With zero physical
common pointing, joint flux/pointing/gain recovery gives 0.575–0.888 arcsec
relative pointing RMSE; fixed wrong fluxes give 6.51–6.90 arcsec. The joint
fits have whitened residual mean squares 0.964–0.988.

Absolute flux is not independently recovered: joint cases allow every flux
to vary, and the amplitude/flux scale is set by their priors. The resulting
nearly common fractional flux shifts are about -0.10% to +0.43% across seeds.
A separate noiseless automated test fixes the central source flux and checks
recovery of gains, off-axis fluxes and relative pointing simultaneously.

With physical (+30,-30) arcsec common pointing, the joint model still gives
4.33–4.74 arcsec relative error and residual mean squares 31.7–32.8. Smooth
achromatic gains and power-law source spectra are not flexible enough to
absorb this model mismatch. This does not test the unrestricted Gaussian
gauge, nor prove failure of more flexible gain/sky models.

Held-out prediction, gain-only comparison, beam-error robustness and free
channel-dependent gains remain outstanding. These results establish the
joint nonlinear implementation under matched assumptions, not deployment
readiness or a new calibration algorithm.

## Held-out times and gain/sky-only control

[Per-seed archived results](results/joint-gain-pointing-heldout.json).

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out
```

This withholds all baselines and frequencies at six complete time samples
(time index modulo four equals two): 2016 training and 672 test complex rows.
All spline knots and prior strengths are fixed before scoring. Test times are
inside the training time span; this is interpolation, not extrapolation.
`fit_pointing=False` supplies the gain/sky-only baseline with identical gain
and flux models and priors. It holds pointing at zero exactly.

Across three seeds with zero physical common offset:

| Test metric | Joint pointing/gain/flux | Gain/flux only |
| --- | --- | --- |
| Clean complex visibility RMS discrepancy | 0.220–0.247 mJy | 2.566–2.724 mJy |
| Mean squared whitened real residual | 1.011–1.023 | 4.329–4.773 |

Relative pointing RMSE over all times is 0.748–1.023 arcsec for the joint fit.
The clean test discrepancy improves by about 11–12 times. This is a matched
synthetic beam/spline test, not evidence of equivalent real-data performance.
Flux scale remains prior-dependent even where predictions pass this check.

With the physical (+30,-30) arcsec shared offset, test residual mean squares
remain 31.4–32.4 for joint fitting versus 35.3–36.3 for gain/flux-only. The
joint model improves predictions modestly but remains inadequate. The test
therefore distinguishes predictive improvement from adequate calibration.

[Historical pointing inspection](pointing-history.md) found irregular and
repeated records that cannot yet justify an empirical within-track drift
model. The historical arrays were **not** used as these simulated trajectories.
Beam mismatch, alternative spline complexity and real-data tests remain open.
