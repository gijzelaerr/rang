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
