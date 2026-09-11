# Separating physical common and relative pointing

`solve_pointing(..., zero_mean_pointing=True,
common_pointing_prior_arcmin=sigma)` now fits a shared two-axis spline in
addition to the zero-mean antenna contrasts. Shared coefficients have
independent zero-centred Gaussian knot priors of width sigma. The basis
supplies smoothness; no extra common-mode curvature penalty is imposed.
This is a controlled prior assumption, not an external measurement.

The return values distinguish:

- `relative_offsets_arcmin`: exactly zero antenna mean at each time.
- `common_offsets_arcmin`: shared physical trajectory.
- `offsets_arcmin`: their sum, used for visibility prediction.

The option requires relative zero-mean constraints and `fit_pointing=True`.
Uncertainty `offset_std_arcmin` applies to the **total** offsets, not the
relative component alone when the common term is enabled. The gain-only
experiment baseline still fixes all pointing at zero.

## Prior sweep

Run for sigma0.1,0.5,3:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/relative_pointing.py --joint-gains --held-out --common-prior-arcmin 0.1
```

The same three-seed katbeam toy injects either zero common pointing or a
constant antenna-frame (+30,-30)arcsec common offset. All72 fits converge.
[Archived results](results/common-pointing-priors.json).

For the nonzero common offset and joint flux fitting:

| Prior sigma (arcmin) | Relative RMSE (arcsec) | Common RMSE (arcsec) | Test residual statistic |
| --- | --- | --- | --- |
| 0.1 | 0.740–1.021 | 0.224–0.370 | 1.012–1.022 |
| 0.5 | 0.743–1.025 | 0.176–0.336 | 1.012–1.023 |
| 3 | 0.743–1.025 | 0.175–0.341 | 1.013–1.023 |

Thus an explicit shared trajectory removes the earlier imposed-zero-mean
model mismatch for this particular truth and model. The relative zero-mean
convention remains exact. This is not a contradiction of Oleg's recommendation.

## Why this does not prove absolute identifiability

The fitted gains are smooth and achromatic. Sky sources have fixed spectral
indices and a single fitted reference flux, not free channel fluxes. Both
pointing components use four knots. The injected constant antenna-frame common
offset is also not the general rotating trajectory of the exact Gaussian
gauge. These restrictions and priors provide information absent from the
unrestricted gauge experiment. Modest sensitivity to these three prior widths
does not establish recovery independent of all assumptions.

The next control must broaden gain/sky/trajectory freedom and compare the
resulting uncertainty and common-mode bias. Common recovery on measured data
requires observation-specific validation; historical array-average pointing
scatter alone cannot supply a calibrated temporal prior.
