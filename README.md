# Rang — Radio Astronomy Next Generation

A research demo exploring calibration and imaging methods for MeerKAT radio interferometry.

The active scope is [MeerKAT calibration research](research/meerkat-scope.md), initially targeting L-band continuum imaging and physical beam/pointing corrections. Satellite-assisted calibration is outside the initial scope.

The [initial cross-field investigation](research/calibration-research-2026-09-11.md) provides the literature review and mathematical proposal. Its original low-frequency atmospheric experiment is superseded by the MeerKAT scope.

There is now a [runnable toy experiment](examples/README.md), with Rust numerical kernels and Python orchestration:

```sh
python3 examples/toy.py --seeds 7 11 19 --output outputs/demo
```

It compares pointing-only and joint sky/pointing fitting with a prototype of adaptive mode protection, and writes JSON results plus PNG/PDF plots. Rust/Cargo and Python are required; plotting additionally needs Matplotlib. Use `--no-plot` to run without plotting dependencies.

The toy demonstrates signal suppression from an incomplete calibration sky. The protection step currently matches ordinary joint fitting, with no demonstrated additional benefit. It uses a simplified beam and an approximate eight-antenna MeerKAT core layout; it is not a validated MeerKAT calibrator.

The [first results note](research/first-results.md) reports a 20-seed beam-mismatch experiment: near-perfect injection response can coexist with a large flux bias, and fitting a shared beam width resolves the bias in this controlled model.
