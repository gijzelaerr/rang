# Hierarchical nuisance elimination

The local pointing-information audit now optionally eliminates independent channel gains before eliminating pointing nuisances shared across channels. This avoids explicitly zero-padding every channel's gain columns into the full time-block matrix. Thin QR preserves retained inner products; no visibility subsampling is involved.

This is standard structured least-squares elimination, not a novel calibration algorithm. It is invalid when eliminated channel gains have cross-channel or cross-time priors. The nonlinear smooth-gain solver is not replaced by this local audit.

## Numerical controls

Dense, time-blocked and frequency-blocked calculations agree on the scalar katbeam fixture, including constrained covariance. Gaussian exact-null and flux-anchor rank tests pass. An extra adversarial test revealed that ranking projected shared columns relative to their own residual scale promotes round-off to rank when gains already explain them. The implementation now measures shared rank relative to the original shared matrix Frobenius norm (threshold 1e−12). This is an explicit, scale-dependent numerical convention, not a guarantee for arbitrary ill-conditioned parameterizations.

## Measured pilot

Native ARM Mac, five warmed repetitions, `OPENBLAS_NUM_THREADS=1`, same eight-antenna toy and prior audit. Times include Jacobian preparation, elimination and the final audit, excluding one warmup per method.

NumPy uses Apple's Accelerate backend here. The OpenBLAS environment setting does not establish single-thread execution; actual BLAS thread counts were not instrumented. These pilot measurements precede the subsequent Rust internal-layout optimization.

| Method | Median seconds |
| --- | ---: |
| Dense NumPy | 1.5205 |
| Time-blocked Rust | 0.1777 |
| Time-blocked NumPy | 0.1612 |
| Frequency-blocked Rust | 0.1533 |
| Frequency-blocked NumPy | 0.1515 |

Maximum absolute difference from dense total information is 2.6e−11. All methods give the same data-only common-mode standard deviations (7.9581, 7.9978 arcsec). Frequency blocking reduces the Rust-backed path by about 14%; NumPy remains slightly faster. The large gain versus dense comes from blocking, not evidence that Rust outperforms optimized numerical libraries.

Reproduce with `OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/blocked_benchmark.py --repeats 5`. Full timing output is archived in `results/frequency-blocking.json`. This is one small local-information calculation, not a full MeerKAT calibration runtime or state-of-the-art comparison.

## Larger structural benchmark

`examples/projector_scaling.py` generates gain derivatives with the correct baseline-incidence structure at 8, 16, 32 and 64 antennas, four independent channels, random shared nuisance columns and two random targets. It is deliberately not a beam/sky simulation: it tests computational scaling without pretending to have verified full-array geometry.

Each method/dimension runs in a fresh worker. Timings exclude fixture creation and one warmup but include gain-column expansion where required. Peak RSS includes the entire worker, fixture and warmup; it is not an incremental kernel allocation. Environment, numerical-library backend and source hashes are recorded. The archive `results/projector-scaling.json` contains all repetitions and numerical checks.

The investigation exposed strided memory access in the dependency-free Rust QR implementation. Internal column-major storage and four independent dot-product accumulators improve its 64-size grouped pilot from 1.36 seconds to about 0.17 seconds. The row-major Python/FFI contract is unchanged. This is an implementation improvement, not mathematical novelty; optimized NumPy remains faster on large blocks. The exact Gaussian-null and dense-information controls continue to pass.

Final five-repeat 64-size measurements, run after the test suite finished:

| Method | Median seconds | Worker peak RSS (MiB) |
| --- | ---: | ---: |
| Full NumPy | 0.1525 | 312.2 |
| Full Rust | 1.9204 | 305.7 |
| Grouped NumPy | 0.0621 | 133.5 |
| Grouped Rust | 0.1713 | 134.0 |

All ranks are634 and relative retained-Gram differences are below5e−16. Grouped NumPy uses about43% of the full NumPy worker's peak memory and is about2.46× faster in this isolated calculation. Rust remains about2.76× slower than grouped NumPy at this size; no optimized-library speed advantage is claimed. These RSS ratios include interpreter overhead and are machine/run dependent.

Reproduce with `OPENBLAS_NUM_THREADS=1 .venv/bin/python examples/projector_scaling.py --repeats 5`. Run separately from test suites and other benchmarks. No end-to-end imaging or same-resource state-of-the-art claim follows from these timings.
