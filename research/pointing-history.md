# Public pointing-history inspection

Source: M. S. de Villiers and W. D. Cotton (2022),
[MeerKAT Primary-beam Measurements in the L Band, data release](https://doi.org/10.48479/s9nh-3s43).
The release is **CC BY-NC 4.0**, not the repository's software licence.
The original NPZ remains in ignored `outputs/pointing-history/` and is not
redistributed. No archive visibility data or large beam cubes were downloaded.

## Reproduce

Download `lband_antenna_pointing_history.npz` using the link on the
[release page](https://archive-gw-1.kat.ac.za/public/repository/10.48479/s9nh-3s43/index.html),
then run:

```sh
.venv/bin/python examples/pointing_history.py outputs/pointing-history/lband_antenna_pointing_history.npz
```

The script disables pickle loading and records a SHA-256 digest. Inspected
file digest: `4eba9bc9c798a0a7e3312c9496b2f275bfe48856deb9f399b264bede5d9975b7`.

## Findings and limits

- Arrays: `antpoint_history_rad` has shape `(2, 64, 881)`; `timestamps` has 881 entries.
- Interpreting timestamps as Unix seconds gives March 2019 to January 2022.
- There are 876 distinct timestamps. Median gap between distinct timestamps is
  2013 seconds, but the largest is approximately 28.4 days.
- Between 30 and 64 antennas have finite values at an epoch. There are 773
  entries with finite values for all 64 antennas; finiteness does **not** prove
  all those antennas were measured simultaneously.
- About 53.8% of adjacent, finite antenna records repeat exactly in both
  coordinates. This may reflect record assembly or repeated values; we have
  not established the cause. They cannot be assumed independent measurements.
- Median absolute coordinates are about 21 and 25 arcsec, with extreme values
  exceeding 2300 arcsec. Raw RMS is outlier-sensitive and is not a reproduction
  of the paper's quoted pointing accuracy statistic.

The [paper, §3.2 and Figure 3](https://arxiv.org/pdf/2202.02101) describes the
historical holography measurements and occasional severe antenna errors.
It also notes that only part of the array scans in a typical holography cycle
(§3.1). Coordinate order/sign, record assembly, and measurement freshness must
be established before interpreting covariance or fitting a stochastic drift
model. In particular, do not interpolate these multi-year records into a
fictional six-hour science observation.

The immediate benchmark therefore remains **synthetic**, with measured-data
inspection informing its limitations rather than supplying trajectories.
Next useful external clarification is the NPZ assembly convention, not an
arbitrarily fitted temporal correlation length.
