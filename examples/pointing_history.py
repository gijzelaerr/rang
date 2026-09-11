"""Inspect the public SARAO history safely, without interpolating missing data."""

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    with np.load(args.path, allow_pickle=False) as data:
        pointing = data["antpoint_history_rad"]
        timestamps = data["timestamps"]
    if pointing.shape != (2, 64, len(timestamps)):
        raise ValueError("unexpected history dimensions")
    valid = np.isfinite(pointing).all(axis=0)
    counts = valid.sum(axis=0)
    arcsec = np.where(valid[None], pointing * (180 * 3600 / np.pi), np.nan)
    means = np.nanmean(arcsec, axis=1)
    gaps = np.diff(np.unique(timestamps))
    adjacent_valid = valid[:, :-1] & valid[:, 1:]
    repeats = (pointing[:, :, :-1] == pointing[:, :, 1:]).all(axis=0)
    summary = {
        "source": "https://doi.org/10.48479/s9nh-3s43",
        "attribution": "M. S. de Villiers and W. D. Cotton (2022), MeerKAT Primary-beam Measurements in the L Band",
        "licence": "CC BY-NC 4.0; original data not redistributed in this repository",
        "sha256": sha256(args.path.read_bytes()).hexdigest(),
        "shape": list(pointing.shape),
        "timestamp_range_utc_assuming_unix_seconds": [
            datetime.fromtimestamp(float(t), UTC).isoformat()
            for t in (timestamps.min(), timestamps.max())
        ],
        "finite_antenna_count_min_median_max": [
            int(counts.min()),
            float(np.median(counts)),
            int(counts.max()),
        ],
        "unique_timestamp_count": len(np.unique(timestamps)),
        "gap_seconds_min_median_max": [
            float(gaps.min()),
            float(np.median(gaps)),
            float(gaps.max()),
        ],
        "coordinate_rms_arcsec": np.sqrt(np.nanmean(arcsec**2, axis=(1, 2))).tolist(),
        "available_antenna_mean_rms_arcsec": np.sqrt(
            np.nanmean(means**2, axis=1)
        ).tolist(),
        "complete_64_antenna_epochs": int(np.sum(counts == 64)),
        "exact_repeat_fraction_adjacent_valid_antenna_records": float(
            repeats[adjacent_valid].mean()
        ),
        "absolute_coordinate_arcsec_percentiles_50_90_99_100": np.nanpercentile(
            np.abs(arcsec), [50, 90, 99, 100], axis=(1, 2)
        ).tolist(),
        "warning": "Coordinate names and missing-value encoding require source verification. Historical irregular samples are not a continuous science-track trajectory. Available-antenna means need not be full-array means.",
    }
    print(json.dumps(summary, indent=2))
    args.path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
