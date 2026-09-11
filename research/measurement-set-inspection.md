# Public MeerKAT pointing-scan inspection

The official [SKA pointing-offset pipeline test data](https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-pointing-offset/-/tree/main/tests/data) provide a small real-observation ingestion test. They are dedicated reference-pointing scans, not a continuous science-field calibration benchmark. No pointing accuracy result is claimed here.

## Observed contents

Inspection on 2026-09-11 of `outer_scans.zip`:

- SHA256: `d4dae56f7ad1bb4a49ab8c4a3846871d5b8ebb4b731ee958cb9f1baf3306144f`.
- Download size 45,452,147 bytes; unpacked file sizes total 83,571,601 bytes.
- Five Measurement Sets: scans 1, 4, 5, 8 and 9. The upstream README says six but lists these five.
- Antennas m000–m003; field J1939−6342; 4,096 channels from 856 to 1711.791 MHz; four correlation products.
- Main-table rows: 20, 20, 20, 20 and 140. FLAG fractions: 44.53%, 42.28%, 43.19%, 43.06% and 46.83%, including all correlations and autocorrelations.
- ANTENNA positions carry ITRF metre metadata. These are observation-file coordinates, not a verified full-array survey.

## Convention barrier

Every POINTING `ANTENNA_ID` is zero although each timestamp has four rows. Timestamp ordering is antenna-major and matches the reshape assumed by the [upstream reader](https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-pointing-offset/-/blob/main/src/ska_sdp_wflow_pointing_offset/read_data.py). That is consistent with intended antenna ordering, but does not independently identify the four row blocks.

Both TARGET and SOURCE_OFFSET have direction reference `J2000` in the column metadata. The upstream reader instead interprets TARGET as commanded azimuth/elevation and SOURCE_OFFSET as cross-elevation/elevation, converting radians to degrees. Consequently, a generic metadata-driven coordinate conversion cannot be trusted for this fixture. We preserve the original data and report the inconsistency; there is no automatic relabelling or coordinate repair.

The archive includes reference offset outputs. They are outputs of another estimator, not independent pointing truth. Comparing them requires reproducing its channel selection, flags, polarization treatment, beam model and coordinate conventions. Agreement alone would validate reproduction, not establish superior calibration or imaging.

The [upstream export metadata](https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-pointing-offset/-/blob/main/src/ska_sdp_wflow_pointing_offset/export_data.py) specifies fitted pointing and voltage-beam widths in radians, whereas intermediate on-sky scan offsets are in degrees. These must not be mixed.

## Reproduction

`examples/inspect_measurement_sets.py ARCHIVE OUTPUT` checks ZIP members and extracts into a fresh output directory, then opens all tables read-only. Output includes `inspection.json`; retain it locally with the original archive. Raw observations are not redistributed in Rang.

On this ARM Mac, no native casacore installation was available. `tools/ms-reader/Dockerfile` pins a Python image digest and public-PyPI NumPy/casacore wheels. Build with:

```sh
docker build --platform linux/amd64 -t rang-ms-reader:local tools/ms-reader
```

Run the image with the archive directory mounted read-only, a separate writable output directory, the script mounted as `/ms_reader.py`, `--network none`, `--read-only` and a `/tmp` tmpfs. Do not mount the script as `/inspect.py`: that shadows Python's standard-library module. The image entrypoint is Python. This emulated reader is not used for native performance claims.

Next required evidence is an independently verified antenna/coordinate mapping or an explicit, documented reproduction-only convention. Science-field superiority additionally requires a different observation and matched calibrator/image-quality comparisons.
