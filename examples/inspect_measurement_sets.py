"""Read-only inspection of the official SKA MeerKAT pointing test archive.

Run with optional python-casacore (e.g. tools/ms-reader Docker image).
The ZIP is checked before extraction to a fresh output directory. Source
Measurement Sets are never opened writable, and source data are not committed.
"""

import argparse
import json
import stat
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath

import numpy as np


def json_default(value):
    """Preserve casacore's NumPy-valued column metadata in JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported metadata type: {type(value).__name__}")


def validate_members(members):
    """Reject path traversal, links, duplicate targets and oversized archives."""
    if sum(m.file_size for m in members) > 200_000_000:
        raise ValueError("archive exceeds 200 MB uncompressed limit")
    seen = set()
    for member in members:
        path = PurePosixPath(member.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.filename
            or stat.S_ISLNK(member.external_attr >> 16)
            or path in seen
        ):
            raise ValueError("unsafe archive member")
        seen.add(path)


def pointing_layout(times, antenna_ids, antenna_count):
    """Diagnose storage layout without guessing missing antenna identities."""
    times, antenna_ids = np.asarray(times), np.asarray(antenna_ids)
    unique, counts = np.unique(times, return_counts=True)
    diagnostic = {
        "unique_timestamps": len(unique),
        "rows_per_timestamp": np.unique(counts).tolist(),
        "unique_antenna_ids": np.unique(antenna_ids).tolist(),
        "complete_antenna_labels": bool(
            np.array_equal(np.unique(antenna_ids), np.arange(antenna_count))
        ),
        "time_major_rectangular": False,
        "antenna_major_rectangular": False,
    }
    if times.size and times.size == antenna_count * len(unique):
        diagnostic["time_major_rectangular"] = bool(
            np.all(times.reshape(-1, antenna_count) == unique[:, None])
        )
        diagnostic["antenna_major_rectangular"] = bool(
            np.all(times.reshape(antenna_count, -1) == unique[None, :])
        )
    return diagnostic


def main():
    from casacore.tables import table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.infolist()
        validate_members(members)
        archive.extractall(args.output)
    results = []
    for path in sorted((args.output / "outer_scans").glob("*.ms")):
        result = {"name": path.name}
        with table(str(path), readonly=True, ack=False, lockoptions="autonoread") as ms:
            result["row_count"] = ms.nrows()
            result["columns"] = ms.colnames()
            for name in (
                "ANTENNA1",
                "ANTENNA2",
                "FIELD_ID",
                "DATA_DESC_ID",
                "SCAN_NUMBER",
            ):
                result[name] = np.unique(ms.getcol(name)).tolist()
            result["time_range_ms_seconds"] = [
                float(np.min(ms.getcol("TIME"))),
                float(np.max(ms.getcol("TIME"))),
            ]
            result["data_cell_shape"] = list(ms.getcell("DATA", 0).shape)
            result["flagged_fraction"] = float(np.mean(ms.getcol("FLAG")))
        for subtable in (
            "ANTENNA",
            "SPECTRAL_WINDOW",
            "POLARIZATION",
            "FIELD",
            "POINTING",
        ):
            with table(
                str(path / subtable), readonly=True, ack=False, lockoptions="autonoread"
            ) as sub:
                description = {"row_count": sub.nrows(), "columns": sub.colnames()}
                wanted = {
                    "ANTENNA": ["NAME", "POSITION", "DISH_DIAMETER"],
                    "SPECTRAL_WINDOW": ["CHAN_FREQ", "CHAN_WIDTH"],
                    "POLARIZATION": ["CORR_TYPE"],
                    "FIELD": ["NAME", "PHASE_DIR"],
                    "POINTING": ["TIME", "ANTENNA_ID", "TARGET", "SOURCE_OFFSET"],
                }[subtable]
                for name in wanted:
                    if name not in sub.colnames() or sub.nrows() == 0:
                        continue
                    values = np.asarray(sub.getcol(name))
                    description[name] = {
                        "shape": list(values.shape),
                        "first_values": values.reshape(-1)[:24].tolist(),
                    }
                    if np.issubdtype(values.dtype, np.number):
                        description[name]["min_max"] = [
                            float(np.nanmin(values)),
                            float(np.nanmax(values)),
                        ]
                    description[name]["keywords"] = sub.getcolkeywords(name)
                result[subtable] = description
                if subtable == "POINTING":
                    result["pointing_layout"] = pointing_layout(
                        sub.getcol("TIME"),
                        sub.getcol("ANTENNA_ID"),
                        result["ANTENNA"]["row_count"],
                    )
        results.append(result)
    payload = {
        "archive_sha256": sha256(args.archive.read_bytes()).hexdigest(),
        "source": "https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-wflow-pointing-offset/-/tree/main/tests/data",
        "limitations": "Dedicated reference-pointing scan, not continuous science-field imaging. Upstream reference solutions are algorithm outputs, not independent pointing truth.",
        "measurement_sets": results,
    }
    encoded = json.dumps(payload, indent=2, default=json_default)
    (args.output / "inspection.json").write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
