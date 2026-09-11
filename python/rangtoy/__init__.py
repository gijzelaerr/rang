"""Run Rust calibration experiments from Python.

The first interface uses a JSON subprocess boundary. Rust owns the numerical
work; Python owns configuration, campaigns, tables and plotting. Set RANG_TOY_BIN
when using an installed package outside the source checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "Cargo.toml").is_file():
        raise RuntimeError(
            "Run from the Rang checkout, or pass a binary / set RANG_TOY_BIN"
        )
    return root


def build() -> Path:
    """Build the numerical core in release mode, without downloading crates."""
    root = repository_root()
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--release",
            "--manifest-path",
            str(root / "Cargo.toml"),
        ],
        cwd=root,
        check=True,
    )
    # Respect Cargo's target directory setting if the user has supplied one.
    metadata = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--offline", "--no-deps", "--format-version", "1"],
            cwd=root,
            text=True,
        )
    )
    suffix = ".exe" if os.name == "nt" else ""
    return Path(metadata["target_directory"]) / "release" / f"rang-toy{suffix}"


def run_experiment(
    *,
    seed: int = 7,
    noise: float = 0.01,
    pointing: float = 0.6,
    beam_error: float = 0.0,
    budget: float = 0.01,
    target_flux: float = 0.02,
    binary: str | Path | None = None,
) -> dict:
    """Return one reproducible experiment as a dictionary.

    noise: Jy per real/imaginary component. pointing: sigma per axis in arcmin.
    beam_error: fractional true beam width mismatch. budget: local operator-norm
    distortion bound, not an image accuracy guarantee. See examples/README.md.
    """
    configured = binary or os.environ.get("RANG_TOY_BIN")
    executable = Path(configured).resolve() if configured else build()
    command = [str(executable)]
    for key, value in {
        "seed": seed,
        "noise": noise,
        "pointing": pointing,
        "beam-error": beam_error,
        "budget": budget,
        "target-flux": target_flux,
    }.items():
        command.extend([f"--{key}", str(value)])
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "Rust experiment failed")
    result = json.loads(completed.stdout)
    result["execution_seconds"] = time.perf_counter() - started
    result["binary_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    result["command"] = command
    return result


__all__ = ["build", "run_experiment"]
