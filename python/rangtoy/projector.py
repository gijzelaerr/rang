"""Checked NumPy/ctypes boundary for the dependency-free Rust QR projector."""

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import build


@lru_cache(maxsize=1)
def _native_projector():
    configured = os.environ.get("RANG_TOY_LIB")
    if configured:
        path = Path(configured).resolve()
    else:
        filename = (
            "rang_toy.dll"
            if os.name == "nt"
            else "librang_toy.dylib"
            if sys.platform == "darwin"
            else "librang_toy.so"
        )
        path = build().parent / filename
    library = ctypes.CDLL(str(path))
    project = library.rang_project_out
    vector = np.ctypeslib.ndpointer(
        dtype=np.float64, ndim=2, flags=("C_CONTIGUOUS", "ALIGNED")
    )
    project.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        vector,
        vector,
        vector,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_double,
    ]
    project.restype = ctypes.c_int
    return project


def project_out(nuisance, targets, *, relative_tolerance=1e-12):
    """Return rotated residual coordinates and nuisance rank.

    The returned (rows-rank, target_columns) matrix preserves the Gram matrix
    of targets projected orthogonally off the nuisance column space. It is
    not expressed in the original visibility-row coordinates. No data rows
    are statistically discarded: the removed rows span fitted nuisances.
    """
    if np.iscomplexobj(nuisance) or np.iscomplexobj(targets):
        raise ValueError("projector requires real matrices")
    a, z = (
        np.require(x, dtype=np.float64, requirements=["C", "A"])
        for x in (nuisance, targets)
    )
    if (
        a.ndim != 2
        or z.ndim != 2
        or a.shape[0] != z.shape[0]
        or z.shape[0] == 0
        or z.shape[1] == 0
    ):
        raise ValueError("incompatible nonempty matrix dimensions")
    if (
        not np.isfinite(a).all()
        or not np.isfinite(z).all()
        or not np.isfinite(relative_tolerance)
        or not 0 < relative_tolerance < 1
    ):
        raise ValueError("finite matrices and tolerance in (0,1) required")
    output = np.empty_like(z)
    rank = ctypes.c_size_t()
    status = _native_projector()(
        len(z),
        a.shape[1],
        z.shape[1],
        a,
        z,
        output,
        ctypes.byref(rank),
        relative_tolerance,
    )
    if status:
        raise RuntimeError(f"Rust nuisance projection failed (status {status})")
    return output[rank.value :], rank.value
