"""Controlled zero-mean pointing recovery with known gains and a katbeam beam."""

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rangtoy import build
from rangtoy.beams import load_katbeam, make_beam_predictor
from rangtoy.pointing import Observation, component_list, solve_pointing, spline_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    raw = subprocess.check_output([str(build()), "--pointing-full-reference"])
    fixture = json.loads(raw)
    r, s = np.asarray(fixture["rows"]), np.asarray(fixture["sources"])
    obs = Observation(
        jnp.asarray(r[:, :3]),
        jnp.asarray(r[:, 3]),
        *(jnp.asarray(r[:, i], dtype=int) for i in (4, 5, 6)),
        jnp.asarray(r[:, 7]),
    )
    sky = component_list(s[:, 0], s[:, 1], s[:, 2], s[:, 3])
    table, metadata = load_katbeam("H")
    predictor = make_beam_predictor(table, metadata=metadata)
    times, knots = np.linspace(0, 21600, 24), np.linspace(0, 21600, 4)
    design, _ = spline_design(times, knots)
    sigma = 0.001
    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(0, 0.3, (4, 8, 2))
        coefficients -= coefficients.mean(axis=1, keepdims=True)
        relative = np.einsum("tk,kad->tad", design, coefficients)
        noise = sigma * (rng.normal(size=len(r)) + 1j * rng.normal(size=len(r)))
        for common_arcmin in (0.0, 0.5):
            truth = relative + np.array([common_arcmin, -common_arcmin])[None, None, :]
            clean = np.asarray(predictor(sky, obs, jnp.asarray(truth)))
            data = clean + noise
            for case in ("known_sky", "wrong_fixed_sky", "joint_flux"):
                supplied = (
                    sky
                    if case == "known_sky"
                    else sky._replace(
                        flux_jy=sky.flux_jy * jnp.array([1.0, 1.02, 0.98, 1.02])
                    )
                )
                fit = solve_pointing(
                    supplied,
                    obs,
                    data,
                    times,
                    knots,
                    8,
                    noise_jy=sigma,
                    smoothness=0.01,
                    predictor=predictor,
                    zero_mean_pointing=True,
                    flux_prior_jy=np.asarray(supplied.flux_jy) * 0.05
                    if case == "joint_flux"
                    else None,
                )
                recovered_sky = supplied._replace(flux_jy=jnp.asarray(fit["flux_jy"]))
                model = np.asarray(
                    predictor(recovered_sky, obs, jnp.asarray(fit["offsets_arcmin"]))
                )
                result = {
                    "seed": seed,
                    "case": case,
                    "common_offset_per_axis_arcmin": common_arcmin,
                    "success": fit["success"],
                    "nfev": fit["nfev"],
                    "relative_rmse_arcsec": float(
                        60 * np.sqrt(np.mean((fit["offsets_arcmin"] - relative) ** 2))
                    ),
                    "maximum_mean_arcsec": float(
                        60 * np.max(np.abs(fit["offsets_arcmin"].mean(axis=1)))
                    ),
                    "clean_visibility_rms_mjy": float(
                        1000 * np.sqrt(np.mean(np.abs(model - clean) ** 2))
                    ),
                    "whitened_residual_mean_square": float(
                        np.mean(np.abs(model - data) ** 2) / (2 * sigma**2)
                    ),
                    "flux_fractional_error": (
                        np.asarray(fit["flux_jy"]) / np.asarray(sky.flux_jy) - 1
                    ).tolist(),
                }
                results.append(result)
                print(json.dumps(result), flush=True)
    payload = {
        "fixture_sha256": sha256(raw).hexdigest(),
        "beam": metadata,
        "noise_per_real_component_jy": sigma,
        "limitations": "Known unit gains and exact beam; in-sample residuals, not held-out validation. Common offset is physical, not absorbed into the input sky.",
        "results": results,
    }
    output = ROOT / "outputs/relative-pointing"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
