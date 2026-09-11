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
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Hold out every fourth time and add a gain/sky-only baseline",
    )
    parser.add_argument(
        "--joint-gains",
        action="store_true",
        help="Inject and fit smooth achromatic gains",
    )
    args = parser.parse_args()
    if args.held_out and not args.joint_gains:
        parser.error(
            "--held-out requires --joint-gains for the gain/sky-only comparison"
        )
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
    test = (
        np.asarray(obs.time_index) % 4 == 2
        if args.held_out
        else np.zeros(len(r), dtype=bool)
    )
    train_obs = Observation(*(a[~test] for a in obs))
    results = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        coefficients = rng.normal(0, 0.3, (4, 8, 2))
        coefficients -= coefficients.mean(axis=1, keepdims=True)
        relative = np.einsum("tk,kad->tad", design, coefficients)
        noise = sigma * (rng.normal(size=len(r)) + 1j * rng.normal(size=len(r)))
        gains = np.ones((24, 8), dtype=complex)
        if args.joint_gains:
            logamp = design @ rng.normal(0, 0.02, (4, 8))
            phase = design @ rng.normal(0, 0.03, (4, 8))
            phase -= phase[:, :1]
            gains = np.exp(logamp + 1j * phase)
        baseline_gain = (
            gains[obs.time_index, obs.antenna1]
            * gains[obs.time_index, obs.antenna2].conj()
        )
        for common_arcmin in (0.0, 0.5):
            truth = relative + np.array([common_arcmin, -common_arcmin])[None, None, :]
            clean = np.asarray(predictor(sky, obs, jnp.asarray(truth)))
            clean = clean * baseline_gain
            data = clean + noise
            cases = ["known_sky", "wrong_fixed_sky", "joint_flux"]
            if args.held_out:
                cases.append("gain_sky_only")
            for case in cases:
                supplied = (
                    sky
                    if case == "known_sky"
                    else sky._replace(
                        flux_jy=sky.flux_jy * jnp.array([1.0, 1.02, 0.98, 1.02])
                    )
                )
                fit = solve_pointing(
                    supplied,
                    train_obs,
                    data[~test],
                    times,
                    knots,
                    8,
                    noise_jy=sigma,
                    smoothness=0.01,
                    predictor=predictor,
                    zero_mean_pointing=True,
                    fit_pointing=case != "gain_sky_only",
                    gain_prior_sigma=(0.1, 0.1) if args.joint_gains else None,
                    flux_prior_jy=np.asarray(supplied.flux_jy) * 0.05
                    if case in ("joint_flux", "gain_sky_only")
                    else None,
                )
                recovered_sky = supplied._replace(flux_jy=jnp.asarray(fit["flux_jy"]))
                model = np.asarray(
                    predictor(recovered_sky, obs, jnp.asarray(fit["offsets_arcmin"]))
                )
                recovered_gains = fit["gains"]
                model = (
                    model
                    * recovered_gains[obs.time_index, obs.antenna1]
                    * recovered_gains[obs.time_index, obs.antenna2].conj()
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
                if args.held_out:
                    for label, mask in (("train", ~test), ("test", test)):
                        result[label + "_whitened_residual_mean_square"] = float(
                            np.mean(np.abs(model[mask] - data[mask]) ** 2)
                            / (2 * sigma**2)
                        )
                        result[label + "_clean_visibility_rms_mjy"] = float(
                            1000
                            * np.sqrt(np.mean(np.abs(model[mask] - clean[mask]) ** 2))
                        )
                print(json.dumps(result), flush=True)
    payload = {
        "fixture_sha256": sha256(raw).hexdigest(),
        "beam": metadata,
        "noise_per_real_component_jy": sigma,
        "joint_gains": args.joint_gains,
        "held_out": args.held_out,
        "train_rows": int((~test).sum()),
        "test_rows": int(test.sum()),
        "limitations": "Exact beam and matched spline family. When held_out is true, times with index modulo 4 equal to 2 are withheld at all baselines/frequencies (interpolation, not extrapolation); hyperparameters fixed beforehand. Otherwise residuals are in-sample. Gains are known unity unless joint_gains is enabled, then smooth achromatic with 0.1 log-amplitude/radian knot priors. Joint flux absolute scale is prior-dependent. Common offset is physical, not absorbed into the input sky. Historical pointing records are not injected into this synthetic campaign.",
        "results": results,
    }
    output = ROOT / (
        "outputs/joint-gain-pointing"
        if args.joint_gains
        else "outputs/relative-pointing"
    )
    if args.held_out:
        output = output.with_name(output.name + "-heldout")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
