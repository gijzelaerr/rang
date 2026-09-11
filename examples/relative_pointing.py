"""Controlled zero-mean pointing recovery with known gains and a katbeam beam."""

import argparse
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rangtoy import build
from rangtoy.beams import beam_table, load_katbeam, make_beam_predictor
from rangtoy.pointing import Observation, component_list, solve_pointing, spline_design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    parser.add_argument(
        "--missing-source-jy",
        type=float,
        default=0.0,
        help="Inject an unmodelled point source at direction cosines (0.003,-0.004)",
    )
    parser.add_argument(
        "--gain-per-channel",
        action="store_true",
        help="Fit independent time-smooth gains per frequency to the same injected achromatic gains",
    )
    parser.add_argument(
        "--beam-width-error",
        type=float,
        default=0.0,
        help="Fractional true FWHM change absent from fitted beam",
    )
    parser.add_argument(
        "--pointing-ripple-arcmin",
        type=float,
        default=0.0,
        help="Out-of-family sinusoidal pointing amplitude",
    )
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
    if not np.isfinite(args.missing_source_jy) or args.missing_source_jy < 0:
        parser.error("missing source flux must be finite and nonnegative")
    if args.gain_per_channel and not args.joint_gains:
        parser.error("--gain-per-channel requires --joint-gains")
    if not np.isfinite(args.beam_width_error) or args.beam_width_error <= -1:
        parser.error("beam width error must be finite and greater than -1")
    if not np.isfinite(args.pointing_ripple_arcmin) or args.pointing_ripple_arcmin < 0:
        parser.error("pointing ripple must be finite and nonnegative")
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
    true_sky = sky
    if args.missing_source_jy:
        true_sky = component_list(
            np.r_[s[:, 0], 0.003],
            np.r_[s[:, 1], -0.004],
            np.r_[s[:, 2], args.missing_source_jy],
            np.r_[s[:, 3], -0.7],
        )
    table, metadata = load_katbeam("H")
    predictor = make_beam_predictor(table, metadata=metadata)
    true_predictor = make_beam_predictor(
        beam_table(
            table.frequency_hz,
            table.squint_rad,
            table.fwhm_rad * (1 + args.beam_width_error),
        )
    )
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
        ripple = args.pointing_ripple_arcmin * np.sin(
            2 * np.pi * times[:, None, None] / 5400 + np.arange(16).reshape(1, 8, 2)
        )
        relative += ripple - ripple.mean(axis=1, keepdims=True)
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
            clean = np.asarray(true_predictor(true_sky, obs, jnp.asarray(truth)))
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
                started = perf_counter()
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
                    gain_per_channel=args.gain_per_channel,
                    flux_prior_jy=np.asarray(supplied.flux_jy) * 0.05
                    if case in ("joint_flux", "gain_sky_only")
                    else None,
                )
                solve_seconds = perf_counter() - started
                recovered_sky = supplied._replace(flux_jy=jnp.asarray(fit["flux_jy"]))
                model = np.asarray(
                    predictor(recovered_sky, obs, jnp.asarray(fit["offsets_arcmin"]))
                )
                recovered_gains = fit["gains"]
                if args.gain_per_channel:
                    fi = np.searchsorted(
                        fit["gain_frequencies_hz"], np.asarray(obs.frequency_hz)
                    )
                    recovered_gains = recovered_gains[np.asarray(obs.time_index), fi]
                    model = (
                        model
                        * recovered_gains[np.arange(len(r)), obs.antenna1]
                        * recovered_gains[np.arange(len(r)), obs.antenna2].conj()
                    )
                else:
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
                    "solve_seconds_including_compilation": solve_seconds,
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
        "gain_per_channel": args.gain_per_channel,
        "true_fractional_beam_width_error": args.beam_width_error,
        "pointing_ripple_arcmin": args.pointing_ripple_arcmin,
        "missing_source_jy": args.missing_source_jy,
        "held_out": args.held_out,
        "train_rows": int((~test).sum()),
        "test_rows": int(test.sum()),
        "limitations": "Synthetic beam/spline model, with deliberate width and fast-ripple mismatch when the corresponding settings are nonzero. Held-out times have index modulo 4 equal to 2, at all baselines/frequencies (interpolation, not extrapolation); hyperparameters fixed beforehand. Injected gains are achromatic; fitted gains are independent per frequency when gain_per_channel is true, otherwise achromatic. Gain priors are 0.1 in log-amplitude/radians. Absolute flux scale is prior-dependent. Historical pointing records are not injected. Timings include compilation and are not a controlled performance benchmark.",
        "results": results,
    }
    output = ROOT / (
        "outputs/joint-gain-pointing"
        if args.joint_gains
        else "outputs/relative-pointing"
    )
    if args.held_out:
        output = output.with_name(output.name + "-heldout")
    if args.gain_per_channel:
        output = output.with_name(output.name + "-channelgains")
    if args.missing_source_jy:
        output = output.with_name(output.name + f"-missing{args.missing_source_jy:g}")
    if args.beam_width_error or args.pointing_ripple_arcmin:
        output = output.with_name(
            output.name
            + f"-width{args.beam_width_error:g}-ripple{args.pointing_ripple_arcmin:g}"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
