"""Finite Gaussian gauge and non-Gaussian beam identifiability controls."""

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from rangtoy import build
from rangtoy.gauge import gaussian_gauge, predict_channels
from rangtoy.observability import sky_locked_information
from rangtoy.pointing import Observation, component_list


def main():
    jax.config.update("jax_enable_x64", True)
    fixture = json.loads(
        subprocess.check_output([str(build()), "--pointing-full-reference"])
    )
    rows, sources = np.asarray(fixture["rows"]), np.asarray(fixture["sources"])
    sky = component_list(sources[:, 0], sources[:, 1], sources[:, 2], sources[:, 3])
    obs = Observation(
        jnp.asarray(rows[:, :3]),
        jnp.asarray(rows[:, 3]),
        *(jnp.asarray(rows[:, i], dtype=int) for i in (4, 5, 6)),
        jnp.asarray(rows[:, 7]),
    )
    rng = np.random.default_rng(61)
    time = np.linspace(0, 2 * np.pi, 24)
    offsets = rng.normal(0, 0.3, (1, 8, 2)) + 0.2 * np.sin(
        time[:, None, None] + rng.uniform(0, 6, (1, 8, 2))
    )
    flux = np.asarray(sky.flux_jy)[None, :] * rng.uniform(0.8, 1.2, (4, 4))
    gains = np.exp(
        rng.normal(0, 0.03, (24, 8, 4)) + 1j * rng.normal(0, 0.1, (24, 8, 4))
    )
    finite, audits = [], []
    for ratio in (1.0, 1.1):
        changed = gaussian_gauge(obs, sky, offsets, flux, gains, [0.7, -0.4], ratio)
        for quartic in (0.0, 0.01, 0.1):
            before = np.asarray(
                predict_channels(sky, obs, offsets, flux, gains, ratio, quartic)
            )
            after = np.asarray(
                predict_channels(
                    sky,
                    obs,
                    changed["offsets_arcmin"],
                    changed["channel_flux_jy"],
                    changed["antenna_gains"],
                    ratio,
                    quartic,
                )
            )
            row = {
                "axis_ratio": ratio,
                "quartic": quartic,
                "max_visibility_difference_jy": float(np.max(np.abs(after - before))),
                "rms_difference_jy_per_component": float(
                    np.sqrt(np.mean(np.abs(after - before) ** 2) / 2)
                ),
                "pointing_change_rms_arcsec": float(
                    60 * np.sqrt(np.mean(changed["common_shift_arcmin"] ** 2))
                ),
            }
            finite.append(row)
            print(json.dumps(row), flush=True)
            audit = sky_locked_information(
                sky,
                obs,
                8,
                noise_jy=0.001,
                beam_axis_ratio=ratio,
                beam_quartic=quartic,
                gain_model="per_time_channel",
                differential_pointing=True,
                common_time_variation=True,
            )
            audits.append(audit)
            print(json.dumps(audit), flush=True)
    anchors = []
    for fixed_sources in ([], [0], [0, 1], [0, 1, 2]):
        audit = sky_locked_information(
            sky,
            obs,
            8,
            noise_jy=0.001,
            beam_axis_ratio=1.1,
            gain_model="per_time_channel",
            differential_pointing=True,
            common_time_variation=True,
            fixed_flux_sources=fixed_sources,
        )
        anchors.append(audit)
        print(json.dumps(audit), flush=True)
    result = {
        "seed": 61,
        "complex_rows": len(rows),
        "shift_parameter_arcmin": [0.7, -0.4],
        "finite_transformations": finite,
        "local_audits": audits,
        "flux_anchor_audits": anchors,
    }
    output = ROOT / "outputs/gaussian-gauge"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
