"""Stable beam derivatives, upstream comparisons and predictor integration."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rangtoy import build
from rangtoy.beams import (
    beam_table,
    cosine_taper,
    load_katbeam,
    make_beam_predictor,
    predict_tabulated,
    voltage_beam,
)
from rangtoy.observability import sky_locked_information
from rangtoy.pointing import C, Observation, component_list, predict, solve_pointing


@pytest.fixture(scope="module")
def fixture():
    reference = json.loads(
        subprocess.check_output([str(build()), "--pointing-reference"])
    )
    r, s = np.asarray(reference["rows"]), np.asarray(reference["sources"])
    obs = Observation(
        jnp.asarray(r[:, :3]),
        jnp.asarray(r[:, 3]),
        *(jnp.asarray(r[:, i], dtype=int) for i in (4, 5, 6)),
        jnp.asarray(r[:, 7]),
    )
    return component_list(s[:, 0], s[:, 1], s[:, 2], s[:, 3]), obs


def test_cosine_taper_center_half_power_and_removable_pole():
    scale = 1.1889647809329453
    assert cosine_taper(0.0) == pytest.approx(1.0)
    assert cosine_taper(0.25) == pytest.approx(np.sqrt(0.5), abs=1e-14)
    pole = 0.25 / scale**2
    assert cosine_taper(pole) == pytest.approx(np.pi / 4, abs=1e-14)
    assert jax.grad(cosine_taper)(pole) == pytest.approx(
        -np.pi * scale**2 / 4, abs=1e-12
    )
    assert jax.grad(cosine_taper)(0.0) == pytest.approx((4 - np.pi**2 / 2) * scale**2)
    hessian = jax.hessian(lambda x: cosine_taper(jnp.sum(x**2)))(jnp.zeros(2))
    np.testing.assert_allclose(
        hessian, 2 * (4 - np.pi**2 / 2) * scale**2 * np.eye(2), atol=1e-12
    )
    # Signed sidelobes, rather than sqrt(power), retain the voltage convention.
    assert cosine_taper(4.0) < 0


@pytest.mark.parametrize("pol", ["H", "V"])
def test_beam_matches_pinned_katbeam(pol):
    katbeam = pytest.importorskip("katbeam")
    table, metadata = load_katbeam(pol)
    rng = np.random.default_rng(8)
    x, y = rng.uniform(-2, 2, (6, 25)), rng.uniform(-2, 2, (6, 25))
    frequencies = np.array([900.0, 975.0, 1150.0, 1284.0, 1550.0, 1650.0])
    actual = np.asarray(
        voltage_beam(
            jnp.deg2rad(x), jnp.deg2rad(y), jnp.asarray(frequencies * 1e6), table
        )
    )
    model = katbeam.JimBeam("MKAT-AA-L-JIM-2020")
    evaluate = model.HH if pol == "H" else model.VV
    expected = np.stack([evaluate(x[i], y[i], f) for i, f in enumerate(frequencies)])
    # Upstream 0.1 rounds the radius scale to eight decimal places.
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=2e-9)
    assert metadata["katbeam_version"] == "0.1"


def test_tabulated_gaussian_reproduces_original_dft(fixture):
    sky, obs = fixture
    frequencies = np.unique(obs.frequency_hz)
    width = 1.02 * C / frequencies / 13.5
    ratio = 1.1
    table = beam_table(
        frequencies,
        np.zeros((4, 2)),
        width[:, None] * np.array([1 / np.sqrt(ratio), np.sqrt(ratio)]),
    )
    offsets = jnp.asarray(np.random.default_rng(21).normal(0, 0.4, (24, 8, 2)))
    np.testing.assert_allclose(
        predict_tabulated(sky, obs, offsets, table, "gaussian"),
        predict(sky, obs, offsets, beam_axis_ratio=ratio),
        atol=1e-13,
    )


def test_table_validation_and_no_extrapolation(fixture):
    _, obs = fixture
    with pytest.raises(ValueError, match="strictly increasing"):
        beam_table([1e9, 1e9], np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        beam_table([1e9, 2e9], np.zeros((2, 2)), -np.ones((2, 2)))
    table = beam_table([1e9, 1.5e9], np.zeros((2, 2)), np.ones((2, 2)))
    predictor = make_beam_predictor(table)
    with pytest.raises(ValueError, match="outside"):
        predictor.validate_observation(obs)
    assert np.isnan(
        voltage_beam(jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.array([2e9]), table)
    ).all()


def test_custom_predictor_derivatives_and_solver(fixture):
    pytest.importorskip("katbeam")
    sky, obs = fixture
    table, metadata = load_katbeam()
    predictor = make_beam_predictor(table, metadata=metadata)
    offsets = jnp.ones((24, 8, 2)) * 0.2
    derivative = jax.jacfwd(lambda a: predictor(sky, obs, offsets.at[:, 0, 0].add(a)))(
        0.0
    )
    step = 1e-4
    finite = (
        predictor(sky, obs, offsets.at[:, 0, 0].add(step))
        - predictor(sky, obs, offsets.at[:, 0, 0].add(-step))
    ) / (2 * step)
    np.testing.assert_allclose(derivative, finite, atol=2e-11)
    fitted = solve_pointing(
        sky,
        obs,
        predictor(sky, obs, offsets),
        np.linspace(0, 21600, 24),
        [0, 21600],
        8,
        noise_jy=1e-5,
        offset_prior_arcmin=1e4,
        predictor=predictor,
    )
    assert fitted["success"]
    np.testing.assert_allclose(fitted["offsets_arcmin"], offsets, atol=1e-6)
    with pytest.raises(ValueError, match="cannot be combined"):
        sky_locked_information(
            sky, obs, 8, noise_jy=0.001, predictor=predictor, beam_axis_ratio=1.1
        )


def test_chromatic_shape_lifts_the_gaussian_gauge(fixture):
    pytest.importorskip("katbeam")
    sky, _ = fixture
    reference = json.loads(
        subprocess.check_output([str(build()), "--pointing-full-reference"])
    )
    r = np.asarray(reference["rows"])
    obs = Observation(
        jnp.asarray(r[:, :3]),
        jnp.asarray(r[:, 3]),
        *(jnp.asarray(r[:, i], dtype=int) for i in (4, 5, 6)),
        jnp.asarray(r[:, 7]),
    )
    table, metadata = load_katbeam()
    zero_squint = table._replace(squint_rad=jnp.zeros_like(table.squint_rad))
    kwargs = {
        "noise_jy": 0.001,
        "gain_model": "per_time_channel",
        "differential_pointing": True,
        "common_time_variation": True,
    }
    native = sky_locked_information(
        sky,
        obs,
        8,
        predictor=make_beam_predictor(zero_squint, "gaussian", metadata),
        **kwargs,
    )
    assert native["observable_common_modes"] == 2
    np.testing.assert_allclose(
        native["local_crlb_arcsec"], [8.40284, 8.30909], atol=0.002
    )
    widths = np.asarray(table.fwhm_rad)
    geometric = np.sqrt(np.prod(widths, axis=1))
    fixed = zero_squint._replace(
        fwhm_rad=jnp.asarray(
            geometric[:, None] * np.array([1 / np.sqrt(1.03), np.sqrt(1.03)])
        )
    )
    null = sky_locked_information(
        sky, obs, 8, predictor=make_beam_predictor(fixed, "gaussian"), **kwargs
    )
    assert null["observable_common_modes"] == 0
