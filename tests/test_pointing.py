"""JAX DFT, derivative, spline and recovery controls."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rangtoy import build
from rangtoy.observability import (
    channel_design,
    sky_locked_information,
    solve_sky_locked,
)
from rangtoy.pointing import (
    Observation,
    component_list,
    image_components,
    predict,
    real_stack,
    solve_pointing,
    spline_design,
)


@pytest.fixture(scope="module")
def reference():
    fixture = json.loads(
        subprocess.check_output([str(build()), "--pointing-reference"])
    )
    rows = np.asarray(fixture["rows"])
    sky = np.asarray(fixture["sources"])
    obs = Observation(
        jnp.asarray(rows[:, :3]),
        jnp.asarray(rows[:, 3]),
        *(jnp.asarray(rows[:, i], dtype=int) for i in (4, 5, 6)),
        jnp.asarray(rows[:, 7]),
    )
    components = component_list(sky[:, 0], sky[:, 1], sky[:, 2], sky[:, 3])
    return fixture, components, obs


def test_dft_and_autodiff_agree_with_rust(reference):
    fixture, components, obs = reference
    theta = jnp.asarray(fixture["pointing_arcmin"])

    def forward(flat):
        offsets = jnp.broadcast_to(flat.reshape(1, 8, 2), (24, 8, 2))
        return real_stack(predict(components, obs, offsets))

    np.testing.assert_allclose(
        forward(theta), fixture["vis_re_im"], atol=2e-9, rtol=2e-9
    )
    jac = jax.jacfwd(forward)(theta)
    np.testing.assert_allclose(jac, fixture["jacobian"], atol=1e-10, rtol=1e-7)
    step = np.zeros(16)
    step[3] = 1e-4
    finite = (forward(theta + step) - forward(theta - step)) / 2e-4
    np.testing.assert_allclose(jac[:, 3], finite, atol=1e-10, rtol=1e-6)


def test_conjugacy_and_w_phase(reference):
    _, components, obs = reference
    offsets = jnp.zeros((24, 8, 2))
    reversed_obs = obs._replace(
        uvw_m=-obs.uvw_m, antenna1=obs.antenna2, antenna2=obs.antenna1
    )
    np.testing.assert_allclose(
        predict(components, reversed_obs, offsets),
        predict(components, obs, offsets).conj(),
        atol=1e-12,
    )
    flat = obs._replace(uvw_m=obs.uvw_m.at[:, 2].set(0))
    assert (
        np.max(
            np.abs(
                predict(components, flat, offsets) - predict(components, obs, offsets)
            )
        )
        > 1e-3
    )


def test_image_flux_and_spline_null_space():
    c = image_components([[1.0, 0], [-0.2, 0.5]], np.zeros((2, 2)), np.zeros((2, 2)))
    assert float(c.flux_jy.sum()) == pytest.approx(1.3)
    empty = image_components(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    assert empty.flux_jy.size == 0
    times = np.linspace(0, 600, 31)
    knots = np.array([0, 100, 350, 600])
    design, penalty = spline_design(times, knots)
    np.testing.assert_allclose(design @ knots, times, atol=1e-12)
    np.testing.assert_allclose(penalty @ knots, 0, atol=1e-10)
    scaled, p_scaled = spline_design(times / 60, knots / 60)
    np.testing.assert_allclose(design, scaled, atol=1e-12)
    np.testing.assert_allclose(penalty, p_scaled, atol=1e-12)
    assert np.linalg.norm(penalty @ [0, 1, -1, 0]) > 1


def test_smooth_pointing_recovery(reference):
    _, components, obs = reference
    times = np.linspace(0, 21600, 24)
    knots = np.linspace(0, 21600, 4)
    design, _ = spline_design(times, knots)
    coefficients = np.random.default_rng(11).normal(0, 0.3, (4, 8, 2))
    truth = np.einsum("tk,kad->tad", design, coefficients)
    data = predict(components, obs, jnp.asarray(truth))
    fit = solve_pointing(
        components, obs, data, times, knots, 8, noise_jy=1e-5, smoothness=0.01
    )
    assert fit["success"], fit["message"]
    assert np.sqrt(np.mean((fit["offsets_arcmin"] - truth) ** 2)) < 1e-3


@pytest.mark.parametrize("knots", [[0, 0, 1], [0], [0, np.nan]])
def test_invalid_knots(knots):
    with pytest.raises(ValueError):
        spline_design([0, 1], knots)


def test_joint_flux_recovers_pointing_with_wrong_sky(reference):
    _, components, obs = reference
    times = np.linspace(0, 21600, 24)
    knots = np.linspace(0, 21600, 4)
    design, _ = spline_design(times, knots)
    truth = np.einsum(
        "tk,kad->tad", design, np.random.default_rng(19).normal(0, 0.3, (4, 8, 2))
    )
    data = predict(components, obs, jnp.asarray(truth))
    wrong = components._replace(
        flux_jy=components.flux_jy * jnp.array([1, 1.02, 0.98, 1.02])
    )
    fixed = solve_pointing(
        wrong, obs, data, times, knots, 8, noise_jy=1e-4, smoothness=0.01
    )
    joint = solve_pointing(
        wrong,
        obs,
        data,
        times,
        knots,
        8,
        noise_jy=1e-4,
        smoothness=0.01,
        flux_prior_jy=[0, 0.035, 0.025, 0.015],
    )
    assert joint["success"], joint["message"]
    assert np.sqrt(np.mean((fixed["offsets_arcmin"] - truth) ** 2)) > 0.2
    assert np.sqrt(np.mean((joint["offsets_arcmin"] - truth) ** 2)) < 0.002
    np.testing.assert_allclose(joint["flux_jy"], components.flux_jy, atol=1e-5)
    assert joint["flux_jy"][0] == float(components.flux_jy[0])
    assert joint["parameter_count"] == fixed["parameter_count"] + 3


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf])
def test_invalid_flux_prior(reference, sigma):
    _, components, obs = reference
    with pytest.raises(ValueError, match="flux prior"):
        solve_pointing(
            components,
            obs,
            np.zeros(len(obs.frequency_hz)),
            np.linspace(0, 21600, 24),
            [0, 21600],
            8,
            noise_jy=0.001,
            flux_prior_jy=sigma,
        )


def test_joint_spectrum_recovery(reference):
    _, components, obs = reference
    times = np.linspace(0, 21600, 24)
    truth = jnp.zeros((24, 8, 2))
    data = predict(components, obs, truth)
    wrong = components._replace(
        flux_jy=components.flux_jy * jnp.array([1, 1.02, 0.98, 1.02]),
        spectral_index=components.spectral_index + jnp.array([0, 0.1, -0.1, 0.1]),
    )
    fit = solve_pointing(
        wrong,
        obs,
        data,
        times,
        [0, 7200, 14400, 21600],
        8,
        noise_jy=1e-5,
        smoothness=0.01,
        flux_prior_jy=[0, 0.035, 0.025, 0.015],
        spectral_index_prior=[0, 0.2, 0.2, 0.2],
    )
    assert fit["success"], fit["message"]
    np.testing.assert_allclose(
        fit["spectral_index"], components.spectral_index, atol=1e-4
    )
    assert np.sqrt(np.mean(fit["offsets_arcmin"] ** 2)) < 0.001


def test_mode_selection_can_freeze_every_pointing_parameter(reference):
    _, components, obs = reference
    times = np.linspace(0, 21600, 24)
    data = predict(components, obs, jnp.zeros((24, 8, 2)))
    fit = solve_pointing(
        components,
        obs,
        data,
        times,
        [0, 21600],
        8,
        noise_jy=0.01,
        minimum_mode_information=1e30,
    )
    assert fit["success"]
    assert fit["retained_pointing_modes"] == fit["parameter_count"] == 0
    assert fit["data_jacobian_rank"] == 0
    np.testing.assert_array_equal(fit["offsets_arcmin"], 0)


def test_full_information_basis_preserves_the_solution(reference):
    _, components, obs = reference
    times = np.linspace(0, 21600, 24)
    data = predict(components, obs, jnp.ones((24, 8, 2)) * 0.2)
    kwargs = {
        "noise_jy": 0.001,
        "flux_prior_jy": [0, 0.035, 0.025, 0.015],
        "spectral_index_prior": [0, 0.2, 0.2, 0.2],
    }
    full = solve_pointing(components, obs, data, times, [0, 21600], 8, **kwargs)
    selected = solve_pointing(
        components,
        obs,
        data,
        times,
        [0, 21600],
        8,
        minimum_mode_information=0,
        **kwargs,
    )
    assert selected["retained_pointing_modes"] == 32
    np.testing.assert_allclose(
        selected["offsets_arcmin"], full["offsets_arcmin"], atol=1e-5
    )


def test_circular_sky_locked_gauge_is_exact(reference):
    _, components, obs = reference
    frequencies = np.unique(obs.frequency_hz)
    a = np.array([0.3, -0.2])
    radians = a * np.pi / (180 * 60)
    k = 2 * np.log(2) / (1.02 * 299792458.0 / frequencies / 13.5) ** 2
    lm = np.asarray(components.lmn[:, :2])
    flux = np.tile(np.asarray(components.flux_jy), (len(frequencies), 1))
    changed = flux * np.exp(
        -4 * k[:, None] * (lm @ radians)[None, :] + 2 * k[:, None] * np.sum(radians**2)
    )
    before = np.asarray(channel_design(components, obs, 8, jnp.zeros(2))) @ flux.ravel()
    after = (
        np.asarray(channel_design(components, obs, 8, jnp.asarray(a))) @ changed.ravel()
    )
    np.testing.assert_allclose(after, before, atol=2e-12, rtol=2e-12)


def test_rotation_and_asymmetry_are_both_needed(reference):
    _, components, obs = reference
    circular = sky_locked_information(components, obs, 8, noise_jy=0.001)
    elliptical = sky_locked_information(
        components, obs, 8, noise_jy=0.001, beam_axis_ratio=1.1
    )
    fixed = sky_locked_information(
        components,
        obs._replace(beam_angle_rad=jnp.zeros_like(obs.beam_angle_rad)),
        8,
        noise_jy=0.001,
        beam_axis_ratio=1.1,
    )
    assert circular["observable_common_modes"] == fixed["observable_common_modes"] == 0
    assert elliptical["observable_common_modes"] == 2
    assert circular["retained_derivative_norm_fraction"] < 1e-12


def test_sky_locked_fit_recovers_without_spectral_priors(reference):
    _, components, obs = reference
    a = np.array([0.3, -0.2])
    flux = np.tile(np.asarray(components.flux_jy), 4) * np.linspace(0.9, 1.1, 16)
    y = np.asarray(channel_design(components, obs, 8, jnp.asarray(a), 1.1)) @ flux
    data = y[::2] + 1j * y[1::2]
    result = solve_sky_locked(
        components, obs, data, 8, noise_jy=0.001, beam_axis_ratio=1.1
    )
    assert result["success"]
    np.testing.assert_allclose(result["shift_arcmin"], a, atol=1e-7)
    inferred_shape = solve_sky_locked(
        components,
        obs,
        data,
        8,
        noise_jy=0.001,
        beam_axis_ratio=1.0,
        fit_beam_axis_ratio=True,
    )
    assert inferred_shape["success"]
    assert inferred_shape["fitted_beam_axis_ratio"] == pytest.approx(1.1, abs=1e-7)
    np.testing.assert_allclose(inferred_shape["shift_arcmin"], a, atol=1e-7)
    refused = solve_sky_locked(
        components, obs, data, 8, noise_jy=0.001, beam_axis_ratio=1
    )
    assert not refused["identifiable"]
    assert refused["shift_arcmin"] is None


def test_gain_nuisances_cannot_increase_pointing_information(reference):
    _, components, obs = reference
    results = [
        sky_locked_information(
            components, obs, 8, noise_jy=0.001, beam_axis_ratio=1.1, gain_model=model
        )
        for model in ("fixed", "constant", "per_time")
    ]
    fractions = [result["retained_derivative_norm_fraction"] for result in results]
    assert fractions[0] >= fractions[1] >= fractions[2]
    assert all(result["observable_common_modes"] == 2 for result in results)
    circular = sky_locked_information(
        components, obs, 8, noise_jy=0.001, gain_model="per_time"
    )
    assert circular["observable_common_modes"] == 0
    with pytest.raises(ValueError, match="gain_model"):
        sky_locked_information(components, obs, 8, noise_jy=0.001, gain_model="typo")
