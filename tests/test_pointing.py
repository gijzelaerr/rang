"""JAX DFT, derivative, spline and recovery controls."""

import json
import subprocess

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rangtoy import build
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
