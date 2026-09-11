"""Component DFT and smooth pointing inference; optional JAX reference path.

Enable JAX 64-bit mode in the application before creating arrays. Importing
this module does not change global JAX settings. Pointing is in arcminutes
in the antenna beam tangent plane; other angles are radians.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import OptimizeResult, least_squares

C = 299_792_458.0
ARCMIN = np.pi / (180 * 60)


class Components(NamedTuple):
    """Point components: direction cosines, integrated Jy, spectral index."""

    lmn: jax.Array
    flux_jy: jax.Array
    spectral_index: jax.Array
    reference_hz: jax.Array


class Observation(NamedTuple):
    """One row per visibility, frequencies flattened; baseline sign p minus q.

    time_index indexes unique solution times. beam_angle_rad is the known
    sky-to-antenna rotation at each row, e.g. parallactic angle.
    """

    uvw_m: jax.Array
    frequency_hz: jax.Array
    antenna1: jax.Array
    antenna2: jax.Array
    time_index: jax.Array
    beam_angle_rad: jax.Array


def component_list(l, m, flux_jy, spectral_index=-0.7, reference_hz=1.28e9):
    """Convert direction cosines and integrated component fluxes to arrays.

    No 1/n factor: inputs are integrated fluxes, not surface brightness.
    Extended emission may be represented by sufficiently resolved components.
    """
    l, m, flux = np.broadcast_arrays(l, m, flux_jy)
    alpha = np.broadcast_to(spectral_index, l.shape)
    if l.ndim != 1:
        raise ValueError("components must be one-dimensional arrays")
    if not all(np.isfinite(a).all() for a in (l, m, flux, alpha)):
        raise ValueError("component values must be finite")
    if np.any(l * l + m * m >= 1):
        raise ValueError("components must be inside the forward hemisphere")
    if not np.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("reference frequency must be positive and finite")
    return Components(
        jnp.asarray(np.column_stack((l, m, np.sqrt(1 - l * l - m * m)))),
        jnp.asarray(flux, dtype=float),
        jnp.asarray(alpha, dtype=float),
        jnp.asarray(reference_hz),
    )


def image_components(image_jy_per_pixel, l, m, **kwargs):
    """Convert Jy/pixel image to components; retain negative, omit zero pixels.

    l and m must have the image shape. FITS/WCS and Jy/beam conversion are not
    implicit: the caller must provide integrated pixel flux and coordinates.
    """
    image = np.asarray(image_jy_per_pixel)
    if image.shape != np.shape(l) or image.shape != np.shape(m):
        raise ValueError("image and coordinate grids must have equal shapes")
    if not np.isfinite(image).all():
        raise ValueError("image values must be finite")
    keep = image != 0
    return component_list(
        np.asarray(l)[keep], np.asarray(m)[keep], image[keep], **kwargs
    )


@jax.jit
def predict(
    components,
    observation,
    offsets_arcmin,
    dish_diameter_m=13.5,
    beam_axis_ratio=1.0,
    beam_quartic=0.0,
):
    """Scalar DFT with full w phase and differentiable Gaussian voltage beams.

    offsets_arcmin: (time, antenna, 2). Geometric-mean power FWHM is
    1.02 lambda / D. beam_axis_ratio is FWHM_y/FWHM_x, keeping beam area fixed.
    Optional beam_quartic adds -beam_quartic * z**2 to log voltage, where
    z is the Gaussian exponent magnitude. For nonzero values the quoted
    FWHM is only the Gaussian-core scale, not the actual beam FWHM.
    Memory is O(visibility rows * components); this reference is for small
    problems. Chunk prediction externally for larger component lists.
    """
    obs = observation
    lm = components.lmn[:, :2]
    cos = jnp.cos(obs.beam_angle_rad)[:, None]
    sin = jnp.sin(obs.beam_angle_rad)[:, None]
    x = cos * lm[:, 0] + sin * lm[:, 1]
    y = -sin * lm[:, 0] + cos * lm[:, 1]
    fwhm = 1.02 * C / obs.frequency_hz / dish_diameter_m

    def beam(antenna):
        offset = offsets_arcmin[obs.time_index, antenna] * ARCMIN
        radius2 = (
            beam_axis_ratio * (x - offset[:, 0, None]) ** 2
            + (y - offset[:, 1, None]) ** 2 / beam_axis_ratio
        )
        z = 2 * jnp.log(2.0) * radius2 / fwhm[:, None] ** 2
        return jnp.exp(-z - beam_quartic * z**2)

    direction = components.lmn - jnp.array([0.0, 0.0, 1.0])
    phase = -2 * jnp.pi * (obs.uvw_m @ direction.T) * obs.frequency_hz[:, None] / C
    flux = (
        components.flux_jy
        * (obs.frequency_hz[:, None] / components.reference_hz)
        ** components.spectral_index
    )
    return jnp.sum(
        flux * beam(obs.antenna1) * beam(obs.antenna2) * jnp.exp(1j * phase), axis=1
    )


def real_stack(values):
    """Interleaved real/imaginary components for real-parameter derivatives."""
    return jnp.stack((values.real, values.imag), axis=-1).reshape(-1)


def spline_design(times_s, knots_s):
    """Natural cubic interpolation and exact integrated-curvature penalty.

    Time is normalized by the knot span. ||P c||² integrates squared second
    derivative in normalized time. Two-point Gauss quadrature is exact for
    the squared piecewise-linear second derivative of a cubic spline.
    """
    times, knots = np.asarray(times_s, float), np.asarray(knots_s, float)
    if times.ndim != 1 or knots.ndim != 1 or times.size == 0 or knots.size < 2:
        raise ValueError("supply nonempty times and at least two knots")
    if not np.isfinite(times).all() or not np.isfinite(knots).all():
        raise ValueError("times and knots must be finite")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(knots) <= 0):
        raise ValueError("times and knots must be strictly increasing")
    if times[0] < knots[0] or times[-1] > knots[-1]:
        raise ValueError("solution times must be inside the knot span")
    span = knots[-1] - knots[0]
    u = (knots - knots[0]) / span
    basis = CubicSpline(u, np.eye(len(knots)), bc_type="natural")
    midpoint, half = (u[:-1] + u[1:]) / 2, np.diff(u) / 2
    nodes = np.column_stack(
        (midpoint - half / np.sqrt(3), midpoint + half / np.sqrt(3))
    )
    penalty = basis(nodes.ravel(), 2) * np.sqrt(np.repeat(half, 2))[:, None]
    return basis((times - knots[0]) / span), penalty


def solve_pointing(
    components,
    observation,
    visibilities,
    times_s,
    knots_s,
    antenna_count,
    *,
    noise_jy,
    smoothness=1.0,
    offset_prior_arcmin=3.0,
    flux_prior_jy=None,
    spectral_index_prior=None,
    minimum_mode_information=None,
    beam_axis_ratio=1.0,
    max_nfev=100,
):
    """Fit smooth offsets, optionally jointly fitting component flux densities.

    flux_prior_jy is a scalar or per-source Gaussian prior sigma around the
    supplied flux. Zero fixes that source; None holds the entire sky fixed.
    spectral_index_prior similarly enables power-law index corrections.
    Positions and beam shape remain fixed. Additive flux corrections
    support signed CLEAN components; positivity is not imposed.

    Minimize whitened residuals + smoothness * integrated curvature² + a
    zero-centred knot-value prior. Curvature leaves constant/linear drift
    unconstrained; the proper offset prior regularizes those modes. Noise
    sigma applies separately to real and imaginary components. This is a
    penalized point estimate: sky/beam mismatch can bias the inferred motion.
    """
    if not jax.config.x64_enabled:
        raise ValueError("enable JAX 64-bit mode before creating input arrays")
    if not np.isfinite(beam_axis_ratio) or beam_axis_ratio <= 0:
        raise ValueError("beam axis ratio must be finite and positive")
    if not isinstance(antenna_count, int) or antenna_count < 2:
        raise ValueError("antenna_count must be an integer >= 2")
    if not np.isfinite(smoothness) or smoothness < 0:
        raise ValueError("smoothness must be finite and nonnegative")
    if not np.isfinite(offset_prior_arcmin) or offset_prior_arcmin <= 0:
        raise ValueError("offset prior must be positive and finite")
    design, penalty = spline_design(times_s, knots_s)
    obs = Observation(*(jnp.asarray(a) for a in observation))
    data = np.asarray(visibilities)
    n = data.size
    if data.shape != (n,) or n == 0 or not np.isfinite(data).all():
        raise ValueError("visibilities must be a nonempty finite vector")
    if np.shape(obs.uvw_m) != (n, 3):
        raise ValueError("uvw_m must have shape (visibility_count, 3)")
    for a in obs[1:]:
        if np.shape(a) != (n,) or not np.isfinite(a).all():
            raise ValueError("observation columns must be finite row vectors")
    if not np.isfinite(obs.uvw_m).all() or np.any(np.asarray(obs.frequency_hz) <= 0):
        raise ValueError("finite coordinates and positive frequencies required")
    for indices, limit in [
        (obs.antenna1, antenna_count),
        (obs.antenna2, antenna_count),
        (obs.time_index, len(times_s)),
    ]:
        values = np.asarray(indices)
        if not np.issubdtype(values.dtype, np.integer) or np.any(
            (values < 0) | (values >= limit)
        ):
            raise ValueError("antenna/time indices must be integers within range")
    sigma = np.broadcast_to(np.asarray(noise_jy, float), (n,))
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("noise_jy must be finite and positive")
    design_jax, penalty_jax = jnp.asarray(design), jnp.asarray(penalty)
    shape = (len(knots_s), antenna_count, 2)
    pointing_size = int(np.prod(shape))
    flux_sigma = np.broadcast_to(
        np.asarray(0.0 if flux_prior_jy is None else flux_prior_jy, float),
        np.shape(components.flux_jy),
    )
    if not np.isfinite(flux_sigma).all() or np.any(flux_sigma < 0):
        raise ValueError("flux prior sigmas must be finite and nonnegative")
    free = np.flatnonzero(flux_sigma > 0)
    alpha_sigma = np.broadcast_to(
        np.asarray(
            0.0 if spectral_index_prior is None else spectral_index_prior, float
        ),
        np.shape(components.spectral_index),
    )
    if not np.isfinite(alpha_sigma).all() or np.any(alpha_sigma < 0):
        raise ValueError("spectral index prior sigmas must be finite and nonnegative")
    alpha_free = np.flatnonzero(alpha_sigma > 0)
    flux_end = pointing_size + len(free)
    total_size = flux_end + len(alpha_free)

    def sky(flat):
        flux = components.flux_jy.at[free].add(
            flat[pointing_size:flux_end] * flux_sigma[free]
        )
        alpha = components.spectral_index.at[alpha_free].add(
            flat[flux_end:] * alpha_sigma[alpha_free]
        )
        return components._replace(flux_jy=flux, spectral_index=alpha)

    def residual(flat):
        coefficients = flat[:pointing_size].reshape(shape)
        offsets = jnp.einsum("tk,kad->tad", design_jax, coefficients)
        model = predict(sky(flat), obs, offsets, beam_axis_ratio=beam_axis_ratio)
        curvature = jnp.einsum("rk,kad->rad", penalty_jax, coefficients)
        return jnp.concatenate(
            (
                real_stack((model - data) / sigma),
                jnp.sqrt(smoothness) * curvature.reshape(-1),
                flat[:pointing_size] / offset_prior_arcmin,
                flat[pointing_size:],
            )
        )

    residual_jit = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    transform = np.eye(total_size)
    mode_information = None
    if minimum_mode_information is not None:
        if not np.isfinite(minimum_mode_information) or minimum_mode_information < 0:
            raise ValueError("minimum mode information must be finite and nonnegative")
        initial_jac = np.asarray(jacobian(np.zeros(total_size)))
        p, s = (
            initial_jac[: 2 * n, :pointing_size],
            initial_jac[: 2 * n, pointing_size:],
        )
        information = p.T @ p
        if s.shape[1]:
            cross = p.T @ s
            information -= cross @ np.linalg.solve(
                s.T @ s + np.eye(s.shape[1]), cross.T
            )
        prior_jac = initial_jac[2 * n :, :pointing_size]
        prior_precision = prior_jac.T @ prior_jac
        values, vectors = np.linalg.eigh(prior_precision)
        whitening = (vectors / np.sqrt(values)) @ vectors.T
        relative = whitening @ information @ whitening
        mode_information, rotation = np.linalg.eigh((relative + relative.T) / 2)
        retained = mode_information >= minimum_mode_information
        basis = whitening @ rotation[:, retained]
        transform = np.zeros((total_size, basis.shape[1] + total_size - pointing_size))
        transform[:pointing_size, : basis.shape[1]] = basis
        transform[pointing_size:, basis.shape[1] :] = np.eye(total_size - pointing_size)

    # Selection is frozen at zero offsets and the supplied sky, without test
    # data. It is a prior-dependent truncated-information baseline, not a
    # claim of nonlinear protection or a novel inference method.
    if transform.shape[1]:
        result = least_squares(
            lambda p: np.asarray(residual_jit(transform @ p)),
            np.zeros(transform.shape[1]),
            jac=lambda p: np.asarray(jacobian(transform @ p)) @ transform,
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
    else:
        residual_zero = np.asarray(residual_jit(np.zeros(total_size)))
        result = OptimizeResult(
            x=np.zeros(0),
            jac=np.zeros((len(residual_zero), 0)),
            success=True,
            message="All pointing modes frozen; sky fixed",
            nfev=1,
            cost=float(residual_zero @ residual_zero / 2),
            optimality=0.0,
        )
    full_solution = transform @ result.x
    coefficients = full_solution[:pointing_size].reshape(shape)
    singular = np.linalg.svd(result.jac[: 2 * n], compute_uv=False)
    cutoff = (
        (singular[0] if singular.size else 0)
        * max(2 * n, result.x.size)
        * np.finfo(float).eps
    )
    rank = int(np.sum(singular > cutoff))
    return {
        "flux_jy": np.asarray(sky(jnp.asarray(full_solution)).flux_jy),
        "spectral_index": np.asarray(sky(jnp.asarray(full_solution)).spectral_index),
        "spectral_index_prior": alpha_sigma.copy(),
        "retained_pointing_modes": transform.shape[1] - total_size + pointing_size,
        "mode_information": mode_information,
        "flux_prior_jy": flux_sigma.copy(),
        "offsets_arcmin": np.einsum("tk,kad->tad", design, coefficients),
        "knot_offsets_arcmin": coefficients,
        "knots_s": np.asarray(knots_s),
        "times_s": np.asarray(times_s),
        "success": bool(result.success),
        "message": result.message,
        "nfev": result.nfev,
        "cost": result.cost,
        "optimality": result.optimality,
        "data_jacobian_rank": rank,
        "parameter_count": int(result.x.size),
        "data_singular_values": singular,
    }
