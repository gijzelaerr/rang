"""Local sky-prior-free observability of an array-common sky-frame shift."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from .pointing import predict, real_stack


def channel_design(
    components, observation, antenna_count, shift_arcmin, beam_axis_ratio=1.0
):
    """Real-stacked DFT matrix for independent source/channel integrated fluxes.

    Columns ordered frequency then source. All antennas share a constant
    sky-frame shift, rotated row-by-row into antenna coordinates. No spectral
    smoothness is imposed. The spectral indices in components are ignored.
    """
    ntime = int(np.max(np.asarray(observation.time_index))) + 1
    angles = np.zeros(ntime)
    for t in range(ntime):
        values = np.asarray(observation.beam_angle_rad)[
            np.asarray(observation.time_index) == t
        ]
        if len(values):
            angles[t] = values[0]
    cos, sin = jnp.cos(jnp.asarray(angles)), jnp.sin(jnp.asarray(angles))
    a = shift_arcmin
    shifted = jnp.stack((cos * a[0] + sin * a[1], -sin * a[0] + cos * a[1]), axis=1)
    offsets = jnp.broadcast_to(shifted[:, None, :], (ntime, antenna_count, 2))

    def flux_model(flux):
        sky = components._replace(flux_jy=flux, spectral_index=jnp.zeros_like(flux))
        return predict(sky, observation, offsets, beam_axis_ratio=beam_axis_ratio)

    responses = jax.jacfwd(flux_model)(jnp.zeros_like(components.flux_jy))
    columns = jnp.concatenate(
        [
            responses * jnp.asarray(np.asarray(observation.frequency_hz) == f)[:, None]
            for f in np.unique(observation.frequency_hz)
        ],
        axis=1,
    )
    return jnp.stack((columns.real, columns.imag), axis=1).reshape(
        2 * columns.shape[0], columns.shape[1]
    )


def solve_sky_locked(
    components,
    observation,
    visibilities,
    antenna_count,
    *,
    noise_jy,
    beam_axis_ratio=1.0,
    fit_beam_axis_ratio=False,
):
    """Profile arbitrary channel fluxes and fit just two shared pointing modes.

    Refuse if the local sky-prior-free audit finds a null direction. Other
    pointing modes and direction-independent gains are held fixed. This is
    an isolated ambiguity experiment, not the full per-antenna solver. When
    fit_beam_axis_ratio is True, jointly fit log axial ratio and check local
    rank at the resulting solution; the initial shape audit is not a veto.
    """
    data = np.asarray(visibilities)
    if data.shape != np.shape(observation.frequency_hz) or not np.isfinite(data).all():
        raise ValueError(
            "visibilities must be a finite vector matching observation rows"
        )
    audit = sky_locked_information(
        components,
        observation,
        antenna_count,
        noise_jy=noise_jy,
        beam_axis_ratio=beam_axis_ratio,
    )
    if audit["observable_common_modes"] < 2 and not fit_beam_axis_ratio:
        return {
            "identifiable": False,
            "success": False,
            "shift_arcmin": None,
            "reason": "Common sky-frame pointing is degenerate with free channel fluxes",
            "audit": audit,
        }
    y = real_stack(jnp.asarray(visibilities)) / noise_jy

    def profiled(parameters):
        a = parameters[:2]
        ratio = jnp.exp(parameters[2]) if fit_beam_axis_ratio else beam_axis_ratio
        design = (
            channel_design(components, observation, antenna_count, a, ratio) / noise_jy
        )
        flux = jnp.linalg.solve(design.T @ design, design.T @ y)
        return design @ flux - y

    fun = jax.jit(profiled)
    jac = jax.jit(jax.jacfwd(profiled))
    fit = least_squares(
        lambda a: np.asarray(fun(a)),
        np.array([0.0, 0.0, np.log(beam_axis_ratio)])
        if fit_beam_axis_ratio
        else np.zeros(2),
        jac=lambda a: np.asarray(jac(a)),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-10:
        return {
            "identifiable": False,
            "success": False,
            "shift_arcmin": None,
            "reason": "Profiled pointing/shape Jacobian is rank deficient at the fitted solution",
            "audit": audit,
        }
    fitted_ratio = float(np.exp(fit.x[2])) if fit_beam_axis_ratio else beam_axis_ratio
    design = np.asarray(
        channel_design(
            components, observation, antenna_count, jnp.asarray(fit.x[:2]), fitted_ratio
        )
    )
    flux = np.linalg.lstsq(
        design, np.asarray(real_stack(jnp.asarray(visibilities))), rcond=None
    )[0]
    return {
        "identifiable": True,
        "success": bool(fit.success),
        "message": fit.message,
        "shift_arcmin": fit.x[:2].tolist(),
        "fitted_beam_axis_ratio": fitted_ratio,
        "fit_beam_axis_ratio": fit_beam_axis_ratio,
        "channel_flux_jy": flux.tolist(),
        "local_sigma_arcsec": (
            60 * np.sqrt(np.diag(np.linalg.inv(fit.jac.T @ fit.jac))[:2])
        ).tolist(),
        "cost": float(fit.cost),
        "audit": audit,
    }


def sky_locked_information(
    components,
    observation,
    antenna_count,
    *,
    noise_jy,
    beam_axis_ratio=1.0,
    gain_model="fixed",
):
    """Project common pointing derivatives off free source/channel fluxes.

    Linearize at zero pointing. Each source has an independent amplitude at
    each observed frequency, with no sky prior. This is a local Fisher audit,
    not a fit or an uncertainty interval for a nonlinear trajectory estimate.
    Optional complex antenna gains are unconstrained nuisances, constant over
    the observation (constant) or independent at each time (per_time). Both
    models share gains across frequency. Linearization is at unit gains;
    least-squares projection handles the redundant gain/sky gauge columns.
    """
    if not jax.config.x64_enabled:
        raise ValueError("enable JAX 64-bit mode before constructing inputs")
    if not np.isfinite(noise_jy) or noise_jy <= 0:
        raise ValueError("noise must be finite and positive")
    if not np.isfinite(beam_axis_ratio) or beam_axis_ratio <= 0:
        raise ValueError("beam axis ratio must be finite and positive")
    if gain_model not in ("fixed", "constant", "per_time"):
        raise ValueError("gain_model must be fixed, constant or per_time")
    ntime = int(np.max(np.asarray(observation.time_index))) + 1
    angles = np.zeros(ntime)
    for t in range(ntime):
        values = np.asarray(observation.beam_angle_rad)[
            np.asarray(observation.time_index) == t
        ]
        if len(values):
            if not np.allclose(values, values[0], atol=1e-12, rtol=0):
                raise ValueError("one common beam angle per time required")
            angles[t] = values[0]
    cos, sin = jnp.cos(jnp.asarray(angles)), jnp.sin(jnp.asarray(angles))

    def common_offsets(a):
        shift = jnp.stack((cos * a[0] + sin * a[1], -sin * a[0] + cos * a[1]), axis=1)
        return jnp.broadcast_to(shift[:, None, :], (ntime, antenna_count, 2))

    def forward(a, flux):
        return (
            real_stack(
                predict(
                    components._replace(flux_jy=flux),
                    observation,
                    common_offsets(a),
                    beam_axis_ratio=beam_axis_ratio,
                )
            )
            / noise_jy
        )

    pointing = np.asarray(
        jax.jacfwd(forward, argnums=0)(jnp.zeros(2), components.flux_jy)
    )
    source = np.asarray(
        jax.jacfwd(forward, argnums=1)(jnp.zeros(2), components.flux_jy)
    )
    frequency = np.asarray(observation.frequency_hz)
    nuisance = np.concatenate(
        [source * np.repeat(frequency == f, 2)[:, None] for f in np.unique(frequency)],
        axis=1,
    )
    sky_count = nuisance.shape[1]
    if gain_model != "fixed":
        visibility = (
            np.asarray(
                predict(
                    components,
                    observation,
                    common_offsets(jnp.zeros(2)),
                    beam_axis_ratio=beam_axis_ratio,
                )
            )
            / noise_jy
        )
        p, q = np.asarray(observation.antenna1), np.asarray(observation.antenna2)
        time = np.asarray(observation.time_index)
        groups = (
            [np.ones(len(time), dtype=bool)]
            if gain_model == "constant"
            else [time == t for t in np.unique(time)]
        )
        columns = []
        for group in groups:
            for antenna in range(antenna_count):
                first, second = (
                    (p == antenna).astype(float),
                    (q == antenna).astype(float),
                )
                # g_p conjugate(g_q): log amplitudes add, phases subtract.
                for derivative in (first + second, 1j * (first - second)):
                    value = visibility * group * derivative
                    columns.append(np.stack((value.real, value.imag), axis=1).ravel())
        nuisance = np.column_stack((nuisance, *columns))
    residual = pointing - nuisance @ np.linalg.lstsq(nuisance, pointing, rcond=None)[0]
    singular = np.linalg.svd(residual, compute_uv=False)
    raw_singular = np.linalg.svd(pointing, compute_uv=False)
    tolerance = raw_singular[0] * 1e-10
    rank = int(np.sum(singular > tolerance))
    crlb = None
    if rank == 2:
        crlb = (60 * np.sqrt(np.diag(np.linalg.inv(residual.T @ residual)))).tolist()
    return {
        "beam_axis_ratio": beam_axis_ratio,
        "noise_jy_per_component": noise_jy,
        "sky_nuisance_parameters": sky_count,
        "gain_model": gain_model,
        "gain_nuisance_parameters": nuisance.shape[1] - sky_count,
        "observable_common_modes": rank,
        "singular_values_per_arcmin": singular.tolist(),
        "known_sky_singular_values_per_arcmin": raw_singular.tolist(),
        "retained_derivative_norm_fraction": float(
            np.linalg.norm(residual) / max(np.linalg.norm(pointing), 1e-30)
        ),
        "local_crlb_arcsec": crlb,
    }
