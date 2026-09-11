"""Integration tests of the scientific controls and Python/Rust boundary."""

import subprocess

import pytest
from rangtoy import build, run_experiment


@pytest.fixture(scope="session")
def binary():
    return build()


def rows(experiment):
    return {row["method"]: row for row in experiment["results"]}


def test_no_error_control_does_not_invent_pointing(binary):
    ex = run_experiment(binary=binary, noise=0, pointing=0, target_flux=0)
    for row in ex["results"]:
        assert abs(row["flux_jy"]) < 1e-10
        assert row["pointing_rmse_arcmin"] < 1e-9
        assert row["heldout_rms_jy"] < 1e-10


def test_recovery_and_no_claim_of_extra_benefit(binary):
    ex = run_experiment(binary=binary)
    r = rows(ex)
    assert ex["visibility_count"] == 2688
    assert ex["max_abs_w"] > 1000
    # Known pointing is an essential independent check on sky discretization.
    assert abs(r["known_pointing"]["flux_jy"] - 0.02) < 0.002
    assert abs(r["joint_sky"]["flux_jy"] - 0.02) < 0.002
    assert r["pointing_only"]["template_transfer"] < 0.9
    assert r["joint_sky"]["template_transfer"] > 0.98
    assert r["protected_modes"]["local_loss"] <= ex["budget"]
    assert r["protected_modes"]["flux_jy"] == pytest.approx(
        r["joint_sky"]["flux_jy"], abs=1e-8
    )
    assert r["joint_sky"]["heldout_rms_jy"] < r["fixed_beam"]["heldout_rms_jy"]


def test_reproducibility_and_seed_changes(binary):
    a = run_experiment(binary=binary, seed=3, noise=0, pointing=0, target_flux=0)
    b = run_experiment(binary=binary, seed=3, noise=0, pointing=0, target_flux=0)
    assert a["results"] == b["results"]
    c = run_experiment(binary=binary, seed=4)
    assert a["results"] != c["results"]


def test_shared_beam_fit_resolves_width_mismatch(binary):
    r = rows(run_experiment(binary=binary, beam_error=0.02))
    assert r["joint_sky"]["flux_jy"] > 0.03
    assert r["joint_sky"]["template_transfer"] > 0.99
    assert abs(r["joint_beam"]["flux_jy"] - 0.02) < 0.002
    assert abs(r["joint_beam"]["fitted_beam_width_error"] - 0.02) < 0.002
    assert len(r["joint_beam"]["pointing_arcmin"]) == 16
    assert r["joint_beam"]["retained_modes"] == 17
    assert r["joint_beam"]["heldout_rms_jy"] < r["joint_sky"]["heldout_rms_jy"]


@pytest.mark.parametrize(
    "option,value",
    [("--noise", "nan"), ("--budget", "-1"), ("--seed", "-1"), ("--pointing", "inf")],
)
def test_invalid_options_are_rejected(binary, option, value):
    result = subprocess.run(
        [str(binary), option, value], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert "rang-toy:" in result.stderr
    assert not result.stdout
