"""The catalog's specified motion and characteristic primitives against answers known by construction (written before the code).

step_response (a 2nd-order system's overshoot, rise and 2 % settling, found here by root-finding on its closed form), hysteresis_loop
(a loop of known stiffness and width), ISO 9283 pose and path accuracy (hand-computed AP, RP, AT, RT), Allan deviation (white rate
noise -> slope -1/2 and N = sigma sqrt(dt); a rate random walk -> slope +1/2 and K).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robotdx.diagnostics import FAIL, NOT_RUN, PASS, WARN, Limits, evaluate
from robotdx.diagnostics.stock import (AllanConfig, HysteresisConfig, PathAccuracyConfig, PoseAccuracyConfig, StepResponseConfig)
from robotdx.diagnostics.stock.characteristic import RateSeries
from robotdx.diagnostics.stock.iso9283 import PathCycle, PoseSample

from conftest import tape


def measures(result) -> dict:
    return {m.key: m.value for m in result.measurements}


def bisect(f, a: float, b: float, tol: float = 1e-12) -> float:
    fa = f(a)
    assert fa * f(b) < 0
    while b - a > tol:
        m = 0.5 * (a + b)
        if f(m) * fa > 0:
            a, fa = m, f(m)
        else:
            b = m
    return 0.5 * (a + b)


# ----- step_response: MATLAB stepinfo / ISO 9283 overshoot and stabilisation time --------------------------------------------
WN, ZETA = 10.0, 0.3
WD = WN * math.sqrt(1 - ZETA ** 2)


def second_order(tau):
    tau = np.asarray(tau, float)
    return np.where(tau < 0, 0.0, 1 - np.exp(-ZETA * WN * tau) * (np.cos(WD * tau) + ZETA / math.sqrt(1 - ZETA ** 2) * np.sin(WD * tau)))


def analytic_stepinfo() -> dict:
    y = lambda tau: float(second_order(tau))  # noqa: E731
    peak = math.pi / WD
    rise = bisect(lambda s: y(s) - 0.9, 0.0, peak) - bisect(lambda s: y(s) - 0.1, 0.0, peak)
    k = max(k for k in range(1, 50) if math.exp(-ZETA * WN * k * math.pi / WD) > 0.02)  # the last extremum outside the 2 % band
    settling = bisect(lambda s: abs(y(s) - 1) - 0.02, k * math.pi / WD, (k + 1) * math.pi / WD)
    return {"overshoot": 100 * math.exp(-ZETA * math.pi / math.sqrt(1 - ZETA ** 2)), "rise": rise, "settling": settling, "peak": peak}


def _step_run(response, dt: float = 0.0005):
    """Target 0 -> 1 at t = 1 s, then 1 -> 0.5 at t = 4 s; the response is ``response`` of each step (scaled by its size)."""

    t = np.arange(0.0, 6.0, dt)
    target = np.where(t < 1.0, 0.0, np.where(t < 4.0, 1.0, 0.5))
    q = response(t - 1.0) - 0.5 * response(t - 4.0)
    base = tape(n=t.size)
    return replace(base, t=t, q_rad=q, q_target_rad=target, qd_target_rad_s=np.zeros(t.size))


def test_step_response_of_a_second_order_system_is_its_closed_form():
    oracle = analytic_stepinfo()
    assert oracle["overshoot"] == pytest.approx(37.23, abs=0.01)
    run = _step_run(second_order)
    out = evaluate("stock.step_response", StepResponseConfig(), [run], subject="s")
    m = measures(out)
    assert out.metric == "overshoot" and out.unit == "%" and m["moves"] == 2
    assert out.value == pytest.approx(oracle["overshoot"], abs=0.02)  # both steps, the down one too (sign handled)
    assert m["rise_s"] == pytest.approx(oracle["rise"], abs=5e-4)
    assert m["settling_s"] == pytest.approx(oracle["settling"], abs=5e-4)
    assert out.payload["y"] == [pytest.approx(oracle["overshoot"], abs=0.02)] * 2
    settle = evaluate("stock.step_response", StepResponseConfig(headline="settling", settling=Limits.at_most(1.0)), [run], subject="s")
    assert settle.value == pytest.approx(oracle["settling"], abs=5e-4) and settle.unit == "s" and settle.verdict == FAIL


def test_step_response_of_a_first_order_lag_has_no_overshoot_and_known_times():
    lag = 0.1
    first = lambda tau: np.where(np.asarray(tau) < 0, 0.0, 1 - np.exp(-np.asarray(tau) / lag))  # noqa: E731
    out = evaluate("stock.step_response", StepResponseConfig(overshoot=Limits.at_most(5.0)), [_step_run(first)], subject="s")
    m = measures(out)
    assert out.value == pytest.approx(0.0, abs=1e-9) and out.verdict == PASS
    assert m["rise_s"] == pytest.approx(lag * math.log(9), abs=5e-4)  # 10 -> 90 %
    assert m["settling_s"] == pytest.approx(lag * math.log(50), abs=5e-4)  # into 2 % and stays


def test_step_response_that_never_settles_is_never_a_clean_pass():
    half = lambda tau: np.where(np.asarray(tau) < 0, 0.0, 0.5)  # noqa: E731  (stops halfway: never inside 2 %)
    out = evaluate("stock.step_response", StepResponseConfig(), [_step_run(half)], subject="s")
    assert (out.verdict, out.code) == (WARN, "NotSettled") and measures(out)["unsettled"] == 2


def test_step_response_is_not_run_without_a_target_or_a_move():
    no_target = replace(tape(), q_target_rad=None)
    assert evaluate("stock.step_response", StepResponseConfig(), [no_target], subject="s").code == "NoTarget"
    flat = replace(tape(), q_target_rad=np.zeros(600))
    out = evaluate("stock.step_response", StepResponseConfig(), [flat], subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "NoMoves")


# ----- hysteresis_loop: torque vs angle with the output locked ----------------------------------------------------------------
def _loop(compliance, gap: float = 0.0, flank: float = 0.0, cycles: int = 3, dt: float = 0.001, amplitude: float = 2.0, noise: float = 0.0):
    """Torque a triangle wave +-amplitude; motor angle = compliance curve - (gap/2) dir (friction lags the loading direction)
    + (flank/2) sign(tau) (gear backlash: a dead zone at zero torque)."""

    t = np.arange(0.0, 4.0 * cycles, dt)
    phase = (t / 4.0) % 1.0
    tau = amplitude * np.where(phase < 0.25, 4 * phase, np.where(phase < 0.75, 2 - 4 * phase, 4 * phase - 4))
    direction = np.sign(np.gradient(tau))
    q = compliance(tau) - 0.5 * gap * direction + 0.5 * flank * np.sign(tau)
    q = q + np.random.default_rng(1).normal(0, noise, t.size) if noise else q
    base = tape(n=t.size)
    return replace(base, t=t, q_rad=q, tau_nm=tau, q_target_rad=None, qd_target_rad_s=None)


ARCMIN = 60 * 180 / math.pi


def test_hysteresis_loop_reads_the_width_and_the_stiffness_of_a_linear_loop():
    run = _loop(lambda tau: tau / 200.0, gap=1e-3, noise=1e-6)
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(), [run], subject="s")
    m = measures(out)
    assert out.metric == "gap" and out.unit == "arcmin"
    assert out.value == pytest.approx(1e-3 * ARCMIN, rel=2e-3)
    assert m["stiffness"] == pytest.approx(200.0, rel=2e-3) and m["compliance"] == pytest.approx(5.0, rel=2e-3)
    assert 0 < m["gap_ci_half"] < 0.01  # arcmin: a clean loop is a tight estimate
    stiff = evaluate("stock.hysteresis_loop", HysteresisConfig(headline="stiffness"), [run], subject="s")
    assert stiff.value == pytest.approx(200.0, rel=2e-3) and stiff.unit == "N.m/rad"


def test_hysteresis_loop_reads_a_backlash_dead_zone_and_piecewise_stiffness():
    backlash = _loop(lambda tau: tau / 200.0, flank=2e-3)
    assert evaluate("stock.hysteresis_loop", HysteresisConfig(), [backlash], subject="s").value == pytest.approx(2e-3 * ARCMIN, rel=2e-3)
    both = _loop(lambda tau: tau / 200.0, gap=1e-3, flank=2e-3)  # lost motion = backlash + the friction loop's width
    assert evaluate("stock.hysteresis_loop", HysteresisConfig(), [both], subject="s").value == pytest.approx(3e-3 * ARCMIN, rel=2e-3)

    def stiffening(tau):  # Harmonic Drive's three-slope curve: K1 100, K2 200, K3 400 N.m/rad with breaks at 0.5 and 1 N.m
        a = np.abs(tau)
        return np.sign(tau) * np.where(a < 0.5, a / 100, np.where(a < 1.0, 0.005 + (a - 0.5) / 200, 0.0075 + (a - 1.0) / 400))
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(), [_loop(stiffening, gap=1e-3)], subject="s")
    m = measures(out)
    assert (m["K1"], m["K2"], m["K3"]) == (pytest.approx(100.0, rel=1e-2), pytest.approx(200.0, rel=1e-2), pytest.approx(400.0, rel=1e-2))


def test_hysteresis_loop_leg_median_matches_per_leg_slopes():
    run = _loop(lambda tau: tau / 250.0, gap=5e-4)
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(stiffness="leg_median", headline="compliance"), [run], subject="s")
    assert out.value == pytest.approx(4.0, rel=2e-3) and out.unit == "mrad/N.m" and measures(out)["legs"] == 7  # 0->2, then 6 reversals


def test_hysteresis_loop_is_not_run_without_torque_or_legs():
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(), [replace(tape(), tau_nm=None)], subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "NoChannel")
    still = evaluate("stock.hysteresis_loop", HysteresisConfig(), [tape()], subject="s")  # torque never ramps
    assert (still.verdict, still.code) == (NOT_RUN, "NoLegs")
    one_way = _loop(lambda tau: tau / 200.0, gap=1e-3)
    up = one_way.tau_nm > -10
    rising = np.gradient(one_way.tau_nm) > 0
    only_up = replace(one_way, t=one_way.t[rising & up], q_rad=one_way.q_rad[rising & up], tau_nm=one_way.tau_nm[rising & up],
                      mode=one_way.mode[rising & up])
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(), [only_up], subject="s")
    assert out.verdict == NOT_RUN and out.code == "NeedsBothDirections"


# ----- ISO 9283 pose accuracy (AP) and repeatability (RP) ---------------------------------------------------------------------
def test_pose_accuracy_is_iso_9283_ap_and_rp():
    commanded = (100.0, 0.0, 50.0)
    # P1: four attained points around a barycentre 5 mm off (3, 4, 0); each 1 mm from it -> l = 1, S_l = 0, RP = 1
    p1 = [(103.0 + dx, 4.0 + dy, 50.0) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    # P2: barycentre on the command; distances 1, 1, 3, 3 -> l = 2, S_l = sqrt(4/3), RP = 2 + 3 sqrt(4/3)
    p2 = [(dx, 0.0, 0.0) for dx in (1.0, -1.0, 3.0, -3.0)]
    samples = ([PoseSample("P1", a, commanded, cycle) for cycle, a in enumerate(p1)]
               + [PoseSample("P2", a, (0.0, 0.0, 0.0), cycle) for cycle, a in enumerate(p2)])
    out = evaluate("stock.pose_accuracy", PoseAccuracyConfig(iso_cycles=0), samples, subject="s")
    assert out.metric == "ap" and out.value == pytest.approx(5.0) and out.unit == "mm"
    assert out.payload["x"] == ["P1", "P2"] and out.payload["y"] == [pytest.approx(5.0), pytest.approx(0.0)]
    assert out.payload["rp"] == [pytest.approx(1.0), pytest.approx(2 + 3 * math.sqrt(4 / 3))]
    rp = evaluate("stock.pose_accuracy", PoseAccuracyConfig(headline="rp", iso_cycles=0, rp=Limits.at_most(5.0)), samples, subject="s")
    assert rp.value == pytest.approx(2 + 3 * math.sqrt(4 / 3)) and (rp.verdict, rp.code) == (FAIL, "RpAboveErrorLimit")
    short = evaluate("stock.pose_accuracy", PoseAccuracyConfig(), samples, subject="s")  # 4 cycles, ISO asks for 30
    assert (short.verdict, short.code) == (WARN, "FewerCyclesThanIso9283")


def test_pose_accuracy_orientation_terms():
    abc = [(10.0, 0.0, 0.0), (12.0, 0.0, 0.0), (14.0, 0.0, 0.0)]  # mean 12 vs commanded 10 -> APa 2; S_a 2 -> RPa 6
    samples = [PoseSample("P1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), i, attained_abc_deg=a, commanded_abc_deg=(10.0, 0.0, 0.0)) for i, a in enumerate(abc)]
    m = measures(evaluate("stock.pose_accuracy", PoseAccuracyConfig(iso_cycles=0), samples, subject="s"))
    assert m["ap_a_deg"] == pytest.approx(2.0) and m["rp_a_deg"] == pytest.approx(6.0) and m["ap_b_deg"] == pytest.approx(0.0)


def test_pose_accuracy_is_not_run_on_too_few_cycles():
    samples = [PoseSample("P1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0), PoseSample("P1", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1)]
    out = evaluate("stock.pose_accuracy", PoseAccuracyConfig(), samples, subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "TooFewCycles")


# ----- ISO 9283 path accuracy (AT) and repeatability (RT) ---------------------------------------------------------------------
def _circle(n: int = 200) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n)
    return np.stack([100 * np.cos(a), 100 * np.sin(a), np.zeros(n)], axis=1)


def test_path_accuracy_is_iso_9283_at_and_rt():
    target = _circle()
    offsets = [np.array([0.3, 0.4, 0.1]), np.array([0.3, 0.4, -0.1]), np.array([0.3, 0.4, 0.0])]  # barycentre 0.5 mm off
    cycles = [PathCycle(target + offset, target, run=f"c{i}") for i, offset in enumerate(offsets)]
    out = evaluate("stock.path_accuracy", PathAccuracyConfig(), cycles, subject="s")
    m = measures(out)
    assert out.metric == "at" and out.value == pytest.approx(0.5) and out.unit == "mm"
    # l per cycle = 0.1, 0.1, 0 -> l = 1/15, S_l = sqrt(((1/30)^2 * 2 + (1/15)^2) / 2); RT = l + 3 S_l (the same at every point)
    l_bar = 0.2 / 3
    s_l = math.sqrt((2 * (0.1 - l_bar) ** 2 + l_bar ** 2) / 2)
    assert m["rt"] == pytest.approx(l_bar + 3 * s_l)
    assert m["max"] == pytest.approx(math.sqrt(0.26)) and len(out.payload["x"]) == len(out.payload["y"]) == 200
    tight = evaluate("stock.path_accuracy", PathAccuracyConfig(tolerance_mm=0.4), cycles, subject="s")
    assert (tight.verdict, tight.code) == (FAIL, "AboveErrorLimit")


def test_path_accuracy_rotation_error_and_refusals():
    target = _circle(50)
    angle = 0.002  # rad about z
    quat = np.tile([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], (50, 1))
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
    out = evaluate("stock.path_accuracy", PathAccuracyConfig(), [PathCycle(target, target, quat_wxyz=quat, target_quat_wxyz=identity)], subject="s")
    assert measures(out)["rot_max_mrad"] == pytest.approx(2.0, rel=1e-6) and measures(out).get("rt") is None
    other = evaluate("stock.path_accuracy", PathAccuracyConfig(frame="belt"), [PathCycle(target, target)], subject="s")
    assert (other.verdict, other.code) == (NOT_RUN, "FrameMismatch")
    ragged = evaluate("stock.path_accuracy", PathAccuracyConfig(), [PathCycle(target, target), PathCycle(target[:40], target[:40])], subject="s")
    assert (ragged.verdict, ragged.code) == (NOT_RUN, "MisalignedCycles")


# ----- Allan deviation (IEEE Std 952-1997 Annex C; overlapping estimator) ------------------------------------------------------
def test_allan_deviation_of_white_rate_noise_has_slope_minus_half_and_its_arw():
    rate_hz, sigma = 100.0, 0.01  # rad/s white noise at 100 Hz -> N = sigma / sqrt(rate) = 1e-3 rad/sqrt(s)
    rng = np.random.default_rng(7)
    n = 360_000  # one hour
    data = RateSeries(np.arange(n) / rate_hz, rng.normal(0.0, sigma, n), "gyro_z")
    out = evaluate("stock.allan_variance", AllanConfig(), [data], subject="s")
    m = measures(out)
    assert out.metric == "arw" and out.value == pytest.approx(sigma / math.sqrt(rate_hz), rel=0.03)
    taus, adev = np.array(out.payload["x"]), np.array(out.payload["y"])
    middle = (taus > 0.05) & (taus < 50)
    slope = np.polyfit(np.log10(taus[middle]), np.log10(adev[middle]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.02)
    assert adev[np.argmin(abs(taus - 1.0))] == pytest.approx(sigma / math.sqrt(rate_hz), rel=0.05)  # sigma(1 s) = N for white noise
    # headline in deg/sqrt(h) (the IMU datasheet unit): N [rad/sqrt(s)] x 180/pi x 60
    deg = evaluate("stock.allan_variance", AllanConfig(scale=180 / math.pi * 60, unit="deg/sqrt(h)"), [data], subject="s")
    assert deg.value == pytest.approx(out.value * 180 / math.pi * 60) and deg.unit == "deg/sqrt(h)"
    assert {"bias_instability", "rrw"} <= set(m) and "arw" not in m  # the headline is never repeated as a measurement


def test_allan_deviation_finds_a_rate_random_walk_and_bias_instability_is_the_floor_over_0664():
    rate_hz, white, walk = 100.0, 0.01, 1e-4  # K = 1e-4 rad/s/sqrt(s)
    rng = np.random.default_rng(11)
    n = 720_000
    rate = rng.normal(0.0, white, n) + np.cumsum(rng.normal(0.0, walk / math.sqrt(rate_hz), n))
    out = evaluate("stock.allan_variance", AllanConfig(headline="rrw"), [RateSeries(np.arange(n) / rate_hz, rate, "gyro")], subject="s")
    m = measures(out)
    assert out.value == pytest.approx(walk, rel=0.3)  # one realisation: the +1/2 tail is a few clusters
    assert m["arw"] == pytest.approx(white / math.sqrt(rate_hz), rel=0.05)
    # IEEE 952's 0.664 is sqrt(2 ln 2 / pi) = 0.66428 rounded: the library uses the exact constant
    assert m["bias_instability"] == pytest.approx(min(out.payload["y"]) / math.sqrt(2 * math.log(2) / math.pi), rel=1e-9)
    assert m["bias_instability"] == pytest.approx(min(out.payload["y"]) / 0.664, rel=1e-3)


def test_allan_deviation_is_not_run_on_too_short_a_record():
    out = evaluate("stock.allan_variance", AllanConfig(), [RateSeries(np.arange(50) / 100.0, np.zeros(50), "g")], subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "TooShort")
