"""JTC diagnostics (stock/jtc.py): each channel on a run whose answer is known by construction, and the one rule that
lets the monitor run live: a tick's numbers never depend on a later tick."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_diagnostics import Limits, evaluate
from robot_diagnostics.stock.jtc import (COLLISION, PATH_TOLERANCE_VIOLATED, SUCCEEDED, CHANNELS, JtcCheckConfig, JtcConfig,
                                         JtcMonitor, JtcRun, JtcTick, first_event, scan)

NAMES = ("shoulder", "elbow")


def ramp_run(*, stuck_at: float | None = None, residual_jump_at: float | None = None, gap_at: float | None = None,
             end: float = 1.0, duration: float = 1.5, period: float = 0.01, speed: float = 0.5, mode_11_at: float | None = None):
    """Both joints follow a 0.5 rad/s ramp until ``end`` and then hold; optionally the elbow stops at ``stuck_at``, its
    torque jumps by 3 N·m at ``residual_jump_at``, the host skips 50 ms at ``gap_at`` or the boards time out."""

    ticks, t, q_stuck = [], 0.0, None
    while t <= duration + 1e-9:
        if gap_at is not None and gap_at <= t < gap_at + 0.05:
            t += period
            continue
        clock = min(t, end)
        q_target = np.array([speed * clock, -speed * clock])
        qd_target = np.array([speed, -speed]) if t < end else np.zeros(2)
        q, qd = q_target.copy(), qd_target.copy()
        if stuck_at is not None and t >= stuck_at:
            q_stuck = -speed * stuck_at if q_stuck is None else q_stuck
            q[1], qd[1] = q_stuck, 0.0
        effort = np.array([0.2, 0.1])
        if residual_jump_at is not None and t >= residual_jump_at:
            effort[1] += 3.0
        mode = [10, 11] if mode_11_at is not None and t >= mode_11_at else [10, 10]
        ticks.append(JtcTick(t=t, q_target=q_target, qd_target=qd_target, effort_target=[0.2, 0.1], q_command=q_target,
                             qd_command=qd_target, effort_command=[0.2, 0.1], max_effort=[12.0, 8.0], q=q, qd=qd,
                             effort=effort, mode=mode, state_t=t - 0.001, target_clock_s=t, end_s=end))
        t = round(t + period, 9)
    return JtcRun(ticks=ticks, joint_names=NAMES, run="synthetic")


def test_every_channel_is_emitted_with_its_shape():
    series = scan(JtcConfig(), ramp_run())
    assert set(series.channels) == set(CHANNELS)
    for name, values in series.channels.items():
        assert values.shape[0] == len(series.t), name


def test_causal_a_prefix_scans_to_the_same_numbers():
    """WHY: the monitor runs inside controller.tick on a live robot; a number that changed once later ticks arrived
    would mean the playback shows something the controller could not have known."""

    run = ramp_run(stuck_at=0.4, residual_jump_at=0.6, gap_at=0.8)
    full = scan(JtcConfig(), run)
    for k in (1, 17, 40, 63, 90):
        prefix = scan(JtcConfig(), JtcRun(run.ticks[:k], run.joint_names))
        for name in CHANNELS:
            np.testing.assert_array_equal(prefix.channels[name], full.channels[name][:k], err_msg=name)


def test_a_clean_run_flags_nothing_and_meets_its_goal():
    series = scan(JtcConfig(), ramp_run())
    assert not [e for e in series.events if e["kind"] != "outcome"]
    assert series.outcome == SUCCEEDED
    assert np.nanmax(np.abs(series.channels["position_error_rad"])) == 0.0
    assert np.all(series.channels["command_sent"] == 1.0)


def test_a_stuck_joint_is_blocked_after_the_lead_and_dwell():
    config = JtcConfig()
    series = scan(config, ramp_run(stuck_at=0.4))
    event = first_event(series, ("blocked",))
    assert event["joint"] == "elbow"
    # the lead passes 1° (0.0175 rad) after 0.0175 / 0.5 = 35 ms (the 0.44 s tick), then 20 ms of dwell: 0.46 s
    assert event["t"] == pytest.approx(0.46)
    assert series.outcome == COLLISION
    assert first_event(series, ("collision",))["t"] == event["t"]


def test_a_torque_residual_is_contact_then_collision():
    series = scan(JtcConfig(), ramp_run(residual_jump_at=0.3))
    contact, collision = first_event(series, ("contact",)), first_event(series, ("collision",))
    assert contact["joint"] == collision["joint"] == "elbow"
    # over both lines from the 0.30 s tick; 10 ms of dwell: both rise at 0.31 s
    assert contact["t"] == pytest.approx(0.31) and collision["t"] == pytest.approx(0.31)


def test_a_host_gap_counts_the_skipped_periods():
    series = scan(JtcConfig(nominal_period_s=0.01), ramp_run(gap_at=0.5))
    k = int(np.nanargmax(series.channels["tick_period_s"]))
    assert series.channels["tick_period_s"][k] == pytest.approx(0.06)
    assert series.channels["missed_ticks"][k] == 5.0


def test_a_joint_not_commanded_shows_zero():
    run = ramp_run()
    ticks = [JtcTick(**{**tick.__dict__, "q_command": [tick.q_command[0], math.nan], "qd_command": [tick.qd_command[0], math.nan],
                        "effort_command": [tick.effort_command[0], math.nan]}) for tick in run.ticks]
    series = scan(JtcConfig(), JtcRun(ticks, NAMES))
    assert np.all(series.channels["command_sent"][:, 0] == 1.0) and np.all(series.channels["command_sent"][:, 1] == 0.0)


def test_path_tolerance_is_the_ros_rule_and_a_watchdog_shows_in_servo_mode():
    series = scan(JtcConfig(path_tolerance_rad=0.01, lead_rad=10.0, collision_nm=100.0), ramp_run(stuck_at=0.4, mode_11_at=1.2))
    assert series.outcome == PATH_TOLERANCE_VIOLATED
    assert np.nanmin(series.channels["path_margin_rad"]) < 0.0
    assert first_event(series, ("servo_timeout",))["joint"] == "elbow"


def test_registered_like_any_stock_check():
    result = evaluate("jtc.collision", JtcCheckConfig(limits=Limits.at_most(0.5)), ramp_run(stuck_at=0.4), subject="arm")
    assert result.verdict == "fail" and result.value == 1.0 and {"x", "y"} <= set(result.payload)
    ok = evaluate("jtc.collision", JtcCheckConfig(limits=Limits.at_most(0.5)), ramp_run(), subject="arm")
    assert ok.verdict == "pass"
    gap = evaluate("jtc.tick_period", JtcCheckConfig(limits=Limits.at_most(0.02)), ramp_run(gap_at=0.5), subject="arm")
    assert gap.verdict == "fail" and gap.value == pytest.approx(0.06)


def test_step_refuses_time_going_backwards():
    monitor = JtcMonitor(JtcConfig(), NAMES)
    tick = ramp_run().ticks[5]
    monitor.step(tick)
    with pytest.raises(ValueError, match="backwards"):
        monitor.step(JtcTick(**{**tick.__dict__, "t": tick.t - 0.01}))
