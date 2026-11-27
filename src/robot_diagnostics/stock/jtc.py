"""Joint trajectory controller (JTC) diagnostics: what a JTC can know every tick, as numbers you can scrub through.

A JTC streams a target trajectory to the servo boards and reads their state back. Three signals meet in it each tick
(Zach, 2026-11-27: "There is the target trajectory, Then we have the commanded trajectory, Then we have the measured
trajectory"):

- **target**: the trajectory sampled at the controller's clock (position, velocity, and effort = the feed-forward torque a
  model expects);
- **command**: what was actually sent (position, velocity, feed-forward, torque cap; NaN = a field not sent);
- **measured**: what the board reported (position, velocity, torque, mode, fault) and when it sampled it.

``JtcMonitor.step(tick)`` turns one tick into every diagnostic channel below. It keeps only what its detectors need
(the last tick time, dwell timers, a latch) and never looks ahead, so it runs inside ``controller.tick`` on a live robot.
``scan(run)`` runs the same ``step`` over a recorded run. WHY one rule for both (the robot_msgs ``ToleranceMonitor``
lesson): a live run and its replay can never disagree, and the playback shows exactly what the controller saw.

The ``jtc.*`` registry entries wrap ``scan`` as ordinary robot_diagnostics series diagnostics, so a host evaluates them
like any stock check (``evaluate("jtc.collision", JtcConfig(), run, subject="arm")``). WHY a drop-in: Zach, 2026-11-27,
"This should basically be building on the robot_diagnostics package, its just like drop in of that".

Channels (per joint unless marked *scalar*; SI units, angles in rad; viewers convert to degrees):

| group | channel | what | prior art |
|---|---|---|---|
| command | ``command_sent`` | 1 when a position, velocity or torque field was sent this tick | "quickly see if the joints are being sent anything" |
| command | ``command_offset_rad`` | command − target position (0 for a plain JTC; held, aborted or shifted otherwise) | ros2 JTC ``output`` vs ``reference`` |
| command | ``effort_cap_nm`` | the torque cap sent (NaN = the board's own cap) | moteus 0x025 maximum torque |
| timing | ``tick_period_s`` *scalar* | time since the previous tick | ros2_control ``periodicity`` |
| timing | ``missed_ticks`` *scalar* | periods the last gap skipped beyond one (> ``missed_factor`` x nominal) | ros2_control overruns |
| timing | ``state_age_s`` *scalar* | tick time − the state's sample time | ROS ``TimeStampStatus`` |
| timing | ``progress`` *scalar* | target clock / trajectory end, clipped to [0, 1] | JTC ``traj_time_`` |
| timing | ``late_s`` *scalar* | time past the trajectory end while the goal is not met | JTC ``goal_time``, MoveIt duration budget |
| tracking | ``position_error_rad`` | target − measured position | ros2 JTC state ``error`` |
| tracking | ``velocity_error_rad_s`` | target − measured velocity | ros2 JTC state ``error`` |
| tracking | ``path_margin_rad`` | path tolerance − abs(position error), while the target runs (< 0 = violated) | ``constraints.<joint>.trajectory`` |
| tracking | ``settled`` | after the end: abs(error) <= goal tolerance and abs(velocity) <= stopped velocity | ``constraints.<joint>.goal``, ``stopped_velocity_tolerance`` |
| effort | ``effort_residual_nm`` | measured torque − target effort | Haddadin 2017 eq. 32-34 (inverse-dynamics residual) |
| effort | ``effort_residual_filtered_nm`` | the residual through a first-order low-pass at ``residual_filter_hz`` (= the raw one when 0): what contact and collision judge | Haddadin 2017 eq. 47 |
| effort | ``at_cap`` | 1 when abs(measured torque) >= ``cap_fraction`` x the cap | moteus fault-register code 102 |
| contact | ``lead_rad`` | following error along the target's motion: sign(target velocity) x (target − measured) | robot_msgs ``Stall.lead`` |
| contact | ``blocked`` | the target moves, the joint lags by more than ``lead_rad`` and is slow, for ``blocked_dwell_s`` | homing's ``Stall``, generalised to a moving target |
| contact | ``contact`` | abs(residual) > ``contact_nm`` for ``residual_dwell_s``; not latched (warn) | libfranka ``joint_contact`` |
| contact | ``collision`` | blocked, or abs(residual) > ``collision_nm`` for the dwell, or at the cap while lagging; latched (fail) | libfranka ``joint_collision`` |
| speed | ``overspeed_rad_s`` | abs(measured velocity) − ``max_speed_rad_s`` (> 0 = over) | ``JointCommandSequence.max_speed`` |
| speed | ``drive_power_w`` | measured torque x measured velocity (> 0 = the drive adds energy) | the sequence's positive-work guard |
| limits | ``limit_margin_rad`` | distance to the nearest position limit (NaN when none are given) | ``servopos.position_min/max`` |
| servo | ``servo_timeout`` | board mode 11 (watchdog timed out) | moteus mode register |
| servo | ``servo_fault`` | board mode 1 (fault) | moteus fault codes 32-52 |
| verdict | ``outcome`` *scalar* | 0 running, 1 goal met, -4 path violated, -5 goal violated, -6 collision, -7 servo fault | FollowJointTrajectory result codes |
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .._numpy import np
from ..core import Evidence, Limits, Measure, result

# FollowJointTrajectory result codes (control_msgs), plus two of ours below the ROS range
RUNNING, SUCCEEDED = 0, 1
PATH_TOLERANCE_VIOLATED, GOAL_TOLERANCE_VIOLATED = -4, -5
COLLISION, SERVO_FAULT = -6, -7
OUTCOME_NAMES = {RUNNING: "running", SUCCEEDED: "goal met", PATH_TOLERANCE_VIOLATED: "path tolerance violated",
                 GOAL_TOLERANCE_VIOLATED: "goal tolerance violated", COLLISION: "collision", SERVO_FAULT: "servo fault"}

MODE_FAULT, MODE_POSITION, MODE_TIMEOUT = 1, 10, 11  # moteus mode register

PER_JOINT = ("command_sent", "command_offset_rad", "effort_cap_nm", "position_error_rad", "velocity_error_rad_s",
             "path_margin_rad", "settled", "effort_residual_nm", "at_cap", "lead_rad", "blocked", "contact", "collision",
             "overspeed_rad_s", "drive_power_w", "limit_margin_rad", "servo_timeout", "servo_fault",
             "effort_residual_filtered_nm")
SCALAR = ("tick_period_s", "missed_ticks", "state_age_s", "progress", "late_s", "outcome")
CHANNELS = PER_JOINT + SCALAR
# The flags whose first rise is an event (what a viewer marks and a scorer counts)
FLAGS = ("blocked", "contact", "collision", "servo_timeout", "servo_fault")

PerJoint = float | tuple[float, ...]


@dataclass(frozen=True)
class JtcTick:
    """One tick as the controller saw it. Arrays are per joint, in ``joint_names`` order; None = not known."""

    t: float  # the tick's time, s (one monotonic clock for the run)
    q_target: Sequence[float]  # rad: the target trajectory at the controller's clock
    q: Sequence[float]  # rad: measured
    qd_target: Sequence[float] | None = None
    effort_target: Sequence[float] | None = None  # N·m: the trajectory's effort (the feed-forward a model expects)
    q_command: Sequence[float] | None = None  # rad: the position sent (NaN = not sent)
    qd_command: Sequence[float] | None = None
    effort_command: Sequence[float] | None = None  # N·m: the feed-forward sent
    max_effort: Sequence[float] | None = None  # N·m: the cap sent (NaN = the board's own)
    qd: Sequence[float] | None = None
    effort: Sequence[float] | None = None  # N·m: the board's reported torque
    mode: Sequence[int] | None = None  # moteus mode per joint
    fault: Sequence[int] | None = None
    state_t: float | None = None  # when the measured state was sampled, same clock as t
    target_clock_s: float | None = None  # the controller's own trajectory clock (for progress)
    end_s: float | None = None  # the active trajectory's end on the target clock


@dataclass(frozen=True)
class JtcRun:
    """A recorded run: its ticks in time order, the joint names, and where it came from (an evidence pointer)."""

    ticks: Sequence[JtcTick]
    joint_names: tuple[str, ...]
    run: str = ""


@dataclass(frozen=True)
class JtcConfig:
    """Every threshold the JTC diagnostics use. A per-joint value is a scalar (every joint) or one value per joint.

    The defaults are a starting point, not a measured envelope: thresholds belong to a calibration on clean runs
    (Haddadin 2017 eq. 54: the worst free-motion residual plus a margin), which the JTC Diagnostics track does.
    """

    nominal_period_s: float = 0.01  # the host's tick period (the rack host's 100 Hz)
    missed_factor: float = 1.5  # a gap longer than this x nominal counts the skipped periods
    path_tolerance_rad: PerJoint = 0.05  # ros2 JTC constraints.<joint>.trajectory; 0 = not checked (bam_controllers v1's 0.05)
    goal_tolerance_rad: PerJoint = 0.0175  # constraints.<joint>.goal (1°); 0 = not checked
    stopped_velocity_rad_s: float = 0.01  # constraints.stopped_velocity_tolerance (the ROS default)
    goal_time_s: float = 1.0  # constraints.goal_time: how long after the end the goal may take
    moving_speed_rad_s: PerJoint = 0.05  # a target faster than this "should be moving": lead and blocked are defined
    lead_rad: PerJoint = 0.0175  # following error along the motion before a joint counts as lagging (1°)
    stall_speed_rad_s: PerJoint = 0.05  # measured speed below which a lagging joint is blocked
    blocked_dwell_s: float = 0.02
    contact_nm: PerJoint = 1.0  # abs(residual) above it: contact (warn, not latched)
    collision_nm: PerJoint = 2.5  # abs(residual) above it: collision (fail, latched)
    residual_dwell_s: float = 0.01
    # first-order low-pass on the residual before the contact/collision lines; 0 = off. WHY: the momentum observer's
    # residual is a low-pass of the external torque (Haddadin 2017 eq. 47), and the raw measured-minus-feed-forward
    # torque swings +-5 N·m in plain free motion on the bench (2026-11-27), so an unfiltered line cries wolf
    residual_filter_hz: float = 0.0
    # after the host missed ticks or a board timed out, the contact lines wait this long (s); 0 = off. WHY: a
    # time-driven JTC skips ahead after a stall and its boards catch up with large torques; that torque is explained by
    # the stall the monitor itself just saw, not by a contact (R6.HostStall, 2026-11-27)
    recover_s: float = 0.0
    cap_fraction: float = 0.95
    max_speed_rad_s: PerJoint = 3.0  # 171.9°/s: JointCommandSequence.max_speed's default
    position_limits_rad: tuple[tuple[float, float], ...] = ()  # (low, high) per joint; () = none known


def _per_joint(value: PerJoint, count: int, name: str):
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1:
        return np.full(count, float(values[0]))
    if values.size != count:
        raise ValueError(f"{name} has {values.size} values for {count} joints")
    return values.copy()


def _array(values, count: int, fill: float = math.nan):
    if values is None:
        return np.full(count, fill)
    return np.asarray(values, dtype=float).reshape(count)


def _since(since, condition, t: float):
    """When each joint's condition became true (NaN while false): the start of its dwell."""

    return np.where(condition, np.where(np.isnan(since), t, since), np.nan)


class JtcMonitor:
    """Every JTC diagnostic for one tick at a time. Causal: a tick's numbers depend only on it and the ticks before."""

    def __init__(self, config: JtcConfig, joint_names: Sequence[str]) -> None:
        self.config = config
        self.joint_names = tuple(joint_names)
        count = len(self.joint_names)
        self._count = count
        self._path_tol = _per_joint(config.path_tolerance_rad, count, "path_tolerance_rad")
        self._goal_tol = _per_joint(config.goal_tolerance_rad, count, "goal_tolerance_rad")
        self._moving = _per_joint(config.moving_speed_rad_s, count, "moving_speed_rad_s")
        self._lead = _per_joint(config.lead_rad, count, "lead_rad")
        self._stall = _per_joint(config.stall_speed_rad_s, count, "stall_speed_rad_s")
        self._contact = _per_joint(config.contact_nm, count, "contact_nm")
        self._collision = _per_joint(config.collision_nm, count, "collision_nm")
        self._max_speed = _per_joint(config.max_speed_rad_s, count, "max_speed_rad_s")
        if config.position_limits_rad:
            limits = np.asarray(config.position_limits_rad, dtype=float).reshape(count, 2)
            self._low, self._high = limits[:, 0], limits[:, 1]
        else:
            self._low = self._high = None
        self._last_t: float | None = None
        # WHY "since" times, not summed dts: a dwell is time spent over the line, measured from the first tick that was
        # over it; summing dt would count the gap before that tick (a 10 ms dwell would pass on its first tick)
        self._blocked_since = np.full(count, np.nan)
        self._contact_since = np.full(count, np.nan)
        self._collision_since = np.full(count, np.nan)
        self._latched = np.zeros(count, dtype=bool)
        self._filtered = None  # the low-passed residual (None until the first finite one)
        self._disturbed_t = -math.inf  # the last tick that missed periods or saw a board time out or fault
        self._outcome = RUNNING
        self._flag_state = {name: np.zeros(count, dtype=bool) for name in FLAGS}
        self.events: list[dict[str, Any]] = []  # {"t", "kind", "joint"}: each flag's rise on each joint, and the outcome

    @property
    def outcome(self) -> int:
        return self._outcome

    def step(self, tick: JtcTick) -> dict[str, Any]:
        config, count = self.config, self._count
        t = float(tick.t)
        dt = 0.0 if self._last_t is None else t - self._last_t
        if dt < 0.0:
            raise ValueError(f"tick times go backwards: {t} after {self._last_t}")
        period = math.nan if self._last_t is None else dt
        self._last_t = t
        nominal = config.nominal_period_s
        missed = 0.0 if not (period > config.missed_factor * nominal) else float(round(period / nominal) - 1)

        q_target = _array(tick.q_target, count)
        q = _array(tick.q, count)
        qd_target = _array(tick.qd_target, count, 0.0)
        qd = _array(tick.qd, count)
        effort = _array(tick.effort, count)
        effort_target = _array(tick.effort_target, count, 0.0)
        q_command = _array(tick.q_command, count)
        qd_command = _array(tick.qd_command, count)
        effort_command = _array(tick.effort_command, count)
        cap = _array(tick.max_effort, count)
        mode = None if tick.mode is None else np.asarray(tick.mode, dtype=float).reshape(count)

        sent = (np.isfinite(q_command) | np.isfinite(qd_command) | np.isfinite(effort_command)).astype(float)
        error = q_target - q
        velocity_error = qd_target - qd
        clock = t if tick.target_clock_s is None else float(tick.target_clock_s)
        end = math.nan if tick.end_s is None else float(tick.end_s)
        running = not (clock >= end)  # NaN end: the target is always running
        progress = math.nan if not math.isfinite(end) else (1.0 if end <= 0.0 else min(max(clock / end, 0.0), 1.0))

        # tracking (ros2 JTC: path while running, goal after the end)
        path_margin = np.where(running & (self._path_tol > 0.0), self._path_tol - np.abs(error), np.nan)
        speed_ok = np.abs(qd) <= config.stopped_velocity_rad_s if tick.qd is not None else np.ones(count, dtype=bool)
        goal_ok = (np.abs(error) <= np.where(self._goal_tol > 0.0, self._goal_tol, np.inf)) & speed_ok
        settled = np.where(running, np.nan, goal_ok.astype(float))
        late = 0.0 if running or goal_ok.all() else max(clock - end, 0.0)

        # effort and contact
        residual = effort - effort_target
        cap_known = np.isfinite(cap) & (cap > 0.0)
        at_cap = np.where(cap_known & np.isfinite(effort), np.abs(effort) >= config.cap_fraction * cap, False)
        moving = np.abs(qd_target) > self._moving
        lead = np.where(moving, np.sign(qd_target) * error, np.nan)
        slow = np.abs(qd) < self._stall if tick.qd is not None else np.zeros(count, dtype=bool)
        lagging = moving & (np.nan_to_num(lead, nan=-np.inf) > self._lead)
        filtered = self._low_pass(residual, dt)
        over_contact = np.nan_to_num(np.abs(filtered), nan=0.0) > self._contact
        over_collision = np.nan_to_num(np.abs(filtered), nan=0.0) > self._collision
        self._blocked_since = _since(self._blocked_since, lagging & slow, t)
        self._contact_since = _since(self._contact_since, over_contact, t)
        self._collision_since = _since(self._collision_since, over_collision, t)
        blocked = (lagging & slow) & (t - self._blocked_since + 1e-12 >= config.blocked_dwell_s)
        contact = over_contact & (t - self._contact_since + 1e-12 >= config.residual_dwell_s)
        hit = blocked | (over_collision & (t - self._collision_since + 1e-12 >= config.residual_dwell_s)) | (at_cap & lagging)
        if missed > 0.0 or (mode is not None and bool(np.any((mode == MODE_TIMEOUT) | (mode == MODE_FAULT)))):
            self._disturbed_t = t
        if config.recover_s > 0.0 and t - self._disturbed_t <= config.recover_s:
            blocked = np.zeros(count, dtype=bool)
            contact = np.zeros(count, dtype=bool)
            hit = np.zeros(count, dtype=bool)
        self._latched |= hit
        collision = self._latched.copy()

        # speed, limits, servo
        overspeed = np.abs(qd) - self._max_speed
        drive_power = effort * qd
        if self._low is not None:
            limit_margin = np.minimum(q - self._low, self._high - q)
        else:
            limit_margin = np.full(count, np.nan)
        timeout = np.zeros(count) if mode is None else (mode == MODE_TIMEOUT).astype(float)
        fault = np.zeros(count) if mode is None else (mode == MODE_FAULT).astype(float)
        if mode is None:
            timeout[:] = np.nan
            fault[:] = np.nan

        # verdict: the first decisive thing that happened stays the outcome (ros2 JTC aborts once)
        if self._outcome == RUNNING:
            if np.nan_to_num(fault).any():
                self._outcome = SERVO_FAULT
            elif collision.any():
                self._outcome = COLLISION
            elif (np.nan_to_num(path_margin, nan=0.0) < 0.0).any():
                self._outcome = PATH_TOLERANCE_VIOLATED
            elif not running and goal_ok.all():
                self._outcome = SUCCEEDED
            elif not running and clock - end > config.goal_time_s:
                self._outcome = GOAL_TOLERANCE_VIOLATED
            if self._outcome != RUNNING:
                self.events.append({"t": t, "kind": "outcome", "joint": "", "code": self._outcome,
                                    "name": OUTCOME_NAMES[self._outcome]})

        flags = {"blocked": blocked, "contact": contact, "collision": collision,
                 "servo_timeout": np.nan_to_num(timeout) > 0.5, "servo_fault": np.nan_to_num(fault) > 0.5}
        for name, now in flags.items():
            rose = now & ~self._flag_state[name]
            for index in np.flatnonzero(rose):
                self.events.append({"t": t, "kind": name, "joint": self.joint_names[index]})
            self._flag_state[name] = now.copy()

        state_age = math.nan if tick.state_t is None else t - float(tick.state_t)
        return {
            "command_sent": sent, "command_offset_rad": q_command - q_target, "effort_cap_nm": cap,
            "tick_period_s": period, "missed_ticks": missed, "state_age_s": state_age, "progress": progress,
            "late_s": late, "position_error_rad": error, "velocity_error_rad_s": velocity_error,
            "path_margin_rad": path_margin, "settled": settled, "effort_residual_nm": residual,
            "at_cap": at_cap.astype(float), "lead_rad": lead, "blocked": blocked.astype(float),
            "contact": contact.astype(float), "collision": collision.astype(float), "overspeed_rad_s": overspeed,
            "drive_power_w": drive_power, "limit_margin_rad": limit_margin, "servo_timeout": timeout,
            "servo_fault": fault, "outcome": float(self._outcome), "effort_residual_filtered_nm": filtered,
        }

    def _low_pass(self, residual, dt: float):
        hz = self.config.residual_filter_hz
        if not hz > 0.0:
            return residual
        if self._filtered is None or dt <= 0.0:
            self._filtered = np.where(np.isfinite(residual), residual, 0.0)
            return self._filtered.copy()
        alpha = dt / (dt + 1.0 / (2.0 * math.pi * hz))
        self._filtered = self._filtered + alpha * (np.nan_to_num(residual, nan=0.0) - self._filtered)
        return self._filtered.copy()


@dataclass(frozen=True)
class JtcSeries:
    """``scan``'s answer: every channel as an array over the run's ticks ((K,) scalar, (K, J) per joint) + the events."""

    t: Any
    joint_names: tuple[str, ...]
    channels: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...] = ()
    outcome: int = RUNNING


def scan(config: JtcConfig, run: JtcRun) -> JtcSeries:
    """The same ``step`` over a recorded run, tick by tick (so it is causal by construction)."""

    monitor = JtcMonitor(config, run.joint_names)
    rows = [monitor.step(tick) for tick in run.ticks]
    t = np.asarray([tick.t for tick in run.ticks], dtype=float)
    channels = {}
    for name in CHANNELS:
        values = [row[name] for row in rows]
        channels[name] = np.asarray(values, dtype=float) if values else np.zeros((0, len(run.joint_names)) if name in PER_JOINT else 0)
    return JtcSeries(t=t, joint_names=run.joint_names, channels=channels, events=tuple(monitor.events),
                     outcome=monitor.outcome)


def first_event(series: JtcSeries, kinds: Iterable[str]) -> Mapping[str, Any] | None:
    kinds = set(kinds)
    return next((event for event in series.events if event["kind"] in kinds), None)


# ----- the registry entries: each picks one question out of the same scan ---------------------------------------
@dataclass(frozen=True)
class JtcCheckConfig:
    """A ``jtc.*`` check: the monitor's thresholds plus the limits its headline number is judged against."""

    monitor: JtcConfig = field(default_factory=JtcConfig)
    limits: Limits = field(default_factory=Limits)


def _series_result(config: JtcCheckConfig, run: JtcRun, *, channel: str, value_of, unit: str, metric: str, title: str,
                   message, reduce_joints=None):
    series = scan(config.monitor, run)
    values = np.asarray(series.channels[channel], dtype=float)
    y = values if values.ndim == 1 else (reduce_joints(values) if reduce_joints else np.nanmax(values, axis=1))
    value = value_of(series, values) if len(series.t) else None
    worst = int(np.nanargmax(np.abs(np.nan_to_num(y, nan=0.0)))) if len(series.t) else 0
    return result(config, value=value, unit=unit, metric=metric, limits=config.limits, title=title,
                  message=message(series, value), payload={"x": series.t.tolist(), "y": np.where(np.isfinite(y), y, None).tolist(),
                                                           "channel": channel},
                  measurements=(Measure("ticks", len(series.t)), Measure("outcome", series.outcome)),
                  evidence=(Evidence(run.run, channel),) if run.run else (), run=run.run,
                  t0=float(series.t[worst]) if len(series.t) else None, t1=float(series.t[worst]) if len(series.t) else None)


def collision(config: JtcCheckConfig, run: JtcRun):
    """Seconds from the run's start to the first collision; NaN-free: no collision = the run's length, judged pass."""

    def value_of(series, values):
        event = first_event(series, ("collision",))
        return 0.0 if event is None else 1.0

    def message(series, value):
        event = first_event(series, ("collision",))
        return "no collision" if event is None else f"collision on {event['joint']} at {event['t']:.3f} s"

    return _series_result(config, run, channel="collision", value_of=value_of, unit="1", metric="collided",
                          title="JTC collision", message=message)


def path_margin(config: JtcCheckConfig, run: JtcRun):
    """The smallest path margin over the run (rad): below 0 = the path tolerance was violated."""

    def value_of(series, values):
        finite = values[np.isfinite(values)]
        return float(finite.min()) if finite.size else None

    return _series_result(config, run, channel="path_margin_rad", value_of=value_of, unit="rad", metric="min",
                          title="JTC path margin", reduce_joints=lambda v: np.nanmin(np.where(np.isfinite(v), v, np.inf), axis=1),
                          message=lambda s, v: "no path tolerance checked" if v is None else f"smallest path margin {math.degrees(v):.2f}°")


def tick_period(config: JtcCheckConfig, run: JtcRun):
    """The longest gap between ticks (s)."""

    def value_of(series, values):
        finite = values[np.isfinite(values)]
        return float(finite.max()) if finite.size else None

    return _series_result(config, run, channel="tick_period_s", value_of=value_of, unit="s", metric="max",
                          title="JTC tick period", message=lambda s, v: "one tick" if v is None else f"longest gap {v * 1e3:.1f} ms")


def command_sent(config: JtcCheckConfig, run: JtcRun):
    """Share of (tick, joint) cells that carried a command: 1 = every joint got something every tick."""

    return _series_result(config, run, channel="command_sent", value_of=lambda s, v: float(np.mean(v)), unit="1",
                          metric="share", title="JTC command sent", reduce_joints=lambda v: np.nanmin(v, axis=1),
                          message=lambda s, v: f"{v * 100:.1f} % of joint-ticks carried a command")


def servo_mode(config: JtcCheckConfig, run: JtcRun):
    """1 when any board timed out or faulted during the run."""

    def value_of(series, values):
        bad = np.nan_to_num(series.channels["servo_timeout"]) + np.nan_to_num(series.channels["servo_fault"])
        return float(bad.max() > 0) if bad.size else None

    return _series_result(config, run, channel="servo_timeout", value_of=value_of, unit="1", metric="any",
                          title="JTC servo mode", message=lambda s, v: "boards stayed in control" if not v else "a board timed out or faulted")


def _specs():
    from ..registry import Spec

    return [
        Spec("jtc.collision", collision, JtcCheckConfig, "JTC collision", "health", "health", "level_history", "series", "1",
             summary="blocked (lagging and slow), a torque residual over the collision level, or at the cap while lagging; latched"),
        Spec("jtc.path_margin", path_margin, JtcCheckConfig, "JTC path margin", "real_vs_target", "driver", "threshold_band", "series", "rad",
             summary="path tolerance - abs(target - measured) while the target runs (ros2 JTC constraints.<joint>.trajectory)"),
        Spec("jtc.tick_period", tick_period, JtcCheckConfig, "JTC tick period", "health", "health", "threshold_band", "series", "s",
             summary="time between controller ticks; skipped periods counted (ros2_control periodicity)"),
        Spec("jtc.command_sent", command_sent, JtcCheckConfig, "JTC command sent", "health", "health", "level_history", "series", "1",
             summary="share of joint-ticks that carried a position, velocity or torque command"),
        Spec("jtc.servo_mode", servo_mode, JtcCheckConfig, "JTC servo mode", "health", "health", "level_history", "series", "1",
             summary="any board in watchdog timeout (mode 11) or fault (mode 1) during the run"),
    ]


JTC = _specs()
