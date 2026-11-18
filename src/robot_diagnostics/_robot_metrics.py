"""robot_metrics, imported on first use: ``from ._robot_metrics import metrics, streams`` in place of importing robot_metrics at module load.

WHY robot_metrics (Zach, 2026-11-17: "a diagnostic = a metric + a threshold"; docs/peps/0004-stock-diagnostics-on-robot-metrics.md): the
stock checks robot_metrics v0 already computes - tracking error, the error at every stop, the limiter shares - take their metric from it.
robot_metrics owns the math; RobotDx keeps its configs, Limits, judge, Result, records and UI. A metric's identity is its class and frozen
config, never a limit, so a new target never makes a new metric (the prior-art note's fix to Limits riding inside each config).

WHY lazy (as ``_numpy`` and ``_sim2real``): robot_metrics is in the ``diagnostics`` extra beside numpy, never a dependency of the host.
Without it every module still imports and every stock Spec still registers; running a check that needs it raises ``ModuleNotFoundError``
(name ``robot_metrics``, or ``numpy``), which the host reports as not_run.
"""

from __future__ import annotations

import importlib


class _Module:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attribute: str):
        if self._module is None:
            self._module = importlib.import_module(self._name)  # raises ModuleNotFoundError(name="robot_metrics") without the extra
        # WHY looked up each time (the module cached, never the class): what runs is always robot_metrics' current metric
        return getattr(self._module, attribute)


metrics = _Module("robot_metrics")


def streams(t, measured, target=None, target_speed=None, *, name: str = "signal", channels=None):
    """A run's rows as the (target, measured) pair a robot_metrics metric reads: robot_msgs JointTrajectories on one clock, the compared
    signal in ``positions`` (a joint angle, a velocity, a gripper width alike), the commanded speed in the target's ``velocities``, and any
    ``channels`` (a fault code) on the measured stream. ``target`` None: only the measured stream (a code share reads nothing else).

    WHY the signal always in ``positions``: a base metric is math only and its meaning comes from the data passed in (Zach, 2026-11-17), so
    a velocity's tracking error is the same ErrorStat as an angle's; the stream's joint name carries which column it was."""

    from robot_msgs.trajectory import JointTrajectory, Trajectory

    if target is None:
        return None, Trajectory(t=t, channels=dict(channels or {}))
    names = (name,)
    reference = JointTrajectory(t=t, joint_names=names, positions=target[:, None],
                                velocities=None if target_speed is None else target_speed[:, None])
    response = JointTrajectory(t=t, joint_names=names, positions=measured[:, None], channels=dict(channels or {}))
    return reference, response


__all__ = ["metrics", "streams"]
