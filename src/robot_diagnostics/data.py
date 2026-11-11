"""The data a stock diagnostic takes: a ``Trajectory`` (one run of one device, columns on one clock) and the helpers
every comparison shares.

Names follow data-ui PEP 0010 §4 (the trajectory contract): every column name ends in its unit (``q_rad``, ``qd_rad_s``,
``tau_nm``, ``i_q_a``, ``v_bus_v``, ``temp_c``); ``t`` is seconds from the run's start; the target is a situation of its
own (``source = "target"``); every model, twin or replay is ``source = "sim"``. The arrays stay SI (rad), as the PEP's
"SI stays in the producer's own npz" rule says; ``rows.py`` writes the display projection (deg) data-ui reads.
Data-ui never needs to know what these mean (Zach, 2026-10-28: "Data UI doesn't need to know the robotic semantic").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from . import _sim2real
from ._numpy import np

MODE_POSITION = 10  # moteus mode 10 = position control: the rows a controller was actually tracking
COLUMNS = ("q_rad", "q_target_rad", "qd_rad_s", "qd_target_rad_s", "tau_nm", "i_q_a", "mode", "fault", "temp_c", "v_bus_v", "tau_ff_nm")
# A measured column and its commanded twin (the target situation's column, carried on the same clock)
TARGET_OF = {"q_rad": "q_target_rad", "qd_rad_s": "qd_target_rad_s"}


def target_of(name: str) -> str:
    """The commanded twin of a measured column: ``TARGET_OF``'s, else PEP 0010's naming with ``_target`` before the unit
    (``width_mm`` -> ``width_target_mm``, ``force_n`` -> ``force_target_n``). "" for a name without a unit suffix."""

    if name in TARGET_OF:
        return TARGET_OF[name]
    stem, _, unit = name.rpartition("_")
    return f"{stem}_target_{unit}" if stem and unit else ""


@dataclass(frozen=True, eq=False)
class Trajectory:
    t: np.ndarray  # s from the run's start (monotonic)
    q_rad: np.ndarray  # what was measured (or predicted, for a model)
    q_target_rad: np.ndarray | None = None
    qd_rad_s: np.ndarray | None = None
    qd_target_rad_s: np.ndarray | None = None
    tau_nm: np.ndarray | None = None  # as reported: current x the board's Kt
    i_q_a: np.ndarray | None = None
    mode: np.ndarray | None = None
    fault: np.ndarray | None = None
    temp_c: np.ndarray | None = None
    v_bus_v: np.ndarray | None = None
    tau_ff_nm: np.ndarray | None = None  # the feed-forward torque each command carried, same frame as tau_nm (stock.feedback_share)
    run: str = ""  # a stable, unique id (for the rack, the npz path): every Result's evidence pointer
    device: str = ""  # "CAN 12"
    situation: str = "real"  # real | target | "twin v7" | "perfect servo" ...: one series on screen
    source: str = "real"  # the situation's family: real | sim | target (picks the hue: blue real, orange sim)
    frame: str = "joint"  # the frame the columns are written in (a joint's own axis here; "base", "belt" for poses)
    prelude_s: float = 0.0  # the runner's move to the tape's origin; not part of the tape
    labels: Mapping[str, Any] = field(default_factory=dict)  # class, seed, can_id, ff_fraction ...
    # WHY: a part's own channels beyond the joint's (gyro_rad_s, force_n, width_mm, cpu_temp_c), on the same clock and named by PEP 0010's
    # rule; a box or a tracking error reads them like any column (io-diagnostics feeds them from an IMU's, load cell's or gripper's State)
    extra: Mapping[str, np.ndarray] = field(default_factory=dict)

    def _values(self, name: str):
        return getattr(self, name, None) if name in COLUMNS else self.extra.get(name)

    def channel(self, name: str) -> np.ndarray:
        values = self._values(name)
        if values is None:
            raise KeyError(f"{self.run or 'trajectory'} has no {name!r} column")
        return np.asarray(values, float)

    def has(self, name: str) -> bool:
        return self._values(name) is not None

    def rows(self, *, skip_prelude: bool = False, controlled_only: bool = False) -> np.ndarray:
        """Boolean mask: the tape (prelude dropped) and/or the position-controlled rows."""

        keep = np.ones(len(self.t), bool)
        if skip_prelude:
            keep &= (self.t - self.t[0]) >= self.prelude_s
        if controlled_only and self.mode is not None:
            keep &= np.asarray(self.mode) == MODE_POSITION
        return keep

    def as_situation(self, q_rad: np.ndarray, situation: str, source: str = "sim") -> "Trajectory":
        """The same run with ``q_rad`` replaced: a model's prediction riding on the real run's clock and commands."""

        return replace(self, q_rad=np.asarray(q_rad, float), situation=situation, source=source)


CONTROLLED_BY = ("real", "both")  # whose position mode decides a scored row of a (real, model) pair


def scored_rows(real: Trajectory, model: Trajectory, *, skip_prelude: bool = False, controlled_only: bool = True,
                controlled_by: str = "real") -> np.ndarray:
    """The rows of the real run's clock a model is scored on: the tape (``skip_prelude``) and, with ``controlled_only``, the rows in
    position mode, judged by the real run alone (``real``) or by the real run AND the model (``both``).

    WHY ``both`` (bam_rack_twin budget.q_rms_mrad, 2026-11-04): a twin that timed out or left position mode is not tracking, so its
    prediction on those rows is not a fidelity statement; the rack scores only the rows both were in mode 10. A model without a mode
    column (a perfect servo, a lagged target) is taken as tracking everywhere."""

    if controlled_by not in CONTROLLED_BY:
        raise ValueError(f"controlled_by {controlled_by!r} not in {CONTROLLED_BY}")
    keep = real.rows(skip_prelude=skip_prelude, controlled_only=controlled_only)
    if controlled_only and controlled_by == "both" and model.mode is not None:
        keep &= np.asarray(model.mode) == MODE_POSITION
    return keep


# WHY the rulers are sim2real's (2026-11-10, proposal 2 step 4): the floor (real vs real) and fidelity (model vs real) share one RMS, one
# clock and one centre, and bam_rack_twin's gates T1-T4 read the same; they live once in sim2real.metrics.series, unchanged to the bit.


def rms(values: np.ndarray) -> float:
    """RMS of a series, NaN when empty (sim2real.metrics.series.rms: robot_msgs.compare's for a finite series)."""

    return _sim2real.series.rms(values)


def difference(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """data-ui PEP 0010 §4.4 ``difference(a, b)``: on a's own samples inside b's span, b(t_i) - a(t_i), b linearly
    interpolated; no row outside b's span (never extrapolates). Returns (t_i, b - a). sim2real.metrics.series.difference."""

    return _sim2real.series.difference(a, b)


def on_common_clock(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Two (t from 0, signal) series linearly interpolated onto one grid over their overlap: ``arange(0, min end, dt)``; the floor's
    (gate T1) shared clock. sim2real.metrics.series.on_common_clock."""

    return _sim2real.series.on_common_clock(a, b, dt)


def pair_rms(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], *, clock: str = "common", dt: float = 0.002) -> float:
    """RMS(b - a) on one clock: ``common`` (a shared grid of step dt) or ``a`` (a's own samples, PEP 0010)."""

    return _sim2real.series.pair_rms(a, b, clock=clock, dt=dt)


def pairwise_rms(series: Sequence[tuple[np.ndarray, np.ndarray]], *, clock: str = "common", dt: float = 0.002) -> list[tuple[int, int, float]]:
    """RMS of the difference for every pair: the shared primitive of the floor and of fidelity, so a fidelity ratio is never a ratio of
    two different rulers."""

    return _sim2real.series.pairwise_rms(series, clock=clock, dt=dt)


def center(values: Sequence[float], how: str = "median") -> float:
    """``median`` or ``median_high`` (the stdlib names) of the finite values; bam_rack_twin's per-class twin ratio takes ``sorted(v)[n // 2]``
    (median_high), the floor ``statistics.median``. sim2real.metrics.series.center."""

    return _sim2real.series.center(values, how)
