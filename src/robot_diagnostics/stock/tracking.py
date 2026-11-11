"""Real vs target: tracking error over a run, and the error left at every stop.

Zach, DP - Reality as Oracle (2026-10-02): "avg error, max error, (in the various angles), error at stop, etc." and the
goal "improve net trackectory tracking performance". Gate T4 (static accuracy) is ``stop_error`` with the joint's share
of the 5 mm tool gate as its limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np
from .._sim2real import actuator

from ..core import NOT_RUN, WARN, Evidence, Limits, Measure, result
from ..data import Trajectory, center, target_of


@dataclass(frozen=True)
class TrackingErrorConfig:
    statistic: str = "rms"  # rms | p95 | max | final (mean |e| over the last final_window_s)
    channel: str = "q_rad"  # q_rad (position) or qd_rad_s (velocity); its target column is data.target_of(channel)
    across_runs: str = "median"  # median | median_high: how per-run values combine
    skip_prelude: bool = False  # drop the runner's move to the origin
    controlled_only: bool = False  # only rows in position mode (moteus mode 10)
    final_window_s: float = 0.5
    scale: float = 1e3  # rad -> mrad
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)


def _statistic(error: np.ndarray, t: np.ndarray, config: TrackingErrorConfig) -> float:
    # WHY sim2real (2026-11-10, proposal 2 step 4): the statistic is sim2real.metrics.actuator's (rms, p95 and max robot_msgs.compare's)
    return actuator.tracking_statistic(error, t, config.statistic, final_window_s=config.final_window_s)


def tracking_error(config: TrackingErrorConfig, runs: Sequence[Trajectory]):
    """Per run, ``statistic(actual - target)`` on the kept, finite rows; combined over runs by ``across_runs``."""

    per_run, evidence, spans = [], [], []
    target_name = target_of(config.channel)
    for run in runs:
        if not run.has(target_name):
            continue
        error = run.channel(config.channel) - run.channel(target_name)
        keep = run.rows(skip_prelude=config.skip_prelude, controlled_only=config.controlled_only) & np.isfinite(error)
        if keep.sum() < 2:
            continue
        per_run.append(config.scale * _statistic(error[keep], run.t[keep], config))
        evidence.append(Evidence(run.run, f"{config.channel} - {target_name}"))
        spans.append((float(run.t[keep][0] - run.t[0]), float(run.t[keep][-1] - run.t[0])))
    value = center(per_run, config.across_runs) if per_run else None
    worst = int(np.argmax(per_run)) if per_run else 0
    title = f"tracking error {config.statistic}"
    if not per_run:
        return result(config, value=None, unit=config.unit, metric=config.statistic, limits=config.limits, title=title,
                      message="no run with a target on this channel", code="NoTarget")
    return result(config, value=value, unit=config.unit, metric=config.statistic, limits=config.limits, title=title,
                  message=f"{config.across_runs} {config.statistic} of {config.channel} - target over {len(per_run)} runs: {value:.3g} {config.unit}",
                  measurements=(Measure("runs", len(per_run)), Measure("worst_run", max(per_run), config.unit),
                                Measure("best_run", min(per_run), config.unit)),
                  payload={"breakdown": {"runs": [e.path for e in evidence], "values": per_run}},
                  evidence=tuple(evidence), run=evidence[worst].path, t0=spans[worst][0], t1=spans[worst][1],
                  why="compare the worst run against the target in playback; a steady offset is a stop-band (friction/kp) problem, a lag a gain or delay one")


@dataclass(frozen=True)
class StopErrorConfig:
    dwell_min_s: float = 0.25  # a stop is >= this long at zero commanded velocity and one constant target
    settle_window_s: float = 0.05  # its error = mean(q - q_target) over its last this-many seconds
    percentile: float = 90.0
    skip_prelude: bool = True  # the runner's settle at the origin is not a stop
    side: str = "encoder"  # what measured q: encoder (rotor over the ratio) or output (camera, output encoder)
    require_side: str = "output"  # WHY: the encoder is blind to backlash and gear compliance, so an encoder-side
    #                                 pass is only "partial" (bam_rack_twin acceptance T4); it reads WARN here
    scale: float = 1e3
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)


def stops(run: Trajectory, config: StopErrorConfig) -> list[tuple[float, float, float]]:
    """(signed stop error (scaled), dwell start, dwell end) of every dwell followed by more motion; times from the run's start. WHY a window's
    last dwell counts: on a tape the last dwell is parking, not a stop (T4), but a window of a stream (``stream.windows`` labels it) is cut
    by the clock, so the dwell it ends in is a stop still going; without it a joint holding still reads NoStops for as long as it holds."""

    keep = run.rows(skip_prelude=config.skip_prelude)
    # WHY sim2real (2026-11-10, proposal 2 step 4): the dwell rule is T4's, written once in sim2real.metrics.actuator.stops; the rows are this
    # run's tape (the prelude dropped), the times from the run's own start
    return actuator.stops(run.t[keep], run.channel("q_rad")[keep], run.channel("q_target_rad")[keep], run.channel("qd_target_rad_s")[keep],
                          dwell_min_s=config.dwell_min_s, settle_window_s=config.settle_window_s, scale=config.scale, origin=run.t[0],
                          open_end="window_end_s" in run.labels)


def stop_error(config: StopErrorConfig, runs: Sequence[Trajectory]):
    """The ``percentile`` of |stop error| over every stop of every run (T4 when ``limits`` is the joint's allocation)."""

    signed, labels, windows, evidence = [], [], [], []
    for run in runs:
        if not (run.has("q_target_rad") and run.has("qd_target_rad_s")):
            continue
        found = stops(run, config)
        signed += [error for error, _, _ in found]
        windows += [(start, end) for _, start, end in found]
        labels += [run.run] * len(found)
        evidence.append(Evidence(run.run, "q_rad - q_target_rad at dwells"))
    title = f"stop error p{config.percentile:g}"
    metric = f"p{config.percentile:g}"
    if not signed:
        return result(config, value=None, unit=config.unit, metric=metric, limits=config.limits, title=title,
                      message="no dwell found (no stop >= dwell_min_s followed by motion, or that a live window ends in)", code="NoStops")
    summary = actuator.stop_summary(signed, config.percentile)
    value = summary["value"]
    measurements = (Measure("n", summary["n"]), Measure("median", summary["median"], config.unit), Measure("max", summary["max"], config.unit),
                    Measure("mean_signed", summary["mean_signed"], config.unit), Measure("runs", len(evidence)))
    worst = summary["worst"]
    common = dict(measurements=measurements, evidence=tuple(evidence), run=labels[worst], t0=windows[worst][0], t1=windows[worst][1],
                  why="the stop band is about (Coulomb friction + unbalanced gravity) / kp: try Coulomb feed-forward (ff_study) before gains; "
                      "an encoder-side pass still needs the output side (camera)",
                  payload={"x": list(range(len(signed))), "y": signed, "group": labels, "x_label": "stop", "y_label": f"signed stop error ({config.unit})"})
    message = f"{metric} {value:.3g} {config.unit} over {len(signed)} stops in {len(evidence)} runs ({config.side} side)"
    within = result(config, value=value, unit=config.unit, metric=metric, limits=config.limits, title=title, message=message, **common)
    if within.verdict == "pass" and config.require_side and config.side != config.require_side:
        return result(config, value=value, unit=config.unit, metric=metric, limits=config.limits, title=title,
                      message=message + f"; within the limit, but measured on the {config.side} side, not the {config.require_side}",
                      verdict=WARN, code="MeasuredOnEncoderSide" if config.side == "encoder" else "NotOnRequiredSide", **common)
    return within


__all__ = ["TrackingErrorConfig", "tracking_error", "StopErrorConfig", "stop_error", "stops", "NOT_RUN"]
