"""Counts over events: the share of good ones (with its confidence), the mean operating time between bad ones, and how old each gate's
newest evidence is.

- ``share``: Zach's ``PercentTask`` with its bug fixed (an explicit ``better``, so 100 % success can never read ERROR) and the Wilson
  score interval beside the share (Wilson 1927; Newcombe 1998 compares it with the alternatives and recommends it for small n).
- ``event_interval``: MTBF / MTTF as operating time over failures (IEC 60050-192 "mean operating time between failures"); with no
  failures the operating time is only a lower bound, and it is judged as one.
- ``staleness``: the age of each gate's newest evidence against its validity window (IEC 61508's proof-test interval; OBD readiness):
  a gate whose evidence is older than its window is stale, a gate with no evidence is never green.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from ..core import FAIL, NOT_RUN, PASS, WARN, Evidence, Limits, Measure, Result, judge, result, worst


@dataclass(frozen=True)
class Event:
    t: float  # s (epoch or from a run's start; one clock per list)
    ok: bool  # the outcome a share counts (a hold that converged, a reply lost, a grasp that held)
    labels: Mapping[str, Any] = field(default_factory=dict)  # what a share slices by (class, joint speed bin, bus)
    code: str = ""  # what event_interval filters on ("fault", "intervention")
    evidence: str = ""  # the run or record behind it


# ----- share ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ShareConfig:
    better: str = "higher"  # higher (success) | lower (loss): which side of the interval is conservative
    statistic: str = "share"  # share: true events / events | count: true events
    judge_on: str = "share"  # share: the point estimate | bound: the Wilson bound on the bad side (lower when higher is better)
    min_events: int = 1  # fewer events than this = not run
    confidence: float = 0.95  # the Wilson interval's coverage
    slice_by: str = ""  # a label key: the payload gets one share per value
    scale: float = 1.0  # 100 for %
    unit: str = "1"
    limits: Limits = field(default_factory=Limits)


def wilson(true: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """The Wilson score interval for ``true`` of ``total`` (Wilson 1927, eq. as in Newcombe 1998 method 3)."""

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = true / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z / (1 + z * z / total) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return max(0.0, centre - half), min(1.0, centre + half)


def share(config: ShareConfig, events: Sequence[Event]):
    """Share (or count) of true events, with the Wilson interval; optionally judged on its conservative bound."""

    events = list(events)
    title = "share" if config.statistic == "share" else "count"
    unit = config.unit if config.statistic == "share" else "count"
    if len(events) < max(1, config.min_events):
        return result(config, value=None, unit=unit, metric=config.statistic, limits=config.limits, title=title, better=config.better,
                      message=f"{len(events)} event(s): needs {max(1, config.min_events)}", code="TooFewEvents")
    true = sum(bool(e.ok) for e in events)
    low, high = wilson(true, len(events), config.confidence)
    point = true / len(events)
    slices: dict[str, dict[str, Any]] = {}
    if config.slice_by:
        for event in events:
            key = str(event.labels.get(config.slice_by, ""))
            row = slices.setdefault(key, {"events": 0, "true": 0})
            row["events"] += 1
            row["true"] += bool(event.ok)
        for row in slices.values():
            row["share"] = config.scale * row["true"] / row["events"]
    if config.statistic == "count":
        value = true
    else:
        value = config.scale * point
    verdict = code = None
    if config.judge_on == "bound" and config.statistic == "share":
        verdict, code = judge(config.scale * (low if config.better == "higher" else high), config.limits)
    return result(config, value=value, unit=unit, metric=config.statistic, limits=config.limits, title=title, better=config.better,
                  verdict=verdict, code=code,
                  message=f"{true} of {len(events)} true: {config.scale * point:.4g} {config.unit} "
                          f"(Wilson {config.confidence:.0%}: {config.scale * low:.4g} to {config.scale * high:.4g})",
                  measurements=(Measure("events", len(events)), Measure("true", true), Measure("wilson_low", config.scale * low, config.unit),
                                Measure("wilson_high", config.scale * high, config.unit)),
                  payload={"slices": slices} if slices else None,
                  evidence=tuple(Evidence(e.evidence, "ok") for e in events if e.evidence)[:50],
                  why="look at the slices: a share that is low in one class or speed bin is a coverage or tuning problem there")


# ----- event_interval -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EventLog:
    events: Sequence[Event]
    operating: Sequence[tuple[float, float]]  # (start, end) s on the events' clock: when the thing was running


@dataclass(frozen=True)
class EventIntervalConfig:
    codes: tuple[str, ...] = ()  # the event codes that count; empty = every event
    min_events: int = 1  # fewer: the operating time is only a lower bound (judged as one)
    scale: float = 1 / 3600  # s -> h
    unit: str = "h"
    limits: Limits = field(default_factory=Limits)


def _merged(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def event_interval(config: EventIntervalConfig, data: EventLog):
    """Operating time / counted events inside it. With fewer than ``min_events``, the operating time is a lower bound on the mean
    interval: it passes only if the bound already clears the limits, else not run (WHY: "no failures yet" is not a measured MTBF)."""

    title = "mean time between events"
    intervals = _merged(data.operating)
    operating_s = sum(b - a for a, b in intervals)
    if operating_s <= 0:
        return result(config, value=None, unit=config.unit, metric="mean_interval", limits=config.limits, title=title, better="higher",
                      message="no operating time recorded", code="NoOperatingTime")
    wanted = [e for e in data.events if not config.codes or e.code in config.codes]
    inside = [e for e in wanted if any(a <= e.t <= b for a, b in intervals)]
    hours = operating_s * config.scale
    common = dict(unit=config.unit, metric="mean_interval", limits=config.limits, title=title, better="higher",
                  evidence=tuple(Evidence(e.evidence, e.code) for e in inside if e.evidence)[:50])
    if len(inside) < max(1, config.min_events):
        censored = (Measure("events", len(inside)), Measure("outside", len(wanted) - len(inside)),
                    Measure("operating", hours, config.unit), Measure("censored", 1))
        bound = f"{len(inside)} event(s) in {hours:.4g} {config.unit} of operation: the mean interval is at least {hours:.4g} {config.unit}"
        if judge(hours, config.limits)[0] == PASS:
            return result(config, value=hours, message=bound, measurements=censored, **common)
        return result(config, value=hours, message=bound + ", which does not settle the limit", verdict=NOT_RUN, code="TooFewEvents",
                      measurements=censored, **common)
    value = hours / len(inside)
    return result(config, value=value, message=f"{len(inside)} event(s) in {hours:.4g} {config.unit} of operation: one per {value:.4g} {config.unit}",
                  measurements=(Measure("events", len(inside)), Measure("outside", len(wanted) - len(inside)),
                                Measure("operating", hours, config.unit), Measure("censored", 0)),
                  why="read the events' codes in the fault history: one recurring code is one cause", **common)


# ----- staleness ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvidenceStamps:
    now: float  # s, the same clock as the stamps
    stamps: Sequence[Any]  # (gate, stamp) pairs, or Results (gate = its diagnostic name, stamp = its evidence time)
    reset_at: float | None = None  # a restart / replug / firmware change: evidence stamped before it no longer counts


@dataclass(frozen=True)
class StalenessConfig:
    valid_for: tuple[tuple[str, float], ...] = (("comms*", 86400.0), ("motion*", 604800.0))  # (gate pattern, s): 24 h comms, 7 d motion
    default_valid_s: float = 86400.0  # a gate no pattern matches
    warn_fraction: float = 0.8  # warn once the evidence is this far through its window
    expected: tuple[str, ...] = ()  # gates that must have evidence: one without any is not run, never green
    scale: float = 1 / 3600  # s -> h
    unit: str = "h"


def _window(config: StalenessConfig, gate: str) -> float:
    return next((float(seconds) for pattern, seconds in config.valid_for if fnmatch.fnmatchcase(gate, pattern)), config.default_valid_s)


def staleness(config: StalenessConfig, data: EvidenceStamps):
    """Per gate, the age of its newest evidence over its window: > 1 fails (Stale), > warn_fraction warns, no evidence is not run."""

    newest: dict[str, float] = {}
    for item in data.stamps:
        gate, stamp = (item.diagnostic, item.stamp) if isinstance(item, Result) else (str(item[0]), item[1])
        if stamp is None or not math.isfinite(stamp) or stamp > data.now:
            continue
        if data.reset_at is not None and stamp < data.reset_at:
            continue
        newest[gate] = max(stamp, newest.get(gate, -math.inf))
    gates = sorted(set(newest) | set(config.expected))
    title = "evidence age"
    if not newest:
        return result(config, value=None, unit=config.unit, metric="age", limits=Limits(), title=title,
                      message="no gate has evidence" + (" since the reset" if data.reset_at is not None else ""), code="NoEvidence")
    rows = []
    for gate in gates:
        if gate not in newest:
            rows.append((gate, None, _window(config, gate), NOT_RUN, "NoEvidence"))
            continue
        age, window = data.now - newest[gate], _window(config, gate)
        ratio = age / window
        verdict, code = (FAIL, "Stale") if ratio > 1 else (WARN, "NearlyStale") if ratio > config.warn_fraction else (PASS, "")
        rows.append((gate, age, window, verdict, code))
    verdict = worst(row[3] for row in rows)
    culprit = next(row for row in rows if row[3] == verdict) if verdict != PASS else None
    aged = [row for row in rows if row[1] is not None]
    head = max(aged, key=lambda row: row[1] / row[2])
    value = head[1] * config.scale
    stale = [row[0] for row in rows if row[3] == FAIL]
    missing = [row[0] for row in rows if row[1] is None]
    return result(config, value=value, unit=config.unit, metric="age", limits=Limits(), title=title, verdict=verdict,
                  code=culprit[4] if culprit else "",
                  message=f"{head[0]}: newest evidence {value:.4g} {config.unit} old (valid {head[2] * config.scale:.4g} {config.unit})"
                          + (f"; stale: {', '.join(stale)}" if stale else "") + (f"; no evidence: {', '.join(missing)}" if missing else ""),
                  measurements=(Measure("gates", len(rows)), Measure("stale", len(stale)), Measure("missing", len(missing))),
                  payload={"x": [row[0] for row in aged], "y": [row[1] * config.scale for row in aged],
                           "valid": [row[2] * config.scale for row in aged], "x_label": "gate", "y_label": f"age ({config.unit})"},
                  why="re-run the stale gates' checks; a restart, replug or firmware change resets every gate's evidence")


__all__ = ["Event", "EventLog", "EvidenceStamps", "ShareConfig", "share", "wilson", "EventIntervalConfig", "event_interval",
           "StalenessConfig", "staleness"]
