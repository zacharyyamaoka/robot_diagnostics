"""Sim vs real, judged against the rig's own floor.

- ``repeatability_floor`` (gate T1): run one tape N times from one start; the floor is the median over all pairs of
  RMS(e_a - e_b). Zach, DP - Reality as Oracle: "FANTASTIC POINT!! So important to baseline this and repeat it ... Two
  aspects accuracy and repeatability".
- ``model_error``: RMS(real - model) per run, for any number of models (a twin, a perfect servo, a lagged target).
  Zach, 2026-10-28: "we should support many way comparisons basically".
- ``fidelity_ratio`` (gates T2 on q, T3 on tau): model error / floor, per tape class.
- ``move_hold_split``, ``rest_offset``, ``split_fidelity`` (T2 split, 2026-11-04): the error's moving part and hold part, where the model
  comes to rest against the real joint, and the judgement moves / floor's moving part and |hold offset| / floor's hold part <= 1.5.

A model is scored on ``data.scored_rows``: the real run's position-mode rows, or (``controlled_by="both"``) the rows both ran in mode 10.
The first three use ``data.pairwise_rms`` / ``on_common_clock``: one ruler for real-vs-real and model-vs-real.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np

from ..core import FAIL, PASS, WARN, Evidence, Limits, Measure, result
from ..data import CONTROLLED_BY, TARGET_OF, Trajectory, center, difference, on_common_clock, pairwise_rms, rms, scored_rows, target_of


def _signal(run: Trajectory, channel: str, relative_to_target: bool, skip_prelude: bool) -> tuple[np.ndarray, np.ndarray]:
    """(t from the tape's start, signal): ``channel - channel_target`` when asked, else the raw channel; finite rows only."""

    values = run.channel(channel)
    if relative_to_target:
        values = values - run.channel(TARGET_OF[channel])
    keep = np.isfinite(values) & run.rows(skip_prelude=skip_prelude)
    t = run.t[keep]
    return t - t[0], values[keep]


@dataclass(frozen=True)
class FloorConfig:
    channel: str = "q_rad"
    relative_to_target: bool = True  # q_rad: the tracking error; tau_nm: the raw torque (gate T3's floor)
    skip_prelude: bool = True
    clock: str = "common"  # common: one grid of step dt over each pair's overlap (bam_rack_twin T1, the oracle);
    #                        a: the first run's own samples (data-ui PEP 0010 difference(a, b))
    dt: float = 0.002  # the common clock's step
    min_repeats: int = 5  # T1: >= 5 identical repeats
    scale: float = 1e3
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)  # a floor is a reading, not a target; a bound flags a noisy rig


def repeatability_floor(config: FloorConfig, repeats: Sequence[Trajectory]):
    """Median pairwise RMS of the signal over identical repeats of one tape (each pair on its own overlap)."""

    title = f"repeatability floor ({config.channel})"
    if len(repeats) < 2:
        return result(config, value=None, unit=config.unit, metric="floor", limits=config.limits, title=title,
                      message=f"{len(repeats)} repeat(s): a floor needs at least 2", code="TooFewRepeats")
    missing = [run.run or "a repeat" for run in repeats if not run.has(config.channel)]
    if missing:  # WHY not run, never a KeyError: a part whose runs lack the channel has no floor yet, which is not a broken diagnostic
        return result(config, value=None, unit=config.unit, metric="floor", limits=config.limits, title=title,
                      message=f"{len(missing)} of {len(repeats)} repeats have no {config.channel} column", code="NoChannel")
    series = [_signal(run, config.channel, config.relative_to_target, config.skip_prelude) for run in repeats]
    pairs = pairwise_rms(series, clock=config.clock, dt=config.dt)
    values = [config.scale * value for _, _, value in pairs]
    floor = center(values, "median")
    n = len(repeats)
    cells = [[0.0] * n for _ in range(n)]
    for (i, j, _), value in zip(pairs, values):
        cells[i][j] = cells[j][i] = value
    names = [run.run for run in repeats]
    worst_i = pairs[max(range(len(values)), key=values.__getitem__)][0]
    common = dict(measurements=(Measure("repeats", n), Measure("pairs", len(pairs)), Measure("max_pair", max(values), config.unit)),
                  payload={"rows": names, "cols": names, "cells": cells}, evidence=tuple(Evidence(name, config.channel) for name in names),
                  run=names[worst_i], t0=0.0, t1=float(series[worst_i][0][-1]),
                  why="repeats must start from one pose at one temperature; a pair far above the rest is a disturbed run")
    message = f"median of {len(pairs)} pairwise RMS over {n} repeats: {floor:.4g} {config.unit}"
    if n < config.min_repeats:
        return result(config, value=floor, unit=config.unit, metric="floor", limits=config.limits, title=title,
                      message=message + f" (needs {config.min_repeats} repeats)", verdict=FAIL, code="TooFewRepeats", **common)
    return result(config, value=floor, unit=config.unit, metric="floor", limits=config.limits, title=title, message=message, **common)


@dataclass(frozen=True)
class ModelErrorConfig:
    channel: str = "q_rad"
    skip_prelude: bool = False
    controlled_only: bool = True  # score where the real controller was tracking (mode 10)
    controlled_by: str = "real"  # real: the real run's mode decides a row; both: the real run's AND the model's (data.scored_rows)
    clock: str = "rows"  # rows: row by row on the real run's clock (a replay); a: difference(real, model) on the real
    #                      run's samples, the model interpolated (data-ui PEP 0010 §4.4)
    across_runs: str = "median"
    scale: float = 1e3
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)

    def __post_init__(self):
        if self.controlled_by not in CONTROLLED_BY:
            raise ValueError(f"controlled_by {self.controlled_by!r} not in {CONTROLLED_BY}")


@dataclass(frozen=True)
class ModelRow:
    """One (model, run) score: what ``fidelity_ratio`` consumes. Produced by ``model_error`` or read from a twin's file."""

    model: str
    run_id: str
    cls: str
    error: float  # in the floor's unit


def model_error(config: ModelErrorConfig, pairs: Sequence[tuple[Trajectory, Trajectory]]):
    """RMS(real - model) per run; the headline is ``across_runs`` over runs. ``payload.breakdown.rows`` feeds fidelity_ratio."""

    rows = []
    for real, model in pairs:
        keep = scored_rows(real, model, skip_prelude=config.skip_prelude, controlled_only=config.controlled_only,
                           controlled_by=config.controlled_by)
        if config.clock == "rows":
            gap = model.channel(config.channel)[keep] - real.channel(config.channel)[keep]
        else:
            mine = model.rows(skip_prelude=config.skip_prelude, controlled_only=config.controlled_only)
            gap = difference((real.t[keep] - real.t[0], real.channel(config.channel)[keep]),
                             (model.t[mine] - model.t[0], model.channel(config.channel)[mine]))[1]
        gap = gap[np.isfinite(gap)]
        rows.append(ModelRow(model.situation, real.run, str(real.labels.get("class", "")), config.scale * rms(gap)))
    models = sorted({row.model for row in rows})
    title = f"model error ({config.channel})"
    if not rows:
        return result(config, value=None, unit=config.unit, metric="rms", limits=config.limits, title=title,
                      message="no (real, model) pair", code="NoPairs")
    value = center([row.error for row in rows], config.across_runs)
    worst = max(rows, key=lambda row: row.error)
    return result(config, value=value, unit=config.unit, metric="rms", limits=config.limits, title=title, run=worst.run_id,
                  message=f"{config.across_runs} RMS(real - {', '.join(models)}) over {len(rows)} runs: {value:.4g} {config.unit}",
                  measurements=(Measure("runs", len(rows)), Measure("worst_run", max(row.error for row in rows), config.unit)),
                  payload={"breakdown": {"rows": [[row.model, row.run_id, row.cls, row.error] for row in rows]}},
                  evidence=tuple(Evidence(row.run_id, f"{config.channel} real - {row.model}") for row in rows))


def rows_of(model_error_result) -> list[ModelRow]:
    return [ModelRow(*row) for row in (model_error_result.payload or {}).get("breakdown", {}).get("rows", [])]


@dataclass(frozen=True)
class FidelityInput:
    floor: float  # same unit as the rows' error
    rows: Sequence[ModelRow]
    floor_kind: str = "repeats"  # repeats | proxy: a proxy floor can never pass (T2 "partial" at best)
    floor_class: str = ""  # the tape class the floor was measured on
    unit: str = "mrad"  # the floor's (and the rows') unit


@dataclass(frozen=True)
class FidelityRatioConfig:
    model: str = "twin"  # the model judged; every other model in the rows is drawn beside it
    on_floor_class: bool = True  # T2: judge the twin on the floor's own tape class (WHY: bam_rack_twin acceptance)
    center: str = "median_high"  # bam_rack_twin takes sorted(v)[n // 2] per class
    class_limit: float | None = 2.0  # T2: every dataset class <= 2.0
    limits: Limits = field(default_factory=lambda: Limits.at_most(1.5))  # T2: <= 1.5 on the floor's tape


def fidelity_ratio(config: FidelityRatioConfig, data: FidelityInput):
    """Model error / floor per (model, class); the headline is the judged model on the floor's class."""

    by: dict[tuple[str, str], list[float]] = {}
    for row in data.rows:
        by.setdefault((row.model, row.cls), []).append(row.error / data.floor)
    ratios = {key: center(values, config.center) for key, values in by.items()}
    classes = sorted({cls for _, cls in ratios})
    models = sorted({model for model, _ in ratios})
    title = f"fidelity ratio ({config.model} / floor)"
    mine = {cls: value for (model, cls), value in ratios.items() if model == config.model}
    if not mine or not data.floor:
        return result(config, value=None, unit="1", metric="ratio", limits=config.limits, title=title,
                      message=f"no rows for model {config.model!r}" if data.floor else "no floor", code="NoRows" if data.floor else "NoFloor")
    if config.on_floor_class and data.floor_class in mine:
        value, basis = mine[data.floor_class], f"on the floor's {data.floor_class} tape"
    else:
        value = center([v for (model, _), values in by.items() if model == config.model for v in values], config.center)
        basis = "over all classes"
    worst_class = max(mine, key=mine.get)
    payload = {"x": [cls for model in models for cls in classes if (model, cls) in ratios],
               "y": [ratios[(model, cls)] for model in models for cls in classes if (model, cls) in ratios],
               "group": [model for model in models for cls in classes if (model, cls) in ratios],
               "x_label": "tape class", "y_label": "model error / floor"}
    measurements = (Measure("floor", data.floor, data.unit), Measure("worst_class_ratio", mine[worst_class]),
                    Measure("classes", len(mine)), Measure("models", len(models)))
    message = f"{config.model} is {value:.3g}x the floor {basis}; worst class {worst_class} {mine[worst_class]:.3g}x"
    judged_rows = [row for row in data.rows if row.model == config.model]
    worst_row = max(judged_rows, key=lambda row: row.error)
    common = dict(measurements=measurements, payload=payload, run=worst_row.run_id,
                  why="the term budget names the missing physics; a class far above the others is outside the fitted envelope")
    judged = result(config, value=value, unit="1", metric="ratio", limits=config.limits, title=title, message=message, **common)
    if config.class_limit is not None and mine[worst_class] > config.class_limit and judged.verdict == PASS:
        return result(config, value=value, unit="1", metric="ratio", limits=config.limits, title=title, verdict=FAIL,
                      code="ClassAboveLimit", message=message + f" > {config.class_limit}", **common)
    if data.floor_kind != "repeats" and judged.verdict == PASS:
        return result(config, value=value, unit="1", metric="ratio", limits=config.limits, title=title, verdict=WARN,
                      code="ProxyFloor", message=message + " (on a proxy floor, not identical repeats)", **common)
    return judged


# --------------------------------------------------------------------------- the moving part and the hold part (T2 split)
# WHY a split (bam_rack_twin acceptance T2, 2026-10-25, Zach's default pick taken while he was away and his to overrule): with no integrator
# a stop lands anywhere in the friction band and the rack does not repeat where, so one RMS mixes two questions. Moves are judged by RMS over
# the floor's moving part; holds by where the model comes to rest against the floor's hold part (``rest_offset``). The definitions below are
# bam_rack_twin's (round7_study.floor_split / rms_split, budget.hold_rest_differences, stations_d_pass.t2_of), reproduced to 1e-9.
HOLD_SPEED_RAD_S = 0.02  # |commanded speed| at or under this is a hold (bam_rack_twin budget.HOLD_SPEED_RAD_S)
PARTS = ("all", "moves", "holds")


def _rms_or_none(values: np.ndarray, scale: float) -> float | None:
    return scale * rms(values) if values.size else None


def _speed(run: Trajectory, channel: str) -> np.ndarray:
    """The commanded speed that classes a row as hold or move; a run without it is all holds (nothing commanded to move)."""

    return run.channel(channel) if run.has(channel) else np.zeros(len(run.t))


@dataclass(frozen=True)
class MoveHoldSplitConfig:
    channel: str = "q_rad"
    speed_channel: str = "qd_target_rad_s"  # the commanded speed: a row is a hold where |speed| <= hold_speed
    hold_speed: float = HOLD_SPEED_RAD_S
    skip_prelude: bool = True  # the tape: the runner's move to the origin is not part of it
    # repeats (the floor's parts): the tracking error, each pair on the common clock (FloorConfig's ruler)
    relative_to_target: bool = True
    dt: float = 0.002
    min_repeats: int = 2
    # (real, model) pairs: row by row on the real run's clock, the rows ``scored_rows`` keeps
    controlled_only: bool = True
    controlled_by: str = "both"  # the rack's ruler: rows both the real run and the model ran in position mode
    on_class: str = ""  # the headline run: the worst (by its whole RMS) run of this tape class; "" = of every run
    scale: float = 1e3
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)  # a reading; split_fidelity is the judgement

    def __post_init__(self):
        if self.controlled_by not in CONTROLLED_BY:
            raise ValueError(f"controlled_by {self.controlled_by!r} not in {CONTROLLED_BY}")


def _pair_parts(config: MoveHoldSplitConfig, a: Trajectory, b: Trajectory) -> dict[str, float | None]:
    """round7_study.floor_split's pair: each run's tape from 0, a dt grid over the overlap, RMS(e_a - e_b) over all points, the holds
    (a's commanded speed, NaN taken as 0, interpolated: |v| <= hold_speed) and the moves."""

    series = []
    for run in (a, b):
        values = run.channel(config.channel)
        if config.relative_to_target:
            values = values - run.channel(TARGET_OF[config.channel])
        keep = np.isfinite(values) & run.rows(skip_prelude=config.skip_prelude)
        speed = _speed(run, config.speed_channel)[keep]
        t = run.t[keep]
        series.append((t - t[0], values[keep], np.where(np.isfinite(speed), speed, 0.0)))
    (t_a, e_a, v_a), (t_b, e_b, _) = series
    grid = np.arange(0.0, min(t_a[-1], t_b[-1]), config.dt)
    gap = np.interp(grid, t_a, e_a) - np.interp(grid, t_b, e_b)
    hold = np.abs(np.interp(grid, t_a, v_a)) <= config.hold_speed
    return {"all": _rms_or_none(gap, config.scale), "moves": _rms_or_none(gap[~hold], config.scale),
            "holds": _rms_or_none(gap[hold], config.scale), "hold_fraction": float(hold.mean()) if hold.size else None}


def _run_parts(config: MoveHoldSplitConfig, real: Trajectory, model: Trajectory) -> dict[str, float | None]:
    """round7_study.rms_split: RMS(real - model) on the scored rows, on their hold rows and on their moving rows."""

    rows = scored_rows(real, model, skip_prelude=config.skip_prelude, controlled_only=config.controlled_only, controlled_by=config.controlled_by)
    hold = rows & (np.abs(_speed(real, config.speed_channel)) <= config.hold_speed)
    gap = real.channel(config.channel) - model.channel(config.channel)
    finite = np.isfinite(gap)
    return {"all": _rms_or_none(gap[rows & finite], config.scale), "moves": _rms_or_none(gap[rows & ~hold & finite], config.scale),
            "holds": _rms_or_none(gap[hold & finite], config.scale), "hold_fraction": float(hold.sum() / rows.sum()) if rows.any() else None}


def _middle(values) -> float | None:
    values = [value for value in values if value is not None]
    return center(values, "median") if values else None


def move_hold_split(config: MoveHoldSplitConfig, data):
    """A trajectory error split into its moving part and its hold part (mrad); the headline is the moving part.

    ``data`` is either identical repeats of one tape (``Sequence[Trajectory]``: the floor's parts, the median over every pair) or
    ``(real, model)`` pairs (per run; the headline run is the worst of ``on_class`` by its whole RMS, as T2 judges the worst pickplace tape)."""

    data = list(data)
    title = f"move / hold split ({config.channel})"
    common = dict(unit=config.unit, metric="moves", limits=config.limits, title=title)
    if not data:
        return result(config, value=None, message="no runs", code="NoRuns", **common)
    if all(isinstance(item, Trajectory) for item in data):
        if len(data) < max(2, config.min_repeats):
            return result(config, value=None, message=f"{len(data)} repeat(s): a split floor needs at least {max(2, config.min_repeats)}",
                          code="TooFewRepeats", **common)
        missing = [run.run or "a repeat" for run in data if not run.has(config.channel)]
        if missing:
            return result(config, value=None, message=f"{len(missing)} of {len(data)} repeats have no {config.channel} column", code="NoChannel", **common)
        pairs = [(i, j, _pair_parts(config, data[i], data[j])) for i, j in itertools.combinations(range(len(data)), 2)]
        parts = {name: _middle(part[name] for _, _, part in pairs) for name in (*PARTS, "hold_fraction")}
        names = [run.run for run in data]
        rows = [[names[i], names[j], *(part[name] for name in (*PARTS, "hold_fraction"))] for i, j, part in pairs]
        counts = (Measure("pairs", len(pairs)), Measure("repeats", len(data)))
        run, basis = "", f"median over {len(pairs)} pairs of {len(data)} repeats"
        evidence = tuple(Evidence(name, config.channel) for name in names)
    elif all(isinstance(item, tuple) and len(item) == 2 for item in data):
        missing = [real.run or "a run" for real, model in data if not (real.has(config.channel) and model.has(config.channel))]
        if missing:
            return result(config, value=None, message=f"{len(missing)} of {len(data)} pairs have no {config.channel} column", code="NoChannel", **common)
        scored = [(real, model, _run_parts(config, real, model)) for real, model in data]
        rows = [[model.situation, real.run, str(real.labels.get("class", "")), *(part[name] for name in (*PARTS, "hold_fraction"))]
                for real, model, part in scored]
        judged = [item for item in scored if item[2]["all"] is not None]
        mine = [item for item in judged if str(item[0].labels.get("class", "")) == config.on_class] if config.on_class else judged
        if not mine:
            return result(config, value=None, message=f"no scored run{' of class ' + config.on_class if config.on_class else ''}",
                          code="NoScoredRows", payload={"breakdown": {"rows": rows}}, **common)
        real, model, parts = max(mine, key=lambda item: item[2]["all"])
        counts = (Measure("runs", len(scored)),)
        run, basis = real.run, f"the worst {config.on_class + ' ' if config.on_class else ''}run ({real.run}) of {len(scored)}"
        evidence = tuple(Evidence(real.run, f"{config.channel} real - {model.situation}") for real, model, _ in scored)
    else:
        raise TypeError("move_hold_split takes repeats (Trajectory, ...) or (real, model) pairs")
    measurements = (*(Measure(name, parts[name], config.unit) for name in PARTS), Measure("hold_fraction", parts["hold_fraction"]), *counts)
    payload = {"breakdown": {"rows": rows}}
    if parts["moves"] is None:
        return result(config, value=None, message=f"no moving rows ({basis})", code="NoMovingRows", measurements=measurements, payload=payload,
                      run=run, evidence=evidence, **common)
    holds = "no hold rows" if parts["holds"] is None else f"holds {parts['holds']:.4g} {config.unit}"
    return result(config, value=parts["moves"], message=f"moves {parts['moves']:.4g} {config.unit}, {holds} ({basis})", measurements=measurements,
                  payload=payload, run=run, evidence=evidence,
                  why="a moving part far above the hold part is the dynamics; a hold part above it is where stops land", **common)


@dataclass(frozen=True)
class RestOffsetConfig:
    channel: str = "q_rad"
    speed_channel: str = "qd_target_rad_s"
    hold_speed: float = HOLD_SPEED_RAD_S
    min_hold_s: float = 0.3  # a hold shorter than this has no rest (bam_rack_twin budget.HOLD_MIN_S)
    tail_s: float = 0.2  # the rest is the mean over the hold's last tail_s, after the stop's transient (budget.HOLD_TAIL_S)
    skip_prelude: bool = True
    controlled_only: bool = True
    controlled_by: str = "both"
    min_holds: int = 4  # stations_d_pass.t2_of: fewer holds than this give no median (not run)
    scale: float = 1e3
    unit: str = "mrad"
    limits: Limits = field(default_factory=Limits)  # a reading; split_fidelity judges |offset| against the floor's hold part

    def __post_init__(self):
        if self.controlled_by not in CONTROLLED_BY:
            raise ValueError(f"controlled_by {self.controlled_by!r} not in {CONTROLLED_BY}")


def rest_holds(config: RestOffsetConfig, real: Trajectory, model: Trajectory) -> list[dict[str, float]]:
    """bam_rack_twin budget.hold_rest_differences: every hold of the real run's commanded path (scored rows, a finite target, |speed| <=
    hold_speed, lasting >= min_hold_s), its rest = mean(target - q) over the last tail_s, folded to the real run's side, so
    ``model_minus_real`` > 0 is a model resting FARTHER from the target than the real joint, < 0 nearer or across it."""

    target = real.channel(target_of(config.channel))
    rows = (scored_rows(real, model, skip_prelude=config.skip_prelude, controlled_only=config.controlled_only, controlled_by=config.controlled_by)
            & np.isfinite(target) & (np.abs(_speed(real, config.speed_channel)) <= config.hold_speed))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], rows.astype(int), [0]])))
    t, q_real, q_model = real.t - real.t[0], real.channel(config.channel), model.channel(config.channel)
    holds = []
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if t[end - 1] - t[start] < config.min_hold_s:
            continue
        tail = np.arange(start, end)[t[start:end] >= t[end - 1] - config.tail_s]
        rest_real = float(np.mean(target[tail] - q_real[tail]))
        rest_model = float(np.mean(target[tail] - q_model[tail]))
        side = 1.0 if rest_real >= 0.0 else -1.0
        holds.append({"t_start_s": float(t[start]), "t_end_s": float(t[end - 1]), "real": config.scale * side * rest_real,
                      "model": config.scale * side * rest_model, "model_minus_real": config.scale * side * (rest_model - rest_real)})
    return holds


def rest_offset(config: RestOffsetConfig, pairs):
    """The model's steady hold offset at rest against the real joint: the median, over every hold of every run, of where the model came to
    rest minus where the real joint did (folded to the real's side; + = the model rests farther from the target)."""

    pairs = list(pairs)
    title = f"rest offset ({config.channel})"
    common = dict(unit=config.unit, metric="median", limits=config.limits, title=title)
    if not pairs:
        return result(config, value=None, message="no (real, model) pair", code="NoPairs", **common)
    found = [(real, model, hold) for real, model in pairs for hold in rest_holds(config, real, model)]
    if len(found) < config.min_holds:
        return result(config, value=None, message=f"{len(found)} hold(s) at rest over {len(pairs)} runs: the offset needs {config.min_holds}",
                      code="TooFewHolds", measurements=(Measure("holds", len(found)),), **common)
    d = np.array([hold["model_minus_real"] for _, _, hold in found])
    value = float(np.median(d))
    worst = max(found, key=lambda item: abs(item[2]["model_minus_real"]))
    measurements = (Measure("holds", int(d.size)), Measure("median_abs", float(np.median(np.abs(d))), config.unit),
                    Measure("p10", float(np.percentile(d, 10)), config.unit), Measure("p90", float(np.percentile(d, 90)), config.unit),
                    Measure("farther_fraction", float(np.mean(d > 0.0))), Measure("runs", len(pairs)))
    payload = {"x": [hold["t_start_s"] for _, _, hold in found], "y": d.tolist(), "group": [real.run for real, _, _ in found],
               "x_label": "hold start (s)", "y_label": f"model - real rest ({config.unit}, + = model farther)"}
    return result(config, value=value, measurements=measurements, payload=payload, run=worst[0].run, t0=worst[2]["t_start_s"], t1=worst[2]["t_end_s"],
                  message=f"median model - real rest {value:+.4g} {config.unit} over {d.size} holds ({float(np.mean(d > 0.0)):.0%} farther)",
                  evidence=tuple(Evidence(real.run, f"{config.channel} rest, real vs {model.situation}") for real, model in pairs),
                  why="a model resting farther out captures the stop early (friction too high); nearer, too late", **common)


@dataclass(frozen=True)
class SplitFidelityInput:
    floor_moves: float | None  # the floor's moving part (move_hold_split over repeats), in ``unit``
    floor_holds: float | None  # the floor's hold part
    moves: float | None  # the model's moving part on its headline run (move_hold_split over pairs)
    rest_offset: float | None  # the model's median rest offset (rest_offset); judged by its size
    floor_kind: str = "repeats"  # repeats | proxy: a proxy floor can never pass
    unit: str = "mrad"


def split_input(floor, split, rest, *, floor_kind: str = "repeats") -> SplitFidelityInput:
    """The three Results (the floor's split, the model's split, the rest offset) as ``split_fidelity``'s input."""

    def part(out, key):
        return next((m.value for m in out.measurements if m.key == key), None) if out is not None else None

    return SplitFidelityInput(part(floor, "moves"), part(floor, "holds"), None if split is None else split.value,
                              None if rest is None else rest.value, floor_kind, floor.unit if floor is not None else "mrad")


@dataclass(frozen=True)
class SplitFidelityConfig:
    # T2 split: moves <= 1.5x the floor's moving part AND |hold offset| <= 1.5x the floor's hold part (bam_rack_twin acceptance
    # FIDELITY_TAPES, an agent's default, "Zach to confirm"); one bound for both parts, the binding part heads
    limits: Limits = field(default_factory=lambda: Limits.at_most(1.5))


def split_fidelity(config: SplitFidelityConfig, data: SplitFidelityInput):
    """T2's split judgement: moves / floor's moving part and |rest offset| / floor's hold part, each against ``limits``; the headline is the
    larger (the part that binds)."""

    title = "split fidelity (moves and hold offset / floor)"
    common = dict(unit="1", metric="ratio", limits=config.limits, title=title)
    for name, code in (("floor_moves", "NoFloor"), ("floor_holds", "NoFloor"), ("moves", "NoMoves"), ("rest_offset", "NoRestOffset")):
        value = getattr(data, name)
        if value is None or not math.isfinite(value) or (name.startswith("floor") and value <= 0.0):
            return result(config, value=None, message=f"no {name.replace('_', ' ')}", code=code, **common)
    moves, hold = data.moves / data.floor_moves, abs(data.rest_offset) / data.floor_holds
    value = max(moves, hold)
    binding = "moves" if moves >= hold else "hold offset"
    measurements = (Measure("moves_over_floor", moves), Measure("hold_offset_over_floor", hold), Measure("floor_moves", data.floor_moves, data.unit),
                    Measure("floor_holds", data.floor_holds, data.unit), Measure("moves", data.moves, data.unit),
                    Measure("rest_offset", data.rest_offset, data.unit))
    payload = {"x": ["moves", "hold offset"], "y": [moves, hold], "x_label": "part", "y_label": "model / floor's part"}
    message = f"moves {moves:.3g}x, hold offset {hold:.3g}x the floor's parts ({binding} binds)"
    out = result(config, value=value, measurements=measurements, payload=payload, message=message,
                 why="moves: the dynamics (inertia, friction, delay); hold offset: where the friction band captures the stop", **common)
    if data.floor_kind != "repeats" and out.verdict == PASS:
        return result(config, value=value, measurements=measurements, payload=payload, verdict=WARN, code="ProxyFloor",
                      message=message + " (on a proxy floor, not identical repeats)", **common)
    return out
