"""The response to each commanded move: overshoot, rise time and settling time.

Definitions are MATLAB ``stepinfo``'s, which ISO 9283 (§7.5 overshoot, §7.4 stabilisation time) shares in spirit: per move from one held
target level to the next,
- overshoot = the furthest the response goes past the new level, in % of the move's size (0 when it never passes it);
- rise time = the time from 10 % to 90 % of the way (first crossings, linearly interpolated between samples);
- settling time = from the moment the target reaches its new level until the response enters, and stays within, ``settle_band`` x the
  move's size (or ``settle_abs``) around the final value (interpolated at the last exit from the band).
WHY "from the moment the target arrives": for a step that is the step; for a ramped move it is when the response is left on its own,
which is what ISO 9283's stabilisation time measures. The final value is the commanded level (``final="target"``) by default, since this
is a real-vs-target check; ``final="response"`` uses the response's own mean over the hold's last ``settle_tail_s`` (stepinfo's yfinal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np

from ..core import NOT_RUN, PASS, WARN, Evidence, Limits, Measure, judge, result, worst
from ..data import Trajectory, target_of

UNITS = {"overshoot": "%", "rise": "s", "settling": "s"}


@dataclass(frozen=True)
class StepResponseConfig:
    channel: str = "q_rad"  # q_rad (position moves) or qd_rad_s (velocity steps); its target is data.target_of(channel)
    hold_min_s: float = 0.2  # a target level held at least this long is a move's start or end
    min_step: float = 1e-6  # smaller level changes are not moves (channel units)
    settle_band: float = 0.02  # fraction of the move's size: the textbook 2 % band
    settle_abs: float | None = None  # an absolute band in channel units instead (ISO 9283 practice: the arm's RP)
    final: str = "target"  # target: the commanded level | response: the response's mean over the hold's last settle_tail_s
    settle_tail_s: float = 0.1
    skip_prelude: bool = True
    combine: str = "max"  # max: the worst move | median
    headline: str = "overshoot"  # overshoot | rise | settling
    overshoot: Limits = field(default_factory=Limits)  # %
    rise: Limits = field(default_factory=Limits)  # s
    settling: Limits = field(default_factory=Limits)  # s


def holds(t: np.ndarray, target: np.ndarray, hold_min_s: float, tolerance: float) -> list[tuple[int, int]]:
    """(first, last) row of every stretch where the target stays within ``tolerance`` of its first value for >= hold_min_s."""

    out, start, count = [], 0, len(t)
    while start < count:
        if not np.isfinite(target[start]):
            start += 1
            continue
        end = start
        while end + 1 < count and np.isfinite(target[end + 1]) and abs(target[end + 1] - target[start]) <= tolerance:
            end += 1
        if t[end] - t[start] >= hold_min_s:
            out.append((start, end))
        start = end + 1
    return out


def _crossing(t: np.ndarray, y: np.ndarray, level: float, direction: float) -> float | None:
    """First time ``direction * (y - level)`` turns >= 0, linearly interpolated; None if it never does."""

    above = direction * (y - level) >= 0
    if not above.any():
        return None
    i = int(np.argmax(above))
    if i == 0:
        return float(t[0])
    y0, y1 = y[i - 1], y[i]
    return float(t[i - 1] + (level - y0) / (y1 - y0) * (t[i] - t[i - 1]))


def _settling(t: np.ndarray, y: np.ndarray, final: float, band: float, t_arrive: float) -> tuple[float | None, bool]:
    """(time after t_arrive until |y - final| <= band for good, settled?). Interpolated at the last exit from the band."""

    outside = np.abs(y - final) > band
    if outside[-1]:
        return float(t[-1] - t_arrive), False
    if not outside.any():
        return 0.0, True
    i = int(np.flatnonzero(outside)[-1])  # last row outside; i + 1 is inside
    e0, e1 = abs(y[i] - final) - band, abs(y[i + 1] - final) - band
    t_in = t[i] + e0 / (e0 - e1) * (t[i + 1] - t[i])
    return max(0.0, float(t_in - t_arrive)), True


def step_response(config: StepResponseConfig, runs: Sequence[Trajectory]):
    """Per move: overshoot (%), rise time (s), settling time (s); the headline is the worst (or median) move."""

    target_name = target_of(config.channel)
    moves, evidence = [], []
    title = f"step response ({config.channel})"
    have_target = False
    for run in runs:
        if not (run.has(config.channel) and run.has(target_name)):
            continue
        have_target = True
        keep = run.rows(skip_prelude=config.skip_prelude)
        t, y, target = run.t[keep], run.channel(config.channel)[keep], run.channel(target_name)[keep]
        found = holds(t, target, config.hold_min_s, config.min_step / 10)
        for (_, end_a), (start_b, end_b) in zip(found, found[1:]):
            y_start, y_final = target[end_a], target[start_b]
            if config.final == "response":
                tail_a = (t >= t[end_a] - config.settle_tail_s) & (t <= t[end_a])
                tail_b = (t >= t[end_b] - config.settle_tail_s) & (t <= t[end_b])
                y_start, y_final = float(np.mean(y[tail_a])), float(np.mean(y[tail_b]))
            size = y_final - y_start
            if abs(target[start_b] - target[end_a]) < config.min_step or abs(size) < config.min_step:
                continue
            sign = 1.0 if size > 0 else -1.0
            span = slice(end_a + 1, end_b + 1)
            tt, yy = t[span], y[span]
            ok = np.isfinite(yy)
            tt, yy = tt[ok], yy[ok]
            if tt.size < 3:
                continue
            t_move = float(t[end_a + 1])
            overshoot = max(0.0, 100.0 * float(np.max(sign * (yy - y_final))) / abs(size))
            t10 = _crossing(tt, yy, y_start + 0.1 * size, sign)
            t90 = _crossing(tt, yy, y_start + 0.9 * size, sign)
            band = config.settle_abs if config.settle_abs is not None else config.settle_band * abs(size)
            settle, settled = _settling(tt, yy, y_final, band, float(t[start_b]))
            moves.append({"overshoot": overshoot, "rise": None if t10 is None or t90 is None else t90 - t10, "settling": settle,
                          "settled": settled, "run": run.run, "t0": t_move - float(run.t[0]), "t1": float(t[end_b] - run.t[0]), "size": size})
        evidence.append(Evidence(run.run, f"{config.channel} vs {target_name}"))
    unit = UNITS[config.headline]
    limits = getattr(config, config.headline)
    if not have_target:
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title,
                      message=f"no run has {config.channel} and {target_name}", code="NoTarget")
    if not moves:
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title,
                      message=f"no move between two held target levels (>= {config.hold_min_s:g} s each)", code="NoMoves")

    def combined(name: str) -> float | None:
        values = [m[name] for m in moves if m[name] is not None]
        if not values:
            return None
        return float(np.max(values)) if config.combine == "max" else float(np.median(values))

    stats = {name: combined(name) for name in UNITS}
    judged = {name: judge(stats[name], getattr(config, name)) for name in UNITS if not getattr(config, name).empty}
    verdict = worst(v for v, _ in judged.values()) if judged else (PASS if stats[config.headline] is not None else NOT_RUN)
    culprit = next((name for name in ("settling", "overshoot", "rise") if name in judged and judged[name][0] == verdict), config.headline)
    code = "" if verdict == PASS else (f"{culprit.capitalize()}{judged[culprit][1]}" if culprit in judged else "NoValue")
    unsettled = sum(not m["settled"] for m in moves)
    if unsettled and verdict == PASS:
        verdict, code = WARN, "NotSettled"
    worst_move = max(moves, key=lambda m: (m[config.headline] if m[config.headline] is not None else -1.0))
    return result(config, value=stats[config.headline], unit=unit, metric=config.headline, limits=limits, title=title, verdict=verdict, code=code,
                  message=f"{len(moves)} moves: overshoot {stats['overshoot']:.3g} %, rise "
                          + (f"{stats['rise']:.3g} s" if stats["rise"] is not None else "n/a") + f", settling {stats['settling']:.3g} s"
                          + (f"; {unsettled} never settled within the hold" if unsettled else ""),
                  measurements=(Measure("moves", len(moves)), Measure("unsettled", unsettled), Measure("overshoot_pct", stats["overshoot"], "%"),
                                Measure("rise_s", stats["rise"], "s"), Measure("settling_s", stats["settling"], "s")),
                  payload={"x": list(range(len(moves))), "y": [m[config.headline] for m in moves], "group": [m["run"] for m in moves],
                           "settled": [m["settled"] for m in moves], "size": [m["size"] for m in moves],
                           "x_label": "move", "y_label": f"{config.headline} ({unit})"},
                  evidence=tuple(evidence), run=worst_move["run"], t0=worst_move["t0"], t1=worst_move["t1"],
                  why="overshoot is too little damping (kd) or too much integral; a slow settle with no overshoot is too much damping or friction")


__all__ = ["StepResponseConfig", "step_response", "holds"]
