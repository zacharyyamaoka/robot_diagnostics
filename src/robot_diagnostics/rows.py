"""The display projection data-ui reads (PEP 0010 §4 and §10): diagnostic rows, sample rows and difference rows.

- ``result_row``: one Result as the diagnostic row every hint view reads, with exactly the PEP's columns (plus ``emits``).
- ``sample_rows`` / ``diff_rows``: a Trajectory as ``samples.json`` / ``diff_samples.json`` rows, columns ending in
  their display unit (deg, not rad). Comparisons are computed here, once (``data.difference``), and written as data.
"""

from __future__ import annotations

import math
from typing import Callable

from ._numpy import np

from .core import RANK, Result
from .data import Trajectory, difference
from .record import LEVEL_OF

ROW_COLUMNS = ("subject", "diagnostic", "title", "verdict", "rank", "value", "unit", "warn_lo", "warn_hi", "error_lo", "error_hi",
               "target", "better", "t", "t0", "t1", "run", "message", "why", "evidence", "basis", "source", "sourceUrl", "record",
               "parent", "hint", "emits")


def playback_link(run: str, t0: float | None, t1: float | None) -> str:
    """``<run>#t=t0&tr=t0,t1``: the run's playback at the evidence window (PEP 0010 §10 ``evidence``)."""

    if not run:
        return ""
    if t0 is None:
        return run
    return f"{run}#t={t0:.3f}&tr={t0:.3f},{(t1 if t1 is not None else t0):.3f}"


def result_row(result: Result, *, parent: str = "", vocabulary: str = "verdict", source_url: str = "",
               link: Callable[[str, float | None, float | None], str] = playback_link) -> dict:
    """One diagnostic row. ``vocabulary``: ``verdict`` (pass/warn/fail/not_run, the gates', sim2real) or ``level``
    (ok/warn/error/unknown, RobotDx's); the column is named after it. ``parent``: the composite it rolls up into."""

    limits = result.limits
    first = next((e for e in result.evidence if e.path == result.run), result.evidence[0] if result.evidence else None)
    row = {
        "subject": result.subject, "diagnostic": result.diagnostic, "title": result.title,
        vocabulary: result.verdict if vocabulary == "verdict" else LEVEL_OF[result.verdict],
        "rank": RANK[result.verdict], "value": result.value, "unit": result.unit,
        "warn_lo": limits.warn_low, "warn_hi": limits.warn_high, "error_lo": limits.error_low, "error_hi": limits.error_high,
        "target": result.target, "better": result.better or None, "t": result.stamp, "t0": result.t0, "t1": result.t1,
        "run": result.run or None, "message": result.message, "why": result.why or None,
        "evidence": link(result.run, result.t0, result.t1) or None, "basis": result.basis or None,
        "source": f"robotdx.diagnostics · {result.diagnostic}", "sourceUrl": source_url or None,
        "record": f"{first.path} · {first.key}" if first else None, "parent": parent or None,
        "hint": result.hint, "emits": result.emits,
    }
    return {key: row[key] for key in (*ROW_COLUMNS[:3], vocabulary, *ROW_COLUMNS[4:])}


def _deg(values) -> list:
    return [None if not math.isfinite(v) else round(float(np.degrees(v)), 6) for v in values]


def sample_rows(run: Trajectory, *, every: int = 1) -> list[dict]:
    """``samples.json`` rows: t, run, situation, source, device + the servo/driver channels in display units."""

    out = []
    columns = {"q_deg": run.q_rad, "qd_deg_s": run.qd_rad_s, "tau_nm": run.tau_nm, "i_q_a": run.i_q_a, "v_bus_v": run.v_bus_v,
               "temp_c": run.temp_c}
    columns = {name: (_deg(values) if name.endswith(("_deg", "_deg_s")) else [None if not math.isfinite(v) else float(v) for v in values])
               for name, values in columns.items() if values is not None}
    for i in range(0, len(run.t), every):
        out.append({"t": float(run.t[i] - run.t[0]), "run": run.run, "situation": run.situation, "source": run.source,
                    "device": run.device, **{name: values[i] for name, values in columns.items()}})
    return out


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def diff_rows(a: Trajectory, b: Trajectory) -> list[dict]:
    """``diff_samples.json`` rows: b - a at a's own sample times inside b's span (PEP 0010 §4.4; data-ui's
    gallery/robotics difference fixture, tests/fixtures/data_ui_difference): ``e_q_mrad``, plus ``e_qd_deg_s`` and
    ``e_tau_nm`` when both runs carry the channel; a null on either side is null in the row."""

    # WHY a and b keep their own t, not t - t[0]: both are on one run clock (CONTRACT §5 pairs rows by time), so zeroing each
    # at its own first sample moves a run that starts before b by that lead and pairs the wrong samples (146 mrad on the fixture).
    channels = [("e_q_mrad", "q_rad", 1e3)]
    if a.has("qd_rad_s") and b.has("qd_rad_s"):
        channels.append(("e_qd_deg_s", "qd_rad_s", 180.0 / math.pi))
    if a.has("tau_nm") and b.has("tau_nm"):
        channels.append(("e_tau_nm", "tau_nm", 1.0))
    t = difference((a.t, a.channel(channels[0][1])), (b.t, b.channel(channels[0][1])))[0]
    errors = {column: difference((a.t, a.channel(name)), (b.t, b.channel(name)))[1] * scale for column, name, scale in channels}
    return [{"t": float(ti), "run": a.run, "situation": a.situation, "source": a.source, "reference": b.situation, "device": a.device,
             **{column: _finite_or_none(values[i]) for column, values in errors.items()}} for i, ti in enumerate(t)]
