"""How much of the torque the feedback loop still supplies: ``RMS(tau - tau_ff) / RMS(tau)``.

Zach's "shrinking feedback term": as a model's feed-forward carries more of the torque, the PID is left with less to do. The definition
is bam_rack_twin's ``vt_kpi.feedback_share`` (branch claude/sim-to-real-motor-models-f1ee43): 1.0 when no feed-forward was sent, pooled
over the controlled rows of a station's runs; ``tests/diagnostics/test_rack_oracle_specified.py`` reproduces its published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np

from ..core import Evidence, Limits, Measure, result
from ..data import Trajectory, center, rms


@dataclass(frozen=True)
class FeedbackShareConfig:
    combine: str = "pooled"  # pooled: one ratio over every run's rows (vt_kpi._pooled_share) | per_run: the centre of per-run ratios
    across_runs: str = "median"  # per_run only: median | median_high
    controlled_only: bool = True  # only position-controlled rows (moteus mode 10), as vt_kpi._run_share
    skip_prelude: bool = False
    normalize: bool = True  # True: RMS(tau - tau_ff) / RMS(tau), unit 1; False: RMS(tau - tau_ff) in N.m (the hold effort)
    limits: Limits = field(default_factory=Limits)


def feedback_share(config: FeedbackShareConfig, runs: Sequence[Trajectory]):
    """RMS of what the feedback supplied (tau - tau_ff) over RMS of the torque, on the kept finite rows."""

    taus, feedforwards, evidence, per_run = [], [], [], []
    for run in runs:
        if not (run.has("tau_nm") and run.has("tau_ff_nm")):
            continue
        tau, feedforward = run.channel("tau_nm"), run.channel("tau_ff_nm")
        keep = run.rows(skip_prelude=config.skip_prelude, controlled_only=config.controlled_only) & np.isfinite(tau) & np.isfinite(feedforward)
        if not keep.any():
            continue
        taus.append(tau[keep])
        feedforwards.append(feedforward[keep])
        evidence.append(Evidence(run.run, "tau_nm - tau_ff_nm"))
        per_run.append(_share(tau[keep], feedforward[keep], config.normalize))
    unit = "1" if config.normalize else "N.m"
    metric = "feedback_share" if config.normalize else "feedback_rms"
    title = "feedback share" if config.normalize else "feedback torque RMS"
    if not taus:
        return result(config, value=None, unit=unit, metric=metric, limits=config.limits, title=title, better="lower",
                      message="no run carries both the reported torque and the feed-forward it was sent", code="NoFeedForward")
    pooled = _share(np.concatenate(taus), np.concatenate(feedforwards), config.normalize)
    value = pooled if config.combine == "pooled" else center([v for v in per_run if v is not None], config.across_runs)
    if value is None or not np.isfinite(value):
        return result(config, value=None, unit=unit, metric=metric, limits=config.limits, title=title, better="lower",
                      message="the reported torque is zero on every kept row: no share to take", code="NoTorque")
    rows = int(sum(t.size for t in taus))
    return result(config, value=value, unit=unit, metric=metric, limits=config.limits, title=title, better="lower",
                  message=f"{config.combine} over {len(taus)} runs ({rows} rows): feedback supplies {value:.4g}"
                          + (" of the torque (RMS)" if config.normalize else " N.m RMS"),
                  measurements=(Measure("runs", len(taus)), Measure("rows", rows),
                                Measure("tau_rms", float(np.sqrt(np.mean(np.concatenate(taus) ** 2))), "N.m")),
                  payload={"breakdown": {"runs": [e.path for e in evidence], "values": per_run}}, evidence=tuple(evidence),
                  why="a share near 1 means the feed-forward model carries little: fit friction/gravity feed-forward (ff_study) before gains")


def _share(tau: np.ndarray, feedforward: np.ndarray, normalize: bool) -> float | None:
    feedback = rms(tau - feedforward)
    if not normalize:
        return feedback
    torque = rms(tau)
    return feedback / torque if torque > 0 else None


__all__ = ["FeedbackShareConfig", "feedback_share"]
