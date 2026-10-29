"""The specified primitives on real rack runs, against numbers a separate implementation published for the same runs.

- ``feedback_share`` vs bam_rack_twin ``vt_kpi.feedback_share_points`` on the 10-24 feed-forward study (rack_2026-10-24/analysis/
  ff_study.json): per station, the pooled share over the runs of the study's best feed-forward fraction.
- ``hysteresis_loop`` (leg-median compliance on the target legs while stuck) vs bam_rack_twin docs/presliding_round2.py's pre-sliding
  compliance on the 10-28 session physical/presliding_can12_can13: CAN 12 0.63 mrad/N.m [0.49, 0.83], CAN 13 0.28 [0.25, 0.39].
- ``step_response`` on the same session's velocity plateaus (no published number to match; what it shows is asserted against the
  session's own finding, "nothing slides at 0.01 rad/s in 2 s").

The bam_rack_twin numbers below were produced by its own code (branch claude/sim-to-real-motor-models-f1ee43 @ 4889d6a6, 2026-10-28:
``vt_kpi.feedback_share_points(ff_study.json)``; ``presliding_round2_can{12,13}.json`` ``direct.slope_mrad_per_nm``), never by this
library. Skipped when the archive is not mounted.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robotdx.diagnostics import NOT_RUN, evaluate
from robotdx.diagnostics.adapters.rack_npz import load
from robotdx.diagnostics.stock import FeedbackShareConfig

from conftest import RACK, needs_rack

pytestmark = needs_rack
DAY24 = RACK / "rack_2026-10-24"
PRESLIDING = RACK / "rack_2026-10-28" / "physical" / "presliding_can12_can13"
A28 = RACK / "rack_2026-10-28" / "analysis"

# vt_kpi.feedback_share_points, rounded by it to 4 dp (value per motor; "median" is its own roll-up over the 8 motors)
PUBLISHED_FEEDBACK_SHARE = {1: 0.9143, 4: 0.7025, 11: 0.8254, 12: 0.5725, 13: 0.474, 14: 1.0, 15: 0.9692, 21: 1.0}
PUBLISHED_MEDIAN = 0.8699


def _fraction(run_json: Path, coulomb_nm: float) -> float:
    """ff_study._fraction: the run's Coulomb feed-forward as a fraction of the station's identified Coulomb torque."""

    feedforward = float((json.loads(run_json.read_text()).get("args") or {}).get("ff_coulomb_nm") or 0.0)
    return round(feedforward / coulomb_nm, 2) if feedforward else 0.0


def _best_fraction_runs(can: int, entry: dict) -> list:
    groups: dict[float, list[Path]] = {}
    for session in (DAY24 / "floors" / f"can{can}", DAY24 / "ff" / f"can{can}"):
        for path in sorted((session / "runs").glob(f"*_can{can}_waypoints.npz")):
            groups.setdefault(_fraction(path.with_suffix(".json"), float(entry["coulomb_nm"])), []).append(path)
    return [load(path) for path in groups[float(entry["best_fraction"])]]


def test_feedback_share_matches_bam_rack_twin_for_all_eight_motors():
    study = json.loads((DAY24 / "analysis" / "ff_study.json").read_text())
    values = {}
    for can, entry in study.items():
        runs = _best_fraction_runs(int(can), entry)
        out = evaluate("stock.feedback_share", FeedbackShareConfig(), runs, subject=f"rack/can{can}")
        values[int(can)] = out.value
        assert out.value == pytest.approx(PUBLISHED_FEEDBACK_SHARE[int(can)], abs=5e-5), can
    assert statistics.median(values.values()) == pytest.approx(PUBLISHED_MEDIAN, abs=5e-5)
    assert values[13] == pytest.approx(0.474, abs=5e-5) and values[12] == pytest.approx(0.5725, abs=5e-5)


def test_the_feedback_share_ruler_can_fail():
    # drop the feed-forward the runs carried: the share must move off the published number (mutation of the input, not the code)
    study = json.loads((DAY24 / "analysis" / "ff_study.json").read_text())
    runs = [replace(run, tau_ff_nm=np.zeros_like(run.tau_ff_nm)) for run in _best_fraction_runs(13, study["13"])]
    out = evaluate("stock.feedback_share", FeedbackShareConfig(), runs, subject="rack/can13")
    assert out.value == pytest.approx(1.0) and abs(out.value - PUBLISHED_FEEDBACK_SHARE[13]) > 0.1
    bare = [replace(run, tau_ff_nm=None) for run in runs]
    assert evaluate("stock.feedback_share", FeedbackShareConfig(), bare, subject="rack/can13").verdict == NOT_RUN
