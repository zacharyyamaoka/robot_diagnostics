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
PRESLIDING_SESSION = RACK / "rack_2026-10-28" / "physical" / "presliding_can12_can13"
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


# ----- 10-28 pre-sliding session (physical/presliding_can12_can13; integral off, no feed-forward) ----------------------------------
# bam_rack_twin presliding_round2_can{12,13}.json "direct": median stuck-leg slope of fine q on the push, over the session's waypoint runs
PUBLISHED_COMPLIANCE = {12: (0.6316627031218227, (0.49427753211848, 0.8297959432363987)),
                        13: (0.28289887241557826, (0.2523712685411179, 0.38790059657589926))}
ROTOR_PER_OUTPUT = 10.0  # both stations' boards are configured with ratio 0.1 (q = 0.1 x rotor angle; measured slope 0.09999 / -0.09996)
ROTOR_SIGN = {12: 1.0, 13: -1.0}  # round4_study.rotor_relation's measured slopes: CAN 13's rotor turns the other way


def _fine(path: Path, can: int):
    """The run with q_rad from the rotor encoder (rotor_rev / ratio): the board's q is quantised to ~0.1 mrad, the rotor is not.
    The same construction as presliding_round2.fine_q (unwrapped rotor angle / ratio, offset to the board's q by the median)."""

    run = load(path)
    with np.load(path) as archive:
        rotor = np.asarray(archive["rotor_rev"], float)
    angle = ROTOR_SIGN[can] * np.unwrap(2 * np.pi * rotor) / ROTOR_PER_OUTPUT
    return replace(run, q_rad=angle + float(np.median(run.q_rad - angle)))


# round2's stuck_legs rule as config: target legs after the prelude + 0.5 s, 7-row means, 4 rows trimmed, >= 20 rows, push span
# >= 0.15 N.m, the joint within 0.6 mrad of the leg's start on >= 95 % of rows; the leg-median slope
PRESLIDING = dict(legs="target", stiffness="leg_median", headline="compliance", smooth_rows=7, trim_rows=4, min_leg_rows=20,
                  min_torque_span_nm=0.15, max_leg_travel_rad=0.6e-3, travel_share=0.95, skip_s=0.5)


@pytest.mark.parametrize("can", [12, 13])
def test_presliding_compliance_matches_bam_rack_twin(can):
    from robotdx.diagnostics.stock import HysteresisConfig

    runs = [_fine(path, can) for path in sorted((PRESLIDING_SESSION / "runs").glob(f"*_can{can}_waypoints.npz"))]
    out = evaluate("stock.hysteresis_loop", HysteresisConfig(**PRESLIDING), runs, subject=f"rack/can{can}")
    published, (low, high) = PUBLISHED_COMPLIANCE[can]
    assert {m.key: m.value for m in out.measurements}["legs"] == 8  # round2's stuck_legs: 8 per station
    # WHY 1 %, not exact: round2 also subtracts the plant's fitted cogging ripple from the push (coefficients from its own fit, not
    # data this library reads); on stuck legs the rotor barely turns, so that term moves the median slope by < 0.5 %
    assert out.value == pytest.approx(published, rel=0.01) and low <= out.value <= high


def test_presliding_compliance_ruler_can_fail():
    # without round2's "still stuck" box the legs that slid come in (12, not 8) and the compliance moves off the published number:
    # the selection rule is part of the ruler, and the check notices when it changes
    from robotdx.diagnostics.stock import HysteresisConfig

    runs = [_fine(path, 12) for path in sorted((PRESLIDING_SESSION / "runs").glob("*_can12_waypoints.npz"))]
    slid = evaluate("stock.hysteresis_loop", HysteresisConfig(**{**PRESLIDING, "max_leg_travel_rad": None}), runs, subject="rack/can12")
    assert {m.key: m.value for m in slid.measurements}["legs"] == 12
    assert abs(slid.value - PUBLISHED_COMPLIANCE[12][0]) > 0.05


def test_step_response_on_the_velocity_plateaus_runs_and_finds_no_settled_reversal_at_001_rad_s():
    """No published number to match: what the session itself found ("nothing slides at 0.01 rad/s in 2 s") must show as an unsettled
    reversal into the -0.01 rad/s plateau, on both stations."""

    from robotdx.diagnostics.stock import StepResponseConfig

    for can, name in ((12, "s08_can12_friction"), (13, "s16_can13_friction")):
        run = load(PRESLIDING_SESSION / "runs" / f"{name}.npz")
        out = evaluate("stock.step_response", StepResponseConfig(channel="qd_rad_s", headline="settling", settle_band=0.1, skip_prelude=False),
                       [run], subject=f"rack/can{can}")
        sizes, settled = out.payload["size"], out.payload["settled"]
        assert len(sizes) == 7 and sizes[0] == pytest.approx(-0.02)  # +0.01 -> -0.01 rad/s, then up the ladder
        assert settled[0] is False and out.code == "NotSettled"
