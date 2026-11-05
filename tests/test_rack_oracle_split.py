"""The oracle for ``stock.move_hold_split`` on repeats: on the rack's real floor sessions it reproduces the floor's moving and hold parts
bam_rack_twin published (round7_study.floor_split: each pair on the 2 ms common clock, a point is a hold where the first run's commanded
speed interpolates to <= 0.02 rad/s, medians over the pairs). The numbers are read from the archive's scored files, never recomputed
here; skipped when the archive is not mounted. The model half (the twin's split and rest offset against followup_scored_can11.json) needs
the twin's replay, so it lives in bam_ws: src/dev/bam_rack_twin/tests/test_metrics_split.py.

Hardware: none. Run (from a fresh terminal):

    cd ~/io-diagnostics
    env -u VIRTUAL_ENV uv run --group dev pytest -q -p no:cacheprovider tests/diagnostics/test_rack_oracle_split.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotdx.diagnostics import evaluate
from robotdx.diagnostics.adapters.rack_npz import load
from robotdx.diagnostics.stock import MoveHoldSplitConfig

from conftest import RACK

ANALYSIS = RACK / "rack_2026-10-28" / "analysis"
CALIBRATION = ANALYSIS / "calibration_scored.json"
FOLLOWUP = ANALYSIS / "followup_scored_can11.json"
pytestmark = pytest.mark.skipif(not (CALIBRATION.is_file() and FOLLOWUP.is_file()), reason=f"the rack archive is not mounted: {CALIBRATION}")
EXACT = 1e-9


def _floors() -> list[tuple[str, dict]]:
    stations = json.loads(CALIBRATION.read_text())["stations"]
    return [*sorted(stations.items()), ("can11 il0.5", json.loads(FOLLOWUP.read_text())["floor"])]


@pytest.mark.parametrize(("station", "published"), _floors() if CALIBRATION.is_file() and FOLLOWUP.is_file() else [], ids=lambda value: str(value)[:12])
def test_the_floor_split_reproduces_each_floor_sessions_published_parts(station, published):
    runs = [load(path) for path in sorted((Path(published["session"]) / "runs").glob("*.npz"))]
    out = evaluate("stock.move_hold_split", MoveHoldSplitConfig(), runs, subject=f"rack/{station}")
    parts = {m.key: m.value for m in out.measurements}
    assert out.value == pytest.approx(published["moves_mrad"], abs=EXACT)
    assert parts["holds"] == pytest.approx(published["holds_mrad"], abs=EXACT)
    assert parts["all"] == pytest.approx(published["all_mrad"], abs=EXACT)
    assert parts["hold_fraction"] == pytest.approx(published["hold_fraction"], abs=EXACT)
    assert parts["pairs"] == published["pairs"] == 10


def test_the_split_ruler_can_fail():
    # mutation check: a different hold speed is a different ruler, and the oracle must notice
    published = json.loads(FOLLOWUP.read_text())["floor"]
    runs = [load(path) for path in sorted((Path(published["session"]) / "runs").glob("*.npz"))]
    moved = evaluate("stock.move_hold_split", MoveHoldSplitConfig(hold_speed=0.05), runs, subject="rack/can11")
    assert abs({m.key: m.value for m in moved.measurements}["holds"] - published["holds_mrad"]) > 1e3 * EXACT
