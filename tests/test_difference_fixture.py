"""diff_rows against data-ui's difference(a, b) fixture (PEP 0010 / gallery/robotics CONTRACT section 5).

The fixture is a numpy/scipy oracle copied verbatim into tests/fixtures/data_ui_difference (PROVENANCE.md there): two
trajectories of one run, a = ``real`` and b = ``target``, and the rows ``difference(a, b)`` must return, compared with the
fixture's own checker (1e-9 in a column's unit, null as null). A servo whose q is in degrees is a ``Trajectory``; a
prismatic axis (``q_mm``) and the poses table are outside what a joint ``Trajectory`` holds, so those rows are not judged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "data_ui_difference"
sys.path.insert(0, str(FIXTURE_DIR))
import fixture  # noqa: E402  (data-ui's stdlib checker, in the fixture directory)

from robotdx.diagnostics.data import Trajectory  # noqa: E402
from robotdx.diagnostics.rows import diff_rows  # noqa: E402

RUN = "fixture/r1"
SERVO = "J3 elbow"


def _column(rows: list[dict], name: str, scale: float = 1.0) -> np.ndarray:
    return np.array([np.nan if row[name] is None else row[name] * scale for row in rows], float)


def _servo(rows: list[dict]) -> Trajectory:
    mine = [row for row in rows if row["device"] == SERVO and row["run"] == RUN]
    return Trajectory(t=_column(mine, "t"), q_rad=_column(mine, "q_deg", np.pi / 180), qd_rad_s=_column(mine, "qd_deg_s", np.pi / 180),
                      tau_nm=_column(mine, "tau_nm"), run=RUN, device=SERVO, situation=mine[0]["situation"], source=mine[0]["source"])


def test_diff_rows_match_the_data_ui_fixture_for_a_revolute_servo():
    a_rows, b_rows = fixture.trajectories("samples")
    expected = [row for row in fixture.rows("diff_samples") if row["device"] == SERVO and row["run"] == RUN]
    problems = fixture.compare(diff_rows(_servo(a_rows), _servo(b_rows)), expected)
    assert problems == [], "\n".join(problems)
    assert len(expected) == 152 and expected[0]["t"] == 0.0  # a runs -0.05..1.75 s, b 0..1.6 s: no rows outside b's span
