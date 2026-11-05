"""The three primitives the rack twin's metrics() worked around (2026-11-04), against answers known by construction:

- ``stock.move_hold_split``: a trajectory error split into its moving part and its hold part (|commanded speed| <= 0.02 rad/s), for
  repeats (the floor's parts, pairwise on the common clock) and for (real, model) pairs (per run; the headline is the worst run of a class).
- ``stock.rest_offset``: where the model comes to rest at every hold against where the real joint did, folded to the real's side.
- mode masking: a model row is scored only where BOTH the real run and the model were in position mode (``controlled_by="both"``).
- ``stock.split_fidelity``: T2's split judgement, moves / floor's moving part <= 1.5 and |hold offset| / floor's hold part <= 1.5.

Written before the code. The rack oracle for the floor's parts is tests/diagnostics/test_rack_oracle_split.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robotdx.diagnostics import FAIL, NOT_RUN, PASS, WARN, Limits, evaluate
from robotdx.diagnostics.stock import ModelErrorConfig, MoveHoldSplitConfig, RestOffsetConfig, SplitFidelityConfig
from robotdx.diagnostics.stock.fidelity import SplitFidelityInput, split_input

from conftest import tape

DT = 0.01  # conftest.tape's step: the floor's common clock on the tape's own samples, so no interpolation blurs a part


def _parts(run, moving: float, holding: float, name: str):
    """``run`` with its tracking error set to ``moving`` on the moving rows and ``holding`` on the hold rows (rad)."""

    hold = np.abs(run.qd_target_rad_s) <= 0.02
    return replace(run, q_rad=run.q_target_rad + np.where(hold, holding, moving), run=name)


def _measures(result) -> dict:
    return {m.key: m.value for m in result.measurements}


# --------------------------------------------------------------------------- the move/hold split


def test_the_floor_split_of_two_repeats_is_each_parts_own_difference():
    a, b = _parts(tape(), 0.0, 0.0, "a"), _parts(tape(), 0.003, 0.001, "b")
    out = evaluate("stock.move_hold_split", MoveHoldSplitConfig(dt=DT), [a, b], subject="s")
    parts = _measures(out)
    assert out.metric == "moves" and out.unit == "mrad" and out.verdict == PASS  # a reading: no limits
    assert out.value == pytest.approx(3.0, abs=1e-9) and parts["moves"] == pytest.approx(3.0, abs=1e-9)
    assert parts["holds"] == pytest.approx(1.0, abs=1e-9)
    hold_share = float(np.mean(np.abs(tape().qd_target_rad_s) <= 0.02))
    assert parts["hold_fraction"] == pytest.approx(hold_share, abs=0.01)
    assert parts["all"] == pytest.approx(np.sqrt(hold_share * 1.0 + (1 - hold_share) * 9.0), abs=0.02)
    assert parts["pairs"] == 1 and parts["repeats"] == 2


def test_the_floor_split_is_the_median_over_pairs():
    runs = [_parts(tape(), 0.001 * i, 0.0005 * i, f"r{i}") for i in range(4)]  # pair (i, j) differs by |i - j| mrad moving
    out = evaluate("stock.move_hold_split", MoveHoldSplitConfig(dt=DT), runs, subject="s")
    assert out.value == pytest.approx(float(np.median([1, 2, 3, 1, 2, 1])), abs=1e-9)
    assert _measures(out)["holds"] == pytest.approx(0.5 * float(np.median([1, 2, 3, 1, 2, 1])), abs=1e-9)


def test_the_model_split_is_per_run_and_its_headline_is_the_worst_run_of_the_class():
    real = [tape(run=f"r{i}", **{"class": cls}) for i, cls in enumerate(("pickplace", "pickplace", "sweep"))]
    errors = {"r0": (0.002, 0.001), "r1": (0.004, 0.0005), "r2": (0.009, 0.009)}  # r1 is the worst pickplace run; r2 another class
    pairs = [(run, run.as_situation(run.q_rad + np.where(np.abs(run.qd_target_rad_s) <= 0.02, errors[run.run][1], errors[run.run][0]), "twin"))
             for run in real]
    out = evaluate("stock.move_hold_split", MoveHoldSplitConfig(on_class="pickplace"), pairs, subject="s")
    assert out.value == pytest.approx(4.0, abs=1e-9) and out.run == "r1"
    assert _measures(out)["holds"] == pytest.approx(0.5, abs=1e-9)
    rows = {row[1]: row for row in out.payload["breakdown"]["rows"]}  # [model, run, class, all, moves, holds, hold_fraction]
    assert rows["r0"][4] == pytest.approx(2.0, abs=1e-9) and rows["r0"][5] == pytest.approx(1.0, abs=1e-9)
    assert rows["r2"][2] == "sweep"
    everyone = evaluate("stock.move_hold_split", MoveHoldSplitConfig(), pairs, subject="s")
    assert everyone.value == pytest.approx(9.0, abs=1e-9) and everyone.run == "r2"


def test_the_model_split_scores_only_rows_both_ran_in_position_mode():
    real = tape(run="r0", **{"class": "pickplace"})
    wild = real.q_rad.copy()
    wild[:120] += 1.0  # a wild prediction ...
    mode = real.mode.copy()
    mode[:120] = 0.0  # ... on rows the model was not tracking (a timeout)
    model = replace(real.as_situation(wild + 0.001, "twin"), mode=mode)
    both = evaluate("stock.move_hold_split", MoveHoldSplitConfig(), [(real, model)], subject="s")
    assert both.value == pytest.approx(1.0, abs=1e-9) and _measures(both)["holds"] == pytest.approx(1.0, abs=1e-9)
    real_only = evaluate("stock.move_hold_split", MoveHoldSplitConfig(controlled_by="real"), [(real, model)], subject="s")
    assert real_only.value > 100.0


def test_the_split_is_not_run_without_its_inputs():
    assert evaluate("stock.move_hold_split", MoveHoldSplitConfig(), [tape()], subject="s").verdict == NOT_RUN  # one repeat
    assert evaluate("stock.move_hold_split", MoveHoldSplitConfig(), [], subject="s").verdict == NOT_RUN
    still = replace(tape(), qd_target_rad_s=np.zeros(600))  # nothing moves: no moving part
    out = evaluate("stock.move_hold_split", MoveHoldSplitConfig(), [(still, still.as_situation(still.q_rad + 0.001, "twin"))], subject="s")
    assert (out.verdict, out.code, out.value) == (NOT_RUN, "NoMovingRows", None)


# --------------------------------------------------------------------------- the rest offset


def _rested(run, rest_mrad: list[float]):
    """``run`` resting ``rest_mrad[k]`` short of the target (target - q, mrad) over hold k; exact on the moving rows."""

    hold = np.abs(run.qd_target_rad_s) <= 0.02
    edges = np.flatnonzero(np.diff(np.concatenate([[0], hold.astype(int), [0]])))
    q = run.q_target_rad.copy()
    for k, (start, end) in enumerate(zip(edges[::2], edges[1::2])):
        q[start:end] -= 1e-3 * rest_mrad[k]
    return q


def test_the_rest_offset_is_the_median_model_minus_real_rest_folded_to_the_real_side():
    real = tape(run="r0")  # four dwells of 0.5 s, each >= 0.3 s
    real = replace(real, q_rad=_rested(real, [2.0, -2.0, 1.0, 3.0]))
    model = real.as_situation(_rested(real, [3.0, -1.0, 1.5, 3.0]), "twin")
    out = evaluate("stock.rest_offset", RestOffsetConfig(), [(real, model)], subject="s")
    # folded to the real's side: +1 (farther), -1 (nearer: real -2, model -1), +0.5, 0 -> median 0.25
    assert out.value == pytest.approx(0.25, abs=1e-9) and out.unit == "mrad" and out.metric == "median"
    parts = _measures(out)
    assert parts["holds"] == 4 and parts["farther_fraction"] == pytest.approx(0.5)
    assert parts["median_abs"] == pytest.approx(0.75, abs=1e-9)
    assert sorted(out.payload["y"]) == pytest.approx([-1.0, 0.0, 0.5, 1.0], abs=1e-9)


def test_the_rest_is_the_mean_over_the_holds_tail_only():
    real = tape(run="r0")
    model_q = real.q_target_rad.copy()
    hold = np.abs(real.qd_target_rad_s) <= 0.02
    early = hold & (np.arange(real.t.size) % 150 < 120)  # the first 0.2 s of each 0.5 s dwell: the transient, not the rest
    model_q[early] -= 0.050
    out = evaluate("stock.rest_offset", RestOffsetConfig(), [(real, real.as_situation(model_q, "twin"))], subject="s")
    assert out.value == pytest.approx(0.0, abs=1e-9)


def test_the_rest_offset_skips_short_holds_and_rows_the_model_left_position_mode():
    real = tape(run="r0")
    model = real.as_situation(real.q_rad - 0.001, "twin")
    assert evaluate("stock.rest_offset", RestOffsetConfig(), [(real, model)], subject="s").value == pytest.approx(1.0, abs=1e-9)
    assert evaluate("stock.rest_offset", RestOffsetConfig(min_hold_s=0.6), [(real, model)], subject="s").code == "TooFewHolds"
    mode = real.mode.copy()
    mode[100:150] = 0.0  # the model timed out over the first dwell
    out = evaluate("stock.rest_offset", RestOffsetConfig(min_holds=1), [(real, replace(model, mode=mode))], subject="s")
    assert _measures(out)["holds"] == 3


def test_the_rest_offset_is_not_run_on_fewer_holds_than_its_minimum():
    real = tape(n=300, run="r0")  # two dwells
    out = evaluate("stock.rest_offset", RestOffsetConfig(), [(real, real.as_situation(real.q_rad, "twin"))], subject="s")
    assert (out.verdict, out.code, out.value) == (NOT_RUN, "TooFewHolds", None)
    assert evaluate("stock.rest_offset", RestOffsetConfig(), [], subject="s").verdict == NOT_RUN


# --------------------------------------------------------------------------- mode masking in model_error


def test_model_error_masks_by_both_modes_when_asked():
    real = tape(run="r0")
    q, mode = real.q_rad + 0.0005, real.mode.copy()
    q[:50] += 1.0
    mode[:50] = 0.0
    model = replace(real.as_situation(q, "twin"), mode=mode)
    assert evaluate("stock.model_error", ModelErrorConfig(controlled_by="both"), [(real, model)], subject="s").value == pytest.approx(0.5)
    # the default stays the real run's mode alone (every existing caller and oracle unchanged)
    assert evaluate("stock.model_error", ModelErrorConfig(), [(real, model)], subject="s").value > 100.0
    with pytest.raises(ValueError, match="controlled_by"):
        ModelErrorConfig(controlled_by="twin")


# --------------------------------------------------------------------------- T2's split judgement


def test_split_fidelity_judges_moves_and_hold_offset_each_against_the_floors_part():
    data = SplitFidelityInput(floor_moves=2.0, floor_holds=4.0, moves=2.8, rest_offset=-2.0)
    out = evaluate("stock.split_fidelity", SplitFidelityConfig(), data, subject="s")
    parts = _measures(out)
    assert parts["moves_over_floor"] == pytest.approx(1.4) and parts["hold_offset_over_floor"] == pytest.approx(0.5)
    assert out.value == pytest.approx(1.4) and out.verdict == PASS  # the binding part heads
    assert out.limits == Limits.at_most(1.5)
    moves_fail = evaluate("stock.split_fidelity", SplitFidelityConfig(), replace(data, moves=3.2), subject="s")
    assert (moves_fail.verdict, moves_fail.code) == (FAIL, "AboveErrorLimit") and moves_fail.value == pytest.approx(1.6)
    hold_fail = evaluate("stock.split_fidelity", SplitFidelityConfig(), replace(data, rest_offset=6.4), subject="s")
    assert hold_fail.verdict == FAIL and _measures(hold_fail)["hold_offset_over_floor"] == pytest.approx(1.6)
    proxy = evaluate("stock.split_fidelity", SplitFidelityConfig(), replace(data, floor_kind="proxy"), subject="s")
    assert (proxy.verdict, proxy.code) == (WARN, "ProxyFloor")
    missing = evaluate("stock.split_fidelity", SplitFidelityConfig(), replace(data, rest_offset=None), subject="s")
    assert missing.verdict == NOT_RUN and missing.code == "NoRestOffset"


def test_split_input_reads_the_three_results():
    a, b = _parts(tape(), 0.0, 0.0, "a"), _parts(tape(), 0.002, 0.004, "b")
    floor = evaluate("stock.move_hold_split", MoveHoldSplitConfig(dt=DT), [a, b], subject="s")
    real = replace(tape(run="r0"), q_rad=_rested(tape(), [1.0] * 4))
    model = real.as_situation(_rested(real, [4.0] * 4), "twin")
    split = evaluate("stock.move_hold_split", MoveHoldSplitConfig(), [(real, model)], subject="s")
    rest = evaluate("stock.rest_offset", RestOffsetConfig(), [(real, model)], subject="s")
    data = split_input(floor, split, rest)
    assert (data.floor_moves, data.floor_holds) == (pytest.approx(2.0), pytest.approx(4.0))
    assert data.rest_offset == pytest.approx(3.0) and data.moves == split.value
    out = evaluate("stock.split_fidelity", SplitFidelityConfig(), data, subject="s")
    assert _measures(out)["hold_offset_over_floor"] == pytest.approx(0.75)
