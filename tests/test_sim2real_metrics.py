"""The stock checks that are also the servo gates compute through sim2real.metrics: one formula, not two.

    uv run --project /home/bam/io-diagnostics --group dev pytest -q /home/bam/io-diagnostics/tests/diagnostics/test_sim2real_metrics.py

Zach, 2026-11-10 ("Sim2Real as a Pure Library on robot_msgs", step 4): RobotDx's stock checks duplicated bam_rack_twin's gates T1-T4 (the
same checks twice, the Sim2Real MOC of Nov 4). Point them at the same sim2real.metrics functions. Written before the checks called sim2real
(red first). A spy proves each check calls sim2real's function; a sentinel answer proves the check's number IS that function's, so a check
that called it and then computed its own number again would fail. The numbers themselves are held by the existing oracles (test_rack_oracle*,
test_spec_ratios, test_move_hold), unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import sim2real.metrics.actuator as actuator
import sim2real.metrics.coverage as coverage
import sim2real.metrics.series as series
from robotdx.diagnostics import data, evaluate
from robotdx.diagnostics.data import Trajectory
from robotdx.diagnostics.stock import (CoverageConfig, FidelityRatioConfig, FloorConfig, ModelErrorConfig, MoveHoldSplitConfig, RestOffsetConfig,
                                       SplitFidelityConfig, StopErrorConfig, TrackingErrorConfig)
from robotdx.diagnostics.stock.fidelity import FidelityInput, ModelRow, SplitFidelityInput


def _tape(offset_mrad: float = 0.0, name: str = "run") -> Trajectory:
    """Move to 0.5 rad, dwell 1 s, move back, dwell 1 s, move on: two stops, then parking."""

    t = np.round(np.arange(0.0, 5.0, 0.01), 9)
    q_target = np.interp(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.3])
    qd_target = np.gradient(q_target, t)
    qd_target[(t >= 1.0) & (t <= 2.0)] = 0.0
    qd_target[(t >= 3.0) & (t <= 4.0)] = 0.0
    q = q_target + offset_mrad * 1e-3 + 1e-4 * np.sin(7 * t)
    return Trajectory(t=t, q_rad=q, q_target_rad=q_target, qd_target_rad_s=qd_target, qd_rad_s=np.gradient(q, t), tau_nm=0.1 * q,
                      mode=np.full(t.size, 10.0), run=name, labels={"class": "pickplace"})


def _spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


REPEATS = [_tape(offset, f"r{index}") for index, offset in enumerate((0.0, 0.4, -0.3, 0.1, 0.2))]


def test_the_floor_is_sim2reals(monkeypatch):
    calls = _spy(monkeypatch, actuator, "floor")
    result = evaluate("stock.repeatability_floor", FloorConfig(), REPEATS, subject="x")
    assert calls == ["floor"] and result.value is not None
    monkeypatch.setattr(actuator, "floor", lambda series_, **kwargs: (123.0, [(i, j, 1.0) for i in range(5) for j in range(i + 1, 5)]))
    assert evaluate("stock.repeatability_floor", FloorConfig(), REPEATS, subject="x").value == 123.0


def test_tracking_and_stops_are_sim2reals(monkeypatch):
    calls = _spy(monkeypatch, actuator, "tracking_statistic")
    assert evaluate("stock.tracking_error", TrackingErrorConfig(), REPEATS[:2], subject="x").value is not None
    assert calls == ["tracking_statistic"] * 2
    stops, summary = _spy(monkeypatch, actuator, "stops"), _spy(monkeypatch, actuator, "stop_summary")
    result = evaluate("stock.stop_error", StopErrorConfig(side="output"), REPEATS[:2], subject="x")
    assert stops == ["stops"] * 2 and summary == ["stop_summary"] and result.value is not None
    monkeypatch.setattr(actuator, "stop_summary", lambda signed, percentile: {"value": 7.0, "n": len(signed), "median": 1.0, "max": 9.0,
                                                                               "mean_signed": 0.0, "worst": 0})
    assert evaluate("stock.stop_error", StopErrorConfig(side="output"), REPEATS[:2], subject="x").value == 7.0


def test_model_error_and_the_fidelity_ratio_are_sim2reals(monkeypatch):
    pairs = [(real, real.as_situation(real.q_rad + 2e-3, "twin")) for real in REPEATS[:3]]
    calls = _spy(monkeypatch, series, "rms")
    assert evaluate("stock.model_error", ModelErrorConfig(), pairs, subject="x").value == pytest.approx(2.0, abs=1e-6)
    assert len(calls) == 3
    ratios = _spy(monkeypatch, actuator, "fidelity_ratios")
    rows = [ModelRow("twin", f"r{i}", "pickplace", 3.0) for i in range(3)]
    assert evaluate("stock.fidelity_ratio", FidelityRatioConfig(), FidelityInput(2.0, rows, floor_class="pickplace"), subject="x").value == 1.5
    assert ratios == ["fidelity_ratios"]
    monkeypatch.setattr(actuator, "fidelity_ratios", lambda rows_, floor, how: ({("twin", "pickplace"): [0.25]}, {("twin", "pickplace"): 0.25}))
    assert evaluate("stock.fidelity_ratio", FidelityRatioConfig(), FidelityInput(2.0, rows, floor_class="pickplace"), subject="x").value == 0.25


def test_t2s_split_is_sim2reals(monkeypatch):
    pair_parts, run_parts = _spy(monkeypatch, actuator, "pair_parts"), _spy(monkeypatch, actuator, "run_parts")
    evaluate("stock.move_hold_split", MoveHoldSplitConfig(), REPEATS[:3], subject="x")
    pairs = [(real, real.as_situation(real.q_rad + 1e-3, "twin")) for real in REPEATS[:2]]
    evaluate("stock.move_hold_split", MoveHoldSplitConfig(), pairs, subject="x")
    assert pair_parts == ["pair_parts"] * 3 and run_parts == ["run_parts"] * 2
    holds, summary = _spy(monkeypatch, actuator, "rest_holds"), _spy(monkeypatch, actuator, "rest_summary")
    evaluate("stock.rest_offset", RestOffsetConfig(min_holds=1), pairs, subject="x")
    assert holds == ["rest_holds"] * 2 and summary == ["rest_summary"]
    monkeypatch.setattr(actuator, "split_ratios", lambda moves, floor_moves, rest, floor_holds: (0.5, 2.5))
    result = evaluate("stock.split_fidelity", SplitFidelityConfig(), SplitFidelityInput(1.0, 1.0, 1.0, 1.0), subject="x")
    assert result.value == 2.5 and result.verdict == "fail"


def test_coverage_bins_with_robot_msgs_through_sim2real(monkeypatch):
    calls = _spy(monkeypatch, coverage, "cell_counts")
    result = evaluate("stock.coverage", CoverageConfig(x="q_rad", y="qd_rad_s", x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), bins=(8, 8)), REPEATS,
                      subject="x")
    assert calls == ["cell_counts"] and result.value is not None


@pytest.mark.parametrize("name", ["rms", "difference", "on_common_clock", "pair_rms", "pairwise_rms", "center"])
def test_the_shared_rulers_are_sim2reals(monkeypatch, name):
    # data.<name> answers whatever sim2real.metrics.series.<name> answers: it is a delegate, not a second copy
    monkeypatch.setattr(series, name, lambda *args, **kwargs: ("sim2real", name))
    a = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    arguments = {"rms": ([1.0],), "difference": (a, a), "on_common_clock": (a, a, 0.1), "pair_rms": (a, a), "pairwise_rms": ([a, a],),
                 "center": ([1.0],)}[name]
    assert getattr(data, name)(*arguments) == ("sim2real", name)
