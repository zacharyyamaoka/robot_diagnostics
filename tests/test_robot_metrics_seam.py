"""The stock checks robot_metrics v0 computes - tracking error, the error at every stop, the limiter shares - compute through it.

    uv run --project /home/bam/io-diagnostics --group dev pytest -q /home/bam/io-diagnostics/tests/diagnostics/test_robot_metrics_seam.py

Zach, 2026-11-17 ("a diagnostic = a metric + a threshold"; docs/peps/0004-stock-diagnostics-on-robot-metrics.md): robot_metrics owns the
math, RobotDx keeps its configs, Limits, judge, Result, records and UI. Written before the checks called robot_metrics (red first), in
test_sim2real_metrics' pattern: a spy proves each check calls robot_metrics' metric, and a sentinel answer proves the check's number IS
that metric's. The numbers themselves are held by the existing oracles (test_rack_oracle*, test_stock) and by the byte-identical re-run
of the stock checks over the rack's real runs (PEP 0004, "Proof").
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import robot_metrics
from robot_metrics import functional as rm_functional
from robotdx.diagnostics import Limits, evaluate
from robotdx.diagnostics.data import Trajectory
from robotdx.diagnostics.stock import FaultCodesConfig, StopErrorConfig, TrackingErrorConfig, health, tracking

STOCK = Path(__file__).resolve().parents[2] / "robotdx" / "diagnostics" / "stock"


def _tape(offset_mrad: float = 0.0, name: str = "run") -> Trajectory:
    """Move to 0.5 rad, dwell 1 s, move back, dwell 1 s, move on: two stops, then parking (test_sim2real_metrics' tape)."""

    t = np.round(np.arange(0.0, 5.0, 0.01), 9)
    q_target = np.interp(t, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 0.5, 0.5, 0.0, 0.0, 0.3])
    qd_target = np.gradient(q_target, t)
    qd_target[(t >= 1.0) & (t <= 2.0)] = 0.0
    qd_target[(t >= 3.0) & (t <= 4.0)] = 0.0
    q = q_target + offset_mrad * 1e-3 + 1e-4 * np.sin(7 * t)
    fault = np.zeros(t.size)
    fault[100:130] = 102.0  # the torque cap on 30 rows
    fault[300:305] = 99.0  # max current on 5
    return Trajectory(t=t, q_rad=q, q_target_rad=q_target, qd_target_rad_s=qd_target, qd_rad_s=np.gradient(q, t), tau_nm=0.1 * q,
                      mode=np.full(t.size, 10.0), fault=fault, run=name, labels={"class": "pickplace"})


REPEATS = [_tape(offset, f"r{index}") for index, offset in enumerate((0.0, 0.4, -0.3, 0.1, 0.2))]


def _spy(monkeypatch, owner, name):
    calls = []
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        calls.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, spy)
    return calls


@pytest.mark.parametrize("statistic, base", [("rms", "ErrorStat"), ("max", "ErrorStat"), ("final", "ErrorStat"), ("p95", "ErrorPercentile")])
def test_tracking_error_is_robot_metrics(monkeypatch, statistic, base):
    metric = getattr(robot_metrics, base)
    calls = _spy(monkeypatch, metric, "update")
    assert evaluate("stock.tracking_error", TrackingErrorConfig(statistic=statistic), REPEATS[:2], subject="x").value is not None
    assert calls == ["update"] * 2  # one run, one fold
    monkeypatch.setattr(metric, "compute", lambda self, state: 0.25)
    assert evaluate("stock.tracking_error", TrackingErrorConfig(statistic=statistic), REPEATS[:2], subject="x").value == 250.0


def test_stop_error_is_robot_metrics(monkeypatch):
    updates, summaries = _spy(monkeypatch, robot_metrics.StopStat, "update"), _spy(monkeypatch, rm_functional, "stop_summary")
    result = evaluate("stock.stop_error", StopErrorConfig(side="output"), REPEATS[:2], subject="x")
    assert updates == ["update"] * 2 and summaries == ["stop_summary"] and result.value is not None and result.measurements[0].value == 4
    monkeypatch.setattr(rm_functional, "stop_summary", lambda signed, percentile: {"value": 7.0, "n": len(signed), "median": 1.0, "max": 9.0,
                                                                                    "mean_signed": 0.0, "worst": 0})
    assert evaluate("stock.stop_error", StopErrorConfig(side="output"), REPEATS[:2], subject="x").value == 7.0


def test_the_public_stops_helper_reads_robot_metrics(monkeypatch):
    calls = _spy(monkeypatch, robot_metrics.StopStat, "stops")
    found = tracking.stops(REPEATS[0], StopErrorConfig())
    assert calls == ["stops"] and len(found) == 2  # two stops; the last dwell of a tape is parking
    assert [round(start, 6) for _, start, _ in found] == [1.0, 3.0]


def test_the_limiter_shares_are_robot_metrics(monkeypatch):
    calls = _spy(monkeypatch, robot_metrics.CodeShare, "update")
    result = evaluate("stock.fault_codes", FaultCodesConfig(), REPEATS[:2], subject="x")
    shares = {m.key: m.value for m in result.measurements}
    assert shares["limiter_99_share"] == 10 / 1000 and shares["limiter_102_share"] == 60 / 1000
    assert calls == ["update"] * 4  # two limiter codes x two runs
    assert result.verdict == "warn" and result.code == "LimiterEngaged"  # 6 % on the cap, over the 2 % warn
    monkeypatch.setattr(robot_metrics.CodeShare, "compute", lambda self, state: 0.0)
    calm = evaluate("stock.fault_codes", FaultCodesConfig(), REPEATS[:2], subject="x")
    assert {m.key: m.value for m in calm.measurements}["limiter_102_share"] == 0.0 and calm.verdict == "pass"


def test_a_metrics_identity_excludes_the_limits():
    # WHY (the prior-art note on RobotDx: its Limits ride inside each config, so a new target changes the config's identity): the
    # config keeps its limits (its public API), and the metric it computes through is the same object whatever the target
    a = tracking.metric(TrackingErrorConfig(limits=Limits.at_most(1.0)))
    b = tracking.metric(TrackingErrorConfig(limits=Limits.at_most(2.0, warn=1.0), scale=1.0, unit="rad", across_runs="median_high"))
    assert TrackingErrorConfig(limits=Limits.at_most(1.0)) != TrackingErrorConfig(limits=Limits.at_most(2.0))
    assert a == b and a.identity == ("ErrorStat", (("statistic", "rms"), ("rows", "all"), ("field", "positions"), ("joint", 0)))
    assert tracking.metric(TrackingErrorConfig(statistic="p95")) == robot_metrics.ErrorPercentile(percentile=95.0)
    assert tracking.metric(TrackingErrorConfig(statistic="max")) == robot_metrics.ErrorStat(statistic="max_abs")
    assert tracking.metric(TrackingErrorConfig(statistic="final")) == robot_metrics.ErrorStat(statistic="mean_abs")
    with pytest.raises(ValueError, match="unknown statistic"):
        tracking.metric(TrackingErrorConfig(statistic="median"))
    t4 = tracking.stop_metric(StopErrorConfig(limits=Limits.at_most(3.6)))
    assert t4 == tracking.stop_metric(StopErrorConfig(limits=Limits.at_most(6.7), side="output", require_side="", scale=1.0, unit="mm"))
    assert t4 == robot_metrics.StopStat(of="error", statistic="p90", include_last_dwell=False, dwell_min_s=0.25, settle_window_s=0.05,
                                        count_first_dwell=True)
    assert tracking.stop_metric(StopErrorConfig(), open_end=True).include_last_dwell is True
    assert tracking.stop_metric(StopErrorConfig(percentile=95.0)).statistic == "p95"
    assert tracking.stop_metric(StopErrorConfig(percentile=97.5)).statistic == "p97.5"
    assert tracking.stop_metric(StopErrorConfig(dwell_min_s=0.5)) != t4
    assert health.limiter_metric(102) == robot_metrics.CodeShare(codes=(102,), channel="fault")


def test_the_covered_checks_import_nothing_from_sim2real():
    for name in ("tracking.py", "health.py"):
        tree = ast.parse((STOCK / name).read_text())
        imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not [module for module in imported if "sim2real" in module], (name, imported)


PROBE = r"""
import json, sys
sys.modules["sim2real"] = None  # sim2real absent: importing it raises ModuleNotFoundError
import numpy as np
from robotdx.diagnostics import evaluate
from robotdx.diagnostics.data import Trajectory
from robotdx.diagnostics.stock import FaultCodesConfig, StopErrorConfig
t = np.round(np.arange(0.0, 4.0, 0.01), 9)
q_target = np.interp(t, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 0.5, 0.0, 0.0])
qd_target = np.gradient(q_target, t)
qd_target[(t >= 1.0) & (t <= 2.0)] = 0.0
fault = np.zeros(t.size)
fault[:50] = 102.0
run = Trajectory(t=t, q_rad=q_target + 1e-3, q_target_rad=q_target, qd_target_rad_s=qd_target, fault=fault, run="r")
stop = evaluate("stock.stop_error", StopErrorConfig(), [run], subject="x")
faults = evaluate("stock.fault_codes", FaultCodesConfig(), [run], subject="x")
print(json.dumps({"stop": stop.value, "share": {m.key: m.value for m in faults.measurements}.get("limiter_102_share"),
                  "sim2real": sys.modules["sim2real"] is None}))
"""


def test_stop_error_and_the_limiter_shares_need_no_sim2real():
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out == {"stop": pytest.approx(1.0), "share": 50 / 400, "sim2real": True}
