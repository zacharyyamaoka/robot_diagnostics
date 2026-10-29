"""Without the diagnostics extra (no numpy): core, registry and record import and work, every stock Spec still registers, and
running a stock diagnostic raises ModuleNotFoundError for numpy (which the host turns into not_run). Proven in a uv environment
of the project's own dependencies only (``--isolated --no-dev``), where numpy is genuinely absent."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UV = "/home/bam/.local/bin/uv"

PROBE = r"""
import importlib.util, json, sys
out = {"numpy": importlib.util.find_spec("numpy") is not None}
from robotdx.diagnostics import core, registry, record
from robotdx.diagnostics import REGISTRY, Limits, evaluate
from robotdx.diagnostics._numpy import available
from robotdx.diagnostics.stock import BoxConfig
out["available"] = available()
out["registered"] = sorted(REGISTRY)
from robotdx.diagnostics.catalog import PRIMITIVES
out["implemented"] = sorted(name for name, p in PRIMITIVES.items() if p.implemented)
out["judge"] = core.judge(61.0, Limits.at_most(70.0, warn=60.0))
line = record.to_record(core.result(BoxConfig(), value=None, unit="Cel", metric="max", limits=Limits(), title="t", message="m", code="NoChannel"))
out["level"] = line["level"]
out["config"] = registry.config_from_dict(BoxConfig, {"max": {"warn_high": 60.0, "error_high": 70.0}}).max.error_high
from robotdx.diagnostics.data import Trajectory
try:
    evaluate("stock.box", BoxConfig(), [Trajectory(t=[0.0, 1.0], q_rad=[0.0, 0.0], temp_c=[40.0, 41.0])], subject="x")
    out["raised"] = None
except ModuleNotFoundError as error:
    out["raised"] = error.name
print(json.dumps(out))
"""


def test_core_registry_and_record_need_no_numpy():
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    done = subprocess.run([UV, "run", "--isolated", "--no-dev", "--project", str(ROOT), "python", "-c", PROBE],
                          capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["numpy"] is False and out["available"] is False  # really no numpy in that environment
    # every primitive the catalog calls implemented registers without numpy (so a page can list and configure it)
    assert "stock.box" in out["registered"] and "stock.fault_codes" in out["registered"]
    assert out["implemented"] == [name for name in out["registered"] if name.startswith("stock.")] and len(out["implemented"]) >= 13
    assert out["judge"] == ["warn", "AboveWarnLimit"]
    assert out["level"] == "unknown"
    assert out["config"] == 70.0
    # a window with data: without numpy its arithmetic cannot start, and the error names numpy (the host's not_run reason)
    assert out["raised"] == "numpy"
