"""sim2real.metrics, imported on first use: ``from ._sim2real import actuator, series`` in place of importing sim2real at module load.

WHY sim2real (Zach, 2026-11-10, "Sim2Real as a Pure Library on robot_msgs", step 4): the stock checks that are also the servo gates T1-T4
(the floor, tracking and stop error, model error, the fidelity ratio, T2's move / hold split and rest offset) and the coverage check's
binning were written twice, here and in bam_rack_twin's acceptance gates. Their formulas now live once in sim2real.metrics; these checks
keep their Trajectory, their row masks, their configs and their Result, and call it.

WHY lazy (as ``_numpy``): sim2real is in the ``diagnostics`` extra beside numpy, never a dependency of the host. Without it every module still
imports and every stock Spec still registers; running a check that needs it raises ``ModuleNotFoundError`` (name ``sim2real``, or ``numpy``),
which the host reports as not_run.
"""

from __future__ import annotations

import importlib


class _Module:
    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attribute: str):
        if self._module is None:
            self._module = importlib.import_module(self._name)  # raises ModuleNotFoundError(name="sim2real") when the extra is absent
        # WHY looked up each time (the module cached, never the function): what runs is always sim2real's current function
        return getattr(self._module, attribute)


series = _Module("sim2real.metrics.series")
actuator = _Module("sim2real.metrics.actuator")
coverage = _Module("sim2real.metrics.coverage")

__all__ = ["series", "actuator", "coverage"]
