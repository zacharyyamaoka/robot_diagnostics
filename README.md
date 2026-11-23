# Robot Diagnostics (RobotDx)

Robot diagnostics as pure functions. A diagnostic is `diagnose(config, data) -> Result`: the config is a frozen dataclass (thresholds,
windows, which channel), the data is a window of arrays or rows, and the Result carries the number, the verdict
(`pass` / `warn` / `fail` / `not_run`), the limits it was judged against and its evidence. Computing the number and checking it against a
threshold are two separate steps, so one metric serves any limit.

```bash
pip install robot-diagnostics            # the core: stdlib only
pip install 'robot-diagnostics[stock]'   # + numpy, Pillow, robot-metrics, sim2real: what the stock checks compute with
```

```python
import numpy as np
from robot_diagnostics import Limits, evaluate
from robot_diagnostics.data import Trajectory
from robot_diagnostics.stock import BoxConfig

t = np.arange(0.0, 10.0, 0.01)
run = Trajectory(t=t, q_rad=np.zeros(t.size), temp_c=55.0 + t, run="r1")
result = evaluate("stock.box", BoxConfig(max=Limits.at_most(70.0, warn=60.0)), [run], subject="arm/joint3")
print(result.verdict, result.code, round(result.value, 2), result.unit)
```

```
warn MaxAboveWarnLimit 64.99 Cel
```

## What is in it

| Module | What it holds |
|---|---|
| `robot_diagnostics` (`core`, `registry`) | `Limits`, `Result`, `Evidence`, `judge`, `worst`; the registry (`evaluate`, `register`, `names`, `load_toml`, entry points in group `robotdx.diagnostics`) |
| `robot_diagnostics.stock` | 33 stock checks: tracking and stop error, repeatability floor, model error and fidelity, coverage, sample rate, fault codes, temperature box, step response, hysteresis, ISO 9283 pose / path accuracy, Allan variance, camera (reprojection, depth plane, sharpness, marker drift), spectral, event interval, staleness, roll-ups and gate matrices |
| `robot_diagnostics.catalog`, `defaults` | One catalog entry per part type and check, with default thresholds and where each threshold came from; a proposed limit shows but never gates |
| `robot_diagnostics.data` | `Trajectory`: a run's columns, each name ending in its unit (`q_rad`, `qd_rad_s`, `temp_c`) |
| `robot_diagnostics.stream` | Slice a live stream into windows and evaluate each (`monitor`, `windows`) |
| `robot_diagnostics.record`, `rows` | A Result as a `diagnostics-record/0` status (`to_record` / `from_record`, lossless) and as a table row |
| `robot_diagnostics.adapters.rack_npz` | Loads a recorded servo run (`.npz`) as a `Trajectory` |

## Dependencies

The core (`core`, `registry`, `record`, `rows`, `catalog`, `defaults`) imports nothing outside the standard library, so a host can list,
configure and record every diagnostic without numpy. The `stock` extra adds what the checks compute with, imported on first use:
without it every module still imports and every check still registers, and running one raises `ModuleNotFoundError` naming the missing
package, which a host reports as `not_run`.

robot-metrics and sim2real are not on PyPI yet; until they are, install them from their repos to use the checks that call them
(tracking / stop error and the limiter shares go through robot-metrics; the floor, model error, fidelity, the move / hold split and
coverage go through sim2real).

## Tests

```bash
uv run --extra test --extra stock python -m pytest -q
```

Tests that need the rack's recorded runs, OpenCV or SciPy skip, naming what is missing.

## Provenance

Split out of io-diagnostics (the RobotDx app, where it was the `robotdx.diagnostics` package) at io-diagnostics commit `b64d85a`, with
its history kept (`git filter-repo`). The app now depends on this package. `robotdx.diagnostics` stays the record `source` and the entry
point group, so records written before the split read the same.

MIT licence.
