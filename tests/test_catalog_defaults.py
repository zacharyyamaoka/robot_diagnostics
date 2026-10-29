"""Every catalog entry is ready to run: a real default config (defaults.py) that builds, validates, survives the TOML round trip and
runs on a fixture, not run when its inputs are absent; its Result carries the data-ui view hint; and a proposed or agent limit shows but
never gates (Zach's rule for the catalog)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from robotdx.diagnostics import REGISTRY, catalog, record
from robotdx.diagnostics.core import FAIL, HINTS, NOT_RUN, PASS, VERDICTS, WARN, Limits, Result, config_dict
from robotdx.diagnostics.defaults import DEFAULTS, GATING, LIMIT_SOURCES
from robotdx.diagnostics.registry import NO_DATA, config_from_dict
from robotdx.diagnostics.stock.camera import CalibrationView, DepthFrames, Frames, MarkerTrack
from robotdx.diagnostics.stock.characteristic import RateSeries
from robotdx.diagnostics.stock.events import Event, EventLog, EvidenceStamps
from robotdx.diagnostics.stock.fidelity import FidelityInput, ModelRow
from robotdx.diagnostics.stock.health import StampPairs, Stamps
from robotdx.diagnostics.stock.iso9283 import PathCycle, PoseSample
from robotdx.diagnostics.stock.spectral import SpectralSamples, SpectralTargets
from robotdx.diagnostics.stock.truthfulness import KnownLoadInput, KtReadings, Pair
from robotdx.diagnostics.data import Trajectory

from conftest import tape

IDS = [entry.id for entry in catalog.ENTRIES]
ONE = Result(diagnostic="x", title="member", verdict=PASS, value=1.0, unit="1", subject="rack/can1")


def _bare() -> Trajectory:
    """A run that carries none of a diagnostic's channels: a clock and a position that never reported."""

    return Trajectory(t=np.arange(20) * 0.05, q_rad=np.full(20, np.nan))


# What each primitive takes, twice: ABSENT = the data's own shape with nothing in it (the inputs are absent); PRESENT = a small fixture
# that carries them. WHY both shapes: None is the host's "I have no such input", an empty shape is a recording that held none.
ABSENT = {
    "stock.tracking_error": [_bare()], "stock.stop_error": [_bare()], "stock.box": [_bare()], "stock.fault_codes": [_bare()],
    "stock.coverage": [_bare()], "stock.feedback_share": [_bare()], "stock.step_response": [_bare()], "stock.hysteresis_loop": [_bare()],
    "stock.repeatability_floor": [_bare()], "stock.model_error": [], "stock.fidelity_ratio": FidelityInput(None, ()),
    "stock.known_load_gain": KnownLoadInput(loaded=()), "stock.kt_consistency": KtReadings(None, None), "stock.agreement": [],
    "stock.sample_rate": [], "stock.stamp_delay": [], "stock.share": [], "stock.event_interval": EventLog((), ()),
    "stock.staleness": EvidenceStamps(1e9, ()), "stock.allan_variance": [], "stock.pose_accuracy": [], "stock.path_accuracy": [],
    "stock.reprojection_error": [], "stock.depth_plane": DepthFrames(np.full((3, 8, 8), np.nan)), "stock.image_sharpness": Frames(()),
    "stock.marker_drift": MarkerTrack(t=np.zeros(0), tag_id=()), "stock.spectral_quality": SpectralTargets(),
    "stock.spectral_angle": SpectralSamples({}, {}), "stock.rollup": [], "stock.gate_matrix": [],
}


def _present(primitive: str):
    rng = np.random.default_rng(3)
    run = tape()
    if primitive == "stock.repeatability_floor":
        return [tape(run=f"r{i}", offset=0.001 * i) for i in range(5)]
    if primitive == "stock.model_error":
        return [(run, tape(offset=0.002, run="twin"))]
    if primitive in ("stock.tracking_error", "stock.stop_error", "stock.box", "stock.fault_codes", "stock.coverage", "stock.feedback_share",
                     "stock.step_response", "stock.hysteresis_loop"):
        return [run]
    return {
        "stock.fidelity_ratio": FidelityInput(1.0, (ModelRow("twin", "r1", "pickplace", 1.2), ModelRow("twin", "r2", "pickplace", 1.1)),
                                              "repeats", "pickplace"),
        "stock.known_load_gain": KnownLoadInput(loaded=(run,), unloaded=(run,)),
        "stock.kt_consistency": KtReadings(100.0, 101.0),
        "stock.agreement": [Pair(101.0, 100.0, "speed_max")],
        "stock.sample_rate": [Stamps(np.arange(400) / 50.0, "can1")],
        "stock.stamp_delay": [StampPairs(np.arange(1.0, 50.0), np.arange(1.0, 50.0) + 0.02, "cam")],
        "stock.share": [Event(float(i), i % 5 != 0, {"class": "pet", "speed_band": "slow", "stream": "a"}) for i in range(50)],
        "stock.event_interval": EventLog((Event(100.0, False, code="fault"),), ((0.0, 7200.0),)),
        "stock.staleness": EvidenceStamps(1e6, (("comms.reply_loss", 1e6 - 60.0), ("white", 1e6 - 60.0), ("dark", 1e6 - 60.0))),
        "stock.allan_variance": [RateSeries(np.arange(20000) * 0.01, rng.normal(0.0, 0.01, 20000))],
        "stock.pose_accuracy": [PoseSample("P1", tuple(rng.normal(0.0, 0.1, 3)), (0.0, 0.0, 0.0), cycle) for cycle in range(5)],
        "stock.path_accuracy": [PathCycle(np.zeros((10, 3)) + 0.1 * c, np.zeros((10, 3))) for c in range(3)],
        "stock.reprojection_error": [CalibrationView(rng.normal(100.0, 10.0, (20, 2)), projected_px=rng.normal(100.0, 10.0, (20, 2)))],
        "stock.depth_plane": DepthFrames(np.full((5, 16, 16), 1000.0) + rng.normal(0.0, 1.0, (5, 16, 16)), ground_truth_mm=1000.0),
        "stock.image_sharpness": Frames((rng.uniform(0.0, 255.0, (32, 32)),)),
        "stock.marker_drift": MarkerTrack(t=np.arange(3.0), tag_id=(0, 0, 0), center_px=np.full((3, 2), 101.0), position_mm=np.zeros((3, 3)),
                                          calibrated_px=(100.0, 100.0), calibrated_mm=(0.0, 0.0, 0.0)),
        "stock.spectral_quality": SpectralTargets(uniform=1000.0 + rng.normal(0.0, 2.0, (20, 50, 4))),
        "stock.spectral_angle": SpectralSamples({"pet": np.array([[1.0, 0.1]])}, {"pet": np.array([1.0, 0.0])}),
        "stock.rollup": [ONE],
        "stock.gate_matrix": [("CAN 1", "T1", ONE)],
    }[primitive]


def test_every_entry_has_exactly_one_default_and_it_is_its_primitives_config():
    assert set(DEFAULTS) == set(IDS) and len(IDS) == 89
    for entry in catalog.ENTRIES:
        found = catalog.default(entry.id)
        assert isinstance(found.config, REGISTRY[entry.primitive].config_type), entry.id
        assert found.limits in LIMIT_SOURCES, entry.id
        assert dataclasses.is_dataclass(found.config) and type(found.config).__dataclass_params__.frozen


@pytest.mark.parametrize("entry_id", IDS)
def test_every_default_builds_validates_and_survives_the_toml_round_trip(entry_id):
    config = catalog.config(entry_id)
    spec = REGISTRY[catalog.entry(entry_id).primitive]

    def plain(value):  # what a TOML file can hold: no None, lists for tuples
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [plain(v) for v in value]
        return value

    values = plain(config_dict(config))
    json.dumps(values)  # plain data
    assert config_from_dict(spec.config_type, values) == config
    for field in dataclasses.fields(config):  # each Limits is ordered (Limits.__post_init__ ran) and in the value's unit
        if isinstance(getattr(config, field.name), Limits):
            Limits(**dataclasses.asdict(getattr(config, field.name)))


@pytest.mark.parametrize("entry_id", IDS)
def test_every_entry_is_not_run_when_its_inputs_are_absent(entry_id):
    primitive = catalog.entry(entry_id).primitive
    none = catalog.evaluate(entry_id, None, subject="fixture")
    assert (none.verdict, none.code) == (NOT_RUN, NO_DATA) and none.value is None
    empty = catalog.evaluate(entry_id, ABSENT[primitive], subject="fixture")
    assert empty.verdict == NOT_RUN and empty.code and empty.value is None, (entry_id, empty.code, empty.message)


@pytest.mark.parametrize("entry_id", IDS)
def test_every_entry_runs_on_a_fixture_and_serialises_with_its_hint(entry_id):
    item = catalog.entry(entry_id)
    out = catalog.evaluate(entry_id, _present(item.primitive), subject="fixture/part", stamp=1.0)
    assert out.verdict in VERDICTS and out.diagnostic == entry_id and out.title == item.title
    # the stock view: data-ui's vocabulary, exactly
    assert out.hint == item.the_hint and out.hint in HINTS
    if out.verdict == NOT_RUN:  # a fixture that lacks a per-unit fact (a known mass) or a channel this part names: says which
        assert out.code and out.message, entry_id
    else:
        assert out.value is not None, entry_id
    line = record.to_record(out)
    values = {pair["key"]: pair["value"] for pair in line["values"]}
    assert values["dx.hint"] == item.the_hint and values["dx.diagnostic"] == entry_id
    assert record.from_record(line) == out  # no loss: the hint and everything else come back


def test_the_fixtures_compute_a_value_for_most_entries():
    # WHY a floor: "runs" must mean computed, not a pile of not_run; the rest name a channel or a per-unit fact the fixture lacks
    computed = [entry.id for entry in catalog.ENTRIES
                if catalog.evaluate(entry.id, _present(entry.primitive), subject="f").verdict != NOT_RUN]
    assert len(computed) >= 60, sorted(set(IDS) - set(computed))


def test_a_proposed_limit_shows_but_never_fails():
    entry_id = "driver.bus_voltage"  # proposed 30-45 V
    assert catalog.default(entry_id).limits == "proposed" and not catalog.gating(entry_id)
    sagging = tape()
    sagging = dataclasses.replace(sagging, v_bus_v=np.full(sagging.t.size, 20.0))
    out = catalog.evaluate(entry_id, [sagging], subject="rack/can1")
    assert out.verdict == WARN and out.code.startswith("Proposed") and "not a gate" in out.message and out.value == 20.0
    # the same number from the bare primitive fails: the rule is the catalog's, not the function's
    bare = REGISTRY["stock.box"].function(catalog.config(entry_id), [sagging])
    assert bare.verdict == FAIL


def test_a_limit_a_person_sets_gates():
    entry_id = "driver.bus_voltage"
    mine = dataclasses.replace(catalog.config(entry_id), min=Limits.at_least(22.0))
    assert catalog.gating(entry_id, mine)
    sagging = dataclasses.replace(tape(), v_bus_v=np.full(tape().t.size, 20.0))
    assert catalog.evaluate(entry_id, [sagging], subject="rack/can1", config_value=mine).verdict == FAIL


def test_his_limits_gate():
    assert set(GATING) == {"his", "his code"}
    entry_id = "driver.board_temp"  # his code: max warn 60 / error 70
    assert catalog.gating(entry_id)
    hot = dataclasses.replace(tape(), temp_c=np.full(tape().t.size, 80.0))
    assert catalog.evaluate(entry_id, [hot], subject="rack/can1").verdict == FAIL


def test_a_reading_has_no_thresholds():
    # "none" means a reading (a floor, a KPI to track): no Limits that could fail it, so its provenance can never hide a gate
    for entry_id, found in DEFAULTS.items():
        if found.limits != "none":
            continue
        for field in dataclasses.fields(found.config):
            value = getattr(found.config, field.name)
            assert not (isinstance(value, Limits) and not value.empty), (entry_id, field.name)
            assert not (field.name in ("tolerance_mm", "class_limit") and value is not None), (entry_id, field.name)
    assert {d.limits for d in DEFAULTS.values()} == set(LIMIT_SOURCES)


def test_the_proposed_rule_rewrites_only_a_limits_verdict():
    # servo.stop_error's limit is an agent's, but "measured on the encoder side only" is the primitive's own finding: kept as it is
    out = catalog.evaluate("servo.stop_error", [tape()], subject="rack/can1")
    assert out.code == "MeasuredOnEncoderSide" and not out.code.startswith("Proposed")
