"""Every specified primitive's Result (computed and not run) survives the RobotDx record both ways, is a valid record for the real
contract, and projects onto data-ui's row: the same guarantees test_result_record.py gives the first 13."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np

from robotdx.diagnostics import NOT_RUN, REGISTRY, Limits, evaluate
from robotdx.diagnostics.catalog import PRIMITIVES
from robotdx.diagnostics.record import from_record, to_record
from robotdx.diagnostics.rows import ROW_COLUMNS, result_row
from robotdx.diagnostics.stock import (AgreementConfig, AllanConfig, DepthPlaneConfig, EventIntervalConfig, FeedbackShareConfig,
                                       HysteresisConfig, MarkerDriftConfig, PathAccuracyConfig, PoseAccuracyConfig, ReprojectionConfig,
                                       ShareConfig, SharpnessConfig, SpectralAngleConfig, SpectralQualityConfig, StalenessConfig,
                                       StampDelayConfig, StepResponseConfig)
from robotdx.diagnostics.stock.camera import CalibrationView, DepthFrames, Frames, MarkerTrack
from robotdx.diagnostics.stock.characteristic import RateSeries
from robotdx.diagnostics.stock.events import Event, EventLog, EvidenceStamps
from robotdx.diagnostics.stock.health import StampPairs
from robotdx.diagnostics.stock.iso9283 import PathCycle, PoseSample
from robotdx.diagnostics.stock.spectral import SpectralSamples, SpectralTargets
from robotdx.diagnostics.stock.truthfulness import Pair

from conftest import tape

SPECIFIED = ("feedback_share", "share", "agreement", "step_response", "pose_accuracy", "path_accuracy", "hysteresis_loop", "allan_variance",
             "reprojection_error", "depth_plane", "spectral_quality", "spectral_angle", "stamp_delay", "image_sharpness", "marker_drift",
             "event_interval", "staleness")


def computed_and_not_run() -> list:
    rng = np.random.default_rng(0)
    run = replace(tape(), tau_nm=np.sin(np.arange(600) / 10), tau_ff_nm=0.5 * np.sin(np.arange(600) / 10))
    t = np.arange(0.0, 4.0, 0.001)
    loop_tau = 2 * np.abs(((t / 2) % 2) - 1) - 1
    loop = replace(tape(n=t.size), t=t, tau_nm=loop_tau, q_rad=loop_tau / 200 - 5e-4 * np.sign(np.gradient(loop_tau)))
    step = replace(tape(n=t.size), t=t, q_target_rad=np.where(t < 1, 0.0, 1.0), q_rad=np.where(t < 1, 0.0, 1 - np.exp(-(t - 1) / 0.05)),
                   qd_target_rad_s=np.zeros(t.size))
    board = np.stack(np.meshgrid(np.arange(4) * 10.0, np.arange(3) * 10.0), -1).reshape(-1, 2)
    now = 1e9
    pairs = [
        ("stock.feedback_share", FeedbackShareConfig(), [run], [tape()]),
        ("stock.share", ShareConfig(limits=Limits.at_least(0.8)), [Event(float(i), i % 4 != 0, {"c": i % 2}) for i in range(20)], []),
        ("stock.agreement", AgreementConfig(limits=Limits(error_low=-1.0, error_high=1.0)), [Pair(9.7, 9.80665, "g", "imu.npz")], [Pair(1.0, None)]),
        ("stock.step_response", StepResponseConfig(overshoot=Limits.at_most(5.0)), [step], [replace(tape(), q_target_rad=None)]),
        ("stock.pose_accuracy", PoseAccuracyConfig(), [PoseSample("P1", (float(i % 3), 0.0, 0.0), (0.0, 0.0, 0.0), i) for i in range(6)], []),
        ("stock.path_accuracy", PathAccuracyConfig(), [PathCycle(np.ones((20, 3)) * k, np.zeros((20, 3)), run=f"c{k}") for k in (1.0, 1.1)], []),
        ("stock.hysteresis_loop", HysteresisConfig(), [loop], [tape()]),
        ("stock.allan_variance", AllanConfig(), [RateSeries(np.arange(20000) / 100.0, rng.normal(0, 0.01, 20000), "imu")], []),
        ("stock.reprojection_error", ReprojectionConfig(), [CalibrationView(board, projected_px=board + 0.3, view_id="v")], []),
        ("stock.depth_plane", DepthPlaneConfig(), DepthFrames(1000 + rng.normal(0, 1, (3, 40, 50)), fx=50.0, fy=50.0, cx=25.0, cy=20.0), DepthFrames(np.zeros((1, 4, 4)))),
        ("stock.spectral_quality", SpectralQualityConfig(), SpectralTargets(uniform=100 + rng.normal(0, 1, (5, 30, 4))), SpectralTargets()),
        ("stock.spectral_angle", SpectralAngleConfig(), SpectralSamples({"pet": np.array([1.0, 2.0])}, {"pet": np.array([1.0, 1.0])}), SpectralSamples({}, {})),
        ("stock.stamp_delay", StampDelayConfig(), [StampPairs(np.arange(1.0, 10.0), np.arange(1.0, 10.0) + 0.01, "cam")], [StampPairs(np.zeros(0), np.zeros(0))]),
        ("stock.image_sharpness", SharpnessConfig(), Frames([rng.uniform(0, 1, (16, 16))], source_id="cam"), Frames([])),
        ("stock.marker_drift", MarkerDriftConfig(), MarkerTrack(np.arange(3.0), [0, 0, 0], np.ones((3, 2)), calibrated_px=(0.0, 0.0)),
         MarkerTrack(np.arange(3.0), [0, 0, 0], np.ones((3, 2)))),
        ("stock.event_interval", EventIntervalConfig(), EventLog([Event(10.0, False, code="fault")], ((0.0, 7200.0),)), EventLog([], ())),
        ("stock.staleness", StalenessConfig(), EvidenceStamps(now, (("comms.loss", now - 100.0), ("motion.T4", now - 9 * 86400.0))), EvidenceStamps(now, ())),
    ]
    out = []
    for name, config, good, missing in pairs:
        out.append(evaluate(name, config, good, subject="rack/can14", stamp=now))
        out.append(evaluate(name, config, missing, subject="rack/can14"))
    return out


def test_every_specified_primitive_is_registered_and_implemented():
    for name in SPECIFIED:
        assert PRIMITIVES[f"stock.{name}"].implemented and f"stock.{name}" in REGISTRY
    assert all(p.implemented for p in PRIMITIVES.values())  # the catalog's 30, none left specified-only


def test_each_specified_primitive_computes_once_and_is_not_run_once():
    results = computed_and_not_run()
    assert {r.diagnostic for r in results} == {f"stock.{name}" for name in SPECIFIED}
    for computed, missing in zip(results[::2], results[1::2]):
        assert computed.value is not None and math.isfinite(computed.value), computed.diagnostic
        assert missing.verdict == NOT_RUN and missing.code and missing.value is None, missing.diagnostic


def test_every_specified_result_survives_the_record_and_the_real_contract(contract):
    for result in computed_and_not_run():
        line = json.loads(json.dumps(to_record(result)))
        assert from_record(line) == result, result.diagnostic
        status = contract.from_dict(to_record(result, expected_period_s=1.0))
        assert contract.validate(status) == [], (result.diagnostic, contract.validate(status))
        assert from_record(contract.to_dict(status)) == result, result.diagnostic
        row = result_row(result)
        assert tuple(row) == ROW_COLUMNS and row["hint"] == PRIMITIVES[result.diagnostic].hint
