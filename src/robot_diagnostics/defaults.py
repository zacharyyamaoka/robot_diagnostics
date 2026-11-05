"""Every catalog entry's default config, as a real config dataclass instance: what makes it ready to run.

``catalog.ENTRIES`` says in words what makes a primitive a part's diagnostic ("percentile=90, limits 3.6 mrad, require_side=output");
this module is the same thing as an instance of the primitive's frozen config, so a host binds it and a page drills into it without a
person translating prose. One row per entry id; ``tests/diagnostics/test_catalog_defaults.py`` proves every entry has one, that it
builds, survives the TOML round trip and runs.

Each row also says where its THRESHOLDS come from (``limits``), which can differ from where the diagnostic itself came from (the
entry's ``prov``): Zach asked for stop error (his), but the 3.6 mrad bound is an agent's.

- ``his`` / ``his code``: Zach's own number (a quote, or his old bam_diagnostics code). These gate.
- ``agent`` / ``proposed``: an agent's default awaiting his yes, or a number from prior art. WHY they never gate (Zach's rule for the
  catalog: proposed entries show but never gate): ``catalog.evaluate`` reports such a result at most as a warning that names the
  limit as proposed; it never fails. Once a person sets the limit themselves (a value that differs from this default), it gates.
- ``none``: no thresholds; the value is a reading (a floor, a KPI to track), judged pass whenever it can be computed.

A config whose required inputs are a per-unit fact nobody has measured yet (a known mass, a nominal voltage, a datasheet value) keeps
that field empty or at the value named in ``source``; the primitive then returns not_run (never a guessed pass) until it is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .core import Limits
from .rollup import GateMatrixConfig, RollupConfig
from .stock import (AgreementConfig, AllanConfig, BoxConfig, CoverageConfig, DepthPlaneConfig, EventIntervalConfig, FaultCodesConfig,
                    FeedbackShareConfig, FidelityRatioConfig, FloorConfig, HysteresisConfig, KnownLoadConfig, KtConsistencyConfig,
                    MarkerDriftConfig, ModelErrorConfig, MoveHoldSplitConfig, PathAccuracyConfig, PoseAccuracyConfig, RateConfig, ReprojectionConfig, RestOffsetConfig, ShareConfig,
                    SharpnessConfig, SpectralAngleConfig, SplitFidelityConfig, SpectralQualityConfig, StalenessConfig, StampDelayConfig, StepResponseConfig,
                    StopErrorConfig, TrackingErrorConfig)

LIMIT_SOURCES = ("his", "his code", "agent", "proposed", "none")
GATING = ("his", "his code")


@dataclass(frozen=True)
class Default:
    config: Any  # an instance of the primitive's config dataclass
    limits: str  # where its thresholds come from: his | his code | agent | proposed | none
    source: str = ""  # the number's origin, when it is not the entry's own ``where``

    def __post_init__(self):
        if self.limits not in LIMIT_SOURCES:
            raise ValueError(f"limits provenance {self.limits!r} not in {LIMIT_SOURCES}")

    @property
    def gating(self) -> bool:
        """True when a fail may fail: the thresholds are Zach's. ``none`` has nothing to gate."""

        return self.limits in GATING


D = Default
at_most, at_least = Limits.at_most, Limits.at_least
KPIS = "Sources/Notes/BAM KPIs.md"
ACC = "bam_rack_twin acceptance.py (agent default, 'Zach to confirm')"
H = 1 / 3600

DEFAULTS: dict[str, Default] = {
    # ================================================================ servo
    "servo.tracking_rms": D(TrackingErrorConfig(statistic="rms", channel="q_rad", controlled_only=True), "none"),
    "servo.tracking_peak": D(TrackingErrorConfig(statistic="p95", channel="q_rad", controlled_only=True), "none"),
    # WHY 3.6 mrad: the strictest joint's share of the 5 mm tool gate (3.6 / 6.7 / 10 mrad); a looser joint sets its own
    "servo.stop_error": D(StopErrorConfig(percentile=90.0, require_side="output", limits=at_most(3.6)), "agent", ACC + " T4"),
    "servo.hold_effort": D(ShareConfig(better="higher", limits=at_least(0.8)), "agent", "pidtuna K4 0.8"),
    "servo.floor": D(FloorConfig(channel="q_rad", clock="common", dt=0.002, min_repeats=5), "none"),
    "servo.twin_fidelity": D(FidelityRatioConfig(model="twin", on_floor_class=True, class_limit=2.0, limits=at_most(1.5)), "agent", ACC + " T2"),
    "servo.torque_fidelity": D(FidelityRatioConfig(model="twin", center="median", class_limit=2.0, limits=at_most(2.0)), "agent", ACC + " T3"),
    "servo.floor_split": D(MoveHoldSplitConfig(dt=0.002), "none"),
    "servo.twin_split": D(MoveHoldSplitConfig(controlled_by="both", on_class="pickplace"), "none"),
    "servo.rest_offset": D(RestOffsetConfig(controlled_by="both", min_holds=4), "none"),
    # WHY 1.5 for each part: the T2 split rule (2026-10-25, the agent's default pick taken while Zach was away, his to overrule)
    "servo.twin_fidelity_split": D(SplitFidelityConfig(limits=at_most(1.5)), "agent", ACC + " T2 split"),
    "servo.torque_truth_load": D(KnownLoadConfig(limits=Limits(warn_low=-2.0, warn_high=2.0, error_low=-5.0, error_high=5.0)), "agent",
                                 ACC + " T6: +-5 % per unit, +-2 % per type; mass_kg and lever_m are the rig's (unset = not run)"),
    "servo.torque_truth_kt": D(KtConsistencyConfig(limits=Limits(warn_low=0.98, warn_high=1.02, error_low=0.95, error_high=1.05)), "proposed",
                               "maxon motor constants tolerance"),
    "servo.feedback_share": D(FeedbackShareConfig(combine="pooled"), "none"),
    "servo.coverage": D(CoverageConfig(x="q_rad", y="qd_rad_s", limits=at_least(0.5, warn=0.8)), "proposed",
                        "the envelope is the joint's range and speed limit; set x_range / y_range per joint"),
    "servo.envelope": D(AgreementConfig(mode="percent", limits=Limits(error_low=-5.0, error_high=5.0)), "agent", ACC + " T7"),
    "servo.thermal": D(ModelErrorConfig(channel="temp_c", controlled_only=False, scale=1.0, unit="Cel", limits=at_most(5.0)), "agent", ACC + " T8"),
    "servo.warm_cold": D(FidelityRatioConfig(model="twin", limits=at_most(1.5)), "agent", ACC + " T9"),
    "servo.live_reid": D(FidelityRatioConfig(model="twin", limits=at_most(1.5)), "agent", ACC + " T10"),
    "servo.mttf": D(EventIntervalConfig(codes=("fault",), limits=at_least(20000.0)), "his", f"{KPIS}:89-91 (Small / Large Servo MTTF 20,000 hrs)"),
    # ================================================================ BLDC driver board
    "driver.faults": D(FaultCodesConfig(fault_codes=(32, 50), limiter_codes=(96, 105), limiter_share_warn=0.02), "his code"),
    "driver.board_temp": D(BoxConfig(channel="temp_c", unit="Cel", headline="max", max=at_most(70.0, warn=60.0), mean=at_most(65.0, warn=55.0)),
                           "his code"),
    # WHY 30-45 V: the band bam_rack_twin's preflight refuses outside on the 36 V supply (the catalog's +-10 % needs a nominal per rack)
    "driver.bus_voltage": D(BoxConfig(channel="v_bus_v", unit="V", headline="min", min=at_least(30.0), max=at_most(45.0)), "proposed",
                            "bam_rack_twin hardware/board.py PREFLIGHT_VOLTAGE_V (30-45 V)"),
    # WHY 50 Hz: his v5 demo's loop period mean_max warn 0.020 s; set expected_hz per bus
    "driver.loop_rate": D(RateConfig(expected_hz=50.0, tolerance=0.1, gap_factor=3.0), "his code"),
    "driver.link_loss": D(ShareConfig(better="lower", scale=100.0, unit="%", limits=at_most(10.0, warn=1.0)), "proposed",
                          "RobotDx P3 'reply_loss < 1 %' (warn); 10 % fail is proposed"),
    "driver.saturation": D(FaultCodesConfig(limiter_codes=(102, 102), limiter_share_warn=0.02), "agent", "bam_rack_twin envelope.py SATURATION_GATE"),
    "driver.config": D(AgreementConfig(mode="mismatches", tolerance=0.0, limits=at_most(0.0)), "proposed"),
    "driver.ripple": D(BoxConfig(channel="tau_nm", unit="N.m", headline="std"), "none",
                       "the 5 % ripple bound needs plateaus keyed to rotor angle; this reads the spread until that view exists"),
    # ================================================================ gearbox
    # WHY the CI half-width is the headline: the T5 / K3 bound is on how well the gap is measured, not on the gap itself
    "gearbox.backlash": D(HysteresisConfig(legs="torque", headline="gap_ci_half", limits=at_most(2.0)), "agent",
                          "catalog: CI half-width <= 2 arcmin (T5 / K3)"),
    "gearbox.stiffness": D(HysteresisConfig(legs="torque", headline="stiffness"), "none"),
    "gearbox.ratio": D(AgreementConfig(mode="percent", limits=Limits(error_low=-0.1, error_high=0.1)), "proposed",
                       "one output turn on camera: 0.1 % of 2 pi = 0.36 deg"),
    "gearbox.transmission_error": D(TrackingErrorConfig(statistic="max", channel="q_rad", scale=180 / math.pi * 60, unit="arcmin"),
                                    "none"),
    "gearbox.wear": D(AgreementConfig(mode="percent", limits=Limits(error_low=-20.0, error_high=20.0)), "agent", "servo-ID ladder K6: flag a 20 % friction change"),
    "gearbox.flank_changes": D(ShareConfig(statistic="count", better="lower", unit="count"), "none"),
    # ================================================================ encoder
    "encoder.output_agreement": D(TrackingErrorConfig(statistic="rms", channel="q_rad"), "none"),
    "encoder.homing": D(FloorConfig(channel="q_rad", relative_to_target=False, min_repeats=5), "none"),
    # WHY 1.19 mrad: AS5047D RMS output noise 0.068 deg (1 sigma); a moteus' own encoder sets its datasheet's
    "encoder.standstill_noise": D(BoxConfig(channel="q_rad", unit="rad", headline="std", std=at_most(0.068 * math.pi / 180)), "proposed",
                                  "AS5047D datasheet 0.068 deg RMS"),
    "encoder.validity": D(FaultCodesConfig(), "proposed", "moteus fault table; replace with the encoder's own code table"),
    # ================================================================ camera, RGB
    "camera.frame_rate": D(RateConfig(expected_hz=30.0, tolerance=0.1, gap_factor=3.0, stamps_scale=1e-3), "proposed",
                           "ROS FrequencyStatus; 30 fps and hardware ms stamps (RealSense hw_ms) until the stream says"),
    "camera.latency": D(StampDelayConfig(min_acceptable_s=-1.0, max_acceptable_s=5.0), "proposed", "ROS TimeStampStatusParam"),
    "camera.intrinsics": D(ReprojectionConfig(headline="rms", limits=at_most(1.0, warn=0.5)), "proposed", "catalog: 0.5 px warn; 1 px fail proposed"),
    "camera.extrinsic_drift": D(MarkerDriftConfig(tag_id=0, space="px", limits=at_most(10.0, warn=3.0)), "proposed",
                                "io-diagnostics saw a 28 px camera move (Oct 21); set tag_id to the rack's fixed tag"),
    "camera.blur": D(SharpnessConfig(statistic="min"), "none", "site-calibrated on labelled frames: a reading until calibrated"),
    "camera.tag_detect": D(ShareConfig(better="higher", slice_by="speed_band"), "none"),
    "camera.sim_overlay": D(FidelityRatioConfig(model="twin", class_limit=2.0, limits=at_most(1.5)), "proposed", "the servo T2 ratio, borrowed"),
    # ================================================================ camera, RGBD (depth)
    "depth.z_accuracy": D(DepthPlaneConfig(headline="z_accuracy", z_accuracy=Limits(warn_low=-1.0, warn_high=1.0, error_low=-2.0, error_high=2.0)),
                          "proposed", "NIST / Intel depth quality: < 1 % good, > 2 % poor"),
    "depth.plane_rms": D(DepthPlaneConfig(headline="plane_rms", roi_fraction=0.4, plane_rms=at_most(1.0, warn=0.4)), "proposed"),
    "depth.fill_rate": D(DepthPlaneConfig(headline="fill_rate", fill_rate=at_least(98.0, warn=99.0)), "proposed"),
    "depth.temporal_noise": D(DepthPlaneConfig(headline="temporal_noise", temporal_noise=at_most(0.5, warn=0.2)), "proposed"),
    "depth.health_check": D(AgreementConfig(mode="difference", unit="1", limits=at_most(0.75, warn=0.25)), "proposed",
                            "RealSense BKM: <= 0.25 ok, >= 0.75 recalibrate (reference 0)"),
    "depth.cloud_vs_sim": D(FidelityRatioConfig(model="twin", class_limit=2.0, limits=at_most(1.5)), "proposed", "the servo T2 ratio, borrowed"),
    # ================================================================ camera, hyperspectral
    "hsi.snr": D(SpectralQualityConfig(headline="snr", snr=at_least(200.0)), "proposed", "HISUI SNR > 200 (unverified)"),
    "hsi.references": D(StalenessConfig(valid_for=(("white*", 3600.0), ("dark*", 3600.0)), default_valid_s=3600.0, expected=("white", "dark")),
                        "proposed", "a white and dark reference per hour until lighting changes reset it"),
    "hsi.spectral_angle": D(SpectralAngleConfig(statistic="median", limits=at_most(0.1)), "proposed", "SAM max_angle 0.1 rad (ENVI default)"),
    "hsi.smile_keystone": D(SpectralQualityConfig(headline="smile", smile=at_most(1.0), keystone=at_most(1.0)), "proposed", "HySIS / HISUI < 1 px"),
    "hsi.frame_rate": D(RateConfig(expected_hz=100.0, tolerance=0.1), "proposed", "set expected_hz from the belt's scan speed"),
    # ================================================================ IMU
    "imu.allan": D(AllanConfig(headline="bias_instability"), "none", "limits from the IMU's datasheet"),
    "imu.static_bias": D(BoxConfig(channel="gyro_rad_s", unit="rad/s", headline="mean"), "none", "mean and std limits from the datasheet"),
    "imu.gravity": D(AgreementConfig(mode="percent", limits=Limits(error_low=-1.0, error_high=1.0)), "proposed", "reference 9.80665 m/s2, +-1 %"),
    "imu.rate": D(RateConfig(expected_hz=200.0, tolerance=0.1), "proposed", "set expected_hz to the IMU's configured rate"),
    # ================================================================ force / torque sensor
    "ft.known_mass": D(KnownLoadConfig(limits=Limits(warn_low=-2.0, warn_high=2.0, error_low=-5.0, error_high=5.0)), "proposed",
                       "ATI known-mass check; the servo T6 bounds, borrowed"),
    "ft.zero_drift": D(BoxConfig(channel="force_n", unit="N", headline="mean"), "none", "mean limits from the sensor's datasheet"),
    "ft.noise": D(BoxConfig(channel="force_n", unit="N", headline="std"), "none", "std limits per axis from the datasheet"),
    "ft.saturation": D(FaultCodesConfig(), "proposed", "status Error bit as a fault code"),
    # ================================================================ gripper
    "gripper.cycle": D(TrackingErrorConfig(statistic="final", channel="width_mm", scale=1.0, unit="mm"), "none"),
    "gripper.detect": D(ShareConfig(better="higher", limits=at_least(0.8)), "proposed"),
    "gripper.grasp_success": D(ShareConfig(better="higher", judge_on="share", slice_by="class", limits=at_least(0.2)), "his",
                               f"{KPIS}:73 (Grasping: Probability of Success 20 %)"),
    "gripper.compute_time": D(BoxConfig(channel="compute_s", unit="s", headline="max", max=at_most(1.0)), "his", f"{KPIS}:75 (Computation time 1 s)"),
    "gripper.force": D(TrackingErrorConfig(statistic="rms", channel="force_n", scale=1.0, unit="N"), "none"),
    "gripper.repeat": D(FloorConfig(channel="width_mm", min_repeats=100, scale=1.0, unit="mm"), "none", "100 consecutive closes (Schunk)"),
    # ================================================================ arm
    "arm.tool_error_at_stop": D(StopErrorConfig(percentile=90.0, side="output", scale=1.0, unit="mm", limits=at_most(5.0)), "his",
                                "Aruco Challenges: 'within 5mm should be ok for grasping'"),
    "arm.path_tracking": D(PathAccuracyConfig(frame="base", headline="at", tolerance_mm=5.0), "his", "the 5 mm tool gate"),
    "arm.pose_ap_rp": D(PoseAccuracyConfig(headline="ap", iso_cycles=30, ap=at_most(5.0)), "his", "the 5 mm tool gate"),
    "arm.overshoot": D(StepResponseConfig(headline="overshoot"), "none", "settle band = the arm's RP once measured"),
    "arm.completion_time": D(AgreementConfig(mode="percent"), "none"),
    "arm.sim_vs_real": D(FidelityRatioConfig(model="twin", limits=at_most(1.5)), "agent", ACC + " T2, per joint"),
    "arm.hold_effort": D(FeedbackShareConfig(normalize=False), "none"),
    "arm.workspace_coverage": D(CoverageConfig(x="q_rad", y="qd_rad_s", limits=at_least(0.5, warn=0.8)), "proposed",
                                "x_mm / y_mm cells once poses are columns"),
    "arm.calibration_drift": D(MarkerDriftConfig(space="mm", limits=at_most(5.0, warn=2.0)), "proposed", "inside the 5 mm tool gate"),
    "arm.closure": D(BoxConfig(channel="closure_um", unit="um", headline="max", max=at_most(1.0)), "agent", "traj_integration_tests Jul 29: 1.0 um closure"),
    "arm.joints": D(RollupConfig(title="joints", tolerate_fail=0, not_run_is="warn"), "his", "the spider rule"),
    # ================================================================ arm + gripper (the pick)
    "pick.sppm": D(ShareConfig(better="higher", slice_by="stream"), "none"),
    "pick.vs_human": D(AgreementConfig(mode="ratio", limits=at_least(1.0)), "proposed", "the human's SPPM as the reference"),
    "pick.rate": D(RateConfig(expected_hz=1.0, tolerance=0.1, gap_factor=3.0), "his", f"{KPIS}:68 (PPM 60 = 1 pick/s)"),
    "pick.mtti": D(EventIntervalConfig(codes=("intervention",), scale=1 / 60, unit="min"), "none"),
    "pick.grasp_pose_error": D(StopErrorConfig(percentile=90.0, side="output", scale=1.0, unit="mm", limits=at_most(5.0)), "his",
                               "Aruco Challenges: 'within 5mm should be ok for grasping'"),
    # ================================================================ deployment / rack
    "rack.gate_matrix": D(GateMatrixConfig(title="hard gates", gates=("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")), "none"),
    "rack.rollup": D(RollupConfig(title="rack", tolerate_fail=0), "his", "acceptance rig: tolerate_fail 0"),
    "rack.evidence_age": D(StalenessConfig(valid_for=(("comms*", 86400.0), ("motion*", 604800.0))), "proposed", "Diagnostics Rollups note P3"),
    "rack.uptime": D(EventIntervalConfig(codes=("system_fault",), limits=at_least(10000.0)), "his", f"{KPIS}:62 (System MTTF 10,000 hrs)"),
    # WHY 1 / 3600 Hz: one stamp per ton processed; 1 ton/h is his system throughput target
    "rack.throughput": D(RateConfig(expected_hz=H, tolerance=0.1, gap_factor=3.0), "his", f"{KPIS}:61 (Throughput 1 ton/hr)"),
    "rack.host": D(BoxConfig(channel="cpu_temp_c", unit="Cel", headline="max", max=at_most(95.0, warn=85.0)), "proposed",
                   "ROS diagnostic_common_diagnostics CPU temperature defaults"),
    "rack.links": D(ShareConfig(better="lower", statistic="count", unit="count", limits=at_most(10.0, warn=3.0)), "his code",
                    "his transport_error_count CounterTask warn 3 / error 10"),
}

__all__ = ["Default", "DEFAULTS", "LIMIT_SOURCES", "GATING"]
