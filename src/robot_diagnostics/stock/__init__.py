"""The stock diagnostics, registered on import. Each is one pure function and one frozen config; a part type gets its
diagnostics by picking these with its own config (``catalog.py``), not by writing code.

All 30 of the catalog's primitives are implemented (the first 13 since the reference; the 17 it only specified landed 2026-10-28):
tracking.py, fidelity.py, truthfulness.py (+ agreement), health.py (+ stamp_delay), coverage.py, ../rollup.py, feedback.py,
events.py (share, event_interval, staleness), response.py (step_response), iso9283.py (pose / path accuracy), characteristic.py
(hysteresis_loop, allan_variance), camera.py (reprojection_error, depth_plane, image_sharpness, marker_drift), spectral.py.
Each cites the standard it computes in its module docstring.
"""

from __future__ import annotations

from ..registry import Spec, register
from ..rollup import GateMatrixConfig, RollupConfig, gate_matrix, rollup
from .camera import (DepthPlaneConfig, MarkerDriftConfig, ReprojectionConfig, SharpnessConfig, depth_plane, image_sharpness, marker_drift,
                     reprojection_error)
from .characteristic import AllanConfig, HysteresisConfig, allan_variance, hysteresis_loop
from .coverage import CoverageConfig, coverage
from .events import EventIntervalConfig, ShareConfig, StalenessConfig, event_interval, share, staleness
from .feedback import FeedbackShareConfig, feedback_share
from .fidelity import FidelityRatioConfig, FloorConfig, ModelErrorConfig, fidelity_ratio, model_error, repeatability_floor
from .iso9283 import PathAccuracyConfig, PoseAccuracyConfig, path_accuracy, pose_accuracy
from .response import StepResponseConfig, step_response
from .health import BoxConfig, FaultCodesConfig, RateConfig, StampDelayConfig, box, fault_codes, sample_rate, stamp_delay
from .spectral import SpectralAngleConfig, SpectralQualityConfig, spectral_angle, spectral_quality
from .tracking import StopErrorConfig, TrackingErrorConfig, stop_error, tracking_error
from .truthfulness import AgreementConfig, KnownLoadConfig, KtConsistencyConfig, agreement, known_load_gain, kt_consistency

STOCK = [
    Spec("stock.tracking_error", tracking_error, TrackingErrorConfig, "tracking error", "real_vs_target", "driver", "kpi_trend", "scalar", "mrad",
         summary="statistic(actual - target) per run, combined over runs"),
    Spec("stock.stop_error", stop_error, StopErrorConfig, "stop error", "real_vs_target", "driver", "distribution", "series", "mrad",
         summary="error left at every dwell; p90 vs the joint's share of the tool gate (T4)"),
    Spec("stock.repeatability_floor", repeatability_floor, FloorConfig, "repeatability floor", "repeatability", "driver", "evidence", "matrix", "mrad",
         summary="median pairwise RMS over identical repeats (T1); the pair matrix is the evidence"),
    Spec("stock.model_error", model_error, ModelErrorConfig, "model error", "sim_vs_real", "driver", "kpi_trend", "scalar", "mrad",
         summary="RMS(real - model) per run, any number of models"),
    Spec("stock.fidelity_ratio", fidelity_ratio, FidelityRatioConfig, "fidelity ratio", "sim_vs_real", "headline", "threshold_band", "series", "1",
         summary="model error / floor per model and tape class (T2 on q, T3 on tau)"),
    Spec("stock.known_load_gain", known_load_gain, KnownLoadConfig, "torque truthfulness (known load)", "truthfulness", "driver", "evidence", "series", "%",
         consumers=("sim2real",), view_extra="xy scatter, reference vs reported torque with the identity line: a slope is the finding",
         summary="drive gain from load-on minus load-off sweeps (T6)"),
    Spec("stock.kt_consistency", kt_consistency, KtConsistencyConfig, "torque truthfulness (Kt vs calibration)", "truthfulness", "health", "kpi_trend", "scalar", "1",
         summary="board Kv vs an independent calibration: reported / true torque"),
    Spec("stock.box", box, BoxConfig, "box check", "health", "health", "threshold_band", "series", "",
         summary="min/max/mean/std of a channel vs four Limits (Zach's BoxChartTask)"),
    Spec("stock.sample_rate", sample_rate, RateConfig, "sample rate", "health", "health", "threshold_band", "series", "Hz",
         summary="achieved rate vs expected +- tolerance, gaps (ROS FrequencyStatus; Zach's EventPeriodTask)"),
    Spec("stock.fault_codes", fault_codes, FaultCodesConfig, "fault and limiter codes", "health", "health", "level_history", "series", "1",
         summary="hard faults and watchdog fail; limiters engaged warn (moteus code table)"),
    Spec("stock.coverage", coverage, CoverageConfig, "coverage", "coverage", "driver", "distribution", "matrix", "1",
         view_extra="2-D occupancy heatmap of the envelope (Zach Oct 28: 'discretize the space'); slices look like image heatmaps",
         summary="share of the declared envelope's cells the runs visited"),
    Spec("stock.rollup", rollup, RollupConfig, "rollup", "rollup", "headline", "rollup_tree", "series", "1",
         summary="a composite's own verdict from its members; caused_by names them"),
    Spec("stock.gate_matrix", gate_matrix, GateMatrixConfig, "gate matrix", "rollup", "headline", "gate_matrix", "matrix", "1",
         summary="subjects x gates; headline = passing cells"),
    # ---- the catalog's specified primitives (2026-10-28) ----
    Spec("stock.feedback_share", feedback_share, FeedbackShareConfig, "feedback share", "sim_vs_real", "driver", "kpi_trend", "scalar", "1",
         summary="RMS(tau - tau_ff) / RMS(tau) over controlled rows: what the feedback loop still supplies (bam_rack_twin vt_kpi)"),
    Spec("stock.share", share, ShareConfig, "share", "real_vs_target", "health", "kpi_trend", "scalar", "1",
         summary="share of true events with its Wilson interval and an explicit better (Zach's PercentTask, fixed)"),
    Spec("stock.agreement", agreement, AgreementConfig, "agreement", "truthfulness", "health", "kpi_trend", "scalar", "%",
         summary="a reported or predicted scalar vs an independent reference: % error, ratio, difference or mismatches"),
    Spec("stock.stamp_delay", stamp_delay, StampDelayConfig, "stamp delay", "health", "health", "threshold_band", "series", "s",
         summary="received - source stamp per message; early, late and zero stamps fail (ROS TimeStampStatus)"),
    Spec("stock.event_interval", event_interval, EventIntervalConfig, "mean time between events", "health", "health", "kpi_trend", "scalar", "h",
         summary="operating time / events (MTBF, MTTI); with no events, a lower bound judged as one"),
    Spec("stock.staleness", staleness, StalenessConfig, "evidence age", "health", "health", "level_history", "series", "h",
         summary="age of each gate's newest evidence vs its validity window; a gate without evidence is never green"),
    Spec("stock.step_response", step_response, StepResponseConfig, "step response", "real_vs_target", "driver", "threshold_band", "series", "%",
         summary="overshoot, 10-90 % rise and 2 % settling time per move (MATLAB stepinfo; ISO 9283 overshoot / stabilisation time)"),
    Spec("stock.pose_accuracy", pose_accuracy, PoseAccuracyConfig, "pose accuracy (ISO 9283)", "real_vs_target", "driver", "distribution", "series", "mm",
         summary="ISO 9283 AP (barycentre vs command) and RP (l + 3 S_l) per pose; the worst pose heads"),
    Spec("stock.path_accuracy", path_accuracy, PathAccuracyConfig, "path accuracy (ISO 9283)", "real_vs_target", "driver", "path_overlay", "series", "mm",
         summary="ISO 9283 AT_P / RT_P over index-aligned cycles; per-point error as the overlay"),
    Spec("stock.hysteresis_loop", hysteresis_loop, HysteresisConfig, "torque-angle loop", "characteristic", "driver", "evidence", "series", "arcmin",
         view_extra="xy loop, torque vs angle: the width at zero torque and the slopes are the finding",
         summary="gap at zero torque (lost motion), stiffness K1..K3 or pre-sliding compliance from the loop's legs"),
    Spec("stock.allan_variance", allan_variance, AllanConfig, "Allan deviation", "characteristic", "driver", "evidence", "series", "rad/s/sqrt(Hz)",
         view_extra="log-log Allan deviation vs tau: the slopes are the noise terms",
         summary="overlapping Allan deviation (IEEE 952); N at tau 1 s, B = floor / 0.664, K at tau 3 s"),
    Spec("stock.reprojection_error", reprojection_error, ReprojectionConfig, "reprojection error", "truthfulness", "health", "distribution", "series", "px",
         summary="RMS reprojection error over every corner and per view (OpenCV calibrateCamera RMS / perViewErrors)"),
    Spec("stock.depth_plane", depth_plane, DepthPlaneConfig, "depth quality (flat target)", "truthfulness", "health", "threshold_band", "series", "% of Z",
         summary="fill rate, plane-fit RMS, Z accuracy and temporal noise in the central ROI (Intel depth quality metrics)"),
    Spec("stock.image_sharpness", image_sharpness, SharpnessConfig, "sharpness", "health", "health", "threshold_band", "series", "1",
         summary="variance of the Laplacian per frame (Pech-Pacheco 2000); the blurriest frame heads"),
    Spec("stock.marker_drift", marker_drift, MarkerDriftConfig, "marker drift", "truthfulness", "health", "threshold_band", "series", "px",
         summary="a fixed tag's offset from its calibrated pixel centre or pose"),
    Spec("stock.spectral_quality", spectral_quality, SpectralQualityConfig, "spectral quality", "characteristic", "health", "distribution", "series", "1",
         summary="SNR per band on a uniform target; smile and keystone (px) on line and edge targets"),
    Spec("stock.spectral_angle", spectral_angle, SpectralAngleConfig, "spectral angle", "truthfulness", "driver", "distribution", "series", "rad",
         summary="spectral angle mapper (Kruse 1993) between measured and reference spectra per material"),
]

for _spec in STOCK:
    register(_spec)
