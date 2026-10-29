"""The catalog's specified camera and spectral primitives against answers known by construction (written before the code).

reprojection_error (OpenCV's pinhole + Brown-Conrady model, projected here by hand), depth_plane (a tilted plane through a known
camera; Gaussian noise of known sigma), image_sharpness (the variance of the Laplacian of a pixel checkerboard is 64 A^2),
marker_drift (a known offset), spectral_quality (known SNR per band, a line whose centre shifts by a known smile, an edge whose
position shifts by a known keystone), spectral_angle (angles between known vectors).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotdx.diagnostics import FAIL, NOT_RUN, PASS, Limits, evaluate
from robotdx.diagnostics.stock import (DepthPlaneConfig, MarkerDriftConfig, ReprojectionConfig, SharpnessConfig, SpectralAngleConfig,
                                       SpectralQualityConfig)
from robotdx.diagnostics.stock.camera import CalibrationView, Camera, DepthFrames, Frames, MarkerTrack
from robotdx.diagnostics.stock.spectral import SpectralSamples, SpectralTargets


def measures(result) -> dict:
    return {m.key: m.value for m in result.measurements}


# ----- reprojection_error: OpenCV calibrateCamera's RMS and perViewErrors -------------------------------------------------------
CAM = Camera(fx=600.0, fy=610.0, cx=320.0, cy=240.0)


def _board() -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(7) * 30.0, np.arange(5) * 30.0)
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)


def _pinhole(points_camera: np.ndarray, camera: Camera) -> np.ndarray:
    x, y, z = points_camera.T
    return np.stack([camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy], axis=1)


def test_reprojection_of_a_known_camera_is_zero_and_a_known_offset_is_that_offset():
    board, tvec = _board(), np.array([-90.0, -60.0, 1000.0])
    exact = _pinhole(board + tvec, CAM)
    view = CalibrationView(exact, object_points=board, camera=CAM, rvec=(0.0, 0.0, 0.0), tvec=tuple(tvec), view_id="v0")
    zero = evaluate("stock.reprojection_error", ReprojectionConfig(), [view], subject="s")
    assert zero.value == pytest.approx(0.0, abs=1e-9) and zero.unit == "px" and zero.verdict == PASS
    shifted = CalibrationView(exact + np.array([3.0, 4.0]), object_points=board, camera=CAM, tvec=tuple(tvec), view_id="v1")
    out = evaluate("stock.reprojection_error", ReprojectionConfig(limits=Limits.at_most(1.0, warn=0.5)), [view, shifted], subject="s")
    # OpenCV: RMS over every point of both views = sqrt((0 + 25 n) / 2n); per view sqrt(sum / n) = 0 and 5
    assert out.value == pytest.approx(math.sqrt(12.5)) and out.payload["x"] == ["v0", "v1"]
    assert out.payload["y"] == [pytest.approx(0.0, abs=1e-9), pytest.approx(5.0)] and measures(out)["worst_view"] == pytest.approx(5.0)
    assert (out.verdict, out.code) == (FAIL, "AboveErrorLimit")


def test_reprojection_uses_rodrigues_and_opencv_distortion():
    board = _board()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # rvec (0, 0, pi/2)
    tvec = np.array([10.0, -20.0, 900.0])
    exact = _pinhole(board @ rotation.T + tvec, CAM)
    view = CalibrationView(exact, object_points=board, camera=CAM, rvec=(0.0, 0.0, math.pi / 2), tvec=tuple(tvec))
    assert evaluate("stock.reprojection_error", ReprojectionConfig(), [view], subject="s").value == pytest.approx(0.0, abs=1e-6)
    # one point at normalised (0.3, 0.1): r^2 = 0.1, k1 = -0.2 -> radial 0.98; p1 = 0.01 -> x_d = 0.294 + 2 p1 x y, y_d = 0.098 + p1 (r^2 + 2 y^2)
    distorted = Camera(fx=600.0, fy=610.0, cx=320.0, cy=240.0, dist=(-0.2, 0.0, 0.01, 0.0, 0.0))
    x_d, y_d = 0.294 + 2 * 0.01 * 0.3 * 0.1, 0.098 + 0.01 * (0.1 + 2 * 0.01)
    detected = np.array([[600.0 * x_d + 320.0, 610.0 * y_d + 240.0]])
    one = CalibrationView(detected, object_points=np.array([[300.0, 100.0, 1000.0]]), camera=distorted)
    assert evaluate("stock.reprojection_error", ReprojectionConfig(), [one], subject="s").value == pytest.approx(0.0, abs=1e-9)


def test_reprojection_takes_projected_corners_directly_and_is_not_run_without_them():
    detected = np.array([[10.0, 10.0], [20.0, 20.0]])
    out = evaluate("stock.reprojection_error", ReprojectionConfig(), [CalibrationView(detected, projected_px=detected + [0.0, 1.0])], subject="s")
    assert out.value == pytest.approx(1.0)
    none = evaluate("stock.reprojection_error", ReprojectionConfig(), [CalibrationView(detected)], subject="s")
    assert (none.verdict, none.code) == (NOT_RUN, "NoProjection")
    assert evaluate("stock.reprojection_error", ReprojectionConfig(), [], subject="s").code == "NoViews"


# ----- depth_plane: Intel's depth quality metrics on a flat target ---------------------------------------------------------------
H, W, FX, FY, CX, CY = 120, 160, 200.0, 200.0, 80.0, 60.0


def _tilted_plane() -> np.ndarray:
    """Z = 1000 + 0.2 X (mm) seen through the camera: Z(u) = 1000 / (1 - 0.2 (u - cx) / fx). Not linear in pixels."""

    u = np.arange(W)[None, :].repeat(H, axis=0)
    return 1000.0 / (1 - 0.2 * (u - CX) / FX)


def test_depth_plane_fits_in_3d_so_a_tilted_plane_is_flat():
    frames = DepthFrames(np.stack([_tilted_plane()] * 3), fx=FX, fy=FY, cx=CX, cy=CY)
    out = evaluate("stock.depth_plane", DepthPlaneConfig(), frames, subject="s")
    assert out.metric == "plane_rms" and out.unit == "% of Z" and out.value == pytest.approx(0.0, abs=1e-6)
    assert measures(out)["fill_rate"] == pytest.approx(100.0)
    pixel_space = evaluate("stock.depth_plane", DepthPlaneConfig(), DepthFrames(frames.depth), subject="s")  # no intrinsics
    assert pixel_space.value > 1e-3  # the plane is curved in pixel space: only a 3D fit sees it flat


def test_depth_plane_noise_fill_rate_and_z_accuracy():
    rng = np.random.default_rng(3)
    depth = 1000.0 + rng.normal(0.0, 2.0, (40, H, W))
    holes = (np.add.outer(np.arange(H), np.arange(W)) % 4) == 0  # 25 % of any 4-aligned window
    depth[:, holes] = 0.0
    frames = DepthFrames(depth, fx=FX, fy=FY, cx=CX, cy=CY, ground_truth_mm=1010.0)
    out = evaluate("stock.depth_plane", DepthPlaneConfig(plane_rms=Limits.at_most(1.0, warn=0.4)), frames, subject="s")
    m = measures(out)
    assert out.value == pytest.approx(0.2, rel=0.05)  # 2 mm at 1000 mm
    assert m["temporal_noise"] == pytest.approx(0.2, rel=0.05) and m["fill_rate"] == pytest.approx(75.0)
    assert m["z_accuracy"] == pytest.approx(100 * (1000.0 - 1010.0) / 1010.0, abs=0.01)
    assert out.verdict == PASS
    poor_fill = evaluate("stock.depth_plane", DepthPlaneConfig(fill_rate=Limits.at_least(98.0, warn=99.0)), frames, subject="s")
    assert (poor_fill.verdict, poor_fill.code) == (FAIL, "FillRateBelowErrorLimit")


def test_depth_plane_is_not_run_without_valid_depth():
    out = evaluate("stock.depth_plane", DepthPlaneConfig(), DepthFrames(np.zeros((2, H, W))), subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "NoValidDepth")
    assert evaluate("stock.depth_plane", DepthPlaneConfig(), DepthFrames(np.zeros((0, H, W))), subject="s").code == "NoFrames"


# ----- image_sharpness: variance of the Laplacian (Pech-Pacheco 2000; cv2.Laplacian(ksize=1).var()) -------------------------------
def _checker(amplitude: float, size: int = 64) -> np.ndarray:
    return amplitude * (-1.0) ** np.add.outer(np.arange(size), np.arange(size))


def test_sharpness_of_a_pixel_checkerboard_is_64_a_squared_and_flat_is_zero():
    # every pixel's four neighbours are -c, so the Laplacian is -8c everywhere (REFLECT_101 keeps the pattern at the border)
    out = evaluate("stock.image_sharpness", SharpnessConfig(), Frames([_checker(10.0), _checker(5.0)]), subject="s")
    assert out.value == pytest.approx(64 * 25.0) and out.metric == "min"  # the blurrier frame heads
    assert out.payload["y"] == [pytest.approx(6400.0), pytest.approx(1600.0)]
    flat = evaluate("stock.image_sharpness", SharpnessConfig(limits=Limits.at_least(100.0)), Frames([np.full((32, 32), 7.0)]), subject="s")
    assert flat.value == pytest.approx(0.0) and (flat.verdict, flat.code) == (FAIL, "BelowErrorLimit")


def test_sharpness_falls_with_blur_and_reads_an_roi():
    rng = np.random.default_rng(5)
    sharp = rng.uniform(0, 255, (80, 80))
    padded = np.pad(sharp, 1, mode="edge")
    blurred = sum(padded[1 + dy:81 + dy, 1 + dx:81 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    values = evaluate("stock.image_sharpness", SharpnessConfig(statistic="median"), Frames([sharp, blurred]), subject="s").payload["y"]
    assert values[1] < 0.2 * values[0]
    half = np.zeros((64, 64))
    half[:, 32:] = _checker(10.0)[:, 32:]
    left = evaluate("stock.image_sharpness", SharpnessConfig(roi=(0.0, 0.0, 0.4, 1.0)), Frames([half]), subject="s")
    assert left.value == pytest.approx(0.0)
    assert evaluate("stock.image_sharpness", SharpnessConfig(), Frames([]), subject="s").code == "NoFrames"


# ----- marker_drift: a fixed tag's offset from its calibrated value --------------------------------------------------------------
def test_marker_drift_is_the_offset_from_the_calibrated_pose():
    n = 50
    rng = np.random.default_rng(9)
    centres = np.array([320.0, 240.0]) + np.array([3.0, 4.0]) + rng.normal(0.0, 0.05, (n, 2))
    ids = [3] * n
    other = np.full((10, 2), 900.0)  # another tag, far away: not this check's business
    track = MarkerTrack(t=np.arange(n + 10, dtype=float), tag_id=ids + [5] * 10, center_px=np.vstack([centres, other]),
                        position_mm=np.vstack([np.tile([0.0, 0.0, 502.0], (n, 1)), np.zeros((10, 3))]),
                        calibrated_px=(320.0, 240.0), calibrated_mm=(0.0, 0.0, 500.0))
    out = evaluate("stock.marker_drift", MarkerDriftConfig(tag_id=3, limits=Limits.at_most(2.0)), track, subject="s")
    assert out.value == pytest.approx(5.0, abs=0.05) and out.unit == "px" and (out.verdict, out.code) == (FAIL, "AboveErrorLimit")
    mm = evaluate("stock.marker_drift", MarkerDriftConfig(tag_id=3, space="mm"), track, subject="s")
    assert mm.value == pytest.approx(2.0) and mm.unit == "mm"


def test_marker_drift_is_not_run_without_detections_or_calibration():
    track = MarkerTrack(t=np.arange(3.0), tag_id=[1, 1, 1], center_px=np.zeros((3, 2)), calibrated_px=None)
    assert evaluate("stock.marker_drift", MarkerDriftConfig(tag_id=1), track, subject="s").code == "NoCalibration"
    assert evaluate("stock.marker_drift", MarkerDriftConfig(tag_id=2), replace_cal(track), subject="s").code == "NoDetections"


def replace_cal(track):
    from dataclasses import replace

    return replace(track, calibrated_px=(0.0, 0.0))


# ----- spectral_quality: SNR per band, smile, keystone ------------------------------------------------------------------------
def test_spectral_snr_per_band_is_mean_over_temporal_noise():
    rng = np.random.default_rng(13)
    bands = 8
    means = 1000.0 + 10.0 * np.arange(bands)
    # 50 captures of 400 pixels: temporal noise sigma 5, plus a fixed pixel-to-pixel pattern (sigma 20, constant in time) that a
    # temporal SNR must ignore (a spatial std would read it as noise)
    pattern = rng.normal(0.0, 20.0, (1, 400, 1))
    cubes = means[None, None, :] + pattern + rng.normal(0.0, 5.0, (50, 400, bands))
    out = evaluate("stock.spectral_quality", SpectralQualityConfig(snr=Limits.at_least(150.0, warn=180.0)), SpectralTargets(uniform=cubes), subject="s")
    assert out.metric == "snr" and out.value == pytest.approx(1000.0 / 5.0, rel=0.02)  # the worst band is band 0
    assert out.payload["y"] == [pytest.approx(m / 5.0, rel=0.02) for m in means]
    assert out.verdict == PASS


def test_spectral_smile_and_keystone_are_the_peak_to_peak_shifts():
    columns, spectral = 64, 100
    centre = 50.0 + 0.8 * np.arange(columns) / (columns - 1)  # the line drifts 0.8 px across the slit: smile 0.8 px
    s = np.arange(spectral)[None, :]
    line = 1000.0 * np.exp(-0.5 * ((s - centre[:, None]) / 1.5) ** 2)
    bands, width = 16, 100
    edge = 30.0 + 0.5 * np.arange(bands) / (bands - 1)  # the edge moves 0.5 px across the bands: keystone 0.5 px
    x = np.arange(width)[None, :]
    profile = 1.0 / (1.0 + np.exp(-(x - edge[:, None]) / 1.0))
    targets = SpectralTargets(line_frame=line, edge_frame=profile)
    smile = evaluate("stock.spectral_quality", SpectralQualityConfig(headline="smile", smile=Limits.at_most(1.0)), targets, subject="s")
    assert smile.value == pytest.approx(0.8, abs=1e-3) and smile.unit == "px" and smile.verdict == PASS
    assert measures(smile)["keystone"] == pytest.approx(0.5, abs=0.01)
    assert evaluate("stock.spectral_quality", SpectralQualityConfig(headline="snr"), targets, subject="s").code == "NoUniformTarget"
    assert evaluate("stock.spectral_quality", SpectralQualityConfig(), SpectralTargets(), subject="s").code == "NoTargets"


# ----- spectral_angle: SAM (Kruse et al. 1993) --------------------------------------------------------------------------------
def test_spectral_angle_between_known_spectra():
    samples = SpectralSamples(measured={"pet": np.array([[1.0, 1.0], [2.0, 2.0]]), "pp": np.array([3.0, 0.0]), "hdpe": np.array([[0.0, 1.0]])},
                              references={"pet": np.array([1.0, 0.0]), "pp": np.array([1.0, 0.0]), "hdpe": np.array([1.0, 0.0])})
    out = evaluate("stock.spectral_angle", SpectralAngleConfig(materials=("pet", "pp")), samples, subject="s")
    assert out.value == pytest.approx(math.pi / 4) and out.unit == "rad"
    assert out.payload["x"] == ["pet", "pp"] and out.payload["y"] == [pytest.approx(math.pi / 4), pytest.approx(0.0)]  # brightness-blind
    every = evaluate("stock.spectral_angle", SpectralAngleConfig(limits=Limits.at_most(1.0)), samples, subject="s")
    assert every.value == pytest.approx(math.pi / 2) and (every.verdict, every.code) == (FAIL, "AboveErrorLimit")


def test_spectral_angle_is_not_run_without_a_reference():
    samples = SpectralSamples(measured={"pet": np.array([1.0, 1.0, 1.0])}, references={"pet": np.array([1.0, 0.0])})
    assert evaluate("stock.spectral_angle", SpectralAngleConfig(), samples, subject="s").code == "NoReference"  # band counts differ
    assert evaluate("stock.spectral_angle", SpectralAngleConfig(), SpectralSamples({}, {}), subject="s").verdict == NOT_RUN
