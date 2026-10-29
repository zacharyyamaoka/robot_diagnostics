"""Cameras: calibration truthfulness, depth quality on a flat target, blur, and a fixed tag that should not move.

- ``reprojection_error``: OpenCV ``calibrateCamera``'s returned RMS (sqrt of the summed squared corner residuals over every point of
  every view) and its ``perViewErrors`` (the same per view), with OpenCV's projection model (Rodrigues rotation, pinhole, Brown-Conrady
  distortion k1 k2 p1 p2 k3) when the caller gives the model instead of projected corners.
- ``depth_plane``: Intel's depth quality metrics (RealSense "Depth image quality" whitepaper; librealsense depth-quality tool) on a flat
  target in a central ROI: fill rate, plane-fit RMS (orthogonal distances to the least-squares plane through the deprojected points, % of
  the mean Z), Z accuracy (median Z vs the ground-truth distance, %), temporal noise (per-pixel std over frames, median, % of Z).
- ``image_sharpness``: the variance of the Laplacian (Pech-Pacheco et al. 2000), exactly ``cv2.Laplacian(img, CV_64F).var()``:
  the 3x3 kernel [[0,1,0],[1,-4,1],[0,1,0]] with OpenCV's default BORDER_REFLECT_101 (numpy's "reflect") and the population variance.
- ``marker_drift``: a fixed fiducial's detected pixel centre (or pose) against its calibrated value; io-diagnostics' camera-moved check
  (a 28 px move) generalised; RealSense's on-chip health check is the same idea for depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np

from ..core import NOT_RUN, PASS, Evidence, Limits, Measure, judge, result, worst


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


# ----- reprojection_error ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()  # OpenCV order: k1, k2, p1, p2[, k3]


@dataclass(frozen=True)
class CalibrationView:
    detected_px: np.ndarray  # (N, 2) the corners the detector found
    projected_px: np.ndarray | None = None  # (N, 2) the model's projection; or give object_points + camera (+ rvec, tvec)
    object_points: np.ndarray | None = None  # (N, 3) in the target's frame
    camera: Camera | None = None
    rvec: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Rodrigues, target -> camera
    tvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    view_id: str = ""


def rodrigues(rvec: Sequence[float]) -> np.ndarray:
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-15:
        return np.eye(3)
    k = np.asarray(rvec, float) / theta
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * cross + (1 - math.cos(theta)) * cross @ cross


def project(points: np.ndarray, camera: Camera, rvec: Sequence[float] = (0.0, 0.0, 0.0), tvec: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """OpenCV ``projectPoints``: X_c = R X + t; x = X/Z, y = Y/Z; Brown-Conrady distortion; u = fx x_d + cx, v = fy y_d + cy."""

    p = np.asarray(points, float) @ rodrigues(rvec).T + np.asarray(tvec, float)
    x, y = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    k1, k2, p1, p2, k3 = (tuple(camera.dist) + (0.0,) * 5)[:5]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_d = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([camera.fx * x_d + camera.cx, camera.fy * y_d + camera.cy], axis=1)


@dataclass(frozen=True)
class ReprojectionConfig:
    headline: str = "rms"  # rms: OpenCV's overall RMS | worst_view: the largest per-view RMS
    limits: Limits = field(default_factory=Limits)  # px


def reprojection_error(config: ReprojectionConfig, views: Sequence[CalibrationView]):
    """Overall RMS reprojection error (px) over every corner of every view, and per view (OpenCV's two numbers)."""

    title = "reprojection error"
    if not views:
        return result(config, value=None, unit="px", metric=config.headline, limits=config.limits, title=title, message="no views", code="NoViews")
    names, per_view, total, count = [], [], 0.0, 0
    for index, view in enumerate(views):
        detected = np.asarray(view.detected_px, float).reshape(-1, 2)
        if view.projected_px is not None:
            projected = np.asarray(view.projected_px, float).reshape(-1, 2)
        elif view.object_points is not None and view.camera is not None:
            projected = project(view.object_points, view.camera, view.rvec, view.tvec)
        else:
            continue
        if projected.shape != detected.shape or detected.size == 0:
            continue
        squared = float(np.sum((detected - projected) ** 2))
        total, count = total + squared, count + detected.shape[0]
        names.append(view.view_id or f"view {index}")
        per_view.append(math.sqrt(squared / detected.shape[0]))
    if not count:
        return result(config, value=None, unit="px", metric=config.headline, limits=config.limits, title=title,
                      message="no view has both detected corners and a projection (or a camera model to make one)", code="NoProjection")
    overall = math.sqrt(total / count)
    stats = {"rms": overall, "worst_view": max(per_view)}
    other = "worst_view" if config.headline == "rms" else "rms"
    return result(config, value=stats[config.headline], unit="px", metric=config.headline, limits=config.limits, title=title,
                  message=f"{count} corners in {len(per_view)} views: RMS {overall:.3g} px, worst view {names[int(np.argmax(per_view))]} {max(per_view):.3g} px",
                  measurements=(Measure(other, stats[other], "px"), Measure("views", len(per_view)), Measure("corners", count)),
                  payload={"x": names, "y": per_view, "x_label": "view", "y_label": "RMS reprojection error (px)"},
                  why="a single bad view is a blurred or mis-detected board (drop it); all views high is the wrong distortion model")


# ----- depth_plane -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DepthFrames:
    depth: np.ndarray  # (F, H, W); 0 (or non-finite) = no depth
    depth_unit_mm: float = 1.0  # one depth count in mm (a RealSense depth_scale of 0.001 m -> 1.0)
    fx: float | None = None  # intrinsics: with them the plane is fitted to deprojected 3D points; without, to Z over pixels
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    ground_truth_mm: float | None = None  # the measured distance to the flat target (Z accuracy needs it)
    source_id: str = ""


@dataclass(frozen=True)
class DepthPlaneConfig:
    roi_fraction: float = 0.4  # the central window, this fraction of the frame's width and height (the depth-quality tool's 40 % ROI)
    headline: str = "plane_rms"  # plane_rms | fill_rate | z_accuracy | temporal_noise
    plane_rms: Limits = field(default_factory=Limits)  # % of Z
    fill_rate: Limits = field(default_factory=Limits)  # %
    z_accuracy: Limits = field(default_factory=Limits)  # % of Z, signed
    temporal_noise: Limits = field(default_factory=Limits)  # % of Z


DEPTH_UNITS = {"plane_rms": "% of Z", "fill_rate": "%", "z_accuracy": "% of Z", "temporal_noise": "% of Z"}


def _roi(height: int, width: int, fraction: float) -> tuple[slice, slice]:
    h, w = max(1, round(fraction * height)), max(1, round(fraction * width))
    y0, x0 = round((height - h) / 2), round((width - w) / 2)
    return slice(y0, y0 + h), slice(x0, x0 + w)


def _plane_rms(z: np.ndarray, u: np.ndarray, v: np.ndarray, data: DepthFrames) -> float:
    """RMS orthogonal distance to the least-squares plane (3D, via SVD) or, without intrinsics, RMS residual of Z = a u + b v + c."""

    if None not in (data.fx, data.fy, data.cx, data.cy):
        points = np.stack([(u - data.cx) * z / data.fx, (v - data.cy) * z / data.fy, z], axis=1)
        centred = points - points.mean(axis=0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        return float(np.sqrt(np.mean((centred @ normal) ** 2)))
    design = np.stack([u, v, np.ones_like(u)], axis=1)
    coef = np.linalg.lstsq(design, z, rcond=None)[0]
    return float(np.sqrt(np.mean((z - design @ coef) ** 2)))


def depth_plane(config: DepthPlaneConfig, data: DepthFrames):
    """Fill rate, plane-fit RMS, Z accuracy and temporal noise in the central ROI; the worst judged statistic decides."""

    depth = np.asarray(data.depth, float)
    unit = DEPTH_UNITS[config.headline]
    limits = getattr(config, config.headline)
    title = "depth quality (flat target)"
    if depth.ndim != 3 or depth.shape[0] == 0:
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title, message="no depth frames", code="NoFrames")
    rows, cols = _roi(depth.shape[1], depth.shape[2], config.roi_fraction)
    roi = depth[:, rows, cols] * data.depth_unit_mm
    valid = np.isfinite(roi) & (roi > 0)
    if not valid.any():
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title,
                      message="no valid depth in the ROI", code="NoValidDepth")
    v, u = np.mgrid[rows, cols]
    fill = 100.0 * float(valid.mean())
    plane = []
    for frame, ok in zip(roi, valid):
        if ok.sum() >= 3:
            z = frame[ok]
            plane.append(100.0 * _plane_rms(z, u[ok].astype(float), v[ok].astype(float), data) / float(np.mean(z)))
    median_z = float(np.median(roi[valid]))
    always = valid.all(axis=0)
    temporal = (100.0 * float(np.median(np.std(roi[:, always], axis=0, ddof=1))) / median_z) if roi.shape[0] >= 2 and always.any() else None
    stats = {"plane_rms": float(np.median(plane)) if plane else None, "fill_rate": fill, "temporal_noise": temporal,
             "z_accuracy": None if not data.ground_truth_mm else 100.0 * (median_z - data.ground_truth_mm) / data.ground_truth_mm}
    judged = {name: judge(stats[name], getattr(config, name)) for name in DEPTH_UNITS if stats[name] is not None and not getattr(config, name).empty}
    verdict = worst(v for v, _ in judged.values()) if judged else (PASS if stats[config.headline] is not None else NOT_RUN)
    culprit = next((name for name in (config.headline, *DEPTH_UNITS) if name in judged and judged[name][0] == verdict), None)
    code = "" if verdict == PASS else (f"{_camel(culprit)}{judged[culprit][1]}" if culprit else "NoValue")
    names = list(DEPTH_UNITS)
    return result(config, value=stats[config.headline], unit=unit, metric=config.headline, limits=limits, title=title, verdict=verdict, code=code,
                  message=f"{depth.shape[0]} frames, ROI {roi.shape[1]}x{roi.shape[2]}: fill {fill:.4g} %, plane RMS "
                          + (f"{stats['plane_rms']:.3g}" if stats["plane_rms"] is not None else "n/a") + " % of Z"
                          + (f", Z accuracy {stats['z_accuracy']:.3g} %" if stats["z_accuracy"] is not None else "")
                          + (f", temporal noise {temporal:.3g} %" if temporal is not None else ""),
                  measurements=tuple(Measure(name, stats[name], DEPTH_UNITS[name]) for name in names if name != config.headline)
                               + (Measure("median_z", median_z, "mm"), Measure("frames", int(depth.shape[0])),
                                  Measure("fit_3d", int(None not in (data.fx, data.fy, data.cx, data.cy)))),
                  payload={"x": names, "y": [stats[name] for name in names], "x_label": "metric", "y_label": "value (% / % of Z)"},
                  evidence=(Evidence(data.source_id, "depth"),) if data.source_id else (),
                  why="a high plane RMS is a bad calibration or a target too close; a low fill rate is lighting, texture or the laser off")


# ----- image_sharpness ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frames:
    images: Sequence[np.ndarray]  # grey frames (H, W)
    t: Sequence[float] | None = None
    source_id: str = ""


@dataclass(frozen=True)
class SharpnessConfig:
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x0, y0, x1, y1 as fractions of the frame
    statistic: str = "min"  # min: the blurriest frame | median
    limits: Limits = field(default_factory=Limits)  # site-calibrated on labelled frames (the value depends on the scene's texture)


def laplacian_variance(image: np.ndarray) -> float:
    """``cv2.Laplacian(image, cv2.CV_64F).var()`` (ksize 1, BORDER_REFLECT_101)."""

    padded = np.pad(np.asarray(image, float), 1, mode="reflect")
    lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
    return float(lap.var())


def image_sharpness(config: SharpnessConfig, data: Frames):
    """Variance of the Laplacian of each frame's ROI; the headline is the blurriest frame (or the median)."""

    images = [np.asarray(image, float) for image in data.images if np.asarray(image).ndim == 2 and min(np.asarray(image).shape) >= 2]
    title = "sharpness (variance of the Laplacian)"
    if not images:
        return result(config, value=None, unit="1", metric=config.statistic, limits=config.limits, title=title,
                      message="no grey frames", code="NoFrames")
    values = []
    for image in images:
        h, w = image.shape
        x0, y0, x1, y1 = config.roi
        crop = image[int(round(y0 * h)):max(int(round(y1 * h)), int(round(y0 * h)) + 2), int(round(x0 * w)):max(int(round(x1 * w)), int(round(x0 * w)) + 2)]
        values.append(laplacian_variance(crop))
    value = float(np.min(values)) if config.statistic == "min" else float(np.median(values))
    x = list(data.t) if data.t is not None and len(data.t) == len(values) else list(range(len(values)))
    return result(config, value=value, unit="1", metric=config.statistic, limits=config.limits, title=title,
                  message=f"{len(values)} frames: {config.statistic} {value:.4g} (median {float(np.median(values)):.4g})",
                  measurements=(Measure("frames", len(values)), Measure("max_frame", float(np.max(values)))),
                  payload={"x": x, "y": values, "x_label": "frame", "y_label": "variance of the Laplacian"},
                  evidence=(Evidence(data.source_id, "frames"),) if data.source_id else (),
                  why="blur is focus, motion (exposure too long for the belt speed) or a dirty lens; compare with a frame known sharp")


# ----- marker_drift ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarkerTrack:
    t: np.ndarray  # s, one per detection
    tag_id: Sequence[int]
    center_px: np.ndarray | None = None  # (N, 2) detected tag centre
    position_mm: np.ndarray | None = None  # (N, 3) detected tag position in the rack / camera frame
    calibrated_px: tuple[float, float] | None = None  # where the tag was at calibration
    calibrated_mm: tuple[float, float, float] | None = None
    source_id: str = ""


@dataclass(frozen=True)
class MarkerDriftConfig:
    tag_id: int = 0
    space: str = "px"  # px: the centre in the image (camera moved) | mm: the pose (extrinsics drifted)
    statistic: str = "median"  # median (robust to detector noise) | max | last
    limits: Limits = field(default_factory=Limits)


def marker_drift(config: MarkerDriftConfig, data: MarkerTrack):
    """|detected - calibrated| for every detection of the tag; the headline statistic over the window."""

    unit = config.space
    title = f"marker drift (tag {config.tag_id})"
    detected = data.center_px if config.space == "px" else data.position_mm
    calibrated = data.calibrated_px if config.space == "px" else data.calibrated_mm
    if calibrated is None:
        return result(config, value=None, unit=unit, metric=config.statistic, limits=config.limits, title=title,
                      message=f"no calibrated {config.space} value for the tag", code="NoCalibration")
    ids = np.asarray(data.tag_id)
    mine = ids == config.tag_id
    if detected is None or not mine.any():
        return result(config, value=None, unit=unit, metric=config.statistic, limits=config.limits, title=title,
                      message=f"tag {config.tag_id} was not detected in the window", code="NoDetections")
    points = np.asarray(detected, float)[mine]
    offsets = np.linalg.norm(points - np.asarray(calibrated, float), axis=1)
    t = np.asarray(data.t, float)[mine]
    value = {"median": float(np.median(offsets)), "max": float(np.max(offsets)), "last": float(offsets[np.argmax(t)])}[config.statistic]
    mean = points.mean(axis=0) - np.asarray(calibrated, float)
    pick = np.linspace(0, offsets.size - 1, min(400, offsets.size)).astype(int)
    return result(config, value=value, unit=unit, metric=config.statistic, limits=config.limits, title=title,
                  message=f"{offsets.size} detections: {config.statistic} offset {value:.3g} {unit} from calibration (mean shift "
                          + ", ".join(f"{c:.3g}" for c in mean) + f" {unit})",
                  measurements=(Measure("detections", int(offsets.size)), Measure("max_offset", float(offsets.max()), unit)),
                  payload={"x": t[pick].tolist(), "y": offsets[pick].tolist(), "x_label": "t (s)", "y_label": f"offset ({unit})"},
                  evidence=(Evidence(data.source_id, f"tag {config.tag_id}"),) if data.source_id else (),
                  why="a step is a bumped camera (re-run the extrinsic calibration); a slow creep is a loose mount or thermal drift")


__all__ = ["Camera", "CalibrationView", "project", "rodrigues", "ReprojectionConfig", "reprojection_error", "DepthFrames", "DepthPlaneConfig",
           "depth_plane", "Frames", "SharpnessConfig", "image_sharpness", "laplacian_variance", "MarkerTrack", "MarkerDriftConfig", "marker_drift"]
