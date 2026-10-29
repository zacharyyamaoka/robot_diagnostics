"""The specified primitives against the mature libraries whose definitions they cite (OpenCV, SciPy). Neither is a dependency: these
tests skip without them and run with, e.g.

    uv run --project /home/bam/io-diagnostics --offline --group dev --with opencv-python-headless --with scipy pytest /home/bam/io-diagnostics/tests/diagnostics/test_spec_external_oracles.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from robotdx.diagnostics import evaluate
from robotdx.diagnostics.stock import ReprojectionConfig, ShareConfig
from robotdx.diagnostics.stock.camera import CalibrationView, Camera, laplacian_variance, project
from robotdx.diagnostics.stock.events import Event


def test_laplacian_variance_is_opencvs():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(21)
    for shape in ((48, 64), (7, 5), (120, 90)):
        image = rng.uniform(0, 255, shape)
        assert laplacian_variance(image) == pytest.approx(float(cv2.Laplacian(image, cv2.CV_64F).var()), rel=1e-12)


def test_projection_is_opencvs_project_points():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(22)
    points = rng.uniform(-200, 200, (40, 3)) + [0.0, 0.0, 1500.0]
    camera = Camera(fx=612.0, fy=598.0, cx=331.0, cy=247.0, dist=(-0.21, 0.07, 0.001, -0.002, 0.01))
    rvec, tvec = np.array([0.1, -0.2, 0.3]), np.array([5.0, -12.0, 40.0])
    matrix = np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]])
    expected, _ = cv2.projectPoints(points, rvec, tvec, matrix, np.array(camera.dist))
    assert np.allclose(project(points, camera, rvec, tvec), expected.reshape(-1, 2), atol=1e-9)


def test_reprojection_rms_is_calibrate_cameras_returned_rms():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(23)
    xs, ys = np.meshgrid(np.arange(9) * 25.0, np.arange(6) * 25.0)
    board = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(np.float32)
    truth = Camera(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    matrix0 = np.array([[truth.fx, 0, truth.cx], [0, truth.fy, truth.cy], [0, 0, 1.0]])
    object_points, image_points = [], []
    for _ in range(8):
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = np.array([-100.0, -60.0, 900.0]) + rng.uniform(-50, 50, 3)
        seen = project(board, truth, rvec, tvec) + rng.normal(0, 0.3, (board.shape[0], 2))
        object_points.append(board)
        image_points.append(seen.astype(np.float32))
    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (640, 480), matrix0.copy(), None,
                                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    fitted = Camera(fx=matrix[0, 0], fy=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2], dist=tuple(dist.ravel()[:5]))
    views = [CalibrationView(image, object_points=board, camera=fitted, rvec=tuple(r.ravel()), tvec=tuple(t.ravel()), view_id=str(i))
             for i, (image, r, t) in enumerate(zip(image_points, rvecs, tvecs))]
    out = evaluate("stock.reprojection_error", ReprojectionConfig(), views, subject="s")
    assert out.value == pytest.approx(rms, rel=1e-4)  # float32 corners in OpenCV, float64 here


def test_wilson_interval_is_scipys():
    stats = pytest.importorskip("scipy.stats")
    for true, total in ((8, 10), (1, 200), (0, 7), (37, 41)):
        events = [Event(t=float(i), ok=i < true) for i in range(total)]
        out = evaluate("stock.share", ShareConfig(), events, subject="s")
        m = {x.key: x.value for x in out.measurements}
        ci = stats.binomtest(true, total).proportion_ci(confidence_level=0.95, method="wilson")
        assert (m["wilson_low"], m["wilson_high"]) == (pytest.approx(ci.low, abs=1e-12), pytest.approx(ci.high, abs=1e-12))
