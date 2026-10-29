"""Hyperspectral cameras: is the sensor good (SNR, smile, keystone), and does a measured spectrum match its material?

- ``spectral_quality``: SNR per band from repeated captures of a uniform target (mean signal over the temporal standard deviation,
  the HySIS / HISUI practice); smile = the peak-to-peak shift, across the slit, of a monochromatic line's centre (sub-pixel by the
  three-point Gaussian peak fit, exact for a Gaussian line); keystone = the peak-to-peak shift, across bands, of a sharp edge's position
  (where its profile crosses half way between its low and high levels, linearly interpolated).
- ``spectral_angle``: the spectral angle mapper (Kruse et al. 1993): alpha = arccos( <t, r> / (|t| |r|) ), blind to brightness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .._numpy import np

from ..core import PASS, Evidence, Limits, Measure, judge, result, worst


# ----- spectral_quality ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralTargets:
    uniform: np.ndarray | None = None  # (R repeats, P pixels, B bands) of a uniform target (white tile)
    line_frame: np.ndarray | None = None  # (C slit columns, S spectral pixels): one monochromatic line (a laser, a Hg-Ar lamp line)
    edge_frame: np.ndarray | None = None  # (B bands, X spatial pixels): a sharp edge across the slit
    wavelengths_nm: Sequence[float] | None = None  # per band, for the SNR plot's x axis
    source_id: str = ""


@dataclass(frozen=True)
class SpectralQualityConfig:
    headline: str = "snr"  # snr (the worst band's) | smile (px) | keystone (px)
    bands: tuple[int, ...] = ()  # SNR over these band indices only; empty = all
    snr: Limits = field(default_factory=Limits)
    smile: Limits = field(default_factory=Limits)
    keystone: Limits = field(default_factory=Limits)


QUALITY_UNITS = {"snr": "1", "smile": "px", "keystone": "px"}
MISSING = {"snr": "NoUniformTarget", "smile": "NoLineTarget", "keystone": "NoEdgeTarget"}


def gaussian_peak(row: np.ndarray) -> float | None:
    """Sub-pixel peak position: the three-point Gaussian fit (log-parabola) around the maximum, after removing the row's median."""

    row = np.asarray(row, float) - float(np.median(row))
    i = int(np.argmax(row))
    if i == 0 or i == row.size - 1 or min(row[i - 1], row[i], row[i + 1]) <= 0:
        return None
    a, b, c = math.log(row[i - 1]), math.log(row[i]), math.log(row[i + 1])
    return i + 0.5 * (a - c) / (a - 2 * b + c)


def half_crossing(profile: np.ndarray) -> float | None:
    """Sub-pixel position where an edge profile first crosses half way between its low and high levels (linear interpolation)."""

    profile = np.asarray(profile, float)
    low, high = float(np.min(profile)), float(np.max(profile))
    if high <= low:
        return None
    rising = profile[-1] >= profile[0]
    level = 0.5 * (low + high)
    above = (profile >= level) if rising else (profile <= level)
    i = int(np.argmax(above))
    if i == 0:
        return None
    y0, y1 = profile[i - 1], profile[i]
    return (i - 1) + (level - y0) / (y1 - y0)


def spectral_quality(config: SpectralQualityConfig, data: SpectralTargets):
    """SNR per band, smile and keystone; the headline is judged with any other limited statistic (worst decides)."""

    stats: dict[str, float | None] = {"snr": None, "smile": None, "keystone": None}
    snr_per_band, x = None, None
    if data.uniform is not None and np.asarray(data.uniform).ndim == 3 and np.asarray(data.uniform).shape[0] >= 2:
        cube = np.asarray(data.uniform, float)
        bands = list(config.bands) or list(range(cube.shape[2]))
        cube = cube[:, :, bands]
        noise = np.median(np.std(cube, axis=0, ddof=1), axis=0)  # temporal std per pixel, median over pixels
        snr_per_band = (cube.mean(axis=(0, 1)) / noise).tolist()
        stats["snr"] = float(np.min(snr_per_band))
        wavelengths = list(data.wavelengths_nm) if data.wavelengths_nm is not None else None
        x = [wavelengths[b] for b in bands] if wavelengths and len(wavelengths) > max(bands) else bands
    centres = None
    if data.line_frame is not None:
        centres = [gaussian_peak(row) for row in np.asarray(data.line_frame, float)]
        centres = [c for c in centres if c is not None]
        stats["smile"] = float(np.ptp(centres)) if len(centres) >= 2 else None
    edges = None
    if data.edge_frame is not None:
        edges = [half_crossing(row) for row in np.asarray(data.edge_frame, float)]
        edges = [e for e in edges if e is not None]
        stats["keystone"] = float(np.ptp(edges)) if len(edges) >= 2 else None
    unit = QUALITY_UNITS[config.headline]
    limits = getattr(config, config.headline)
    title = "spectral quality"
    if all(value is None for value in stats.values()):
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title,
                      message="no uniform-target repeats, line frame or edge frame", code="NoTargets")
    if stats[config.headline] is None:
        return result(config, value=None, unit=unit, metric=config.headline, limits=limits, title=title,
                      message=f"no target for {config.headline}", code=MISSING[config.headline])
    judged = {name: judge(stats[name], getattr(config, name)) for name in stats if stats[name] is not None and not getattr(config, name).empty}
    verdict = worst(v for v, _ in judged.values()) if judged else PASS
    culprit = next((name for name in (config.headline, *stats) if name in judged and judged[name][0] == verdict), None)
    code = "" if verdict == PASS else f"{culprit.capitalize()}{judged[culprit][1]}"
    if snr_per_band is not None:
        payload = {"x": x, "y": snr_per_band, "x_label": "band", "y_label": "SNR"}
    elif config.headline == "smile":
        payload = {"x": list(range(len(centres))), "y": centres, "x_label": "slit column", "y_label": "line centre (spectral px)"}
    else:
        payload = {"x": list(range(len(edges))), "y": edges, "x_label": "band", "y_label": "edge position (spatial px)"}
    return result(config, value=stats[config.headline], unit=unit, metric=config.headline, limits=limits, title=title, verdict=verdict, code=code,
                  message=", ".join(f"{name} {stats[name]:.4g} {QUALITY_UNITS[name]}" for name in stats if stats[name] is not None),
                  measurements=tuple(Measure(name, stats[name], QUALITY_UNITS[name]) for name in stats if name != config.headline),
                  payload=payload, evidence=(Evidence(data.source_id, "targets"),) if data.source_id else (),
                  why="low SNR in the end bands is the illumination's spectrum or the QE edge; smile/keystone are optics alignment")


# ----- spectral_angle ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpectralSamples:
    measured: Mapping[str, np.ndarray]  # material -> (n, B) pixel spectra (or one (B,) spectrum), reflectance
    references: Mapping[str, np.ndarray]  # material -> (B,) library spectrum
    source_id: str = ""


@dataclass(frozen=True)
class SpectralAngleConfig:
    materials: tuple[str, ...] = ()  # empty = every material with both a measurement and a reference
    statistic: str = "median"  # over a material's pixels: median | mean | max
    limits: Limits = field(default_factory=Limits)  # rad (the classifier's max_angle)


def spectral_angle(config: SpectralAngleConfig, data: SpectralSamples):
    """Per material, the angle (rad) between its measured pixels and its reference; the headline is the worst material."""

    names = [m for m in sorted(data.measured) if m in data.references and (not config.materials or m in config.materials)]
    angles, skipped = {}, []
    for name in names:
        pixels = np.atleast_2d(np.asarray(data.measured[name], float))
        reference = np.asarray(data.references[name], float).ravel()
        if pixels.shape[1] != reference.size or not np.linalg.norm(reference):
            skipped.append(name)
            continue
        norms = np.linalg.norm(pixels, axis=1) * np.linalg.norm(reference)
        ok = norms > 0
        if not ok.any():
            skipped.append(name)
            continue
        each = np.arccos(np.clip((pixels[ok] @ reference) / norms[ok], -1.0, 1.0))
        angles[name] = float({"median": np.median, "mean": np.mean, "max": np.max}[config.statistic](each))
    title = "spectral angle"
    if not angles:
        return result(config, value=None, unit="rad", metric="angle", limits=config.limits, title=title,
                      message="no material has a measured spectrum and a reference with the same bands"
                              + (f" (band counts differ: {', '.join(skipped)})" if skipped else ""), code="NoReference")
    worst_name = max(angles, key=angles.get)
    return result(config, value=angles[worst_name], unit="rad", metric="angle", limits=config.limits, title=title,
                  message=f"{len(angles)} materials: worst {worst_name} {angles[worst_name]:.4g} rad ({math.degrees(angles[worst_name]):.3g} deg)",
                  measurements=(Measure("materials", len(angles)), Measure("skipped", len(skipped))),
                  payload={"x": list(angles), "y": list(angles.values()), "x_label": "material", "y_label": "spectral angle (rad)"},
                  evidence=(Evidence(data.source_id, worst_name),) if data.source_id else (),
                  why="a large angle on one material is a wrong or stale reference (lighting changed); on all, the white reference")


__all__ = ["SpectralTargets", "SpectralQualityConfig", "spectral_quality", "gaussian_peak", "half_crossing", "SpectralSamples",
           "SpectralAngleConfig", "spectral_angle"]
