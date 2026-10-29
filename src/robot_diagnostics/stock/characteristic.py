"""A part's measured characteristic against its datasheet: the torque-angle loop of a drivetrain, and an inertial sensor's noise terms.

- ``hysteresis_loop``: torque vs angle with the output locked (Harmonic Drive's torsional stiffness K1..K3 and hysteresis loss; Nabtesco's
  lost motion), or the joint held still by friction (pre-sliding compliance). The loop is cut into legs (monotonic torque ramps, or the
  stretches where the target moves); each leg's slope dq/dtau is a compliance, and the two loading directions' lines, extrapolated to zero
  torque, are the loop's width there: the gap.
- ``allan_variance``: the overlapping Allan deviation of a rate signal (IEEE Std 952-1997, Annex C; IEEE 1554 for accelerometers) and the
  three noise terms read off it the way IEEE 952 and Kalibr's IMU noise model do: angle random walk N on the tau^-1/2 line at tau = 1 s,
  bias instability B = the curve's floor / 0.664 (sqrt(2 ln 2 / pi)), rate random walk K on the tau^+1/2 line at tau = 3 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Sequence

from .._numpy import np

from ..core import Evidence, Limits, Measure, result
from ..data import Trajectory

ARCMIN_PER_RAD = 60.0 * 180.0 / math.pi


# ----- hysteresis_loop -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HysteresisConfig:
    legs: str = "torque"  # torque: monotonic torque ramps (a clamp test) | target: the stretches where the target moves (pre-sliding)
    reversal_nm: float = 0.05  # torque legs: a reversal must exceed this to start a new leg (noise below it is not a reversal)
    smooth_rows: int = 1  # moving-average window applied to q and tau over the whole run before cutting legs (1 = none)
    trim_rows: int = 4  # rows dropped at each end of a leg (the turnaround)
    min_leg_rows: int = 20
    min_torque_span_nm: float = 0.0  # a leg must sweep at least this much torque
    max_leg_travel_rad: float | None = None  # pre-sliding: drop a leg whose angle leaves this box around its start (it slid)
    travel_share: float = 0.95  # ... on more than (1 - travel_share) of its rows
    skip_prelude: bool = True
    skip_s: float = 0.0  # target legs: also skip this long after the prelude
    stiffness: str = "pooled"  # pooled: one fit of q = c tau + an intercept per (torque side, direction) | leg_median: median leg slope
    band: tuple[float, float] = (0.0, 1.0)  # rows used for the pooled fit, as |tau| / max |tau|
    breaks: tuple[float, float, float, float] = (0.0, 0.25, 0.5, 1.0)  # |tau| / max |tau| bands for K1, K2, K3 (HD's T1, T2 as fractions)
    ci: float = 0.95  # the gap's confidence interval (from the fit's covariance; rows are serially correlated, so it is optimistic)
    headline: str = "gap"  # gap (arcmin) | gap_ci_half (arcmin) | stiffness (N.m/rad) | compliance (mrad/N.m)
    limits: Limits = field(default_factory=Limits)


HEADLINE_UNITS = {"gap": "arcmin", "gap_ci_half": "arcmin", "stiffness": "N.m/rad", "compliance": "mrad/N.m"}


def _smooth(values: np.ndarray, rows: int) -> np.ndarray:
    return values if rows <= 1 else np.convolve(values, np.ones(rows) / rows, mode="same")


def _zigzag(tau: np.ndarray, threshold: float) -> list[int]:
    """Indices of the alternating torque extremes, first and last row included: a turn counts once the torque retraces > threshold
    from the running extreme (the classic zig-zag filter), so noise below the threshold never cuts a leg."""

    points, direction, anchor = [0], 0, 0
    for i in range(1, tau.size):
        if direction == 0:
            if abs(tau[i] - tau[0]) > threshold:
                direction, anchor = (1 if tau[i] > tau[0] else -1), i
        elif direction > 0:
            if tau[i] > tau[anchor]:
                anchor = i
            elif tau[anchor] - tau[i] > threshold:
                points.append(anchor)
                direction, anchor = -1, i
        else:
            if tau[i] < tau[anchor]:
                anchor = i
            elif tau[i] - tau[anchor] > threshold:
                points.append(anchor)
                direction, anchor = 1, i
    if points[-1] != tau.size - 1:
        points.append(tau.size - 1)
    return points


def _legs(run: Trajectory, config: HysteresisConfig, q: np.ndarray, tau: np.ndarray) -> list[slice]:
    """[start, end) row slices of the run's legs (already trimmed)."""

    rows = run.rows(skip_prelude=config.skip_prelude)
    if config.legs == "target":
        if not run.has("qd_target_rad_s"):
            return []
        rows = rows & ((run.t - run.t[0]) >= run.prelude_s + config.skip_s)
        moving = (np.abs(run.channel("qd_target_rad_s")) > 1e-6) & rows
        edges = np.flatnonzero(np.diff(np.concatenate([[0], moving.astype(int), [0]])))
        bounds = list(zip(edges[::2], edges[1::2]))
    else:
        first = int(np.argmax(rows)) if rows.any() else tau.size
        points = [first + p for p in _zigzag(tau[first:], config.reversal_nm)] if first < tau.size else []
        bounds = [(a, b + 1) for a, b in zip(points, points[1:])]
    out = []
    for a, b in bounds:
        a, b = int(a) + config.trim_rows, int(b) - config.trim_rows
        if b - a < config.min_leg_rows:
            continue
        leg = slice(a, b)
        if not (np.all(np.isfinite(q[leg])) and np.all(np.isfinite(tau[leg]))):
            continue
        if np.ptp(tau[leg]) < max(config.min_torque_span_nm, config.reversal_nm if config.legs == "torque" else 0.0):
            continue
        if config.max_leg_travel_rad is not None and np.mean(np.abs(q[leg] - q[a]) < config.max_leg_travel_rad) < config.travel_share:
            continue
        out.append(leg)
    return out


def _fit(tau: np.ndarray, q: np.ndarray, groups: np.ndarray, labels: list) -> tuple[np.ndarray, np.ndarray] | None:
    """q = c tau + one intercept per group: (coefficients [c, I_1..I_k], covariance), or None if under-determined."""

    columns = [tau] + [(groups == label).astype(float) for label in labels]
    design = np.stack(columns, axis=1)
    if design.shape[0] <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coef, _, _, _ = np.linalg.lstsq(design, q, rcond=None)
    residual = q - design @ coef
    sigma2 = float(residual @ residual) / (design.shape[0] - design.shape[1])
    return coef, sigma2 * np.linalg.inv(design.T @ design)


def hysteresis_loop(config: HysteresisConfig, runs: Sequence[Trajectory]):
    """Gap at zero torque (the loop's width there), stiffness (pooled or leg median), K1..K3 per torque band."""

    title = "torque-angle loop"
    unit = HEADLINE_UNITS[config.headline]
    taus, qs, directions, slopes, evidence, first_loop = [], [], [], [], [], None
    seen = False
    for run in runs:
        if not (run.has("tau_nm") and run.has("q_rad")):
            continue
        seen = True
        q, tau = _smooth(run.channel("q_rad"), config.smooth_rows), _smooth(run.channel("tau_nm"), config.smooth_rows)
        legs = _legs(run, config, q, tau)
        for leg in legs:
            direction = 1.0 if tau[leg.stop - 1] > tau[leg.start] else -1.0
            taus.append(tau[leg])
            qs.append(q[leg])
            directions.append(np.full(leg.stop - leg.start, direction))
            slopes.append(float(np.polyfit(tau[leg], q[leg], 1)[0]))
        if legs:
            evidence.append(Evidence(run.run, "q_rad vs tau_nm"))
            if first_loop is None:
                rows = np.concatenate([np.arange(leg.start, leg.stop) for leg in legs])
                pick = rows[np.linspace(0, rows.size - 1, min(400, rows.size)).astype(int)]
                first_loop = (tau[pick], 1e3 * (q[pick] - np.median(q[pick])))
    if not seen:
        return result(config, value=None, unit=unit, metric=config.headline, limits=config.limits, title=title,
                      message="no run has both q_rad and tau_nm", code="NoChannel")
    if not taus:
        return result(config, value=None, unit=unit, metric=config.headline, limits=config.limits, title=title,
                      message="no leg: the torque never ramped (or every leg slid / was too short)", code="NoLegs")
    tau, q, direction = np.concatenate(taus), np.concatenate(qs), np.concatenate(directions)
    scale = float(np.max(np.abs(tau))) or 1.0
    side = np.where(tau >= 0, 1.0, -1.0)
    group = side * 10 + direction  # 11 (+ torque, loading up), 9 (+, unloading down), -9 (-, up), -11 (-, down)
    present = [label for label in (11.0, 9.0, -9.0, -11.0) if np.sum(group == label) >= 3]

    def fit_band(low: float, high: float):
        rows = (np.abs(tau) / scale >= low) & (np.abs(tau) / scale <= high)
        labels = [label for label in present if np.sum(rows & (group == label)) >= 3]
        fitted = _fit(tau[rows], q[rows], group[rows], labels) if labels else None
        return fitted, labels

    fitted, labels = fit_band(*config.band)
    gap = ci_half = None
    if fitted is not None:
        coef, cov = fitted
        index = {label: 1 + i for i, label in enumerate(labels)}
        # the loop's width at zero torque: the unloading line coming down from positive torque (+, down) against the loading line
        # coming up from negative torque (-, up); with one torque side only, that side's two directions
        pair = next(((a, b) for a, b in ((9.0, -9.0), (11.0, 9.0), (-9.0, -11.0)) if a in index and b in index), None)
        if pair is not None:
            i, j = index[pair[0]], index[pair[1]]
            gap = abs(float(coef[i] - coef[j]))
            se = math.sqrt(max(0.0, float(cov[i, i] + cov[j, j] - 2 * cov[i, j])))
            ci_half = NormalDist().inv_cdf(0.5 + config.ci / 2) * se
    if config.stiffness == "leg_median":
        compliance = float(np.median(slopes))
    else:
        compliance = float(fitted[0][0]) if fitted is not None else None
    stiffness = 1.0 / compliance if compliance else None
    bands = {}
    for k, (low, high) in enumerate(zip(config.breaks, config.breaks[1:]), start=1):
        band_fit, _ = fit_band(low, high)
        bands[f"K{k}"] = (1.0 / float(band_fit[0][0])) if band_fit is not None and band_fit[0][0] else None
    stats = {"gap": None if gap is None else gap * ARCMIN_PER_RAD, "gap_ci_half": None if ci_half is None else ci_half * ARCMIN_PER_RAD,
             "stiffness": stiffness, "compliance": None if compliance is None else 1e3 * compliance}
    common = dict(unit=unit, metric=config.headline, limits=config.limits, title=title, evidence=tuple(evidence),
                  measurements=tuple(Measure(name, stats[name], HEADLINE_UNITS[name]) for name in HEADLINE_UNITS if name != config.headline)
                               + tuple(Measure(name, value, "N.m/rad") for name, value in bands.items())
                               + (Measure("legs", len(taus)), Measure("torque_max", scale, "N.m")),
                  payload={"x": first_loop[0].tolist(), "y": first_loop[1].tolist(), "x_label": "tau (N.m)", "y_label": "q (mrad, centred)"})
    if stats[config.headline] is None:
        directions_seen = sorted({float(d) for d in direction})
        code = "NeedsBothDirections" if config.headline.startswith("gap") and len(directions_seen) < 2 else "NoFit"
        return result(config, value=None, message=f"{len(taus)} legs, but no {config.headline}: "
                      + ("the torque was ramped one way only" if code == "NeedsBothDirections" else "the fit is under-determined"), code=code, **common)
    gap_text = f"gap {stats['gap']:.3g} arcmin" + (f" +- {stats['gap_ci_half']:.2g}" if stats["gap_ci_half"] is not None else "") if stats["gap"] is not None else "gap n/a"
    stiff_text = f"stiffness {stiffness:.4g} N.m/rad ({stats['compliance']:.3g} mrad/N.m)" if stiffness else "stiffness n/a"
    return result(config, value=stats[config.headline], message=f"{len(taus)} legs: {gap_text}; {stiff_text}",
                  why="a wide gap is backlash or lost motion (preload, wear); a low stiffness is a compliant stage (belt, flex spline)", **common)


# ----- allan_variance ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RateSeries:
    t: np.ndarray  # s, uniform
    rate: np.ndarray  # a rate signal at rest (gyro rad/s, accelerometer m/s2)
    source_id: str = ""
    unit: str = "rad/s"


@dataclass(frozen=True)
class AllanConfig:
    taus_per_decade: int = 10  # log-spaced cluster times
    min_clusters: int = 9  # the longest tau is the record / this (fewer independent clusters are noise)
    min_samples: int = 100
    slope_window: int = 2  # local log-log slope from a fit over +-this many taus
    slope_tolerance: float = 0.1  # a tau is on the white-noise (or random-walk) line where its local slope is within this of -1/2 (+1/2)
    headline: str = "arw"  # arw (N) | bias_instability (B) | rrw (K)
    scale: float = 1.0  # e.g. 180/pi x 60 for N in deg/sqrt(h), 180/pi x 3600 for B in deg/h
    unit: str = ""  # "" = the SI unit of the headline term, from the series' unit
    limits: Limits = field(default_factory=Limits)


def overlapping_adev(rate: np.ndarray, dt: float, m: np.ndarray) -> np.ndarray:
    """Overlapping Allan deviation at cluster sizes ``m`` (IEEE 952 Annex C; the estimator of MathWorks' allanvar and Kalibr):
    theta = cumulative sum of rate x dt, sigma^2(tau) = sum_k (theta[k+2m] - 2 theta[k+m] + theta[k])^2 / (2 tau^2 (L - 2m))."""

    theta = np.concatenate([[0.0], np.cumsum(rate) * dt])
    length = theta.size
    out = np.empty(m.size)
    for i, size in enumerate(m):
        tau = size * dt
        d = theta[2 * size:] - 2 * theta[size:length - size] + theta[:length - 2 * size]
        out[i] = math.sqrt(float(d @ d) / (2 * tau * tau * (length - 2 * size)))
    return out


def allan_variance(config: AllanConfig, data: Sequence[RateSeries]):
    """Allan deviation of the first usable series; N, B and K beside it (the headline one judged)."""

    series = next((s for s in data if np.asarray(s.rate).size >= config.min_samples), None)
    si = {"arw": "{u}/sqrt(Hz)", "bias_instability": "{u}", "rrw": "{u}*sqrt(Hz)"}
    first_unit = data[0].unit if data else "rad/s"
    unit = config.unit or si[config.headline].format(u=first_unit)
    title = "Allan deviation"
    if series is None:
        return result(config, value=None, unit=unit, metric=config.headline, limits=config.limits, title=title,
                      message=f"no series with >= {config.min_samples} samples", code="TooShort")
    t, rate = np.asarray(series.t, float), np.asarray(series.rate, float)
    dt = float((t[-1] - t[0]) / (t.size - 1))
    rate = rate[np.isfinite(rate)]
    longest = max(1, rate.size // config.min_clusters)
    m = np.unique(np.round(np.logspace(0, math.log10(longest), max(2, int(config.taus_per_decade * math.log10(max(longest, 10))) + 1))).astype(int))
    adev = overlapping_adev(rate, dt, m)
    tau = m * dt
    log_tau, log_adev = np.log10(tau), np.log10(adev)
    slopes = np.array([np.polyfit(log_tau[max(0, i - config.slope_window):i + config.slope_window + 1],
                                  log_adev[max(0, i - config.slope_window):i + config.slope_window + 1], 1)[0] for i in range(tau.size)])
    white = np.abs(slopes + 0.5) <= config.slope_tolerance
    walk = np.abs(slopes - 0.5) <= config.slope_tolerance
    # N: the tau^-1/2 line (slope fixed) through the white-noise taus, read at tau = 1 s; K: the tau^+1/2 line read at tau = 3 s
    terms = {"arw": float(10 ** np.mean(log_adev[white] + 0.5 * log_tau[white])) if white.any() else None,
             "bias_instability": float(adev.min() / math.sqrt(2 * math.log(2) / math.pi)),
             "rrw": float(10 ** np.mean(log_adev[walk] - 0.5 * (log_tau[walk] - math.log10(3.0)))) if walk.any() else None}
    units = {name: pattern.format(u=series.unit) for name, pattern in si.items()}
    head = terms[config.headline]
    value = None if head is None else head * config.scale
    if value is None:
        return result(config, value=None, unit=unit, metric=config.headline, limits=config.limits, title=title,
                      message=f"no stretch of the curve has the {config.headline} slope", code="NoSlopeRegion",
                      payload={"x": tau.tolist(), "y": adev.tolist()})
    return result(config, value=value, unit=unit, metric=config.headline, limits=config.limits, title=title,
                  message=f"{rate.size} samples over {rate.size * dt:.4g} s: N {_fmt(terms['arw'])} {units['arw']}, "
                          f"B {_fmt(terms['bias_instability'])} {units['bias_instability']}, K {_fmt(terms['rrw'])} {units['rrw']}",
                  measurements=tuple(Measure(name, terms[name], units[name]) for name in terms if name != config.headline)
                               + (Measure("duration", rate.size * dt, "s"), Measure("tau_floor", float(tau[int(np.argmin(adev))]), "s")),
                  payload={"x": tau.tolist(), "y": adev.tolist(), "x_label": "tau (s)", "y_label": f"Allan deviation ({series.unit})",
                           "log_x": True, "log_y": True},
                  evidence=(Evidence(series.source_id, "rate"),) if series.source_id else (),
                  why="compare N, B and K with the datasheet; a K far above it is temperature drift (let the sensor warm up)")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


__all__ = ["HysteresisConfig", "hysteresis_loop", "RateSeries", "AllanConfig", "allan_variance", "overlapping_adev", "ARCMIN_PER_RAD"]
