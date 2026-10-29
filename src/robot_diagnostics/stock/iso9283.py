"""ISO 9283:1998 (Manipulating industrial robots: performance criteria and related test methods), pose and path characteristics.

Pose (§7.2.1, §7.2.2), per commanded pose j over n cycles, with x_i the attained positions and x_c the commanded one:
- AP = | x_bar - x_c |, the distance from the barycentre of the attained positions to the command (positional accuracy);
- RP = l_bar + 3 S_l, with l_i = | x_i - x_bar |, l_bar their mean and S_l = sqrt( sum (l_i - l_bar)^2 / (n - 1) ) (repeatability);
- orientation: AP_a = a_bar - a_c and RP_a = +-3 S_a per angle (a, b, c), S_a the sample standard deviation.
Path (§8.2, §8.3), per commanded path point i over n cycles: AT_P = max_i | x_bar_i - x_ci |, RT_P = max_i (l_bar_i + 3 S_li).
WHY the path points are index-aligned: ISO 9283 intersects each cycle with planes normal to the commanded path; a trajectory replayed on
one clock (the target situation riding on each run's own rows) gives the same commanded point at the same index, which is that
intersection when the cycles track the path. Cycles of different lengths are refused, never resampled.
ISO's test asks for 30 cycles at five poses; fewer cycles warn (``iso_cycles``) when everything else passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .._numpy import np

from ..core import PASS, WARN, Evidence, Limits, Measure, judge, result, worst


# ----- pose accuracy --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PoseSample:
    pose: str  # the commanded pose's name (P1 .. P5)
    attained_mm: tuple[float, float, float]
    commanded_mm: tuple[float, float, float]
    cycle: int = 0
    attained_abc_deg: tuple[float, float, float] | None = None  # orientation as the robot's a, b, c angles (optional)
    commanded_abc_deg: tuple[float, float, float] | None = None
    evidence: str = ""


@dataclass(frozen=True)
class PoseAccuracyConfig:
    headline: str = "ap"  # ap (accuracy) | rp (repeatability): the worst pose's
    min_cycles: int = 3  # fewer attained positions at a pose = that pose is not run
    iso_cycles: int = 30  # ISO 9283's 30 cycles; fewer warn when all else passes (0 = do not ask)
    ap: Limits = field(default_factory=Limits)  # mm
    rp: Limits = field(default_factory=Limits)  # mm


def _pose_terms(samples: list[PoseSample]) -> dict:
    attained = np.array([s.attained_mm for s in samples], float)
    commanded = np.mean(np.array([s.commanded_mm for s in samples], float), axis=0)
    barycentre = attained.mean(axis=0)
    distances = np.linalg.norm(attained - barycentre, axis=1)
    terms = {"ap": float(np.linalg.norm(barycentre - commanded)),
             "rp": float(distances.mean() + 3 * distances.std(ddof=1)), "n": len(samples)}
    if all(s.attained_abc_deg is not None and s.commanded_abc_deg is not None for s in samples):
        angles = np.array([s.attained_abc_deg for s in samples], float)
        angle_commanded = np.mean(np.array([s.commanded_abc_deg for s in samples], float), axis=0)
        for k, name in enumerate("abc"):
            terms[f"ap_{name}"] = float(angles[:, k].mean() - angle_commanded[k])
            terms[f"rp_{name}"] = float(3 * angles[:, k].std(ddof=1))
    return terms


def pose_accuracy(config: PoseAccuracyConfig, samples: Sequence[PoseSample]):
    """ISO 9283 AP and RP per pose; the headline is the worst pose's; both are judged (worst of the two decides)."""

    by_pose: dict[str, list[PoseSample]] = {}
    for sample in samples:
        by_pose.setdefault(sample.pose, []).append(sample)
    poses = {name: _pose_terms(items) for name, items in by_pose.items() if len(items) >= max(2, config.min_cycles)}
    title = f"pose {'accuracy' if config.headline == 'ap' else 'repeatability'} (ISO 9283)"
    limits = getattr(config, config.headline)
    if not poses:
        return result(config, value=None, unit="mm", metric=config.headline, limits=limits, title=title,
                      message=f"no pose has {max(2, config.min_cycles)} cycles", code="TooFewCycles")
    stats = {name: max(terms[name] for terms in poses.values()) for name in ("ap", "rp")}
    judged = {name: judge(stats[name], getattr(config, name)) for name in ("ap", "rp")}
    verdict = worst(v for v, _ in judged.values())
    culprit = next((name for name in (config.headline, "ap", "rp") if judged[name][0] == verdict), config.headline)
    code = "" if verdict == PASS else f"{culprit.capitalize()}{judged[culprit][1]}"
    cycles = min(terms["n"] for terms in poses.values())
    if verdict == PASS and config.iso_cycles and cycles < config.iso_cycles:
        verdict, code = WARN, "FewerCyclesThanIso9283"
    other = "rp" if config.headline == "ap" else "ap"
    angle_measures = []
    for key in ("ap_a", "ap_b", "ap_c", "rp_a", "rp_b", "rp_c"):
        values = [terms[key] for terms in poses.values() if key in terms]
        if values:
            angle_measures.append(Measure(f"{key}_deg", max(values, key=abs), "deg"))
    names = sorted(poses)
    worst_pose = max(names, key=lambda name: poses[name][config.headline])
    return result(config, value=stats[config.headline], unit="mm", metric=config.headline, limits=limits, title=title, verdict=verdict, code=code,
                  message=f"{len(poses)} poses x >= {cycles} cycles: worst AP {stats['ap']:.3g} mm, worst RP {stats['rp']:.3g} mm ({worst_pose} worst "
                          f"{config.headline.upper()})" + (f"; ISO 9283 asks for {config.iso_cycles} cycles" if code == "FewerCyclesThanIso9283" else ""),
                  measurements=(Measure(other, stats[other], "mm"), Measure("poses", len(poses)), Measure("cycles", cycles), *angle_measures),
                  payload={"x": names, "y": [poses[name][config.headline] for name in names], other: [poses[name][other] for name in names],
                           "x_label": "pose", "y_label": f"{config.headline.upper()} (mm)"},
                  evidence=tuple(Evidence(s.evidence, s.pose) for s in samples if s.evidence)[:50],
                  why="AP is calibration (kinematic parameters, tool offset); RP is the drive (backlash, stop-band friction, encoder resolution)")


# ----- path accuracy --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathCycle:
    xyz_mm: np.ndarray  # (N, 3) attained, in ``frame``
    target_xyz_mm: np.ndarray  # (N, 3) commanded, the same N points
    frame: str = "base"
    run: str = ""
    quat_wxyz: np.ndarray | None = None  # (N, 4) attained orientation (optional)
    target_quat_wxyz: np.ndarray | None = None


@dataclass(frozen=True)
class PathAccuracyConfig:
    frame: str = "base"  # every cycle must be written in this frame (base, belt): a cycle in another is refused
    headline: str = "at"  # at (AT_P) | rt (RT_P) | rms | max (per-sample |attained - commanded| over every cycle)
    tolerance_mm: float | None = None  # shorthand: when limits is empty, fail above this
    limits: Limits = field(default_factory=Limits)


def _rotation_mrad(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    return 1e3 * 2 * np.arccos(np.clip(np.abs(np.sum(q * r, axis=1)), 0.0, 1.0))


def path_accuracy(config: PathAccuracyConfig, cycles: Sequence[PathCycle]):
    """ISO 9283 AT_P and RT_P over index-aligned cycles, the per-sample error beside them."""

    limits = config.limits if not config.limits.empty or config.tolerance_mm is None else Limits.at_most(config.tolerance_mm)
    title = "path accuracy (ISO 9283)"
    kept = [c for c in cycles if c.frame == config.frame]
    if not kept:
        return result(config, value=None, unit="mm", metric=config.headline, limits=limits, title=title,
                      message=f"no cycle in frame {config.frame!r}" + (f" (got {sorted({c.frame for c in cycles})})" if cycles else ""),
                      code="FrameMismatch" if cycles else "NoCycles")
    attained = [np.asarray(c.xyz_mm, float) for c in kept]
    commanded = [np.asarray(c.target_xyz_mm, float) for c in kept]
    if len({a.shape for a in attained} | {c.shape for c in commanded}) != 1 or attained[0].ndim != 2 or attained[0].shape[1] != 3:
        return result(config, value=None, unit="mm", metric=config.headline, limits=limits, title=title,
                      message="cycles are not sampled at the same commanded points (shapes differ): ISO 9283 compares point by point",
                      code="MisalignedCycles")
    stack = np.stack(attained)  # (n cycles, N points, 3)
    command = np.mean(np.stack(commanded), axis=0)
    barycentre = stack.mean(axis=0)
    accuracy = np.linalg.norm(barycentre - command, axis=1)
    errors = np.linalg.norm(stack - command[None], axis=2)
    stats = {"at": float(accuracy.max()), "rms": float(np.sqrt(np.mean(errors ** 2))), "max": float(errors.max()), "rt": None}
    if len(kept) >= 2:
        spread = np.linalg.norm(stack - barycentre[None], axis=2)  # l_ij
        stats["rt"] = float(np.max(spread.mean(axis=0) + 3 * spread.std(axis=0, ddof=1)))
    rotation = None
    if all(c.quat_wxyz is not None and c.target_quat_wxyz is not None for c in kept):
        rotation = float(max(_rotation_mrad(np.asarray(c.quat_wxyz, float), np.asarray(c.target_quat_wxyz, float)).max() for c in kept))
    value = stats[config.headline]
    if value is None:
        return result(config, value=None, unit="mm", metric=config.headline, limits=limits, title=title,
                      message="RT needs at least 2 cycles", code="TooFewCycles")
    pick = np.linspace(0, accuracy.size - 1, min(400, accuracy.size)).astype(int)
    worst_point = int(np.argmax(accuracy))
    return result(config, value=value, unit="mm", metric=config.headline, limits=limits, title=title,
                  message=f"{len(kept)} cycles x {accuracy.size} points in {config.frame}: AT {stats['at']:.3g} mm"
                          + (f", RT {stats['rt']:.3g} mm" if stats["rt"] is not None else "") + f", RMS {stats['rms']:.3g} mm",
                  measurements=tuple(Measure(name, stats[name], "mm") for name in ("at", "rt", "rms", "max") if name != config.headline)
                               + (Measure("rot_max_mrad", rotation, "mrad"), Measure("cycles", len(kept)), Measure("points", int(accuracy.size))),
                  payload={"x": pick.tolist(), "y": accuracy[pick].tolist(), "x_label": "path point", "y_label": "|barycentre - commanded| (mm)",
                           "frame": config.frame, "worst_point": worst_point},
                  evidence=tuple(Evidence(c.run, "xyz_mm") for c in kept if c.run),
                  why="a path error that grows with speed is lag (feed-forward, bandwidth); one fixed in space is calibration")


__all__ = ["PoseSample", "PoseAccuracyConfig", "pose_accuracy", "PathCycle", "PathAccuracyConfig", "path_accuracy"]
