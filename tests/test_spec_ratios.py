"""The catalog's specified ratio, event and time primitives against answers known by construction (written before the code).

feedback_share, share, agreement, stamp_delay, event_interval, staleness. Every oracle is computed here by hand or is a published
number (the Wilson 95 % interval of 8/10 is [0.4902, 0.9433]: Newcombe 1998, Table I); none calls the code under test.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from robotdx.diagnostics import FAIL, NOT_RUN, PASS, WARN, Limits, evaluate
from robotdx.diagnostics.core import Result
from robotdx.diagnostics.stock import (AgreementConfig, EventIntervalConfig, FeedbackShareConfig, ShareConfig, StalenessConfig,
                                       StampDelayConfig)
from robotdx.diagnostics.stock.events import Event, EventLog, EvidenceStamps
from robotdx.diagnostics.stock.health import StampPairs
from robotdx.diagnostics.stock.truthfulness import Pair

from conftest import tape


def measures(result) -> dict:
    return {m.key: m.value for m in result.measurements}


# ----- feedback_share: RMS(tau - tau_ff) / RMS(tau) (bam_rack_twin vt_kpi.feedback_share) ----------------------------------------
def _torque_run(amplitude: float, ff_fraction: float, run: str, periods: int = 4, per_period: int = 50):
    """tau = A sin over whole periods (mean of sin^2 exactly 1/2) and a feed-forward that carries ``ff_fraction`` of it."""

    n = periods * per_period
    base = tape(n=n, run=run)
    tau = amplitude * np.sin(2 * np.pi * np.arange(n) / per_period)
    return replace(base, tau_nm=tau, tau_ff_nm=ff_fraction * tau)


def test_feedback_share_of_a_feedforward_carrying_a_fixed_fraction_is_the_rest():
    out = evaluate("stock.feedback_share", FeedbackShareConfig(), [_torque_run(1.0, 0.6, "a")], subject="s")
    assert out.value == pytest.approx(0.4, abs=1e-12) and out.verdict == PASS and out.better == "lower"
    # bam_rack_twin's own test case: feedback 0.1 sin 7t on a 0.5 sign(cos t) feed-forward
    t = np.linspace(0.0, 2.0, 400)
    feedback, feedforward = 0.1 * np.sin(7 * t), 0.5 * np.sign(np.cos(t))
    run = replace(tape(n=t.size), t=t, tau_nm=feedback + feedforward, tau_ff_nm=feedforward)
    expected = np.sqrt(np.mean(feedback ** 2)) / np.sqrt(np.mean((feedback + feedforward) ** 2))
    assert evaluate("stock.feedback_share", FeedbackShareConfig(), [run], subject="s").value == pytest.approx(expected, rel=1e-12)


def test_feedback_share_pools_rows_or_centres_per_run_values():
    runs = [_torque_run(1.0, 0.5, "a"), _torque_run(3.0, 0.9, "b")]
    pooled = evaluate("stock.feedback_share", FeedbackShareConfig(), runs, subject="s")
    # by hand: feedback mean squares 0.125 and 0.045, torque mean squares 0.5 and 4.5, equal row counts
    assert pooled.value == pytest.approx(math.sqrt((0.125 + 0.045) / 2) / math.sqrt((0.5 + 4.5) / 2), rel=1e-12)
    per_run = evaluate("stock.feedback_share", FeedbackShareConfig(combine="per_run"), runs, subject="s")
    assert per_run.value == pytest.approx(0.3, rel=1e-12)  # median of 0.5 and 0.1
    effort = evaluate("stock.feedback_share", FeedbackShareConfig(normalize=False), runs[:1], subject="s")
    assert effort.value == pytest.approx(0.5 / math.sqrt(2), rel=1e-12) and effort.unit == "N.m"


def test_feedback_share_uses_only_controlled_rows():
    run = _torque_run(1.0, 0.6, "a")
    mode = run.mode.copy()
    mode[::2] = 0.0  # every other row not position-controlled: its feed-forward is wrong on purpose
    wrong_ff = run.tau_ff_nm.copy()
    wrong_ff[::2] = 0.0
    out = evaluate("stock.feedback_share", FeedbackShareConfig(), [replace(run, mode=mode, tau_ff_nm=wrong_ff)], subject="s")
    assert out.value == pytest.approx(0.4, abs=1e-12)


def test_feedback_share_is_not_run_without_a_feedforward_or_a_torque():
    missing = evaluate("stock.feedback_share", FeedbackShareConfig(), [tape()], subject="s")
    assert (missing.verdict, missing.code, missing.value) == (NOT_RUN, "NoFeedForward", None)
    still = evaluate("stock.feedback_share", FeedbackShareConfig(), [replace(tape(), tau_ff_nm=np.zeros(600))], subject="s")
    assert (still.verdict, still.code) == (NOT_RUN, "NoTorque")


# ----- share: Wilson score interval (Wilson 1927; Newcombe 1998) --------------------------------------------------------------
def _events(ok: int, total: int, **labels) -> list:
    return [Event(t=float(i), ok=i < ok, labels=labels) for i in range(total)]


def test_share_of_8_in_10_carries_the_published_wilson_interval():
    out = evaluate("stock.share", ShareConfig(limits=Limits.at_least(0.8)), _events(8, 10), subject="s")
    assert out.value == pytest.approx(0.8) and out.verdict == PASS  # on the bound is inside
    m = measures(out)
    assert m["wilson_low"] == pytest.approx(0.4902, abs=5e-5) and m["wilson_high"] == pytest.approx(0.9433, abs=5e-5)
    assert m["events"] == 10 and out.better == "higher"
    bound = evaluate("stock.share", ShareConfig(judge_on="bound", limits=Limits.at_least(0.8)), _events(8, 10), subject="s")
    assert (bound.verdict, bound.code, bound.value) == (FAIL, "BelowErrorLimit", pytest.approx(0.8))  # judged on 0.49, not 0.8


def test_share_with_higher_better_never_fails_a_perfect_record():
    # the old PercentTask bug: success_rate had only an upper bound, so 100 % success read ERROR
    out = evaluate("stock.share", ShareConfig(better="higher", limits=Limits.at_least(0.8, warn=0.9)), _events(10, 10), subject="s")
    assert out.verdict == PASS and out.value == 1.0 and measures(out)["wilson_high"] == pytest.approx(1.0)


def test_share_lower_better_in_percent_slices_and_counts():
    loss = [Event(t=float(i), ok=(i == 7), labels={"bus": "a" if i < 100 else "b"}) for i in range(200)]
    out = evaluate("stock.share", ShareConfig(better="lower", scale=100.0, unit="%", slice_by="bus", limits=Limits.at_most(1.0)), loss, subject="s")
    assert out.value == pytest.approx(0.5) and out.unit == "%" and out.verdict == PASS and out.better == "lower"
    assert out.payload["slices"] == {"a": {"events": 100, "true": 1, "share": pytest.approx(1.0)}, "b": {"events": 100, "true": 0, "share": 0.0}}
    bound = evaluate("stock.share", ShareConfig(better="lower", judge_on="bound", scale=100.0, limits=Limits.at_most(1.0)), loss, subject="s")
    assert bound.verdict == FAIL  # Wilson upper bound of 1/200 is 2.8 %, above 1 %
    count = evaluate("stock.share", ShareConfig(statistic="count"), loss, subject="s")
    assert count.value == 1 and count.unit == "count"


def test_share_is_not_run_on_too_few_events():
    out = evaluate("stock.share", ShareConfig(min_events=5), _events(2, 3), subject="s")
    assert (out.verdict, out.code, out.value) == (NOT_RUN, "TooFewEvents", None)
    assert evaluate("stock.share", ShareConfig(), [], subject="s").verdict == NOT_RUN


# ----- agreement: a reported scalar against an independent reference -----------------------------------------------------------
def test_agreement_percent_ratio_and_worst_pair():
    gravity = evaluate("stock.agreement", AgreementConfig(limits=Limits(error_low=-1.0, error_high=1.0)), [Pair(9.70, 9.80665, "g")], subject="s")
    assert gravity.value == pytest.approx(100 * (9.70 - 9.80665) / 9.80665) and gravity.unit == "%"
    assert (gravity.verdict, gravity.code) == (FAIL, "BelowErrorLimit")
    ratio = evaluate("stock.agreement", AgreementConfig(mode="ratio"), [Pair(2.0, 1.6)], subject="s")
    assert ratio.value == pytest.approx(1.25) and ratio.unit == "1"
    worst = evaluate("stock.agreement", AgreementConfig(), [Pair(1.02, 1.0), Pair(0.9, 1.0), Pair(1.05, 1.0)], subject="s")
    assert worst.value == pytest.approx(-10.0) and measures(worst)["pairs"] == 3
    median = evaluate("stock.agreement", AgreementConfig(combine="median"), [Pair(1.02, 1.0), Pair(0.9, 1.0), Pair(1.05, 1.0)], subject="s")
    assert median.value == pytest.approx(2.0)


def test_agreement_counts_config_mismatches():
    stored = [Pair(200.0, 200.0, "kp"), Pair(16000.0, 0.0, "ki"), Pair("board", "board", "gains"), Pair(0.8001, 0.8, "ilimit")]
    out = evaluate("stock.agreement", AgreementConfig(mode="mismatches", tolerance=1e-3, limits=Limits.at_most(0)), stored, subject="s")
    assert out.value == 1 and (out.verdict, out.code) == (FAIL, "AboveErrorLimit")
    assert out.payload["mismatched"] == ["ki"]


def test_agreement_is_not_run_without_a_reference():
    out = evaluate("stock.agreement", AgreementConfig(), [Pair(1.0, None), Pair(None, 2.0)], subject="s")
    assert (out.verdict, out.code, out.value) == (NOT_RUN, "NoReference", None)
    zero = evaluate("stock.agreement", AgreementConfig(), [Pair(1.0, 0.0)], subject="s")  # a percent of zero is undefined
    assert zero.verdict == NOT_RUN


# ----- stamp_delay: ROS diagnostic_updater TimeStampStatus (min_acceptable -1 s, max_acceptable 5 s) --------------------------
def test_stamp_delay_reads_the_delay_and_flags_late_early_and_zero_stamps():
    source = np.arange(0.0, 10.0, 0.1) + 1e9
    good = evaluate("stock.stamp_delay", StampDelayConfig(), [StampPairs(source, source + 0.02, "cam")], subject="s")
    assert good.value == pytest.approx(0.02) and good.verdict == PASS and good.unit == "s"
    late = source + 0.02
    late[40] += 6.0
    out = evaluate("stock.stamp_delay", StampDelayConfig(), [StampPairs(source, late, "cam")], subject="s")
    assert (out.verdict, out.code, out.value) == (FAIL, "LateStamps", pytest.approx(6.02)) and measures(out)["late"] == 1
    median = evaluate("stock.stamp_delay", StampDelayConfig(statistic="median"), [StampPairs(source, late, "cam")], subject="s")
    assert median.value == pytest.approx(0.02) and median.verdict == FAIL  # any late message fails, as ROS does
    early = source + 0.02
    early[3] -= 2.5
    assert evaluate("stock.stamp_delay", StampDelayConfig(), [StampPairs(source, early, "cam")], subject="s").code == "EarlyStamps"
    zero = source.copy()
    zero[0] = 0.0
    out = evaluate("stock.stamp_delay", StampDelayConfig(), [StampPairs(zero, source + 0.02, "cam")], subject="s")
    assert (out.verdict, out.code) == (FAIL, "ZeroStamps") and measures(out)["zero"] == 1


def test_stamp_delay_is_not_run_without_stamps():
    out = evaluate("stock.stamp_delay", StampDelayConfig(), [StampPairs(np.zeros(0), np.zeros(0), "cam")], subject="s")
    assert (out.verdict, out.code) == (NOT_RUN, "NoStamps")


# ----- event_interval: MTBF = operating time / failures (IEC 60050-192) --------------------------------------------------------
def test_event_interval_is_operating_hours_over_failures():
    events = [Event(t=t, ok=False, code="fault") for t in (100.0, 9000.0, 52000.0, 80000.0)] + [Event(t=40000.0, ok=False, code="fault")]
    log = EventLog(events, operating=((0.0, 36000.0), (50000.0, 86000.0)))  # 20 h on; the 40000 s event is while off
    out = evaluate("stock.event_interval", EventIntervalConfig(limits=Limits.at_least(2.0)), log, subject="s")
    assert out.value == pytest.approx(5.0) and out.unit == "h" and out.verdict == PASS
    assert measures(out)["events"] == 4 and measures(out)["outside"] == 1 and measures(out)["operating"] == pytest.approx(20.0)
    only = EventLog(events + [Event(t=200.0, ok=False, code="intervention")], operating=((0.0, 36000.0), (50000.0, 86000.0)))
    picked = evaluate("stock.event_interval", EventIntervalConfig(codes=("intervention",), scale=1 / 60, unit="min"), only, subject="s")
    assert picked.value == pytest.approx(20 * 60.0)


def test_event_interval_with_no_failures_is_a_lower_bound_that_can_pass_but_never_guesses():
    log = EventLog([], operating=((0.0, 72000.0),))
    passes = evaluate("stock.event_interval", EventIntervalConfig(limits=Limits.at_least(10.0)), log, subject="s")
    assert passes.verdict == PASS and passes.value == pytest.approx(20.0) and measures(passes)["censored"] == 1
    unknown = evaluate("stock.event_interval", EventIntervalConfig(limits=Limits.at_least(50.0)), log, subject="s")
    assert (unknown.verdict, unknown.code) == (NOT_RUN, "TooFewEvents")  # MTBF >= 20 h says nothing about >= 50 h
    off = evaluate("stock.event_interval", EventIntervalConfig(), EventLog([Event(t=1.0, ok=False)], operating=()), subject="s")
    assert (off.verdict, off.code) == (NOT_RUN, "NoOperatingTime")


# ----- staleness: age of each gate's newest evidence vs its validity window -----------------------------------------------------
NOW = 1.0e9
HOUR, DAY = 3600.0, 86400.0


def test_staleness_fails_the_gate_past_its_window_and_names_it():
    stamps = EvidenceStamps(NOW, (("comms.reply_loss", NOW - HOUR), ("comms.reply_loss", NOW - 5 * HOUR), ("motion.T4", NOW - 8 * DAY)))
    out = evaluate("stock.staleness", StalenessConfig(), stamps, subject="rack")
    assert (out.verdict, out.code) == (FAIL, "Stale") and out.value == pytest.approx(192.0) and out.unit == "h"
    assert out.payload["x"] == ["comms.reply_loss", "motion.T4"] and out.payload["y"] == [pytest.approx(1.0), pytest.approx(192.0)]
    fresh = EvidenceStamps(NOW, (("comms.reply_loss", NOW - HOUR), ("motion.T4", NOW - 2 * DAY)))
    assert evaluate("stock.staleness", StalenessConfig(), fresh, subject="rack").verdict == PASS
    near = EvidenceStamps(NOW, (("comms.reply_loss", NOW - 20 * HOUR),))
    assert evaluate("stock.staleness", StalenessConfig(), near, subject="rack").code == "NearlyStale"


def test_staleness_never_passes_a_gate_without_evidence():
    stamps = EvidenceStamps(NOW, (("comms.reply_loss", NOW - HOUR),))
    out = evaluate("stock.staleness", StalenessConfig(expected=("comms.reply_loss", "motion.T4")), stamps, subject="rack")
    assert (out.verdict, out.code) == (NOT_RUN, "NoEvidence")
    reset = EvidenceStamps(NOW, (("comms.reply_loss", NOW - HOUR),), reset_at=NOW - 60.0)  # a replug after the evidence
    assert evaluate("stock.staleness", StalenessConfig(), reset, subject="rack").verdict == NOT_RUN
    assert evaluate("stock.staleness", StalenessConfig(), EvidenceStamps(NOW, ()), subject="rack").code == "NoEvidence"


def test_staleness_reads_results_by_their_stamp():
    old = Result(diagnostic="stock.box", title="t", verdict=PASS, value=1.0, unit="1", stamp=NOW - 2 * DAY)
    out = evaluate("stock.staleness", StalenessConfig(valid_for=(("stock.*", DAY),)), EvidenceStamps(NOW, (old,)), subject="rack")
    assert (out.verdict, out.value) == (FAIL, pytest.approx(48.0))
