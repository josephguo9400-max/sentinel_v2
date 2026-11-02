"""Streaming guard (serving/streaming.py): hold-back buffer, check scheduling, flag handling, failure modes.

Every test runs on the virtual-time loop (serving/bench/vclock.py) unless it says otherwise, so latencies are exact
and no test sleeps in wall time. Fakes: a token source at a fixed rate and a classifier that flags any text
containing the marker token, after a configurable latency.
"""
from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

import pytest

from serving.bench.vclock import run_virtual
from serving.streaming import (CheckResult, Done, Release, Stop, StreamGuardConfig, collect, guard_stream,
                               service_check)

MARK = "<INJ>"


def make_tokens(n: int, inject_at: int | None = None, inject_len: int = 5) -> list[str]:
    out = [f"t{i} " for i in range(n)]
    if inject_at is not None:
        for i in range(inject_at, min(n, inject_at + inject_len)):
            out[i] = MARK
    return out


class Source:
    """Async token source at `rate` tokens/s; records whether it was closed early."""

    def __init__(self, toks, rate=50.0):
        self.toks, self.dt, self.closed, self.served = list(toks), 1.0 / rate, False, 0

    async def gen(self):
        try:
            for t in self.toks:
                await asyncio.sleep(self.dt)
                self.served += 1
                yield t
        finally:
            self.closed = self.served < len(self.toks)


class FakeClassifier:
    def __init__(self, latency=0.005, fail=None, failed_flag=False):
        self.latency, self.fail, self.failed_flag, self.calls = latency, fail, failed_flag, []

    async def __call__(self, text: str) -> CheckResult:
        self.calls.append(text)
        lat = self.latency(text) if callable(self.latency) else self.latency
        await asyncio.sleep(lat)
        if self.fail is not None and self.fail(len(self.calls)):
            raise RuntimeError("classifier crashed")
        if MARK in text:
            return CheckResult("flag", failed=self.failed_flag, score=0.99)
        return CheckResult("allow", score=0.01)


def run(toks, cfg, clf=None, rate=50.0):
    src = Source(toks, rate)
    clf = clf or FakeClassifier()
    events, st = run_virtual(collect(src.gen(), clf, cfg))
    return events, st, src, clf


def released_text(events):
    return "".join(e.text for e in events if isinstance(e, Release))


def assert_well_formed(events, st):
    assert isinstance(events[-1], (Stop, Done))
    assert sum(isinstance(e, (Stop, Done)) for e in events) == 1
    pos = 0
    for e in events[:-1]:
        assert isinstance(e, Release) and e.start == pos and e.end > e.start
        pos = e.end
    assert pos == st.released


# ------------------------------------------------------------------ invariant: never release unchecked text

GRID = list(itertools.product([0, 1, 8, 32], [1, 4, 16], [None, 8, 64], [0.002, 0.1, 0.6], [None, 23, 90]))


@pytest.mark.parametrize("B,k,W,lat,inj", GRID)
def test_fail_closed_never_releases_unchecked_text(B, k, W, lat, inj):
    toks = make_tokens(120, inj)
    for cfg in (StreamGuardConfig(B, k, W, "FAIL_CLOSED"),  # stall
                StreamGuardConfig(B, k, W, "FAIL_CLOSED", check_timeout_s=0.3, max_unchecked_tokens=max(k, 40))):
        events, st, _, _ = run(toks, cfg, FakeClassifier(lat))
        assert_well_formed(events, st)
        assert st.released_unchecked == 0
        assert all(e.checked for e in events if isinstance(e, Release))
        # every released token was inside a check that returned allow, and it was released only after that check
        allowed = [c for c in st.checks if c.outcome == "allow"]
        for i, rt in enumerate(st.release_t):
            if rt is not None:
                assert any(c.span[0] <= i < c.span[1] and c.t_end <= rt for c in allowed), i
        # while the stream is running, B checked tokens stay held back
        done_t = st.arrival_t[-1] if st.generated == len(toks) else float("inf")
        for e in events:
            if isinstance(e, Release) and e.t < done_t:
                ends = [c.span[1] for c in allowed if c.t_end <= e.t]
                assert e.end <= max(ends) - B
        if inj is not None and MARK not in released_text(events):
            assert isinstance(events[-1], Stop)


def test_stall_mode_releases_everything_benign_and_coalesces_checks():
    toks = make_tokens(200)
    cfg = StreamGuardConfig(holdback_tokens=8, check_every=8, failure_mode="FAIL_CLOSED")
    assert cfg.stalls
    events, st, _, clf = run(toks, cfg, FakeClassifier(0.5), rate=100)  # a check costs 50 tokens of generation
    assert isinstance(events[-1], Done) and st.released == 200 and st.released_unchecked == 0
    assert len(clf.calls) < 200 // 8  # boundaries coalesced instead of queued
    delays = [r - a for a, r in zip(st.arrival_t, st.release_t)]
    assert max(delays) > 0.5  # the visible stream stalled behind the classifier


# ------------------------------------------------------------------ flag mid-stream

def test_flag_mid_stream_stops_and_retracts_held_back_buffer():
    toks = make_tokens(100, inject_at=40)
    cfg = StreamGuardConfig(holdback_tokens=16, check_every=8, window_tokens=None)
    events, st, src, _ = run(toks, cfg, FakeClassifier(0.001))
    stop = events[-1]
    assert isinstance(stop, Stop) and stop.reason == "flag"
    assert stop.check_span == (0, 48)  # the first check that contains token 40
    assert st.released == 24  # check (0, 40) passed and released 40 - B
    assert MARK not in released_text(events)
    assert stop.retracted_tokens == stop.generated_tokens - stop.released_tokens > 0
    assert src.closed  # generation was stopped, not drained
    assert not any(isinstance(e, Release) for e in events[events.index(stop):])


def test_flag_with_zero_holdback_can_leak_the_start_of_the_injection():
    """With B = 0, text up to the last passed check is shown; an injection starting inside a check window that
    the classifier only recognizes later leaks. The fake recognizes it only once 3 marker tokens are present."""
    toks = make_tokens(100, inject_at=38, inject_len=10)

    class Late(FakeClassifier):
        async def __call__(self, text):
            await asyncio.sleep(0.001)
            return CheckResult("flag" if text.count(MARK) >= 3 else "allow")

    events, st, _, _ = run(toks, StreamGuardConfig(0, 4), Late())
    assert isinstance(events[-1], Stop)
    assert released_text(events).count(MARK) == 2  # tokens 38, 39 released by the check ending at 40
    events, st, _, _ = run(toks, StreamGuardConfig(8, 4), Late())
    assert released_text(events).count(MARK) == 0


def test_service_failure_flag_stops_even_under_fail_open():
    """A flag produced by the classifier's own failure policy is the bundle's decision: never overridden."""
    toks = make_tokens(60, inject_at=20)
    events, _, _, _ = run(toks, StreamGuardConfig(4, 8, failure_mode="FAIL_OPEN"), FakeClassifier(failed_flag=True))
    assert isinstance(events[-1], Stop) and events[-1].reason == "classifier_failure"


# ------------------------------------------------------------------ slow / failing classifier

def test_fail_closed_timeout_stops_with_classifier_failure():
    events, st, src, _ = run(make_tokens(100), StreamGuardConfig(8, 8, check_timeout_s=0.1), FakeClassifier(0.3))
    assert isinstance(events[-1], Stop) and events[-1].reason == "classifier_failure"
    assert st.released == 0 and src.closed


def test_fail_closed_lag_budget_stops_when_classifier_falls_behind():
    cfg = StreamGuardConfig(8, 8, max_unchecked_tokens=32)
    events, st, _, _ = run(make_tokens(200), cfg, FakeClassifier(0.5), rate=100)
    assert isinstance(events[-1], Stop) and events[-1].reason == "lag"
    assert st.released_unchecked == 0


def test_fail_open_timeout_releases_unchecked_and_finishes():
    cfg = StreamGuardConfig(8, 8, failure_mode="FAIL_OPEN", check_timeout_s=0.1)
    events, st, _, _ = run(make_tokens(100), cfg, FakeClassifier(0.3))
    assert isinstance(events[-1], Done) and st.released == 100
    assert st.released_unchecked > 0
    assert any(not e.checked for e in events if isinstance(e, Release))


def test_fail_open_lag_bounds_unchecked_backlog_and_a_later_flag_still_stops():
    toks = make_tokens(300, inject_at=150)
    cfg = StreamGuardConfig(8, 8, failure_mode="FAIL_OPEN", max_unchecked_tokens=40)
    events, st, _, _ = run(toks, cfg, FakeClassifier(0.5), rate=100)
    stop = events[-1]
    assert isinstance(stop, Stop) and stop.reason == "flag"
    assert st.released_unchecked > 0
    # backlog bound: at every arrival at most 40 tokens are both unreleased and unresolved
    assert stop.generated_tokens - stop.released_tokens <= 40 + 8
    assert MARK in released_text(events)  # the price of FAIL_OPEN: the injection leaked before the late check


@pytest.mark.parametrize("mode,last", [("FAIL_CLOSED", Stop), ("FAIL_OPEN", Done)])
def test_classifier_exception(mode, last):
    clf = FakeClassifier(0.001, fail=lambda n: n == 3)
    events, st, _, _ = run(make_tokens(64), StreamGuardConfig(4, 8, failure_mode=mode), clf)
    assert isinstance(events[-1], last)
    if mode == "FAIL_CLOSED":
        assert events[-1].reason == "classifier_failure" and "RuntimeError" in events[-1].detail
        assert st.released_unchecked == 0
    else:
        # the failed check (0, 24) left tokens 16..24 unchecked; 16..20 were released before the next check, and
        # the next full-prefix check (0, 32) passed 20..24 while they were still held back (B = 4)
        assert st.released == 64 and st.released_unchecked == 8 - 4


# ------------------------------------------------------------------ k / B / W edge cases

def test_k1_checks_every_token():
    events, st, _, clf = run(make_tokens(20), StreamGuardConfig(0, 1), FakeClassifier(0.0001), rate=10)
    assert isinstance(events[-1], Done) and len(clf.calls) == 20


def test_holdback_longer_than_stream_releases_only_at_the_end():
    toks = make_tokens(30)
    events, st, _, _ = run(toks, StreamGuardConfig(64, 8), FakeClassifier(0.001))
    rel = [e for e in events if isinstance(e, Release)]
    assert len(rel) == 1 and (rel[0].start, rel[0].end) == (0, 30)
    assert rel[0].t >= st.arrival_t[-1]


def test_tail_not_a_multiple_of_k_gets_a_final_check():
    events, st, _, clf = run(make_tokens(21, inject_at=20, inject_len=1), StreamGuardConfig(0, 8), FakeClassifier())
    assert isinstance(events[-1], Stop) and events[-1].check_span == (0, 21)
    assert st.released == 16  # checks at 8 and 16 passed; B = 0


def test_empty_stream():
    events, st, _, clf = run([], StreamGuardConfig(4, 4))
    assert len(events) == 1 and isinstance(events[0], Done) and events[0].released_tokens == 0 and not clf.calls


def scored_tokens(st):
    """Tokens inside a check that actually scored them: outcome allow (not a failed-allow) or flag."""
    cov = set()
    for c in st.checks:
        if c.outcome in ("allow", "flag"):
            cov.update(range(*c.span))
    return cov


@pytest.mark.parametrize("W,k,lat", [(4, 16, 0.001), (8, 8, 0.4), (64, 8, 0.4), (1, 1, 0.05)])
def test_windows_cover_every_token(W, k, lat):
    toks = make_tokens(150)
    _, st, _, _ = run(toks, StreamGuardConfig(0, k, W), FakeClassifier(lat), rate=100)
    for c in st.checks:
        assert c.span[1] - c.span[0] >= min(W, c.span[1])
    assert scored_tokens(st) == set(range(150))  # a timed-out, raised or failed-allow span does not count


class Flaky(FakeClassifier):
    """Fails on the given call numbers (1-based) in one of three ways; otherwise the marker classifier."""

    def __init__(self, how, on, latency=0.005):
        super().__init__(latency)
        self.how, self.on = how, set(on)

    async def __call__(self, text):
        n = len(self.calls) + 1
        if n in self.on:
            self.calls.append(text)
            if self.how == "timeout":
                await asyncio.sleep(10.0)
            await asyncio.sleep(self.latency)
            if self.how == "exception":
                raise RuntimeError("classifier crashed")
            return CheckResult("allow", failed=True)  # the classifier's own failure policy allowed unscored text
        return await super().__call__(text)


@pytest.mark.parametrize("W", [None, 1, 8, 16, 64])
@pytest.mark.parametrize("how", ["timeout", "exception", "failed_allow"])
def test_fail_open_rescores_a_failed_window(W, how):
    """Review streaming-r1 M1: under FAIL_OPEN a check that did not score releases its text unchecked, and the next
    check starts at the first unscored token, so every token still ends up in a scored window."""
    cfg = StreamGuardConfig(0, 8, W, "FAIL_OPEN", check_timeout_s=0.2 if how == "timeout" else None)
    events, st, _, clf = run(make_tokens(150), cfg, Flaky(how, on={3, 4}))
    assert isinstance(events[-1], Done) and st.released == 150
    assert st.released_unchecked > 0 and any(not e.checked for e in events if isinstance(e, Release))
    assert [c.outcome for c in st.checks][2:4] == [{"failed_allow": "failed_open"}.get(how, how)] * 2
    assert scored_tokens(st) == set(range(150))
    last_scored = st.checks[1].span[1]  # the failed checks' unscored part starts here
    assert st.checks[4].span[0] <= last_scored and st.checks[4].outcome == "allow"  # re-covers the failed span


def test_fail_open_window_rescores_a_timed_out_injection_window():
    """Review streaming-r1 probe P3: k = W = 16, FAIL_OPEN with a deadline, and a classifier that is slow on the
    first window holding the injection (64, 80). That check times out and its tokens are released unchecked; the
    next window must start at 64 again, see the injection and stop the stream. The old scheduler started it at 80,
    and the stream ended Done with the injection never scored."""
    seen = {"slow": False}

    class SlowOnce(FakeClassifier):
        async def __call__(self, text):
            if MARK in text and not seen["slow"]:
                seen["slow"] = True
                self.calls.append(text)
                await asyncio.sleep(1.0)
                return CheckResult("flag")
            return await super().__call__(text)

    cfg = StreamGuardConfig(0, 16, window_tokens=16, failure_mode="FAIL_OPEN", check_timeout_s=0.2)
    events, st, _, _ = run(make_tokens(200, inject_at=70), cfg, SlowOnce())
    assert [(c.span, c.outcome) for c in st.checks if c.outcome != "allow"][0] == ((64, 80), "timeout")
    stop = events[-1]
    assert isinstance(stop, Stop) and stop.reason == "flag" and stop.check_span[0] == 64
    assert set(range(70, 75)) <= scored_tokens(st)
    # the price of FAIL_OPEN: the timed-out span was shown, marked unchecked
    assert all(not e.checked for e in events if isinstance(e, Release) and e.start < 75 and e.end > 70)


def test_fail_open_window_keeps_rescoring_while_checks_keep_failing():
    """Probe P3 as the reviewer wrote it: every check containing the injection takes 1 s against a 0.2 s deadline.
    FAIL_OPEN then releases the rest of the stream unchecked (counted, marked checked=False), but every later check
    still runs over the failed span; none skips it."""
    cfg = StreamGuardConfig(0, 16, window_tokens=16, failure_mode="FAIL_OPEN", check_timeout_s=0.2)
    clf = FakeClassifier(lambda text: 1.0 if MARK in text else 0.005)
    events, st, _, _ = run(make_tokens(200, inject_at=70), cfg, clf)
    assert isinstance(events[-1], Done)
    failed = [c for c in st.checks if c.outcome == "timeout"]
    assert failed[0].span == (64, 80) and all(c.span[0] <= 64 for c in failed)
    assert all(c.outcome == "timeout" for c in st.checks if c.span[1] > 64)
    assert not set(range(70, 75)) & scored_tokens(st)
    assert all(not st.checked[i] for i in range(64, 200)) and st.released_unchecked == 200 - 64


def test_fail_closed_window_stops_on_a_failed_allow():
    """Review probe P3b: stall FAIL_CLOSED, W = 16, and the classifier's own failure policy answers allow
    (failed=True) on the short window holding the injection. The guard must not release that unscored text."""
    class FailedOnMark(FakeClassifier):
        async def __call__(self, text):
            if MARK in text and text.count(" ") < 40:
                self.calls.append(text)
                await asyncio.sleep(self.latency)
                return CheckResult("allow", failed=True)
            return await super().__call__(text)

    events, st, _, _ = run(make_tokens(200, inject_at=66), StreamGuardConfig(0, 16, window_tokens=16), FailedOnMark())
    stop = events[-1]
    assert isinstance(stop, Stop) and stop.reason == "classifier_failure" and stop.check_span == (64, 80)
    assert MARK not in released_text(events) and st.released == 64 and st.released_unchecked == 0
    assert st.checks[-1].outcome == "failed_open"


# Review streaming-r1 m1: a check that raises, times out or comes back as a failed-allow, under each guard setting.
FAIL_SETTINGS = {
    "stall_closed": dict(failure_mode="FAIL_CLOSED"),
    "budget_closed": dict(failure_mode="FAIL_CLOSED", check_timeout_s=0.3, max_unchecked_tokens=40),
    "stall_open": dict(failure_mode="FAIL_OPEN"),
    "budget_open": dict(failure_mode="FAIL_OPEN", check_timeout_s=0.3, max_unchecked_tokens=40),
}


@pytest.mark.parametrize("setting", sorted(FAIL_SETTINGS))
@pytest.mark.parametrize("how", ["timeout", "exception", "failed_allow"])
@pytest.mark.parametrize("W", [None, 16])
@pytest.mark.parametrize("inj", [None, 60])
def test_failed_check_under_each_setting(setting, how, W, inj):
    kw = FAIL_SETTINGS[setting]
    if how == "timeout" and kw.get("check_timeout_s") is None:
        pytest.skip("no deadline: a slow check stalls instead of failing")
    cfg = StreamGuardConfig(4, 8, W, **kw)
    events, st, _, _ = run(make_tokens(120, inj), cfg, Flaky(how, on={3}))
    assert_well_formed(events, st)
    if cfg.failure_mode == "FAIL_CLOSED":
        # the guaranteed setting: nothing a check did not score is ever released, stall or not
        stop = events[-1]
        assert isinstance(stop, Stop) and stop.reason == "classifier_failure"
        assert st.released_unchecked == 0 and all(st.checked[i] for i in range(st.released))
        allowed = [c for c in st.checks if c.outcome == "allow"]
        for i in range(st.released):
            assert any(c.span[0] <= i < c.span[1] and c.t_end <= st.release_t[i] for c in allowed), i
    else:
        # FAIL_OPEN, stall included: the failed check's text goes out unchecked, and later checks re-cover it
        assert st.released_unchecked > 0
        assert any(not e.checked for e in events if isinstance(e, Release))
        if inj is None:
            assert isinstance(events[-1], Done) and scored_tokens(st) == set(range(120))
        else:
            assert isinstance(events[-1], Stop) and events[-1].reason == "flag"
            assert MARK not in released_text(events)


def test_failed_allow_on_every_check_stops_fail_closed_immediately():
    """Review probe P1: stall FAIL_CLOSED and a bundle whose decide member fails open on every call. The old guard
    released all 64 tokens unchecked and ended Done; it must stop at the first check and release nothing."""
    class AlwaysFailedAllow(FakeClassifier):
        async def __call__(self, text):
            self.calls.append(text)
            await asyncio.sleep(self.latency)
            return CheckResult("allow", failed=True)

    events, st, _, clf = run(make_tokens(64, 20), StreamGuardConfig(4, 8), AlwaysFailedAllow())
    assert isinstance(events[-1], Stop) and events[-1].reason == "classifier_failure"
    assert st.released == 0 and len(clf.calls) == 1


def test_window_misses_an_injection_split_across_checks_only_if_window_is_too_small():
    """A classifier that needs the whole 6-token injection in one window: W = 4 never sees it (a documented limit
    of windowed rescoring), the full prefix does."""
    toks = make_tokens(80, inject_at=30, inject_len=6)

    class Whole(FakeClassifier):
        async def __call__(self, text):
            await asyncio.sleep(0.001)
            return CheckResult("flag" if text.count(MARK) >= 6 else "allow")

    ev_small, _, _, _ = run(toks, StreamGuardConfig(0, 4, window_tokens=4), Whole())
    ev_full, _, _, _ = run(toks, StreamGuardConfig(0, 4, window_tokens=None), Whole())
    assert isinstance(ev_small[-1], Done) and isinstance(ev_full[-1], Stop)


@pytest.mark.parametrize("kw", [dict(holdback_tokens=-1), dict(check_every=0), dict(window_tokens=0),
                                dict(failure_mode="FALLBACK"), dict(check_timeout_s=0),
                                dict(check_every=16, max_unchecked_tokens=8),
                                dict(holdback_tokens=64, check_every=8, failure_mode="FAIL_OPEN",
                                     max_unchecked_tokens=40)])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        StreamGuardConfig(**kw)


def test_lag_budget_below_holdback_is_allowed_only_fail_closed():
    """Review n1: under FAIL_OPEN the lag rule releases the held-back buffer too, so a lag budget below B would make B
    meaningless and is rejected; under FAIL_CLOSED the lag rule stops the stream instead, so it is fine."""
    StreamGuardConfig(holdback_tokens=64, check_every=8, failure_mode="FAIL_CLOSED", max_unchecked_tokens=40)
    StreamGuardConfig(holdback_tokens=40, check_every=8, failure_mode="FAIL_OPEN", max_unchecked_tokens=40)


# ------------------------------------------------------------------ real event loop and the service adapter

def test_runs_on_a_real_event_loop():
    async def main():
        src = Source(make_tokens(40, inject_at=25), rate=2000)
        return await collect(src.gen(), FakeClassifier(0.001), StreamGuardConfig(4, 8))
    events, st = asyncio.run(main())
    assert isinstance(events[-1], Stop) and events[-1].reason == "flag" and st.released_unchecked == 0


def test_consumer_closing_early_closes_the_source():
    async def main():
        src = Source(make_tokens(100), rate=1000)
        agen = guard_stream(src.gen(), FakeClassifier(0.001), StreamGuardConfig(0, 4))
        first = await agen.__anext__()
        await agen.aclose()
        return first, src
    first, src = run_virtual(main())
    assert isinstance(first, Release) and src.closed


def test_service_check_adapter_reads_decision_score_and_failure():
    def outcome(decision, policy_applied=None, score=0.7):
        r_active = SimpleNamespace(role="active", stage="stage2", score=score)
        r_obs = SimpleNamespace(role="observe", stage="stage1", score=0.1)
        return SimpleNamespace(decision=decision, policy_applied=policy_applied, coverage_policy_applied=None,
                               decided_by="deberta-pi-v2", results=[r_active, r_obs])

    class Svc:
        def __init__(self, out):
            self.out = out

        async def classify(self, text):
            return self.out

    ok = asyncio.run(service_check(Svc(outcome("flag")))("x"))
    assert ok == CheckResult("flag", failed=False, reason=None, score=0.7)
    failed = asyncio.run(service_check(Svc(outcome("flag", "fail_closed", None)))("x"))
    assert failed.failed and failed.decision == "flag" and failed.score is None
    # a coverage rule deciding content no step scored is a failure too, whichever way it decided
    cov = outcome("allow")
    cov.coverage_policy_applied = "allow"
    res = asyncio.run(service_check(Svc(cov))("x"))
    assert res.failed and res.decision == "allow"
