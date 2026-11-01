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


@pytest.mark.parametrize("W,k,lat", [(4, 16, 0.001), (8, 8, 0.4), (64, 8, 0.4), (1, 1, 0.05)])
def test_windows_cover_every_token(W, k, lat):
    toks = make_tokens(150)
    _, st, _, _ = run(toks, StreamGuardConfig(0, k, W), FakeClassifier(lat), rate=100)
    covered = set()
    for c in st.checks:
        assert c.span[1] - c.span[0] >= min(W, c.span[1])
        covered.update(range(*c.span))
    assert covered == set(range(150))


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
                                dict(check_every=16, max_unchecked_tokens=8)])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        StreamGuardConfig(**kw)


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
