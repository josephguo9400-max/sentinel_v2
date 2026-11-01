"""Output-side streaming guard: classify a model's generated text while it streams (design: docs/design/streaming.md).

`guard_stream(tokens, check, config)` consumes an async iterator of generated token strings and yields what the
user may see:

- **Hold-back buffer (B = `holdback_tokens`).** Only text a check has passed is released, and the last B checked
  tokens are held back as well, so a later flag can still retract them. The visible stream lags generation by at
  least B tokens plus the time the pending check takes. When generation ends, the final check covers everything
  and the held-back tail is released.
- **Check every k tokens (`check_every`).** One check runs at a time. When a check finishes, the next one covers
  up to the newest k-token boundary that has arrived, so a slow classifier coalesces boundaries (the effective k
  grows) instead of queueing checks.
- **Window (`window_tokens`).** `None` re-scores the full prefix on every check. W > 0 scores the last W tokens,
  extended back to the end of the previous check if the coalesced span is longer than W, so every token is in at
  least one scored window.
- **Flag.** Any check that returns `flag` stops the stream: the source is closed, every unreleased token is
  retracted (never shown) and a `Stop` event is the last event. Text already released stays released; this guard
  cannot unsay it, which is what B is for.
- **Failures the guard sees** (the check raised, `check_timeout_s` passed, or more than `max_unchecked_tokens`
  generated tokens are waiting for a check) follow `failure_mode`: `FAIL_CLOSED` stops the stream; `FAIL_OPEN`
  releases the affected text marked `checked=False` and continues. With `check_timeout_s=None` and
  `max_unchecked_tokens=None` nothing fails for slowness: the visible stream stalls until the check returns
  (FAIL_CLOSED with an unbounded lag budget), which is the only setting that cannot release unchecked text even
  under FAIL_OPEN.
- **The bundle's decision is never overridden.** A check result of `flag` stops the stream even when the
  classifier's own failure policy produced it (`CheckResult.failed`): the guard's failure mode covers only what the
  classifier service could not see (the guard's own deadline and lag), so it is never more permissive than the
  served bundle.

This module is outside the engine's decision path (serving/engine_version.json): it calls the classifier service
the same way any client does and changes neither the policy engine nor any bundle_sha256. Its own settings are not
part of any ExecutionPolicy, so a stream decision is not bound to a gated bundle (docs/design/streaming.md, open
questions); no endpoint serves it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Callable, Literal

FailureMode = Literal["FAIL_OPEN", "FAIL_CLOSED"]
StopReason = Literal["flag", "classifier_failure", "lag"]


@dataclass(frozen=True)
class StreamGuardConfig:
    holdback_tokens: int = 16  # B
    check_every: int = 16  # k
    window_tokens: int | None = None  # None: full prefix
    failure_mode: FailureMode = "FAIL_CLOSED"
    check_timeout_s: float | None = None  # None: wait for the check (stall)
    max_unchecked_tokens: int | None = None  # generated tokens waiting for a check; None: unbounded (stall)

    def __post_init__(self):
        if not isinstance(self.holdback_tokens, int) or self.holdback_tokens < 0:
            raise ValueError("holdback_tokens must be an int >= 0")
        if not isinstance(self.check_every, int) or self.check_every < 1:
            raise ValueError("check_every must be an int >= 1")
        if self.window_tokens is not None and (not isinstance(self.window_tokens, int) or self.window_tokens < 1):
            raise ValueError("window_tokens must be None (full prefix) or an int >= 1")
        if self.failure_mode not in ("FAIL_OPEN", "FAIL_CLOSED"):
            raise ValueError("failure_mode must be FAIL_OPEN or FAIL_CLOSED")
        if self.check_timeout_s is not None and not self.check_timeout_s > 0:
            raise ValueError("check_timeout_s must be None or > 0")
        if self.max_unchecked_tokens is not None and self.max_unchecked_tokens < self.check_every:
            # below k the lag rule would fire before the first check could be scheduled
            raise ValueError("max_unchecked_tokens must be None or >= check_every")

    @property
    def stalls(self) -> bool:
        """True iff slowness never fails a check: the visible stream waits instead."""
        return self.check_timeout_s is None and self.max_unchecked_tokens is None


@dataclass(frozen=True)
class CheckResult:
    decision: Literal["allow", "flag"]
    failed: bool = False  # the classifier did not score; its own failure policy produced `decision`
    reason: str | None = None
    score: float | None = None


Check = Callable[[str], Awaitable[CheckResult]]


@dataclass(frozen=True)
class Release:
    text: str
    start: int  # token span [start, end)
    end: int
    checked: bool
    t: float


@dataclass(frozen=True)
class Stop:
    reason: StopReason
    generated_tokens: int  # tokens received from the source when the stream stopped
    released_tokens: int  # tokens the user saw
    retracted_tokens: int  # generated but never released
    check_span: tuple[int, int] | None  # the check that stopped it (None for a lag stop)
    detail: str | None
    t: float


@dataclass(frozen=True)
class Done:
    generated_tokens: int
    released_tokens: int
    t: float


Event = Release | Stop | Done


@dataclass
class CheckRecord:
    span: tuple[int, int]
    t_start: float
    t_end: float | None = None
    outcome: str | None = None  # allow | flag | failed_open | failed_closed | timeout | exception | abandoned
    score: float | None = None


@dataclass
class StreamStats:
    """Per-stream bookkeeping for measurement and tests."""
    t0: float = 0.0
    arrival_t: list[float] = field(default_factory=list)
    release_t: list[float | None] = field(default_factory=list)
    checked: list[bool] = field(default_factory=list)  # token passed a check (or the final check)
    released_unchecked: int = 0
    checks: list[CheckRecord] = field(default_factory=list)
    stop: Stop | None = None
    end_t: float | None = None

    @property
    def generated(self) -> int:
        return len(self.arrival_t)

    @property
    def released(self) -> int:
        return sum(t is not None for t in self.release_t)


async def guard_stream(tokens: AsyncIterable[str], check: Check, config: StreamGuardConfig,
                       stats: StreamStats | None = None) -> AsyncIterator[Event]:
    """Yield Release events for checked text, then exactly one Stop or Done."""
    cfg = config
    st = stats if stats is not None else StreamStats()
    loop = asyncio.get_running_loop()
    st.t0 = loop.time()
    it = tokens.__aiter__()
    toks: list[str] = []
    released = 0  # r: tokens [0, r) are visible
    checked_to = 0  # c: the last resolved check ended here
    sched_to = 0  # end of the last scheduled check
    source_done = False
    next_tok: asyncio.Task | None = asyncio.ensure_future(it.__anext__())
    inflight: asyncio.Task | None = None
    inflight_span: tuple[int, int] | None = None
    inflight_rec: CheckRecord | None = None

    def release_to(end: int) -> list[Release]:
        nonlocal released
        out: list[Release] = []
        end = min(end, len(toks))
        now = loop.time()
        while released < end:
            ok = st.checked[released]
            j = released
            while j < end and st.checked[j] == ok:
                j += 1
            for i in range(released, j):
                st.release_t[i] = now
            if not ok:
                st.released_unchecked += j - released
            out.append(Release("".join(toks[released:j]), released, j, ok, now))
            released = j
        return out

    async def close_source():
        if next_tok is not None and not next_tok.done():
            next_tok.cancel()
            try:
                await next_tok
            except BaseException:  # noqa: BLE001 - cancelled or the source's own error; the stream is over
                pass
        aclose = getattr(it, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass

    async def stop(reason: StopReason, span, detail) -> Stop:
        nonlocal inflight
        if inflight is not None and not inflight.done():
            inflight.cancel()  # abandon: a real ONNX call cannot be cancelled, it finishes in the service
            if inflight_rec is not None and inflight_rec.outcome is None:
                inflight_rec.outcome, inflight_rec.t_end = "abandoned", loop.time()
        await close_source()
        ev = Stop(reason, len(toks), released, len(toks) - released, span, detail, loop.time())
        st.stop, st.end_t = ev, ev.t
        return ev

    async def run_check(text: str) -> CheckResult:
        if cfg.check_timeout_s is None:
            return await check(text)
        return await asyncio.wait_for(check(text), cfg.check_timeout_s)

    try:
        while True:
            # ---- schedule the next check (one at a time; coalesce to the newest k boundary)
            if inflight is None:
                n = len(toks)
                if source_done:
                    p = n if n > sched_to else None
                else:
                    p = sched_to + cfg.check_every * ((n - sched_to) // cfg.check_every)
                    p = p if p > sched_to else None
                if p is not None:
                    s = 0 if cfg.window_tokens is None else max(0, min(p - cfg.window_tokens, sched_to))
                    inflight_span = (s, p)
                    inflight_rec = CheckRecord(span=inflight_span, t_start=loop.time())
                    st.checks.append(inflight_rec)
                    inflight = asyncio.ensure_future(run_check("".join(toks[s:p])))
                    sched_to = p
            if source_done and inflight is None:
                for ev in release_to(len(toks)):
                    yield ev
                st.end_t = loop.time()
                yield Done(len(toks), released, st.end_t)
                return

            waiting = [t for t in (next_tok, inflight) if t is not None]
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)

            # ---- a check finished (handled first: it may stop the stream)
            if inflight is not None and inflight.done():
                task, span, rec = inflight, inflight_span, inflight_rec
                inflight = inflight_span = inflight_rec = None
                rec.t_end = loop.time()
                try:
                    res: CheckResult | None = task.result()
                    err = None
                except asyncio.TimeoutError:
                    res, err = None, "timeout"
                except Exception as exc:  # noqa: BLE001 - a broken classifier is a failure, not a crash
                    res, err = None, f"exception: {type(exc).__name__}"
                if res is not None:
                    rec.score = res.score
                if res is not None and res.decision == "flag":
                    rec.outcome = "flag" if not res.failed else "failed_closed"
                    yield await stop("flag" if not res.failed else "classifier_failure", span,
                                     res.reason)
                    return
                if res is None and cfg.failure_mode == "FAIL_CLOSED":
                    rec.outcome = err.split(":")[0]
                    yield await stop("classifier_failure", span, err)
                    return
                if res is None:  # FAIL_OPEN: the span counts as resolved, its tokens stay unchecked
                    rec.outcome = err.split(":")[0]
                else:
                    rec.outcome = "allow" if not res.failed else "failed_open"
                    if not res.failed:
                        for i in range(span[0], span[1]):
                            st.checked[i] = True
                checked_to = max(checked_to, span[1])  # a FAIL_OPEN lag release may already be past it
                final = source_done and checked_to == len(toks)
                for ev in release_to(len(toks) if final else checked_to - cfg.holdback_tokens):
                    yield ev

            # ---- a token arrived (or the source ended)
            if next_tok is not None and next_tok.done():
                task, next_tok = next_tok, None
                try:
                    tok = task.result()
                except StopAsyncIteration:
                    source_done = True
                else:
                    toks.append(tok)
                    st.arrival_t.append(loop.time())
                    st.release_t.append(None)
                    st.checked.append(False)
                    next_tok = asyncio.ensure_future(it.__anext__())
                    lag = len(toks) - checked_to
                    if cfg.max_unchecked_tokens is not None and lag > cfg.max_unchecked_tokens:
                        if cfg.failure_mode == "FAIL_CLOSED":
                            yield await stop("lag", None, f"{lag} tokens waiting for a check")
                            return
                        # FAIL_OPEN: release the oldest tokens so at most max_unchecked_tokens wait
                        for ev in release_to(len(toks) - cfg.max_unchecked_tokens):
                            yield ev
                        checked_to = max(checked_to, released)
    finally:
        if st.stop is None and st.end_t is None:  # the consumer stopped iterating early
            if inflight is not None and not inflight.done():
                inflight.cancel()
            await close_source()


async def collect(tokens: AsyncIterable[str], check: Check, config: StreamGuardConfig
                  ) -> tuple[list[Event], StreamStats]:
    st = StreamStats()
    events = [ev async for ev in guard_stream(tokens, check, config, st)]
    return events, st


def service_check(service: Any) -> Check:
    """A Check backed by a started ClassifierService (the served bundle): one /classify-equivalent call per check.

    `failed` is set when the decide member did not score and its failure policy decided (`policy_applied`), or
    when a coverage rule decided content no step scored, so the guard can tell a model flag from a failure flag.
    """
    async def _check(text: str) -> CheckResult:
        out = await service.classify(text)
        active = [r for r in out.results if r.role == "active" and r.stage != "stage1"]
        score = max((r.score for r in active if r.score is not None), default=None)
        failed = out.policy_applied is not None
        reason = out.decided_by if failed or out.coverage_policy_applied else None
        return CheckResult(decision=out.decision, failed=failed, reason=reason, score=score)
    return _check
