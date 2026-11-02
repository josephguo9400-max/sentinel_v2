"""The streaming harness's bookkeeping (serving/bench/streaming.py), with a fake scorer: no model, no lock."""
from serving.bench import streaming as sb


class FakeScorer:
    """Flags any text containing the marker; latency proportional to length."""

    def __init__(self, ms_per_char=0.2):
        self.ms, self.calls = ms_per_char, 0

    def get(self, text):
        self.calls += 1
        return {"decision": "flag" if "<INJ>" in text else "allow", "failed": False, "score": None,
                "latency_ms": self.ms * len(text)}


def stream(kind, n=100, at=40, length=10):
    toks = [f"w{i} " for i in range(n)]
    span = None
    if kind == "injection":
        toks[at:at + length] = ["<INJ>"] * length
        span = [at, at + length]
    return {"id": f"{kind}-1", "kind": kind, "pool": "newsletter", "group": "g", "tokens": toks, "span": span}


def test_detection_delay_and_leak_are_counted_from_the_injection_start():
    cfg = sb.StreamGuardConfig(holdback_tokens=0, check_every=8)
    rec = sb.simulate(stream("injection"), cfg, 50.0, FakeScorer(0.0001))
    assert rec["stopped"] and rec["stop_check_end"] == 48
    assert rec["leaked"] == 0 and rec["released"] == 40  # check (0, 40) passed, B = 0
    assert rec["delay_tokens"] == rec["generated"] - 40 >= 8


def test_aggregate_counts_false_stops_and_catches():
    cfg = sb.StreamGuardConfig(holdback_tokens=8, check_every=8)
    recs = [sb.simulate(stream("injection"), cfg, 50.0, FakeScorer()), sb.simulate(stream("benign"), cfg, 50.0,
                                                                                    FakeScorer())]
    m = sb.aggregate(recs, 50.0)
    assert m["injection"]["caught"]["k"] == 1 and m["injection"]["caught_zero_leak"]["k"] == 1
    assert m["benign"]["false_stop"]["k"] == 0
    assert m["unchecked"]["released_unchecked_tokens"] == 0
    assert m["compute"]["checks_per_1k_generated_tokens"] > 0


def test_a_flag_whose_window_misses_the_injection_is_not_a_catch():
    """W = 8: a flag on benign text after the injection (the window no longer contains it) is not counted as caught."""
    toks = [f"w{i} " for i in range(100)]
    toks[90] = "<INJ>"  # the scorer flags at 90; the "injection" span is earlier
    s = {"id": "x", "kind": "injection", "pool": "newsletter", "group": "g", "tokens": toks, "span": [40, 50]}
    rec = sb.simulate(s, sb.StreamGuardConfig(0, 8, window_tokens=8), 50.0, FakeScorer(0.0001))
    m = sb.aggregate([rec], 50.0)
    assert rec["stop_reason"] == "flag" and m["injection"]["caught"]["k"] == 0
    assert m["injection"]["stopped_by_flag_outside_injection"]["k"] == 1


def test_a_benign_flag_before_the_injection_is_reported_and_excluded():
    """Review streaming-r1 m4: a stream stopped by a check that ended before the injection started never reached it;
    it is counted in stopped_before_injection and left out of the caught_excluding_stopped_before denominator."""
    toks = [f"w{i} " for i in range(100)]
    toks[10] = "<INJ>"  # the scorer flags benign text at 10; the injection span starts at 40
    early = {"id": "e", "kind": "injection", "pool": "newsletter", "group": "g", "tokens": toks, "span": [40, 50]}
    cfg = sb.StreamGuardConfig(0, 8)
    recs = [sb.simulate(early, cfg, 50.0, FakeScorer(0.0001)),
            sb.simulate(stream("injection"), cfg, 50.0, FakeScorer(0.0001))]
    m = sb.aggregate(recs, 50.0)["injection"]
    assert m["caught"]["k"] == 1 and m["caught"]["n"] == 2
    assert m["stopped_before_injection"]["k"] == 1 and m["stopped_before_injection_by_flag"]["k"] == 1
    assert (m["caught_excluding_stopped_before"]["k"], m["caught_excluding_stopped_before"]["n"]) == (1, 1)


def test_mcnemar_exact_p():
    assert sb._mcnemar_p(0, 0) is None
    assert sb._mcnemar_p(4, 0) == 0.125  # 2 * 0.5**4
    assert sb._mcnemar_p(3, 3) == 1.0
