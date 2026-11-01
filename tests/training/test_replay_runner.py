"""Replay runner against a fake gateway (a threaded stdlib HTTP server; no serving code is imported)."""
import asyncio
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from training.common import ROOT, read_jsonl
from training.replay_runner import Config, Runner, main

REPLAY = ROOT / "data" / "replay" / "replay-v1.jsonl"


def _dr(rid, stage, status="ok", score=0.1, reason=None, role="active"):
    ok = status == "ok"
    return {"schema_version": "1.2", "request_id": rid, "detector_id": f"fake-{stage}-{role}",
            "detector_version": "0", "status": status, "score": score if ok else None, "threshold": 0.5,
            "predicted_injection": (score >= 0.5) if ok else None, "latency_ms": 0.1,
            "input_sha256": hashlib.sha256(rid.encode()).hexdigest(), "input_tokens": 3, "n_chunks": 1,
            "truncated": not ok, "aggregation": "max", "stage": stage, "role": role, "failure_reason": reason}


def fake_response(rid: str, text: str) -> tuple[int, dict | str]:
    """Markers in the text choose the outcome. Non-ASCII text escalates (like a language gate)."""
    if "ERR500" in text:
        return 500, {"detail": "boom"}
    if "BADJSON" in text:
        return 200, {"request_id": rid, "decision": "maybe"}
    flag = "FLAG" in text
    escalate = "ESC" in text or not text.isascii()
    shadow = [_dr(rid, "stage2", score=0.9 if "SHADOWDISAGREE" in text else (0.9 if flag else 0.1), role="shadow")]
    if "SHADOWFAIL" in text:
        shadow = [_dr(rid, "stage2", "timeout", reason="timeout", role="shadow")]
    if "OVERLOAD" in text:
        return 200, {"request_id": rid, "decision": "flag", "decided_by": "fail_closed", "policy_applied": "fail_closed",
                     "coverage_policy_applied": None, "stages_run": ["stage1", "stage2"],
                     "results": [_dr(rid, "stage1", score=0.6), _dr(rid, "stage2", "error", reason="overloaded")]}
    results = [_dr(rid, "stage1", score=0.6 if escalate else (0.95 if flag else 0.05))]
    stages = ["stage1"]
    if escalate:
        results.append(_dr(rid, "stage2", score=0.9 if flag else 0.1))
        stages.append("stage2")
    return 200, {"request_id": rid, "decision": "flag" if flag else "allow", "decided_by": f"fake-{stages[-1]}-active",
                 "policy_applied": None, "coverage_policy_applied": None, "stages_run": stages,
                 "results": results + shadow}


class FakeGateway:
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s
        self.seen = []
        self.max_concurrent = 0
        self._cur = 0
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer._cur += 1
                    outer.max_concurrent = max(outer.max_concurrent, outer._cur)
                    outer.seen.append((self.path, sorted(body)))
                try:
                    if outer.delay_s:
                        time.sleep(outer.delay_s)
                    code, payload = fake_response(body["request_id"], body["text"])
                finally:
                    with outer._lock:
                        outer._cur -= 1
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()


def _items(texts, langs=None, labels=None, delays=None, gap=0.1):
    out = []
    for i, t in enumerate(texts):
        out.append({"id": f"i{i}", "seq": i, "text": t, "lang": (langs or ["en"] * len(texts))[i],
                    "label": (labels or [None] * len(texts))[i],
                    "label_available_after_s": (delays or [None] * len(texts))[i], "arrival_offset_s": i * gap})
    return out


def _run(items, url, out_dir=None, **kw):
    cfg = Config(url=url, nominal_rps=10.0, **kw)
    return asyncio.run(Runner(items, cfg, out_dir).run())


def test_windows_validate_and_count_routing_errors_and_failure_reasons(tmp_path):
    texts = (["hello"] * 6 + ["FLAG now"] * 2 + ["ESC please"] * 2 + ["OVERLOAD"] + ["ERR500"] + ["BADJSON"]
             + ["SHADOWFAIL ok"] + ["SHADOWDISAGREE ok"] + ["hello"])
    with FakeGateway() as gw:
        res = _run(_items(texts), gw.url, tmp_path, rps=200, window_s=0.05, mode="open", arrivals="uniform")
    assert gw.seen[0] == ("/classify", ["request_id", "text"])
    assert res["n_sent"] == 16 and res["n_responses"] == 14
    assert res["errors"] == {"http_500": 1, "invalid_response": 1}
    assert res["failure_reason"] == {"overloaded": 1}
    windows = [json.loads(x) for x in (tmp_path / "windows.jsonl").read_text().splitlines()]
    assert sum(w["n_sent"] for w in windows) == 16 and [w["window"] for w in windows] == list(range(len(windows)))
    tot = lambda k: sum(w[k]["numerator"] for w in windows)  # noqa: E731
    assert tot("flag_rate") == 3  # 2 FLAG + the overload fail_closed
    assert tot("escalation_rate") == 3  # 2 ESC + overload
    assert sum(w["failures"]["overloaded"] for w in windows) == 1
    assert sum(w["failures"]["timeout_or_exception"] for w in windows) == 0
    assert sum(w["shadow"]["failure_reason"].get("timeout", 0) for w in windows) == 1
    agree = [w["shadow"]["agreement_with_served_decision"] for w in windows]
    # 14 responses: OVERLOAD has no shadow, SHADOWFAIL failed -> 12 ok shadows; SHADOWDISAGREE disagrees
    assert sum(a["denominator"] for a in agree) == 12 and sum(a["numerator"] for a in agree) == 11
    assert sum(sum(w["policy_applied"].values()) for w in windows) == 14
    req = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert len(req) == 16 and all("text" not in r for r in req)
    assert json.loads((tmp_path / "summary.json").read_text())["n_sent"] == 16


def test_drift_is_visible_per_window():
    # first half English, second half non-English (escalated by the fake gateway's language gate).
    # 40 items at 20 rps over 0.5 s windows: about 10 per window, so window 0 is all English and the last all not,
    # even if the host delays a few sends.
    texts = ["plain words"] * 20 + ["Grüße aus Köln"] * 20
    langs = ["en"] * 20 + ["de"] * 20
    with FakeGateway() as gw:
        res = _run(_items(texts, langs), gw.url, rps=20, window_s=0.5, mode="open", arrivals="uniform")
    w = [x for x in res["windows"] if x["n_sent"]]
    assert len(w) >= 3
    assert w[0]["non_en_share"] == 0.0 and w[0]["escalation_rate"] == 0.0
    assert w[-1]["non_en_share"] == 1.0 and w[-1]["escalation_rate"] == 1.0
    shares = [x["non_en_share"] for x in w]
    assert shares == sorted(shares)


def test_delayed_labels_are_used_only_after_their_delay(tmp_path):
    texts = ["FLAG x", "hello", "FLAG y", "hello"]
    labels = [1, 0, 1, None]
    delays = [0.0, 3.0, None, 1.0]  # nominal seconds; item 2 never revealed; item 3 has no label
    with FakeGateway() as gw:
        # rps 20 -> time scale 0.5: delays become 0, 1.5 s. window 0.25 s; run ends ~0.2 s, so item 1 is pending.
        res = _run(_items(texts, labels=labels, delays=delays), gw.url, rps=20, window_s=0.25, mode="open",
                   arrivals="uniform")
        assert res["labels"]["used"]["n_labels"] == 1 and res["labels"]["pending_at_end"] == 1
        assert res["labels"]["never_revealed"] == 2
        assert res["labels"]["used"]["recall_as_served"]["numerator"] == 1
        drained = _run(_items(texts, labels=labels, delays=delays), gw.url, tmp_path, rps=20, window_s=0.25,
                       mode="open", arrivals="uniform", drain_labels=True)
    assert drained["labels"]["used"]["n_labels"] == 2 and drained["labels"]["pending_at_end"] == 0
    windows = [json.loads(x) for x in (tmp_path / "windows.jsonl").read_text().splitlines()]
    revealed = [w["delayed_labels"]["revealed_in_window"]["n_labels"] for w in windows]
    assert revealed[0] == 1 and sum(revealed) == 2
    reveal_w = next(i for i, w in enumerate(windows) if i > 0 and w["delayed_labels"]["revealed_in_window"]["n_labels"])
    assert windows[reveal_w]["t_end_s"] >= 1.5  # item 1's label (sent ~0.05 s, delay 1.5 s) not used before then
    assert windows[reveal_w]["delayed_labels"]["cumulative"]["fpr_as_served"]["denominator"] == 1


def test_open_loop_keeps_the_rate_closed_loop_waits_for_responses():
    texts = ["hello"] * 20
    with FakeGateway(delay_s=0.05) as gw:
        t0 = time.monotonic()
        o = _run(_items(texts), gw.url, rps=100, window_s=1, mode="open", arrivals="uniform")
        t_open = time.monotonic() - t0
        open_conc = gw.max_concurrent
        gw.max_concurrent = 0
        c = _run(_items(texts), gw.url, rps=100, window_s=1, mode="closed", concurrency=1)
    assert o["n_responses"] == c["n_responses"] == 20
    assert open_conc >= 2 and gw.max_concurrent == 1
    assert c["duration_s"] >= 20 * 0.05 * 0.9  # bounded by latency, not by the 100 rps cap
    assert o["duration_s"] < c["duration_s"] and t_open < 1.5


def test_open_loop_counts_client_saturation_instead_of_slowing_down():
    with FakeGateway(delay_s=0.2) as gw:
        res = _run(_items(["hello"] * 10), gw.url, rps=200, window_s=1, mode="open", arrivals="uniform",
                   max_inflight=2)
    assert res["errors"].get("client_saturated", 0) >= 5 and res["n_sent"] == 10


def test_unreachable_gateway_is_a_transport_error():
    res = _run(_items(["hello"] * 3), "http://127.0.0.1:9", rps=100, window_s=1, timeout_s=1)
    assert res["n_responses"] == 0 and sum(res["errors"].values()) == 3
    assert all(k.startswith("transport:") for k in res["errors"])


def test_schedule_offsets_are_scaled_by_rate():
    items = _items(["hello"] * 5, gap=1.0)  # nominal 1 s gaps at nominal 10 rps
    r = Runner(items, Config(url="http://x", rps=40, nominal_rps=10))
    assert [r.scheduled(i, x) for i, x in enumerate(items)] == [0.0, 0.25, 0.5, 0.75, 1.0]
    with pytest.raises(ValueError):
        Runner(items, Config(url="http://x", rps=0))


@pytest.mark.skipif(not REPLAY.exists(), reason="replay set not built")
def test_cli_replays_the_committed_set_against_the_fake_gateway(tmp_path):
    with FakeGateway() as gw:
        assert main(["--url", gw.url, "--rps", "400", "--limit", "120", "--window-s", "0.1", "--out-dir",
                     str(tmp_path), "--mode", "open"]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_sent"] == 120 and s["n_responses"] == 120 and s["errors"] == {}
    assert s["config"]["replay_sha256"] and s["latency_conditions"].endswith("not a benchmark")
    ids = [json.loads(x)["id"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert ids == [r["id"] for r in list(read_jsonl(REPLAY))[:120]]


def test_observe_stage1_is_reported_apart_from_shadow_agreement():
    """D23: the observe-only Stage 1 (role="shadow", stage="stage1") only ever clears, so it gets would-allow
    figures (below-low share, below-low-but-served-flag) and never enters the shadow agreement figure."""
    from training.replay_runner import Rec, summarize_window

    def rec(i, decision, s1, policy=None):
        return Rec(item={"id": f"i{i}", "lang": "en", "label": None}, window=0, sent_at=0.0, label_at=None,
                   parsed={"decision": decision, "stages": ["stage2"], "decided_by": "deb", "policy_applied": policy,
                           "coverage_policy_applied": None, "failure_reasons": [], "shadow": [s1]})

    def s1(status="ok", pred=False, truncated=False):
        return {"status": status, "predicted_injection": pred if status == "ok" else None, "stage": "stage1",
                "truncated": truncated if status == "ok" else True, "detector_id": "stage1-tfidf-lr",
                "failure_reason": None if status == "ok" else "timeout"}

    recs = [rec(0, "allow", s1()), rec(1, "flag", s1()), rec(2, "flag", s1(), policy="fail_closed"),
            rec(3, "flag", s1(pred=True)), rec(4, "allow", s1(truncated=True)), rec(5, "allow", s1("timeout"))]
    w = summarize_window(0, 0.0, 1.0, recs, recs, -1.0)
    assert w["shadow"]["n_ok"] == 0 and w["shadow"]["agreement_with_served_decision"]["denominator"] == 0
    o = w["observe_stage1"]
    assert o["n_ok"] == 5 and o["n_partial"] == 1
    assert (o["below_low_share"]["numerator"], o["below_low_share"]["denominator"]) == (3, 5)
    assert o["below_low_but_served_flag"] == 2
    assert o["below_low_and_served_flag_by_policy"] == {"None": 1, "fail_closed": 1}
    assert o["failure_reason"] == {"timeout": 1}
