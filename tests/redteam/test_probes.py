"""Probe generators place injections at exact DeBERTa token offsets that defeat chunked scoring.

Placement is verified three ways: against the module's mirrored constants, against the original
serving.encoding chunk planner, and against the planner the gateway uses now (ClassifierService._prepare:
serving/structure.py build_chunks) with its Stage 2 chunk selection when no Stage 1 clears anything (observe
mode, D23: stage2_order(...)[:max_chunks]), so a drift in any of them is caught.
"""
import pytest

from redteam import probes
from redteam.probes import (MAX_CHUNKS, STRIDE, WINDOW, generate_probes, load_probes, n_tokens,
                            token_span_of)

serving_encoding = pytest.importorskip("serving.encoding")
from serving.config import GatewayConfig  # noqa: E402
from serving.encoding import plan_chunks  # noqa: E402

CLS, SEP = 1, 2


def _tok(text):
    return probes._tokenizer().encode(text, add_special_tokens=False).ids


def _chunk_contains(chunk_ids, needle_ids):
    inner = chunk_ids[1:-1]  # strip [CLS]/[SEP]
    return any(inner[i:i + len(needle_ids)] == needle_ids for i in range(len(inner) - len(needle_ids) + 1))


def _plan(body):
    return plan_chunks(_tok(body), CLS, SEP, 512, STRIDE, MAX_CHUNKS)


ALL = generate_probes()
ATTACKS = [p for p in ALL if p.kind == "attack"]


def test_constants_match_the_gateway_defaults():
    c = GatewayConfig()
    assert WINDOW == c.max_len - 2 and STRIDE == c.stride and MAX_CHUNKS == c.max_chunks


def test_generation_is_deterministic():
    a = {p.probe_id: p.body for p in generate_probes()}
    b = {p.probe_id: p.body for p in generate_probes()}
    assert a == b
    assert {p.probe_id: p.body for p in generate_probes(seed=1)} != a


def test_committed_fixtures_match_generator():
    """The checked-in fixtures must be exactly what the seeded generator produces (no manual drift)."""
    gen = {p.probe_id: p.to_json() for p in generate_probes()}
    loaded = {p.probe_id: p.to_json() for p in load_probes()}
    assert loaded == gen


def test_composition():
    assert len(ALL) == 24
    by = {}
    for p in ALL:
        by.setdefault((p.placement, p.kind), 0)
        by[(p.placement, p.kind)] += 1
    assert by == {("middle_overlong", "attack"): 6, ("middle_overlong", "control"): 6,
                  ("boundary_split", "attack"): 6, ("boundary_split", "control"): 6}
    # every attack has a matched control of equal token length
    controls = {p.control_of: p for p in ALL if p.kind == "control"}
    for a in ATTACKS:
        assert a.probe_id in controls
        assert controls[a.probe_id].total_tokens == a.total_tokens


def test_addresses_are_invalid_only():
    for p in ATTACKS:
        assert p.exfil_address.endswith(".invalid")


@pytest.mark.parametrize("p", [p for p in ATTACKS if p.placement == "middle_overlong"],
                         ids=lambda p: p.probe_id)
def test_middle_injection_is_past_the_reach_and_dropped(p):
    span = token_span_of(p.body, p.injection_text)
    assert span == p.injection_token_span
    # well past the four-window reach (1,662)
    assert span[0] > WINDOW + STRIDE * (MAX_CHUNKS - 1)
    plan = _plan(p.body)
    assert plan.truncated is True
    # the whole injection appears in no scored chunk
    assert not any(_chunk_contains(ch, _tok(p.injection_text)) for ch in plan.chunks)
    # not even the exfil address survives into a chunk
    assert not any(_chunk_contains(ch, _tok(p.exfil_address)) for ch in plan.chunks)


@pytest.mark.parametrize("p", [p for p in ATTACKS if p.placement == "boundary_split"],
                         ids=lambda p: p.probe_id)
def test_boundary_injection_straddles_and_is_split(p):
    s, e = token_span_of(p.body, p.injection_text)
    assert (s, e) == p.injection_token_span
    assert s < WINDOW < e  # straddles the 510 boundary
    verb_span = token_span_of(p.body, probes.INJECTIONS[[i["id"] for i in probes.INJECTIONS].index(
        p.injection_id)]["verb"])
    assert verb_span[1] <= STRIDE  # verb half invisible to window 1
    plan = _plan(p.body)
    # no single scored chunk holds the whole injection
    assert not any(_chunk_contains(ch, _tok(p.injection_text)) for ch in plan.chunks)
    # the address half sits past the 510 boundary (so window 0 never sees it) but within window 1's
    # reach (so it is not simply dropped) -- i.e. it is split from the verb, not truncated.
    addr_span = token_span_of(p.body, p.exfil_address)
    assert addr_span[0] >= WINDOW
    windows = probes._selected_windows(p.total_tokens)
    assert probes._any_overlap(windows, addr_span)  # address survives into a scored window
    assert not probes._any_overlap(windows, verb_span) or verb_span[0] < STRIDE  # verb only in window 0


@pytest.mark.parametrize("p", [p for p in ALL if p.kind == "control"], ids=lambda p: p.probe_id)
def test_controls_carry_no_injection(p):
    assert p.injection_text is None and p.exfil_address is None and p.injection_id is None
    low = p.body.lower()
    assert "send_email" not in low and "ignore all previous" not in low and ".invalid" not in low


def test_token_span_helper_is_exact():
    body = "alpha beta gamma delta"
    assert token_span_of(body, "gamma") == (2, 3)


def test_filler_is_sentences_not_word_salad():
    """Probe set v2: long bodies are built from distinct sentences; single-word filler only tops up the tail of
    each padded segment. v1 filled most of a 2,600-token body with random words, which DeBERTa flags whether or
    not an injection is present, so the controls could not separate attack from length."""
    import re

    from redteam.probes import FILLER_MAX, FILLER_WORDS, PROBE_SET_VERSION, _sentence_pool, load_probes

    assert PROBE_SET_VERSION == 2
    pool = set(_sentence_pool())
    filler = set(FILLER_WORDS)
    for p in load_probes():
        sentences = re.findall(r"[^.!?]*[.!?]", p.body)
        known = [s.strip() for s in sentences if s.strip() in pool]
        assert len(known) == len(set(known)), f"{p.probe_id}: a filler sentence repeats"
        # words outside any sentence (trailing top-ups) are few
        loose = [w for w in re.sub(r"[^.!?]*[.!?]", " ", p.body).split() if w in filler]
        n_segments = 3 if p.placement == "middle_overlong" else 4
        assert len(loose) <= FILLER_MAX * n_segments, f"{p.probe_id}: {len(loose)} loose filler words"


class _RealTokenizer:
    """Just enough of a Stage 2 detector for ClassifierService._prepare: the real DeBERTa tokenizer."""
    detector_id, detector_version, threshold, max_len = "probe-tok", "0", 0.5, 512

    def __init__(self):
        self._tok = probes._tokenizer()
        self.cls_id, self.sep_id = self._tok.token_to_id("[CLS]"), self._tok.token_to_id("[SEP]")

    def tokenize(self, text):
        return self._tok.encode(text, add_special_tokens=False).ids

    def tokenize_with_offsets(self, text):
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids, [tuple(o) for o in enc.offsets]

    def infer(self, batch):
        raise AssertionError("not used")


@pytest.fixture(scope="module")
def served_planner():
    from serving.service import ClassifierService, stage2_order

    config = GatewayConfig(shadow_log_path="")
    svc = ClassifierService(_RealTokenizer(), config)

    def selected(body):
        cs = svc._prepare(body)
        assert len(cs.segments) == 1  # probe bodies are one segment: chunk spans are body token offsets
        idx = stage2_order(len(cs.chunks), set(), {})[: config.max_chunks]
        return [(a, b) for i in idx for _, a, b in cs.chunks[i].spans], len(cs.chunks)
    return selected


@pytest.mark.parametrize("p", ATTACKS, ids=lambda p: p.probe_id)
def test_probe_geometry_holds_for_the_served_planner(p, served_planner):
    windows, n_chunks = served_planner(p.body)
    span = tuple(p.injection_token_span)
    assert not probes._covered_by_any(windows, span)  # no scored chunk holds the whole injection
    if p.placement == "middle_overlong":
        assert n_chunks > MAX_CHUNKS and not probes._any_overlap(windows, span)  # dropped middle
    else:
        addr = token_span_of(p.body, p.exfil_address)
        assert probes._any_overlap(windows, addr)  # split from the verb, not dropped
