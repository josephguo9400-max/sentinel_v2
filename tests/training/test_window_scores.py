"""Small-window rescoring (training/window_scores.py), the dilution curve and the window/two-pass evaluation."""
import json

import pytest

from training import dilution_curve as dc
from training import window_eval as we
from training import window_scores as ws
from training.common import ROOT, read_jsonl, sha256_file

DEBERTA = ROOT / "models" / "deberta-pi-v2" / "onnx" / "model.onnx"
CAS_VAL = ROOT / "training" / "results" / "cascade_scores_val.jsonl"
MAX_BYTES = 5 * 1024 * 1024


# ------------------------------------------------------------------ windowing

def test_stride_is_the_gateways_for_510_and_half_the_window_otherwise():
    assert ws.stride_for(510) == 384
    assert [ws.stride_for(w) for w in (256, 128, 64)] == [128, 64, 32]
    with pytest.raises(ValueError):
        ws.stride_for(1)


@pytest.mark.parametrize("n", [0, 1, 63, 64, 65, 100, 511, 1000])
@pytest.mark.parametrize("w", [64, 128, 510])
def test_window_rows_cover_every_token_and_end_at_the_tail(n, w):
    ids = list(range(100, 100 + n))
    rows = ws.window_rows(ids, w, ws.stride_for(w), 1, 2)
    assert all(r[0] == 1 and r[-1] == 2 and len(r) <= w + 2 for r in rows)
    seen = {t for r in rows for t in r[1:-1]}
    assert seen == set(ids)
    if n:
        assert rows[-1][-2] == ids[-1]
    if n > w:
        assert all(len(r) == w + 2 for r in rows)  # equal lengths: batching a request needs no padding


def test_510_windows_are_the_gateway_chunks():
    from serving.encoding import chunk_ids

    ids = list(range(5, 1500))
    assert ws.window_rows(ids, 510, 384, 1, 2) == chunk_ids(ids, 1, 2, 512, 384)


def test_cascade_cache_reuse_rule():
    c = {"deb_tokens": 100, "chunks": [[0, 9, 0.25, 0.1]]}
    assert ws.from_cascade_cache(c, 510) == [0.25]
    assert ws.from_cascade_cache(c, 128) == [0.25]  # one window holding the whole request: same input
    assert ws.from_cascade_cache(c, 64) is None  # needs the model
    long = {"deb_tokens": 900, "chunks": [[0, 1, 0.1, 0], [1, 2, 0.7, 0]]}
    assert ws.from_cascade_cache(long, 510) == [0.1, 0.7]
    assert ws.from_cascade_cache(long, 256) is None
    with pytest.raises(ValueError):
        ws.max_score([])


# ------------------------------------------------------------------ dilution composition

def test_compose_places_the_injection_exactly():
    inj, ctx = [7, 8, 9], list(range(100, 120))
    ids, span = dc.compose(inj, ctx, 10, "start")
    assert ids == inj + ctx[:10] and span == (0, 3)
    ids, span = dc.compose(inj, ctx, 10, "end")
    assert ids == ctx[:10] + inj and span == (10, 13)
    ids, span = dc.compose(inj, ctx, 11, "middle")
    assert ids == ctx[:5] + inj + ctx[5:11] and span == (5, 8)
    assert dc.compose(inj, ctx, 0, "middle") == (inj, (0, 3))
    with pytest.raises(ValueError):
        dc.compose(inj, ctx, 21, "start")
    with pytest.raises(ValueError):
        dc.compose(inj, ctx, 5, "nowhere")


def test_stratified_selection_fills_short_sources_from_spml():
    el = ([{"id": f"a{i}", "source": "spml"} for i in range(50)] + [{"id": f"b{i}", "source": "jackhhao"} for i in range(5)]
          + [{"id": f"c{i}", "source": "deepset"} for i in range(30)])
    got = dc.select_val_injections(el, per_source=10, total=30)
    by = {s: sum(r["source"] == s for r in got) for s in dc.SOURCES}
    assert by == {"spml": 15, "jackhhao": 5, "deepset": 10}
    assert got == dc.select_val_injections(list(reversed(el)), per_source=10, total=30)  # order-independent


def test_items_enumerate_every_length_position_and_context_alone():
    injections = [{"id": "x", "group": "g", "ids": [1, 2]}, {"id": "y", "group": "g", "ids": [3]}]
    contexts = {(i, p): list(range(10, 10 + dc.CONTEXT_TOKENS)) for i in ("x", "y") for p in dc.POOLS}
    it = dc.items(injections, contexts)
    per = 1 + (len(dc.LENGTHS) - 1) * len(dc.POSITIONS)
    assert sum(x["kind"] == "injection" for x in it) == 2 * len(dc.POOLS) * per
    assert sum(x["kind"] == "context" for x in it) == 2 * len(dc.POOLS) * (len(dc.LENGTHS) - 1)
    x = next(x for x in it if x["key"] == "x|newsletter|510|end")
    assert len(x["ids"]) == 512 and x["span"] == [510, 512]


# ------------------------------------------------------------------ evaluation logic

def test_model_tokens_count_specials_and_overlap():
    assert we.model_tokens(50, 64) == 52
    assert we.model_tokens(100, 64) == 3 * 66  # starts 0, 32, 36
    assert we.model_tokens(510, 510) == 512
    assert we.model_tokens(900, 510) == 3 * 512  # starts 0, 384, 390


def test_two_pass_flag():
    assert we.two_pass_flag(0.6, 0.0, 10, 128, 0.9) == 1  # pass 1 alone flags
    assert we.two_pass_flag(0.1, 0.95, 300, 128, 0.9) == 1  # pass 2 flags a long request
    assert we.two_pass_flag(0.1, 0.95, 128, 128, 0.9) == 0  # not longer than N: pass 2 does not run
    assert we.two_pass_flag(0.1, 0.85, 300, 128, 0.9) == 0  # below the pass-2 threshold


def test_paired_difference_counts_discordant_items():
    d = we.paired([1, 1, 0, 0, 1], [1, 0, 0, 0, 0])
    assert d["only_first"] == 2 and d["only_second"] == 0 and d["diff"] == pytest.approx(0.4)
    assert d["ci95"][0] <= 0.4 <= d["ci95"][1]
    assert we.paired([], []) is None


def _r(i, label, n, s510, s_small, source="s"):
    r = {"id": str(i), "label": label, "n": n, "source": source, "family": None}
    for w in we.WINDOWS:
        s = s510 if (w == 510 or n <= w) else s_small
        r[f"s{w}"], r[f"k{w}"] = s, len(ws.window_starts(n, w, ws.stride_for(w)))
        r[f"f{w}"] = int(s >= 0.5)
    return r


def test_selection_never_buys_recall_with_benign_hard_false_positives():
    # A long injection pass 2 recovers at t2 <= 0.9; a long benign_hard item pass 2 would flag at t2 <= 0.8.
    rows = [_r(0, 1, 400, 0.01, 0.95), _r(1, 0, 400, 0.01, 0.85, "benign_hard")]
    rows += [_r(10 + i, 0, 20, 0.01, 0.01) for i in range(300)]
    g = we.grid(rows)
    best = we.select(g)
    assert best["recall"]["numerator"] == 1
    assert best["benign_hard_fp"] == 0 and best["t2"] > 0.85
    assert all(not x["admissible"] for x in g if x["t2"] <= 0.85 and x["N"] < 400)


def test_buckets():
    assert we.bucket(0) == "0-64" and we.bucket(64) == "0-64" and we.bucket(65) == "65-128"
    assert we.bucket(510) == "257-510" and we.bucket(511) == ">510"


# ------------------------------------------------------------------ committed outputs

@pytest.mark.parametrize("name", ws.SETS)
def test_committed_window_scores_are_complete_and_small(name):
    p = ws.out_path(name)
    if not p.exists():
        pytest.skip("window scores not computed")
    assert p.stat().st_size < MAX_BYTES
    rows = we.load(name)  # raises if any W is missing for any item
    for r in rows:
        assert all(r[f"k{w}"] == len(ws.window_starts(r["n"], w, ws.stride_for(w))) for w in ws.WINDOWS)
        for w in ws.WINDOWS:
            if r["n"] <= w:
                assert r[f"s{w}"] == r["s510"]  # single window: identical input


@pytest.mark.skipif(not we.OUT.exists(), reason="window eval not run")
def test_committed_window_eval_is_reproducible():
    committed = json.loads(we.OUT.read_text())
    for name, d in committed["inputs"].items():
        assert d["sha256"] == sha256_file(ROOT / d["file"])
    assert json.loads(json.dumps(we.build())) == committed
    assert committed["two_pass"]["chosen_on"] == "val"


@pytest.mark.skipif(not dc.OUT.exists(), reason="dilution curve not summarized")
def test_committed_dilution_curve_is_reproducible():
    committed = json.loads(dc.OUT.read_text())
    assert committed["scores_file_sha256"] == sha256_file(dc.CACHE)
    assert dc.CACHE.stat().st_size < MAX_BYTES
    got = dc.summarize([int(w) for w in committed["by_window"]])
    assert json.loads(json.dumps(got)) == committed


@pytest.mark.skipif(not DEBERTA.exists(), reason="DeBERTa model not fetched")
@pytest.mark.skipif(not ws.out_path("val").exists(), reason="window scores not computed")
def test_cached_window_scores_match_a_fresh_computation_on_a_sample():
    """Covers both cache paths: model records, and cascade-cache copies for n <= W (run under the CPU lock)."""
    from training.cascade_scores import DebertaScorer

    texts = {r["id"]: r["text"] for r in ws.load_set("val")}
    recs = list(read_jsonl(ws.out_path("val")))
    model = [d for d in recs if d["from"] == "model" and d["W"] == 128 and d["n_tokens"] < 300][:2]
    copied = [d for d in recs if d["from"] == "cascade_cache" and d["W"] == 64 and 20 < d["n_tokens"] <= 64][:2]
    assert model and copied
    deb = DebertaScorer(threads=1)
    for d in model + copied:
        ids = deb.tok.encode(texts[d["id"]], add_special_tokens=False).ids
        got = [deb.infer_one(x) for x in ws.window_rows(ids, d["W"], d["stride"], deb.cls_id, deb.sep_id)]
        assert got == pytest.approx(d["scores"], abs=1e-5)
