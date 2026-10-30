"""Round-3 reviewer probes (fakes only; no model weights, no frozen test).

`fixed_*` asserts a round-2 finding no longer reproduces. `r3_*` asserts a round-3 observation, so a pass means it
is reproduced. Reuses the fake specs from tests/training/test_replay_policy_equivalence.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "training"))

import test_replay_policy_equivalence as T  # noqa: E402

from training import replay_policy_equivalence as eq  # noqa: E402


def _patch_specs(monkeypatch):
    holder = {}

    def specs(pointers, intra_op_threads=0):
        holder["specs"] = T.fake_specs(pointers, intra_op_threads)
        return holder["specs"]

    monkeypatch.setattr(eq, "load_specs", specs)
    return holder


def test_fixed_r2_m1_no_cache_sends_every_row_to_the_model_once_per_engine(tmp_path, monkeypatch):
    """Behaviour only (no new report fields): with --no-cache the shared model sees 2 x rows. At 22ec311 it saw
    1 x rows (one call served both engines)."""
    h = _patch_specs(monkeypatch)
    out = tmp_path / "t3.json"
    assert eq.main(["t3", "--out", str(out), "--limit", "3", "--no-cache"]) == 0
    rows = json.loads(out.read_text())["deberta_rows"]["total_scored_per_engine"]
    assert rows > 0
    assert h["specs"]["stage2"].detector.rows_scored == 2 * rows


def test_r3_m1_no_cache_resume_relabels_cached_items_as_uncached(tmp_path, monkeypatch):
    """A --no-cache invocation that resumes a progress file written with the shared cache scores nothing new but
    labels the whole report "cache: none" with the "each engine called the model" note. Per-item records do not
    carry the cache mode, so the report cannot tell."""
    _patch_specs(monkeypatch)
    out = tmp_path / "t3.json"
    assert eq.main(["t3", "--out", str(out), "--limit", "3"]) == 0          # shared cache
    assert eq.main(["t3", "--out", str(out), "--limit", "3", "--no-cache"]) == 0  # resumes, scores nothing
    rep = json.loads(out.read_text())
    assert rep["n_items_this_run"] == 0 and rep["n_items_compared"] == 3 and rep["complete"]
    assert rep["deberta_rows"]["cache"] == "none"
    assert rep["deberta_rows"]["note"].startswith("no cache: each engine called the model")
    assert rep["provenance"]["this_run"]["no_cache"] is True
    progress = out.with_suffix(".progress.jsonl").read_text().splitlines()
    assert all("cache" not in json.dumps(json.loads(line)["code"]) for line in progress)
