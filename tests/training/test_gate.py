"""The release gate (training/gate.py): verdict rules, the report's binding to a bundle_sha256, promotion with an
explicit approval and an audit record, and rollback as a pointer flip.

Every test here uses fixtures or a temporary registry. The gate is never run on data/splits/frozen_test.jsonl:
the owner decides when the final split is read (and the gate then refuses without an explicit approval, which is
asserted below).
"""
import json
from pathlib import Path

import pytest

from training import gate
from training.common import ROOT

BUNDLE = "1" * 64
OTHER = "2" * 64


# ---------------------------------------------------------------- fixtures: synthetic per-item records

def record(rid, label, predicted, *, source="spml", family=None, escalated=True, error=None,
           policy_applied=None, coverage=None, failure_reasons=()):
    parsed = {"error": error} if error else {
        "predicted": predicted, "model_prediction": policy_applied is None and coverage is None,
        "status": "ok", "decision": "flag" if predicted else "allow", "decided_by": "d",
        "policy_applied": policy_applied, "coverage_policy_applied": coverage, "score": 0.9 if predicted else 0.1,
        "stages": ["stage2"] if escalated else ["stage1"], "statuses": ["ok"],
        "failure_reasons": list(failure_reasons), "escalated": escalated, "truncated": False}
    return {"id": str(rid), "label": label, "source": source, "family": family, "stage2_overlap": None,
            "lang": "en", "parsed": parsed, "bundle_sha256": BUNDLE}


def perfect(n_pos=250, n_neg=150, *, benign_hard=120, fpr_hits=0, source="spml"):
    """Enough positives for the recall check and enough benign-hard negatives for the FPR check."""
    rows = [record(f"p{i}", 1, True, source=source) for i in range(n_pos)]
    rows += [record(f"n{i}", 0, False, source=source) for i in range(n_neg)]
    rows += [record(f"bh{i}", 0, i < fpr_hits, source="benign_hard", family="f1") for i in range(benign_hard)]
    return rows


def checks_for(rows, paired=None, comparison=None, policy=None):
    slices = gate.slice_metrics(rows)
    return gate.checks(rows, slices, paired, comparison, policy)


class _Policy:
    def __init__(self, gate_step=None):
        self.gate = gate_step


# ---------------------------------------------------------------- verdict rules

def test_recall_needs_200_positives():
    c = checks_for(perfect(n_pos=199))["recall"]
    assert c["verdict"] == gate.INSUFFICIENT and "199 positives" in c["why"]
    c = checks_for(perfect(n_pos=250))["recall"]
    assert c["verdict"] == gate.INSUFFICIENT and "no active bundle" in c["why"]  # nothing to pair with


def test_recall_is_paired_against_the_active_bundle():
    rows = perfect()
    worse = {"diff": -0.05, "ci95": [-0.09, -0.01], "only_candidate": 0, "only_active": 12, "n_pairs": 250}
    assert checks_for(rows, worse, OTHER)["recall"]["verdict"] == gate.FAIL
    ok = {"diff": -0.004, "ci95": [-0.019, 0.011], "only_candidate": 2, "only_active": 3, "n_pairs": 250}
    assert checks_for(rows, ok, OTHER)["recall"]["verdict"] == gate.PASS
    edge = {"diff": -0.02, "ci95": [-0.02, 0.0], "only_candidate": 0, "only_active": 5, "n_pairs": 250}
    assert checks_for(rows, edge, OTHER)["recall"]["verdict"] == gate.PASS  # -2 pt exactly is allowed


def test_paired_recall_counts_pairs_over_shared_items():
    cand = [record("a", 1, True), record("b", 1, False), record("c", 1, True), record("d", 0, True)]
    active = [record("a", 1, True), record("b", 1, True), record("c", 1, False), record("x", 1, True)]
    out = gate.paired_recall(cand, active)
    assert out["n_pairs"] == 3 and out["only_candidate"] == 1 and out["only_active"] == 1
    assert out["diff"] == pytest.approx(0.0)


def test_benign_hard_fpr_rule():
    c = checks_for(perfect(benign_hard=99))["benign_hard_fpr"]
    assert c["verdict"] == gate.INSUFFICIENT and "99 benign-hard" in c["why"]
    c = checks_for(perfect(benign_hard=120, fpr_hits=0))["benign_hard_fpr"]
    assert c["verdict"] == gate.PASS
    c = checks_for(perfect(benign_hard=120, fpr_hits=12))["benign_hard_fpr"]  # 10%: upper bound above 5%
    assert c["verdict"] == gate.FAIL and "upper bound" in c["why"]


def test_compute_saved_only_applies_to_a_policy_with_a_gate_step():
    rows = perfect()  # every item escalated
    assert checks_for(rows, policy=_Policy(None))["compute_saved"]["verdict"] == gate.NOT_EVALUATED
    assert checks_for(rows, policy=_Policy("a gate"))["compute_saved"]["verdict"] == gate.FAIL
    half = [record(f"p{i}", 1, True, escalated=i % 4 == 0) for i in range(200)]
    c = checks_for(half, policy=_Policy("a gate"))["compute_saved"]
    assert c["verdict"] == gate.PASS and c["escalation_share"]["value"] == 0.25


def test_error_rate_counts_faults_and_separates_load_shedding():
    rows = perfect(n_pos=200, n_neg=100, benign_hard=100)
    c = checks_for(rows)["error_rate"]
    assert c["verdict"] == gate.PASS and c["error_rate"]["value"] == 0.0
    rows = [record(f"p{i}", 1, True, failure_reasons=["timeout"] if i < 5 else []) for i in range(200)]
    c = checks_for(rows)["error_rate"]
    assert c["verdict"] == gate.FAIL and c["failures"] == {"timeout": 5}
    rows = [record(f"p{i}", 1, True, failure_reasons=["overloaded"] if i < 50 else []) for i in range(200)]
    c = checks_for(rows)["error_rate"]
    assert c["verdict"] == gate.PASS and c["shed"] == 50  # shedding is a capacity signal, not a detector fault


def test_overall_verdict():
    assert gate.verdict_of({"a": {"verdict": gate.PASS}}) == gate.PASS
    assert gate.verdict_of({"a": {"verdict": gate.PASS}, "b": {"verdict": gate.INSUFFICIENT}}) == gate.INSUFFICIENT
    assert gate.verdict_of({"a": {"verdict": gate.NOT_EVALUATED}}) == gate.INSUFFICIENT
    assert gate.verdict_of({"a": {"verdict": gate.FAIL}, "b": {"verdict": gate.INSUFFICIENT}}) == gate.FAIL


def test_as_served_counts_policy_and_coverage_decisions():
    """A failure-policy or coverage-policy decision is the decision the user got (gate_view as_served)."""
    rows = [record("a", 1, True, policy_applied="fail_closed"), record("b", 1, False, policy_applied="fail_open"),
            record("c", 0, True, coverage="flag")]
    m = gate.slice_metrics(rows)["overall"]
    assert m["n"] == 3 and m["confusion"]["tp"] == 1 and m["confusion"]["fn"] == 1 and m["confusion"]["fp"] == 1


def test_errors_are_excluded_from_the_metrics():
    rows = perfect(n_pos=200, n_neg=100, benign_hard=100) + [record("e", 1, None, error="transport: X")]
    m = gate.slice_metrics(rows)["overall"]
    assert m["n"] == 400


# ---------------------------------------------------------------- the frozen test is the owner's call

def test_gate_refuses_the_frozen_test_without_an_explicit_approval(tmp_path):
    args = gate.main.__globals__["argparse"].Namespace(
        registry=ROOT / "artifacts" / "registry.json", split="frozen_test", splits_dir=ROOT / "data" / "splits",
        out=tmp_path / "r.json", max_seconds=None, limit=1, active_report=None, frozen_test_approved_by=None)
    with pytest.raises(SystemExit) as exc:
        gate.evaluate(args)
    assert "frozen_test" in str(exc.value) and "--frozen-test-approved-by" in str(exc.value)
    assert not (tmp_path / "r.json").exists()


def test_no_committed_gate_report_used_the_frozen_test():
    """This milestone evaluates the gate on fixtures and val only; the owner decides when the final split is read."""
    for path in sorted((ROOT / "reports").rglob("gate_*.json")):
        rep = json.loads(path.read_text())
        assert rep["run"]["split"] != "frozen_test", f"{path} is a gate run on the frozen test"
        assert rep["run"]["frozen_test_approved_by"] is None
    log = ROOT / "artifacts" / "audit_log.jsonl"
    if log.is_file():
        for line in log.read_text().splitlines():
            if line.strip():
                assert json.loads(line)["action"] != "gate_on_frozen_test"


# Every non-test Python file that mentions the frozen test. All but training/gate.py predate the gate and are
# training-side analysis (split building, leakage checks, Stage 1 training, harness --split choices), not release
# decisions. A new file on this list needs a reason.
FROZEN_TEST_MENTIONS = {
    "training/benign_hard.py", "training/build_splits.py", "training/cascade_scores.py", "training/check_leakage.py",
    "training/common.py", "training/dilution_curve.py", "training/eval_harness.py", "training/gate.py",
    "training/publish_stage1.py", "training/replay.py", "training/train_stage1.py", "training/window_scores.py",
}
MILESTONE_BASE = ("origin/milestone/cascade-and-shadow", "milestone/cascade-and-shadow", "36ce683")


def test_no_new_file_mentions_the_frozen_test():
    """Works without git (review r1 n6): the set of non-test Python files that mention frozen_test is pinned."""
    found = {str(p.relative_to(ROOT)) for top in ("training", "serving", "redteam", "scripts", "benchmarks",
                                                  "contracts", "web")
             for p in (ROOT / top).rglob("*.py") if (ROOT / top).is_dir() and "frozen_test" in p.read_text()}
    assert found <= FROZEN_TEST_MENTIONS, f"new readers of the frozen test: {sorted(found - FROZEN_TEST_MENTIONS)}"


def test_the_gate_is_the_only_new_reader_of_the_frozen_test():
    """No file added or modified since the cascade-and-shadow milestone adds a line mentioning
    data/splits/frozen_test.jsonl, except training/gate.py.

    Earlier readers are training-side analysis that predates the gate (e.g. training/eval_harness.py's --split
    choice, training/build_splits.py, training/check_leakage.py); they are not release decisions and are left
    alone. What this guards is that the gate did not grow a second reader alongside it. The base falls back from the
    remote branch to the local branch to the merge-base commit; without any of them (a shallow clone) the git-free
    test above still applies, and this one skips with the reason.
    """
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)

    base = next((b for b in MILESTONE_BASE if git("rev-parse", "--verify", "--quiet", f"{b}^{{commit}}").returncode
                 == 0), None)
    if base is None:
        pytest.skip(f"none of {MILESTONE_BASE} is available (shallow clone?); "
                    f"test_no_new_file_mentions_the_frozen_test still applies")
    diff = git("diff", "--diff-filter=AM", "-U0", f"{base}...HEAD", "--", "*.py")
    assert diff.returncode == 0, diff.stderr
    offenders, current = set(), None
    for line in diff.stdout.splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and current and "frozen_test" in line:
            if current != "training/gate.py" and not current.startswith("tests/"):
                offenders.add(current)
    assert not offenders, f"these add a reference to the frozen test: {sorted(offenders)}"


# ---------------------------------------------------------------- report binding, promote, rollback

def fake_report(tmp_path: Path, bundle_sha: str, policy_path: str, verdict: str = gate.PASS,
                split: str = "val") -> Path:
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({"verdict": verdict, "promotes": False,
                             "bundle": {"bundle_sha256": bundle_sha, "policy_path": policy_path},
                             "run": {"split": split, "frozen_test_approved_by": None, "complete": True,
                                     "limit": None, "n_items_in_split": 3214, "n_items_evaluated": 3214}}))
    return p


def tmp_registry(tmp_path: Path, active: str, previous: str | None = None, status="pending_gate") -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"schema": "sentinel-registry-v2", "active": active, "previous": previous,
                             "capabilities_root": "capabilities", "status": status}))
    return p


ACTIVE = "policies/d23-deberta-decides/1/policy.json"
CANDIDATE = "policies/cascade-tfidf-allow-only/1/policy.json"


def bundle_sha(rel: str) -> str:
    from serving.policy import bundle_sha256
    from serving.registry import load_policy_file
    return bundle_sha256(load_policy_file(ROOT / "artifacts", rel))


def test_promote_requires_a_pass_and_records_an_audit_entry(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE)
    log = tmp_path / "audit.jsonl"
    rep = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE, verdict=gate.INSUFFICIENT)
    with pytest.raises(SystemExit) as exc:
        gate.promote(reg, rep, "owner", log=log)
    assert "INSUFFICIENT_EVIDENCE" in str(exc.value) and "nothing is promoted" in str(exc.value)
    assert json.loads(reg.read_text())["active"] == ACTIVE  # untouched
    assert not log.exists()

    rep = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE, verdict=gate.PASS)
    entry = gate.promote(reg, rep, "owner <o@example.com>", log=log)
    state = json.loads(reg.read_text())
    assert state["active"] == CANDIDATE and state["previous"] == ACTIVE and state["status"] == "approved"
    assert entry["actor"] == "owner <o@example.com>" and entry["action"] == "promote"
    assert entry["bundle_sha256"] == bundle_sha(CANDIDATE) and entry["previous_bundle_sha256"] == bundle_sha(ACTIVE)
    assert entry["gate_verdict"] == gate.PASS and entry["gate_split"] == "val"
    assert [json.loads(line)["action"] for line in log.read_text().splitlines()] == ["promote"]


def test_promote_refuses_a_report_bound_to_another_bundle(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE)
    rep = fake_report(tmp_path, "9" * 64, CANDIDATE)
    with pytest.raises(SystemExit) as exc:
        gate.promote(reg, rep, "owner", log=tmp_path / "a.jsonl")
    assert "re-run the gate" in str(exc.value)
    assert json.loads(reg.read_text())["active"] == ACTIVE


def test_promote_requires_an_approver(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE)
    rep = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE)
    with pytest.raises(SystemExit) as exc:
        gate.promote(reg, rep, "   ", log=tmp_path / "a.jsonl")
    assert "--approved-by is required" in str(exc.value)


def test_an_override_is_recorded(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE)
    log = tmp_path / "audit.jsonl"
    rep = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE, verdict=gate.INSUFFICIENT)
    entry = gate.promote(reg, rep, "owner", log=log, allow_not_pass=True)
    assert entry["override_not_pass"] is True and entry["gate_verdict"] == gate.INSUFFICIENT


def test_promote_and_rollback_keep_the_registry_valid(tmp_path):
    """The schema and comment of the file survive a pointer change (it must still load afterwards)."""
    reg = tmp_registry(tmp_path, ACTIVE)
    raw = json.loads(reg.read_text())
    raw["comment"] = "keep me"
    reg.write_text(json.dumps(raw))
    rep = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE)
    gate.promote(reg, rep, "owner", log=tmp_path / "a.jsonl")
    after = json.loads(reg.read_text())
    assert after["schema"] == "sentinel-registry-v2" and after["comment"] == "keep me"
    gate.read_registry_v2_state(reg)
    gate.rollback(reg, "owner", log=tmp_path / "a.jsonl")
    assert json.loads(reg.read_text())["schema"] == "sentinel-registry-v2"
    gate.read_registry_v2_state(reg)


def test_rollback_is_a_pointer_flip(tmp_path):
    reg = tmp_registry(tmp_path, CANDIDATE, ACTIVE, status="approved")
    log = tmp_path / "audit.jsonl"
    entry = gate.rollback(reg, "owner", log=log)
    state = json.loads(reg.read_text())
    assert state["active"] == ACTIVE and state["previous"] == CANDIDATE
    assert entry["action"] == "rollback" and entry["bundle_sha256"] == bundle_sha(ACTIVE)
    assert entry["previous_bundle_sha256"] == bundle_sha(CANDIDATE)
    gate.rollback(reg, "owner", log=log)  # flipping back is another audited rollback
    assert json.loads(reg.read_text())["active"] == CANDIDATE
    assert len(log.read_text().splitlines()) == 2


def test_rollback_without_a_previous_is_refused(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE, None)
    with pytest.raises(SystemExit) as exc:
        gate.rollback(reg, "owner", log=tmp_path / "a.jsonl")
    assert "nothing to roll back" in str(exc.value)


def test_the_gate_never_promotes_by_itself():
    """`evaluate` has no path that writes a registry or the audit log."""
    src = (ROOT / "training" / "gate.py").read_text()
    body = src[src.index("def evaluate(args)"):src.index("def main(")]
    assert "write_text" not in body and "promote(" not in body
    assert "append_audit" in body  # only to record that the frozen test was read
    assert "\"promotes\": False" in src


def test_committed_gate_report_is_bound_to_the_served_bundle():
    """The val dry run is bound to the D23 bundle as it was under the engine version it ran on (1.0.0). After the
    engine bump to 1.1.0 (review r1 fixes) the same policy is a new bundle, so that report no longer describes the
    served bundle and promote refuses it (D25.2: no verdict carries over)."""
    path = ROOT / "reports" / "policy-runtime" / "gate_val_d23.json"
    if not path.is_file():
        pytest.skip("no committed gate run")
    rep = json.loads(path.read_text())
    from serving.engine_version import ENGINE_VERSION
    from serving.policy import bundle_sha256
    from serving.registry import load_policy_file, read_registry_v2
    state = read_registry_v2(ROOT / "artifacts" / "registry.json")
    policy = load_policy_file(ROOT / "artifacts", state["active"])
    assert rep["bundle"]["bundle_sha256"] == bundle_sha256(policy, rep["bundle"]["engine_version"])
    assert rep["promotes"] is False and rep["run"]["split"] == "val"
    # val has 92 benign-hard samples (< 100), so a val run can never be a PASS.
    assert rep["checks"]["benign_hard_fpr"]["verdict"] == gate.INSUFFICIENT
    assert rep["verdict"] == gate.INSUFFICIENT
    if rep["bundle"]["engine_version"] != ENGINE_VERSION:
        assert rep["bundle"]["bundle_sha256"] != bundle_sha(state["active"])


# ---------------------------------------------------------------- promote refusals (review r1 m2, m3)

def test_promote_of_the_already_active_bundle_is_refused_and_keeps_previous(tmp_path):
    """Review r1 m2 (probe test_r3_...): promoting the active bundle used to set previous = active and lose the real
    rollback target. It is refused, even with --allow-not-pass, and nothing is written."""
    reg = tmp_registry(tmp_path, ACTIVE, CANDIDATE)
    log = tmp_path / "audit.jsonl"
    rep = fake_report(tmp_path, bundle_sha(ACTIVE), ACTIVE, verdict=gate.INSUFFICIENT)
    before = reg.read_text()
    with pytest.raises(SystemExit) as exc:
        gate.promote(reg, rep, "owner", log=log, allow_not_pass=True)
    assert "already the active bundle" in str(exc.value)
    assert reg.read_text() == before and not log.exists()
    state = json.loads(reg.read_text())
    assert state["active"] == ACTIVE and state["previous"] == CANDIDATE


@pytest.mark.parametrize("run_over,why", [
    ({"complete": False, "n_items_evaluated": 10}, "incomplete"),
    ({"limit": 50, "n_items_in_split": 50, "n_items_evaluated": 50}, "incomplete"),
    ({"complete": None}, "incomplete"),
    ({"split": "train"}, "not a gate split"),
    ({"split": None}, "not a gate split"),
])
def test_promote_refuses_an_incomplete_or_non_gate_run(tmp_path, run_over, why):
    """Review r1 m3 (probe test_r3b_...): a PASS on a partial run, a --limit run or a non-gate split is refused,
    and --allow-not-pass does not override it."""
    reg = tmp_registry(tmp_path, ACTIVE)
    path = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE, verdict=gate.PASS)
    rep = json.loads(path.read_text())
    rep["run"].update(run_over)
    path.write_text(json.dumps(rep))
    for allow in (False, True):
        with pytest.raises(SystemExit) as exc:
            gate.promote(reg, path, "owner", log=tmp_path / "a.jsonl", allow_not_pass=allow)
        assert why in str(exc.value)
    assert json.loads(reg.read_text())["active"] == ACTIVE and not (tmp_path / "a.jsonl").exists()


def test_promote_accepts_a_complete_frozen_test_report(tmp_path):
    reg = tmp_registry(tmp_path, ACTIVE)
    path = fake_report(tmp_path, bundle_sha(CANDIDATE), CANDIDATE, verdict=gate.PASS, split="frozen_test")
    entry = gate.promote(reg, path, "owner", log=tmp_path / "a.jsonl")
    assert entry["gate_split"] == "frozen_test" and json.loads(reg.read_text())["active"] == CANDIDATE


# ---------------------------------------------------------------- frozen-test audit before the run (review r1 m4)

def _frozen_args(tmp_path):
    import argparse
    return argparse.Namespace(split="frozen_test", frozen_test_approved_by="owner", audit_log=tmp_path / "audit.jsonl",
                              registry=tmp_path / "registry.json", out=tmp_path / "gate_frozen.json",
                              max_seconds=None, limit=None, splits_dir=tmp_path, active_report=None)


def test_a_frozen_test_run_is_audited_before_it_starts_even_if_it_crashes(tmp_path, monkeypatch):
    """The split is never read here: run_bundle is replaced. The "started" entry must exist when run_bundle is
    entered, and a crash adds "failed"."""
    args = _frozen_args(tmp_path)
    seen = {}

    def crashing_run_bundle(*a, **k):
        seen["log_at_start"] = [json.loads(line) for line in args.audit_log.read_text().splitlines()]
        raise MemoryError("simulated OOM")

    monkeypatch.setattr(gate, "run_bundle", crashing_run_bundle)
    with pytest.raises(MemoryError):
        gate.evaluate(args)
    assert [(e["action"], e["status"]) for e in seen["log_at_start"]] == [("gate_on_frozen_test", "started")]
    entries = [json.loads(line) for line in args.audit_log.read_text().splitlines()]
    assert [e["status"] for e in entries] == ["started", "failed"]
    assert entries[0]["actor"] == "owner" and entries[1]["error"] == "MemoryError"


def test_a_completed_frozen_test_run_has_a_started_and_a_completed_entry(tmp_path, monkeypatch):
    args = _frozen_args(tmp_path)
    meta = {"split": "frozen_test", "complete": True, "n_items_evaluated": 0, "n_items_in_split": 0}
    monkeypatch.setattr(gate, "run_bundle", lambda *a, **k: ([], meta, object(), 0.0, 0))
    monkeypatch.setattr(gate, "build_report", lambda *a, **k: {"verdict": gate.INSUFFICIENT, "run": meta,
                                                               "bundle": {"bundle_sha256": BUNDLE}})
    gate.evaluate(args)
    entries = [json.loads(line) for line in args.audit_log.read_text().splitlines()]
    assert [e["status"] for e in entries] == ["started", "completed"]
    assert entries[1]["bundle_sha256"] == BUNDLE and entries[1]["verdict"] == gate.INSUFFICIENT


def test_a_val_run_writes_no_audit_entry(tmp_path, monkeypatch):
    args = _frozen_args(tmp_path)
    args.split, args.frozen_test_approved_by = "val", None
    meta = {"split": "val", "complete": True, "n_items_evaluated": 0, "n_items_in_split": 0}
    monkeypatch.setattr(gate, "run_bundle", lambda *a, **k: ([], meta, object(), 0.0, 0))
    monkeypatch.setattr(gate, "build_report", lambda *a, **k: {"verdict": gate.INSUFFICIENT, "run": meta,
                                                               "bundle": {"bundle_sha256": BUNDLE}})
    gate.evaluate(args)
    assert not args.audit_log.exists()


def test_benign_hard_why_states_the_comparison_that_held():
    """Review r1 n5: a FAIL says '>' rather than always '<= 5%'."""
    fail = checks_for(perfect(benign_hard=120, fpr_hits=30))["benign_hard_fpr"]
    assert fail["verdict"] == gate.FAIL and " > " in fail["why"] and "<=" not in fail["why"]
    ok = checks_for(perfect(benign_hard=120, fpr_hits=0))["benign_hard_fpr"]
    assert ok["verdict"] == gate.PASS and " <= " in ok["why"]
