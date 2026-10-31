"""The gate reads the as-served view: failure-policy decisions count as what the user received."""
import json

from training.eval_harness import add_as_served, as_served, evaluate, main


class _Backend:
    """Item 0: model flags an injection. Item 1: gateway timed out on an injection and fail_open allowed it.
    Item 2: model allows a benign item. Item 3: gateway timed out on a benign item and fail_closed flagged it."""

    def __init__(self):
        self.replies = [
            {"predicted": 1, "model_prediction": True, "truncated": False, "latency_ms": 1.0},
            {"model_prediction": False, "status": "timeout", "policy_applied": "fail_open", "decision": "allow",
             "truncated": True, "latency_ms": 1.0},
            {"predicted": 0, "model_prediction": True, "truncated": False, "latency_ms": 1.0},
            {"model_prediction": False, "status": "timeout", "policy_applied": "fail_closed", "decision": "flag",
             "truncated": True, "latency_ms": 1.0},
        ]

    def predict(self, text, request_id):
        return self.replies.pop(0)

    def describe(self):
        return {"backend": "fake"}


ROWS = [{"id": str(i), "text": "t", "label": y, "source": "s", "lang": "en"} for i, y in enumerate([1, 1, 0, 0])]


def test_model_view_excludes_policy_decisions_but_as_served_counts_them():
    res = evaluate(ROWS, _Backend())
    o = res["slices"]["overall"]
    assert o["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert o["recall"]["value"] == 1.0
    served = o["as_served"]
    assert served["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert served["recall"]["numerator"] == 1 and served["recall"]["denominator"] == 2
    assert served["fpr"]["numerator"] == 1 and served["fpr"]["denominator"] == 2
    assert res["gate_view"] == "as_served"


def test_add_as_served_cli_is_idempotent(tmp_path):
    res = evaluate(ROWS, _Backend())
    for m in res["slices"].values():
        m.pop("as_served")
    p = tmp_path / "r.json"
    p.write_text(json.dumps(res))
    assert main(["--add-as-served", str(p)]) == 0
    first = json.loads(p.read_text())
    assert main(["--add-as-served", str(p)]) == 0
    assert json.loads(p.read_text()) == first
    assert first["slices"]["overall"]["as_served"]["confusion"]["fn"] == 1


def test_as_served_equals_model_view_without_policy_decisions():
    m = {"confusion": {"tp": 3, "fp": 1, "fn": 2, "tn": 4}, "policy_decisions": {"by_decision_and_label": {}}}
    assert as_served(m)["confusion"] == m["confusion"]
