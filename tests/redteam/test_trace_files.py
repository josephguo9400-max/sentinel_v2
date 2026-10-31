"""Every emitted event validates against contracts.TraceEvent and is persisted as one JSONL file per run."""
import json

import httpx

import pytest

from contracts import TraceEvent
from redteam import run as cli
from redteam.classifier import FixedClassifier, KeywordClassifier
from redteam.defenders import BenignDefender, GullibleDefender
from redteam.gateway_client import GatewayClient
from redteam.sandbox import run_attempt
from redteam.scenario import load_split
from redteam.trace import RunTrace, group_by_attempt, read_jsonl

ALL = load_split("dev") + load_split("heldout")


def test_every_event_of_every_mode_validates_and_persists(tmp_path):
    t = RunTrace(run_id="all-modes", out_dir=tmp_path)
    unreachable = GatewayClient("http://127.0.0.1:9", timeout_s=0.2)

    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    timing_out = GatewayClient("http://gateway.invalid", transport=httpx.MockTransport(slow))
    for sc in ALL:
        for defender in (BenignDefender(), GullibleDefender()):
            run_attempt(sc, "A", defender, t)
            for clf in (KeywordClassifier(), FixedClassifier("allow"), unreachable, timing_out):
                run_attempt(sc, "B", defender, t, classifier=clf)
                run_attempt(sc, "C", defender, t, classifier=clf)
    unreachable.close()
    timing_out.close()
    assert t.path == tmp_path / "all-modes.jsonl"
    lines = t.path.read_text().splitlines()
    assert len(lines) == len(t.events) > 0
    for line in lines:
        e = TraceEvent.model_validate(json.loads(line))
        assert e.source == "scripted" and e.run_id == "all-modes"
    loaded = read_jsonl(t.path)
    assert [e.model_dump() for e in loaded] == [e.model_dump() for e in t.events]
    types = {e.event_type for e in loaded}
    assert {"tool_proposed", "tool_executed", "tool_blocked", "content_quarantined", "detector_prediction",
            "detector_timeout", "model_response", "attempt_completed"} <= types
    ids = [e.event_id for e in loaded]
    assert len(ids) == len(set(ids))
    for attempt_id, evs in group_by_attempt(loaded).items():
        assert len({e.mode for e in evs}) == 1
        assert sum(e.event_type == "attempt_completed" for e in evs) == 1
        assert evs[-1].event_type == "attempt_completed"


def test_one_file_per_run_and_no_overwrite(tmp_path):
    a, b = RunTrace(out_dir=tmp_path), RunTrace(out_dir=tmp_path)
    assert a.path != b.path
    with pytest.raises(FileExistsError):
        RunTrace(run_id=a.run_id, out_dir=tmp_path)
    with pytest.raises(ValueError):
        RunTrace(run_id="../escape", out_dir=tmp_path)


def test_new_runs_dir_is_self_gitignored(tmp_path):
    d = tmp_path / "runs"
    RunTrace(out_dir=d)
    assert (d / ".gitignore").read_text().strip().endswith("*")


def test_cli_writes_trace_and_rejudges(tmp_path, capsys):
    rc = cli.main(["--scenario", "dev_daily_digest", "--modes", "A,C", "--defender", "gullible",
                   "--classifier", "allow-all", "--runs-dir", str(tmp_path), "--run-id", "cli-test"])
    assert rc == 0
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [o["mode"] for o in out[:2]] == ["A", "C"]
    assert out[0]["attack_success"] and not out[1]["attack_success"] and out[1]["blocked"]
    assert out[2]["summary"]["n_attempts"] == 2
    cli.main(["--judge", str(tmp_path / "cli-test.jsonl")])
    re_out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert re_out[-1]["summary"]["attack_success"] == 1
