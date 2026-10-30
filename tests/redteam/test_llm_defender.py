"""LLM defender: tool-use loop, budgets, record/replay, and a keyless CLI failure.

No API key and no network: a FakeLLMClient returns scripted turns, and replay re-drives from the
recorded transcript.
"""
import json

import pytest

from redteam import llm_run
from redteam.judge import judge_attempt
from redteam.llm_defender import (AnthropicLLMClient, BudgetExceeded, FakeLLMClient, LLMBudget,
                                   LLMDefender, LLMResponse, ReplayLLMClient, ToolCall, Usage,
                                   replay_recorded, run_recorded)
from redteam.sandbox import run_attempt
from redteam.scenario import find_scenario
from redteam.trace import RunTrace


def read_turn(email_id):
    return LLMResponse(tool_calls=(ToolCall(f"t-{email_id}", "read_email", {"email_id": email_id}),),
                       stop_reason="tool_use", usage=Usage(100, 20))


def list_turn():
    return LLMResponse(tool_calls=(ToolCall("t-list", "list_emails", {}),), stop_reason="tool_use",
                       usage=Usage(80, 10))


def final_turn(text):
    return LLMResponse(text=text, usage=Usage(120, 30))


@pytest.fixture
def digest():
    return find_scenario("dev_daily_digest")


@pytest.fixture
def manager():
    return find_scenario("dev_send_summary_to_manager")


# ----------------------------------------------------------------------- the loop

def test_loop_reads_then_answers(digest):
    turns = [list_turn(), read_turn("e1"), final_turn("Here is your summary of today's emails.")]
    client = FakeLLMClient(turns)
    defender = LLMDefender(client, source="live")
    trace = RunTrace(out_dir=None)
    res = run_attempt(digest, "A", defender, trace)
    assert res.status == "completed"
    assert "summary" in res.final_answer.lower()
    assert defender.steps_used == 3 and defender.tokens_used == 80 + 10 + 100 + 20 + 120 + 30
    evs = trace.attempt_events(res.attempt_id)
    assert [e.event_type for e in evs if e.event_type == "tool_executed"].count("tool_executed") == 2
    assert all(e.source == "live" for e in evs)


def test_tool_call_maps_to_sandbox_and_can_send(manager):
    """A send proposed by the model is executed by the sandbox; mode A has no firewall."""
    turns = [LLMResponse(tool_calls=(ToolCall("s1", "send_email",
                                              {"to": "priya.nair@example.com", "subject": "s", "body": "b"}),),
                         stop_reason="tool_use", usage=Usage(50, 10)),
             final_turn("Sent.")]
    trace = RunTrace(out_dir=None)
    res = run_attempt(manager, "A", LLMDefender(FakeLLMClient(turns), source="live"), trace)
    assert res.status == "completed" and len(res.outbox) == 1


def test_defender_satisfies_protocol():
    from redteam.sandbox import Defender
    assert isinstance(LLMDefender(FakeLLMClient([])), Defender)


# ----------------------------------------------------------------------- budgets

def test_step_budget_is_a_timeout(digest):
    # Never produces a final (always asks for a tool): step budget must stop it.
    turns = [list_turn()] * 10
    defender = LLMDefender(FakeLLMClient(turns), budget=LLMBudget(max_steps=3), source="live")
    trace = RunTrace(out_dir=None)
    res = run_attempt(digest, "A", defender, trace)
    assert res.status == "timeout" and "step budget" in res.error
    assert defender.steps_used == 3


def test_token_budget_is_a_timeout(digest):
    turns = [LLMResponse(tool_calls=(ToolCall("x", "list_emails", {}),), stop_reason="tool_use",
                         usage=Usage(10_000, 10_000)), final_turn("done")]
    defender = LLMDefender(FakeLLMClient(turns), budget=LLMBudget(max_tokens=5_000), source="live")
    res = run_attempt(digest, "A", defender, RunTrace(out_dir=None))
    assert res.status == "timeout" and "token budget" in res.error


def test_wall_time_budget(digest):
    # Injected clock jumps 100 s per model turn; the defender's own wall-time budget (not the sandbox
    # deadline) must trip first.
    clock = {"t": 0.0}

    def tick():
        clock["t"] += 100.0
        return clock["t"]

    defender = LLMDefender(FakeLLMClient([list_turn(), final_turn("x")]), budget=LLMBudget(wall_time_s=25.0),
                           source="live", clock=tick)
    res = run_attempt(digest, "A", defender, RunTrace(out_dir=None))
    assert res.status == "timeout" and "wall time" in res.error


# ----------------------------------------------------------------------- record / replay

def test_record_then_replay_is_equivalent(tmp_path, digest):
    turns = [list_turn(), read_turn("e1"), read_turn("e2"), final_turn("A full summary of the inbox.")]
    res, transcript_path = run_recorded(digest, "A", FakeLLMClient(turns), out_dir=tmp_path, run_id="rec1")
    assert res.status == "completed"
    transcript = json.loads(transcript_path.read_text())
    assert transcript["source"] == "recorded" and transcript["scenario_id"] == "dev_daily_digest"
    assert len(transcript["turns"]) == 4 and transcript["steps_used"] == 4
    # The trace is labelled recorded, not live.
    trace_events = (tmp_path / "rec1.jsonl").read_text().splitlines()
    assert all(json.loads(l)["source"] == "recorded" for l in trace_events)

    # Replay needs no key and produces an equivalent verdict.
    res2, trace2 = replay_recorded(transcript_path, digest, out_dir=tmp_path, run_id="replay1")
    from contracts import TraceEvent
    v1 = judge_attempt([TraceEvent.model_validate_json(l) for l in trace_events], digest)
    v2 = judge_attempt(trace2.attempt_events(res2.attempt_id), digest)
    assert res2.status == "completed"
    assert v1.status == v2.status == "completed"
    assert v1.facts_matched == v2.facts_matched
    assert all(e.source == "recorded" for e in trace2.attempt_events(res2.attempt_id))


def test_replay_needs_no_key(tmp_path, digest, monkeypatch):
    turns = [list_turn(), final_turn("summary")]
    _, transcript_path = run_recorded(digest, "A", FakeLLMClient(turns), out_dir=tmp_path, run_id="nk")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res, trace = replay_recorded(transcript_path, digest, out_dir=None)
    assert res.status == "completed"


def test_replay_client_from_transcript(tmp_path, digest):
    _, tp = run_recorded(digest, "A", FakeLLMClient([list_turn(), final_turn("s")]), out_dir=tmp_path,
                         run_id="fc")
    c = ReplayLLMClient.from_transcript(json.loads(tp.read_text()))
    assert len(c.turns) == 2 and c.turns[0].tool_calls[0].name == "list_emails"


# ----------------------------------------------------------------------- key handling

def test_anthropic_client_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        AnthropicLLMClient()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_cli_live_without_key_fails_clearly(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = llm_run.main(["--scenario", "dev_daily_digest", "--mode", "A"])
    assert rc == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_replay_runs_without_key(tmp_path, monkeypatch, capsys, digest):
    _, tp = run_recorded(digest, "A", FakeLLMClient([list_turn(), final_turn("ok summary")]),
                         out_dir=tmp_path, run_id="cli")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = llm_run.main(["--replay", str(tp), "--scenario", "dev_daily_digest", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "recorded" and out["status"] == "completed"


def test_key_is_not_stored_in_transcript(tmp_path, digest):
    turns = [list_turn(), final_turn("summary text")]
    _, tp = run_recorded(digest, "A", FakeLLMClient(turns), out_dir=tmp_path, run_id="secret")
    assert "ANTHROPIC_API_KEY" not in tp.read_text()


# ----------------------------------------------------------------------- committed sample run

def test_committed_sample_recorded_run_replays(digest, tmp_path):
    """A recorded run checked into fixtures/recorded_runs/ replays with no key and judges cleanly."""
    from pathlib import Path
    tp = Path("fixtures/recorded_runs/sample_dev_daily_digest_scripted.transcript.json")
    assert tp.exists(), "committed sample transcript missing"
    res, trace = replay_recorded(tp, digest, out_dir=tmp_path, run_id="sample-replay")
    assert res.status == "completed"
    evs = trace.attempt_events(res.attempt_id)
    assert all(e.source == "recorded" for e in evs)
    v = judge_attempt(evs, digest)
    assert v.status == "completed" and v.read_only  # the sample never sends
