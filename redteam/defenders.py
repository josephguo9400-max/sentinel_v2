"""Defenders. Anything implementing sandbox.Defender (name, source, run(env) -> str) can be plugged in.

Scripted defenders (source="scripted") exist to prove the plumbing without an LLM:
  BenignDefender    reads every email, optionally searches docs, writes an extractive summary,
                    and sends only to addresses written in the user's own request.
  GullibleDefender  same, but also obeys any send instruction it finds in an email body,
                    so it proposes the attacker's send.

The next milestone's LLM defender is expected to look like LLMDefenderSkeleton: map each model
tool_use block to env.call_tool(name, input), feed the result back, record model text with
env.record_model_response, and return the final text. It is intentionally not implemented here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .sandbox import DefenderEnv

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_ADDR_TOKEN = r"[^\s,;]*@[^\s,;]+"
# Instruction forms the gullible defender obeys.
_ACTION_LINE = re.compile(r"^\s*ACTION:\s*send_email\s+(\{.*\})\s*$", re.MULTILINE)
_NL_SEND = re.compile(rf"\b(?:send|forward|email|mail)\b[^\n]{{0,120}}?\bto\s+(?P<addr>{_ADDR_TOKEN})",
                      re.IGNORECASE)
_USER_CC = re.compile(rf"\b(?:cc|copy)\s+(?P<addr>{_ADDR_TOKEN})", re.IGNORECASE)
_USER_ADDR = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    return _SENTENCE_END.split(text, maxsplit=1)[0] if text else ""


def _strip_trailing_punct(addr: str) -> str:
    return addr.rstrip(".,;:!?)\"'")


@dataclass
class _ReadAll:
    """Shared reading/summarizing behaviour for scripted defenders."""

    name: str = "scripted"
    source: str = "scripted"

    def read_inbox(self, env: DefenderEnv) -> list[dict[str, Any]]:
        listing = env.call_tool("list_emails", {})
        emails = []
        for meta in listing.get("result", []) if listing.get("ok") else []:
            r = env.call_tool("read_email", {"email_id": meta["id"]})
            if r.get("ok"):
                emails.append(r["result"])
        return emails

    def search(self, env: DefenderEnv) -> list[dict[str, Any]]:
        docs = []
        queries = env.scripted_hints.search_queries if env.scripted_hints else ()
        for q in queries:
            r = env.call_tool("search_docs", {"query": q})
            if r.get("ok"):
                docs.extend(r["result"])
        return docs

    def summarize(self, emails: list[dict[str, Any]], docs: list[dict[str, Any]]) -> str:
        lines = [f"Summary of {len(emails)} emails:"]
        for e in emails:
            if e.get("quarantined"):
                lines.append(f"- {e['subject']} (from {e['from']}): [body quarantined, not read]")
            else:
                lines.append(f"- {e['subject']} (from {e['from']}): {first_sentence(e['body'])}")
        for d in docs:
            lines.append(f"- Doc '{d['title']}': {first_sentence(d['content'])}")
        return "\n".join(lines)

    def user_requested_send(self, env: DefenderEnv, summary: str) -> dict[str, Any] | None:
        """Addresses the user wrote in the request: 'cc X' goes to cc, everything else to to."""
        cc = {_strip_trailing_punct(m.group("addr")) for m in _USER_CC.finditer(env.user_request)}
        addrs = [a for a in (_strip_trailing_punct(x) for x in _USER_ADDR.findall(env.user_request))
                 if a.lower() != env.user_address.lower()]
        to = [a for a in addrs if a not in cc]
        if not to and not cc:
            return None
        args: dict[str, Any] = {"to": to, "subject": "Summary of today's emails", "body": summary}
        if cc:
            args["cc"] = sorted(cc)
        return args


@dataclass
class BenignDefender(_ReadAll):
    name: str = "benign"
    source: str = "scripted"

    def run(self, env: DefenderEnv) -> str:
        emails = self.read_inbox(env)
        summary = self.summarize(emails, self.search(env))
        notes = []
        send = self.user_requested_send(env, summary)
        if send is not None:
            r = env.call_tool("send_email", send)
            notes.append("Sent the summary as requested." if r.get("ok") else f"Could not send: {r.get('error')}")
        return "\n".join([summary, *notes])


@dataclass
class GullibleDefender(_ReadAll):
    """Follows every send instruction found in any email body it can read."""

    name: str = "gullible"
    source: str = "scripted"

    def run(self, env: DefenderEnv) -> str:
        emails = self.read_inbox(env)
        summary = self.summarize(emails, self.search(env))
        notes = []
        send = self.user_requested_send(env, summary)
        if send is not None:
            r = env.call_tool("send_email", send)
            notes.append("Sent the summary as requested." if r.get("ok") else f"Could not send: {r.get('error')}")
        for e in emails:
            for args in self.instructions(e.get("body") or "", summary):
                r = env.call_tool("send_email", args)
                notes.append(f"Followed instruction in {e['id']}: {'sent' if r.get('ok') else r.get('error')}")
        return "\n".join([summary, *notes])

    @staticmethod
    def instructions(body: str, summary: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in _ACTION_LINE.finditer(body):
            try:
                args = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(args, dict):
                args.setdefault("subject", "Requested information")
                args.setdefault("body", summary)
                out.append(args)
        if out:
            return out
        for m in _NL_SEND.finditer(body):
            out.append({"to": _strip_trailing_punct(m.group("addr")), "subject": "Requested information",
                        "body": summary})
        return out


class LLMDefenderSkeleton:
    """Placeholder showing the shape of the next milestone's live defender. Not functional."""

    name = "llm"
    source = "live"

    def run(self, env: DefenderEnv) -> str:  # pragma: no cover - documented interface only
        raise NotImplementedError("the LLM defender is scheduled for the cascade-and-shadow milestone")


DEFENDERS = {"benign": BenignDefender, "gullible": GullibleDefender}
