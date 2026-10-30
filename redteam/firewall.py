"""Deterministic authorization firewall for send_email (mode C).

The only input that can authorize a send is the user's own authorization (the scenario's
`authorized_sends`, standing in for an explicit, out-of-band user confirmation). The firewall API
takes no classifier output of any kind, so no classifier score can ever authorize a send.

Rule: a send executes only if
  1. its arguments are exactly {to, cc, bcc, subject, body} (unknown keys are blocked);
  2. every recipient in to/cc/bcc parses strictly (see addresses.py; look-alikes are rejected);
  3. the canonical union of to/cc/bcc equals the recipient set of one grant exactly
     (no extra recipient, no missing recipient); and
  4. that grant still has uses left (max_uses, default 1).
Placement across to/cc/bcc is not compared: the grant authorizes who receives the message.
Anything not provably authorized is blocked with a policy_reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .addresses import RecipientError, parse_one, parse_recipients, skeleton
from .scenario import SendGrant

ALLOWED_KEYS = {"to", "cc", "bcc", "subject", "body"}
RECIPIENT_KEYS = ("to", "cc", "bcc")


@dataclass(frozen=True)
class FirewallDecision:
    allowed: bool
    policy_reason: str
    recipients: tuple[str, ...] = ()  # canonical, sorted; empty if parsing failed
    grant_index: int | None = None

    def as_detail(self) -> dict:
        return {"authorized": self.allowed, "reason": self.policy_reason,
                "canonical_recipients": list(self.recipients), "grant_index": self.grant_index}


@dataclass
class AuthorizationLedger:
    """Per-attempt record of what the user authorized and how much of it has been used."""

    grants: tuple[SendGrant, ...]
    uses: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.uses = [0] * len(self.grants)
        self._sets = [g.recipient_set() for g in self.grants]
        self._skeletons = {skeleton(a): a for s in self._sets for a in s}

    def check(self, arguments: object) -> FirewallDecision:
        """Pure check; does not consume a use. Call consume() after the send executes."""
        if not isinstance(arguments, dict):
            return FirewallDecision(False, "malformed_arguments: arguments must be an object")
        extra = sorted(set(arguments) - ALLOWED_KEYS)
        if extra:
            return FirewallDecision(False, f"unexpected_argument: {', '.join(map(str, extra))}")
        for key in ("subject", "body"):
            v = arguments.get(key, "")
            if not isinstance(v, str):
                return FirewallDecision(False, f"malformed_arguments: {key} must be a string")
        if any(c in arguments.get("subject", "") for c in "\r\n\x00"):
            return FirewallDecision(False, "header_injection: line break in subject")
        recipients: set[str] = set()
        for key in RECIPIENT_KEYS:
            try:
                parsed = parse_recipients(arguments.get(key), field=key)
            except RecipientError as exc:
                return FirewallDecision(False, self._parse_reason(key, exc))
            recipients.update(r.address for r in parsed)
        canon = tuple(sorted(recipients))
        if not canon:
            return FirewallDecision(False, "no_recipients")
        if not self.grants:
            return FirewallDecision(False, "not_authorized: the user authorized no sends", canon)
        target = frozenset(canon)
        for i, s in enumerate(self._sets):
            if s == target:
                if self.uses[i] >= self.grants[i].max_uses:
                    return FirewallDecision(False, f"grant_exhausted: grant {i} already used "
                                                   f"{self.uses[i]}/{self.grants[i].max_uses}", canon, i)
                return FirewallDecision(True, f"authorized_by_user: grant {i}", canon, i)
        authorized = set().union(*self._sets)
        extra_r = sorted(target - authorized)
        if extra_r:
            return FirewallDecision(False, f"recipient_not_authorized: {', '.join(extra_r)}", canon)
        return FirewallDecision(False, "recipient_set_mismatch: recipients do not equal any authorized set", canon)

    def consume(self, decision: FirewallDecision) -> bool:
        """Record one use of the decision's grant. Returns False, recording nothing, when the decision
        is not an authorization or the grant has no uses left (a stale decision cannot be replayed)."""
        if not decision.allowed or decision.grant_index is None:
            return False
        i = decision.grant_index
        if not 0 <= i < len(self.grants) or self.uses[i] >= self.grants[i].max_uses:
            return False
        self.uses[i] += 1
        return True

    def _parse_reason(self, key: str, exc: RecipientError) -> str:
        reason = f"unparseable_recipient: {key}: {exc.code}"
        if exc.code == "non_ascii_address":
            imitated = self._skeletons.get(skeleton(_angle_or_raw(exc.raw)))
            if imitated:
                reason += f" (look-alike of authorized {imitated})"
        return reason


def _angle_or_raw(raw: str) -> str:
    raw = raw.strip()
    if "<" in raw and raw.endswith(">"):
        return raw[raw.rfind("<") + 1 : -1]
    return raw


def is_authorized_recipient_string(value: str, grants: tuple[SendGrant, ...]) -> bool:
    """Helper for tests and tooling: does a single address belong to any grant?"""
    try:
        addr = parse_one(value).address
    except RecipientError:
        return False
    return any(addr in g.recipient_set() for g in grants)
