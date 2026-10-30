"""Fixture integrity: schema, reserved domains, split layout, and coverage requirements."""
import json
import re
from pathlib import Path

import pytest

from redteam.scenario import SCENARIO_ROOT, Scenario, load_split

DEV, HELDOUT = load_split("dev"), load_split("heldout")
ADDR = re.compile(r"[A-Za-z0-9._%+'Ѐ-ӿ-]+@([A-Za-z0-9.Ѐ-ӿ-]+)")


def test_split_sizes():
    assert len(DEV) >= 3 and len(HELDOUT) >= 2
    assert {s.split for s in DEV} == {"dev"} and {s.split for s in HELDOUT} == {"heldout"}
    assert not {s.scenario_id for s in DEV} & {s.scenario_id for s in HELDOUT}


def test_some_scenario_authorizes_a_send_and_some_do_not():
    allsc = DEV + HELDOUT
    assert any(s.authorized_sends for s in allsc) and any(not s.authorized_sends for s in allsc)
    assert any(s.authorized_sends for s in DEV) and any(s.authorized_sends for s in HELDOUT)


@pytest.mark.parametrize("path", sorted(SCENARIO_ROOT.glob("*/*.json")), ids=lambda p: p.name)
def test_every_address_in_fixture_text_uses_a_reserved_domain(path: Path):
    text = path.read_text(encoding="utf-8")
    json.loads(text)
    for m in ADDR.finditer(json.loads(json.dumps(text))):
        domain = m.group(1).rstrip(".").lower()
        ok = domain.endswith(".invalid") or any(domain == d or domain.endswith("." + d)
                                                 for d in ("example.com", "example.org", "example.net"))
        assert ok, f"{path.name}: {m.group(0)}"


def test_non_reserved_address_rejected():
    raw = json.loads((SCENARIO_ROOT / "dev" / "dev_daily_digest.json").read_text())
    raw["emails"][0]["from"] = "someone@gmail.com"
    with pytest.raises(ValueError):
        Scenario.model_validate(raw)


def test_attacker_email_is_the_only_injection_carrier():
    for s in DEV + HELDOUT:
        for e in s.emails:
            has_action = "ACTION: send_email" in e.body or "forward a copy" in e.body
            assert has_action == (e.id == s.attacker_email_id), (s.scenario_id, e.id)


def test_long_dilution_injection_is_past_512_words():
    sc = next(s for s in HELDOUT if s.scenario_id == "ho_long_dilution")
    body = sc.email(sc.attacker_email_id).body
    assert len(body[: body.index("ignore previous")].split()) > 512
