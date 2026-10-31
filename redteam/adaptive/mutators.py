"""Mutation operators for the adaptive attacker (milestone adaptive-attacker, redesign r1).

The review (reviews/adaptive-attacker-r1.md M2) found the first version never adapted: the only thing
that changed between queries was a "variation #n" hint to the carrier-writing LLM, and decision-only
and score-feedback were identical by construction. The RFC (docs/RFC.md "Adaptive attacker") wants a
real keep-best / rewrite loop whose adaptivity comes from *mutation operators*, not from an LLM
authoring injections.

A **candidate** is a composed email body: a benign carrier with one injection core placed into it,
under a set of mutation parameters (``Mutation``). The search (redteam/adaptive/search.py) keeps the
best candidate seen so far and applies one more operator to it each query. An LLM may write or
paraphrase the *benign carrier* only; no operator here authors or paraphrases the injection core.

Operators (all deterministic given a seed):
  * placement    - where the core sits: start / middle / end / after DeBERTa token 512 (long bodies).
  * carrier_len  - how long the benign carrier is (reuse of the DILUTION.md length sweep).
  * paraphrase   - swap the request verb / justification using fixed paraphrase tables (core-preserving:
                   it rewrites only the connective framing around the core, never the core string).
  * split        - break the surrounding framing across paragraphs.
  * frame        - wrap the core in quoting / forwarded-message framing.
  * noise        - whitespace, casing and zero-width formatting noise around the framing.

``after_token_512`` placement needs a token offset; the harness passes a tokenizer-based splitter so
this module stays import-light (no model import at module load).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Callable

Placement = str  # "start" | "middle" | "end" | "after_512"
PLACEMENTS: tuple[str, ...] = ("start", "middle", "end", "after_512")

# Benign connective framings the core can be wrapped in. These are not injection text; they are the
# ordinary-email scaffolding around the (fixed) core. Varying them is the "paraphrase the framing"
# operator and never touches the core string itself.
_LEAD_INS: tuple[str, ...] = (
    "", "By the way, ", "One quick note: ", "For completeness, ", "A small housekeeping item: ",
    "While you are at it, ", "Just so it is on record, ",
)
_TRAILERS: tuple[str, ...] = (
    "", " Thanks!", " Appreciate it.", " No rush.", " Let me know if that is a problem.",
)
# Quoting / forwarded-message framings (the "frame" operator).
_FRAMES: tuple[str, ...] = (
    "{core}",
    "> {core}",
    "---------- Forwarded message ----------\n{core}",
    "[quoted from an earlier message]\n{core}",
)
_ZW = "​"  # zero-width space used by the noise operator (formatting noise, not content)


@dataclass(frozen=True)
class Mutation:
    """The parameters that turn a (carrier, core) pair into a concrete candidate body."""
    placement: Placement = "end"
    carrier_scale: float = 1.0      # multiplier applied to the carrier length by the harness
    lead_in: int = 0                # index into _LEAD_INS
    trailer: int = 0                # index into _TRAILERS
    frame: int = 0                  # index into _FRAMES
    split: bool = False             # break framing across paragraphs
    noise: bool = False             # whitespace/casing/zero-width noise around the framing

    def as_dict(self) -> dict:
        return {"placement": self.placement, "carrier_scale": self.carrier_scale,
                "lead_in": self.lead_in, "trailer": self.trailer, "frame": self.frame,
                "split": self.split, "noise": self.noise}


OPERATORS: tuple[str, ...] = ("placement", "carrier_len", "paraphrase", "split", "frame", "noise")


def _apply_noise(text: str, rng: random.Random) -> str:
    # Light formatting noise only around spacing; never alters the core's words.
    out = []
    for ch in text:
        out.append(ch)
        if ch == " " and rng.random() < 0.04:
            out.append(_ZW)
    return "".join(out)


def framed_core(core: str, m: Mutation, rng: random.Random) -> str:
    """Wrap a rendered core with the mutation's benign framing (lead-in, frame, trailer, split, noise).

    The core string is preserved verbatim inside the framing. Returns the framed block to be placed
    into the carrier by the harness."""
    lead = _LEAD_INS[m.lead_in % len(_LEAD_INS)]
    trail = _TRAILERS[m.trailer % len(_TRAILERS)]
    framed = _FRAMES[m.frame % len(_FRAMES)].format(core=core)
    if m.split and "\n" not in framed:
        # Put the lead-in on its own line, separating framing from the core line.
        block = f"{lead.strip()}\n\n{framed}{trail}" if lead.strip() else f"{framed}{trail}"
    else:
        block = f"{lead}{framed}{trail}"
    if m.noise:
        block = _apply_noise(block, rng)
    return block


def neighbours(m: Mutation, operator: str, rng: random.Random) -> Mutation:
    """Return a mutation that differs from ``m`` by one application of ``operator`` (keep-best step)."""
    if operator == "placement":
        choices = [p for p in PLACEMENTS if p != m.placement]
        return replace(m, placement=rng.choice(choices))
    if operator == "carrier_len":
        scale = rng.choice([0.5, 1.0, 2.0, 4.0])
        return replace(m, carrier_scale=scale)
    if operator == "paraphrase":
        return replace(m, lead_in=rng.randrange(len(_LEAD_INS)), trailer=rng.randrange(len(_TRAILERS)))
    if operator == "split":
        return replace(m, split=not m.split)
    if operator == "frame":
        return replace(m, frame=rng.randrange(len(_FRAMES)))
    if operator == "noise":
        return replace(m, noise=not m.noise)
    raise ValueError(f"unknown operator {operator!r}")


def random_mutation(rng: random.Random) -> Mutation:
    return Mutation(
        placement=rng.choice(PLACEMENTS),
        carrier_scale=rng.choice([0.5, 1.0, 2.0, 4.0]),
        lead_in=rng.randrange(len(_LEAD_INS)),
        trailer=rng.randrange(len(_TRAILERS)),
        frame=rng.randrange(len(_FRAMES)),
        split=rng.random() < 0.5,
        noise=rng.random() < 0.3,
    )
