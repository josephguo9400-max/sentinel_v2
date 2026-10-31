"""Compose a candidate body from a benign carrier + a rendered, framed injection core at a placement.

Placement is by DeBERTa content-token offset so ``after_512`` really lands past the first 512-token
scoring window (review M2: the first version's long_dilution cores started at tokens 508/415, never
past 512). The tokenizer helpers are reused from redteam/probes.py so there is one tokenizer of record.

A ``CompositionResult`` records the measured token span of the framed block within the final body and
whether the requested placement was actually reachable (e.g. after_512 needs a carrier long enough),
so a report can never silently claim a placement it did not achieve.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .mutators import Mutation, framed_core

AFTER_512_OFFSET = 512


@dataclass(frozen=True)
class CompositionResult:
    body: str
    core_token_start: int        # first content-token index of the framed block in the body
    core_token_end: int          # last+1
    body_tokens: int
    placement_requested: str
    placement_reached: bool      # False if after_512 was requested but the body was too short
    note: str = ""


def _tok():
    from ..probes import _tokenizer
    return _tokenizer()


def n_tokens(text: str) -> int:
    from ..probes import n_tokens as _n
    return _n(text)


def _char_offset_of_token(text: str, token_index: int) -> int:
    """Character offset at which content-token ``token_index`` begins (end-of-text if beyond)."""
    enc = _tok().encode(text, add_special_tokens=False)
    if token_index <= 0:
        return 0
    if token_index >= len(enc.ids):
        return len(text)
    return enc.offsets[token_index][0]


def _insert_at_token(carrier: str, block: str, token_index: int) -> str:
    cut = _char_offset_of_token(carrier, token_index)
    # Snap to a word boundary so we never split a carrier word.
    while 0 < cut < len(carrier) and not carrier[cut].isspace() and not carrier[cut - 1].isspace():
        cut += 1
    left, right = carrier[:cut].rstrip(), carrier[cut:].lstrip()
    parts = [p for p in (left, block, right) if p]
    return "\n\n".join(parts)


def compose(carrier: str, rendered_core: str, m: Mutation, *, seed: int = 0) -> CompositionResult:
    """Place ``rendered_core`` (already target-substituted) into ``carrier`` under mutation ``m``."""
    rng = random.Random(seed)
    block = framed_core(rendered_core, m, rng)
    carrier = carrier.strip()
    ctoks = n_tokens(carrier)
    reached = True
    note = ""
    if m.placement == "start":
        idx = 0
    elif m.placement == "end":
        idx = ctoks
    elif m.placement == "middle":
        idx = ctoks // 2
    elif m.placement == "after_512":
        if ctoks > AFTER_512_OFFSET:
            idx = AFTER_512_OFFSET
        else:
            idx = ctoks  # too short to reach 512; fall back to end and record it
            reached = False
            note = f"carrier only {ctoks} tokens; after_512 fell back to end"
    else:
        raise ValueError(f"unknown placement {m.placement!r}")

    body = _insert_at_token(carrier, block, idx)
    # Measure where the block actually landed (by its unique framed text if possible, else the core).
    from ..probes import token_span_of
    anchor = block if body.count(block) == 1 else rendered_core
    try:
        start, end = token_span_of(body, anchor)
    except (ValueError, AssertionError):
        start, end = (-1, -1)
    return CompositionResult(body=body, core_token_start=start, core_token_end=end,
                             body_tokens=n_tokens(body), placement_requested=m.placement,
                             placement_reached=reached, note=note)
