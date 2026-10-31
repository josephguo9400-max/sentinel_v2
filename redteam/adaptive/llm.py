"""Attacker LLM clients. The attacker asks a model for the next candidate email body.

Providers plug in behind ``AttackerClient``:
  * ``AnthropicAttackerClient`` -- Anthropic Messages API. The key is read ONLY from ANTHROPIC_API_KEY
    and is held by the SDK; it is never logged, cached or persisted. The SDK is imported lazily.
    Budget is enforced BEFORE each billable call and the response cache (content-keyed, header-free)
    is consulted first so reruns are free.
  * ``FakeAttackerClient`` -- scripted completions for tests: no key, no network.

Every billable call is: cache-lookup -> budget preflight -> API call -> record usage -> cache store.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .budget import SpendTracker
from .cache import ResponseCache, request_hash

MISSING_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not set. The adaptive attacker needs a spend-capped Anthropic API key in "
    "that environment variable (load it from /root/.sentinel/.env with `set -a; . /root/.sentinel/.env; "
    "set +a`). The key is never printed, logged, cached or committed. Use the replay/cache or a "
    "FakeAttackerClient to run without a key.")


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False


@runtime_checkable
class AttackerClient(Protocol):
    model: str

    def complete(self, *, system: str, messages: list[dict[str, Any]], max_tokens: int,
                 temperature: float) -> Completion: ...


def _estimate_tokens(system: str, messages: list[dict[str, Any]]) -> int:
    """Conservative (over-)estimate of input tokens for a budget preflight: ~chars/3 (real tokenizers
    give ~chars/4, so this over-counts, which is the safe direction for a cap)."""
    chars = len(system)
    for m in messages:
        c = m.get("content", "")
        chars += len(c) if isinstance(c, str) else len(str(c))
    return max(1, chars // 3)


class AnthropicAttackerClient:
    """Anthropic Messages API client with budget + cache. Returns plain text completions."""

    def __init__(self, model: str, spend: SpendTracker, cache: ResponseCache,
                 api_key: str | None = None):
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(MISSING_KEY_MESSAGE)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the 'anthropic' package is not installed; install it for live runs") from exc
        self.model = model
        self.spend = spend
        self.cache = cache
        # The SDK holds the key; we never store it on self in a logged field. trust_env is left at its
        # default so the pre-configured HTTPS proxy + CA bundle are honoured.
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, *, system: str, messages: list[dict[str, Any]], max_tokens: int,
                 temperature: float) -> Completion:
        key = request_hash(model=self.model, system=system, messages=messages, max_tokens=max_tokens,
                           temperature=temperature)
        hit = self.cache.get(key)
        if hit is not None:
            self.spend.record_cache_hit()
            return Completion(text=hit["text"], input_tokens=hit["input_tokens"],
                              output_tokens=hit["output_tokens"], model=self.model, cached=True)
        # Budget preflight with a conservative upper bound BEFORE any network call.
        self.spend.preflight(self.model, _estimate_tokens(system, messages), max_tokens)
        # The v2 Messages API does not accept a top-level `temperature`; sampling uses the API default.
        # `temperature` is kept in the cache key (so a flag change yields a distinct run) but not sent.
        # Candidate diversity comes from the varied prompt (seed angle + growing feedback history).
        resp = self._client.messages.create(  # pragma: no cover - needs a key
            model=self.model, system=system, messages=messages, max_tokens=max_tokens)
        text = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        in_tok = getattr(resp.usage, "input_tokens", 0)
        out_tok = getattr(resp.usage, "output_tokens", 0)
        self.spend.record(self.model, in_tok, out_tok)
        self.cache.put(key, {"text": text, "input_tokens": in_tok, "output_tokens": out_tok})
        return Completion(text=text, input_tokens=in_tok, output_tokens=out_tok, model=self.model)


@dataclass
class FakeAttackerClient:
    """Scripted completions for tests. ``script`` is a callable (system, messages) -> text, or a list of
    texts returned in order. Tokens are derived deterministically so budget/cache paths are exercised."""

    model: str
    script: Callable[[str, list[dict[str, Any]]], str] | list[str]
    spend: SpendTracker | None = None
    cache: ResponseCache | None = None
    _i: int = 0

    def complete(self, *, system: str, messages: list[dict[str, Any]], max_tokens: int,
                 temperature: float) -> Completion:
        if self.cache is not None:
            key = request_hash(model=self.model, system=system, messages=messages,
                               max_tokens=max_tokens, temperature=temperature)
            hit = self.cache.get(key)
            if hit is not None:
                if self.spend is not None:
                    self.spend.record_cache_hit()
                return Completion(hit["text"], hit["input_tokens"], hit["output_tokens"], self.model, True)
        if callable(self.script):
            text = self.script(system, messages)
        else:
            text = self.script[min(self._i, len(self.script) - 1)]
            self._i += 1
        in_tok = _estimate_tokens(system, messages)
        out_tok = max(1, len(text) // 4)
        if self.spend is not None:
            self.spend.preflight(self.model, in_tok, max_tokens)
            self.spend.record(self.model, in_tok, out_tok)
        if self.cache is not None:
            self.cache.put(key, {"text": text, "input_tokens": in_tok, "output_tokens": out_tok})
        return Completion(text=text, input_tokens=in_tok, output_tokens=out_tok, model=self.model)
