"""Hard dollar-cap accounting for the adaptive attacker.

Spend is computed from each API response's token usage times a published per-token price, and the
run must stop BEFORE issuing any call that could push spend over the cap. Nothing here touches the
API key; prices are the only secret-free input.

Prices are per million tokens, from Anthropic's published pricing as reported 2026-10-23 (third
party, see docs/design/adaptive-attacker.md §7 and §9: owner to confirm against billing). The Haiku
family has a long-prompt tier at 100K input tokens; we implement it so the cap is never
under-counted, though our prompts sit far below that break.
"""
from __future__ import annotations

from dataclasses import dataclass, field

PRICES_AS_OF = "2026-10-23"
PRICES_SOURCE = "Anthropic published pricing, third-party reported; owner to verify against billing"

# Each entry: list of (input_tokens_upper_bound, usd_per_1M_input, usd_per_1M_output) tiers, applied
# by the request's input-token count. The last tier has an infinite upper bound.
_PRICES: dict[str, list[tuple[float, float, float]]] = {
    "claude-haiku-5-5": [(100_000, 0.10, 0.50), (float("inf"), 0.50, 2.50)],
    "claude-sonnet-5-5": [(float("inf"), 2.00, 10.00)],
    # Older models kept so recorded runs and tests remain priceable.
    "claude-3-5-haiku-20241022": [(float("inf"), 0.80, 4.00)],
}


class BudgetExceeded(RuntimeError):
    """Raised instead of issuing a call that could exceed the dollar cap."""


def is_priced(model: str) -> bool:
    return model in _PRICES


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call. Unknown models raise, so an unpriceable call never silently costs $0."""
    tiers = _PRICES.get(model)
    if tiers is None:
        raise KeyError(f"no published price for model {model!r}; add it to budget._PRICES before running")
    for upper, in_price, out_price in tiers:
        if input_tokens <= upper:
            return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    upper, in_price, out_price = tiers[-1]
    return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price


def _max_input_price(model: str) -> float:
    return max(t[1] for t in _PRICES[model])


def _max_output_price(model: str) -> float:
    return max(t[2] for t in _PRICES[model])


@dataclass
class SpendTracker:
    """Tracks cumulative spend against a hard cap. Not thread-safe; the attacker loop is sequential.

    ``preflight`` must be called before every billable API call with a conservative upper bound on the
    call's token use; it raises BudgetExceeded if that call could exceed the cap. ``record`` is called
    after the call with the real usage from the response. A cache hit records nothing (it is free)."""

    cap_usd: float
    spent_usd: float = 0.0
    n_calls: int = 0
    n_cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cap_usd <= 0:
            raise ValueError("cap_usd must be positive")

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.spent_usd

    def would_exceed(self, model: str, est_input_tokens: int, est_max_output_tokens: int) -> bool:
        est = cost_usd(model, est_input_tokens, est_max_output_tokens)
        return self.spent_usd + est > self.cap_usd

    def preflight(self, model: str, est_input_tokens: int, est_max_output_tokens: int) -> None:
        """Raise BudgetExceeded if a call with this (upper-bound) usage could exceed the cap."""
        if not is_priced(model):
            raise KeyError(f"no published price for model {model!r}")
        if self.would_exceed(model, est_input_tokens, est_max_output_tokens):
            est = cost_usd(model, est_input_tokens, est_max_output_tokens)
            raise BudgetExceeded(
                f"next call (~{est_input_tokens} in / <= {est_max_output_tokens} out tokens, "
                f"~${est:.4f}) would exceed cap ${self.cap_usd:.2f} (spent ${self.spent_usd:.4f})")

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = cost_usd(model, input_tokens, output_tokens)
        self.spent_usd += cost
        self.n_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        m = self.by_model.setdefault(model, {"n_calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        m["n_calls"] += 1
        m["input_tokens"] += input_tokens
        m["output_tokens"] += output_tokens
        m["usd"] += cost
        return cost

    def record_cache_hit(self) -> None:
        self.n_cache_hits += 1

    def as_dict(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "n_calls": self.n_calls,
            "n_cache_hits": self.n_cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_model": {k: {**v, "usd": round(v["usd"], 6)} for k, v in self.by_model.items()},
            "prices_as_of": PRICES_AS_OF,
            "prices_source": PRICES_SOURCE,
        }
