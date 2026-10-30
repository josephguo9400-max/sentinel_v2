"""HTTP client for the classifier gateway: POST {url}/classify.

Request:  contracts.ClassifyRequest  {"text": str, "request_id": str}
Response: contracts.ClassifyResponse (v1.2), validated strictly: unknown keys, a request_id other than
          the one sent, or any invalid DetectorResult make the response unusable (bad_response).
          stages_run, coverage_policy_applied and each result's failure_reason are kept in the verdict.

When the gateway cannot be used (timeout, connection error, non-2xx, malformed response) the
client applies `on_unavailable`:
  "flag"  (default) -> decision=flag, i.e. the email is quarantined (fail closed), fallback=True
  "allow"           -> decision=allow, the email reaches the defender (fail open), fallback=True
  "error"           -> raise ClassifierUnavailable; the sandbox records the attempt as an error
Every fallback is recorded in the verdict (gateway_status, error) so the trace shows it.

Timeout. The client must never give up before the gateway does: the gateway enforces its own
per-detector budget (GatewayConfig.timeout_ms, default 5000 ms) and answers with its failure policy
when the budget runs out. A shorter client timeout would turn a slow-but-scored email into a client
fallback. With `timeout_s=None` (the default) the client reads `config.timeout_ms` from
GET {url}/health and uses budget + TIMEOUT_MARGIN_S; if /health is unusable it uses
DEFAULT_TIMEOUT_S. `timeout_source` records which rule applied ("explicit", "health", "health_capped", "default");
a budget from /health is capped at MAX_DERIVED_TIMEOUT_S because /health is untrusted input.
"""
from __future__ import annotations

import time
from typing import Literal

import httpx
from pydantic import ValidationError

from contracts import ClassifyRequest, ClassifyResponse

from .classifier import ClassifierUnavailable, ClassifierVerdict, GatewayStatus

OnUnavailable = Literal["flag", "allow", "error"]

DEFAULT_TIMEOUT_S = 10.0  # used when the gateway budget cannot be read from /health
TIMEOUT_MARGIN_S = 2.0  # added to the gateway budget: HTTP, JSON and event-loop overhead
HEALTH_TIMEOUT_S = 2.0  # for the one GET /health probe at construction
# A budget read from /health is untrusted input: never let it make one call wait longer than this.
# 30 s matches the sandbox's default per-attempt deadline (Limits.deadline_s).
MAX_DERIVED_TIMEOUT_S = 30.0


class GatewayClient:
    def __init__(self, url: str, timeout_s: float | None = None, on_unavailable: OnUnavailable = "flag",
                 transport: httpx.BaseTransport | None = None):
        if on_unavailable not in ("flag", "allow", "error"):
            raise ValueError(f"on_unavailable must be flag, allow or error: {on_unavailable!r}")
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        base = url.rstrip("/")
        self.url = base + "/classify"
        self.on_unavailable = on_unavailable
        # trust_env=False: never route sandbox traffic through a proxy picked up from the environment.
        self._client = httpx.Client(transport=transport, trust_env=False)
        # An explicit timeout is used as given (no /health probe), so tests can bound waits precisely.
        self.gateway_timeout_ms: float | None = None if timeout_s is not None else self._read_budget_ms(
            base + "/health")
        if timeout_s is not None:
            self.timeout_s, self.timeout_source = float(timeout_s), "explicit"
        elif self.gateway_timeout_ms is not None:
            derived = self.gateway_timeout_ms / 1000.0 + TIMEOUT_MARGIN_S
            if derived > MAX_DERIVED_TIMEOUT_S:
                self.timeout_s, self.timeout_source = MAX_DERIVED_TIMEOUT_S, "health_capped"
            else:
                self.timeout_s, self.timeout_source = derived, "health"
        else:
            self.timeout_s, self.timeout_source = DEFAULT_TIMEOUT_S, "default"
        self._timeout = httpx.Timeout(self.timeout_s)

    def _read_budget_ms(self, health_url: str) -> float | None:
        try:
            resp = self._client.get(health_url, timeout=HEALTH_TIMEOUT_S)
            data = resp.json() if resp.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        cfg = data.get("config") if isinstance(data, dict) else None
        budget = cfg.get("timeout_ms") if isinstance(cfg, dict) else None
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget < 3_600_000:
            return None
        return float(budget)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def classify(self, text: str, request_id: str) -> ClassifierVerdict:
        t0 = time.perf_counter()
        try:
            payload = ClassifyRequest(text=text, request_id=request_id).model_dump()
            resp = self._client.post(self.url, json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            return self._fallback("timeout", f"{type(exc).__name__}: {exc}", t0)
        except httpx.HTTPError as exc:
            return self._fallback("unreachable", f"{type(exc).__name__}: {exc}", t0)
        if resp.status_code // 100 != 2:
            return self._fallback("bad_response", f"HTTP {resp.status_code}", t0)
        try:
            data = resp.json()
        except ValueError:
            return self._fallback("bad_response", "response is not JSON", t0)
        return self._parse(data, request_id, t0)

    def _parse(self, data: object, request_id: str, t0: float) -> ClassifierVerdict:
        if not isinstance(data, dict):
            return self._fallback("bad_response", "response is not a JSON object", t0)
        try:
            resp = ClassifyResponse.model_validate(data)
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = ".".join(str(x) for x in err.get("loc", ()))
            return self._fallback("bad_response", f"invalid ClassifyResponse at {loc or '<root>'}: {err['msg']}", t0)
        if resp.request_id != request_id:
            return self._fallback("bad_response", f"response request_id {resp.request_id!r} != sent {request_id!r}",
                                  t0)
        return ClassifierVerdict(
            decision=resp.decision, decided_by=resp.decided_by, policy_applied=resp.policy_applied,
            results=tuple(resp.results), gateway_status="ok", latency_ms=_ms(t0),
            request_id=resp.request_id, stages_run=tuple(resp.stages_run),
            coverage_policy_applied=resp.coverage_policy_applied,
        )

    def _fallback(self, status: GatewayStatus, error: str, t0: float) -> ClassifierVerdict:
        if self.on_unavailable == "error":
            raise ClassifierUnavailable(f"gateway {status}: {error}")
        return ClassifierVerdict(
            decision="flag" if self.on_unavailable == "flag" else "allow",
            decided_by=None, policy_applied=f"client_on_unavailable={self.on_unavailable}",
            gateway_status=status, fallback=True, error=error, latency_ms=_ms(t0),
        )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)
