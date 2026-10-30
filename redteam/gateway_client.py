"""HTTP client for the classifier gateway: POST {url}/classify.

Request:  {"text": str, "request_id": str}
Response: {"decision": "allow"|"flag", "decided_by": str, "policy_applied": str,
           "results": [DetectorResult v1.1 dicts]}   (extra keys are ignored)

When the gateway cannot be used (timeout, connection error, non-2xx, malformed response) the
client applies `on_unavailable`:
  "flag"  (default) -> decision=flag, i.e. the email is quarantined (fail closed), fallback=True
  "allow"           -> decision=allow, the email reaches the defender (fail open), fallback=True
  "error"           -> raise ClassifierUnavailable; the sandbox records the attempt as an error
Every fallback is recorded in the verdict (gateway_status, error) so the trace shows it.
"""
from __future__ import annotations

import time
from typing import Literal

import httpx
from pydantic import ValidationError

from contracts import DetectorResult

from .classifier import ClassifierUnavailable, ClassifierVerdict, GatewayStatus

OnUnavailable = Literal["flag", "allow", "error"]


class GatewayClient:
    def __init__(self, url: str, timeout_s: float = 2.0, on_unavailable: OnUnavailable = "flag",
                 transport: httpx.BaseTransport | None = None):
        if on_unavailable not in ("flag", "allow", "error"):
            raise ValueError(f"on_unavailable must be flag, allow or error: {on_unavailable!r}")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.url = url.rstrip("/") + "/classify"
        self.timeout_s = timeout_s
        self.on_unavailable = on_unavailable
        # trust_env=False: never route sandbox traffic through a proxy picked up from the environment.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), transport=transport, trust_env=False)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def classify(self, text: str, request_id: str) -> ClassifierVerdict:
        t0 = time.perf_counter()
        try:
            resp = self._client.post(self.url, json={"text": text, "request_id": request_id})
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
        return self._parse(data, t0)

    def _parse(self, data: object, t0: float) -> ClassifierVerdict:
        if not isinstance(data, dict):
            return self._fallback("bad_response", "response is not a JSON object", t0)
        decision = data.get("decision")
        if decision not in ("allow", "flag"):
            return self._fallback("bad_response", f"invalid decision {decision!r}", t0)
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            return self._fallback("bad_response", "results must be a list", t0)
        try:
            results = tuple(DetectorResult.model_validate(r) for r in raw_results)
        except ValidationError as exc:
            return self._fallback("bad_response", f"invalid DetectorResult: {exc.errors()[0]['msg']}", t0)
        decided_by, policy = data.get("decided_by"), data.get("policy_applied")
        return ClassifierVerdict(
            decision=decision,
            decided_by=decided_by if isinstance(decided_by, str) else None,
            policy_applied=policy if isinstance(policy, str) else None,
            results=results, gateway_status="ok", latency_ms=_ms(t0),
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
