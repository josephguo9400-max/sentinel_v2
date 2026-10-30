# Contract change requests

Agents never edit `contracts/` directly. Append a request here; the owner decides and bumps the schema version.

| Date | Requester | Contract | Proposed change | Reason | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-10-29 | Agent A (serving) | `DetectorResult` | Add optional `failure_reason: Literal["timeout", "exception", "overloaded"] \| None` (None unless status != ok) | `status="error"` cannot distinguish a detector exception from a gateway queue-full rejection; callers (eval harness, gate error-rate check) may want to exclude load shedding. Workaround: reason is only in the `sentinel_detector_failures_total{reason}` metric. | pending |
| 2026-10-29 | Agent C | `TraceEvent` | Add optional `seq: int` (per-attempt monotonic order) and `scenario_id: str` | Timestamps can tie, and judging needs the scenario; both are carried in `detail` (`detail.seq`, `detail.scenario_id`) as a workaround | |
| 2026-10-29 | Agent C | new `ClassifyRequest` / `ClassifyResponse` | Freeze the `/classify` wire shape: request `{text, request_id}`; response `{decision: allow\|flag, decided_by, policy_applied, results: [DetectorResult]}` | Agent A builds it and Agent C consumes it in parallel; today it exists only in prose. `redteam/gateway_client.py` validates `results` with `DetectorResult` and ignores extra keys | |
| 2026-10-29 | Agent C | `TraceEvent` | Allow `detector_version=None` (or define a sentinel) on `detector_prediction`/`detector_timeout` when the gateway was unreachable and no detector answered | Field is required for these types; the client writes `"unavailable"` and sets `detail.fallback=true` as a workaround | |
