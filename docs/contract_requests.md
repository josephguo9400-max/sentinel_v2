# Contract change requests

Agents never edit `contracts/` directly. Append a request here; the owner decides and bumps the schema version.

| Date | Requester | Contract | Proposed change | Reason | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-10-29 | Agent A (serving) | `DetectorResult` | Add optional `failure_reason: Literal["timeout", "exception", "overloaded"] \| None` (None unless status != ok) | `status="error"` cannot distinguish a detector exception from a gateway queue-full rejection; callers (eval harness, gate error-rate check) may want to exclude load shedding. Workaround: reason is only in the `sentinel_detector_failures_total{reason}` metric. | pending |
