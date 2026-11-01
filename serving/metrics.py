"""Prometheus metrics. Each gateway instance owns its own registry so tests can build many apps."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS_S = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0,
                     7.5, 10.0, 20.0)


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("sentinel_classify_requests", "Classify requests by final decision",
                                ["decision", "policy_applied"], registry=r)
        self.paths = Counter("sentinel_cascade_path", "Requests by routing outcome (stage1_allow, stage2_flag, "
                                                      "coverage_flag, stage2_timeout, overloaded, ...)",
                             ["path"], registry=r)
        self.coverage = Counter("sentinel_coverage_policy_applied",
                                "Requests decided by the coverage policy (content scored by neither stage)",
                                ["policy"], registry=r)
        self.unscored = Counter("sentinel_unscored_requests",
                                "Requests with content neither cleared by the gate nor scored by a decide detector "
                                "(ClassifyResponse.unscored_by_any_step), by decision", ["decision"], registry=r)
        self.detector_latency = Histogram("sentinel_detector_latency_seconds",
                                          "Detector latency per request; role=active is the decision path, "
                                          "role=shadow the observe-only Stage 1",
                                          ["detector_id", "status", "role"], buckets=LATENCY_BUCKETS_S, registry=r)
        self.detector_failures = Counter("sentinel_detector_failures",
                                         "Detector results with status != ok (role=active: decision path; "
                                         "role=shadow: observe-only Stage 1, never part of a decision)",
                                         ["detector_id", "status", "reason", "role"], registry=r)
        self.request_chunks = Histogram("sentinel_request_chunks", "Chunks in the request's chunk plan",
                                        buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128), registry=r)
        self.chunks = Histogram("sentinel_chunks_per_request", "Chunks scored per request, per stage",
                                ["stage", "role"], buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128), registry=r)
        self.truncated = Counter("sentinel_truncated_requests",
                                 "Detector results where that detector did not score some content",
                                 ["detector_id", "role"], registry=r)
        self.batch_size = Histogram("sentinel_inference_batch_size", "Chunks per Stage 2 ONNX call",
                                    buckets=(1, 2, 4, 8, 16, 32, 64), registry=r)
        # Padding cost of Stage 2 batches: computed = rows x longest row, real = sum of row lengths.
        self.real_tokens = Counter("sentinel_inference_real_tokens", "Non-padding token positions sent to the model",
                                   registry=r)
        self.computed_tokens = Counter("sentinel_inference_computed_tokens",
                                       "Token positions the model computed, padding included", registry=r)
        self.inflight = Gauge("sentinel_inflight_requests", "Admitted requests not yet finished", registry=r)
        self.queue_depth = Gauge("sentinel_queue_depth", "Chunks waiting for a worker", registry=r)
        self.busy_workers = Gauge("sentinel_busy_workers", "Inference workers currently running (incl. abandoned)",
                                  registry=r)
        # Stage 1 (registry stage1_mode, serving/service.py)
        self.stage1_observed = Counter("sentinel_stage1_observed",
                                       "Observe-only Stage 1: what routed Stage 1 would have done "
                                       "(would_finalize_allow / would_escalate / not_scored / dropped)", ["outcome"],
                                       registry=r)
        self.stage1_deadline_hits = Counter("sentinel_stage1_deadline_hits",
                                            "Stage 1 stopped at its soft deadline with chunks left unscored",
                                            ["detector_id", "stage1_mode"], registry=r)
        # Registry
        self.registry_loads = Counter("sentinel_registry_loads", "Registry (re)loads by outcome", ["outcome"],
                                      registry=r)
        self.registry_rejections = Counter("sentinel_registry_rejections", "Detectors rejected at load",
                                           ["pointer", "code"], registry=r)
        # Shadow
        self.shadow_results = Counter("sentinel_shadow_results", "Shadow results by status and failure_reason",
                                      ["detector_id", "status", "reason"], registry=r)
        self.shadow_latency = Histogram("sentinel_shadow_latency_seconds", "Shadow job latency (excl. queue wait)",
                                        ["detector_id", "status"], buckets=LATENCY_BUCKETS_S, registry=r)
        self.shadow_comparisons = Counter("sentinel_shadow_comparisons",
                                          "Shadow vs active decision (agree / disagree / not_comparable)",
                                          ["detector_id", "outcome"], registry=r)
        self.shadow_dropped = Counter("sentinel_shadow_dropped", "Mirrored requests dropped (shadow queue full)",
                                      registry=r)
        self.shadow_queue_depth = Gauge("sentinel_shadow_queue_depth", "Requests waiting for the shadow", registry=r)

    def observe_batch(self, rows: list[list[int]]) -> None:
        """Record one ONNX call: its batch size, and its real and computed (padded) token positions."""
        self.batch_size.observe(len(rows))
        self.real_tokens.inc(sum(len(x) for x in rows))
        self.computed_tokens.inc(len(rows) * max((len(x) for x in rows), default=0))
