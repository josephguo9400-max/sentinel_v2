"""Prometheus metrics. Each gateway instance owns its own registry so tests can build many apps."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS_S = (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 20.0)


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("sentinel_classify_requests", "Classify requests by final decision",
                                ["decision", "policy_applied"], registry=r)
        self.detector_latency = Histogram("sentinel_detector_latency_seconds",
                                          "Detector latency per request (tokenize + queue + inference)",
                                          ["detector_id", "status"], buckets=LATENCY_BUCKETS_S, registry=r)
        self.detector_failures = Counter("sentinel_detector_failures", "Detector results with status != ok",
                                         ["detector_id", "status", "reason"], registry=r)
        self.chunks = Histogram("sentinel_chunks_per_request", "Chunks scored per request",
                                buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32), registry=r)
        self.truncated = Counter("sentinel_truncated_requests", "Requests where some content was not scored",
                                 ["detector_id"], registry=r)
        self.batch_size = Histogram("sentinel_inference_batch_size", "Chunks per ONNX call",
                                    buckets=(1, 2, 4, 8, 16, 32, 64), registry=r)
        self.inflight = Gauge("sentinel_inflight_requests", "Admitted requests not yet finished", registry=r)
        self.queue_depth = Gauge("sentinel_queue_depth", "Chunks waiting for a worker", registry=r)
        self.busy_workers = Gauge("sentinel_busy_workers", "Inference workers currently running (incl. abandoned)",
                                  registry=r)
