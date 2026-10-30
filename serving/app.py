"""Sentinel gateway: FastAPI app exposing POST /classify, GET /health and GET /metrics.

Run: uvicorn serving.app:app --host 127.0.0.1 --port 8000   (config via SENTINEL_* env vars, see config.py)
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from contracts import SCHEMA_VERSION, ClassifyRequest, ClassifyResponse

from .config import GatewayConfig
from .detector import Detector, OnnxDebertaDetector
from .metrics import GatewayMetrics
from .service import ClassifierService

log = logging.getLogger("sentinel.serving")


MAX_REQUEST_ID_CHARS = 200


def load_onnx_detector(config: GatewayConfig) -> OnnxDebertaDetector:
    from .detector import DEBERTA_SHA256

    return OnnxDebertaDetector(config.model_dir, detector_id=config.detector_id,
                               detector_version=config.detector_version, threshold=config.threshold,
                               max_len=config.max_len, intra_op_threads=config.intra_op_threads,
                               expected_sha256=DEBERTA_SHA256 if config.verify_model_sha256 else None)


def create_app(config: GatewayConfig | None = None, detector: Detector | None = None) -> FastAPI:
    """Build the app. Without `detector`, the ONNX DeBERTa model is loaded from config.model_dir at startup."""
    config = config or GatewayConfig.from_env()
    metrics = GatewayMetrics()
    state: dict = {"service": None, "load_error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            det = detector if detector is not None else load_onnx_detector(config)
            service = ClassifierService(det, config, metrics)
            await service.start()
            state["service"] = service
        except Exception as exc:  # keep serving /health so the failure is visible
            log.exception("detector failed to load")
            state["load_error"] = f"{type(exc).__name__}: {exc}"
        yield
        if state["service"] is not None:
            await state["service"].stop()

    app = FastAPI(title="Sentinel gateway", version="0.2.0", lifespan=lifespan)
    app.state.gateway = state
    app.state.config = config
    app.state.metrics = metrics

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> ClassifyResponse:
        service: ClassifierService | None = state["service"]
        if service is None:
            # No detector at all is not a per-detector timeout/error; refuse rather than guess a decision.
            raise HTTPException(status_code=503, detail=f"detector not loaded: {state['load_error']}")
        if req.request_id is not None and len(req.request_id) > MAX_REQUEST_ID_CHARS:
            raise HTTPException(status_code=422, detail=f"request_id longer than {MAX_REQUEST_ID_CHARS} characters")
        if len(req.text) > config.max_text_chars:
            raise HTTPException(status_code=413, detail=f"text longer than {config.max_text_chars} characters")
        outcome = await service.classify(req.text, req.request_id)
        return outcome.to_response()  # validated against contracts.ClassifyResponse before it leaves

    @app.get("/health")
    async def health(response: Response) -> dict:
        service: ClassifierService | None = state["service"]
        if service is None:
            response.status_code = 503
            return {"status": "unavailable", "model_loaded": False, "error": state["load_error"]}
        d = service.detector
        return {
            "status": "ok", "model_loaded": True, "schema_version": SCHEMA_VERSION,
            "detector": {"detector_id": d.detector_id, "detector_version": d.detector_version,
                         "threshold": d.threshold, "max_len": d.max_len,
                         "model_sha256": getattr(d, "model_sha256", None)},
            "config": {"stride": config.stride, "max_chunks": config.max_chunks, "timeout_ms": config.timeout_ms,
                       "failure_policy": config.failure_policy, "max_batch": config.max_batch,
                       "max_wait_ms": config.max_wait_ms, "workers": config.workers, "max_queue": config.max_queue,
                       "max_inflight": config.max_inflight, "max_inflight_chars": config.max_inflight_chars,
                       "tokenize_workers": config.tokenize_workers},
            "queue_depth": service.batcher.queue_depth, "busy_workers": service.batcher.busy_workers,
            "inflight_requests": service.inflight,
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        service: ClassifierService | None = state["service"]
        if service is not None:
            metrics.queue_depth.set(service.batcher.queue_depth)
            metrics.busy_workers.set(service.batcher.busy_workers)
            metrics.inflight.set(service.inflight)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def __getattr__(name: str):
    # `uvicorn serving.app:app` gets a lazily built app, so importing this module (tests) loads no model.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
