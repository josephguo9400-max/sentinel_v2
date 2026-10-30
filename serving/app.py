"""Sentinel gateway: FastAPI app exposing POST /classify, GET /health, GET /metrics and POST /admin/reload.

Run: uvicorn serving.app:app --host 127.0.0.1 --port 8200   (config via SENTINEL_* env vars, see config.py)

Detectors come from the registry file (default artifacts/registry.json, see serving/registry.py). Load rules:
- startup: Stage 2 is required; without a verified Stage 2 the gateway answers 503 (no failure policy covers "no
  detector"). A rejected Stage 1 or shadow is reported in /health ("status": "degraded") and the gateway serves the
  rest (Stage 2 alone scores every chunk, so dropping Stage 1 costs compute, not coverage).
- reload (POST /admin/reload, or every `registry_poll_s` when the file's sha256 changes): all or nothing. If any
  pointer is rejected, the new registry is not applied, the last good configuration keeps serving, and /health
  lists the rejections under "last_reload".
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from contracts import SCHEMA_VERSION, ClassifyRequest, ClassifyResponse

from .config import GatewayConfig
from .metrics import GatewayMetrics
from .registry import DetectorSpec, ModelCache, RegistryError, load_registry
from .service import ClassifierService

log = logging.getLogger("sentinel.serving")

MAX_REQUEST_ID_CHARS = 200


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spec_from_detector(det: Any, config: GatewayConfig, *, role: str = "single", kind: str = "token",
                       gray_zone=None, input_languages=None, failure_policy=None, timeout_ms=None) -> DetectorSpec:
    """Wrap an injected detector object (tests) in a DetectorSpec using the config's injected-only fields."""
    if isinstance(det, DetectorSpec):
        return det
    if kind == "token" and getattr(det, "max_len", config.max_len) != config.max_len:
        raise ValueError(f"detector max_len {det.max_len} != config max_len {config.max_len}")
    return DetectorSpec(detector=det, detector_id=det.detector_id, detector_version=det.detector_version,
                        threshold=det.threshold, kind=kind, role=role,
                        failure_policy=failure_policy or config.failure_policy,
                        timeout_ms=timeout_ms or config.timeout_ms, gray_zone=gray_zone,
                        input_languages=input_languages, max_input_tokens=getattr(det, "max_len", 512),
                        tokenizer=getattr(det, "tokenizer_name", None), manifest_schema=None,
                        artifact_dir=None, sha256=getattr(det, "model_sha256", None))


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def create_app(config: GatewayConfig | None = None, detector: Any = None, *, stage1: Any = None,
               shadow: Any = None, language: Any = None, stage1_mode: str = "route") -> FastAPI:
    """Build the app. Without `detector`, detectors are loaded from the registry at startup.

    `detector` / `stage1` / `shadow` inject detector objects or DetectorSpecs (tests); `language` injects a
    language identifier; `stage1_mode` applies to injected detectors only (the registry carries its own).
    """
    config = config or GatewayConfig.from_env()
    metrics = GatewayMetrics()
    cache = ModelCache()
    state: dict[str, Any] = {"service": None, "load_error": None, "rejections": [], "pointers": None,
                             "registry_sha256": None, "loaded_at": None, "last_reload": None, "source": None}
    lock = asyncio.Lock()

    def build_from_registry() -> tuple[dict[str, DetectorSpec], list, dict, str | None]:
        reg_path = Path(config.registry_path)
        sha = _file_sha(reg_path)
        loaded = load_registry(reg_path, Path(config.artifacts_root), intra_op_threads=config.intra_op_threads,
                               intra_op_by_pointer={"shadow": config.shadow_intra_op_threads}, cache=cache,
                               known_tokenizers=config.known_tokenizers)
        for rj in loaded.rejections:
            metrics.registry_rejections.labels(rj.pointer, rj.code).inc()
        return loaded.specs, [r.as_dict() for r in loaded.rejections], loaded.pointers, sha

    async def install(specs: dict[str, DetectorSpec], mode: str) -> None:
        service = ClassifierService(specs["stage2"], config, metrics, stage1=specs.get("stage1"),
                                    shadow=specs.get("shadow"), language=language, stage1_mode=mode)
        await service.start()
        old = state["service"]
        state["service"] = service
        state["loaded_at"] = _now()
        if old is not None:
            asyncio.get_running_loop().create_task(_retire(old))

    async def _retire(old: ClassifierService) -> None:
        deadline = time.monotonic() + 30
        while old.inflight and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        await old.stop()

    async def initial_load() -> None:
        if detector is not None:
            specs = {"stage2": spec_from_detector(detector, config)}
            if stage1 is not None:
                specs["stage1"] = stage1 if isinstance(stage1, DetectorSpec) else spec_from_detector(
                    stage1, config, role="stage1", kind="text", gray_zone=(0.2, 0.8))
            if shadow is not None:
                specs["shadow"] = spec_from_detector(shadow, config)
            state["source"] = "injected"
            await install(specs, stage1_mode)
            return
        state["source"] = "registry"
        try:
            specs, rejections, pointers, sha = await asyncio.to_thread(build_from_registry)
        except RegistryError as exc:
            log.error("registry load failed: %s", exc)
            state["load_error"] = str(exc)
            metrics.registry_loads.labels("failed").inc()
            return
        state["rejections"], state["pointers"], state["registry_sha256"] = rejections, pointers, sha
        if "stage2" not in specs:
            state["load_error"] = "stage2 rejected: " + "; ".join(
                f"{r['code']}: {r['detail']}" for r in rejections if r["pointer"] == "stage2")
            metrics.registry_loads.labels("failed").inc()
            return
        await install(specs, pointers["stage1_mode"])
        metrics.registry_loads.labels("degraded" if rejections else "ok").inc()

    async def reload() -> dict:
        async with lock:
            if state["source"] == "injected":
                return {"applied": False, "reason": "detectors were injected; nothing to reload"}
            try:
                specs, rejections, pointers, sha = await asyncio.to_thread(build_from_registry)
            except RegistryError as exc:
                rep = {"at": _now(), "applied": False, "error": str(exc), "rejections": []}
                state["last_reload"] = rep
                metrics.registry_loads.labels("rejected").inc()
                log.error("registry reload rejected, keeping last good config: %s", exc)
                return rep
            if rejections or "stage2" not in specs:
                rep = {"at": _now(), "applied": False, "registry_sha256": sha, "pointers": pointers,
                       "rejections": rejections,
                       "error": "rejected; the last good configuration keeps serving"}
                state["last_reload"] = rep
                metrics.registry_loads.labels("rejected").inc()
                log.error("registry reload rejected, keeping last good config: %s", rejections)
                if state["service"] is None:  # nothing to keep: take what verified, like startup
                    if "stage2" in specs:
                        await install(specs, pointers["stage1_mode"])
                        state["load_error"] = None
                return rep
            await install(specs, pointers["stage1_mode"])
            state.update(rejections=[], pointers=pointers, registry_sha256=sha, load_error=None)
            rep = {"at": _now(), "applied": True, "registry_sha256": sha, "pointers": pointers, "rejections": []}
            state["last_reload"] = rep
            metrics.registry_loads.labels("ok").inc()
            log.info("registry reloaded: %s", pointers)
            return rep

    async def poll() -> None:
        while True:
            await asyncio.sleep(config.registry_poll_s)
            sha = _file_sha(Path(config.registry_path))
            if sha is not None and sha != state["registry_sha256"] and sha != (state["last_reload"] or {}).get(
                    "registry_sha256"):
                await reload()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await initial_load()
        except Exception as exc:  # keep serving /health so the failure is visible
            log.exception("detector failed to load")
            state["load_error"] = f"{type(exc).__name__}: {exc}"
        poller = asyncio.create_task(poll()) if config.registry_poll_s > 0 and state["source"] == "registry" else None
        yield
        if poller is not None:
            poller.cancel()
        if state["service"] is not None:
            await state["service"].stop()

    app = FastAPI(title="Sentinel gateway", version="0.3.0", lifespan=lifespan)
    app.state.gateway = state
    app.state.config = config
    app.state.metrics = metrics
    app.state.reload = reload

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

    @app.post("/admin/reload")
    async def admin_reload(response: Response) -> dict:
        rep = await reload()
        if not rep.get("applied"):
            response.status_code = 409
        return rep

    @app.get("/health")
    async def health(response: Response) -> dict:
        service: ClassifierService | None = state["service"]
        registry_info = {"path": config.registry_path, "sha256": state["registry_sha256"],
                         "pointers": state["pointers"], "rejections": state["rejections"],
                         "last_reload": state["last_reload"], "loaded_at": state["loaded_at"],
                         "source": state["source"]}
        if service is None:
            response.status_code = 503
            return {"status": "unavailable", "model_loaded": False, "error": state["load_error"],
                    "registry": registry_info}
        d = service.stage2
        degraded = bool(state["rejections"]) or bool((state["last_reload"] or {}).get("applied") is False)
        return {
            "status": "degraded" if degraded else "ok", "model_loaded": True, "schema_version": SCHEMA_VERSION,
            "detector": {**d.describe(), "max_len": d.detector.max_len},  # active Stage 2 (decides by default)
            "stage1": service.stage1.describe() if service.stage1 else None,
            "stage1_mode": service.stage1_mode if service.stage1 else None,
            "stage1_languages": service.stage1_languages,
            "language_detector": service.language.describe() if service.language is not None else None,
            "shadow": service.shadow.describe() if service.shadow else None,
            "registry": registry_info,
            "config": {"timeout_ms": service.request_budget_ms, "stage2_timeout_ms": int(service.s2_timeout_s * 1000),
                       "stage1_timeout_ms": int(service.s1_timeout_s * 1000) if service.stage1 else None,
                       "stage1_soft_deadline_ms": round(service.s1_soft_budget_s * 1000, 3) if service.stage1
                       else None,
                       "stride": service.stride, "max_chunks": config.max_chunks,
                       "stage1_max_chunks": config.stage1_max_chunks,
                       "stage1_finalize_flag": config.stage1_finalize_flag,
                       "structure_aware": config.structure_aware,
                       "coverage_policy": config.coverage_policy,
                       "deep_scan_max_chunks": config.deep_scan_max_chunks,
                       "deep_scan_timeout_ms": config.deep_scan_timeout_ms,
                       "deep_scan_fallback": config.deep_scan_fallback,
                       "failure_policy": d.failure_policy, "max_batch": config.max_batch,
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
            if service.shadow is not None:
                metrics.shadow_queue_depth.set(service.shadow.queue_depth)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def __getattr__(name: str):
    # `uvicorn serving.app:app` gets a lazily built app, so importing this module (tests) loads no model.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
