import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

_request_id = ContextVar("request_id", default="")
_tracer = trace.get_tracer("enterprise-nim-agent")
_allowed = {
    "request_id",
    "model",
    "tool_name",
    "retrieval_count",
    "approval_required",
    "approval_status",
    "latency_ms",
    "success",
    "error.code",
    "http.method",
    "http.status_code",
    "model.path",
}
logger = logging.getLogger("enterprise_agent")


def configure(settings, span_exporter=None):
    global _tracer
    provider = TracerProvider(resource=Resource.create({"service.name": "enterprise-nim-agent"}))
    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif settings.otel_exporter_otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=settings.otel_exporter_otlp_endpoint,
                    timeout=3,
                )
            )
        )
    _tracer = provider.get_tracer("enterprise-nim-agent", "0.1.0")
    logger.setLevel(settings.log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return provider


@contextmanager
def bind_request(request_id):
    token = _request_id.set(request_id)
    try:
        yield
    finally:
        _request_id.reset(token)


def trace_id():
    return f"{trace.get_current_span().get_span_context().trace_id:032x}"


@contextmanager
def span(name, *, context=None, **attributes):
    defaults = {
        "request_id": _request_id.get(),
        "model": "",
        "tool_name": "",
        "retrieval_count": 0,
        "approval_required": False,
        "approval_status": "not_required",
        "success": True,
    }
    defaults.update({k: v for k, v in attributes.items() if k in _allowed and v is not None})
    started = time.perf_counter()
    with _tracer.start_as_current_span(
        name,
        context=context,
        attributes=defaults,
        record_exception=False,
        set_status_on_exception=False,
    ) as current:
        try:
            yield current
        except BaseException as exc:
            current.set_attribute("success", False)
            current.set_attribute("error.code", getattr(exc, "code", "internal_error"))
            current.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            current.set_attribute("latency_ms", (time.perf_counter() - started) * 1000)
            if getattr(current, "attributes", {}).get("success") is False:
                current.set_status(Status(StatusCode.ERROR))


def log_request(request_id, current_trace, status, latency_ms, success=None):
    logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "api_request",
                "request_id": request_id,
                "trace_id": current_trace,
                "status_code": status,
                "latency_ms": round(latency_ms, 3),
                "success": status < 400 if success is None else success,
            }
        )
    )
