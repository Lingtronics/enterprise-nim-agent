import json
import logging
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.errors import RuntimeErrorBase
from app.telemetry.instrumentation import bind_request, configure, span, trace_id

REQUIRED_ATTRIBUTES = {
    "request_id",
    "model",
    "tool_name",
    "retrieval_count",
    "approval_required",
    "approval_status",
    "latency_ms",
    "success",
}


@pytest.fixture(autouse=True)
def capture_runtime_logs(caplog):
    # Runtime logs intentionally do not propagate to the application's root logger.
    logger = logging.getLogger("enterprise_agent")
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def span_text(spans):
    return json.dumps(
        [
            {
                "name": item.name,
                "attributes": dict(item.attributes),
                "status": item.status.description,
                "events": [
                    {"name": event.name, "attributes": dict(event.attributes)}
                    for event in item.events
                ],
            }
            for item in spans
        ]
    )


def test_nested_spans_share_trace_and_bound_request(settings):
    exporter = InMemorySpanExporter()
    provider = configure(settings)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    request_id = str(uuid4())
    with bind_request(request_id):
        with span("unit.outer", model="unit-model") as outer:
            current_trace = trace_id()
            with span("unit.inner", retrieval_count=2) as inner:
                assert trace_id() == current_trace
                assert inner.get_span_context().trace_id == outer.get_span_context().trace_id
    provider.force_flush()
    recorded = {item.name: item for item in exporter.get_finished_spans()}
    assert set(recorded) == {"unit.inner", "unit.outer"}
    assert len(current_trace) == 32
    assert recorded["unit.inner"].parent.span_id == recorded["unit.outer"].context.span_id
    for item in recorded.values():
        assert REQUIRED_ATTRIBUTES <= set(item.attributes)
        assert item.attributes["request_id"] == request_id
        assert item.attributes["success"] is True
        assert item.attributes["latency_ms"] >= 0
    provider.shutdown()


def test_unknown_content_attributes_and_exception_messages_are_not_exported(settings, caplog):
    exporter = InMemorySpanExporter()
    provider = configure(settings)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    content = "PRIVATE_USER_CONTENT_telemetry"
    credential = "PRIVATE_CREDENTIAL_telemetry"
    with bind_request(str(uuid4())):
        with pytest.raises(RuntimeErrorBase):
            with span("unit.failure", query=content, authorization=credential):
                raise RuntimeErrorBase(
                    "model_unavailable", f"provider echoed {content} {credential}", 503, True
                )
    provider.force_flush()
    recorded = exporter.get_finished_spans()
    assert len(recorded) == 1
    assert recorded[0].attributes["success"] is False
    assert recorded[0].status.status_code is StatusCode.ERROR
    assert "query" not in recorded[0].attributes
    assert "authorization" not in recorded[0].attributes
    assert not recorded[0].events
    exported = span_text(recorded) + caplog.text
    assert content not in exported
    assert credential not in exported
    provider.shutdown()
