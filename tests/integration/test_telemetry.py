import json
import logging
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from app.api.main import create_app
from app.errors import RuntimeErrorBase
from app.models.nim_client import NIMClient

READER = {"Authorization": "Bearer test-reader-token"}
OPERATOR = {"Authorization": "Bearer test-operator-token"}
APPROVER = {"Authorization": "Bearer test-approver-token"}
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
INBOUND_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
INBOUND_PARENT = "00f067aa0ba902b7"
SECOND_TRACE = "0af7651916cd43dd8448eb211c80319c"


@pytest.fixture(autouse=True)
def capture_runtime_logs(caplog):
    logger = logging.getLogger("enterprise_agent")
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def configured_nim(settings):
    return settings.model_copy(
        update={
            "nim_api_key": SecretStr("PRIVATE_NIM_CREDENTIAL_telemetry"),
            "nim_max_retries": 1,
            "nim_backoff_seconds": 0,
        }
    )


def grounded_completion(request):
    payload = json.loads(request.content)
    context = json.loads(payload["messages"][-1]["content"])
    document = next(
        item
        for item in context["documents"]
        if item["metadata"]["document_id"] == "atlas-requirements"
    )
    quote = "Project Atlas requires GPU inference in the production and staging environments."
    assert quote in document["content"]
    content = json.dumps(
        {"evidence": [{"chunk_id": document["metadata"]["chunk_id"], "quote": quote}]}
    )
    return httpx.Response(
        200,
        json={
            "model": "meta/llama-3.1-8b-instruct",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
    )


def assert_correlated(spans, request_id, expected_trace=None):
    assert spans
    trace_ids = {f"{item.context.trace_id:032x}" for item in spans}
    assert len(trace_ids) == 1
    if expected_trace is not None:
        assert trace_ids == {expected_trace}
    for item in spans:
        assert REQUIRED_ATTRIBUTES <= set(item.attributes)
        assert item.attributes["request_id"] == request_id
        assert isinstance(item.attributes["success"], bool)
        assert item.attributes["latency_ms"] >= 0
    return next(iter(trace_ids))


def serialized(spans):
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


def test_rag_and_nim_propagate_w3c_trace_without_exporting_content(settings, caplog):
    exporter = InMemorySpanExporter()
    model = NIMClient(configured_nim(settings), transport=httpx.MockTransport(grounded_completion))
    request_id = str(uuid4())
    query = "Summarise Atlas deployment requirements. PRIVATE_QUERY_telemetry"
    with TestClient(create_app(settings, model, span_exporter=exporter)) as client:
        response = client.post(
            "/v1/query",
            headers={
                **READER,
                "traceparent": f"00-{INBOUND_TRACE}-{INBOUND_PARENT}-01",
            },
            json={"query": query, "request_id": request_id},
        )
        assert response.status_code == 200
        assert response.json()["trace_id"] == INBOUND_TRACE
        assert response.headers["X-Trace-ID"] == INBOUND_TRACE
        assert response.headers["X-Request-ID"] == request_id
    spans = exporter.get_finished_spans()
    assert_correlated(spans, request_id, INBOUND_TRACE)
    names = {item.name for item in spans}
    assert {
        "http.request",
        "agent.workflow",
        "model.generate",
        "nim.request",
        "retrieval.search",
    } <= names
    inbound = next(item for item in spans if item.name == "http.request")
    assert inbound.parent.span_id == int(INBOUND_PARENT, 16)
    model_span = next(item for item in spans if item.name == "model.generate")
    assert model_span.attributes["model"] == settings.nim_model
    assert model_span.attributes["model.path"] == "primary"
    assert (
        next(item for item in spans if item.name == "nim.request").attributes["model"]
        == settings.nim_model
    )
    assert (
        next(item for item in spans if item.name == "retrieval.search").attributes[
            "retrieval_count"
        ]
        > 0
    )
    exported = serialized(spans) + caplog.text
    assert '"event": "api_request"' in caplog.text
    for sensitive in (
        query,
        "PRIVATE_QUERY_telemetry",
        "PRIVATE_NIM_CREDENTIAL_telemetry",
        "test-reader-token",
        "Project Atlas requires GPU inference in the production and staging environments.",
    ):
        assert sensitive not in exported


def test_read_tool_has_its_own_trace_and_request_correlation(settings):
    exporter = InMemorySpanExporter()
    request_id = str(uuid4())
    with TestClient(create_app(settings, span_exporter=exporter)) as client:
        response = client.post(
            "/v1/tools/call",
            headers=READER,
            json={
                "request_id": request_id,
                "name": "get_deployment_status",
                "arguments": {"project_id": "atlas"},
            },
        )
        assert response.status_code == 200
    spans = exporter.get_finished_spans()
    current_trace = assert_correlated(spans, request_id)
    assert response.json()["trace_id"] == current_trace
    assert response.headers["X-Trace-ID"] == current_trace
    assert {"http.request", "agent.workflow", "tool.call"} <= {item.name for item in spans}
    assert "model.generate" not in {item.name for item in spans}
    tool_span = next(item for item in spans if item.name == "tool.call")
    assert tool_span.attributes["tool_name"] == "get_deployment_status"
    assert tool_span.attributes["approval_required"] is False


def test_approval_decision_uses_current_trace_and_original_action_correlation(settings):
    exporter = InMemorySpanExporter()
    request_id = str(uuid4())
    with TestClient(create_app(settings, span_exporter=exporter)) as client:
        pending = client.post(
            "/v1/query",
            headers=OPERATOR,
            json={
                "query": "Change Project Atlas deployment status to approved.",
                "request_id": request_id,
            },
        )
        assert pending.status_code == 200
        original_trace = pending.json()["trace_id"]
        first_spans = exporter.get_finished_spans()
        assert_correlated(first_spans, request_id, original_trace)
        exporter.clear()
        approved = client.post(
            f"/v1/approvals/{request_id}",
            headers={
                **APPROVER,
                "traceparent": f"00-{SECOND_TRACE}-{INBOUND_PARENT}-01",
            },
            json={"decision": "approve"},
        )
        assert approved.status_code == 200
        assert approved.json()["trace_id"] == original_trace
        assert approved.headers["X-Trace-ID"] == SECOND_TRACE
        assert approved.headers["X-Request-ID"] == request_id
    decision_spans = exporter.get_finished_spans()
    assert_correlated(decision_spans, request_id, SECOND_TRACE)
    assert {"http.request", "approval.decision", "tool.call"} <= {
        item.name for item in decision_spans
    }
    decision = next(item for item in decision_spans if item.name == "approval.decision")
    assert decision.attributes["approval_required"] is True
    assert decision.attributes["tool_name"] == "update_deployment_status"


def test_nim_timeout_fallback_exports_failed_primary_span_with_bounded_attempts(settings, caplog):
    exporter = InMemorySpanExporter()
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("PRIVATE_PROVIDER_TIMEOUT_telemetry", request=request)

    model = NIMClient(configured_nim(settings), transport=httpx.MockTransport(handler))
    request_id = str(uuid4())
    with TestClient(create_app(settings, model, span_exporter=exporter)) as client:
        response = client.post(
            "/v1/query",
            headers=READER,
            json={
                "query": "Summarise Atlas deployment requirements.",
                "request_id": request_id,
            },
        )
        assert response.status_code == 200
        assert response.json()["model"]["path"] == "fallback"
        assert attempts == 2
    spans = exporter.get_finished_spans()
    assert_correlated(spans, request_id)
    nim_spans = [item for item in spans if item.name == "nim.request"]
    assert nim_spans
    assert all(item.attributes["success"] is False for item in nim_spans)
    assert all(item.attributes["error.code"] == "model_timeout" for item in nim_spans)
    assert all(item.status.status_code is StatusCode.ERROR for item in nim_spans)
    assert all(not item.events for item in nim_spans)
    assert (
        next(item for item in spans if item.name == "model.generate").attributes["model.path"]
        == "fallback"
    )
    assert "PRIVATE_PROVIDER_TIMEOUT_telemetry" not in serialized(spans) + caplog.text
    assert "PRIVATE_NIM_CREDENTIAL_telemetry" not in serialized(spans) + caplog.text


def test_permanent_model_failure_marks_error_spans_without_provider_body(settings, caplog):
    exporter = InMemorySpanExporter()
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            401, text="PRIVATE_PROVIDER_BODY_telemetry PRIVATE_NIM_CREDENTIAL_telemetry"
        )

    model = NIMClient(configured_nim(settings), transport=httpx.MockTransport(handler))
    request_id = str(uuid4())
    with TestClient(create_app(settings, model, span_exporter=exporter)) as client:
        response = client.post(
            "/v1/query",
            headers=READER,
            json={
                "query": "Atlas requirements",
                "request_id": request_id,
            },
        )
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "model_authentication"
        assert attempts == 1
    spans = exporter.get_finished_spans()
    assert_correlated(spans, request_id)
    for name in ("http.request", "agent.workflow", "model.generate", "nim.request"):
        assert any(item.name == name and item.attributes["success"] is False for item in spans)
    exported = serialized(spans) + caplog.text + response.text
    assert "PRIVATE_PROVIDER_BODY_telemetry" not in exported
    assert "PRIVATE_NIM_CREDENTIAL_telemetry" not in exported


def test_missing_nim_credentials_fail_startup_clearly(settings):
    invalid_test_configuration = settings.model_copy(
        update={"model_mode": "nim", "nim_api_key": None}
    )
    with pytest.raises(RuntimeErrorBase, match="NIM_API_KEY") as caught:
        with TestClient(create_app(invalid_test_configuration)):
            pass
    assert caught.value.code == "model_configuration"
