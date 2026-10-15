import json

import httpx
import pytest

from app.config.settings import Settings
from app.errors import RuntimeErrorBase
from app.models.nim_client import NIMClient


def configuration(**kwargs):
    return Settings(
        api_tokens_json=json.dumps({"test-token": {"user_id": "alice", "scopes": ["public"]}}),
        nim_api_key="example-secret",
        nim_backoff_seconds=0,
        _env_file=None,
        **kwargs,
    )


def completion(content="Grounded answer."):
    return {
        "model": "meta/llama-3.1-8b-instruct",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


@pytest.mark.asyncio
async def test_nim_request_and_metadata():
    def handler(request):
        assert str(request.url) == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer example-secret"
        payload = json.loads(request.content)
        assert payload["model"] == "meta/llama-3.1-8b-instruct"
        assert payload["messages"] == [{"role": "user", "content": "Question"}]
        assert payload["stream"] is False
        return httpx.Response(200, json=completion())

    client = NIMClient(configuration(), transport=httpx.MockTransport(handler))
    try:
        result = await client.generate([{"role": "user", "content": "Question"}])
        assert result.content == "Grounded answer."
        assert result.path == "primary"
        assert result.usage["total_tokens"] == 15
        assert result.duration_ms >= 0
        assert result.ttft_ms is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503) if attempts < 3 else httpx.Response(200, json=completion())

    client = NIMClient(configuration(nim_max_retries=2), transport=httpx.MockTransport(handler))
    try:
        await client.generate([{"role": "user", "content": "Question"}])
        assert attempts == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_authentication_error_is_permanent_and_redacted():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, text="example-secret leaked by upstream")

    client = NIMClient(configuration(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeErrorBase) as caught:
            await client.generate([{"role": "user", "content": "Question"}])
        assert caught.value.code == "model_authentication"
        assert caught.value.retriable is False
        assert "example-secret" not in str(caught.value)
        assert attempts == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_timeout_exhausts_bounded_retries():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("provider timed out", request=request)

    client = NIMClient(configuration(nim_max_retries=2), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeErrorBase) as caught:
            await client.generate([{"role": "user", "content": "Question"}])
        assert caught.value.code == "model_timeout"
        assert caught.value.retriable is True
        assert attempts == 3
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": [{"message": {"content": 123}}]}]
)
async def test_malformed_completion_is_permanent(body):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json=body)

    client = NIMClient(configuration(), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeErrorBase) as caught:
            await client.generate([{"role": "user", "content": "Question"}])
        assert caught.value.code == "model_invalid_response"
        assert caught.value.retriable is False
        assert attempts == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tool_only_completion_and_schema_pass_through():
    calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "get_deployment_status", "arguments": '{"project_id":"atlas"}'},
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "get_deployment_status", "parameters": {"type": "object"}},
        }
    ]

    def handler(request):
        assert json.loads(request.content)["tools"] == tools
        return httpx.Response(
            200, json={"choices": [{"message": {"content": None, "tool_calls": calls}}]}
        )

    client = NIMClient(configuration(), transport=httpx.MockTransport(handler))
    try:
        result = await client.generate([{"role": "user", "content": "Status?"}], tools=tools)
        assert result.content == ""
        assert result.tool_calls == calls
    finally:
        await client.aclose()


def test_nim_client_rejects_missing_credentials_even_in_demo_configuration():
    settings = Settings(
        api_tokens_json='{"token":{"user_id":"alice","scopes":[]}}',
        nim_api_key=None,
        _env_file=None,
    )
    with pytest.raises(RuntimeErrorBase, match="NIM_API_KEY"):
        NIMClient(settings)
