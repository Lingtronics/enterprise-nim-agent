"""Async NVIDIA NIM adapter with bounded transient-error retries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.config.settings import Settings
from app.errors import RuntimeErrorBase
from app.models.types import Generation
from app.telemetry.instrumentation import span


class NIMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if settings.nim_api_key is None or not settings.nim_api_key.get_secret_value().strip():
            raise RuntimeErrorBase(
                code="model_configuration",
                message="NIM_API_KEY is required to initialize the NIM model client.",
                status_code=503,
                retriable=False,
            )
        if client is not None and transport is not None:
            raise ValueError("Provide either an HTTP client or transport, not both")
        self.settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            transport=transport,
            timeout=settings.nim_timeout_seconds,
            follow_redirects=False,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generation:
        with span("nim.request", model=self.settings.nim_model):
            return await self._generate(messages, tools)

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generation:
        if not messages:
            raise RuntimeErrorBase(
                code="model_request_invalid",
                message="At least one model message is required.",
                status_code=422,
                retriable=False,
            )
        payload: dict[str, Any] = {
            "model": self.settings.nim_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        started = time.perf_counter()
        for attempt in range(self.settings.nim_max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.settings.nim_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.nim_api_key.get_secret_value()}",
                        "Accept": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.nim_timeout_seconds,
                )
                self._check_status(response)
                result = self._parse_response(response)
                result.duration_ms = (time.perf_counter() - started) * 1000
                return result
            except httpx.TimeoutException:
                error = RuntimeErrorBase(
                    code="model_timeout",
                    message="The NIM endpoint timed out.",
                    status_code=504,
                    retriable=True,
                )
            except httpx.RequestError:
                error = RuntimeErrorBase(
                    code="model_unavailable",
                    message="The NIM endpoint could not be reached.",
                    status_code=503,
                    retriable=True,
                )
            except RuntimeErrorBase as exc:
                error = exc
            if not error.retriable or attempt >= self.settings.nim_max_retries:
                raise error from None
            await asyncio.sleep(min(self.settings.nim_backoff_seconds * (2**attempt), 30))
        raise AssertionError("bounded retry loop must return or raise")

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status in {401, 403}:
            raise RuntimeErrorBase(
                code="model_authentication",
                message="The NIM endpoint rejected the configured credentials.",
                status_code=502,
                retriable=False,
            )
        if status == 408:
            raise RuntimeErrorBase(
                code="model_timeout",
                message="The NIM endpoint timed out.",
                status_code=504,
                retriable=True,
            )
        if status == 429 or status >= 500:
            raise RuntimeErrorBase(
                code="model_unavailable",
                message="The NIM endpoint is temporarily unavailable.",
                status_code=503,
                retriable=True,
            )
        raise RuntimeErrorBase(
            code="model_request_invalid",
            message="The NIM endpoint rejected the model request.",
            status_code=502,
            retriable=False,
        )

    def _parse_response(self, response: httpx.Response) -> Generation:
        try:
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content")
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list) or any(
                not isinstance(call, dict) for call in tool_calls
            ):
                raise ValueError("invalid tool calls")
            if content is None and tool_calls:
                content = ""
            if not isinstance(content, str) or (not content.strip() and not tool_calls):
                raise ValueError("empty or invalid model content")
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            token_usage = {
                key: value
                for key, value in usage.items()
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            }
            model = data.get("model", self.settings.nim_model)
            if not isinstance(model, str) or not model:
                raise ValueError("invalid model")
            return Generation(
                content=content,
                model=model,
                usage=token_usage,
                tool_calls=tool_calls,
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeErrorBase(
                code="model_invalid_response",
                message="The NIM endpoint returned an invalid completion response.",
                status_code=502,
                retriable=False,
            ) from None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
