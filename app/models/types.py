"""Model-provider-independent generation contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Generation:
    content: str
    model: str
    path: str = "primary"
    duration_ms: float = 0
    usage: dict[str, int] = field(default_factory=dict)
    ttft_ms: float | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True


class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generation: ...

    async def aclose(self) -> None: ...
