from typing import Any, Literal, TypedDict


class AgentState(TypedDict, total=False):
    request_id: str
    user_id: str
    scopes: list[str]
    query: str
    request_type: Literal["query", "tool"]
    retrieved_documents: list[dict[str, Any]]
    tool_name: str | None
    tool_arguments: dict[str, Any]
    approval_required: bool
    approval_status: Literal["not_required", "pending", "approved", "denied"]
    model_response: dict[str, Any]
    final_response: dict[str, Any]
    errors: list[dict[str, Any]]
    trace_id: str
    timings_ms: dict[str, float]
    tool_result: dict[str, Any]
    pending_action: dict[str, Any]
