"""Server-side approval policy independent of model output and caller privileges."""

from app.errors import RuntimeErrorBase

READ_TOOLS = frozenset({"get_deployment_status"})
WRITE_TOOLS = frozenset({"update_deployment_status", "create_deployment_ticket"})


def requires_approval(tool_name: str) -> bool:
    if tool_name in WRITE_TOOLS:
        return True
    if tool_name in READ_TOOLS:
        return False
    raise RuntimeErrorBase("unknown_tool", "The requested tool is not registered.", 422)
