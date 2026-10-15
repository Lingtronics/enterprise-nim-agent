import pytest

from app.agent.graph import AgentRuntime
from app.config.settings import TokenIdentity
from app.errors import RuntimeErrorBase
from app.models.router import ModelRouter
from app.tools.registry import ToolRegistry


async def test_read_tool_graph(settings):
    tools = ToolRegistry(settings.database_path)
    runtime = AgentRuntime(ModelRouter(demo_mode=True), tools=tools)
    result = await runtime.run(
        "What is the current deployment status of Project Atlas?",
        TokenIdentity(user_id="reader", scopes=["public", "tools:read"]),
    )
    assert result["tool_name"] == "get_deployment_status"
    assert result["tool_result"]["status"] == "pending"
    assert not result["approval_required"]


async def test_missing_approval_service_cannot_write(settings):
    tools = ToolRegistry(settings.database_path)
    runtime = AgentRuntime(ModelRouter(demo_mode=True), tools=tools)
    with pytest.raises(RuntimeErrorBase) as error:
        await runtime.run(
            "Change Project Atlas deployment status to approved.",
            TokenIdentity(user_id="operator", scopes=["public", "tools:write"]),
        )
    assert error.value.code == "approval_unavailable"
    assert (
        tools.execute_read(
            "get_deployment_status", {"project_id": "atlas"}, ["public", "tools:read"]
        )["status"]
        == "pending"
    )
