from uuid import uuid4

import pytest

from app.agent.graph import AgentRuntime
from app.approval.service import ApprovalService
from app.config.settings import TokenIdentity
from app.models.router import ModelRouter
from app.tools.registry import ToolRegistry


@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_graph_write_pauses_until_independent_human_decision(settings, decision):
    tools = ToolRegistry(settings.database_path)
    approvals = ApprovalService(tools)
    runtime = AgentRuntime(ModelRouter(demo_mode=True), tools=tools, approvals=approvals)
    operator = TokenIdentity(user_id="operator", scopes=["public", "tools:write", "tools:read"])
    approver = TokenIdentity(
        user_id="approver", scopes=["public", "tools:write", "approvals:decide"]
    )
    query = "Change Project Atlas deployment status to approved."
    request_id = str(uuid4())
    pending = await runtime.run(query, operator, request_id=request_id)
    assert pending["status"] == "pending_approval"
    assert pending["approval_required"]
    assert pending["approval"]["request_id"] == request_id
    assert tools.execute_read(
        "get_deployment_status", {"project_id": "atlas"}, operator.scopes
    )["status"] == "pending"
    terminal = approvals.decide(request_id, decision, approver)
    approved = decision == "approve"
    assert terminal["status"] == ("completed" if approved else "denied")
    current = tools.execute_read(
        "get_deployment_status", {"project_id": "atlas"}, operator.scopes
    )
    assert current["status"] == ("approved" if approved else "pending")
    assert current["version"] == int(approved)
    assert approvals.decide(request_id, decision, approver) == terminal


async def test_ticket_creation_has_one_durable_receipt_after_approval(settings):
    tools = ToolRegistry(settings.database_path)
    approvals = ApprovalService(tools)
    runtime = AgentRuntime(ModelRouter(demo_mode=True), tools=tools, approvals=approvals)
    operator = TokenIdentity(user_id="operator", scopes=["public", "tools:write"])
    approver = TokenIdentity(
        user_id="approver", scopes=["public", "tools:write", "approvals:decide"]
    )
    request_id = str(uuid4())
    pending = await runtime.run(
        "Create a deployment ticket for Project Atlas: investigate readiness probes",
        operator,
        request_id=request_id,
    )
    assert pending["tool_name"] == "create_deployment_ticket"
    connection = tools.storage.connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0
    finally:
        connection.close()
    result = approvals.decide(request_id, "approve", approver)
    assert result["tool_result"]["ticket_id"] == request_id
    assert result["tool_result"]["description"] == "investigate readiness probes"
    reopened = ApprovalService(ToolRegistry(settings.database_path))
    assert reopened.decide(request_id, "approve", approver) == result
    connection = tools.storage.connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM tool_executions").fetchone()[0] == 1
        assert {
            row["event"] for row in connection.execute("SELECT event FROM audit")
        } == {"approval_decision", "tool_executed"}
    finally:
        connection.close()


async def test_read_graph_avoids_approval_storage(settings):
    tools = ToolRegistry(settings.database_path)
    approvals = ApprovalService(tools)
    runtime = AgentRuntime(ModelRouter(demo_mode=True), tools=tools, approvals=approvals)
    result = await runtime.run(
        "What is the current deployment status of Project Atlas?",
        TokenIdentity(user_id="reader", scopes=["public", "tools:read"]),
    )
    assert result["status"] == "completed"
    assert not result["approval_required"]
    connection = tools.storage.connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0
    finally:
        connection.close()
