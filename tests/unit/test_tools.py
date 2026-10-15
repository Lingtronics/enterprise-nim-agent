import json

import pytest

from app.errors import RuntimeErrorBase
from app.tools.registry import ToolRegistry

READER = ["public", "tools:read"]
WRITER = ["public", "tools:read", "tools:write"]


@pytest.fixture
def tools(tmp_path):
    return ToolRegistry(tmp_path / "tools.db")


def test_unknown_tool_is_rejected(tools):
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.validate("shell_command", {"command": "echo unsafe"}, WRITER)
    assert caught.value.code == "unknown_tool"


@pytest.mark.parametrize(
    "arguments",
    [
        {"project_id": "atlas", "new_status": "invalid"},
        {"project_id": "atlas", "new_status": "approved", "approved": True},
        {"project_id": 123, "new_status": "approved"},
        {"project_id": "atlas"},
    ],
)
def test_invalid_schema_rejected(tools, arguments):
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.validate("update_deployment_status", arguments, WRITER)
    assert caught.value.code == "tool_arguments_invalid"


def test_canonical_arguments_and_structured_read(tools):
    assert tools.validate("get_deployment_status", {"project_id": " Atlas "}, READER) == {
        "project_id": "atlas"
    }
    assert tools.execute_read("get_deployment_status", {"project_id": "atlas"}, READER) == {
        "project_id": "atlas",
        "status": "pending",
        "version": 0,
    }


@pytest.mark.parametrize(
    "name,arguments,scopes",
    [
        ("get_deployment_status", {"project_id": "atlas"}, ["public"]),
        ("update_deployment_status", {"project_id": "atlas", "new_status": "approved"}, READER),
        ("get_deployment_status", {"project_id": "orion"}, READER),
    ],
)
def test_scope_permissions_enforced(tools, name, arguments, scopes):
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.validate(name, arguments, scopes)
    assert caught.value.code == "tool_forbidden"


def test_unknown_project_rejected(tools):
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.validate("get_deployment_status", {"project_id": "missing"}, READER)
    assert caught.value.code == "project_not_found"


def test_read_write_policy_and_direct_write_denial(tools):
    assert tools.requires_approval("get_deployment_status") is False
    assert tools.requires_approval("update_deployment_status") is True
    assert tools.requires_approval("create_deployment_ticket") is True
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.execute_read(
            "update_deployment_status", {"project_id": "atlas", "new_status": "approved"}, WRITER
        )
    assert caught.value.code == "approval_required"
    assert (
        tools.execute_read("get_deployment_status", {"project_id": "atlas"}, READER)["status"]
        == "pending"
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "What is the current deployment status of Project Atlas?",
            ("get_deployment_status", {"project_id": "atlas"}),
        ),
        ("Show Zephyr deployment status", ("get_deployment_status", {"project_id": "zephyr"})),
        (
            "Change Project Atlas deployment status to approved.",
            ("update_deployment_status", {"project_id": "atlas", "new_status": "approved"}),
        ),
        (
            "Please set Orion status to failed",
            ("update_deployment_status", {"project_id": "orion", "new_status": "failed"}),
        ),
        (
            "Create a deployment ticket for Project Atlas: Roll out the GPU pilot",
            (
                "create_deployment_ticket",
                {"project_id": "atlas", "description": "Roll out the GPU pilot"},
            ),
        ),
    ],
)
def test_natural_tool_selection(tools, query, expected):
    assert tools.select(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Update Atlas and Zephyr deployment status to approved",
        "Update deployment status to approved",
        "Update Atlas deployment status to approved and deploy it",
        "Update Atlas deployment status to approved and set status to failed",
        "Show Atlas deployment status and update it to approved",
        "Create a deployment ticket for Atlas and update status to approved",
        "Explain deployment status for Atlas",
        "Update Atlas deployment status",
    ],
)
def test_ambiguous_or_explanatory_queries_rejected(tools, query):
    with pytest.raises(RuntimeErrorBase) as caught:
        tools.select(query)
    assert caught.value.code == "tool_request_ambiguous"


def test_mcp_contracts_expose_strict_schemas_and_policy(tools):
    contracts = {tool["name"]: tool for tool in tools.list_tools()}
    assert set(contracts) == {
        "get_deployment_status",
        "update_deployment_status",
        "create_deployment_ticket",
    }
    assert contracts["update_deployment_status"]["inputSchema"]["additionalProperties"] is False
    assert contracts["get_deployment_status"]["annotations"]["readOnlyHint"] is True
    assert contracts["create_deployment_ticket"]["approval_required"] is True


def create_capability(
    connection,
    request_id="request-1",
    name="update_deployment_status",
    arguments=None,
    status="approved",
):
    connection.execute(
        "CREATE TABLE IF NOT EXISTS actions(request_id TEXT PRIMARY KEY,tool_name TEXT,"
        "arguments_json TEXT,requester_scopes_json TEXT,status TEXT)"
    )
    connection.execute(
        "INSERT INTO actions VALUES(?,?,?,?,?)",
        (
            request_id,
            name,
            json.dumps(arguments or {"project_id": "atlas", "new_status": "approved"}),
            json.dumps(WRITER),
            status,
        ),
    )


def test_pending_capability_and_missing_transaction_cannot_write(tools):
    connection = tools.storage.connect()
    try:
        arguments = {"project_id": "atlas", "new_status": "approved"}
        with pytest.raises(RuntimeErrorBase) as caught:
            tools.execute_approved(
                connection, "request-1", "update_deployment_status", arguments, WRITER
            )
        assert caught.value.code == "approval_transaction_required"
        create_capability(connection, status="pending")
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeErrorBase) as caught:
            tools.execute_approved(
                connection, "request-1", "update_deployment_status", arguments, WRITER
            )
        assert caught.value.code == "approval_required"
        connection.rollback()
    finally:
        connection.close()
    assert (
        tools.execute_read("get_deployment_status", {"project_id": "atlas"}, READER)["status"]
        == "pending"
    )


def test_approved_tool_receipt_prevents_replay_and_arguments_mismatch(tools):
    connection = tools.storage.connect()
    try:
        arguments = {"project_id": "atlas", "new_status": "approved"}
        create_capability(connection)
        connection.execute("BEGIN IMMEDIATE")
        first = tools.execute_approved(
            connection, "request-1", "update_deployment_status", arguments, WRITER
        )
        second = tools.execute_approved(
            connection, "request-1", "update_deployment_status", arguments, WRITER
        )
        assert first == second == {"project_id": "atlas", "status": "approved", "version": 1}
        with pytest.raises(RuntimeErrorBase) as caught:
            tools.execute_approved(
                connection,
                "request-1",
                "update_deployment_status",
                {"project_id": "atlas", "new_status": "failed"},
                WRITER,
            )
        assert caught.value.code == "approval_mismatch"
        # Registry cannot commit independently from approval state and audit.
        connection.rollback()
    finally:
        connection.close()
    assert (
        tools.execute_read("get_deployment_status", {"project_id": "atlas"}, READER)["version"] == 0
    )


def test_approval_cannot_gain_new_requester_scopes(tools):
    connection = tools.storage.connect()
    try:
        create_capability(connection)
        connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeErrorBase) as caught:
            tools.execute_approved(
                connection,
                "request-1",
                "update_deployment_status",
                {"project_id": "atlas", "new_status": "approved"},
                WRITER + ["internal"],
            )
        assert caught.value.code == "approval_mismatch"
        connection.rollback()
    finally:
        connection.close()


def test_ticket_id_is_deterministic_and_transactional(tools):
    connection = tools.storage.connect()
    try:
        arguments = {"project_id": "atlas", "description": "GPU pilot"}
        create_capability(
            connection,
            request_id="ticket-request",
            name="create_deployment_ticket",
            arguments=arguments,
        )
        connection.execute("BEGIN IMMEDIATE")
        first = tools.execute_approved(
            connection, "ticket-request", "create_deployment_ticket", arguments, WRITER
        )
        second = tools.execute_approved(
            connection, "ticket-request", "create_deployment_ticket", arguments, WRITER
        )
        assert first == second
        assert first["ticket_id"] == "ticket-request"
        assert connection.execute("SELECT count(*) FROM tickets").fetchone()[0] == 1
        connection.commit()
    finally:
        connection.close()
