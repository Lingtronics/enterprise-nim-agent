from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from app.approval.service import ApprovalService
from app.config.settings import TokenIdentity
from app.errors import RuntimeErrorBase
from app.tools.registry import ToolRegistry


@pytest.fixture
def tools(settings):
    return ToolRegistry(settings.database_path)


@pytest.fixture
def approvals(tools):
    return ApprovalService(tools)


@pytest.fixture
def operator():
    return TokenIdentity(user_id="operator", scopes=["public", "tools:write", "tools:read"])


@pytest.fixture
def approver():
    return TokenIdentity(
        user_id="approver", scopes=["public", "tools:write", "approvals:decide"]
    )


@pytest.fixture
def state(operator):
    return {
        "request_id": str(uuid4()),
        "user_id": operator.user_id,
        "scopes": list(operator.scopes),
        "query": "Change Project Atlas deployment status to approved.",
        "tool_name": "update_deployment_status",
        "tool_arguments": {"project_id": "atlas", "new_status": "approved"},
        "trace_id": "initial-request-trace",
        "timings_ms": {},
    }


def counts(tools):
    connection = tools.storage.connect()
    try:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("actions", "tool_executions", "audit", "tickets")
        }
    finally:
        connection.close()


def atlas(tools):
    return tools.execute_read(
        "get_deployment_status", {"project_id": "atlas"}, ["public", "tools:read"]
    )


def test_create_persists_canonical_pending_action_without_effect(approvals, tools, state):
    state["tool_arguments"]["project_id"] = " ATLAS "
    pending = approvals.create(state)
    assert pending["status"] == "pending"
    assert pending["arguments"] == {"project_id": "atlas", "new_status": "approved"}
    assert pending["reason"] == state["query"]
    assert pending["requested_by"] == "operator"
    assert pending["timestamp"]
    assert atlas(tools) == {"project_id": "atlas", "status": "pending", "version": 0}
    assert counts(tools) == {"actions": 1, "tool_executions": 0, "audit": 0, "tickets": 0}
    assert approvals.create(state) == pending


def test_approved_action_executes_once_and_audit_commits(approvals, tools, state, approver):
    approvals.create(state)
    result = approvals.decide(state["request_id"], "approve", approver)
    assert result["status"] == "completed"
    assert result["approval_status"] == "approved"
    assert result["approval"]["status"] == "executed"
    assert result["approval"]["decided_by"] == approver.user_id
    assert result["approval"]["decided_at"]
    assert result["tool_result"] == {"project_id": "atlas", "status": "approved", "version": 1}
    assert result["trace_id"] == "initial-request-trace"
    assert approvals.decide(state["request_id"], "approve", approver) == result
    assert approvals.create(state)["result"] == result
    assert atlas(tools)["version"] == 1
    assert counts(tools) == {"actions": 1, "tool_executions": 1, "audit": 2, "tickets": 0}


def test_denied_action_has_no_effect_and_replays(approvals, tools, state, approver):
    approvals.create(state)
    result = approvals.decide(state["request_id"], "deny", approver)
    assert result["status"] == "denied"
    assert result["approval_status"] == "denied"
    assert result["tool_result"] is None
    assert approvals.decide(state["request_id"], "deny", approver) == result
    assert approvals.create(state)["result"] == result
    assert atlas(tools)["version"] == 0
    assert counts(tools) == {"actions": 1, "tool_executions": 0, "audit": 1, "tickets": 0}


@pytest.mark.parametrize("initial,opposite", [("approve", "deny"), ("deny", "approve")])
def test_conflicting_decisions_rejected(approvals, tools, state, approver, initial, opposite):
    approvals.create(state)
    result = approvals.decide(state["request_id"], initial, approver)
    before = counts(tools)
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(state["request_id"], opposite, approver)
    assert error.value.status_code == 409
    assert counts(tools) == before
    assert approvals.get(state["request_id"], approver)["result"] == result


@pytest.mark.parametrize("changed", ["user_id", "query", "tool_name", "tool_arguments"])
def test_request_id_collision_cannot_change_action(approvals, state, changed):
    approvals.create(state)
    different = dict(state)
    if changed == "user_id":
        different[changed] = "another-operator"
    elif changed == "query":
        different[changed] = "A different rationale"
    elif changed == "tool_name":
        different[changed] = "create_deployment_ticket"
        different["tool_arguments"] = {"project_id": "atlas", "description": "Investigate"}
    else:
        different[changed] = {"project_id": "atlas", "new_status": "failed"}
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.create(different)
    assert error.value.code == "request_id_conflict"


@pytest.mark.parametrize("decision", ["approve", "deny"])
@pytest.mark.parametrize(
    "scopes",
    [
        ["public", "tools:write"],
        ["public", "approvals:decide"],
        ["internal", "tools:write", "approvals:decide"],
    ],
)
def test_decision_requires_approval_write_and_target_permissions(
    approvals, tools, state, scopes, decision
):
    approvals.create(state)
    denied_identity = TokenIdentity(user_id="outsider", scopes=scopes)
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(state["request_id"], decision, denied_identity)
    assert error.value.status_code == 403
    assert atlas(tools)["version"] == 0
    assert counts(tools)["audit"] == 0


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_self_decision_is_forbidden_even_with_all_permissions(approvals, state, decision):
    approvals.create(state)
    self_approver = TokenIdentity(
        user_id="operator", scopes=["public", "tools:write", "approvals:decide"]
    )
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(state["request_id"], decision, self_approver)
    assert error.value.code == "self_approval_forbidden"


def test_pending_visibility_is_owner_or_authorised_independent_approver(
    approvals, state, operator, approver
):
    pending = approvals.create(state)
    assert approvals.get(state["request_id"], operator) == pending
    assert approvals.get(state["request_id"], approver) == pending
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.get(
            state["request_id"], TokenIdentity(user_id="reader", scopes=["public", "tools:read"])
        )
    assert error.value.status_code == 403


def test_restart_preserves_pending_and_execution_receipt(settings, state, approver):
    tools = ToolRegistry(settings.database_path)
    original = ApprovalService(tools)
    original.create(state)
    reopened = ApprovalService(ToolRegistry(settings.database_path))
    result = reopened.decide(state["request_id"], "approve", approver)
    restarted = ApprovalService(ToolRegistry(settings.database_path))
    assert restarted.decide(state["request_id"], "approve", approver) == result
    assert restarted.create(state)["result"] == result
    assert atlas(tools)["version"] == 1


@pytest.mark.parametrize("scopes", [["public", "tools:write"], ["internal", "tools:read"]])
def test_weaker_token_for_same_owner_cannot_read_internal_action(approvals, state, scopes):
    state["tool_arguments"] = {"project_id": "orion", "new_status": "approved"}
    state["scopes"] = ["internal", "tools:write"]
    approvals.create(state)
    weaker_identity = TokenIdentity(user_id=state["user_id"], scopes=scopes)
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.get(state["request_id"], weaker_identity)
    assert error.value.status_code == 403


def test_concurrent_approvals_across_connections_execute_once(
    approvals, tools, state, approver
):
    approvals.create(state)
    independent_service = ApprovalService(tools)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(service.decide, state["request_id"], "approve", approver)
            for service in (approvals, independent_service)
        ]
        results = [future.result() for future in futures]
    assert results[0] == results[1]
    assert atlas(tools)["version"] == 1
    assert counts(tools)["tool_executions"] == 1
    assert counts(tools)["audit"] == 2


def test_concurrent_opposite_decisions_have_one_terminal_outcome(
    approvals, tools, state, approver
):
    approvals.create(state)
    independent_service = ApprovalService(tools)

    def decide(service, decision):
        try:
            return service.decide(state["request_id"], decision, approver)
        except RuntimeErrorBase as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(decide, approvals, "approve"),
            pool.submit(decide, independent_service, "deny"),
        ]
        results = [future.result() for future in futures]
    failures = [result for result in results if isinstance(result, RuntimeErrorBase)]
    successes = [result for result in results if isinstance(result, dict)]
    assert len(failures) == len(successes) == 1
    assert failures[0].code == "approval_decision_conflict"
    approved = successes[0]["approval_status"] == "approved"
    assert atlas(tools)["version"] == int(approved)
    assert counts(tools)["tool_executions"] == int(approved)
    assert counts(tools)["audit"] == 1 + int(approved)


def test_failed_effect_rolls_back_decision_effect_receipt_and_audit(
    approvals, tools, state, approver, monkeypatch
):
    approvals.create(state)
    original = tools.execute_approved

    def fail_after_effect(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated tool failure after writing")

    monkeypatch.setattr(tools, "execute_approved", fail_after_effect)
    with pytest.raises(RuntimeError, match="simulated tool failure"):
        approvals.decide(state["request_id"], "approve", approver)
    assert atlas(tools)["version"] == 0
    assert approvals.get(state["request_id"], approver)["status"] == "pending"
    assert counts(tools)["audit"] == counts(tools)["tool_executions"] == 0
    monkeypatch.setattr(tools, "execute_approved", original)
    assert approvals.decide(state["request_id"], "approve", approver)["status"] == "completed"


def test_other_connections_never_see_intermediate_approval_or_effect(
    approvals, tools, state, approver, monkeypatch
):
    approvals.create(state)
    original = tools.execute_approved

    def observe_uncommitted(*args, **kwargs):
        result = original(*args, **kwargs)
        connection = tools.storage.connect()
        try:
            assert connection.execute(
                "SELECT status FROM actions WHERE request_id=?", (state["request_id"],)
            ).fetchone()[0] == "pending"
            assert connection.execute(
                "SELECT version FROM deployments WHERE project_id='atlas'"
            ).fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 0
        finally:
            connection.close()
        return result

    monkeypatch.setattr(tools, "execute_approved", observe_uncommitted)
    approvals.decide(state["request_id"], "approve", approver)
    assert atlas(tools)["version"] == 1


def test_requester_project_permissions_rechecked_at_execution(
    approvals, tools, state, approver
):
    approvals.create(state)
    connection = tools.storage.connect()
    try:
        connection.execute(
            "UPDATE deployments SET security_scope='internal' WHERE project_id='atlas'"
        )
    finally:
        connection.close()
    approver = approver.model_copy(update={"scopes": approver.scopes + ["internal"]})
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(state["request_id"], "approve", approver)
    assert error.value.status_code == 403
    assert approvals.get(state["request_id"], approver)["status"] == "pending"
    assert counts(tools)["audit"] == counts(tools)["tool_executions"] == 0


@pytest.mark.parametrize("decision", ["approved", "", None, True])
def test_only_explicit_decisions_allowed(approvals, state, approver, decision):
    approvals.create(state)
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(state["request_id"], decision, approver)
    assert error.value.code == "approval_decision_invalid"


def test_read_tool_cannot_create_approval(approvals, state):
    state["tool_name"] = "get_deployment_status"
    state["tool_arguments"] = {"project_id": "atlas"}
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.create(state)
    assert error.value.code == "approval_not_required"


def test_unknown_action_is_structured_not_found(approvals, approver):
    with pytest.raises(RuntimeErrorBase) as error:
        approvals.decide(str(uuid4()), "approve", approver)
    assert error.value.code == "approval_not_found"
