"""Independent human approvals with atomic local effects and durable replay."""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Literal

from app.config.settings import TokenIdentity
from app.errors import RuntimeErrorBase
from app.telemetry.instrumentation import span
from app.tools.registry import ToolRegistry


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ApprovalService:
    """One transaction commits the decision, effect, audit and final response.

    The exactly-once guarantee covers tools whose effects share this SQLite
    transaction. External APIs require receiver-supported idempotency instead.
    """

    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        connection = tools.storage.connect()
        try:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    request_id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    requester_scopes_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('pending','approved','executed','denied')),
                    state_json TEXT NOT NULL,
                    result_json TEXT,
                    decided_by TEXT,
                    decided_at TEXT
                )
            """)
        finally:
            connection.close()

    @staticmethod
    def _pending_view(row: sqlite3.Row, *, include_result: bool = False) -> dict[str, Any]:
        view = {
            "request_id": row["request_id"],
            "tool_name": row["tool_name"],
            "arguments": json.loads(row["arguments_json"]),
            "reason": row["reason"],
            "requested_by": row["requested_by"],
            "timestamp": row["timestamp"],
            "status": row["status"],
        }
        if row["decided_by"] is not None:
            view["decided_by"] = row["decided_by"]
            view["decided_at"] = row["decided_at"]
        if include_result and row["result_json"] is not None:
            view["result"] = json.loads(row["result_json"])
        return view

    @staticmethod
    def _find(connection: sqlite3.Connection, request_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM actions WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise RuntimeErrorBase("approval_not_found", "The action is unavailable.", 404)
        return row

    def _require_decider(
        self, connection: sqlite3.Connection, row: sqlite3.Row, identity: TokenIdentity
    ) -> None:
        if identity.user_id == row["requested_by"]:
            raise RuntimeErrorBase(
                "self_approval_forbidden", "A different human must decide this action.", 403
            )
        if "approvals:decide" not in identity.scopes:
            raise RuntimeErrorBase(
                "approval_forbidden", "Your identity cannot decide this action.", 403
            )
        # Validate both the write capability and the target project's current scope.
        self.tools._validate(
            row["tool_name"], json.loads(row["arguments_json"]), identity.scopes, connection
        )

    def create(self, state: dict[str, Any]) -> dict[str, Any]:
        connection = self.tools.storage.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            name = state["tool_name"]
            if not self.tools.requires_approval(name):
                raise RuntimeErrorBase(
                    "approval_not_required", "Read tools do not require approval.", 422
                )
            arguments = self.tools._validate(
                name, state["tool_arguments"], state["scopes"], connection
            )
            canonical_arguments = _json(arguments)
            request_id = state["request_id"]
            existing = connection.execute(
                "SELECT * FROM actions WHERE request_id=?", (request_id,)
            ).fetchone()
            if existing is not None:
                if (
                    existing["requested_by"] != state["user_id"]
                    or existing["reason"] != state["query"]
                    or existing["tool_name"] != name
                    or existing["arguments_json"] != canonical_arguments
                ):
                    raise RuntimeErrorBase(
                        "request_id_conflict", "This request ID belongs to another action.", 409
                    )
                result = self._pending_view(existing, include_result=True)
                connection.commit()
                return result
            stored_state = dict(state) | {"tool_arguments": arguments}
            connection.execute(
                """
                INSERT INTO actions (
                    request_id,tool_name,arguments_json,requested_by,requester_scopes_json,
                    reason,timestamp,status,state_json
                ) VALUES(?,?,?,?,?,?,?,'pending',?)
                """,
                (
                    request_id,
                    name,
                    canonical_arguments,
                    state["user_id"],
                    _json(state["scopes"]),
                    state["query"],
                    datetime.now(timezone.utc).isoformat(),
                    _json(stored_state),
                ),
            )
            result = self._pending_view(self._find(connection, request_id))
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, request_id: str, identity: TokenIdentity) -> dict[str, Any]:
        connection = self.tools.storage.connect()
        try:
            row = self._find(connection, request_id)
            if identity.user_id != row["requested_by"]:
                self._require_decider(connection, row, identity)
            else:
                self.tools._validate(
                    row["tool_name"],
                    json.loads(row["arguments_json"]),
                    identity.scopes,
                    connection,
                )
            return self._pending_view(row, include_result=True)
        finally:
            connection.close()

    @staticmethod
    def _audit(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        event: str,
        actor: str,
        timestamp: str,
        details: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit(request_id,event,tool_name,actor,timestamp,details_json)
            VALUES(?,?,?,?,?,?)
            """,
            (row["request_id"], event, row["tool_name"], actor, timestamp, _json(details)),
        )

    def decide(
        self, request_id: str, decision: Literal["approve", "deny"], identity: TokenIdentity
    ) -> dict[str, Any]:
        if decision not in ("approve", "deny"):
            raise RuntimeErrorBase(
                "approval_decision_invalid",
                "An explicit approve or deny decision is required.",
                422,
            )
        started = time.perf_counter()
        connection = self.tools.storage.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = self._find(connection, request_id)
            self._require_decider(connection, row, identity)
            if row["status"] in ("executed", "denied"):
                expected = "executed" if decision == "approve" else "denied"
                if row["status"] != expected:
                    raise RuntimeErrorBase(
                        "approval_decision_conflict",
                        "This action already has a different decision.",
                        409,
                    )
                result = json.loads(row["result_json"])
                connection.commit()
                return result
            if row["status"] != "pending":
                raise RuntimeErrorBase(
                    "approval_invalid", "The persisted approval state is invalid.", 409
                )
            state = json.loads(row["state_json"])
            timestamp = datetime.now(timezone.utc).isoformat()
            approval_status = "approved" if decision == "approve" else "denied"
            connection.execute(
                "UPDATE actions SET status=?,decided_by=?,decided_at=? WHERE request_id=?",
                (approval_status, identity.user_id, timestamp, request_id),
            )
            self._audit(
                connection,
                row,
                "approval_decision",
                identity.user_id,
                timestamp,
                {"decision": decision, "requested_by": row["requested_by"]},
            )
            tool_started = time.perf_counter()
            if decision == "approve":
                with span(
                    "tool.call",
                    tool_name=row["tool_name"],
                    approval_required=True,
                    approval_status="approved",
                ):
                    tool_result = self.tools.execute_approved(
                        connection,
                        request_id,
                        row["tool_name"],
                        json.loads(row["arguments_json"]),
                        json.loads(row["requester_scopes_json"]),
                    )
                self._audit(
                    connection,
                    row,
                    "tool_executed",
                    row["requested_by"],
                    timestamp,
                    {"project_id": tool_result["project_id"], "approved_by": identity.user_id},
                )
                terminal_status = "executed"
                answer = "Approved action executed successfully."
            else:
                tool_result = None
                terminal_status = "denied"
                answer = "The action was rejected."
            tool_ms = (time.perf_counter() - tool_started) * 1000 if tool_result else 0.0
            connection.execute(
                "UPDATE actions SET status=? WHERE request_id=?", (terminal_status, request_id)
            )
            view = self._pending_view(self._find(connection, request_id))
            result = {
                "request_id": request_id,
                "status": "completed" if decision == "approve" else "denied",
                "answer": answer,
                "sources": [],
                "tool_name": row["tool_name"],
                "tool_result": tool_result,
                "approval_required": True,
                "approval_status": approval_status,
                "approval": view,
                "model": {},
                "errors": [],
                "trace_id": state.get("trace_id", ""),
                "timings_ms": {
                    **state.get("timings_ms", {}),
                    "tool": tool_ms,
                    "total": (time.perf_counter() - started) * 1000,
                },
            }
            connection.execute(
                "UPDATE actions SET result_json=? WHERE request_id=?", (_json(result), request_id)
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
