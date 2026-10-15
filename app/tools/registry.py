"""Explicit tool allowlist, deterministic routing and capability-checked writes."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.agent.policies import requires_approval
from app.errors import RuntimeErrorBase
from app.tools.schemas import (
    CreateDeploymentTicketArguments,
    GetDeploymentStatusArguments,
    UpdateDeploymentStatusArguments,
)
from app.tools.storage import ToolStorage

_SCHEMAS = {
    "get_deployment_status": GetDeploymentStatusArguments,
    "update_deployment_status": UpdateDeploymentStatusArguments,
    "create_deployment_ticket": CreateDeploymentTicketArguments,
}
_DESCRIPTIONS = {
    "get_deployment_status": "Read an authorised project's deployment status and version.",
    "update_deployment_status": "Update deployment status after persisted human approval.",
    "create_deployment_ticket": "Create a deployment ticket after persisted human approval.",
}


class ToolRegistry:
    def __init__(self, database_path: Path):
        self.storage = ToolStorage(database_path)

    @staticmethod
    def requires_approval(name: str) -> bool:
        return requires_approval(name)

    def _validate(
        self, name: str, arguments: Any, scopes: list[str], connection: sqlite3.Connection
    ) -> dict[str, Any]:
        if name not in _SCHEMAS:
            raise RuntimeErrorBase("unknown_tool", "The requested tool is not registered.", 422)
        required_scope = "tools:write" if self.requires_approval(name) else "tools:read"
        if required_scope not in scopes:
            raise RuntimeErrorBase("tool_forbidden", "Your identity cannot use this tool.", 403)
        try:
            canonical = _SCHEMAS[name].model_validate(arguments).model_dump()
        except ValidationError:
            raise RuntimeErrorBase(
                "tool_arguments_invalid", "Tool arguments do not match the registered schema.", 422
            ) from None
        project = connection.execute(
            "SELECT security_scope FROM deployments WHERE project_id=?", (canonical["project_id"],)
        ).fetchone()
        if project is None:
            raise RuntimeErrorBase(
                "project_not_found", "The requested project is unavailable.", 404
            )
        if project["security_scope"] not in scopes:
            raise RuntimeErrorBase(
                "tool_forbidden", "Your identity cannot access this project.", 403
            )
        return canonical

    def validate(self, name: str, arguments: Any, scopes: list[str]) -> dict[str, Any]:
        connection = self.storage.connect()
        try:
            return self._validate(name, arguments, scopes, connection)
        finally:
            connection.close()

    def select(self, query: str) -> tuple[str, dict[str, Any]]:
        """Parse a narrow single-project grammar; ambiguity must never become a write."""
        text = query.lower().strip()
        connection = self.storage.connect()
        try:
            known_projects = {
                row["project_id"]
                for row in connection.execute("SELECT project_id FROM deployments")
            }
        finally:
            connection.close()
        project_ids = set(re.findall(r"\bproject\s+([a-z][a-z0-9_-]*)\b", text))
        project_ids.update(
            project
            for project in known_projects
            if re.search(rf"(?<![a-z0-9_-]){re.escape(project)}(?![a-z0-9_-])", text)
        )
        if len(project_ids) != 1:
            raise RuntimeErrorBase(
                "tool_request_ambiguous",
                "Specify exactly one project for an operational request.",
                422,
            )
        project_id = next(iter(project_ids))
        if project_id not in known_projects:
            raise RuntimeErrorBase(
                "project_not_found", "The requested project is unavailable.", 404
            )
        if re.match(r"^(?:please\s+)?(?:change|update|set)\b", text):
            project = rf"(?:project\s+)?{re.escape(project_id)}(?:'s)?"
            subject = (
                rf"(?:{project}\s+(?:deployment\s+)?status|"
                rf"(?:the\s+)?(?:deployment\s+)?status\s+(?:of|for)\s+{project})"
            )
            match = re.match(
                rf"^(?:please\s+)?(?:change|update|set)\s+{subject}\s+"
                r"to\s+(pending|approved|deployed|failed)[.!?]?\s*$",
                text,
            )
            if match is None:
                raise RuntimeErrorBase(
                    "tool_request_ambiguous", "Specify one deployment status and one action.", 422
                )
            return "update_deployment_status", {
                "project_id": project_id,
                "new_status": match.group(1),
            }
        if re.match(r"^(?:please\s+)?create\b", text):
            header = text.split(":", 1)[0].strip()
            if not re.match(
                rf"^(?:please\s+)?create\s+(?:(?:a|an)\s+)?deployment\s+ticket\s+"
                rf"(?:for\s+)?(?:project\s+)?{re.escape(project_id)}[.!?]?$",
                header,
            ):
                raise RuntimeErrorBase(
                    "tool_request_ambiguous", "Specify a deployment-ticket action.", 422
                )
            description = query.split(":", 1)[1].strip() if ":" in query else query.strip()
            return "create_deployment_ticket", {
                "project_id": project_id,
                "description": description,
            }
        if (
            re.match(r"^(?:what(?:\s+is|'s)|show|check|get|tell\s+me|current)\b", text)
            and re.search(r"\b(?:current(?:\s+deployment)?\s+status|deployment\s+status)\b", text)
            and not re.search(r"\b(?:change|update|set|create|delete|execute)\b", text)
        ):
            return "get_deployment_status", {"project_id": project_id}
        raise RuntimeErrorBase(
            "tool_request_ambiguous",
            "The query does not select one supported operational action.",
            422,
        )

    def execute_read(self, name: str, arguments: Any, scopes: list[str]) -> dict[str, Any]:
        if self.requires_approval(name):
            raise RuntimeErrorBase(
                "approval_required", "This tool requires explicit persisted human approval.", 409
            )
        connection = self.storage.connect()
        try:
            canonical = self._validate(name, arguments, scopes, connection)
            row = connection.execute(
                "SELECT project_id,status,version FROM deployments WHERE project_id=?",
                (canonical["project_id"],),
            ).fetchone()
            return dict(row)
        finally:
            connection.close()

    def execute_approved(
        self,
        connection: sqlite3.Connection,
        request_id: str,
        name: str,
        arguments: Any,
        scopes: list[str],
    ) -> dict[str, Any]:
        """Execute a persisted capability inside the approval manager's transaction.

        A caller-supplied approval flag is never sufficient. No commit occurs here;
        approval status, effect, receipt and audit must commit atomically upstream.
        """
        if not connection.in_transaction:
            raise RuntimeErrorBase(
                "approval_transaction_required",
                "Approved execution requires an active transaction.",
                409,
            )
        if not self.requires_approval(name):
            raise RuntimeErrorBase(
                "tool_arguments_invalid", "Read tools do not use approved execution.", 422
            )
        try:
            action = connection.execute(
                "SELECT tool_name,arguments_json,requester_scopes_json,status "
                "FROM actions WHERE request_id=?",
                (request_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            action = None
        if action is None or action["status"] != "approved":
            raise RuntimeErrorBase(
                "approval_required", "No persisted approved action permits this execution.", 409
            )
        try:
            stored_arguments = json.loads(action["arguments_json"])
            stored_scopes = json.loads(action["requester_scopes_json"])
        except (TypeError, ValueError):
            raise RuntimeErrorBase(
                "approval_invalid", "The persisted approval capability is invalid.", 409
            ) from None
        if not isinstance(stored_scopes, list) or any(
            not isinstance(scope, str) for scope in stored_scopes
        ):
            raise RuntimeErrorBase(
                "approval_invalid", "The persisted approval capability is invalid.", 409
            )
        canonical = self._validate(name, arguments, scopes, connection)
        if (
            action["tool_name"] != name
            or set(stored_scopes) != set(scopes)
            or canonical != stored_arguments
        ):
            raise RuntimeErrorBase(
                "approval_mismatch",
                "The action does not match the persisted approval capability.",
                409,
            )
        self._validate(name, stored_arguments, stored_scopes, connection)
        receipt = connection.execute(
            "SELECT result_json FROM tool_executions WHERE request_id=?", (request_id,)
        ).fetchone()
        if receipt is not None:
            return json.loads(receipt["result_json"])
        if name == "update_deployment_status":
            connection.execute(
                "UPDATE deployments SET status=?,version=version+1 WHERE project_id=?",
                (canonical["new_status"], canonical["project_id"]),
            )
            result = dict(
                connection.execute(
                    "SELECT project_id,status,version FROM deployments WHERE project_id=?",
                    (canonical["project_id"],),
                ).fetchone()
            )
        else:
            timestamp = datetime.now(timezone.utc).isoformat()
            connection.execute(
                "INSERT INTO tickets(ticket_id,project_id,description,created_at) VALUES(?,?,?,?)",
                (request_id, canonical["project_id"], canonical["description"], timestamp),
            )
            result = {
                "ticket_id": request_id,
                "project_id": canonical["project_id"],
                "description": canonical["description"],
                "created_at": timestamp,
            }
        connection.execute(
            "INSERT INTO tool_executions(request_id,result_json) VALUES(?,?)",
            (request_id, json.dumps(result, sort_keys=True)),
        )
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": _DESCRIPTIONS[name],
                "inputSchema": schema.model_json_schema(),
                "annotations": {
                    "readOnlyHint": not self.requires_approval(name),
                    "idempotentHint": not self.requires_approval(name),
                },
                "approval_required": self.requires_approval(name),
                "required_scopes": [
                    "tools:write" if self.requires_approval(name) else "tools:read"
                ],
            }
            for name, schema in _SCHEMAS.items()
        ]
