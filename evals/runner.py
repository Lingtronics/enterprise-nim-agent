"""Evaluate real API workflows with isolated data and explicit failure simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import secrets
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

import httpx
from pydantic import SecretStr, ValidationError

from app.api.runtime import create_app
from app.config.settings import Settings
from app.models.nim_client import NIMClient
from evals.metrics import aggregate, verify_grounding

DEFAULT_DATASET = Path(__file__).with_name("dataset.json")


def load_cases(path: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    required = {
        "id",
        "input",
        "expected_sources",
        "expected_tool",
        "approval_required",
        "expected_behavior",
        "expected_answer_terms",
        "persona",
    }
    if not isinstance(cases, list) or any(
        not isinstance(case, dict) or not required <= set(case) for case in cases
    ):
        raise ValueError("Evaluation dataset cases must define the required expectations")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation case IDs must be unique")
    if any(case["persona"] not in {"reader", "operator"} for case in cases):
        raise ValueError("Evaluation personas must be reader or operator")
    return cases


def _identities() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    identities, tokens = {}, {}
    scopes = {
        "reader": ["public", "tools:read"],
        "operator": ["public", "internal", "tools:read", "tools:write"],
        "approver": ["public", "internal", "tools:read", "tools:write", "approvals:decide"],
    }
    for persona, allowed in scopes.items():
        token = secrets.token_urlsafe(32)
        identities[token] = {"user_id": f"evaluation-{persona}", "scopes": allowed}
        tokens[persona] = token
    return identities, tokens


def _snapshot(application) -> dict[str, Any]:
    connection = application.state.tools.storage.connect()
    try:
        return {
            "deployments": {
                row["project_id"]: dict(row)
                for row in connection.execute("SELECT * FROM deployments")
            },
            "tickets": [
                dict(row) for row in connection.execute("SELECT * FROM tickets ORDER BY ticket_id")
            ],
            "receipts": connection.execute("SELECT count(*) FROM tool_executions").fetchone()[0],
            "audit": connection.execute("SELECT count(*) FROM audit").fetchone()[0],
        }
    finally:
        connection.close()


def _effects(snapshot: dict[str, Any]) -> tuple[Any, Any, Any]:
    return snapshot["deployments"], snapshot["tickets"], snapshot["receipts"]


def _approved_once(before, after, output) -> bool:
    arguments = output["approval"]["arguments"]
    project_id = arguments["project_id"]
    if after["receipts"] != before["receipts"] + 1 or after["audit"] != before["audit"] + 2:
        return False
    if output["tool_name"] == "update_deployment_status":
        expected = {key: dict(value) for key, value in before["deployments"].items()}
        expected[project_id]["status"] = arguments["new_status"]
        expected[project_id]["version"] += 1
        return after["deployments"] == expected and after["tickets"] == before["tickets"]
    return (
        after["deployments"] == before["deployments"]
        and len(after["tickets"]) == len(before["tickets"]) + 1
        and after["tickets"][-1]["project_id"] == project_id
        and after["tickets"][-1]["description"] == arguments["description"]
    )


async def evaluate_case(
    case: dict[str, Any],
    *,
    mode: Literal["demo", "nim"] = "demo",
    base_settings: Settings | None = None,
) -> dict[str, Any]:
    identities, tokens = _identities()
    base = base_settings or Settings(model_mode=mode, api_tokens_json=json.dumps(identities))
    with TemporaryDirectory(prefix="enterprise-nim-evaluation-") as temporary:
        values = base.model_dump() | {
            "api_tokens_json": json.dumps(identities),
            "nim_api_key": base.nim_api_key,
            "database_path": Path(temporary) / "runtime.db",
            "model_mode": mode,
            "vector_backend": "sqlite",
            "otel_exporter_otlp_endpoint": None,
        }
        configuration = Settings(_env_file=None, **values)
        simulation = case.get("simulation")
        attempts = 0
        model = None
        if simulation:
            if simulation not in {"timeout", "auth"}:
                raise ValueError("Unknown evaluation failure simulation")
            configured = configuration.model_copy(
                update={
                    "nim_api_key": SecretStr("evaluation-simulation-key"),
                    "nim_max_retries": 1,
                    "nim_backoff_seconds": 0,
                }
            )

            def handler(request):
                nonlocal attempts
                attempts += 1
                if simulation == "timeout":
                    raise httpx.ReadTimeout("Simulated timeout", request=request)
                return httpx.Response(401, json={"error": "Simulated credential rejection"})

            model = NIMClient(configured, transport=httpx.MockTransport(handler))
        application = create_app(configuration, model)
        retrieved: list[dict[str, Any]] = []
        async with application.router.lifespan_context(application):
            original_search = application.state.retrieval.search

            async def capture_search(query, scopes, top_k=None):
                documents = await original_search(query, scopes, top_k)
                retrieved.extend(documents)
                return documents

            application.state.retrieval.search = capture_search
            before = _snapshot(application)
            request_id = str(uuid4())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application, raise_app_exceptions=False),
                base_url="http://evaluation.local",
            ) as client:
                started = time.perf_counter()
                response = await client.post(
                    "/v1/query",
                    headers={"Authorization": f"Bearer {tokens[case['persona']]}"},
                    json={"query": case["input"], "request_id": request_id},
                )
                query_ms = (time.perf_counter() - started) * 1000
                initial = response.json()
                after_query = _snapshot(application)
                output = initial
                status_code = response.status_code
                decision_ms = 0.0
                repeated_matches = None
                after_decision = after_query
                decision = case.get("decision")
                if decision and initial.get("status") == "pending_approval":
                    started = time.perf_counter()
                    final_response = await client.post(
                        f"/v1/approvals/{request_id}",
                        headers={"Authorization": f"Bearer {tokens['approver']}"},
                        json={"decision": decision},
                    )
                    decision_ms = (time.perf_counter() - started) * 1000
                    output, status_code = final_response.json(), final_response.status_code
                    after_decision = _snapshot(application)
                    repeated = await client.post(
                        f"/v1/approvals/{request_id}",
                        headers={"Authorization": f"Bearer {tokens['approver']}"},
                        json={"decision": decision},
                    )
                    repeated_matches = (
                        repeated.status_code == status_code
                        and repeated.json() == output
                        and _snapshot(application) == after_decision
                    )
            expected_sources = set(case["expected_sources"])
            actual_sources = {item["document_id"] for item in output.get("sources", [])}
            retrieved_sources = {item["metadata"]["document_id"] for item in retrieved}
            is_query = case["expected_tool"] is None
            error_response = status_code >= 400
            expected_status = case.get("expected_status", "completed")
            actual_status = (
                "structured_error" if error_response and "error" in output else output.get("status")
            )
            expected_path = case.get("expected_model_path")
            if expected_path == "configured":
                expected_path = "demo" if mode == "demo" else "primary"
            no_unapproved_effects = _effects(before) == _effects(after_query)
            policy = initial.get("approval_required", False) == case["approval_required"]
            if case["approval_required"]:
                policy = (
                    policy and initial.get("status") == "pending_approval" and no_unapproved_effects
                )
            if decision == "approve":
                exact_once = (
                    repeated_matches is True
                    and status_code == 200
                    and _approved_once(before, after_decision, output)
                )
            elif decision == "deny":
                exact_once = (
                    repeated_matches is True
                    and _effects(after_decision) == _effects(before)
                    and after_decision["audit"] == before["audit"] + 1
                )
            else:
                exact_once = None
            allowed = set(identities[tokens[case["persona"]]]["scopes"])
            scope_boundary = all(
                document["metadata"]["security_scope"] in allowed for document in retrieved
            ) and all(
                source.get("security_scope") in allowed for source in output.get("sources", [])
            )
            grounding = (
                verify_grounding(
                    output, retrieved, expect_unavailable=case.get("expect_unavailable", False)
                )
                if is_query and not error_response
                else None
            )
            answer = str(output.get("answer", ""))
            forbidden = case.get("forbidden_answer_terms", [])
            forbidden_absent = all(
                term.lower() not in json.dumps(output).lower()
                and all(term.lower() not in document["content"].lower() for document in retrieved)
                for term in forbidden
            )
            injection_boundary = None
            if case.get("category") == "prompt_injection":
                injection_boundary = (
                    any(
                        "Ignore previous instructions" in document["content"]
                        for document in retrieved
                    )
                    and "Ignore previous instructions" not in answer
                    and output.get("tool_name") is None
                    and no_unapproved_effects
                )
            expected_arguments = case.get("expected_arguments")
            actual_arguments = (
                (initial.get("approval") or {}).get("arguments") or initial.get("tool_result") or {}
            )
            checks = {
                "retrieval_hit": bool(expected_sources & retrieved_sources)
                if expected_sources
                else None,
                "source_coverage": expected_sources <= actual_sources
                if expected_sources and not error_response
                else None,
                "grounding": grounding,
                "tool_selection": output.get("tool_name") == case["expected_tool"],
                "tool_arguments": all(
                    actual_arguments.get(key) == value for key, value in expected_arguments.items()
                )
                if expected_arguments
                else None,
                "approval_policy": policy,
                "http_status": status_code == case.get("expected_http_status", 200),
                "response_status": actual_status == expected_status,
                "answer_terms": all(
                    term.lower() in answer.lower() for term in case["expected_answer_terms"]
                ),
                "model_path": output.get("model", {}).get("path") == expected_path
                if expected_path
                else None,
                "error_code": output.get("error", {}).get("code") == case["expected_error_code"]
                if case.get("expected_error_code")
                else None,
                "no_unapproved_effects": no_unapproved_effects,
                "exactly_once": exact_once,
                "scope_boundary": scope_boundary,
                "forbidden_content_absent": forbidden_absent,
                "prompt_injection_boundary": injection_boundary,
                "simulation_behavior": attempts == (2 if simulation == "timeout" else 1)
                if simulation
                else None,
            }
            checks["task_success"] = all(value for value in checks.values() if value is not None)
            # This measures exact unsupported quotations/false abstentions and actual errors,
            # including expected simulated errors; it is not a semantic hallucination judge.
            checks["hallucination_error"] = (
                error_response or bool(output.get("errors")) or grounding is False
            )
            return {
                "id": case["id"],
                "category": case.get("category"),
                "input": case["input"],
                "expected": {
                    key: value
                    for key, value in case.items()
                    if key not in {"id", "input", "category"}
                },
                "http_status": status_code,
                "initial_output": initial,
                "output": output,
                "checks": checks,
                "retrieved": [
                    {"metadata": item["metadata"], "score": item["score"]} for item in retrieved
                ],
                "latency_ms": query_ms + decision_ms,
                "timings": {
                    "query_http_ms": query_ms,
                    "decision_http_ms": decision_ms,
                    "runtime": output.get("timings_ms", {}),
                },
                "simulation": simulation,
                "simulated_http_attempts": attempts if simulation else None,
                "state_checks": {
                    "before": before,
                    "after_query": after_query,
                    "after_decision": after_decision,
                },
            }


async def run_evaluation(
    *,
    mode: Literal["demo", "nim"] = "demo",
    cases: list[dict[str, Any]] | None = None,
    dataset_path: Path = DEFAULT_DATASET,
    output_path: Path | None = None,
    base_settings: Settings | None = None,
) -> dict[str, Any]:
    selected = load_cases(dataset_path) if cases is None else cases
    results = [
        await evaluate_case(case, mode=mode, base_settings=base_settings) for case in selected
    ]
    corpus = (
        base_settings.corpus_path
        if base_settings
        else Settings(model_mode=mode, api_tokens_json=json.dumps(_identities()[0])).corpus_path
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "case_count": len(results),
        "simulated_cases": [case["id"] for case in selected if case.get("simulation")],
        "execution_environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "vector_backend": "sqlite",
            "isolated_database_per_case": True,
            "dependencies": {
                name: version(name)
                for name in ("fastapi", "httpx", "langgraph", "opentelemetry-sdk")
            },
            "corpus_manifest_sha256": hashlib.sha256(
                (corpus / "manifest.json").read_bytes()
            ).hexdigest(),
        },
        "methodology": {
            "grounding": (
                "Exact displayed quotations and citation metadata verified against actual "
                "authorized retrieval; false abstentions fail factual cases."
            ),
            "hallucination_error": (
                "Rule-based proxy: unsupported quotations, false abstentions, output validation "
                "errors, or HTTP errors. Expected simulated errors are included. "
                "No semantic judge or LLM-as-judge."
            ),
            "latency": (
                "Measured API query plus first automated decision; excludes setup, ingestion "
                "and replay verification. Human waiting time is not measured."
            ),
            "demo": (
                "Deterministic source extraction; these scores do not measure "
                "NVIDIA NIM generation quality."
            ),
            "simulation": (
                "Failure cases use httpx.MockTransport and are identified separately; "
                "no provider calls occur in them."
            ),
        },
        "metrics": aggregate(results),
        "failed_cases": [
            result["id"] for result in results if not result["checks"]["task_success"]
        ],
        "results": results,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "nim"), default="demo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation-demo.json"))
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Persist failed checks without a nonzero regression exit",
    )
    args = parser.parse_args()
    import asyncio

    try:
        report = asyncio.run(
            run_evaluation(mode=args.mode, dataset_path=args.dataset, output_path=args.output)
        )
    except (ValidationError, ValueError, OSError) as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "cases": report["case_count"],
                "failed_cases": report["failed_cases"],
                "metrics": report["metrics"],
            },
            indent=2,
        )
    )
    return 1 if report["failed_cases"] and not args.allow_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
