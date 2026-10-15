"""Extract safe measurements and aggregate measured provider metadata."""

from __future__ import annotations

import math
import re
from typing import Any

from evals.metrics import UNAVAILABLE, percentile


def numeric(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def distribution(values: list[float | None]) -> dict[str, Any]:
    measured = [value for value in values if value is not None]
    return {
        "count": len(measured),
        "p50": percentile(measured, 0.5),
        "p95": percentile(measured, 0.95),
        "max": max(measured) if measured else None,
    }


def measure_response(
    payload: dict[str, Any], status_code: int, elapsed_ms: float, *, kind: str, index: int
) -> dict[str, Any]:
    """HTTP success alone does not make an unsupported known-answer request succeed."""
    answer = payload.get("answer", "")
    sources = payload.get("sources", [])
    success = (
        200 <= status_code < 300
        and payload.get("status") == "completed"
        and not payload.get("errors")
    )
    if kind == "rag":
        attributed = (
            isinstance(answer, str)
            and isinstance(sources, list)
            and bool(sources)
            and all(
                isinstance(source, dict)
                and isinstance(source.get("chunk_id"), str)
                and f"[{source['chunk_id']}]" in answer
                for source in sources
            )
        )
        success = success and attributed and answer != UNAVAILABLE
    else:
        result = payload.get("tool_result") or {}
        success = (
            success
            and payload.get("tool_name") == "get_deployment_status"
            and result.get("project_id") == "atlas"
            and result.get("status") in {"pending", "approved", "deployed", "failed"}
            and isinstance(result.get("version"), int)
        )
    model = payload.get("model") or {}
    timing = payload.get("timings_ms") or {}
    usage = {
        key: value
        for key, value in (model.get("usage") or {}).items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    }
    model_ms = numeric(model.get("duration_ms"))
    tokens_per_second = (
        usage["completion_tokens"] / (model_ms / 1000)
        if usage.get("completion_tokens") is not None and model_ms and model_ms > 0
        else None
    )
    cost = model.get("cost_estimate") or {}
    currency = cost.get("currency")
    currency = (
        currency if isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency) else None
    )
    code = (payload.get("error") or {}).get("code")
    if not success and not code:
        code = "grounding_or_task_failed" if status_code < 400 else "http_error"
    if code and not (isinstance(code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", code)):
        code = "http_error"
    return {
        "index": index,
        "kind": kind,
        "http_status": status_code,
        "success": bool(success),
        "error_code": code,
        "http_latency_ms": elapsed_ms,
        "timings_ms": {
            key: numeric(timing.get(key)) for key in ("total", "model", "retrieval", "tool")
        },
        "model_duration_ms": model_ms,
        "model": model.get("model"),
        "model_path": model.get("path"),
        "usage": usage,
        "ttft_ms": numeric(model.get("ttft_ms")),
        "tokens_per_second": tokens_per_second,
        "cost_estimate": {"amount": numeric(cost.get("amount")), "currency": currency},
    }


def summarize(
    records: list[dict[str, Any]], elapsed_seconds: float, concurrency: int
) -> dict[str, Any]:
    successes = sum(record["success"] for record in records)
    model_requests = [record for record in records if record["kind"] == "rag"]
    priced = [record for record in model_requests if record["cost_estimate"]["amount"] is not None]
    currencies = {record["cost_estimate"]["currency"] for record in priced}
    complete_cost = (
        bool(model_requests)
        and len(priced) == len(model_requests)
        and len(currencies) == 1
        and None not in currencies
    )
    total_cost = (
        sum(record["cost_estimate"]["amount"] for record in priced) if complete_cost else None
    )
    token_records = [record for record in records if record["tokens_per_second"] is not None]
    generated_tokens = sum(record["usage"]["completion_tokens"] for record in token_records)
    model_seconds = sum(record["model_duration_ms"] for record in token_records) / 1000
    return {
        "requests": len(records),
        "concurrency": concurrency,
        "elapsed_seconds": elapsed_seconds,
        "successes": successes,
        "errors": len(records) - successes,
        "success_rate": successes / len(records) if records else None,
        "error_rate": (len(records) - successes) / len(records) if records else None,
        "requests_per_second": len(records) / elapsed_seconds if elapsed_seconds > 0 else None,
        "successful_requests_per_second": successes / elapsed_seconds
        if elapsed_seconds > 0
        else None,
        "mix": {"rag": len(model_requests), "read_tool": len(records) - len(model_requests)},
        "http_latency_ms": distribution([record["http_latency_ms"] for record in records]),
        "component_latency_ms": {
            key: distribution([record["timings_ms"][key] for record in records])
            for key in ("total", "model", "retrieval", "tool")
        },
        "ttft_ms": distribution([record["ttft_ms"] for record in records]),
        "effective_completion_tokens_per_second": generated_tokens / model_seconds
        if token_records and model_seconds > 0
        else None,
        "token_rate_samples": len(token_records),
        "token_rate_definition": (
            "Completion tokens divided by model call duration, including RPC, retry and "
            "backoff time; not decoding-only throughput."
        ),
        "cost_estimate": {
            "currency": next(iter(currencies)) if complete_cost else None,
            "total": total_cost,
            "per_request": total_cost / len(records)
            if total_cost is not None and records
            else None,
            "per_successful_task": total_cost / successes
            if total_cost is not None and successes
            else None,
            "priced_model_requests": len(priced),
            "model_requests": len(model_requests),
            "basis": (
                "Provider token cost only; read-tool requests make no model call. "
                "CPU, GPU, storage and network resource costs are not estimated."
            ),
        },
        "model_names": sorted(
            {record["model"] for record in records if isinstance(record["model"], str)}
        ),
        "model_paths": sorted(
            {record["model_path"] for record in records if isinstance(record["model_path"], str)}
        ),
        "records": records,
    }
