"""Benchmark bounded read-only API profiles and a separate local retrieval baseline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from app.api.runtime import create_app
from app.config.settings import Settings
from app.retrieval.service import RetrievalService
from benchmarks.latency import distribution, measure_response, summarize

WORKLOAD = [
    ("rag", "Summarise the deployment requirements for Project Atlas."),
    ("rag", "Which deployment environments support GPU inference?"),
    ("rag", "How are API credentials injected through Kubernetes Secrets?"),
    ("read_tool", "What is the current deployment status of Project Atlas?"),
]
DEFAULT_PROFILES = {
    "interactive": {"requests": 10, "concurrency": 1},
    "sequential_baseline": {"requests": 20, "concurrency": 1},
    "moderate": {"requests": 100, "concurrency": 10},
}


def validate_profiles(profiles):
    for profile in profiles.values():
        requests, concurrency = profile["requests"], profile["concurrency"]
        if isinstance(requests, bool) or not isinstance(requests, int) or not 1 <= requests <= 200:
            raise ValueError("Each profile must contain 1 to 200 requests")
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or not 1 <= concurrency <= 50
        ):
            raise ValueError("Each profile must use concurrency between 1 and 50")


def _synthetic_token_mapping():
    token = secrets.token_urlsafe(32)
    return token, json.dumps(
        {token: {"user_id": "benchmark-reader", "scopes": ["public", "tools:read"]}}
    )


def _configured_reader(settings):
    direct = os.environ.get("AGENT_API_TOKEN")
    if direct:
        return direct
    readers = [
        (token, identity)
        for token, identity in settings.api_tokens.items()
        if {"public", "tools:read"} <= set(identity.scopes)
    ]
    if not readers:
        raise ValueError("Set AGENT_API_TOKEN or configure a public reader bearer identity")
    return min(readers, key=lambda item: "tools:write" in item[1].scopes)[0]


@asynccontextmanager
async def benchmark_client(settings, mode, url):
    if url is not None:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Benchmark URL must be an HTTP URL without credentials, query or fragment"
            )
        async with httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=settings.nim_timeout_seconds * (settings.nim_max_retries + 1) + 30,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        ) as client:
            yield client, _configured_reader(settings), None
    else:
        token, mapping = _synthetic_token_mapping()
        with TemporaryDirectory(prefix="enterprise-nim-benchmark-") as temporary:
            values = settings.model_dump() | {
                "nim_api_key": settings.nim_api_key,
                "api_tokens_json": mapping,
                "database_path": Path(temporary) / "runtime.db",
                "model_mode": mode,
                "vector_backend": "sqlite",
                "otel_exporter_otlp_endpoint": None,
            }
            configuration = Settings(_env_file=None, **values)
            application = create_app(configuration)
            async with application.router.lifespan_context(application):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application, raise_app_exceptions=False),
                    base_url="http://benchmark.local",
                    timeout=configuration.nim_timeout_seconds * (configuration.nim_max_retries + 1)
                    + 30,
                ) as client:
                    yield client, token, application


async def run_profile(client, token, requests: int, concurrency: int):
    validate_profiles({"profile": {"requests": requests, "concurrency": concurrency}})
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index):
        kind, query = WORKLOAD[index % len(WORKLOAD)]
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post(
                    "/v1/query", headers={"Authorization": f"Bearer {token}"}, json={"query": query}
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                payload = response.json()
                if not isinstance(payload, dict):
                    payload = {"error": {"code": "invalid_response"}}
                return measure_response(
                    payload, response.status_code, elapsed_ms, kind=kind, index=index
                )
            except (httpx.RequestError, ValueError, TypeError, AttributeError):
                elapsed_ms = (time.perf_counter() - started) * 1000
                return measure_response(
                    {"error": {"code": "transport_or_response_error"}},
                    0,
                    elapsed_ms,
                    kind=kind,
                    index=index,
                )

    started = time.perf_counter()
    records = await asyncio.gather(*(one(index) for index in range(requests)))
    return summarize(records, time.perf_counter() - started, concurrency)


async def benchmark_retrieval(settings, calls=100):
    if not isinstance(calls, int) or isinstance(calls, bool) or not 1 <= calls <= 200:
        raise ValueError("Retrieval baseline must use 1 to 200 calls")
    with TemporaryDirectory(prefix="enterprise-nim-retrieval-benchmark-") as temporary:
        values = settings.model_dump() | {
            "nim_api_key": settings.nim_api_key,
            "api_tokens_json": settings.api_tokens_json,
            "database_path": Path(temporary) / "runtime.db",
            "vector_backend": "sqlite",
        }
        service = RetrievalService(Settings(_env_file=None, **values))
        try:
            ingested = await asyncio.to_thread(service.ingest)
            latencies, counts = [], []
            started = time.perf_counter()
            for index in range(calls):
                submitted = time.perf_counter()
                result = await service.search(WORKLOAD[index % 3][1], ["public", "tools:read"])
                latencies.append((time.perf_counter() - submitted) * 1000)
                counts.append(len(result))
            elapsed_seconds = time.perf_counter() - started
            return {
                "calls": calls,
                "backend": "sqlite",
                "scope": "public",
                "transport": "direct local service",
                "separate_from_http_target": True,
                "ingested": ingested,
                "latency_ms": distribution(latencies),
                "elapsed_seconds": elapsed_seconds,
                "calls_per_second": calls / elapsed_seconds if elapsed_seconds > 0 else None,
                "returned_chunks": counts,
            }
        finally:
            await service.aclose()


def quality_reference(path, mode, manifest_hash, transport):
    if path is None or not Path(path).is_file():
        return {"included": False, "reason": "No separate evaluation artifact is available."}
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {
            "included": False,
            "reason": "The separate evaluation artifact could not be parsed.",
        }
    if (
        report.get("mode") != mode
        or report.get("execution_environment", {}).get("corpus_manifest_sha256") != manifest_hash
    ):
        return {
            "included": False,
            "reason": "Evaluation mode or local corpus manifest fingerprint does not match.",
        }
    return {
        "included": True,
        "artifact": str(path).replace("\\", "/"),
        "generated_at": report.get("generated_at"),
        "case_count": report.get("case_count"),
        "metrics": report.get("metrics"),
        "failed_cases": report.get("failed_cases"),
        "basis": (
            "Separate evaluation execution matching mode and local corpus manifest fingerprint; "
            "not quality inferred from load-test latency."
        ),
        "http_target_corpus_independently_verified": False if transport == "http" else None,
    }


async def run_benchmarks(
    *,
    mode: Literal["demo", "nim"] = "demo",
    url: str | None = None,
    profiles=None,
    retrieval_calls=100,
    settings: Settings | None = None,
    evaluation_path: Path | None = Path("artifacts/evaluation-demo.json"),
    output_path: Path | None = None,
    report_path: Path | None = None,
):
    selected = profiles or DEFAULT_PROFILES
    validate_profiles(selected)
    if settings is None:
        _, mapping = _synthetic_token_mapping()
        values = {"model_mode": mode if url is None else "demo"}
        if url is None or os.environ.get("AGENT_API_TOKEN"):
            values["api_tokens_json"] = mapping
        settings = Settings(**values)
    transport = "http" if url is not None else "asgi"
    manifest_hash = hashlib.sha256(
        (settings.corpus_path / "manifest.json").read_bytes()
    ).hexdigest()
    measurements = {}
    async with benchmark_client(settings, mode, url) as (client, token, application):
        for name, profile in selected.items():
            measurements[name] = await run_profile(
                client, token, profile["requests"], profile["concurrency"]
            )
    retrieval = await benchmark_retrieval(settings, retrieval_calls)
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "os_release": platform.release(),
            "logical_cpu_count": os.cpu_count(),
            "client_transport": transport,
            "target_url": url,
            "api_backend": "sqlite"
            if transport == "asgi"
            else "not independently exposed by HTTP target",
            "local_retrieval_backend": "sqlite",
            "configured_vector_backend": settings.vector_backend,
            "model_names_observed": sorted(
                {model for profile in measurements.values() for model in profile["model_names"]}
            ),
            "dependencies": {
                name: version(name)
                for name in ("fastapi", "httpx", "langgraph", "opentelemetry-sdk")
            },
            "corpus_manifest_sha256": manifest_hash,
        },
        "methodology": {
            "workload": (
                "Cyclic mix of three evaluated RAG queries and one read-only deployment-status "
                "query. No writes or approvals."
            ),
            "setup": (
                "Startup/ingestion excluded; no warmup requests. Semaphore wait excluded from "
                "each submitted request latency but included in profile wall throughput."
            ),
            "ttft": (
                "Not measured: the adapter uses non-streaming completions. "
                "Unavailable TTFT is null, not zero."
            ),
            "tokens": (
                "Only provider-reported token usage is used; demo mode has no invented "
                "token counts or rates."
            ),
            "cost": (
                "Token prices must be explicitly configured and usage present. "
                "Resource/time metrics are reported without GPU cost estimates."
            ),
            "quality": (
                "Task checks require completed responses, no output errors and attributed "
                "non-abstaining answers for these known RAG questions. This load test is not "
                "a semantic answer-quality judge."
            ),
        },
        "profiles": measurements,
        "retrieval_microbenchmark": retrieval,
        "separate_evaluation_reference": quality_reference(
            evaluation_path, mode, manifest_hash, transport
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(report, output_path), encoding="utf-8")
    return report


def _format(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "unavailable"


def render_report(report, artifact_path=None):
    environment = report["environment"]
    lines = [
        "# Benchmark report",
        "",
        f"Measured at {report['generated_at']} using **{report['mode']} mode** "
        f"and **{environment['client_transport']} transport**.",
        "",
    ]
    if artifact_path:
        lines.extend(
            [f"Machine-readable evidence: `{str(artifact_path).replace(chr(92), '/')}`.", ""]
        )
    lines.extend(
        [
            "## Environment and workload",
            "",
            f"Python {environment['python']}; {environment['os']} {environment['os_release']}; "
            f"{environment['logical_cpu_count']} logical CPUs.",
            f"Observed model: {', '.join(environment['model_names_observed']) or 'unavailable'}. "
            f"Target: {environment['target_url'] or 'isolated in-process FastAPI application'}.",
            f"API backend: {environment['api_backend']}. Local retrieval baseline: SQLite. "
            f"Corpus manifest SHA-256: `{environment['corpus_manifest_sha256']}`.",
            "",
            report["methodology"]["workload"],
            report["methodology"]["setup"],
            "",
            "| Profile | Requests / concurrency | HTTP p50 ms | HTTP p95 ms | Max ms "
            "| Requests/s | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, profile in report["profiles"].items():
        latency = profile["http_latency_ms"]
        lines.append(
            f"| {name} | {profile['requests']} / {profile['concurrency']} "
            f"| {_format(latency['p50'])} | {_format(latency['p95'])} "
            f"| {_format(latency['max'])} | {_format(profile['requests_per_second'])} "
            f"| {profile['errors']} / {profile['requests']} |"
        )
    baseline = report["profiles"].get(
        "sequential_baseline", next(iter(report["profiles"].values()))
    )
    lines.extend(
        [
            "",
            "## Baseline and optimization comparison",
            "",
            "| Metric | Baseline | Optimized |",
            "|---|---:|---|",
            f"| p50 latency ms | {_format(baseline['http_latency_ms']['p50'])} | not measured |",
            f"| p95 latency ms | {_format(baseline['http_latency_ms']['p95'])} | not measured |",
            "| TTFT | unavailable (non-streaming) | not measured |",
            f"| requests/s | {_format(baseline['requests_per_second'])} | not measured |",
        ]
    )
    quality = report["separate_evaluation_reference"]
    if quality["included"]:
        metrics = quality["metrics"]
        lines.extend(
            [
                "| retrieval hit rate (separate evaluation) "
                f"| {_format(metrics['retrieval_hit_rate'])} | not measured |",
                "| task success rate (separate evaluation) "
                f"| {_format(metrics['task_success_rate'])} | not measured |",
                "",
                f"Quality reference: `{quality['artifact']}`; {quality['case_count']} separately "
                "executed cases, matching mode and local manifest fingerprint. These are "
                "deterministic regression metrics, not NIM generation-quality results. "
                "HTTP target corpus identity is not independently exposed by the API.",
            ]
        )
    else:
        lines.extend(
            [
                "| retrieval hit rate | unavailable | not measured |",
                "| task success rate (evaluation) | unavailable | not measured |",
                "",
                quality["reason"],
            ]
        )
    lines.extend(
        [
            "",
            "## Component timing and retrieval",
            "",
            "| Profile | Model p50 ms | Retrieval p50 ms | Tool p50 ms | Runtime total p50 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, profile in report["profiles"].items():
        component = profile["component_latency_ms"]
        lines.append(
            f"| {name} | {_format(component['model']['p50'])} "
            f"| {_format(component['retrieval']['p50'])} | {_format(component['tool']['p50'])} "
            f"| {_format(component['total']['p50'])} |"
        )
    retrieval = report["retrieval_microbenchmark"]
    lines.extend(
        [
            "",
            f"Independent local SQLite retrieval: {retrieval['calls']} calls, "
            f"p50 {_format(retrieval['latency_ms']['p50'])} ms, "
            f"p95 {_format(retrieval['latency_ms']['p95'])} ms. "
            "This is a separate direct-service run and does not measure a remote vector store.",
            "",
            "## Observed bottlenecks and tradeoffs",
            "",
        ]
    )
    for name, profile in report["profiles"].items():
        component = profile["component_latency_ms"]
        applicable = {
            key: component[key]["p50"]
            for key in ("model", "retrieval", "tool")
            if component[key]["p50"] is not None
        }
        dominant = max(applicable, key=applicable.get) if applicable else "unavailable"
        lines.append(
            f"For {name}, the largest measured component median is {dominant}. "
            "Component medians describe different applicable request subsets "
            "and should not be added together."
        )
    lines.extend(
        [
            "",
            "The small local corpus and deterministic extractor make CPU scheduling, SQLite "
            "access and HTTP/framework overhead visible. These measurements do not validate "
            "NIM service latency, GPU utilization, container performance, Kubernetes throughput "
            "or production scale.",
            "",
            "Concurrency throughput and per-request latency should be assessed together. "
            "SQLite operational storage and the sample corpus are a single-instance reference "
            "baseline; the workload contains read operations only.",
            "",
            "## Recommendations and limits",
            "",
            "Measure genuine NIM latency and provider token usage with explicit credentials "
            "before choosing serving optimizations. Enable streaming and measure first-token "
            "events before reporting TTFT. Increase corpus/workload realism before comparing "
            "SQLite with Qdrant. Profile retrieval connections and scheduling before considering "
            "caching, and preserve scope filtering in any cache. Repeat comparable runs after "
            "an optimization; none was measured here.",
            "",
            report["methodology"]["ttft"],
            report["methodology"]["tokens"],
            report["methodology"]["cost"],
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "nim"), default="demo")
    parser.add_argument(
        "--url",
        help=(
            "Existing API URL; bearer token comes privately from AGENT_API_TOKEN "
            "or configured reader mapping"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark-demo.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/benchmark-report.md"))
    parser.add_argument("--evaluation", type=Path, default=Path("artifacts/evaluation-demo.json"))
    parser.add_argument("--stress-concurrency", type=int)
    parser.add_argument("--stress-requests", type=int, default=200)
    args = parser.parse_args()
    profiles = {name: dict(profile) for name, profile in DEFAULT_PROFILES.items()}
    if args.stress_concurrency is not None:
        profiles["stress"] = {
            "requests": args.stress_requests,
            "concurrency": args.stress_concurrency,
        }
    try:
        report = asyncio.run(
            run_benchmarks(
                mode=args.mode,
                url=args.url,
                profiles=profiles,
                evaluation_path=args.evaluation,
                output_path=args.output,
                report_path=args.report,
            )
        )
    except (ValidationError, ValueError, OSError) as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "transport": report["environment"]["client_transport"],
                "profiles": {
                    name: {
                        key: profile[key]
                        for key in (
                            "requests",
                            "concurrency",
                            "successes",
                            "errors",
                            "requests_per_second",
                            "http_latency_ms",
                        )
                    }
                    for name, profile in report["profiles"].items()
                },
            },
            indent=2,
        )
    )
    return 1 if any(profile["errors"] for profile in report["profiles"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
