import json

import pytest

from benchmarks.latency import measure_response, summarize
from benchmarks.load_test import quality_reference, run_benchmarks, validate_profiles


def grounded_payload():
    return {
        "status": "completed",
        "answer": "Project Atlas requires GPU inference. [atlas:0000]",
        "sources": [{"chunk_id": "atlas:0000"}],
        "errors": [],
        "model": {"model": "demo", "path": "demo", "usage": {}, "ttft_ms": None, "duration_ms": 2},
        "timings_ms": {"total": 4, "retrieval": 1, "model": 2},
    }


def test_failed_grounding_and_false_abstention_count_as_errors():
    valid = measure_response(grounded_payload(), 200, 5, kind="rag", index=0)
    invalid = grounded_payload() | {"status": "error", "errors": [{"code": "ungrounded_output"}]}
    unavailable = grounded_payload() | {
        "answer": "Information is unavailable in the authorised corpus.",
        "sources": [],
    }
    failed = measure_response(invalid, 502, 6, kind="rag", index=1)
    abstained = measure_response(unavailable, 200, 7, kind="rag", index=2)
    profile = summarize([valid, failed, abstained], 0.1, 2)
    assert profile["successes"] == 1
    assert profile["errors"] == 2
    assert profile["error_rate"] == pytest.approx(2 / 3)
    assert profile["requests_per_second"] == 30


def test_nonstreaming_and_demo_usage_do_not_invent_rates_or_costs():
    record = measure_response(grounded_payload(), 200, 5, kind="rag", index=0)
    profile = summarize([record], 0.01, 1)
    assert record["ttft_ms"] is None
    assert record["tokens_per_second"] is None
    assert profile["ttft_ms"]["count"] == 0
    assert profile["effective_completion_tokens_per_second"] is None
    assert profile["cost_estimate"]["per_request"] is None


def test_provider_rates_and_costs_require_actual_metadata():
    payload = grounded_payload()
    payload["model"].update(
        {
            "usage": {"completion_tokens": 10, "prompt_tokens": 20},
            "duration_ms": 100,
            "cost_estimate": {"amount": 0.001, "currency": "USD"},
        }
    )
    record = measure_response(payload, 200, 110, kind="rag", index=0)
    assert record["tokens_per_second"] == 100
    profile = summarize([record], 0.2, 1)
    assert profile["effective_completion_tokens_per_second"] == 100
    assert profile["cost_estimate"]["per_successful_task"] == 0.001


@pytest.mark.parametrize(
    "profile",
    [
        {"requests": 201, "concurrency": 1},
        {"requests": 10, "concurrency": 51},
        {"requests": True, "concurrency": 1},
    ],
)
def test_profiles_are_bounded(profile):
    with pytest.raises(ValueError):
        validate_profiles({"stress": profile})


def test_quality_reference_rejects_different_mode_or_corpus(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_text(
        json.dumps({"mode": "nim", "execution_environment": {"corpus_manifest_sha256": "other"}}),
        encoding="utf-8",
    )
    assert quality_reference(path, "demo", "current", "http")["included"] is False


@pytest.mark.asyncio
async def test_small_real_asgi_benchmark_measures_read_workload(settings, tmp_path):
    output = tmp_path / "benchmark.json"
    markdown = tmp_path / "report.md"
    report = await run_benchmarks(
        settings=settings,
        profiles={"smoke": {"requests": 4, "concurrency": 2}},
        retrieval_calls=3,
        evaluation_path=None,
        output_path=output,
        report_path=markdown,
    )
    profile = report["profiles"]["smoke"]
    assert profile["requests"] == profile["successes"] == 4
    assert profile["errors"] == 0
    assert profile["mix"] == {"rag": 3, "read_tool": 1}
    assert profile["component_latency_ms"]["model"]["count"] == 3
    assert profile["component_latency_ms"]["tool"]["count"] == 1
    assert report["retrieval_microbenchmark"]["calls"] == 3
    serialized = output.read_text(encoding="utf-8")
    assert "Authorization" not in serialized
    assert "test-reader-token" not in serialized
    assert "not measured" in markdown.read_text(encoding="utf-8")
