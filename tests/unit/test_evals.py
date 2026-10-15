import json

import pytest

from evals.metrics import aggregate, verify_grounding
from evals.runner import load_cases, run_evaluation


def test_metrics_use_only_applicable_denominators():
    results = [
        {
            "checks": {
                "retrieval_hit": True,
                "grounding": True,
                "tool_selection": True,
                "task_success": True,
                "hallucination_error": False,
            },
            "latency_ms": 10,
        },
        {
            "checks": {
                "retrieval_hit": False,
                "grounding": False,
                "tool_selection": False,
                "task_success": False,
                "hallucination_error": True,
            },
            "latency_ms": 20,
        },
        {
            "checks": {
                "retrieval_hit": None,
                "grounding": None,
                "tool_selection": True,
                "task_success": True,
                "hallucination_error": False,
            },
            "latency_ms": 30,
        },
    ]
    metrics = aggregate(results)
    assert metrics["retrieval_hit_rate"] == 0.5
    assert metrics["source_grounding_correctness"] == 0.5
    assert metrics["tool_selection_accuracy"] == pytest.approx(2 / 3)
    assert metrics["hallucination_error_rate"] == pytest.approx(1 / 3)
    assert metrics["samples"]["source_grounding_correctness"]["denominator"] == 2
    assert metrics["source_coverage_rate"] is None
    assert metrics["latency_ms"]["p50"] == 20


def example_output(quote):
    metadata = {
        "chunk_id": "atlas:0000",
        "document_id": "atlas",
        "document_name": "Project Atlas",
        "security_scope": "public",
    }
    document = {"metadata": metadata, "content": "Project Atlas requires GPU inference."}
    output = {
        "status": "completed",
        "answer": f"{quote} [atlas:0000]",
        "sources": [metadata],
        "errors": [],
    }
    return output, [document]


def test_citation_alone_cannot_pass_unsupported_answer():
    output, documents = example_output("Project Atlas costs one million dollars.")
    assert verify_grounding(output, documents) is False
    output, documents = example_output("Project Atlas requires GPU inference.")
    assert verify_grounding(output, documents) is True
    output["sources"][0] = output["sources"][0] | {"security_scope": "finance"}
    assert verify_grounding(output, documents) is False


def test_false_abstention_and_uncited_claims_fail_grounding():
    output, documents = example_output("Project Atlas requires GPU inference.")
    output["answer"] += "\n\nIt has unlimited funding."
    assert verify_grounding(output, documents) is False
    output["answer"] = "Information is unavailable in the authorised corpus."
    output["sources"] = []
    assert verify_grounding(output, documents) is False
    assert verify_grounding(output, documents, expect_unavailable=True) is True


def test_multiline_verbatim_quote_is_supported_and_extra_claim_is_not():
    output, documents = example_output("Project Atlas requires GPU inference.")
    documents[0]["content"] += "\n\nProject Atlas must run on Kubernetes."
    output["answer"] = documents[0]["content"] + " [atlas:0000]"
    assert verify_grounding(output, documents) is True
    output["answer"] += "\n\nAn invented extra claim."
    assert verify_grounding(output, documents) is False


def test_conflict_notice_requires_different_counts_for_same_project():
    atlas, atlas_documents = example_output("Project Atlas requires 2 replicas.")
    atlas_documents[0]["content"] = "Project Atlas requires 2 replicas."
    zephyr_metadata = {
        "chunk_id": "zephyr:0000",
        "document_id": "zephyr",
        "document_name": "Project Zephyr",
        "security_scope": "public",
    }
    zephyr = {"metadata": zephyr_metadata, "content": "Project Zephyr requires one replica."}
    documents = atlas_documents + [zephyr]
    atlas["answer"] += "\n\nProject Zephyr requires one replica. [zephyr:0000]"
    atlas["sources"].append(zephyr_metadata)
    assert verify_grounding(atlas, documents) is True
    notice = "The sources conflict on replica requirements; review both versions.\n\n"
    atlas["answer"] = notice + atlas["answer"]
    assert verify_grounding(atlas, documents) is False
    zephyr["metadata"] = zephyr_metadata | {"document_name": "Project Atlas Archived Requirements"}
    zephyr["content"] = "Project Atlas requires 3 replicas."
    atlas["sources"][1] = zephyr["metadata"]
    atlas["answer"] = (
        notice
        + "Project Atlas requires 2 replicas. [atlas:0000]\n\n"
        "Project Atlas requires 3 replicas. [zephyr:0000]"
    )
    assert verify_grounding(atlas, documents) is True


def test_dataset_has_required_category_coverage():
    cases = load_cases()
    assert len(cases) >= 24
    minimums = {
        "factual": 5,
        "multi_document": 5,
        "tool_selection": 3,
        "approval_policy": 3,
        "prompt_injection": 2,
        "failure": 2,
    }
    for category, minimum in minimums.items():
        assert sum(case["category"] == category for case in cases) >= minimum


@pytest.mark.asyncio
async def test_harness_runs_real_read_approval_denial_and_failure_workflows(settings, tmp_path):
    selected_ids = {"tool-001", "approval-002", "approval-003", "failure-001", "missing-001"}
    cases = [case for case in load_cases() if case["id"] in selected_ids]
    output_path = tmp_path / "evaluation.json"
    report = await run_evaluation(cases=cases, base_settings=settings, output_path=output_path)
    assert report["failed_cases"] == []
    assert report["case_count"] == 5
    assert report["execution_environment"]["isolated_database_per_case"] is True
    by_id = {result["id"]: result for result in report["results"]}
    assert (
        by_id["approval-002"]["state_checks"]["after_decision"]["deployments"]["atlas"]["version"]
        == 1
    )
    assert (
        by_id["approval-003"]["state_checks"]["after_decision"]["deployments"]["atlas"]["version"]
        == 0
    )
    assert by_id["failure-001"]["simulated_http_attempts"] == 2
    assert by_id["failure-001"]["output"]["model"]["path"] == "fallback"
    serialized = output_path.read_text(encoding="utf-8")
    assert json.loads(serialized)["metrics"]["task_success_rate"] == 1
    assert "Authorization" not in serialized
    assert "test-reader-token" not in serialized
    assert "evaluation-simulation-key" not in serialized
