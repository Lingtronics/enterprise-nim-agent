import json

from app.agent.graph import AgentRuntime, is_tool_request
from app.config.settings import TokenIdentity
from app.models.types import Generation


class Model:
    def __init__(self, evidence):
        self.evidence = evidence

    async def generate(self, messages):
        return Generation(
            content=json.dumps({"evidence": self.evidence}), model="fixture", path="demo"
        )


class Retrieval:
    async def search(self, query, scopes):
        return [
            {
                "content": "Atlas requires a GPU.",
                "metadata": {"chunk_id": "atlas-0", "document_name": "Atlas"},
            }
        ]


async def test_read_does_not_enter_approval():
    runtime = AgentRuntime(
        Model([{"chunk_id": "atlas-0", "quote": "Atlas requires a GPU."}]), Retrieval()
    )
    result = await runtime.run(
        "Summarise Atlas requirements", TokenIdentity(user_id="reader", scopes=["public"])
    )
    assert result["status"] == "completed"
    assert not result["approval_required"]
    assert result["sources"][0]["chunk_id"] == "atlas-0"


async def test_invented_quote_fails_closed():
    runtime = AgentRuntime(
        Model([{"chunk_id": "atlas-0", "quote": "Atlas costs $5."}]), Retrieval()
    )
    result = await runtime.run(
        "Atlas requirements", TokenIdentity(user_id="reader", scopes=["public"])
    )
    assert result["status"] == "error"
    assert not result["sources"]
    assert "$5" not in result["answer"]


def test_routing_uses_original_request_only():
    assert not is_tool_request("Summarise the deployment requirements for Project Atlas.")
    assert is_tool_request("What is the current deployment status of Project Atlas?")
    assert is_tool_request("Change Project Atlas deployment status to approved.")
    assert not is_tool_request("Explain the deployment status approval requirements.")
    assert not is_tool_request("Compare deployment status policies in two documents.")


async def test_partial_quote_cannot_drop_source_negation():
    class NegatedRetrieval:
        async def search(self, query, scopes):
            return [
                {
                    "content": "Atlas must not run without human approval.",
                    "metadata": {"chunk_id": "atlas-0", "document_name": "Atlas"},
                }
            ]

    runtime = AgentRuntime(
        Model([{"chunk_id": "atlas-0", "quote": "run without human approval."}]), NegatedRetrieval()
    )
    result = await runtime.run("Atlas approval", TokenIdentity(user_id="reader", scopes=["public"]))
    assert "must not run without human approval" in result["answer"]


async def test_word_number_conflicts_are_visible():
    class ConflictingRetrieval:
        async def search(self, query, scopes):
            return [
                {
                    "content": "Atlas requires two replicas. Atlas requires three replicas.",
                    "metadata": {"chunk_id": "atlas-0", "document_name": "Atlas"},
                }
            ]

    runtime = AgentRuntime(
        Model(
            [
                {"chunk_id": "atlas-0", "quote": "Atlas requires two replicas."},
                {"chunk_id": "atlas-0", "quote": "Atlas requires three replicas."},
            ]
        ),
        ConflictingRetrieval(),
    )
    result = await runtime.run("Atlas replicas", TokenIdentity(user_id="reader", scopes=["public"]))
    assert "sources conflict" in result["answer"]
