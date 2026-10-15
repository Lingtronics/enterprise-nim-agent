import json

from app.agent.graph import AgentRuntime
from app.config.settings import TokenIdentity
from app.models.types import Generation


async def test_different_projects_have_different_requirements_without_a_conflict():
    class Retrieval:
        async def search(self, query, scopes):
            return [
                {
                    "content": "Atlas requires two replicas.",
                    "metadata": {
                        "chunk_id": "atlas:0",
                        "document_name": "Project Atlas Requirements",
                    },
                },
                {
                    "content": "Zephyr requires one replica.",
                    "metadata": {
                        "chunk_id": "zephyr:0",
                        "document_name": "Project Zephyr Requirements",
                    },
                },
            ]

    class Model:
        async def generate(self, messages):
            return Generation(
                model="fixture",
                content=json.dumps(
                    {
                        "evidence": [
                            {"chunk_id": "atlas:0", "quote": "Atlas requires two replicas."},
                            {"chunk_id": "zephyr:0", "quote": "Zephyr requires one replica."},
                        ]
                    }
                ),
            )

    result = await AgentRuntime(Model(), Retrieval()).run(
        "Compare Atlas and Zephyr requirements", TokenIdentity(user_id="reader", scopes=["public"])
    )
    assert "sources conflict" not in result["answer"]
    assert len(result["sources"]) == 2
