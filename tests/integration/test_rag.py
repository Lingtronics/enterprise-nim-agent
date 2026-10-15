import json

import httpx
from pydantic import SecretStr

from app.agent.graph import AgentRuntime
from app.config.settings import TokenIdentity
from app.models.nim_client import NIMClient
from app.models.router import ModelRouter
from app.retrieval.service import RetrievalService


async def test_nim_receives_scoped_context_and_returns_verified_citations(settings):
    retrieval = RetrievalService(settings)
    retrieval.ingest()
    observed = []

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"])
        observed.extend(context["documents"])
        document = next(d for d in observed if "Atlas" in d["content"])
        return httpx.Response(
            200,
            json={
                "model": "fixture-nim",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "evidence": [
                                        {
                                            "chunk_id": document["metadata"]["chunk_id"],
                                            "quote": document["content"],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            },
        )

    nim = NIMClient(
        settings.model_copy(update={"nim_api_key": SecretStr("test-key")}),
        transport=httpx.MockTransport(respond),
    )
    runtime = AgentRuntime(ModelRouter(nim), retrieval)
    result = await runtime.run(
        "Summarise the deployment requirements for Project Atlas.",
        TokenIdentity(user_id="reader", scopes=["public"]),
    )
    assert observed
    assert all(d["metadata"]["security_scope"] == "public" for d in observed)
    assert result["status"] == "completed"
    assert result["sources"]
    assert result["model"]["path"] == "primary"
    assert not result["tool_name"]
    await nim.aclose()
    await retrieval.aclose()


async def test_restricted_context_never_reaches_model(settings):
    retrieval = RetrievalService(settings)
    retrieval.ingest()
    observed = []

    class InspectingModel:
        async def generate(self, messages, tools=None):
            from app.models.router import DemoModel

            observed.extend(json.loads(messages[-1]["content"])["documents"])
            return await DemoModel().generate(messages)

    runtime = AgentRuntime(ModelRouter(InspectingModel()), retrieval)
    result = await runtime.run(
        "What is the confidential annual budget?",
        TokenIdentity(user_id="reader", scopes=["public"]),
    )
    assert all(d["metadata"]["security_scope"] == "public" for d in observed)
    assert "4.2 million GBP" not in json.dumps(observed)
    assert "4.2 million GBP" not in json.dumps(result)
    await retrieval.aclose()
