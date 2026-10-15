"""Exercise the running API over TCP without printing credentials."""

import argparse
import json
from pathlib import Path

import httpx

from app.config.settings import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/http-smoke-demo.json"))
    args = parser.parse_args()
    settings = Settings()
    token = next(key for key, user in settings.api_tokens.items() if user.user_id == "reader")
    with httpx.Client(base_url=args.url, timeout=120) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        anonymous = client.post("/v1/query", json={"query": "Atlas requirements"})
        headers = {"Authorization": "Bearer " + token}
        status = client.post(
            "/v1/query",
            headers=headers,
            json={"query": "What is the current deployment status of Project Atlas?"},
        )
        answer = client.post(
            "/v1/query",
            headers=headers,
            json={"query": "Summarise the deployment requirements for Project Atlas."},
        )
        forbidden = client.post(
            "/v1/tools/call",
            headers=headers,
            json={
                "name": "update_deployment_status",
                "arguments": {"project_id": "atlas", "new_status": "approved"},
            },
        )
        assert (
            live.status_code == ready.status_code == status.status_code == answer.status_code == 200
        )
        assert anonymous.status_code == 401 and forbidden.status_code == 403
        assert answer.json()["sources"] and status.json()["tool_name"] == "get_deployment_status"
        result = {
            "transport": "real HTTP over loopback TCP",
            "mode": settings.model_mode,
            "live": live.status_code,
            "ready": ready.status_code,
            "anonymous_rejected": anonymous.status_code,
            "reader_write_rejected": forbidden.status_code,
            "rag_source_count": len(answer.json()["sources"]),
            "model_path": answer.json()["model"]["path"],
            "trace_id": answer.json()["trace_id"],
            "passed": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
