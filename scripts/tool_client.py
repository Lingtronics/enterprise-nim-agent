"""Authenticated client for the limited MCP-inspired tools gateway."""

import argparse
import json
import os

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("--arguments", default="{}")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    token = os.environ.get("AGENT_API_TOKEN")
    if not token:
        parser.error("Set AGENT_API_TOKEN to an authorised bearer token.")
    method = "tools/call" if args.name else "tools/list"
    params = {"name": args.name, "arguments": json.loads(args.arguments)} if args.name else {}
    response = httpx.post(
        args.url.rstrip("/") + "/mcp",
        headers={"Authorization": "Bearer " + token},
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=120,
    )
    print(json.dumps(response.json(), indent=2))
    response.raise_for_status()


if __name__ == "__main__":
    main()
