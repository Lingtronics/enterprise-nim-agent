"""Create random local identities; never overwrite existing credentials."""

import json
import secrets
from pathlib import Path


def main():
    destination = Path(".env")
    if destination.exists():
        raise SystemExit(".env already exists; edit it explicitly or use it as-is.")
    users = {
        secrets.token_urlsafe(32): {"user_id": "reader", "scopes": ["public", "tools:read"]},
        secrets.token_urlsafe(32): {
            "user_id": "operator",
            "scopes": ["public", "internal", "tools:read", "tools:write"],
        },
        secrets.token_urlsafe(32): {
            "user_id": "approver",
            "scopes": ["public", "internal", "tools:read", "tools:write", "approvals:decide"],
        },
    }
    content = Path(".env.example").read_text(encoding="utf-8")
    content = content.replace("API_TOKENS_JSON=", "API_TOKENS_JSON='" + json.dumps(users) + "'")
    destination.write_text(content, encoding="utf-8")
    print("Created .env with random tokens for reader, operator and approver. Keep it private.")


if __name__ == "__main__":
    main()
