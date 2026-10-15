import json
from pathlib import Path

import pytest


def pytest_configure(config):
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def settings(tmp_path):
    from app.config.settings import Settings

    return Settings(
        _env_file=None,
        api_tokens_json=json.dumps(
            {
                "test-reader-token": {"user_id": "reader", "scopes": ["public", "tools:read"]},
                "test-operator-token": {
                    "user_id": "operator",
                    "scopes": ["public", "internal", "tools:read", "tools:write"],
                },
                "test-approver-token": {
                    "user_id": "approver",
                    "scopes": [
                        "public",
                        "internal",
                        "tools:read",
                        "tools:write",
                        "approvals:decide",
                    ],
                },
            }
        ),
        database_path=tmp_path / "runtime.db",
        model_mode="demo",
    )
