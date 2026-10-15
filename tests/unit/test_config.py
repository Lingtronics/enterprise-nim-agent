import json

import pytest
from pydantic import ValidationError

from app.config.settings import Settings

TOKENS = json.dumps({"test-token": {"user_id": "alice", "scopes": ["public", "tools:read"]}})


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8080/v1/")
    monkeypatch.setenv("NIM_MODEL", "example/model")
    monkeypatch.setenv("NIM_API_KEY", "example-secret")
    monkeypatch.setenv("MODEL_MODE", "nim")
    monkeypatch.setenv("API_TOKENS_JSON", TOKENS)
    settings = Settings(_env_file=None)
    assert settings.nim_base_url == "http://localhost:8080/v1"
    assert settings.nim_model == "example/model"
    assert settings.nim_api_key.get_secret_value() == "example-secret"
    assert settings.api_tokens["test-token"].user_id == "alice"
    assert "example-secret" not in repr(settings)
    assert "test-token" not in repr(settings)
    assert "nim_api_key" not in settings.model_dump()
    assert "api_tokens_json" not in settings.model_dump()


def test_nim_credentials_required():
    with pytest.raises(ValidationError, match="NIM_API_KEY is required"):
        Settings(api_tokens_json=TOKENS, model_mode="nim", nim_api_key=None, _env_file=None)


@pytest.mark.parametrize("mapping", ["not-json", "[]", "{}", '{"token": {"user_id": "alice"}}'])
def test_authentication_configuration_rejected(mapping):
    with pytest.raises(ValidationError, match="API_TOKENS_JSON"):
        Settings(api_tokens_json=mapping, _env_file=None)


def test_credentials_do_not_leak_in_configuration_errors():
    with pytest.raises(ValidationError) as caught:
        Settings(api_tokens_json='{"sensitive-token": {"user_id": ""}}', _env_file=None)
    assert "sensitive-token" not in str(caught.value)


def test_chunk_overlap_validation():
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(api_tokens_json=TOKENS, chunk_size=100, chunk_overlap=100, _env_file=None)


@pytest.mark.parametrize("url", ["file:///tmp/model", "https://secret@example.com/v1", "relative"])
def test_invalid_model_url_rejected(url):
    with pytest.raises(ValidationError, match="service URL"):
        Settings(api_tokens_json=TOKENS, nim_base_url=url, _env_file=None)
