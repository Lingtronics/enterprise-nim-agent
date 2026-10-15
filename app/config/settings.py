"""Environment configuration and validated bearer-token identities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class TokenIdentity(BaseModel):
    """Identity bound to a server-configured opaque bearer token."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    user_id: str = Field(min_length=1)
    scopes: list[str]

    @field_validator("user_id")
    @classmethod
    def nonblank_user(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_id must not be blank")
        return value

    @field_validator("scopes")
    @classmethod
    def nonblank_scopes(cls, values: list[str]) -> list[str]:
        if any(not scope.strip() for scope in values):
            raise ValueError("scopes must contain nonblank strings")
        return list(dict.fromkeys(values))


class Settings(BaseSettings):
    """Load environment variables without an implicit development credential.

    ``API_TOKENS_JSON`` is always required, including in deterministic demo mode.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )

    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = Field(default="meta/llama-3.1-8b-instruct", min_length=1)
    nim_api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    nim_timeout_seconds: float = Field(default=30, gt=0, le=600)
    nim_max_retries: int = Field(default=2, ge=0, le=5)
    nim_backoff_seconds: float = Field(default=0.05, ge=0, le=30)
    model_mode: Literal["nim", "demo"] = "demo"
    fallback_enabled: bool = True
    api_tokens_json: str = Field(repr=False, exclude=True)
    database_path: Path = Path("data/runtime.db")
    corpus_path: Path = Path("sample_data")
    vector_backend: Literal["sqlite", "qdrant"] = "sqlite"
    qdrant_url: str = "http://localhost:6333"
    retrieval_top_k: int = Field(default=6, gt=0, le=100)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    otel_exporter_otlp_endpoint: str | None = None
    price_input_per_million: float | None = Field(default=None, ge=0)
    price_output_per_million: float | None = Field(default=None, ge=0)
    price_currency: str = Field(default="USD", pattern="^[A-Z]{3}$")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    _api_tokens: dict[str, TokenIdentity] = PrivateAttr(default_factory=dict)

    @field_validator("nim_base_url", "qdrant_url")
    @classmethod
    def validate_service_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("service URL must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("service URL must not contain credentials, query, or fragment")
        return value.rstrip("/")

    @field_validator("otel_exporter_otlp_endpoint")
    @classmethod
    def validate_telemetry_url(cls, value: str | None) -> str | None:
        return cls.validate_service_url(value) if value else None

    @model_validator(mode="after")
    def validate_runtime(self) -> Settings:
        if self.model_mode == "nim" and (
            self.nim_api_key is None or not self.nim_api_key.get_secret_value().strip()
        ):
            raise ValueError("NIM_API_KEY is required when MODEL_MODE=nim")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        try:
            parsed = json.loads(self.api_tokens_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("API_TOKENS_JSON must be a JSON token-to-identity object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("API_TOKENS_JSON must be a JSON token-to-identity object")
        if not parsed:
            raise ValueError("API_TOKENS_JSON must configure at least one bearer token")
        identities: dict[str, TokenIdentity] = {}
        for token, identity in parsed.items():
            if not isinstance(token, str) or not token.strip():
                raise ValueError("API_TOKENS_JSON contains an empty bearer token")
            try:
                identities[token] = TokenIdentity.model_validate(identity)
            except ValueError as exc:
                # Keep token strings and nested configuration out of error text.
                raise ValueError(
                    "API_TOKENS_JSON identities require user_id and a list of scopes"
                ) from exc
        self._api_tokens = identities
        return self

    @property
    def api_tokens(self) -> dict[str, TokenIdentity]:
        """Validated token mapping; never add its contents to logs or traces."""
        return self._api_tokens.copy()
