"""Strict schemas shared by the tool registry and MCP-inspired contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    project_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")

    @field_validator("project_id", mode="before")
    @classmethod
    def canonical_project_id(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


class GetDeploymentStatusArguments(ProjectArguments):
    pass


class UpdateDeploymentStatusArguments(ProjectArguments):
    new_status: Literal["pending", "approved", "deployed", "failed"]


class CreateDeploymentTicketArguments(ProjectArguments):
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("description", mode="before")
    @classmethod
    def strip_description(cls, value):
        return value.strip() if isinstance(value, str) else value
