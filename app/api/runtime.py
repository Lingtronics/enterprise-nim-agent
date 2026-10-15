import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.propagate import extract
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.graph import AgentRuntime
from app.api.auth import Identity
from app.api.schemas import JsonRpcRequest, QueryRequest, ToolCallRequest
from app.approval.service import ApprovalService
from app.config.settings import Settings
from app.errors import RuntimeErrorBase
from app.models.nim_client import NIMClient
from app.models.router import ModelRouter
from app.retrieval.service import RetrievalService
from app.telemetry.instrumentation import bind_request, configure, log_request, span, trace_id
from app.tools.registry import ToolRegistry


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=12000)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "deny"]


def create_app(settings: Settings | None = None, model=None, span_exporter=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configuration = settings or Settings()
        application.state.ready = False
        application.state.settings = configuration
        provider = configure(configuration, span_exporter)
        application.state.metrics = {"requests": 0, "errors": 0, "latency_ms_sum": 0.0}
        primary = model or (NIMClient(configuration) if configuration.model_mode == "nim" else None)
        router = ModelRouter(
            primary,
            configuration.fallback_enabled,
            demo_mode=configuration.model_mode == "demo" and primary is None,
        )
        retrieval = RetrievalService(configuration)
        tools = ToolRegistry(configuration.database_path)
        approvals = ApprovalService(tools)
        application.state.model = primary
        application.state.router = router
        application.state.retrieval = retrieval
        application.state.tools = tools
        application.state.approvals = approvals
        application.state.runtime = AgentRuntime(router, retrieval, tools, approvals, configuration)
        try:
            for attempt in range(10):
                try:
                    await asyncio.to_thread(retrieval.ingest)
                    break
                except ValueError:
                    raise
                except Exception:
                    if configuration.vector_backend != "qdrant" or attempt == 9:
                        raise RuntimeErrorBase(
                            "retrieval_unavailable", "Vector index startup failed.", 503
                        ) from None
                    await asyncio.sleep(min(0.2 * 2**attempt, 2))
            application.state.ready = True
            yield
        finally:
            application.state.ready = False
            await router.aclose()
            await retrieval.aclose()
            provider.shutdown()

    application = FastAPI(title="Enterprise NIM Agent", version="0.1.0", lifespan=lifespan)

    @application.middleware("http")
    async def correlation(request: Request, call_next):
        request.state.request_id = str(uuid4())
        started = time.perf_counter()
        with (
            bind_request(request.state.request_id),
            span("http.request", context=extract(dict(request.headers))) as current,
        ):
            current.set_attribute("http.method", request.method)
            request.state.trace_id = trace_id()
            try:
                response = await call_next(request)
            except Exception:
                response = JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "code": "internal_error",
                            "message": "An internal operation failed.",
                            "retriable": False,
                        },
                        "request_id": request.state.request_id,
                        "trace_id": request.state.trace_id,
                    },
                )
            current.set_attribute("request_id", request.state.request_id)
            current.set_attribute("http.status_code", response.status_code)
            success = response.status_code < 400 and not getattr(request.state, "failed", False)
            current.set_attribute("success", success)
            latency_ms = (time.perf_counter() - started) * 1000
            request.app.state.metrics["requests"] += 1
            request.app.state.metrics["errors"] += int(not success)
            request.app.state.metrics["latency_ms_sum"] += latency_ms
            log_request(
                request.state.request_id,
                request.state.trace_id,
                response.status_code,
                latency_ms,
                success,
            )
            response.headers["X-Request-ID"] = request.state.request_id
            response.headers["X-Trace-ID"] = request.state.trace_id
        return response

    @application.exception_handler(RuntimeErrorBase)
    async def runtime_error(request: Request, exc: RuntimeErrorBase):
        return JSONResponse(
            status_code=exc.status_code,
            headers={"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None,
            content={
                "error": exc.as_dict(),
                "request_id": request.state.request_id,
                "trace_id": request.state.trace_id,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "input_invalid",
                    "message": "Request schema validation failed.",
                    "retriable": False,
                },
                "details": [
                    {"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()
                ],
                "request_id": request.state.request_id,
            },
        )

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An internal operation failed.",
                    "retriable": False,
                },
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @application.get("/health/live")
    async def live():
        return {"status": "ok"}

    @application.get("/health/ready")
    async def ready(request: Request):
        if not getattr(request.app.state, "ready", False):
            raise RuntimeErrorBase("not_ready", "The runtime is not ready.", 503)
        try:
            connection = request.app.state.tools.storage.connect()
            try:
                connection.execute("SELECT 1").fetchone()
            finally:
                connection.close()
            await request.app.state.retrieval.health()
        except Exception:
            raise RuntimeErrorBase("not_ready", "Runtime storage is unavailable.", 503) from None
        return {"status": "ok"}

    @application.post("/v1/query")
    async def query(body: QueryRequest, request: Request, identity: Identity):
        request.state.request_id = str(body.request_id or request.state.request_id)
        result = await request.app.state.runtime.run(body.query, identity, request.state.request_id)
        return (
            JSONResponse(status_code=502, content=result) if result["status"] == "error" else result
        )

    async def run_tool(body, request, identity):
        request.state.request_id = str(body.request_id or request.state.request_id)
        return await request.app.state.runtime.run(
            f"Call {body.name}", identity, request.state.request_id, body.name, body.arguments
        )

    @application.post("/v1/tools/call")
    async def tool_call(body: ToolCallRequest, request: Request, identity: Identity):
        return await run_tool(body, request, identity)

    @application.get("/v1/tools")
    async def list_tools(request: Request, identity: Identity):
        return {"tools": request.app.state.tools.list_tools()}

    @application.get("/v1/approvals/{request_id}")
    async def get_approval(request_id: UUID, request: Request, identity: Identity):
        return await asyncio.to_thread(request.app.state.approvals.get, str(request_id), identity)

    @application.post("/v1/approvals/{request_id}")
    async def decide(
        request_id: UUID, body: ApprovalDecision, request: Request, identity: Identity
    ):
        request.state.request_id = str(request_id)
        with (
            bind_request(str(request_id)),
            span(
                "approval.decision",
                approval_required=True,
                approval_status="approved" if body.decision == "approve" else "denied",
            ) as current,
        ):
            result = await asyncio.to_thread(
                request.app.state.approvals.decide, str(request_id), body.decision, identity
            )
            current.set_attribute("tool_name", result["tool_name"])
            return result

    @application.get("/v1/metrics")
    async def metrics(request: Request, identity: Identity):
        if "tools:read" not in identity.scopes:
            raise RuntimeErrorBase("metrics_forbidden", "Read permission is required.", 403)
        return dict(request.app.state.metrics)

    @application.post("/mcp")
    async def mcp(body: JsonRpcRequest, request: Request, identity: Identity):
        try:
            if body.method == "tools/list":
                result = {"tools": request.app.state.tools.list_tools()}
            elif body.method == "tools/call":
                value = await run_tool(
                    ToolCallRequest.model_validate(body.params), request, identity
                )
                result = {
                    "content": [{"type": "text", "text": json.dumps(value)}],
                    "structuredContent": value,
                    "isError": value["status"] == "error",
                }
            else:
                request.state.failed = True
                return {
                    "jsonrpc": "2.0",
                    "id": body.id,
                    "error": {"code": -32601, "message": "Method not supported."},
                }
            return {"jsonrpc": "2.0", "id": body.id, "result": result}
        except ValidationError:
            request.state.failed = True
            return {
                "jsonrpc": "2.0",
                "id": body.id,
                "error": {"code": -32602, "message": "Invalid tool call parameters."},
            }
        except RuntimeErrorBase as exc:
            request.state.failed = True
            return {
                "jsonrpc": "2.0",
                "id": body.id,
                "error": {"code": -32000, "message": exc.message, "data": exc.as_dict()},
            }

    @application.post("/v1/generate")
    async def generate(body: GenerateRequest, request: Request, identity: Identity):
        if request.app.state.model is None:
            raise RuntimeErrorBase(
                "model_disabled", "Direct generation requires MODEL_MODE=nim.", 503
            )
        return asdict(
            await request.app.state.model.generate([{"role": "user", "content": body.prompt}])
        )

    return application
