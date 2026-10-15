# Enterprise Agent Runtime on NVIDIA NIM

A compact enterprise support reference: FastAPI, a bounded LangGraph workflow, configurable
NVIDIA NIM inference, scoped RAG, typed tools, durable human approvals, OpenTelemetry, evaluation
and repeatable performance measurements.

The default **demo** mode works offline with deterministic hash embeddings and extractive
evidence selection. **nim** mode uses the configured NIM chat-completions endpoint. Live NIM,
container and Kubernetes validation require the corresponding credentials and runtimes;
see [validation evidence](docs/validation.md) for what was actually executed.

## Local setup

Python 3.12 or newer is required. Run these commands from this repository directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe scripts/bootstrap_env.py
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
Bootstrap creates an ignored `.env` with random reader, operator and independent approver
tokens. It never overwrites an existing `.env`. All business endpoints require a bearer token;
health endpoints are public. The corpus is ingested from its trusted manifest during startup.

Open [API documentation](http://localhost:8000/docs),
[liveness](http://localhost:8000/health/live) or
[readiness](http://localhost:8000/health/ready). SQLite state is persisted under `data/`.
Readiness checks operational storage and the configured vector index. It does not probe NIM.

To use a real model, edit `.env`:

```dotenv
MODEL_MODE=nim
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_MODEL=your-served-model-name
NIM_API_KEY=your-secret-injected-key
```

`NIM_BASE_URL` must include the API version prefix; the adapter appends `/chat/completions`.
Self-hosted NIM URLs are supported. Credentials are required in NIM mode, including for a
controlled self-hosted endpoint. Permanent authentication/schema failures never fall back.
Retriable failure fallback is labelled explicitly and can be disabled with
`FALLBACK_ENABLED=false`. Timeouts and retry limits are environment-configurable.

## API use

Use a token from the private `.env` mapping in the `Authorization: Bearer <token>` header.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/query` | Original user question or supported operational request |
| `GET /v1/tools` | Registered typed contracts |
| `POST /v1/tools/call` | Precise tool request through the same graph and approval policy |
| `GET /v1/approvals/{request_id}` | Authorised pending/final action view |
| `POST /v1/approvals/{request_id}` | Explicit `approve` or `deny` from an independent approver |
| `POST /mcp` | Limited MCP-inspired JSON-RPC `tools/list` and `tools/call` gateway |
| `POST /v1/generate` | Authenticated direct NIM transport demonstration |
| `GET /v1/metrics` | Authenticated runtime request/error/latency counters |

Example request bodies:

```json
{"query": "Summarise the deployment requirements for Project Atlas."}
```

```json
{"name": "update_deployment_status", "arguments": {"project_id": "atlas", "new_status": "approved"}}
```

The write returns `status: pending_approval` and a request ID. A different configured human
with `approvals:decide`, `tools:write` and project access submits:

```json
{"decision": "approve"}
```

No missing decision or timeout grants approval. Repeated matching decisions return the durable
result; opposite decisions return 409. A request body may supply an optional UUID `request_id`
for idempotency. Reusing it for different action arguments or another requester returns 409.
The side effect, receipt, decision and audit commit in a single transaction.

Natural action grammar covers single-project deployment status reads, status changes to
`pending|approved|deployed|failed`, and deployment ticket creation. Ambiguous operations fail
closed. Typed calls are preferred for integrations. Model output and retrieved text cannot
select tools. The gateway exposes MCP-like contracts and JSON-RPC shapes; it is **not** a full
MCP transport/session implementation.

## Corpus and retrieval

`sample_data/manifest.json` defines document identity, timestamp and exact security scope.
Every `.md`/`.txt` file must appear in the manifest, and paths must stay inside the corpus.
Run `python scripts/ingest.py` to reconcile a changed corpus. SQLite reconciliation is atomic;
Qdrant replacement deletes the old corpus before publishing the new one to avoid stale scope
leaks, so it can briefly make the corpus unavailable.

Retrieval filters scopes before top-k. Documents are untrusted context. The model selects
verbatim evidence JSON; validation expands snippets to containing sentences/lines and renders
verified quotes with source IDs. This deliberately favours extractive grounding over fluent
abstractive summaries. Hash embeddings and demo lexical coverage can miss synonyms. Archived
Atlas requirements intentionally conflict with current replica requirements for evaluation.

## Tests, evaluation and benchmarking

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=.test_tmp/local -q
.\.venv\Scripts\ruff.exe check app tests evals benchmarks scripts
.\.venv\Scripts\python.exe -m evals.runner --mode demo --output artifacts/evaluation-demo.json
.\.venv\Scripts\python.exe -m benchmarks.load_test --mode demo --output artifacts/benchmark-demo.json
```

Tests use mocked HTTP NIM responses, scoped real retrieval and isolated transactional storage.
The evaluation harness checks source coverage, evidence grounding, tool selection, approval
effects, task success and failure behavior. Its deterministic grounding/error metrics are
regression proxies, not an LLM judge. Benchmark profiles include 10 interactive requests,
20 sequential baseline requests and 100 requests at concurrency 10, with component timings.
Reports label in-process or HTTP transport and model mode. TTFT is unavailable because the
current adapter does not stream; demo token rates and GPU costs are not invented.

Optional `PRICE_INPUT_PER_MILLION`, `PRICE_OUTPUT_PER_MILLION` and `PRICE_CURRENCY` estimate
token costs only when provider usage exists. Missing prices/usage produce an unavailable
estimate. [Benchmark report](docs/benchmark-report.md) contains measured results and limits.

## Deployment

```bash
docker build -t enterprise-nim-agent:local .
docker compose up --build
docker compose --profile telemetry up --build
```

Compose runs the API and Qdrant with persistent volumes, using private `.env` credentials.
Optional telemetry export uses an OpenTelemetry Collector. Set
`OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-collector:4318/v1/traces` when enabling the profile.
The variable expects the full trace endpoint. Without an exporter, trace IDs and content-safe
JSON logs remain available. Configure token pricing explicitly if cost estimates are wanted.

Kubernetes manifests and the optional Helm chart use an existing Secret rather than committed
credentials. See [deployment instructions](docs/deployment.md). Operational SQLite storage
supports one API replica with a persistent volume and `Recreate` rollout. Migrate operational
state to a shared transactional database before horizontal scaling.

## Design and limitations

[Architecture and Mermaid diagrams](docs/architecture.md),
[security assumptions](docs/security.md), [evaluation details](docs/evaluation.md) and
[validation evidence](docs/validation.md) describe the trust boundaries and measured scope.

Bearer configuration is a pilot identity mechanism; production OIDC, TLS, rate limiting,
network policies, backup/restore and document administration are integration work. Exactly-once
effects apply only to tools sharing the local SQLite transaction. External enterprise APIs
need their own idempotency guarantees. General semantic contradiction detection, PDF ingestion
and streaming are not implemented. Dynamo is deferred because this outbound-client runtime
has no demonstrated distributed inference scheduling requirement.

Primary references: [NVIDIA NIM API](https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html),
[LangGraph](https://reference.langchain.com/python/langgraph/graph/state),
[Qdrant filtering](https://qdrant.tech/documentation/search/filtering/),
[OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/instrumentation/).
