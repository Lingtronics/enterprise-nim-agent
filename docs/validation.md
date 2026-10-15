# Validation record — 14 October 2026

The implementation and offline reference workflows are exercised locally. The specification's
full deployment Definition of Done remains open until live NIM, Docker and Kubernetes checks
run in an environment that supplies those dependencies.

## Executed locally

The final automated suite passed **211 tests, with zero failures, errors or skips** in
9.05 seconds; totals are recorded in `artifacts/tests.xml`. Tests cover model
configuration/retries/fallback, bounded LangGraph routing, scoped SQLite and in-memory Qdrant
retrieval, source validation, strict tools, independent human approvals, concurrency, rollback,
restart/replay, prompt injection, authentication, HTTP API behavior, tracing and evaluation.
Performance tests exercise the measurement harness without brittle latency thresholds.

| Check | Evidence |
| --- | --- |
| Unit/integration/performance regression suite | `artifacts/tests.xml`, 211 passed |
| Ruff source checks | Executed against app, tests, evals, benchmarks and scripts |
| 24-case offline evaluation | `artifacts/evaluation-demo.json`, 24/24 task success |
| Retrieval hit and grounding regression checks | 15/15 retrieval hits, 17/17 grounded cases |
| Expected failure simulation | One simulated auth error, included in the 1/24 error proxy |
| Real loopback HTTP smoke | `artifacts/http-smoke-demo.json`: health 200, anonymous 401, reader write 403, cited RAG |
| Kubernetes/Compose/Dockerfile structural checks | `artifacts/deployment-validation.json` |
| Helm lint and rendering | `artifacts/helm-lint.txt`, `artifacts/helm-rendered.yaml` |
| Secret-value source scan | `artifacts/secret-scan.json`, no configured credential values found |
| HTTP and ASGI workload measurements | 260 requests, zero errors; benchmark JSON and `docs/benchmark-report.md` |

The primary HTTP moderate profile used 100 requests at concurrency 10 and measured p50
76.883 ms, p95 170.602 ms and 105.297 requests/second. This is a CPU demo measurement, with
provider inference, TTFT, GPU throughput and GPU costs unmeasured. The temporary HTTP server
was stopped after validation.

Helm 4.3.0 was downloaded from the official release URL and SHA-256 verified against the
[tagged release](https://github.com/helm/helm/releases/tag/v4.3.0):
`304ea163cce4d9ad14e189c01846c6a34de9cfdfe48536ae54b2e8ba7884e67c` (Windows amd64 zip).
The chart also rejects unsafe/malformed values in automated tests. Structural validation is a
repository-specific invariant check; it is not Kubernetes API schema or cluster validation.
The source scan checks configured local credential values and NVIDIA key patterns in
deliverable sources/artifacts; it is not a general secret-discovery product. Private `.env`,
virtual environments, local tooling and temporary test data are excluded from deliverables.

## External validation still required

| Required check | Current limitation | Reproduction |
| --- | --- | --- |
| Live NVIDIA NIM inference | No NIM API key configured | Set NIM endpoint/model/key and run `python -m evals.runner --mode nim` |
| Docker build | Docker is not installed on this host | `docker build -t enterprise-nim-agent:0.1.0 .` |
| Docker Compose healthy startup | Docker engine is unavailable | `docker compose up --build --wait` |
| Kubernetes client dry run | No kubectl or kubeconfig/cluster available | `kubectl apply --dry-run=client -f k8s/` with the target cluster configured |
| Kubernetes readiness/PVC behavior | No cluster deployment executed | Follow `docs/deployment.md` and verify probes and persistent restart behavior |
| OTLP collector export | Exporter wiring tested through in-memory spans, collector not started | Enable Compose telemetry profile and inspect collector output |
| CI jobs | Workflow is provided locally; no hosted CI run occurred | Push the repository to an authorised GitHub remote |

No live GPU-serving, distributed Dynamo, on-prem/hybrid, production load or external ticket
system deployment is claimed. The measured demo uses deterministic extraction and local CPU
storage. See the benchmark report for transport, workload, timestamps and measurement limits.
