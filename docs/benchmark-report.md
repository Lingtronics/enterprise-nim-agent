# Benchmark report

Measured at 2026-10-14T17:40:40.720492+00:00 using **demo mode** and **http transport**.

Machine-readable evidence: `artifacts/benchmark-http-demo.json`.

## Environment and workload

Python 3.12.4; Windows 11; 16 logical CPUs.
Observed model: deterministic-extractive-v1. Target: http://127.0.0.1:18000.
API backend: not independently exposed by HTTP target. Local retrieval baseline: SQLite. Corpus manifest SHA-256: `f02a692ec5db8906ae17d8a07608cf319bc44d7adc7ed1b2bc921dec3d288a04`.

Cyclic mix of three evaluated RAG queries and one read-only deployment-status query. No writes or approvals.
Startup/ingestion excluded; no warmup requests. Semaphore wait excluded from each submitted request latency but included in profile wall throughput.

| Profile | Requests / concurrency | HTTP p50 ms | HTTP p95 ms | Max ms | Requests/s | Errors |
|---|---:|---:|---:|---:|---:|---:|
| interactive | 10 / 1 | 9.689 | 22.721 | 29.648 | 83.695 | 0 / 10 |
| sequential_baseline | 20 / 1 | 11.693 | 16.668 | 19.553 | 80.807 | 0 / 20 |
| moderate | 100 / 10 | 76.883 | 170.602 | 247.538 | 105.297 | 0 / 100 |

## Baseline and optimization comparison

| Metric | Baseline | Optimized |
|---|---:|---|
| p50 latency ms | 11.693 | not measured |
| p95 latency ms | 16.668 | not measured |
| TTFT | unavailable (non-streaming) | not measured |
| requests/s | 80.807 | not measured |
| retrieval hit rate (separate evaluation) | 1.000 | not measured |
| task success rate (separate evaluation) | 1.000 | not measured |

Quality reference: `artifacts/evaluation-demo.json`; 24 separately executed cases, matching mode and local manifest fingerprint. These are deterministic regression metrics, not NIM generation-quality results. HTTP target corpus identity is not independently exposed by the API.

## Component timing and retrieval

| Profile | Model p50 ms | Retrieval p50 ms | Tool p50 ms | Runtime total p50 ms |
|---|---:|---:|---:|---:|
| interactive | 0.973 | 2.374 | 1.791 | 6.556 |
| sequential_baseline | 1.137 | 2.589 | 2.413 | 8.009 |
| moderate | 1.141 | 15.351 | 18.162 | 57.377 |

Independent local SQLite retrieval: 100 calls, p50 1.065 ms, p95 1.816 ms. This is a separate direct-service run and does not measure a remote vector store.

## Observed bottlenecks and tradeoffs

For interactive, the largest measured component median is retrieval. Component medians describe different applicable request subsets and should not be added together.
For sequential_baseline, the largest measured component median is retrieval. Component medians describe different applicable request subsets and should not be added together.
For moderate, the largest measured component median is tool. Component medians describe different applicable request subsets and should not be added together.

The small local corpus and deterministic extractor make CPU scheduling, SQLite access and HTTP/framework overhead visible. These measurements do not validate NIM service latency, GPU utilization, container performance, Kubernetes throughput or production scale.

Concurrency throughput and per-request latency should be assessed together. SQLite operational storage and the sample corpus are a single-instance reference baseline; the workload contains read operations only.

## Recommendations and limits

Measure genuine NIM latency and provider token usage with explicit credentials before choosing serving optimizations. Enable streaming and measure first-token events before reporting TTFT. Increase corpus/workload realism before comparing SQLite with Qdrant. Profile retrieval connections and scheduling before considering caching, and preserve scope filtering in any cache. Repeat comparable runs after an optimization; none was measured here.

Not measured: the adapter uses non-streaming completions. Unavailable TTFT is null, not zero.
Only provider-reported token usage is used; demo mode has no invented token counts or rates.
Token prices must be explicitly configured and usage present. Resource/time metrics are reported without GPU cost estimates.
