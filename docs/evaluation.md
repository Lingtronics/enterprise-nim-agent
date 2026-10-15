# Evaluation methodology

Run `python -m evals.runner --mode demo --output artifacts/evaluation-demo.json` from the
repository directory after installing the development dependencies and bootstrapping `.env`.
Use `--mode nim` with a configured endpoint/key/model to assess live model behavior separately.

The dataset covers factual and multi-document retrieval, supported tool selection, pending and
decided approvals, injection instructions, restricted content, missing information, conflicts
and simulated provider failures. Every case uses isolated operational storage and synthetic
API identities. Failure simulations are labelled; they are not evidence of live endpoint
availability. The harness runs the real API through an in-process ASGI transport, including
authentication, graph orchestration, retrieval, policies and output validation.

Retrieved document IDs and returned citation metadata are checked against expectations.
Displayed evidence must occur in the cited authorised chunk. Expected answer terms and
unavailable behavior are separate checks, so a false abstention cannot receive task success
merely because it contains no unsupported claims. Tool selection includes unnecessary-tool
checks. Approval checks inspect operational storage before approval, after denial and after
repeated approval, rather than trusting the response's approval flag alone.

JSON output retains individual cases, checks, actual outputs, latency and aggregate metrics.
Retrieval-hit and source-coverage metrics apply to source-bearing cases. Grounding applies to
RAG answers; tool and approval denominators are stated in the machine-readable output.
The error/hallucination measure is a rule-based regression proxy for unsupported evidence,
unexpected outputs and runtime errors, not a semantic hallucination judge. An intentionally
simulated authentication failure can be a correct task outcome while still counting as a
runtime error. These two measurements answer different questions.

Demo evaluation assesses deterministic lexical retrieval/extraction and the runtime's safety
boundaries. It does not measure NIM reasoning, GPU inference, abstractive completeness or
enterprise business correctness. Live NIM results should be reviewed alongside source
relevance and answer completeness before replacing embeddings or relaxing output validation.
