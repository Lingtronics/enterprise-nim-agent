# Security assumptions and controls

Server-configured opaque bearer tokens bind identity and scopes. Bootstrap generates random
local tokens and writes them only to ignored `.env`. Production deployments must inject
`API_TOKENS_JSON` and `NIM_API_KEY` from a secret manager and terminate TLS. The sample token
mechanism has no expiry, issuer validation or identity-provider lifecycle; replace it with
enterprise JWT/OIDC authentication before exposure beyond a controlled pilot.

Document scopes are exact labels. A principal needs the document's scope; `public` means a
scope granted to an authenticated identity, not anonymous internet access. The ingestion
manifest is trusted administrator input, validates paths and controls classification. Editing
classification requires re-ingestion. Unmanifested files do not become available automatically.

Tools require `tools:read` or `tools:write` plus the project's security scope. Approval requires
`approvals:decide`, `tools:write`, project access and a human identity different from the
requester. The write effect is unreachable through a read call. Decisions are immutable and
must specify approve or deny. A pending action never expires into approval.

All argument schemas reject extra fields and invalid values. The registry rejects unknown
tools. State and audit use parameterised SQL. Model-produced tool calls and instructions in
documents cannot select or authorise operational tools. Output validation guarantees only
that displayed evidence is present in an authorised retrieved source, not that a source's
claims are true or that relevance is perfect. The simple instruction-pattern filter is
defence in depth; the execution boundary does not depend on detecting every injection phrase.

SQLite protects against repeated local effects by atomically storing effects, decisions,
audit and response. Storage must be persistent and backed up. Filesystem access to the DB
can expose authorised corpus and pending action data; protect the volume accordingly. The
reference has no encryption-at-rest management, external ticket-system connector, rate limiter
or tenant isolation beyond configured scopes. One process/replica is the supported deployment
shape. Request bodies are bounded by schemas; edge request-size and rate limits remain a
deployment responsibility.

Telemetry avoids query/document/token capture and raw provider errors. Audit records contain
operational arguments and user IDs, so restrict access and apply retention policy. Evaluation
and benchmark artifacts use synthetic sample data. Price estimates require explicit provider
token pricing; the runtime never invents GPU costs.

Validation evidence and untested deployment requirements are recorded in
`docs/validation.md` and `docs/benchmark-report.md` after local execution.
