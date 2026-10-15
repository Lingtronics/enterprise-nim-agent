# Deployment

The API is a CPU client of a remote NVIDIA NIM endpoint. It does not host model weights or
require a GPU on the API machine. Docker, Compose, Kubernetes and Helm use one uvicorn worker
and one API replica. Durable tool state, pending approvals and audit records live in SQLite
on a persistent volume, even when retrieval vectors live in Qdrant. `Recreate` prevents
overlapping API pods during rollout; the chart rejects replica counts above one. Horizontal
scaling requires a shared transactional state/approval store before adding replicas or an HPA.

## Containerized pilot

Create private local identities with `python scripts/bootstrap_env.py`; `.env` must exist
before Compose starts. The default is `MODEL_MODE=demo`, which makes deterministic evidence
selection without calling NIM. To exercise real inference, set `MODEL_MODE=nim`, provide
`NIM_API_KEY`, and choose the reachable `NIM_BASE_URL` and `NIM_MODEL` in `.env`.

```text
docker build -t enterprise-nim-agent:0.1.0 .
docker compose config --quiet
docker compose up --build -d
docker compose ps
curl --fail http://127.0.0.1:8000/health/ready
docker compose logs agent-api
```

`config --quiet` checks Compose without printing resolved credential values. Compose reads
`.env` using dotenv syntax, including single-quoted JSON. Explicit `VECTOR_BACKEND=qdrant`
and `QDRANT_URL=http://vector-store:6333` override the local SQLite/localhost defaults. The API
retries vector-store initialization a bounded ten times; a service dependency starts Qdrant
first without relying on a curl executable inside its image. The API healthcheck uses Python
and checks both the runtime SQLite connection and the live vector index.

The API runs as UID/GID 10001 with a read-only root filesystem, a writable `/tmp`, and the
`api-state` named volume mounted at `/app/data`. Qdrant uses a separate `vector-data` volume.
No Qdrant host port is published. The standard Qdrant image retains its upstream root user
in Compose; isolate that service on this private pilot network. Kubernetes uses UID 10001
and volume group ownership for both services. Restrict access to the Docker daemon and back
up both volumes. `docker compose down` retains data; adding `--volumes` deletes durable state.

`requirements.lock` fixes Python runtime/build package versions. The Docker image installs
the lock first and installs this project without dependencies or isolated build downloads.
Python's `3.12-slim` base tag and upstream service tags can move: an enterprise image pipeline
should resolve and scan image digests before promoting an image to its private registry.

Qdrant is pinned to `qdrant/qdrant:v1.19.2`, verified against its
[official release](https://github.com/qdrant/qdrant/releases/tag/v1.19.2) on 14 October 2026.
Its HTTP API uses port 6333; the Kubernetes probes use the documented `/livez` and `/readyz`
[health endpoints](https://qdrant.tech/documentation/ops-monitoring/monitoring/).
Named volumes also avoid the Windows bind-mount storage caveat in Qdrant's
[local quickstart](https://qdrant.tech/documentation/quickstart/).

### Optional trace collector

Set the following private `.env` configuration before enabling the telemetry profile:

```text
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-collector:4318/v1/traces
docker compose --profile telemetry up --build -d
docker compose logs telemetry-collector
```

The first line is an environment-file entry, not a shell command. The profile starts the
collector only when requested. It accepts OTLP/HTTP on the Compose network and exports basic
trace summaries to its logs. Application instrumentation excludes prompts, document content,
tool arguments and credentials from attributes. The reference collector is pinned to the
verified [OpenTelemetry 0.162.0 release](https://github.com/open-telemetry/opentelemetry-collector-releases/releases/tag/v0.162.0).
It supplies neither a retained trace database nor a dashboard; replace the debug exporter
with an authenticated enterprise backend for retention. Remove the endpoint configuration
when the collector profile is disabled.

## Kubernetes

The resources in `k8s/` create namespace `enterprise-agent`, API and Qdrant Deployments and
ClusterIP Services, one PVC per service, a ConfigMap, and an API-only Qdrant ingress policy.
The cluster needs a default StorageClass, or add the appropriate `storageClassName` to both
PVCs. Its CNI must implement NetworkPolicy for the vector-store ingress restriction to apply.
The unauthenticated vector store must remain inside this trust boundary.

Build and publish `enterprise-nim-agent:0.1.0` to a registry the cluster can pull, then update
the API image in `k8s/deployment.yaml` to that registry/tag or load the image into your local
cluster. The default ConfigMap selects demo mode. Change `MODEL_MODE` to `nim` and configure
the endpoint/model to test NIM, with network egress to the endpoint.

Before applying the deployments, provision an existing Secret called
`enterprise-agent-secrets` in namespace `enterprise-agent` with exact keys `API_TOKENS_JSON`
and `NIM_API_KEY`. A secret manager/External Secrets controller is preferred. Demo mode still
requires a valid token map and the referenced NIM key may be an empty value. NIM mode requires
a nonempty key. Credentials are absent from the committed YAML and Helm values.

For a local demonstration, first create the namespace:

```text
kubectl apply -f k8s/namespace.yaml
```

The following Python example reads dotenv through validated `Settings`, writes private
temporary files, and sends file references to kubectl. Run it from the repository in the
configured Python environment. It creates a new Secret and intentionally fails if one exists;
rotate existing credentials through the secret manager or a reviewed secret update.

```python
import os
import subprocess
import tempfile
from pathlib import Path
from app.config.settings import Settings

settings = Settings()
with tempfile.TemporaryDirectory(prefix="agent-secrets-") as directory:
    destination = Path(directory)
    tokens = destination / "API_TOKENS_JSON"
    nim_key = destination / "NIM_API_KEY"
    tokens.write_text(settings.api_tokens_json, encoding="utf-8")
    nim_key.write_text(
        settings.nim_api_key.get_secret_value() if settings.nim_api_key else "",
        encoding="utf-8",
    )
    os.chmod(tokens, 0o600)
    os.chmod(nim_key, 0o600)
    subprocess.run([
        "kubectl", "--namespace", "enterprise-agent", "create", "secret", "generic",
        "enterprise-agent-secrets", f"--from-file=API_TOKENS_JSON={tokens}",
        f"--from-file=NIM_API_KEY={nim_key}",
    ], check=True)
```

Do not pass the dotenv file directly to `kubectl --from-env-file`: kubectl preserves dotenv
single quotes around JSON rather than parsing them like Settings/Compose, and includes
unrelated configuration keys. Temporary files disappear after the Python block; protect
your original `.env` and the destination cluster's secret access.

```text
python scripts/validate_deployment.py
kubectl apply --dry-run=client -f k8s/
kubectl apply --dry-run=server -f k8s/
kubectl apply -f k8s/
kubectl --namespace enterprise-agent rollout status deployment/enterprise-qdrant
kubectl --namespace enterprise-agent rollout status deployment/enterprise-agent
kubectl --namespace enterprise-agent port-forward service/enterprise-agent 8000:8000
```

Check `/health/live` and `/health/ready` via the forwarded local port. Startup probes permit
initialization; readiness probes include storage/index availability; liveness probes test the
process without restarting it merely because a downstream service is temporarily unavailable.
Resource requests/limits are starting values for this small corpus and need measurement for
your workload. The API is private by default. A public pilot needs an authenticated TLS
gateway and the enterprise's ingress/network policy before exposure.

## Helm

The chart creates equivalent namespaced resources, requires a pre-provisioned Secret, and
supports image, NIM endpoint/model, resources, storage and telemetry settings. It does not
create or accept literal secret values. Schema validation rejects unknown keys, empty secret
names, unpinned/latest image tags, invalid resource quantities and multiple API replicas.
Template checks also reject chunk overlap greater than or equal to chunk size.

```text
helm lint helm
helm template demo helm --namespace enterprise-agent > artifacts/helm-rendered.yaml
python scripts/validate_deployment.py --rendered artifacts/helm-rendered.yaml
helm upgrade --install demo helm --namespace enterprise-agent --create-namespace --set existingSecret=enterprise-agent-secrets --set image.repository=your-registry/enterprise-nim-agent
```

Use a reviewed values override for `model.mode=nim`, `model.baseUrl`, `model.name`, registry
tag, resource limits and StorageClass. `telemetry.endpoint` accepts the full OTLP/HTTP trace
URL of your separately managed collector. Keep `replicaCount=1` while SQLite provides
transactional approvals. Chart releases create their own PVCs; plan backup, retention and
restoration before uninstalling a release, which can delete its chart-managed PVCs.

## Verification limits

`scripts/validate_deployment.py` parses YAML and checks resource relationships, selectors,
container ports/probes, resource bounds, durable volumes, credential references and Compose
overrides. It labels its result **structural**. It does not start images, contact Kubernetes,
validate admission policies, resolve Secrets or provision storage. Helm renders/validates
chart values; it also does not prove that the image starts in a cluster.

On the implementation host, Docker and kubectl are unavailable and no Kubernetes context
is configured. Container build/Compose startup and kubectl dry runs therefore remain
unexecuted. Portable Helm 4.3.0 lint and rendering passed. Run the commands above on a host
with Docker and a configured target cluster before claiming deployment readiness.
