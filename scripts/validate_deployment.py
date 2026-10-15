"""Offline structural checks, not Kubernetes API validation or container execution.

Use kubectl dry-run/Helm lint and start the containers separately on equipped hosts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

_CREDENTIAL_KEYS = {"API_TOKENS_JSON", "NIM_API_KEY"}


class UniqueKeyLoader(yaml.SafeLoader):
    """Do not silently accept duplicated selectors, credential keys or config fields."""


def _mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError("Duplicate YAML mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_documents(path: Path) -> list[dict]:
    documents = list(yaml.load_all(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader))
    if any(not isinstance(document, dict) for document in documents if document is not None):
        raise ValueError("Every YAML document must be an object")
    return [document for document in documents if document is not None]


def _pinned(image: str) -> bool:
    if not isinstance(image, str):
        return False
    if "@sha256:" in image:
        return bool(re.search(r"@sha256:[a-f0-9]{64}$", image))
    leaf = image.rsplit("/", 1)[-1]
    return ":" in leaf and leaf.rsplit(":", 1)[-1] not in {"", "latest"}


def validate_kubernetes(documents: list[dict]) -> list[str]:
    errors = []
    indexed = {}
    deployments = []
    for document in documents:
        metadata = document.get("metadata", {})
        name = metadata.get("name")
        namespace = metadata.get("namespace", "default")
        kind = document.get("kind")
        if not document.get("apiVersion") or not kind or not name:
            errors.append("Resource requires apiVersion, kind and metadata.name")
            continue
        identity = (namespace, kind, name)
        if identity in indexed:
            errors.append(f"Duplicate resource {kind}/{name}")
        indexed[identity] = document
        if kind == "Deployment":
            deployments.append(document)
        if kind == "Secret" and (document.get("data") or document.get("stringData")):
            errors.append(f"Secret/{name} contains committed credential data")
        if kind == "ConfigMap" and _CREDENTIAL_KEYS.intersection(document.get("data", {})):
            errors.append(f"ConfigMap/{name} contains credential keys")
        if kind == "HorizontalPodAutoscaler":
            errors.append("Horizontal scaling is unsupported with durable SQLite state")
    if not deployments:
        errors.append("At least one Deployment is required")
    api_found = False
    for deployment in deployments:
        name = deployment["metadata"]["name"]
        namespace = deployment["metadata"].get("namespace", "default")
        spec = deployment.get("spec", {})
        template = spec.get("template", {})
        labels = template.get("metadata", {}).get("labels", {})
        selector = spec.get("selector", {}).get("matchLabels", {})
        if not selector or any(labels.get(key) != value for key, value in selector.items()):
            errors.append(f"Deployment/{name} selector does not match pod labels")
        pod = template.get("spec", {})
        volumes = {volume.get("name"): volume for volume in pod.get("volumes", [])}
        for volume in volumes.values():
            claim = volume.get("persistentVolumeClaim", {}).get("claimName")
            if claim and (namespace, "PersistentVolumeClaim", claim) not in indexed:
                errors.append(f"Deployment/{name} references missing PVC/{claim}")
        for container in pod.get("containers", []):
            container_name = container.get("name", "unnamed")
            label = f"Deployment/{name} container {container_name}"
            if not _pinned(container.get("image")):
                errors.append(f"{label} requires a version tag or digest")
            ports = {port.get("name") for port in container.get("ports", [])}
            ports.update(port.get("containerPort") for port in container.get("ports", []))
            for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
                http = container.get(probe, {}).get("httpGet", {})
                if not http.get("path", "").startswith("/") or http.get("port") not in ports:
                    errors.append(f"{label} {probe} must use a declared HTTP port and path")
            resources = container.get("resources", {})
            if any(
                not resources.get(bound, {}).get(resource)
                for bound in ("requests", "limits")
                for resource in ("cpu", "memory")
            ):
                errors.append(f"{label} requires CPU/memory requests and limits")
            for mount in container.get("volumeMounts", []):
                if mount.get("name") not in volumes:
                    errors.append(f"{label} references an undeclared volume")
            for source in container.get("envFrom", []):
                config = source.get("configMapRef", {}).get("name")
                if config and (namespace, "ConfigMap", config) not in indexed:
                    errors.append(f"{label} references missing ConfigMap/{config}")
            credential_env = {}
            for entry in container.get("env", []):
                if entry.get("name") in _CREDENTIAL_KEYS:
                    credential_env[entry["name"]] = entry
                    reference = entry.get("valueFrom", {}).get("secretKeyRef", {})
                    if (
                        "value" in entry
                        or not reference.get("name")
                        or reference.get("key") != entry["name"]
                        or reference.get("optional")
                    ):
                        errors.append(f"{label} credentials require nonoptional Secret references")
            if container_name == "agent-api":
                api_found = True
                if set(credential_env) != _CREDENTIAL_KEYS:
                    errors.append(f"{label} must reference both API and NIM credential keys")
                if (
                    spec.get("replicas", 1) != 1
                    or spec.get("strategy", {}).get("type") != "Recreate"
                ):
                    errors.append(f"{label} must use one replica and Recreate for SQLite")
                security = pod.get("securityContext", {})
                if (
                    security.get("runAsUser") != 10001
                    or security.get("fsGroup") != 10001
                    or not security.get("runAsNonRoot")
                ):
                    errors.append(f"{label} must use nonroot UID and fsGroup 10001")
                state_mounts = [
                    mount
                    for mount in container.get("volumeMounts", [])
                    if mount.get("mountPath") == "/app/data"
                ]
                claim_names = [
                    volumes.get(mount["name"], {}).get("persistentVolumeClaim", {}).get("claimName")
                    for mount in state_mounts
                ]
                if not claim_names or any(not claim for claim in claim_names):
                    errors.append(f"{label} must persist /app/data in a PVC")
                for claim in claim_names:
                    pvc = indexed.get((namespace, "PersistentVolumeClaim", claim), {})
                    if pvc and pvc.get("spec", {}).get("accessModes") != ["ReadWriteOnce"]:
                        errors.append(f"{label} state PVC must be ReadWriteOnce")
    if not api_found:
        errors.append("An agent-api container is required")
    for document in documents:
        if document.get("kind") != "Service":
            continue
        namespace = document["metadata"].get("namespace", "default")
        name = document["metadata"]["name"]
        selector = document.get("spec", {}).get("selector", {})
        matching = [
            deployment
            for deployment in deployments
            if (
                deployment["metadata"].get("namespace", "default") == namespace
                and selector
                and all(
                    deployment["spec"]["template"]["metadata"]["labels"].get(key) == value
                    for key, value in selector.items()
                )
            )
        ]
        if not matching:
            errors.append(f"Service/{name} selector matches no deployment")
            continue
        targets = {
            port.get("name")
            for deployment in matching
            for container in deployment["spec"]["template"]["spec"].get("containers", [])
            for port in container.get("ports", [])
        }
        targets.update(
            port.get("containerPort")
            for deployment in matching
            for container in deployment["spec"]["template"]["spec"].get("containers", [])
            for port in container.get("ports", [])
        )
        if any(
            port.get("targetPort", port.get("port")) not in targets
            for port in document.get("spec", {}).get("ports", [])
        ):
            errors.append(f"Service/{name} targetPort matches no container port")
    return errors


def validate_compose(document: dict) -> list[str]:
    errors = []
    services = document.get("services", {})
    api = services.get("agent-api", {})
    vector = services.get("vector-store", {})
    if not api or not vector:
        return ["Compose requires agent-api and vector-store services"]
    environment = api.get("environment", {})
    if (
        environment.get("VECTOR_BACKEND") != "qdrant"
        or environment.get("QDRANT_URL") != "http://vector-store:6333"
    ):
        errors.append("Compose must override dotenv vector configuration for its service network")
    if _CREDENTIAL_KEYS.intersection(environment):
        errors.append("Compose credentials must come from the private env_file")
    if ".env" not in api.get("env_file", []):
        errors.append("Compose agent-api must load private .env credentials")
    health = api.get("healthcheck", {})
    if "/health/ready" not in str(health.get("test", "")) or not health.get("retries"):
        errors.append("Compose API requires a bounded readiness healthcheck")
    dependency = api.get("depends_on", {}).get("vector-store", {})
    if dependency.get("condition") != "service_started":
        errors.append(
            "Compose dependency must rely on API retries without a Qdrant shell healthcheck"
        )
    if api.get("user") != "10001:10001" or not api.get("read_only"):
        errors.append("Compose API must use nonroot UID and a read-only root filesystem")
    if vector.get("ports"):
        errors.append("Compose Qdrant must remain private without published host ports")
    if not _pinned(vector.get("image")):
        errors.append("Compose Qdrant requires a version tag or digest")
    named = document.get("volumes", {})
    for service, path in ((api, "/app/data"), (vector, "/qdrant/storage")):
        mounts = [
            mount.split(":") for mount in service.get("volumes", []) if isinstance(mount, str)
        ]
        if not any(len(mount) >= 2 and mount[0] in named and mount[1] == path for mount in mounts):
            errors.append(f"Compose {path} must use a declared named volume")
    collector = services.get("telemetry-collector", {})
    if collector and "telemetry" not in collector.get("profiles", []):
        errors.append("Compose collector must remain optional through its telemetry profile")
    return errors


def validate_dockerfile(content: str) -> list[str]:
    checks = {
        "CPU API image requires Python 3.12 slim": "FROM python:3.12-slim" in content,
        "Docker image must install the dependency lock": "COPY requirements.lock" in content
        and "--requirement requirements.lock" in content,
        "Docker image must use nonroot UID 10001": "USER 10001:10001" in content,
        "Docker healthcheck must check readiness": "HEALTHCHECK" in content
        and "/health/ready" in content,
        "Docker API must use a single uvicorn worker": '"--workers", "1"' in content,
    }
    return [message for message, passed in checks.items() if not passed]


def validate_repository(root: Path, rendered: Path | None = None) -> dict:
    errors = []
    checked = []
    try:
        documents = []
        for path in sorted((root / "k8s").glob("*.yaml")):
            documents.extend(load_documents(path))
            checked.append(path.relative_to(root).as_posix())
        errors.extend(validate_kubernetes(documents))
        compose = load_documents(root / "docker-compose.yml")
        if len(compose) != 1:
            errors.append("Compose must be one YAML document")
        else:
            errors.extend(validate_compose(compose[0]))
        checked.extend(["docker-compose.yml", "Dockerfile"])
        errors.extend(validate_dockerfile((root / "Dockerfile").read_text(encoding="utf-8")))
        # Helm itself validates values against this schema when linting/rendering.
        json.loads((root / "helm" / "values.schema.json").read_text(encoding="utf-8"))
        if rendered is not None:
            errors.extend(validate_kubernetes(load_documents(rendered)))
            checked.append(str(rendered))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"Deployment parsing failed ({type(exc).__name__})")
    return {"validation": "structural", "success": not errors, "checked": checked, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rendered", type=Path, help="Also check YAML emitted by helm template")
    arguments = parser.parse_args()
    result = validate_repository(arguments.root.resolve(), arguments.rendered)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
