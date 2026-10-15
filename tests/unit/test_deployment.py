"""Detect broken deployment links and unsafe configuration regressions offline."""

import copy
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.validate_deployment import (
    UniqueKeyLoader,
    load_documents,
    validate_compose,
    validate_kubernetes,
    validate_repository,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def manifests():
    return [
        document
        for path in sorted((ROOT / "k8s").glob("*.yaml"))
        for document in load_documents(path)
    ]


def resource(manifests, kind, name):
    return next(
        item for item in manifests if item["kind"] == kind and item["metadata"]["name"] == name
    )


def test_repository_structural_validation_passes():
    result = validate_repository(ROOT)
    assert result["success"], result["errors"]
    assert result["validation"] == "structural"


@pytest.mark.parametrize(
    "regression",
    [
        "inline_credential",
        "missing_secret_key",
        "configmap_credential",
        "committed_secret",
        "wrong_selector",
        "missing_claim",
        "multiple_replicas",
        "rolling_update",
        "missing_resources",
        "wrong_probe_port",
        "wrong_service_port",
    ],
)
def test_kubernetes_regressions_are_detected(manifests, regression):
    documents = copy.deepcopy(manifests)
    deployment = resource(documents, "Deployment", "enterprise-agent")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if regression == "inline_credential":
        container["env"][0] = {"name": "API_TOKENS_JSON", "value": "unsafe-placeholder"}
    elif regression == "missing_secret_key":
        container["env"][0]["valueFrom"]["secretKeyRef"]["optional"] = True
    elif regression == "configmap_credential":
        resource(documents, "ConfigMap", "enterprise-agent-config")["data"]["NIM_API_KEY"] = "x"
    elif regression == "committed_secret":
        documents.append(
            {
                "apiVersion": "v1",
                "kind": "Secret",
                "metadata": {"name": "bad"},
                "stringData": {"NIM_API_KEY": "unsafe-placeholder"},
            }
        )
    elif regression == "wrong_selector":
        deployment["spec"]["selector"]["matchLabels"]["app"] = "no-pods"
    elif regression == "missing_claim":
        documents.remove(resource(documents, "PersistentVolumeClaim", "enterprise-agent-state"))
    elif regression == "multiple_replicas":
        deployment["spec"]["replicas"] = 2
    elif regression == "rolling_update":
        deployment["spec"]["strategy"]["type"] = "RollingUpdate"
    elif regression == "missing_resources":
        del container["resources"]["limits"]["memory"]
    elif regression == "wrong_probe_port":
        container["readinessProbe"]["httpGet"]["port"] = "missing-port"
    else:
        resource(documents, "Service", "enterprise-agent")["spec"]["ports"][0]["targetPort"] = 99
    assert validate_kubernetes(documents)


@pytest.mark.parametrize("regression", ["dotenv_override", "public_vectors", "lost_state"])
def test_compose_network_and_persistence_regressions_are_detected(regression):
    document = load_documents(ROOT / "docker-compose.yml")[0]
    if regression == "dotenv_override":
        document["services"]["agent-api"]["environment"]["VECTOR_BACKEND"] = "sqlite"
    elif regression == "public_vectors":
        document["services"]["vector-store"]["ports"] = ["6333:6333"]
    else:
        document["services"]["agent-api"]["volumes"] = []
    assert validate_compose(document)


def test_duplicate_yaml_configuration_is_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("kind: ConfigMap\nkind: Secret\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_documents(path)


@pytest.fixture
def helm_binary():
    binary = os.environ.get("HELM_BINARY") or shutil.which("helm")
    if not binary:
        candidates = sorted((ROOT / ".tools").glob("helm-*/windows-amd64/helm.exe"))
        binary = str(candidates[-1]) if candidates else None
    if not binary:
        pytest.skip("Helm binary unavailable; run Helm lint/render validation on an equipped host")
    return binary


def test_helm_rendered_chart_has_valid_resource_links(helm_binary):
    result = subprocess.run(
        [helm_binary, "template", "demo", str(ROOT / "helm"), "--namespace", "enterprise-agent"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    documents = [item for item in yaml.load_all(result.stdout, Loader=UniqueKeyLoader) if item]
    assert validate_kubernetes(documents) == []


@pytest.mark.parametrize(
    "setting",
    [
        "replicaCount=2",
        "existingSecret=",
        "image.tag=latest",
        "resources.limits.memory=invalid",
        "model.apiKey=unsupported",
        "retrieval.chunkOverlap=800",
    ],
)
def test_helm_rejects_unsafe_or_invalid_values(helm_binary, setting):
    result = subprocess.run(
        [helm_binary, "template", "demo", str(ROOT / "helm"), "--set", setting],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, f"Helm accepted invalid setting {setting}"
