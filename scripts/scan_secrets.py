"""Check deliverable sources for the locally configured credential values."""

import json
import re
from pathlib import Path

from app.config.settings import Settings


def main():
    settings = Settings()
    secrets = list(settings.api_tokens)
    if settings.nim_api_key and settings.nim_api_key.get_secret_value():
        secrets.append(settings.nim_api_key.get_secret_value())
    roots = [
        "app",
        "tests",
        "docs",
        "evals",
        "benchmarks",
        "scripts",
        "k8s",
        "helm",
        "sample_data",
        ".github",
        "telemetry",
        "artifacts",
    ]
    files = set()
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".yaml", ".yml", ".toml"}:
                files.add(path)
    files.update(
        Path(name)
        for name in [
            "README.md",
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            "requirements.lock",
            ".env.example",
        ]
        if Path(name).exists()
    )
    findings = []
    provider_key = re.compile(r"nvapi-[A-Za-z0-9_-]{30,}")
    for path in sorted(files):
        content = path.read_text(encoding="utf-8-sig")
        if any(secret in content for secret in secrets) or provider_key.search(content):
            findings.append(path.as_posix())
    report = {
        "files_scanned": len(files),
        "credential_values_found": len(findings),
        "files": findings,
        "scope": (
            "Deliverable sources and artifacts; "
            "ignored private .env and tool/runtime files excluded."
        ),
    }
    output = Path("artifacts/secret-scan.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
