# Project Zephyr Deployment Requirements

Project Zephyr uses CPU inference in the development environment.
Project Zephyr requires a single replica for its inference service.
Project Zephyr does not require a GPU for its development workload.
Service credentials must be injected through Kubernetes Secrets.
