# Project Atlas Deployment Requirements

Project Atlas requires GPU inference in the production and staging environments.
Project Atlas must run on Kubernetes with 2 replicas for the inference service.
API credentials must be injected through Kubernetes Secrets.
Readiness and liveness probes are required before a deployment receives traffic.
An operational deployment status change requires explicit human approval.
