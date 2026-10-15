FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATABASE_PATH=/app/data/runtime.db \
    CORPUS_PATH=/app/sample_data

WORKDIR /app
COPY requirements.lock ./requirements.lock
RUN python -m pip install --requirement requirements.lock

COPY pyproject.toml ./pyproject.toml
COPY app ./app
COPY sample_data ./sample_data
RUN python -m pip install --no-deps --no-build-isolation . \
    && groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --no-create-home agent \
    && mkdir -p /app/data \
    && chown 10001:10001 /app/data

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=4 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
