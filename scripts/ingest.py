"""Reconcile the manifest-controlled corpus with the configured vector index."""

import asyncio
import json

from app.config.settings import Settings
from app.retrieval.service import RetrievalService


async def main():
    service = RetrievalService(Settings())
    try:
        print(json.dumps(service.ingest(), sort_keys=True))
    finally:
        await service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
