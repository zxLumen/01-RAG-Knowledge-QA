#!/usr/bin/env python3
"""Start the RAG Knowledge QA web server."""

import logging

import uvicorn

if __name__ == "__main__":
    # Make the app logger ("rag") visible in server/container logs (otherwise
    # logger.exception output is silently dropped).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        # Behind a reverse proxy (e.g. Caddy on the docker web network) the real
        # client IP arrives via X-Forwarded-For. Trusting it makes per-IP rate
        # limiting meaningful again. Safe here because the port is only exposed
        # on the internal network, never published directly.
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
