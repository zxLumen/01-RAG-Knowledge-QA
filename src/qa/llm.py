"""Unified LLM client supporting Ollama and OpenAI-compatible providers.

Streaming requests are fired from a daemon thread and the generator polls a
cancellation event, so a stop request takes effect instantly even while the
provider is still in its (potentially long) prefill phase. Cancellation reuses
:mod:`src.qa.cancel`: closing the live response aborts generation upstream.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Generator

import requests

from src.qa import cancel, providers
from src.qa.llm_config import resolve_key

REQUEST_TIMEOUT = 300
OLLAMA_NUM_CTX = 8192

# Identify the client with a real name (providers such as OpenCode Go ask clients
# not to masquerade as a generic HTTP library).
APP_USER_AGENT = "rag-knowledge-qa/1.0"
# Providers that require a stable per-conversation session id for routing/caching.
_SESSION_HEADER_PROVIDERS = {"opencode", "opencode-go"}


def _headers(profile: dict, session_id: str | None = None) -> dict:
    key = resolve_key(profile.get("api_key"))
    headers = {
        "Content-Type": "application/json",
        "User-Agent": APP_USER_AGENT,
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if profile.get("provider") in _SESSION_HEADER_PROVIDERS:
        # OpenCode Go returns 400 MissingSessionID without this header; reusing the
        # app's chat session id keeps routing and prompt caching stable per chat.
        headers["x-opencode-session"] = session_id or uuid.uuid4().hex
    return headers


def _endpoint(profile: dict) -> str:
    base = (profile.get("base_url") or "").rstrip("/")
    protocol = profile.get("protocol") or providers.protocol_for(profile.get("provider"))
    path = "/api/chat" if protocol == providers.PROTOCOL_OLLAMA else "/chat/completions"
    return base + path


def _payload(profile: dict, messages: list[dict], stream: bool) -> dict:
    protocol = profile.get("protocol") or providers.protocol_for(profile.get("provider"))
    temperature = profile.get("temperature", 0.0)
    if protocol == providers.PROTOCOL_OLLAMA:
        return {
            "model": profile.get("model", ""),
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
            "stream": stream,
        }
    return {
        "model": profile.get("model", ""),
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }


def _iter_deltas(profile: dict, resp) -> Generator[str, None, None]:
    protocol = profile.get("protocol") or providers.protocol_for(profile.get("provider"))
    if protocol == providers.PROTOCOL_OLLAMA:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            text = (data.get("message") or {}).get("content")
            if text:
                yield text
        return

    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        text = (choices[0].get("delta") or {}).get("content")
        if text:
            yield text


def stream_chat(
    profile: dict, messages: list[dict], gen_id: str | None = None
) -> Generator[str, None, None]:
    """Yield text deltas from ``profile``; cancellation ends iteration early.

    Raises on connection/HTTP errors.
    """
    stop = threading.Event()
    ready = threading.Event()
    box: dict = {}
    err_box: dict = {}

    def _post() -> None:
        try:
            resp = requests.post(
                _endpoint(profile),
                json=_payload(profile, messages, stream=True),
                headers=_headers(profile, gen_id),
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code >= 400:
                detail = resp.text[:300]
                resp.close()
                err_box["exc"] = RuntimeError(f"HTTP {resp.status_code}: {detail}")
                ready.set()
                return
            if stop.is_set() or cancel.is_cancelled(gen_id):
                resp.close()
                return
            box["resp"] = resp
            ready.set()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            err_box["exc"] = exc
            ready.set()

    threading.Thread(target=_post, daemon=True).start()

    try:
        while not ready.is_set():
            if cancel.is_cancelled(gen_id):
                return
            time.sleep(0.1)

        if "exc" in err_box:
            raise err_box["exc"]

        resp = box["resp"]
        with resp:
            cancel.attach(gen_id, resp)
            for delta in _iter_deltas(profile, resp):
                if cancel.is_cancelled(gen_id):
                    return
                yield delta
    finally:
        stop.set()


def complete(profile: dict, messages: list[dict]) -> str:
    """Non-streaming completion; returns the full text."""
    resp = requests.post(
        _endpoint(profile),
        json=_payload(profile, messages, stream=False),
        headers=_headers(profile),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    protocol = profile.get("protocol") or providers.protocol_for(profile.get("provider"))
    if protocol == providers.PROTOCOL_OLLAMA:
        return (data.get("message") or {}).get("content", "")
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "")


def list_models(profile: dict) -> list[str]:
    """Fetch the provider's live model list (Ollama /api/tags or OpenAI /models).

    No fallback: a valid API key is required for OpenAI-compatible providers,
    and errors (e.g. 401) propagate to the caller.
    """
    base = (profile.get("base_url") or "").rstrip("/")
    protocol = profile.get("protocol") or providers.protocol_for(profile.get("provider"))
    if protocol == providers.PROTOCOL_OLLAMA:
        resp = requests.get(f"{base}/api/tags", timeout=15)
        resp.raise_for_status()
        return sorted(
            m.get("name", "") for m in resp.json().get("models", []) if m.get("name")
        )
    resp = requests.get(f"{base}/models", headers=_headers(profile), timeout=15)
    resp.raise_for_status()
    return sorted(m.get("id", "") for m in resp.json().get("data", []) if m.get("id"))
