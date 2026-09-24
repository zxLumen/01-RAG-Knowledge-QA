from __future__ import annotations

from src.qa import llm


class _FakeResp:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def iter_lines(self):
        yield from self._lines


def test_iter_deltas_openai():
    resp = _FakeResp([
        b'data: {"choices":[{"delta":{"content":"\xe4\xbd\xa0"}}]}',
        b'',
        b'data: {"choices":[{"delta":{"content":"\xe5\xa5\xbd"}}]}',
        b'data: [DONE]',
        b'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ])
    profile = {"protocol": "openai"}
    assert list(llm._iter_deltas(profile, resp)) == ["你", "好"]


def test_iter_deltas_ollama():
    resp = _FakeResp([
        b'{"message":{"content":"a"},"done":false}',
        b'',
        b'{"message":{"content":"b"},"done":false}',
        b'{"message":{"content":""},"done":true}',
    ])
    profile = {"protocol": "ollama"}
    assert list(llm._iter_deltas(profile, resp)) == ["a", "b"]


def test_endpoint_and_payload():
    ollama = {
        "provider": "ollama",
        "protocol": "ollama",
        "base_url": "http://localhost:11434/",
        "model": "m",
    }
    assert llm._endpoint(ollama) == "http://localhost:11434/api/chat"
    payload = llm._payload(ollama, [{"role": "user", "content": "hi"}], stream=True)
    assert payload["stream"] is True and "options" in payload

    oai = {
        "provider": "deepseek",
        "protocol": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    }
    assert llm._endpoint(oai) == "https://api.deepseek.com/v1/chat/completions"
    payload = llm._payload(oai, [], stream=False)
    assert payload["stream"] is False and "options" not in payload
