"""StructuredClaude against a mocked HTTP transport: checks the request we send
and how responses are handled, without calling the API."""

import json

import anthropic
import httpx2
import pytest

from responsematch.claude import ClaudeRefusal, StructuredClaude
from responsematch.parsing import ClaudeParser
from responsematch.schemas import ServiceRequest, Urgency


def fake_client(captured: list, payload: dict, stop_reason="end_turn") -> anthropic.Anthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        content = [] if stop_reason == "refusal" else [{"type": "text", "text": json.dumps(payload)}]
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "stop_details": {"type": "refusal", "category": None, "explanation": None} if stop_reason == "refusal" else None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        })

    return anthropic.Anthropic(api_key="test-key", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


PARSED = {
    "category": "medical assistance", "urgency": "high", "keywords": ["injured"], "required_skills": ["first aid"],
    "required_certifications": ["first aid"], "languages": [], "location_text": None, "summary": "Someone is injured.",
}


def test_parser_sends_structured_request_with_fallback():
    captured = []
    parser = ClaudeParser(StructuredClaude("claude-opus-5-5", client=fake_client(captured, PARSED)))
    out = parser.parse(ServiceRequest(id="r", text="Someone is injured", category_hint="medical"))

    assert out.urgency is Urgency.high and out.required_certifications == ["first aid"]
    body = json.loads(captured[0].content)
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "Reported category: medical" in body["messages"][0]["content"]
    assert "server-side-fallback-2026-07-01" in captured[0].headers["anthropic-beta"]


def test_refusal_raises():
    claude = StructuredClaude("claude-opus-5-5", client=fake_client([], {}, stop_reason="refusal"))
    with pytest.raises(ClaudeRefusal):
        ClaudeParser(claude).parse(ServiceRequest(id="r", text="x"))
