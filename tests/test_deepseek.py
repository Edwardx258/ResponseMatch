"""StructuredDeepSeek against a mocked HTTP transport (no network, no key)."""

import json

import httpx2
import openai
import pytest

from responsematch.config import Settings
from responsematch.deepseek import StructuredDeepSeek
from responsematch.graph import make_llm
from responsematch.llm import LLMError
from responsematch.semantic import LLMJudge
from responsematch.schemas import Judgements, ParsedRequest, ServiceRequest, Urgency, Volunteer


def fake_client(captured: list, replies: list[str], finish_reason="stop") -> openai.OpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        content = replies[min(len(captured), len(replies)) - 1]
        return httpx2.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        })

    return openai.OpenAI(api_key="test-key", base_url="https://api.deepseek.com",
                         http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


GOOD = json.dumps({"judgements": [{"volunteer_id": "v1", "score": 1.4, "rationale": "fits", "cited_passages": ["p#0"]}]})


def test_json_mode_request_and_validation():
    captured = []
    llm = StructuredDeepSeek(client=fake_client(captured, [GOOD]))
    out = llm.call(system="Judge.", prompt="go", schema=Judgements)

    assert out.judgements[0].volunteer_id == "v1"
    body = json.loads(captured[0].content)
    assert body["model"] == "deepseek-chat"
    assert body["response_format"] == {"type": "json_object"}
    assert '"judgements"' in body["messages"][0]["content"]  # schema is in the system prompt
    assert str(captured[0].url).startswith("https://api.deepseek.com")


def test_retries_once_with_the_validation_error():
    captured = []
    llm = StructuredDeepSeek(client=fake_client(captured, ['{"judgements": "oops"}', GOOD]))
    out = llm.call(system="Judge.", prompt="go", schema=Judgements)

    assert len(captured) == 2 and out.judgements[0].rationale == "fits"
    retry = json.loads(captured[1].content)["messages"]
    assert retry[-2]["role"] == "assistant" and "does not match the schema" in retry[-1]["content"]


def test_gives_up_after_max_attempts():
    llm = StructuredDeepSeek(client=fake_client([], ["not json at all"]))
    with pytest.raises(LLMError, match="no valid Judgements after 2 attempts"):
        llm.call(system="Judge.", prompt="go", schema=Judgements)


def test_truncated_reply_is_an_error():
    llm = StructuredDeepSeek(client=fake_client([], ['{"judge'], finish_reason="length"), max_attempts=1)
    with pytest.raises(LLMError, match="cut off"):
        llm.call(system="Judge.", prompt="go", schema=Judgements)


def test_judge_clamps_scores_from_any_backend():
    judge = LLMJudge(StructuredDeepSeek(client=fake_client([], [GOOD])))
    parsed = ParsedRequest(category="c", urgency=Urgency.low, keywords=[], required_skills=[], summary="s")
    v = Volunteer(id="v1", name="V", skills=[], lat=0, lon=0)
    out = judge.judge(ServiceRequest(id="r", text="t"), parsed, [v], [])
    assert out["v1"].score == 1.0


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    llm = make_llm(Settings(backend="deepseek", model=""))
    assert isinstance(llm, StructuredDeepSeek) and llm.model == "deepseek-chat"
    assert make_llm(Settings(backend="deepseek", model="deepseek-reasoner")).model == "deepseek-reasoner"
