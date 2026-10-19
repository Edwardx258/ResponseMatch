"""Thin wrapper for structured Claude calls used by the parser and the judge."""

from __future__ import annotations

from typing import TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Server-side fallback: a refused request is re-run on Anthropic's recommended
# model for that refusal category instead of coming back empty.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeError(RuntimeError):
    pass


class ClaudeRefusal(ClaudeError):
    pass


class StructuredClaude:
    def __init__(self, model: str, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def call(self, *, system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise ClaudeRefusal(f"Claude declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise ClaudeError("Response hit max_tokens before the structured output was complete")
        if response.parsed_output is None:
            raise ClaudeError(f"No structured output (stop_reason={response.stop_reason})")
        return response.parsed_output
