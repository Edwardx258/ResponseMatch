"""The one interface the parser and the judge need from a model provider."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class StructuredLLM(Protocol):
    model: str

    def call(self, *, system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
        """Return the model's answer validated as `schema`.

        `effort` is a hint ("low" for extraction, "medium" for judging); providers
        without an equivalent setting ignore it."""
        ...
