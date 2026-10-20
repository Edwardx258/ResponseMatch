"""DeepSeek backend, through its OpenAI-compatible chat completions API.

DeepSeek's JSON mode guarantees valid JSON but not a particular schema, so the
schema goes into the system prompt and the reply is validated here, with one
retry that shows the model its validation error.
"""

from __future__ import annotations

import json
import os

import openai
from pydantic import ValidationError

from .llm import T, LLMError

DEFAULT_MODEL = "deepseek-v4.1-flash"  # if the API reports an unknown model, check the name in DeepSeek's docs
DEFAULT_BASE_URL = "https://api.deepseek.com"

JSON_INSTRUCTIONS = (
    "\n\nReply with a single JSON object, and nothing else, that conforms to this JSON schema:\n{schema}"
)


class StructuredDeepSeek:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: openai.OpenAI | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_attempts: int = 2,
    ):
        self.model = model
        self.client = client or openai.OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=base_url)
        self.max_attempts = max_attempts

    def call(self, *, system: str, prompt: str, schema: type[T], effort: str = "low") -> T:
        # effort has no DeepSeek equivalent here; it is accepted to keep the interface uniform
        schema_json = json.dumps(schema.model_json_schema(), indent=1)
        messages = [
            {"role": "system", "content": system + JSON_INSTRUCTIONS.format(schema=schema_json)},
            {"role": "user", "content": prompt},
        ]
        last_error = "no attempts made"
        for _ in range(self.max_attempts):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=8000,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if choice.finish_reason == "length":
                last_error = "response was cut off at max_tokens"
            elif not content.strip():
                last_error = "empty response"
            else:
                try:
                    return schema.model_validate_json(content)
                except ValidationError as e:
                    last_error = f"schema validation failed: {e.errors(include_url=False)}"
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": f"That JSON does not match the schema: {last_error}. "
                                                    "Reply again with corrected JSON only."},
                    ]
        raise LLMError(f"DeepSeek gave no valid {schema.__name__} after {self.max_attempts} attempts: {last_error}")
