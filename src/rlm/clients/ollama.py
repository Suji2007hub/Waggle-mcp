from collections import defaultdict
from typing import Any

import requests

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaClient(BaseLM):
    """LM client for Ollama's local chat API."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(self, prompt: str | list[dict[str, Any]], model: str | None = None) -> str:
        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Ollama client.")

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": self._to_messages(prompt),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        self._track_usage(payload, model)
        return str(payload.get("message", {}).get("content", "") or "")

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        return self.completion(prompt, model=model)

    def _to_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return [
                {
                    "role": str(item.get("role", "user")),
                    "content": str(item.get("content", "")),
                }
                for item in prompt
            ]
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, payload: dict[str, Any], model: str) -> None:
        self.model_call_counts[model] += 1
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        completion_tokens = int(payload.get("eval_count") or 0)
        self.model_input_tokens[model] += prompt_tokens
        self.model_output_tokens[model] += completion_tokens
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(
            model_usage_summaries={
                model: ModelUsageSummary(
                    total_calls=self.model_call_counts[model],
                    total_input_tokens=self.model_input_tokens[model],
                    total_output_tokens=self.model_output_tokens[model],
                )
                for model in self.model_call_counts
            }
        )

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
        )
