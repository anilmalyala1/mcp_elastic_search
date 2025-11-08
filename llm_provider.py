"""Helpers for initializing planner LLM clients."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def initialize_openai_client(api_key: Optional[str]):
    """Return an OpenAI client instance when the dependency is available."""

    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        return None
    return OpenAI(api_key=api_key)


class PlannerLLMClient:
    """Small wrapper around an OpenAI client for planning queries."""

    def __init__(self, api_key: Optional[str], model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model
        self._client = initialize_openai_client(api_key)

    def plan(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Request a plan from the language model, returning parsed JSON."""

        if not self._client:
            return None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception:
            return None
        if not response.choices:
            return None
        content = response.choices[0].message.content
        if not content:
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed
