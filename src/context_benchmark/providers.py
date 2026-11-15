"""Minimal providers: deterministic offline mock and guarded OpenAI-compatible API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ModelRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    cost: float | None
    raw: dict[str, Any]


class Provider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


class MockProvider:
    """Offline provider. A callable or queued responses can emulate failures."""

    def __init__(self, responses: list[str] | None = None, callback: Callable[[ModelRequest], str] | None = None) -> None:
        self.responses = list(responses or [])
        self.callback = callback
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        started = time.perf_counter()
        if self.callback:
            text = self.callback(request)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            text = '{"constraints":[]}'
        return ModelResponse(
            text=text,
            prompt_tokens=sum(len(message["content"]) for message in request.messages) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=(time.perf_counter() - started) * 1000,
            cost=0.0,
            raw={"mock": True},
        )


class OpenAICompatibleProvider:
    def __init__(
        self,
        base_url: str,
        *,
        api_key_env: str,
        allow_api_calls: bool = False,
        timeout_seconds: int = 120,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ) -> None:
        if not allow_api_calls:
            raise PermissionError("Real API calls are disabled. Verify pinned official model IDs before enabling.")
        if not base_url or not api_key_env:
            raise ValueError("base_url and api_key_env are required")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.input_price = input_price_per_million
        self.output_price = output_price_per_million

    def complete(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise PermissionError(f"Missing API key environment variable: {self.api_key_env}")
        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "seed": request.seed,
        }
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Provider HTTP {exc.code}: {body}") from exc
        latency = (time.perf_counter() - started) * 1000
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        cost = None
        if self.input_price is not None and self.output_price is not None and prompt_tokens is not None and completion_tokens is not None:
            cost = (prompt_tokens * self.input_price + completion_tokens * self.output_price) / 1_000_000
        return ModelResponse(
            text=raw["choices"][0]["message"]["content"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            cost=cost,
            raw=raw,
        )
