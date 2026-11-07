#!/usr/bin/env python3
"""
A tiny stand-in for an expensive LLM (PROVIDED — do not edit).

Real retrieval, caching, and compaction save money AND latency because they
shrink (or skip) the prompt the model has to prefill. We model that here:
"processing" a token costs wall-clock time, but a cache hit costs only the
base overhead — mirroring a generation that never happened.

This lets `run_demo.py` show a real latency drop on cached / compacted turns
without calling (or paying for) a real API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from context_os import RetrievalHit, Usage


@dataclass
class ModelResponse:
    latency_s: float
    output_tokens: int
    text: str


class MockModel:
    """Latency is proportional to prompt tokens on a miss, near-zero on a hit."""

    def __init__(
        self,
        *,
        seconds_per_prefill_token: float = 5e-5,
        seconds_per_output_token: float = 2e-4,
        base_latency_s: float = 0.02,
        output_tokens: int = 80,
    ) -> None:
        self.seconds_per_prefill_token = seconds_per_prefill_token
        self.seconds_per_output_token = seconds_per_output_token
        self.base_latency_s = base_latency_s
        self.output_tokens = output_tokens

    def generate(
        self,
        usage: Usage,
        retrieved: list[RetrievalHit],
        *,
        sleep: bool = True,
    ) -> ModelResponse:
        if usage.cache_hit:
            latency = self.base_latency_s
            text = ""  # Pipeline already has the cached answer
        else:
            latency = (
                self.base_latency_s
                + usage.prompt_tokens * self.seconds_per_prefill_token
                + self.output_tokens * self.seconds_per_output_token
            )
            if retrieved:
                text = f"Based on policy: {retrieved[0].chunk.text}"
            else:
                text = "I don't have enough context to answer."
        if sleep:
            time.sleep(latency)
        return ModelResponse(
            latency_s=latency,
            output_tokens=0 if usage.cache_hit else self.output_tokens,
            text=text,
        )
