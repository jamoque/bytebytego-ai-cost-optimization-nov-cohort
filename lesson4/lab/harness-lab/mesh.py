#!/usr/bin/env python3
"""
The mesh client (PROVIDED).

Every model call in this lab goes through the LiteLLM mesh you built in
Lab 1. That now includes three kinds of model behind one gateway:

    chat models        POST /v1/chat/completions     closed (OpenAI, Anthropic)
                                                     and open (a model on your
                                                     machine, served by Ollama)
    decision models    POST /v1/systemone            Jev (hosted) or an open
                                                     decision model on your machine

`Mesh.chat` returns a `Call`. `Mesh.decide` returns a `Decision`. Both carry
what the request cost, read from the mesh's `x-litellm-response-cost` header
(the number Langfuse shows). The prices in config.yaml are used for two
things only: splitting a call's cost into line items for the ledger, and as
a fallback when the mesh has no price for a model.

A local model has no price, so the mesh reports $0 for it. That is not what
it costs. `Mesh.local_cost` charges wall-clock seconds at the hourly rate in
config.yaml (`local.usd_per_hour`): electricity plus the share of the
machine you are using up.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
LAB2_DIR = HERE.parents[2] / "lesson2" / "lab" / "rag-lab"

# --------------------------------------------------------------------------- #
# Lab 2 is a dependency: its index, embedder, and compression are imported,
# not copied. `lab2.embed`, `lab2.compress`, and `lab2.get_collection` below
# are the functions you ran in week 2.
# --------------------------------------------------------------------------- #
if not (LAB2_DIR / "pipeline.py").exists():
    print(f"Lab 2 pipeline not found at {LAB2_DIR}. This lab builds on it.", flush=True)
    sys.exit(1)
# Loaded by file path rather than by adding Lab 2 to sys.path, so Lab 2's
# run_experiment.py and client_test.py can never shadow this lab's files.
_spec = importlib.util.spec_from_file_location("lab2_pipeline", LAB2_DIR / "pipeline.py")
lab2 = importlib.util.module_from_spec(_spec)
sys.modules["lab2_pipeline"] = lab2
_spec.loader.exec_module(lab2)

log = lab2.log
token_count = lab2.token_count


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else HERE / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if os.environ.get("MESH_BASE_URL"):
        cfg["mesh"]["base_url"] = os.environ["MESH_BASE_URL"]
    return cfg


# --------------------------------------------------------------------------- #
# One record per request that reached the mesh.
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One chat completion."""

    purpose: str             # agent | check | judge | handoff
    tier: str                # weak | mid | strong | local
    alias: str               # what the client asked the mesh for
    model: str               # what the mesh says served it
    prompt_tokens: int       # everything the model read, cached or not
    cached_tokens: int       # the part of prompt_tokens read from the provider's prompt cache
    cache_write_tokens: int  # the part of prompt_tokens written to the cache (Anthropic bills these at 1.25x)
    completion_tokens: int   # everything the model wrote, reasoning included
    reasoning_tokens: int    # the part of completion_tokens that was reasoning
    cost_usd: float          # the gateway's meter. $0 for a local model
    local_usd: float         # wall-clock seconds x local.usd_per_hour, for a local model
    latency_s: float
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    thinking_blocks: list[dict] = field(default_factory=list)   # Anthropic wants these sent back mid tool loop

    @property
    def fresh_tokens(self) -> int:
        """Input tokens billed at the full input price."""
        return max(self.prompt_tokens - self.cached_tokens - self.cache_write_tokens, 0)

    @property
    def total_usd(self) -> float:
        return self.cost_usd + self.local_usd


@dataclass
class Decision:
    """One System One request: a state and one or more typed questions."""

    purpose: str
    alias: str
    model: str
    answers: dict[str, dict[str, Any]]   # question name -> the model's typed answer
    input_tokens: int
    cost_usd: float
    local_usd: float
    latency_s: float

    @property
    def total_usd(self) -> float:
        return self.cost_usd + self.local_usd

    def p(self, name: str) -> float:
        """P(yes) for a noul question."""
        return float(self.answers[name]["noul"])

    def choice(self, name: str) -> tuple[str, float]:
        """(chosen option, confidence) for a choice question."""
        answer = self.answers[name]
        return str(answer["choice"]), float(answer.get("confidence", 0.0))

    def score(self, name: str) -> float:
        """Expected level (0 = the first level) for a score question."""
        return float(self.answers[name]["score"])


class StateTooLarge(RuntimeError):
    """The state and questions do not fit the decision model's context window.

    A chat model's runtime may quietly drop the start of a prompt that is too
    long. A decision request is never truncated: it is refused, because an
    answer about half the evidence would look exactly like a real one.
    """


class Mesh:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tiers: dict[str, dict[str, Any]] = config["tiers"]
        self._client = None
        self._http = None

    # ---- plumbing --------------------------------------------------------- #
    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                log("Install dependencies first: pip install -r requirements.txt")
                sys.exit(1)
            mesh = self.config["mesh"]
            self._client = OpenAI(
                base_url=mesh["base_url"], api_key=mesh["api_key"], timeout=float(mesh.get("timeout_s", 600))
            )
        return self._client

    def alias(self, tier: str) -> str:
        if tier not in self.tiers:
            raise ValueError(f"Unknown tier '{tier}'. config.yaml defines: {list(self.tiers)}")
        return self.tiers[tier]["alias"]

    def is_local(self, tier: str) -> bool:
        return bool(self.tiers.get(tier, {}).get("local"))

    def local_cost(self, seconds: float) -> float:
        """What `seconds` of your own hardware costs. See `local:` in config.yaml."""
        return seconds * float(self.config.get("local", {}).get("usd_per_hour", 0.0)) / 3600.0

    def _metadata(self, purpose: str, tags: list[str] | None) -> dict[str, Any]:
        # Flows through the mesh into Langfuse, exactly as in Lab 1.
        return {
            "generation_name": f"lab4-{purpose}",
            "session_id": self.config["mesh"].get("session_id", "lab4"),
            "tags": ["lab4", *(tags or [])],
        }

    # ---- chat models ------------------------------------------------------ #
    def chat(
        self,
        tier: str,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
        tags: list[str] | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        **params: Any,
    ) -> Call:
        """Send one chat completion through the mesh."""
        mesh = self.config["mesh"]
        alias = self.alias(tier)
        kwargs: dict[str, Any] = {
            "model": alias,
            "messages": messages,
            "max_tokens": int(max_tokens or mesh.get("max_tokens", 600)),
            "temperature": float(mesh.get("temperature", 0)),
            "extra_body": {"metadata": self._metadata(purpose, tags)},
            **params,
        }
        if tools:
            kwargs["tools"] = tools
        if params.get("reasoning_effort") not in (None, "none"):
            # Anthropic rejects any temperature other than 1 while thinking is on.
            # Leave it to the provider (and, for the local model, to its Modelfile).
            kwargs.pop("temperature")
        t0 = time.perf_counter()
        try:
            raw = self._openai().chat.completions.with_raw_response.create(**kwargs)
        except Exception as exc:
            log(
                f"Mesh call failed. Is LiteLLM running at {mesh['base_url']} "
                f"with lesson4/lab/harness-lab/mesh_config.yaml?\n"
                f"  Model alias: {alias}\n  Original error: {exc}"
            )
            raise
        latency = time.perf_counter() - t0
        resp = raw.parse()
        message = resp.choices[0].message
        usage = resp.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        # Prompt-cache accounting. The mesh normalizes every provider to the
        # OpenAI usage shape: `cached_tokens` is what was read from the cache,
        # and Anthropic additionally reports what was written to it.
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        written = int(
            getattr(details, "cache_creation_tokens", 0)
            or getattr(usage, "cache_creation_input_tokens", 0)
            or 0
        )
        out_details = getattr(usage, "completion_tokens_details", None)
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)

        local = self.is_local(tier)
        try:
            cost = float(raw.headers.get("x-litellm-response-cost") or 0.0)
        except ValueError:
            cost = 0.0
        if cost <= 0 and not local:
            cost = self.price(tier, prompt_tokens - cached - written, cached, written, completion_tokens)

        return Call(
            purpose=purpose,
            tier=tier,
            alias=alias,
            model=raw.headers.get("x-litellm-model-name") or resp.model or alias,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached,
            cache_write_tokens=written,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            local_usd=self.local_cost(latency) if local else 0.0,
            latency_s=latency,
            text=(message.content or "").strip(),
            tool_calls=[
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
                for tc in (message.tool_calls or [])
            ],
            thinking_blocks=list((message.model_extra or {}).get("thinking_blocks") or []),
        )

    def price(self, tier: str, fresh: int, cached: int, written: int, completion: int) -> float:
        """Dollars for one call from the config.yaml price list."""
        return sum(self.line_items(tier, fresh, cached, written, completion).values())

    def line_items(self, tier: str, fresh: int, cached: int, written: int, completion: int) -> dict[str, float]:
        """Split a call's tokens into the four lines of the ledger, in dollars."""
        spec = self.tiers.get(tier, {})
        inp = float(spec.get("input_price_per_m", 0.0))
        return {
            "input": fresh * inp / 1e6,
            "cache_write": written * float(spec.get("cache_write_per_m", inp)) / 1e6,
            "cache_read": cached * float(spec.get("cache_read_per_m", inp)) / 1e6,
            "output": completion * float(spec.get("output_price_per_m", 0.0)) / 1e6,
        }

    # ---- decision models -------------------------------------------------- #
    def decide(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        purpose: str = "decide",
        alias: str | None = None,
    ) -> Decision:
        """Ask a decision model typed questions about `state`.

        `state` is the content being judged: a string, or a dict / list that
        the model reads as labeled text. Each question is one of

            {"type": "noul",   "instructions": "..."}                         -> P(yes)
            {"type": "choice", "instructions": "...", "criteria": {name: description}}
            {"type": "score",  "instructions": "...", "criteria": [low, ..., high]}

        The model answers every question in one forward pass and writes no
        text: an answer is a probability distribution over the options you
        supplied. Questions in one request share the state but cannot see each
        other's answers.
        """
        try:
            import httpx
        except ImportError:
            log("Install dependencies first: pip install -r requirements.txt")
            sys.exit(1)
        mesh = self.config["mesh"]
        cfg = self.config.get("decision", {})
        alias = alias or cfg.get("alias", "mesh-decide")
        if self._http is None:
            self._http = httpx.Client(timeout=float(mesh.get("timeout_s", 600)))
        url = mesh["base_url"].rstrip("/") + "/systemone"
        t0 = time.perf_counter()
        try:
            resp = self._http.post(
                url,
                headers={"Authorization": f"Bearer {mesh['api_key']}"},
                json={"model": alias, "state": state, "questions": questions},
            )
        except Exception as exc:
            log(f"Decision call failed. Is LiteLLM running at {mesh['base_url']}?\n  Original error: {exc}")
            raise
        latency = time.perf_counter() - t0
        if resp.status_code != 200:
            too_large = re.search(r"request \(\d+ tokens\) exceeds the available context size \(\d+ tokens\)|must not exceed \d+ KiB", resp.text)
            if too_large or resp.status_code == 413:
                raise StateTooLarge(too_large.group(0) if too_large else resp.text[:200])
            raise RuntimeError(
                f"Decision model '{alias}' returned HTTP {resp.status_code}: {resp.text[:400]}\n"
                "Check that the alias exists in mesh_config.yaml and that its model is pulled "
                "(local) or its API key is exported (hosted). 'Unknown Decisions provider' means "
                "the mesh's LiteLLM is too old (this lab was verified on 1.110.2): "
                "pip install -U 'litellm[proxy]' in the Lab 1 venv and restart it."
            )
        body = resp.json()
        try:
            cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
        except ValueError:
            cost = 0.0
        local = bool(cfg.get("local", True)) if alias == cfg.get("alias") else cost <= 0
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        if cost <= 0 and not local:
            cost = input_tokens * float(cfg.get("input_price_per_m", 0.0)) / 1e6
        return Decision(
            purpose=purpose,
            alias=alias,
            model=str(body.get("model") or alias),
            answers=body.get("answers") or {},
            input_tokens=input_tokens,
            cost_usd=cost,
            local_usd=self.local_cost(latency) if local else 0.0,
            latency_s=latency,
        )


__all__ = ["HERE", "LAB2_DIR", "lab2", "log", "token_count", "load_config", "Call", "Decision", "StateTooLarge", "Mesh"]
