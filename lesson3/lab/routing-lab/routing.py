#!/usr/bin/env python3
"""
Northstar model router (PROVIDED).

This is the working system you observe in Parts 1 and 4. You configure it
through `config.yaml`; you do not have to implement it. The from-scratch
advanced exercise, `../mf-router-lab/`, trains your own preference router.

Request path:

    user question
      │
      ▼
    retrieve top-k chunks          # Lab 2's index and embedder, unchanged
      │
      ▼
    ROUTER: which rung of the price ladder answers this?     # new in Lab 3
      │
      ├─ weak | mid | strong   a fixed rung, no decision (the baselines)
      ├─ gateway               ask for `mesh-auto`; the mesh scores the prompt with rules
      ├─ deterministic         nearest route by embedding similarity; the route names a rung
      ├─ cascade               weak answers first; a verifier gate decides whether to pay for more
      └─ preference            RouteLLM predicts P(strong wins); above a threshold → strong
      │
      ▼
    LLM Gateway (Lab 1 LiteLLM mesh) ──► provider

Every model call goes through the mesh, so every call is traced, tagged,
and priced in one place. This file never talks to a provider directly.
Cost is read from the mesh's `x-litellm-response-cost` header (the number
Langfuse shows); the prices in config.yaml are only a fallback.

Three of the five strategies decide BEFORE any model is called (gateway,
deterministic, preference). The cascade decides AFTER the weak model has
answered, which is why it is the only one that can pay twice.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
LAB2_DIR = HERE.parents[2] / "lesson2" / "lab" / "rag-lab"

# --------------------------------------------------------------------------- #
# Lab 2 is a dependency: its index, embedder, compression, and compaction
# are imported, not copied. `lab2.embed`, `lab2.compress`, `lab2.compact`,
# and `lab2.Message` below are the functions you ran last week.
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
Hit = lab2.Hit
token_count = lab2.token_count

SUPPORT_SYSTEM = (
    "You are a Northstar Commerce support agent. Answer using only the provided "
    "context. If the context is insufficient, say you don't have that information. "
    "Be concise: at most four sentences of reasoning, then finish with one line "
    'that starts with "Answer:" and states the final amount, date, time, or yes/no.'
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else HERE / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    if os.environ.get("MESH_BASE_URL"):
        cfg["mesh"]["base_url"] = os.environ["MESH_BASE_URL"]
    return cfg


@lru_cache(maxsize=1)
def mesh_aliases() -> dict[str, str]:
    """provider model id → mesh alias, read from mesh_config.yaml.

    The gateway reports which provider model served a request. This maps
    that back to an alias so a `mesh-auto` call can be attributed to a tier.
    """
    mesh = yaml.safe_load((HERE / "mesh_config.yaml").read_text())
    out: dict[str, str] = {}
    for entry in mesh.get("model_list", []):
        model = entry.get("litellm_params", {}).get("model", "")
        if model and not model.startswith("auto_router/"):
            out[model] = entry["model_name"]
    return out


# --------------------------------------------------------------------------- #
# Retrieval (Lab 2's Chroma index + MiniLM embedder, plus a source filter
# that Part 3's specialist agents use to search only their own documents).
# --------------------------------------------------------------------------- #
def retrieve(
    query: str,
    *,
    k: int,
    min_score: float = 0.0,
    sources: list[str] | None = None,
) -> list[Hit]:
    collection = lab2.get_collection()
    if collection.count() == 0:
        log(f"Vector DB is empty. Run:  (cd {LAB2_DIR} && python index_kb.py)")
        return []
    kwargs: dict[str, Any] = {
        "query_embeddings": [lab2.embed([query])[0]],
        "n_results": min(int(k), collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if sources:
        kwargs["where"] = {"source": {"$in": list(sources)}}
    result = collection.query(**kwargs)
    hits: list[Hit] = []
    for _id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        score = 1.0 - dist
        if score < min_score:
            continue
        hits.append(Hit(id=_id, source=(meta or {}).get("source", ""), score=score, text=doc or ""))
    return hits


# --------------------------------------------------------------------------- #
# The mesh client. One method, one Call record per request.
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One request that reached the mesh."""

    purpose: str             # answer | gate | plan | synthesize | memory | agent-<name>
    tier: str                # weak | mid | strong (resolved from the served model for mesh-auto)
    alias: str               # what the client asked the mesh for
    model: str               # what the mesh says actually served it
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)  # x-litellm-* response headers
    top_logprobs: list[tuple[str, float]] = field(default_factory=list)  # first token, if requested


class Mesh:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tiers: dict[str, dict[str, Any]] = config["tiers"]
        self._client = None

    def _openai(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                log("Install dependencies first: pip install -r requirements.txt")
                sys.exit(1)
            mesh = self.config["mesh"]
            self._client = OpenAI(base_url=mesh["base_url"], api_key=mesh["api_key"])
        return self._client

    def alias(self, tier: str) -> str:
        if tier not in self.tiers:
            raise ValueError(f"Unknown tier '{tier}'. config.yaml defines: {list(self.tiers)}")
        return self.tiers[tier]["alias"]

    def tier_of(self, served_model: str) -> str:
        """Which tier a provider model id (as reported by the mesh) belongs to."""
        alias = mesh_aliases().get(served_model, served_model)
        for tier, spec in self.tiers.items():
            if spec["alias"] == alias:
                return tier
        return "?"

    def _fallback_cost(self, tier: str, prompt_tokens: int, completion_tokens: int) -> float:
        spec = self.tiers.get(tier)
        if not spec:
            return 0.0
        return (
            prompt_tokens * float(spec["input_price_per_m"]) / 1_000_000
            + completion_tokens * float(spec["output_price_per_m"]) / 1_000_000
        )

    def chat(
        self,
        tier: str,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
        tags: list[str] | None = None,
        alias: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        **params: Any,
    ) -> Call:
        """Send one chat completion through the mesh.

        `tier` picks the alias from config.yaml. Pass `alias` to override it
        (the gateway strategy asks for `mesh-auto` and lets the mesh choose).
        """
        mesh = self.config["mesh"]
        model_alias = alias or self.alias(tier)
        session = mesh.get("session_id", "lab3")
        kwargs: dict[str, Any] = {
            "model": model_alias,
            "messages": messages,
            "max_tokens": int(max_tokens or mesh.get("max_tokens", 300)),
            "temperature": float(mesh.get("temperature", 0)),
            # Metadata flows through the mesh into Langfuse, exactly as in Lab 1.
            "extra_body": {
                "metadata": {
                    "generation_name": f"lab3-{purpose}",
                    "session_id": session,
                    "tags": ["lab3", *(tags or [])],
                }
            },
            **params,
        }
        if tools:
            kwargs["tools"] = tools
        t0 = time.perf_counter()
        try:
            raw = self._openai().chat.completions.with_raw_response.create(**kwargs)
        except Exception as exc:
            log(
                f"Mesh call failed. Is LiteLLM running at {mesh['base_url']} "
                f"with lesson3/lab/routing-lab/mesh_config.yaml?\n"
                f"  Model alias: {model_alias}\n  Original error: {exc}"
            )
            raise
        latency = time.perf_counter() - t0
        resp = raw.parse()
        headers = {
            k.lower(): v for k, v in raw.headers.items() if k.lower().startswith("x-litellm")
        }
        message = resp.choices[0].message
        usage = resp.usage
        prompt_tokens = (usage.prompt_tokens or 0) if usage else 0
        completion_tokens = (usage.completion_tokens or 0) if usage else 0

        served = headers.get("x-litellm-model-name") or resp.model or model_alias
        resolved_tier = self.tier_of(served) if alias else tier

        # The mesh is the meter. Fall back to config prices only if it is silent.
        try:
            cost = float(headers.get("x-litellm-response-cost", ""))
        except ValueError:
            cost = 0.0
        if cost <= 0:
            cost = self._fallback_cost(resolved_tier, prompt_tokens, completion_tokens)

        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
            for tc in (message.tool_calls or [])
        ]
        first_token: list[tuple[str, float]] = []
        logprobs = getattr(resp.choices[0], "logprobs", None)
        if logprobs and getattr(logprobs, "content", None):
            first_token = [(t.token, t.logprob) for t in (logprobs.content[0].top_logprobs or [])]
        return Call(
            purpose=purpose,
            tier=resolved_tier,
            alias=model_alias,
            model=served,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_s=latency,
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            headers=headers,
            top_logprobs=first_token,
        )


# --------------------------------------------------------------------------- #
# Strategy 1b: deterministic vector routing.
#
# The same nearest-neighbor search as Lab 2's semantic cache. There the
# nearest stored query returned an ANSWER. Here the nearest stored utterance
# returns a ROUTE, and the route names a tier. No model is called to decide.
# --------------------------------------------------------------------------- #
@dataclass
class RouteDecision:
    route: str | None                     # None = nothing cleared min_score
    tier: str
    score: float                          # similarity of the winning route
    scores: list[tuple[str, float]]       # every route, best first


class VectorRouter:
    def __init__(self, routes: list[dict[str, Any]], *, min_score: float, default_tier: str) -> None:
        self.routes = {r["name"]: r for r in routes}
        self.min_score = min_score
        self.default_tier = default_tier
        owners: list[str] = []
        texts: list[str] = []
        for r in routes:
            for utterance in r["utterances"]:
                owners.append(r["name"])
                texts.append(utterance)
        self._index = list(zip(owners, lab2.embed(texts)))

    def scores(self, text: str) -> list[tuple[str, float]]:
        """Each route's score is its single closest utterance (vectors are L2-normalized)."""
        query = lab2.embed([text])[0]
        best: dict[str, float] = {name: -1.0 for name in self.routes}
        for name, vec in self._index:
            sim = sum(a * b for a, b in zip(query, vec))
            if sim > best[name]:
                best[name] = sim
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

    def route(self, text: str) -> RouteDecision:
        scores = self.scores(text)
        name, score = scores[0]
        if score < self.min_score:
            return RouteDecision(route=None, tier=self.default_tier, score=score, scores=scores)
        return RouteDecision(route=name, tier=self.routes[name]["tier"], score=score, scores=scores)


def load_routes(config: dict[str, Any]) -> list[dict[str, Any]]:
    import json

    path = HERE / config["deterministic"]["routes"]
    return json.loads(path.read_text())["routes"]


# --------------------------------------------------------------------------- #
# Strategy 2: the cascade's gate.
#
# A cascade is only as good as the function that decides "is this answer
# good enough to keep?". FrugalGPT trains a small scoring model for that.
# This lab uses the version most teams ship first: a second, cheap model call
# that re-derives the answer and says YES or NO. The gate's score is the
# probability the verifier puts on YES, read from the token logprobs, so the
# threshold is a dial rather than a switch.
#
#   abstention ("I don't have that information")  → 0.0, no verifier call
#   otherwise                                      → P(YES) from the verifier
#
# The gate is not free and not perfect. It adds one cheap call to EVERY
# question, it raises false alarms on correct answers, and it waves through
# some wrong ones. Part 1.4 has you measure all three.
# --------------------------------------------------------------------------- #
_ABSTAIN_RE = re.compile(
    r"don'?t have (that|this|the|enough) information|do not have (that|this|the|enough) information"
    r"|not (provided|specified|mentioned|stated) in the context|cannot (determine|answer)|can'?t determine"
    r"|insufficient (context|information)",
    re.I,
)
_ANSWER_LINE_RE = re.compile(r"answer[\s*_]*:[\s*_]*([^\n]+)", re.I)

VERIFY_PROMPT = """You are a strict reviewer of a support agent's draft answer.

Policy context:
{context}

Customer question: {question}

Draft answer:
{answer}

Re-derive the answer yourself from the policy context, checking every condition, threshold, date and calculation. Is the draft's final answer correct? Reply with exactly one word: YES or NO."""


def final_line(answer: str) -> str:
    """The text after the last 'Answer:' marker, or the whole answer if there is none."""
    found = _ANSWER_LINE_RE.findall(answer)
    return found[-1].strip() if found else answer


def abstained(answer: str) -> bool:
    return not answer.strip() or bool(_ABSTAIN_RE.search(answer))


def yes_probability(text: str, top_logprobs: list[tuple[str, float]] | None) -> float:
    """Turn a YES/NO verdict into a confidence in [0, 1].

    With logprobs: P(YES) / (P(YES) + P(NO)) over the first token's candidates.
    Without (some providers do not return them): 1.0 for YES, 0.0 for anything else.
    """
    if top_logprobs:
        p_yes = sum(math.exp(lp) for tok, lp in top_logprobs if tok.strip().upper().startswith("YES"))
        p_no = sum(math.exp(lp) for tok, lp in top_logprobs if tok.strip().upper().startswith("NO"))
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no)
    return 1.0 if text.strip().upper().startswith("YES") else 0.0


# --------------------------------------------------------------------------- #
# Strategy 3: preference-based routing with RouteLLM.
#
# RouteLLM's `mf` router is a matrix-factorization model trained on tens of
# thousands of human preference votes between pairs of models (Chatbot
# Arena), augmented with GPT-4-judged pairs. Given a prompt it predicts the probability that
# the strong model's answer would be preferred over the weak model's. You
# pick a threshold; prompts at or above it go to the strong model.
#
# It scores the QUESTION only. It has never seen Northstar's policies and it
# does not read the retrieved context.
#
# `preference.checkpoint` swaps in different weights for the same router
# class: a directory holding a `model.safetensors` in RouteLLM's format. The
# advanced exercise (`../mf-router-lab/`) trains one on Northstar battles.
# --------------------------------------------------------------------------- #
PUBLISHED_MF_CHECKPOINT = "routellm/mf_gpt4_augmented"


class PreferenceScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._router = None
        self._cache: dict[str, float] = {}

    def _load(self):
        if self._router is not None:
            return self._router
        mesh = self.config["mesh"]
        pref = self.config["preference"]
        if pref.get("router", "mf") != "mf":
            raise ValueError("This lab wires RouteLLM's 'mf' router. Set preference.router: mf")
        checkpoint = self.checkpoint()
        # RouteLLM builds an OpenAI() client at import time, so a key must exist.
        os.environ.setdefault("OPENAI_API_KEY", mesh["api_key"])
        if checkpoint == PUBLISHED_MF_CHECKPOINT:
            log("Loading RouteLLM (first run downloads the mf checkpoint from Hugging Face)...")
        else:
            log(f"Loading RouteLLM with your checkpoint: {checkpoint}")
        try:
            from openai import OpenAI
            from routellm.controller import Controller
            import routellm.routers.matrix_factorization.model as mf_model
        except ImportError:
            log("RouteLLM is not installed. Run: pip install -r requirements.txt")
            sys.exit(1)
        controller = Controller(
            routers=["mf"],
            strong_model=self.config["tiers"]["strong"]["alias"],
            weak_model=self.config["tiers"]["weak"]["alias"],
            config={"mf": {"checkpoint_path": checkpoint}},
        )
        router = controller.routers["mf"]
        # Out of the box RouteLLM calls api.openai.com for its prompt embedding,
        # bypassing the mesh. Point it at the mesh's embedding alias instead so
        # the router's own spend is traced and keyed like every other request.
        mf_model.OPENAI_CLIENT = OpenAI(base_url=mesh["base_url"], api_key=mesh["api_key"])
        router.model.embedding_model = pref.get("embedding_alias", "mesh-embed")
        self._router = router
        return router

    def checkpoint(self) -> str:
        """RouteLLM's published weights, or the directory named by preference.checkpoint."""
        custom = self.config["preference"].get("checkpoint")
        if not custom:
            return PUBLISHED_MF_CHECKPOINT
        path = Path(custom).expanduser()
        if not path.is_absolute():
            path = HERE / path            # relative to this lab, like the other paths in config.yaml
        if not (path / "model.safetensors").exists():
            log(
                f"preference.checkpoint points at {path}, which has no model.safetensors.\n"
                "Train one first (cd ../mf-router-lab && python train.py), or set checkpoint: null."
            )
            sys.exit(1)
        return str(path.resolve())

    @property
    def is_custom(self) -> bool:
        return self.checkpoint() != PUBLISHED_MF_CHECKPOINT

    def win_rate(self, prompt: str) -> float:
        """P(strong model's answer is preferred) for this prompt, in [0, 1]."""
        if prompt not in self._cache:
            self._cache[prompt] = float(self._load().calculate_strong_win_rate(prompt))
        return self._cache[prompt]


def calibrate_threshold(win_rates: list[float], strong_fraction: float) -> float:
    """Threshold that sends about `strong_fraction` of these prompts to strong.

    Calibration needs no labels: only a sample of real traffic and a budget.
    """
    if not win_rates:
        raise ValueError("Need at least one win rate to calibrate")
    ranked = sorted(win_rates, reverse=True)
    n_strong = round(strong_fraction * len(ranked))
    if n_strong <= 0:
        return min(1.0, ranked[0] + 1e-6)
    return ranked[min(n_strong, len(ranked)) - 1]


# --------------------------------------------------------------------------- #
# Evals. Routing without a quality number is just a discount.
# --------------------------------------------------------------------------- #
def grade(answer: str, expect: list[str], reject: list[str] | None = None) -> bool:
    """True if the final answer line matches every `expect` regex and no `reject` regex."""
    text = final_line(answer)
    if not all(re.search(p, text, re.I) for p in expect):
        return False
    return not any(re.search(p, text, re.I) for p in (reject or []))


# --------------------------------------------------------------------------- #
# Public router.
# --------------------------------------------------------------------------- #
@dataclass
class RouteResult:
    question: str
    strategy: str
    answer: str
    tier: str                    # the tier that produced the final answer
    reason: str                  # one line: why the router chose it
    calls: list[Call]            # every model call this question cost (a cascade can have several)
    hits: list[Hit]
    score: float | None = None   # route similarity, gate score, or win rate, depending on strategy

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def llm_calls(self) -> int:
        return len(self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def model(self) -> str:
        answers = [c for c in self.calls if c.purpose == "answer"]
        return answers[-1].model if answers else ""


class Router:
    STRATEGIES = ("gateway", "deterministic", "cascade", "preference")

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mesh = Mesh(config)
        self._vector: VectorRouter | None = None
        self.preference = PreferenceScorer(config)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Router":
        return cls(load_config(path))

    # ---- shared pieces --------------------------------------------------- #
    @property
    def vector_router(self) -> VectorRouter:
        if self._vector is None:
            det = self.config["deterministic"]
            self._vector = VectorRouter(
                load_routes(self.config),
                min_score=float(det.get("min_score", 0.3)),
                default_tier=det.get("default_tier", "strong"),
            )
        return self._vector

    def retrieve(self, question: str, *, k: int | None = None, sources: list[str] | None = None) -> list[Hit]:
        rag = self.config["rag"]
        return retrieve(
            question,
            k=int(k or rag.get("k", 5)),
            min_score=float(rag.get("min_score", 0.0)),
            sources=sources,
        )

    def context_for(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(h.text for h in hits)
        cap = self.config["rag"].get("compress_max_tokens")
        if cap:
            context = lab2.compress(context, question, int(cap))  # Lab 2, unchanged
        return context

    def answer(
        self,
        tier: str,
        question: str,
        context: str,
        *,
        tags: list[str] | None = None,
        alias: str | None = None,
    ) -> Call:
        """One grounded answer from one rung of the ladder."""
        messages = [
            {"role": "system", "content": f"{SUPPORT_SYSTEM}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ]
        return self.mesh.chat(
            tier, messages, purpose="answer", alias=alias, tags=[f"tier-{tier}", *(tags or [])]
        )

    def gate(self, question: str, answer: str, context: str, *, tags: list[str] | None = None) -> tuple[float, Call | None]:
        """Score a draft answer in [0, 1]. Returns (score, the verifier call or None)."""
        if abstained(answer):
            return 0.0, None  # the model already told us; no need to pay a verifier
        tier = self.config["cascade"].get("gate_tier", "weak")
        prompt = VERIFY_PROMPT.format(context=context, question=question, answer=answer)
        call = self.mesh.chat(
            tier,
            [{"role": "user", "content": prompt}],
            purpose="gate",
            tags=tags,
            max_tokens=3,
            logprobs=True,
            top_logprobs=5,
        )
        return yes_probability(call.text, call.top_logprobs), call

    # ---- the one entry point --------------------------------------------- #
    def ask(self, question: str, *, strategy: str | None = None, tags: list[str] | None = None) -> RouteResult:
        strategy = strategy or self.config["router"]["strategy"]
        tags = [f"strategy-{strategy}", *(tags or [])]
        hits = self.retrieve(question)
        context = self.context_for(question, hits)

        if strategy in self.mesh.tiers:
            call = self.answer(strategy, question, context, tags=tags)
            return RouteResult(question, strategy, call.text, strategy, "fixed tier", [call], hits)
        if strategy == "gateway":
            return self._gateway(question, context, hits, tags)
        if strategy == "deterministic":
            return self._deterministic(question, context, hits, tags)
        if strategy == "cascade":
            return self._cascade(question, context, hits, tags)
        if strategy == "preference":
            return self._preference(question, context, hits, tags)
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use a tier ({list(self.mesh.tiers)}) or one of {self.STRATEGIES}"
        )

    # ---- strategy 1a: rules, decided inside the gateway ------------------- #
    def _gateway(self, question: str, context: str, hits: list[Hit], tags: list[str]) -> RouteResult:
        """Ask for the router alias. The mesh picks the model; we read which one from the headers."""
        alias = self.config["gateway"]["alias"]
        call = self.answer("auto", question, context, tags=tags, alias=alias)
        rule_tier = call.headers.get("x-litellm-complexity-router-tier", "?")
        try:
            score: float | None = float(call.headers.get("x-litellm-complexity-router-score", ""))
        except ValueError:
            score = None
        shown = f"{score:+.2f}" if score is not None else "n/a"
        reason = f"gateway rules: {rule_tier} (score {shown}) → {call.model}"
        return RouteResult(question, "gateway", call.text, call.tier, reason, [call], hits, score)

    # ---- strategy 1b: vector routes, decided in the client ---------------- #
    def _deterministic(self, question: str, context: str, hits: list[Hit], tags: list[str]) -> RouteResult:
        decision = self.vector_router.route(question)
        call = self.answer(decision.tier, question, context, tags=[*tags, f"route-{decision.route or 'default'}"])
        if decision.route is None:
            reason = f"no route ≥ {self.vector_router.min_score:.2f} (best {decision.score:.2f}) → default {decision.tier}"
        else:
            reason = f"route {decision.route} ({decision.score:.2f}) → {decision.tier}"
        return RouteResult(question, "deterministic", call.text, decision.tier, reason, [call], hits, decision.score)

    # ---- strategy 2: cascade ---------------------------------------------- #
    def _cascade(self, question: str, context: str, hits: list[Hit], tags: list[str]) -> RouteResult:
        cfg = self.config["cascade"]
        ladder: list[str] = list(cfg["ladder"])
        threshold = float(cfg["threshold"])
        calls: list[Call] = []
        trail: list[str] = []
        score = 0.0
        for i, tier in enumerate(ladder):
            call = self.answer(tier, question, context, tags=[*tags, f"rung-{i}"])
            calls.append(call)
            if i == len(ladder) - 1:
                trail.append(f"{tier} (top rung, accepted)")
                break
            score, verdict = self.gate(question, call.text, context, tags=[*tags, f"rung-{i}"])
            if verdict is not None:
                calls.append(verdict)
            if score >= threshold:
                trail.append(f"{tier} gate {score:.2f} ≥ {threshold:.2f} accepted")
                break
            trail.append(f"{tier} gate {score:.2f} < {threshold:.2f}")
        final = [c for c in calls if c.purpose == "answer"][-1]
        return RouteResult(question, "cascade", final.text, final.tier, " → ".join(trail), calls, hits, score)

    # ---- strategy 3: RouteLLM --------------------------------------------- #
    def _preference(self, question: str, context: str, hits: list[Hit], tags: list[str]) -> RouteResult:
        threshold = float(self.config["preference"]["threshold"])
        win = self.preference.win_rate(question)
        tier = "strong" if win >= threshold else "weak"
        call = self.answer(tier, question, context, tags=tags)
        op = "≥" if win >= threshold else "<"
        reason = f"P(strong wins) {win:.3f} {op} {threshold:.3f} → {tier}"
        return RouteResult(question, "preference", call.text, tier, reason, [call], hits, win)


__all__ = [
    "Router",
    "RouteResult",
    "RouteDecision",
    "VectorRouter",
    "PreferenceScorer",
    "Mesh",
    "Call",
    "Hit",
    "retrieve",
    "abstained",
    "yes_probability",
    "VERIFY_PROMPT",
    "final_line",
    "grade",
    "calibrate_threshold",
    "load_config",
    "load_routes",
    "token_count",
    "log",
    "lab2",
    "SUPPORT_SYSTEM",
]
