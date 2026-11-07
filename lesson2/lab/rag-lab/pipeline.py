#!/usr/bin/env python3
"""
Northstar RAG pipeline (PROVIDED).

This is the working system you observe in Parts 1–4. You configure it through
`config.yaml`; you do not have to implement it. The from-scratch rebuild of
the same ideas is `../context-os-lab/`.

Request path (matches the lecture diagram):

    user query
      │
      ▼
    assemble workflow messages (RAG → compress → compact)   # this process
      │
      ▼
    LLM Gateway (Lab 1 LiteLLM mesh)
      │
      ├─ embed messages → Qdrant similarity search
      │     │
      │     ├─ HIT  (similarity ≥ threshold) ──► response cache ──► return
      │     │
      │     └─ MISS ──► provider (OpenAI / Anthropic)
      │
      ▼
    return answer + x-litellm-semantic-similarity header

Every query enters the gateway. Hit vs miss is decided *inside* the mesh,
not in this file. This client only opts out per-request with
`cache.no-cache` and `cache.no-store` when `semantic_cache.enabled` is
false. `no-store` matters: `no-cache` alone still writes Qdrant, and the
next cached run would hit those bypass answers.

LiteLLM stores a miss in a background task after the HTTP response is
sent. This client waits until that point is visible in Qdrant before
returning a miss, so the next question in a loop can actually hit.

Knowledge-base embeddings for RAG are still local (`all-MiniLM-L6-v2` +
Chroma). Gateway semantic-cache embeddings are `mesh-embed`
(text-embedding-3-small) as configured in mesh_config.yaml.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
KB_DIR = HERE / "knowledge_base"
DB_DIR = HERE / ".chroma"
COLLECTION_NAME = "northstar-kb"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INPUT_PRICE_PER_M = 0.15   # gpt-4o-mini; overridden by mesh.input_price_per_m
DEFAULT_OUTPUT_PRICE_PER_M = 0.60

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def split_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def token_count(text: str) -> int:
    return len(split_tokens(text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    llm_calls: int,
    *,
    input_price_per_m: float = DEFAULT_INPUT_PRICE_PER_M,
    output_price_per_m: float = DEFAULT_OUTPUT_PRICE_PER_M,
) -> float:
    if llm_calls <= 0:
        return 0.0
    return (
        prompt_tokens * (input_price_per_m / 1_000_000)
        + completion_tokens * (output_price_per_m / 1_000_000)
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else HERE / "config.yaml"
    return yaml.safe_load(cfg_path.read_text())


def gateway_cache_params() -> dict[str, Any]:
    mesh = yaml.safe_load((HERE / "mesh_config.yaml").read_text())
    return mesh.get("litellm_settings", {}).get("cache_params", {})


def gateway_similarity_threshold() -> float:
    """Hit/miss cutoff. Lives in the gateway config, not config.yaml."""
    return float(gateway_cache_params().get("similarity_threshold", 0.8))


def _qdrant_base_url() -> str:
    return os.environ.get("QDRANT_API_BASE", "http://localhost:6333").rstrip("/")


def _qdrant_point_count() -> int | None:
    params = gateway_cache_params()
    name = params.get("qdrant_collection_name", "lab2_semantic_cache")
    url = f"{_qdrant_base_url()}/collections/{name}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            payload = json.load(resp)
        return int(payload["result"]["points_count"])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _wait_for_cache_store(before: int | None, timeout: float = 8.0) -> None:
    """Wait until LiteLLM's background cache write is readable.

    The proxy returns the completion, then embeds and upserts in another
    task. A paraphrase sent immediately will search a stale index and miss.
    Waiting here does not decide hit vs miss; the gateway already did that.
    Matching on prompt text is not enough: another end user can already
    have stored the same string under a different cache key.
    """
    if before is None:
        before = _qdrant_point_count()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        now = _qdrant_point_count()
        if before is not None and now is not None and now > before:
            return
        time.sleep(0.1)
    log(
        "Timed out waiting for the gateway to store this response in Qdrant. "
        "The next paraphrase may miss."
    )


def load_documents() -> list[dict[str, str]]:
    docs = []
    for p in sorted(KB_DIR.glob("*.txt")):
        docs.append({"source": p.name, "text": p.read_text()})
    return docs


def chunk_document(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Slice `text` into overlapping token windows, preserving original wording."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("overlap must be >= 0 and < chunk_tokens")
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    if not spans:
        return []
    step = chunk_tokens - overlap
    chunks: list[str] = []
    i = 0
    while i < len(spans):
        j = min(i + chunk_tokens, len(spans))
        chunk = text[spans[i][0] : spans[j - 1][1]].strip()
        if chunk:
            chunks.append(chunk)
        if j >= len(spans):
            break
        i += step
    return chunks


def corpus_token_count() -> int:
    return sum(token_count(d["text"]) for d in load_documents())


# --------------------------------------------------------------------------- #
# Lazy singletons (model load is slow; defer until first use).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("Install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    log(f"Loading embedding model '{MODEL_NAME}' (first run downloads ~90 MB)...")
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_client():
    try:
        import chromadb
    except ImportError:
        log("Install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------- #
# Extractive compression + conversation compaction (same contracts as the
# from-scratch lab, so the two halves of the lab tell one story).
# --------------------------------------------------------------------------- #
def compress(text: str, query: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if token_count(text) <= max_tokens:
        return text
    sentences = split_sentences(text)
    if not sentences:
        return ""
    query_set = {t.lower() for t in split_tokens(query)}

    def score(sentence: str) -> int:
        sent_set = {t.lower() for t in split_tokens(sentence)}
        return sum(1 for t in sent_set if t in query_set)

    ranked = sorted(enumerate(sentences), key=lambda iv: (-score(iv[1]), iv[0]))
    kept: set[int] = set()
    for i, _s in ranked:
        candidate = " ".join(sentences[j] for j in sorted(kept | {i}))
        if token_count(candidate) <= max_tokens:
            kept.add(i)
    if not kept:
        return " ".join(split_tokens(ranked[0][1])[:max_tokens])
    return " ".join(sentences[j] for j in sorted(kept))


@dataclass
class Message:
    role: str
    content: str


def compact(messages: list[Message], budget: int) -> list[Message]:
    if budget <= 0:
        return []
    if not messages:
        return []
    total = sum(token_count(m.content) for m in messages)
    if total <= budget:
        return list(messages)

    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    used = sum(token_count(m.content) for m in system)

    suffix: list[Message] = []
    for m in reversed(rest):
        t = token_count(m.content)
        if used + t <= budget:
            suffix.append(m)
            used += t
        else:
            break
    suffix.reverse()
    dropped = len(rest) - len(suffix)
    if dropped == 0:
        return system + suffix

    marker = Message("system", f"[compacted: {dropped} earlier turns]")
    mt = token_count(marker.content)
    while suffix and used + mt > budget:
        dropped_m = suffix.pop(0)
        used -= token_count(dropped_m.content)
        dropped += 1
        marker = Message("system", f"[compacted: {dropped} earlier turns]")
        mt = token_count(marker.content)
    if used + mt <= budget:
        return system + [marker] + suffix
    return system + suffix


# --------------------------------------------------------------------------- #
# Public pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    id: str
    source: str
    score: float
    text: str

    @property
    def tokens(self) -> int:
        return token_count(self.text)


@dataclass
class AskResult:
    answer: str
    cache_hit: bool
    hits: list[Hit]
    retrieved_tokens: int
    compressed_tokens: int
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int
    estimated_cost_usd: float
    semantic_similarity: float | None = None
    prompt_messages: list[Message] = field(default_factory=list)


class Pipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.collection = get_collection()
        self._mesh_client = None

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Pipeline":
        return cls(load_config(path))

    def _mesh(self):
        if self._mesh_client is None:
            try:
                from openai import OpenAI
            except ImportError:
                log("Install the OpenAI SDK first: pip install openai")
                sys.exit(1)
            mesh = self.config["mesh"]
            self._mesh_client = OpenAI(
                base_url=mesh["base_url"],
                api_key=mesh["api_key"],
            )
        return self._mesh_client

    def _complete(
        self,
        messages: list[Message],
        *,
        tags: list[str],
        use_gateway_cache: bool,
    ) -> tuple[str, int, int, bool, float | None]:
        """Send `messages` through the Lab 1 mesh.

        The gateway embeds the messages and decides hit vs miss. This method
        never short-circuits locally. Returns
        (answer, prompt_tokens, completion_tokens, cache_hit, similarity).
        """
        mesh = self.config["mesh"]
        payload = [{"role": m.role, "content": m.content} for m in messages]
        extra_body: dict[str, Any] = {
            "metadata": {
                "generation_name": "lab2-ask",
                "session_id": mesh.get("session_id", "lab2"),
                "tags": tags,
            },
            "user": mesh.get("session_id", "lab2"),
        }
        points_before = _qdrant_point_count() if use_gateway_cache else None
        if not use_gateway_cache:
            # no-cache skips the read. no-store skips the write. Both are
            # required: a bypass run that still writes makes the next
            # cached question a hit of the bypass answer.
            extra_body["cache"] = {"no-cache": True, "no-store": True}
        try:
            raw = self._mesh().chat.completions.with_raw_response.create(
                model=mesh["model"],
                messages=payload,
                max_tokens=int(mesh.get("max_tokens", 150)),
                temperature=float(mesh.get("temperature", 0)),
                extra_body=extra_body,
            )
        except Exception as exc:
            log(
                f"Mesh call failed. Is LiteLLM running at {mesh['base_url']} "
                f"with lesson2/lab/rag-lab/mesh_config.yaml?\n"
                f"  Original error: {exc}"
            )
            raise
        resp = raw.parse()
        headers = {k.lower(): v for k, v in raw.headers.items()}
        sim_header = headers.get("x-litellm-semantic-similarity")
        similarity: float | None = None
        if sim_header not in (None, ""):
            try:
                similarity = float(sim_header)
            except ValueError:
                similarity = None
        # The gateway stamps x-litellm-semantic-similarity on misses too
        # (the nearest neighbor, often well above 0). A hit is that score
        # at or above the gateway threshold — the same comparison LiteLLM
        # makes. There is no cache-hit header in current LiteLLM; any
        # positive similarity is not a hit.
        hit_header = headers.get("x-litellm-cache-hit", "").lower()
        if similarity is not None:
            cache_hit = similarity >= gateway_similarity_threshold()
        else:
            cache_hit = hit_header in ("true", "1", "yes")
        if use_gateway_cache and not cache_hit:
            _wait_for_cache_store(points_before)
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        if usage is not None:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
        else:
            prompt_tokens = sum(token_count(m.content) for m in messages)
            completion_tokens = token_count(text)
        return text, prompt_tokens, completion_tokens, cache_hit, similarity

    def _prices(self) -> tuple[float, float]:
        mesh = self.config.get("mesh", {})
        return (
            float(mesh.get("input_price_per_m", DEFAULT_INPUT_PRICE_PER_M)),
            float(mesh.get("output_price_per_m", DEFAULT_OUTPUT_PRICE_PER_M)),
        )

    def reset_cache(self) -> None:
        """No-op. Semantic cache state lives in Qdrant, inside the gateway."""
        return None

    def corpus_token_count(self) -> int:
        return corpus_token_count()

    def indexed_chunk_count(self) -> int:
        return self.collection.count()

    def retrieve(self, query: str, *, stuff: bool | None = None) -> list[Hit]:
        retriever_cfg = self.config["retriever"]
        if not retriever_cfg.get("enabled", True):
            return []
        use_stuff = retriever_cfg.get("stuff", False) if stuff is None else stuff
        if use_stuff:
            # Stuff the original documents, not the overlapping chunks — otherwise
            # the baseline is inflated by overlap and the comparison is dishonest.
            return [
                Hit(id=d["source"], source=d["source"], score=1.0, text=d["text"])
                for d in load_documents()
            ]
        if self.collection.count() == 0:
            log("Vector DB is empty. Run:  python index_kb.py")
            return []

        k = min(int(retriever_cfg["k"]), self.collection.count())
        query_vec = embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        min_score = float(retriever_cfg.get("min_score", 0.0))
        hits = []
        for _id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            score = 1.0 - dist
            if score < min_score:
                continue
            hits.append(
                Hit(
                    id=_id,
                    source=(meta or {}).get("source", ""),
                    score=score,
                    text=doc or "",
                )
            )
        return hits

    def ask(
        self,
        query: str,
        history: list[Message] | None = None,
        *,
        stuff: bool | None = None,
        tags: list[str] | None = None,
    ) -> AskResult:
        history = list(history or [])
        hits = self.retrieve(query, stuff=stuff)
        combined = "\n\n".join(h.text for h in hits)
        retrieved_tokens = token_count(combined)

        compression_cfg = self.config["compression"]
        if compression_cfg.get("enabled"):
            compressed = compress(
                combined, query, int(compression_cfg.get("max_tokens", 120))
            )
        else:
            compressed = combined
        compressed_tokens = token_count(compressed)

        if compressed.strip():
            system = (
                "You are a Northstar Commerce support agent. Answer using only the "
                "provided context. If the context is insufficient, say you don't "
                "have that information. Be concise."
            )
            context_msg = Message("system", f"{system}\n\nContext:\n{compressed}")
        else:
            # Retrieval is off (Part 2). The grounded prompt would refuse
            # every question, so a cache hit and a miss return the same
            # string and the checkpoint is meaningless. This prefix is long
            # enough that paraphrases still clear similarity_threshold 0.80
            # while an unrelated question stays under it.
            context_msg = Message(
                "system",
                "You are a Northstar Commerce support agent. When policy context "
                "is provided, answer using only that context. When no policy "
                "context is provided, answer from general knowledge. Be concise.\n\n",
            )
        user_msg = Message("user", query)
        prompt = [context_msg, *history, user_msg]
        compaction_cfg = self.config["compaction"]
        if compaction_cfg.get("enabled"):
            prompt = compact(prompt, int(compaction_cfg.get("token_budget", 400)))

        # LiteLLM semantic cache keys on the *entire* messages array, so a
        # growing conversation looks like "the previous turn plus a little"
        # and false-hits. Only allow gateway caching on single-shot turns.
        cache_cfg = self.config.get("semantic_cache", {})
        use_gateway_cache = bool(cache_cfg.get("enabled")) and not history

        mesh_tags = ["lab2"]
        mesh_tags.append("cache-allowed" if use_gateway_cache else "cache-bypass")
        if tags:
            mesh_tags.extend(tags)

        answer, prompt_tokens, completion_tokens, cache_hit, similarity = (
            self._complete(
                prompt, tags=mesh_tags, use_gateway_cache=use_gateway_cache
            )
        )

        llm_calls = 0 if cache_hit else 1
        in_price, out_price = self._prices()
        return AskResult(
            answer=answer,
            cache_hit=cache_hit,
            hits=hits,
            retrieved_tokens=retrieved_tokens,
            compressed_tokens=compressed_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_calls=llm_calls,
            estimated_cost_usd=estimate_cost(
                prompt_tokens,
                completion_tokens,
                llm_calls,
                input_price_per_m=in_price,
                output_price_per_m=out_price,
            ),
            semantic_similarity=similarity,
            prompt_messages=prompt,
        )
