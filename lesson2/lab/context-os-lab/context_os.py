#!/usr/bin/env python3
"""
EXERCISE: Implement a Context OS from scratch.

In lecture you watched RAG retrieval, semantic caching, and context
compression/compaction as separate ideas. In this exercise you build the
three engines that make them real, then a provided Pipeline wires them
into one request path:

    query
      │
      ├─ semantic cache?  ── hit ──► reuse the stored answer (0 LLM tokens)
      │
      ▼ miss
    retrieve top-k chunks from the knowledge base
      │
      ▼
    compress retrieved text down to a token budget
      │
      ▼
    compact conversation history so the prompt stays bounded
      │
      ▼
    send the prompt to the model, store the answer in the cache

------------------------------------------------------------------------------
WHAT YOU IMPLEMENT
------------------------------------------------------------------------------
You only need to edit the six methods marked with `TODO(student)` below:

  1. chunk_text                 -> overlapping token windows over a document
  2. Retriever.retrieve         -> cosine top-k with a minimum score
  3. SemanticCache.lookup       -> nearest cached query, threshold + TTL
  4. SemanticCache.store        -> write an entry, LRU-evict under budget
  5. compress                   -> extractive sentence selection for a query
  6. compact                    -> bound a conversation to a token budget

Everything else (tokenizer, lexical embedder, dataclasses, Pipeline) is
provided. The embedder is a hashed bag-of-words stand-in for the neural
model you used in lecture: fast, deterministic, no download. Paraphrases
that share wording land close together in this space; unrelated sentences
do not. The *mechanics* of retrieval, caching, and compaction are the same
ones a production system uses on top of real embeddings.

Run `pytest` to check your work. Run `python run_demo.py` to watch your
engines reproduce the four-arm cost experiment from the parent lab.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# Pricing (Claude Sonnet 4.6 list pricing, USD per million input tokens).
# Identical to lesson 1 so the numbers feel comparable.
# --------------------------------------------------------------------------- #
INPUT_PRICE_PER_M = 3.00
EMBED_DIM = 256

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


# --------------------------------------------------------------------------- #
# Tokenizer + embedder (PROVIDED — do not edit).
# --------------------------------------------------------------------------- #
def split_tokens(text: str) -> list[str]:
    """Split text into word and punctuation tokens."""
    return _TOKEN_RE.findall(text)


def tokenize(text: str) -> list[int]:
    """Deterministic token ids. `len(tokenize(text))` is the token count."""
    return [zlib.crc32(tok.encode("utf-8")) for tok in split_tokens(text)]


def split_sentences(text: str) -> list[str]:
    """Split on end-of-sentence punctuation. A document with no terminator
    is returned as a single sentence.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def embed(text: str) -> list[float]:
    """Lexical hashed bag-of-words + bigrams, L2-normalized.

    Shared tokens land in the same buckets, so paraphrases that reuse
    wording score high. This is a stand-in for the neural embeddings from
    lecture — fast, deterministic, no model download.
    """
    toks = [t.lower() for t in split_tokens(text)]
    vec = [0.0] * EMBED_DIM
    for t in toks:
        vec[zlib.crc32(t.encode("utf-8")) % EMBED_DIM] += 1.0
    for a, b in zip(toks, toks[1:]):
        vec[zlib.crc32(f"{a}_{b}".encode("utf-8")) % EMBED_DIM] += 0.5
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def estimate_input_cost(prompt_tokens: int, llm_calls: int) -> float:
    """Cost of the input side of one request, in USD.

    A semantic-cache hit makes zero LLM calls, so it costs $0 on this meter.
    (Embedding cost is treated as negligible, which matches the lecture
    framing: the cache exists to avoid the *generation* bill.)
    """
    if llm_calls <= 0:
        return 0.0
    return prompt_tokens * (INPUT_PRICE_PER_M / 1_000_000)


# --------------------------------------------------------------------------- #
# Data structures (PROVIDED — do not edit).
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """One indexed piece of the knowledge base."""

    id: str
    text: str
    source: str
    vector: list[float]


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float


@dataclass
class CacheRecord:
    query: str
    vector: list[float]
    answer: str
    created_at: float
    last_used: float


@dataclass
class CacheHit:
    answer: str
    similarity: float
    query: str  # the stored query that matched


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class Usage:
    """Per-request accounting for the Context OS pipeline."""

    retrieved_tokens: int
    compressed_tokens: int
    prompt_tokens: int
    llm_calls: int
    cache_hit: bool
    estimated_input_cost_usd: float


# ========================================================================== #
# TODO(student) #1: overlapping token windows
# ========================================================================== #
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into overlapping windows of `chunk_size` tokens.

    Requirements:
      - Tokenize with `split_tokens`. Reconstruct each chunk by joining the
        window's tokens with a single space.
      - Windows advance by `step = chunk_size - overlap` tokens.
      - The final window may be shorter than `chunk_size`. Do not emit an
        extra empty chunk.
      - Empty / whitespace-only text returns `[]`.
      - Raise `ValueError` if `chunk_size <= 0` or if `overlap < 0` or
        `overlap >= chunk_size`.

    Example (chunk_size=4, overlap=2) over 8 tokens:
      tokens 0-3, then 2-5, then 4-7.
    """
    raise NotImplementedError("Implement chunk_text")


# ========================================================================== #
# TODO(student) #2: cosine top-k retrieval
# ========================================================================== #
class Retriever:
    """In-memory vector index over already-chunked texts."""

    def __init__(self, *, min_score: float = 0.0) -> None:
        self.min_score = min_score
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def index(self, texts: list[str], *, source: str = "") -> None:
        """PROVIDED. Embed and store each string as one chunk."""
        for text in texts:
            cid = f"{source}:{len(self._chunks)}" if source else f"c{len(self._chunks)}"
            self._chunks.append(
                Chunk(id=cid, text=text, source=source, vector=embed(text))
            )

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        """Return up to `k` chunks most similar to `query`.

        Requirements:
          - Embed the query with `embed`, score every chunk with `cosine`.
          - Drop chunks whose score is strictly below `self.min_score`.
          - Sort remaining hits by score descending; break ties by chunk id
            ascending (so the result is deterministic).
          - Return at most `k` hits. If `k <= 0` or the index is empty,
            return `[]`.
          - Do not mutate the index.
        """
        raise NotImplementedError("Implement Retriever.retrieve")


# ========================================================================== #
# TODO(student) #3 + #4: semantic cache
# ========================================================================== #
class SemanticCache:
    """Nearest-neighbor cache over previous queries.

    A lookup is a HIT when the closest unexpired entry has cosine similarity
    at or above `threshold`. Hits are free on the LLM meter; misses fall
    through to retrieval + generation, then `store`.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        ttl_seconds: float = 300.0,
        max_entries: int = 10_000,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: list[CacheRecord] = []

    def __len__(self) -> int:
        return len(self._store)

    def _is_expired(self, record: CacheRecord, now: float) -> bool:
        return (now - record.created_at) > self.ttl_seconds

    def lookup(self, query: str, now: float) -> CacheHit | None:
        """Find the nearest unexpired cached query.

        Requirements:
          - Skip expired records (see `_is_expired`). Do not delete them.
          - If the best remaining cosine similarity is >= `self.threshold`,
            it is a HIT: refresh that record's `created_at` and `last_used`
            to `now` (sliding TTL) and return a `CacheHit`.
          - Otherwise return `None`. Do not mutate the store on a miss.
          - An empty store is a miss.
        """
        raise NotImplementedError("Implement SemanticCache.lookup")

    def store(self, query: str, answer: str, now: float) -> None:
        """Insert a new cache record for `query` -> `answer`.

        Requirements:
          - Always append a new `CacheRecord` (do not try to merge with
            existing entries). Set `created_at == last_used == now`.
          - After inserting, while `len(self._store) > self.max_entries`,
            evict the record with the smallest `last_used` (LRU). Ties may
            be broken however you like.
        """
        raise NotImplementedError("Implement SemanticCache.store")


# ========================================================================== #
# TODO(student) #5: extractive compression
# ========================================================================== #
def compress(text: str, query: str, max_tokens: int) -> str:
    """Shrink `text` to `max_tokens` by keeping the sentences most relevant
    to `query`.

    Requirements:
      - If `max_tokens <= 0`, return `""`.
      - If `len(tokenize(text)) <= max_tokens`, return `text` unchanged
        (preserve original whitespace).
      - Otherwise split with `split_sentences`. Score each sentence as the
        number of *distinct* query tokens (lowercased) that also appear in
        the sentence. Rank by score descending, breaking ties by original
        index ascending.
      - Walk that ranking and keep a sentence if adding it (joining kept
        sentences with a single space, in original order) still satisfies
        `len(tokenize(result)) <= max_tokens`.
      - If no full sentence fits, return the highest-ranked sentence
        truncated to `max_tokens` tokens (`" ".join(split_tokens(s)[:max_tokens])`).
    """
    raise NotImplementedError("Implement compress")


# ========================================================================== #
# TODO(student) #6: conversation compaction
# ========================================================================== #
def compact(messages: list[Message], budget: int) -> list[Message]:
    """Bound a conversation to `budget` tokens, keeping recent turns.

    Requirements:
      - Token count of a message is `len(tokenize(message.content))`.
      - If the full list already fits (or `messages` is empty), return a
        shallow copy of `messages` unchanged.
      - If `budget <= 0`, return `[]`.
      - System messages (`role == "system"`) are never dropped and never
        truncated, even if they alone exceed the budget.
      - Of the remaining (non-system) messages, keep the longest *suffix*
        (newest turns) that fits in the leftover budget. Walk from newest
        to oldest and stop at the first message that does not fit — do not
        skip over it to pick a smaller older message.
      - If any non-system message was dropped, insert a marker immediately
        after the system messages:
            Message(role="system", content=f"[compacted: {n} earlier turns]")
        where `n` is the number of dropped non-system messages. The marker
        counts against the budget. If adding it would overflow, drop the
        oldest kept non-system message (and bump `n`) until it fits, or
        until the suffix is empty. If the suffix is empty and the marker
        still doesn't fit, omit the marker.
    """
    raise NotImplementedError("Implement compact")


# --------------------------------------------------------------------------- #
# Pipeline (PROVIDED — do not edit).
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    answer: str
    usage: Usage
    retrieved: list[RetrievalHit]
    prompt_messages: list[Message]


class Pipeline:
    """Wires retrieval, cache, compression, and compaction into one call."""

    def __init__(
        self,
        retriever: Retriever,
        cache: SemanticCache,
        *,
        k: int = 3,
        compress_max_tokens: int | None = None,
        token_budget: int | None = None,
        answer_fn=None,
    ) -> None:
        self.retriever = retriever
        self.cache = cache
        self.k = k
        self.compress_max_tokens = compress_max_tokens
        self.token_budget = token_budget
        self.answer_fn = answer_fn or _default_answer

    def process(self, query: str, history: list[Message], now: float) -> Result:
        hit = self.cache.lookup(query, now)
        if hit is not None:
            usage = Usage(
                retrieved_tokens=0,
                compressed_tokens=0,
                prompt_tokens=0,
                llm_calls=0,
                cache_hit=True,
                estimated_input_cost_usd=0.0,
            )
            return Result(
                answer=hit.answer,
                usage=usage,
                retrieved=[],
                prompt_messages=[],
            )

        retrieved = self.retriever.retrieve(query, self.k)
        combined = " ".join(h.chunk.text for h in retrieved)
        retrieved_tokens = len(tokenize(combined))
        if self.compress_max_tokens is None:
            compressed = combined
        else:
            compressed = compress(combined, query, self.compress_max_tokens)
        compressed_tokens = len(tokenize(compressed))

        context_msg = Message("system", f"Context:\n{compressed}")
        user_msg = Message("user", query)
        prompt = [context_msg, *history, user_msg]
        if self.token_budget is not None:
            prompt = compact(prompt, self.token_budget)
        prompt_tokens = sum(len(tokenize(m.content)) for m in prompt)

        answer = self.answer_fn(query, retrieved, prompt)
        self.cache.store(query, answer, now)
        usage = Usage(
            retrieved_tokens=retrieved_tokens,
            compressed_tokens=compressed_tokens,
            prompt_tokens=prompt_tokens,
            llm_calls=1,
            cache_hit=False,
            estimated_input_cost_usd=estimate_input_cost(prompt_tokens, 1),
        )
        return Result(
            answer=answer,
            usage=usage,
            retrieved=retrieved,
            prompt_messages=prompt,
        )


def _default_answer(query: str, retrieved: list[RetrievalHit], prompt: list[Message]) -> str:
    if not retrieved:
        return "I don't have enough context to answer."
    return f"Based on policy: {retrieved[0].chunk.text}"


__all__ = [
    "chunk_text",
    "compress",
    "compact",
    "Retriever",
    "SemanticCache",
    "Pipeline",
    "Chunk",
    "RetrievalHit",
    "CacheRecord",
    "CacheHit",
    "Message",
    "Usage",
    "Result",
    "tokenize",
    "split_tokens",
    "split_sentences",
    "embed",
    "cosine",
    "estimate_input_cost",
    "INPUT_PRICE_PER_M",
]

_ = (field, Iterable)  # referenced for student stretch goals
