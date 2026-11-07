#!/usr/bin/env python3
"""
REFERENCE SOLUTION — do not open until you've finished `context_os.py`.

A complete, correct implementation of the Context OS engines. Drop-in
compatible with the public API in `context_os.py`. Instructors can validate
the test suite against it with:

    CONTEXT_OS_IMPL=reference_solution pytest -v
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

INPUT_PRICE_PER_M = 3.00
EMBED_DIM = 256

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def split_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def tokenize(text: str) -> list[int]:
    return [zlib.crc32(tok.encode("utf-8")) for tok in split_tokens(text)]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def embed(text: str) -> list[float]:
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
    if llm_calls <= 0:
        return 0.0
    return prompt_tokens * (INPUT_PRICE_PER_M / 1_000_000)


@dataclass
class Chunk:
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
    query: str


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Usage:
    retrieved_tokens: int
    compressed_tokens: int
    prompt_tokens: int
    llm_calls: int
    cache_hit: bool
    estimated_input_cost_usd: float


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    toks = split_tokens(text)
    if not toks:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(toks):
        window = toks[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(toks):
            break
        start += step
    return chunks


class Retriever:
    def __init__(self, *, min_score: float = 0.0) -> None:
        self.min_score = min_score
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def index(self, texts: list[str], *, source: str = "") -> None:
        for text in texts:
            cid = f"{source}:{len(self._chunks)}" if source else f"c{len(self._chunks)}"
            self._chunks.append(
                Chunk(id=cid, text=text, source=source, vector=embed(text))
            )

    def retrieve(self, query: str, k: int) -> list[RetrievalHit]:
        if k <= 0 or not self._chunks:
            return []
        qv = embed(query)
        hits: list[RetrievalHit] = []
        for ch in self._chunks:
            score = cosine(qv, ch.vector)
            if score >= self.min_score:
                hits.append(RetrievalHit(chunk=ch, score=score))
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]


class SemanticCache:
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
        qv = embed(query)
        best: CacheRecord | None = None
        best_sim = -1.0
        for rec in self._store:
            if self._is_expired(rec, now):
                continue
            sim = cosine(qv, rec.vector)
            if sim > best_sim:
                best_sim = sim
                best = rec
        if best is None or best_sim < self.threshold:
            return None
        best.created_at = now
        best.last_used = now
        return CacheHit(answer=best.answer, similarity=best_sim, query=best.query)

    def store(self, query: str, answer: str, now: float) -> None:
        self._store.append(
            CacheRecord(
                query=query,
                vector=embed(query),
                answer=answer,
                created_at=now,
                last_used=now,
            )
        )
        while len(self._store) > self.max_entries:
            lru_i = min(range(len(self._store)), key=lambda i: self._store[i].last_used)
            self._store.pop(lru_i)


def compress(text: str, query: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if len(tokenize(text)) <= max_tokens:
        return text
    sentences = split_sentences(text)
    if not sentences:
        return ""
    query_set = {t.lower() for t in split_tokens(query)}

    def score(sentence: str) -> int:
        sent_set = {t.lower() for t in split_tokens(sentence)}
        return sum(1 for t in sent_set if t in query_set)

    ranked = sorted(
        enumerate(sentences),
        key=lambda iv: (-score(iv[1]), iv[0]),
    )
    kept: set[int] = set()
    for i, _sentence in ranked:
        candidate_idx = sorted(kept | {i})
        candidate = " ".join(sentences[j] for j in candidate_idx)
        if len(tokenize(candidate)) <= max_tokens:
            kept.add(i)
    if not kept:
        _i, sentence = ranked[0]
        return " ".join(split_tokens(sentence)[:max_tokens])
    return " ".join(sentences[j] for j in sorted(kept))


def _msg_tokens(message: Message) -> int:
    return len(tokenize(message.content))


def compact(messages: list[Message], budget: int) -> list[Message]:
    if budget <= 0:
        return []
    if not messages:
        return []
    total = sum(_msg_tokens(m) for m in messages)
    if total <= budget:
        return list(messages)

    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    used = sum(_msg_tokens(m) for m in system)

    suffix: list[Message] = []
    for m in reversed(rest):
        t = _msg_tokens(m)
        if used + t <= budget:
            suffix.append(m)
            used += t
        else:
            break
    suffix.reverse()

    dropped = len(rest) - len(suffix)
    if dropped == 0:
        return system + suffix

    marker = Message(role="system", content=f"[compacted: {dropped} earlier turns]")
    mt = _msg_tokens(marker)
    while suffix and used + mt > budget:
        dropped_m = suffix.pop(0)
        used -= _msg_tokens(dropped_m)
        dropped += 1
        marker = Message(role="system", content=f"[compacted: {dropped} earlier turns]")
        mt = _msg_tokens(marker)
    if used + mt <= budget:
        return system + [marker] + suffix
    return system + suffix


@dataclass
class Result:
    answer: str
    usage: Usage
    retrieved: list[RetrievalHit]
    prompt_messages: list[Message]


class Pipeline:
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
