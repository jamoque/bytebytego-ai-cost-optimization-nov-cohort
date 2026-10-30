#!/usr/bin/env python3
"""
REFERENCE SOLUTION — do not open until you've finished `prompt_cache.py`.

A complete, correct implementation of the prompt-caching engine. Drop-in
compatible with the public API in `prompt_cache.py`. Instructors can validate
the test suite against it with:

    PROMPT_CACHE_IMPL=reference_solution pytest -v
"""

from __future__ import annotations

import hashlib
import re
import zlib
from dataclasses import dataclass

INPUT_PRICE_PER_M = 3.00
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SEED_HASH = hashlib.sha256(b"prompt-cache-seed").hexdigest()


def tokenize(text: str) -> list[int]:
    return [zlib.crc32(tok.encode("utf-8")) for tok in _TOKEN_RE.findall(text)]


@dataclass
class BlockRef:
    index: int
    chain_hash: str
    n_tokens: int
    cumulative_tokens: int


@dataclass
class CacheEntry:
    chain_hash: str
    n_tokens: int
    created_at: float
    last_used: float


@dataclass
class Usage:
    cache_read_tokens: int
    cache_creation_tokens: int
    fresh_tokens: int
    total_tokens: int
    estimated_input_cost_usd: float

    @property
    def cache_hit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.cache_read_tokens / self.total_tokens


def estimate_input_cost(fresh: int, write: int, read: int) -> float:
    per_token = INPUT_PRICE_PER_M / 1_000_000
    return (
        fresh * per_token
        + write * per_token * CACHE_WRITE_MULTIPLIER
        + read * per_token * CACHE_READ_MULTIPLIER
    )


class PrefixCache:
    def __init__(
        self,
        *,
        block_size: int = 16,
        min_cacheable_tokens: int = 0,
        ttl_seconds: float = 300.0,
        max_blocks: int = 10_000,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.min_cacheable_tokens = min_cacheable_tokens
        self.ttl_seconds = ttl_seconds
        self.max_blocks = max_blocks
        self._store: dict[str, CacheEntry] = {}

    def __len__(self) -> int:
        return len(self._store)

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.created_at) > self.ttl_seconds

    def _block_hashes(self, token_ids: list[int]) -> list[BlockRef]:
        blocks: list[BlockRef] = []
        prev_hash = _SEED_HASH
        n_full = len(token_ids) // self.block_size
        for i in range(n_full):
            chunk = token_ids[i * self.block_size : (i + 1) * self.block_size]
            h = hashlib.sha256()
            h.update(prev_hash.encode("utf-8"))
            h.update(b"|")
            h.update(",".join(map(str, chunk)).encode("utf-8"))
            chain = h.hexdigest()
            blocks.append(
                BlockRef(
                    index=i,
                    chain_hash=chain,
                    n_tokens=self.block_size,
                    cumulative_tokens=(i + 1) * self.block_size,
                )
            )
            prev_hash = chain
        return blocks

    def lookup(self, blocks: list[BlockRef], now: float) -> int:
        matched = 0
        for b in blocks:
            entry = self._store.get(b.chain_hash)
            if entry is None or self._is_expired(entry, now):
                break
            matched += 1
        return matched

    def store(self, blocks: list[BlockRef], matched_blocks: int, now: float) -> None:
        for i, b in enumerate(blocks):
            if i < matched_blocks:
                entry = self._store.get(b.chain_hash)
                if entry is not None:
                    entry.created_at = now
                    entry.last_used = now
            else:
                self._store[b.chain_hash] = CacheEntry(
                    chain_hash=b.chain_hash,
                    n_tokens=b.n_tokens,
                    created_at=now,
                    last_used=now,
                )
        self._evict(now)

    def _evict(self, now: float) -> None:
        while len(self._store) > self.max_blocks:
            lru_key = min(self._store, key=lambda k: self._store[k].last_used)
            del self._store[lru_key]

    def process(self, token_ids: list[int], now: float) -> Usage:
        total = len(token_ids)
        if total < self.min_cacheable_tokens:
            return Usage(
                cache_read_tokens=0,
                cache_creation_tokens=0,
                fresh_tokens=total,
                total_tokens=total,
                estimated_input_cost_usd=estimate_input_cost(total, 0, 0),
            )
        blocks = self._block_hashes(token_ids)
        matched = self.lookup(blocks, now)
        read = sum(b.n_tokens for b in blocks[:matched])
        write = sum(b.n_tokens for b in blocks[matched:])
        self.store(blocks, matched, now)
        fresh = total - read - write
        return Usage(
            cache_read_tokens=read,
            cache_creation_tokens=write,
            fresh_tokens=fresh,
            total_tokens=total,
            estimated_input_cost_usd=estimate_input_cost(fresh, write, read),
        )
