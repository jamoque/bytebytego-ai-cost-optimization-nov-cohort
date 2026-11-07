#!/usr/bin/env python3
"""
Test suite for the Context OS exercise.

Run with:   pytest -v

These tests are your definition of "done". They import your implementation
from `context_os.py` by default. (Instructors can point the tests at a
reference implementation with `CONTEXT_OS_IMPL=reference_solution pytest`.)
"""

from __future__ import annotations

import importlib
import os

import pytest

impl = importlib.import_module(os.environ.get("CONTEXT_OS_IMPL", "context_os"))

chunk_text = impl.chunk_text
compress = impl.compress
compact = impl.compact
Retriever = impl.Retriever
SemanticCache = impl.SemanticCache
Pipeline = impl.Pipeline
Message = impl.Message
tokenize = impl.tokenize
split_tokens = impl.split_tokens
embed = impl.embed
cosine = impl.cosine
estimate_input_cost = impl.estimate_input_cost


# --------------------------------------------------------------------------- #
# Tokenizer / embedder (provided — sanity checks)
# --------------------------------------------------------------------------- #
def test_tokenize_is_deterministic():
    assert tokenize("hello world") == tokenize("hello world")


def test_embed_is_normalized():
    v = embed("refunds take five business days")
    norm = sum(x * x for x in v) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_lexical_overlap_outranks_unrelated():
    refunds = embed("Refunds are issued within five business days.")
    shipping = embed("Domestic shipping usually takes three days.")
    query = embed("how long do refunds take")
    assert cosine(query, refunds) > cosine(query, shipping)


# --------------------------------------------------------------------------- #
# #1 chunk_text
# --------------------------------------------------------------------------- #
def test_chunk_text_emits_overlapping_windows():
    text = "one two three four five six seven eight"
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert chunks == [
        "one two three four",
        "three four five six",
        "five six seven eight",
    ]


def test_chunk_text_final_window_may_be_short():
    text = "one two three four five"
    chunks = chunk_text(text, chunk_size=3, overlap=0)
    assert chunks == ["one two three", "four five"]


def test_chunk_text_empty_input():
    assert chunk_text("   ", chunk_size=4, overlap=0) == []
    assert chunk_text("", chunk_size=4, overlap=0) == []


def test_chunk_text_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=4, overlap=4)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=4, overlap=-1)


# --------------------------------------------------------------------------- #
# #2 retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_ranks_relevant_chunk_first():
    r = Retriever()
    r.index(
        [
            "To get a refund use the returns portal and the original payment method.",
            "Domestic packages travel with ground carriers after warehouse pickup.",
            "Reset your password from the login page using the account email.",
        ]
    )
    hits = r.retrieve("how do I get a refund", k=3)
    assert hits, "expected at least one hit"
    assert "refund" in hits[0].chunk.text.lower()


def test_retrieve_respects_k():
    r = Retriever()
    r.index(["alpha token", "beta token", "gamma token"])
    hits = r.retrieve("token", k=2)
    assert len(hits) == 2


def test_retrieve_min_score_filters():
    r = Retriever(min_score=0.99)
    r.index(["completely unrelated galaxy telescope image"])
    hits = r.retrieve("how do I get a refund for my order", k=5)
    assert hits == []


def test_retrieve_k_zero_or_empty():
    r = Retriever()
    assert r.retrieve("anything", k=3) == []
    r.index(["hello world"])
    assert r.retrieve("hello", k=0) == []


# --------------------------------------------------------------------------- #
# #3 lookup + #4 store
# --------------------------------------------------------------------------- #
def test_exact_query_is_a_hit():
    c = SemanticCache(threshold=0.7)
    c.store("how do I get a refund for my order", "Use the returns portal.", now=0.0)
    hit = c.lookup("how do I get a refund for my order", now=1.0)
    assert hit is not None
    assert hit.answer == "Use the returns portal."
    assert hit.similarity == pytest.approx(1.0)


def test_lexical_paraphrase_is_a_hit():
    c = SemanticCache(threshold=0.7)
    c.store("how do I get a refund for my order", "Use the returns portal.", now=0.0)
    hit = c.lookup("how do I get a refund for this order", now=1.0)
    assert hit is not None
    assert hit.answer == "Use the returns portal."


def test_unrelated_query_is_a_miss():
    c = SemanticCache(threshold=0.7)
    c.store("how do I get a refund for my order", "Use the returns portal.", now=0.0)
    hit = c.lookup("the telescope captured a stunning image of a distant galaxy", now=1.0)
    assert hit is None


def test_lookup_on_empty_cache_is_a_miss():
    c = SemanticCache()
    assert c.lookup("anything", now=0.0) is None


def test_entries_expire_after_ttl():
    c = SemanticCache(threshold=0.5, ttl_seconds=300)
    c.store("how do I get a refund for my order", "portal", now=0.0)
    assert c.lookup("how do I get a refund for my order", now=301.0) is None


def test_read_refreshes_ttl():
    c = SemanticCache(threshold=0.5, ttl_seconds=300)
    c.store("how do I get a refund for my order", "portal", now=0.0)
    assert c.lookup("how do I get a refund for my order", now=200.0) is not None
    assert c.lookup("how do I get a refund for my order", now=450.0) is not None


def test_lru_eviction_under_budget():
    c = SemanticCache(threshold=0.5, max_entries=2)
    c.store("query alpha unique words here", "A", now=0.0)
    c.store("query beta different tokens now", "B", now=1.0)
    c.store("query gamma totally new content", "C", now=2.0)
    assert len(c) == 2
    assert c.lookup("query alpha unique words here", now=3.0) is None
    hit = c.lookup("query gamma totally new content", now=3.0)
    assert hit is not None
    assert hit.answer == "C"


def test_recently_used_survives_eviction():
    c = SemanticCache(threshold=0.5, max_entries=2)
    c.store("query alpha unique words here", "A", now=0.0)
    c.store("query beta different tokens now", "B", now=1.0)
    assert c.lookup("query alpha unique words here", now=2.0) is not None  # A most recent
    c.store("query gamma totally new content", "C", now=3.0)  # evict B
    assert c.lookup("query alpha unique words here", now=4.0) is not None
    assert c.lookup("query beta different tokens now", now=4.0) is None


# --------------------------------------------------------------------------- #
# #5 compress
# --------------------------------------------------------------------------- #
def test_compress_under_budget_is_unchanged():
    text = "Refunds take five days."
    assert compress(text, "refunds", max_tokens=100) == text


def test_compress_keeps_relevant_sentence():
    text = (
        "Refunds take five business days after we receive the item. "
        "The cafeteria serves wood-fired pizza on Fridays. "
        "Contact billing if you need a copy of an old invoice."
    )
    out = compress(text, "how long do refunds take", max_tokens=12)
    assert "Refunds take five business days" in out
    assert "pizza" not in out
    assert len(tokenize(out)) <= 12


def test_compress_zero_budget_is_empty():
    assert compress("Refunds take five days.", "refunds", max_tokens=0) == ""


def test_compress_truncates_when_no_sentence_fits():
    text = "Refunds take five business days after we receive the item."
    out = compress(text, "refunds", max_tokens=3)
    assert len(tokenize(out)) <= 3
    assert out  # something survived


# --------------------------------------------------------------------------- #
# #6 compact
# --------------------------------------------------------------------------- #
def test_compact_under_budget_is_unchanged():
    msgs = [
        Message("system", "You are a support agent."),
        Message("user", "Hello"),
        Message("assistant", "Hi there"),
    ]
    assert compact(msgs, budget=10_000) == msgs


def test_compact_keeps_system_and_recent_suffix():
    msgs = [
        Message("system", "You are a support agent."),
        Message("user", "first turn about refunds taking too long for my order"),
        Message("assistant", "I can help with that refund timeline question today"),
        Message("user", "second turn about a replacement shipment status update"),
        Message("assistant", "The replacement is currently at the local hub"),
        Message("user", "third turn I also forgot my password this morning"),
    ]
    budget = (
        len(tokenize(msgs[0].content))
        + len(tokenize("[compacted: 4 earlier turns]"))
        + len(tokenize(msgs[-1].content))
        + 2  # small cushion so only the last user turn plus marker + system fit
    )
    out = compact(msgs, budget=budget)
    roles_contents = [(m.role, m.content) for m in out]
    assert roles_contents[0] == ("system", "You are a support agent.")
    assert any(m.content.startswith("[compacted:") for m in out)
    assert msgs[-1] in out
    assert msgs[1] not in out  # oldest user turn dropped
    assert sum(len(tokenize(m.content)) for m in out) <= budget


def test_compact_zero_budget_is_empty():
    assert compact([Message("user", "hello")], budget=0) == []


def test_compact_inserts_marker_with_dropped_count():
    msgs = [
        Message("system", "You are a support agent."),
        Message("user", "first turn discusses the refund timeline for a blender"),
        Message("assistant", "I can look up that blender refund for you today"),
        Message("user", "second turn is only about resetting the password please"),
    ]
    # Keep system + last user turn + a 2-turn marker; drop the first two non-system messages.
    last = len(tokenize(msgs[-1].content))
    sys_t = len(tokenize(msgs[0].content))
    marker = len(tokenize("[compacted: 2 earlier turns]"))
    budget = sys_t + marker + last
    out = compact(msgs, budget=budget)
    markers = [m for m in out if m.content.startswith("[compacted:")]
    assert len(markers) == 1
    assert markers[0].content == "[compacted: 2 earlier turns]"
    assert out[-1] == msgs[-1]


# --------------------------------------------------------------------------- #
# Pipeline wiring (uses the pieces above)
# --------------------------------------------------------------------------- #
def test_pipeline_cache_hit_skips_llm():
    r = Retriever()
    r.index(["Refunds take five business days after inspection."])
    cache = SemanticCache(threshold=0.7)
    pipe = Pipeline(r, cache, k=1)
    first = pipe.process("how do I get a refund for my order", [], now=0.0)
    assert first.usage.cache_hit is False
    assert first.usage.llm_calls == 1
    second = pipe.process("how do I get a refund for this order", [], now=1.0)
    assert second.usage.cache_hit is True
    assert second.usage.llm_calls == 0
    assert second.usage.estimated_input_cost_usd == 0.0
    assert second.answer == first.answer


def test_pipeline_cost_matches_formula():
    r = Retriever()
    r.index(["Refunds take five business days after inspection."])
    cache = SemanticCache(threshold=0.99)  # force a miss on a slightly different query
    pipe = Pipeline(r, cache, k=1, token_budget=10_000)
    result = pipe.process("how long do refunds take", [], now=0.0)
    expected = estimate_input_cost(result.usage.prompt_tokens, 1)
    assert result.usage.estimated_input_cost_usd == pytest.approx(expected)
    assert result.usage.prompt_tokens > 0
