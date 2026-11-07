#!/usr/bin/env python3
"""
Replay the parent lab's cost experiment through YOUR from-scratch Context OS
(PROVIDED — do not edit).

No API key and no network needed: the corpus and queries come from ./fixtures
and a fake MockModel provides latency. The point is to watch your engines
reproduce the four-arm shape from the hands-on lab — stuffing vs RAG vs
RAG+cache vs the full pipeline — and quantify the token + cost savings.

    python run_demo.py              # both workloads, default settings
    python run_demo.py --no-sleep   # skip latency simulation (fast)
"""

from __future__ import annotations

import argparse
import json
import os
import importlib
from pathlib import Path

from mock_model import MockModel

_impl = importlib.import_module(os.environ.get("CONTEXT_OS_IMPL", "context_os"))
chunk_text = _impl.chunk_text
Retriever = _impl.Retriever
SemanticCache = _impl.SemanticCache
Pipeline = _impl.Pipeline
Message = _impl.Message
tokenize = _impl.tokenize

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_corpus() -> list[dict]:
    return json.loads((FIXTURES / "corpus.json").read_text())["documents"]


def load_queries() -> list[dict]:
    return json.loads((FIXTURES / "queries.json").read_text())


def load_conversation() -> list[str]:
    return json.loads((FIXTURES / "conversation.json").read_text())


def build_retriever(docs: list[dict], *, stuff: bool, chunk_size: int, overlap: int) -> Retriever:
    r = Retriever(min_score=0.0)
    if stuff:
        blob = " ".join(d["text"] for d in docs)
        r.index([blob], source="corpus")
        return r
    for d in docs:
        pieces = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        if not pieces:
            pieces = [d["text"]]
        r.index(pieces, source=d["id"])
    return r


def build_pipeline(
    *,
    docs: list[dict],
    stuff: bool,
    cache_enabled: bool,
    compress_max_tokens: int | None,
    token_budget: int | None,
    k: int,
    chunk_size: int,
    overlap: int,
    threshold: float,
) -> Pipeline:
    retriever = build_retriever(docs, stuff=stuff, chunk_size=chunk_size, overlap=overlap)
    cache = SemanticCache(
        threshold=threshold if cache_enabled else 1.01,  # 1.01 => nothing hits
        ttl_seconds=300.0,
        max_entries=10_000,
    )
    return Pipeline(
        retriever,
        cache,
        k=1 if stuff else k,
        compress_max_tokens=None if stuff else compress_max_tokens,
        token_budget=None if stuff else token_budget,
    )


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    header = (
        f"{'Q':>3} {'Hit':>4} {'Retr':>6} {'Comp':>6} {'Prompt':>7} "
        f"{'LLM':>4} {'Cost$':>10} {'Lat':>7}"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        u = row["usage"]
        print(
            f"{i:>3} {'yes' if u.cache_hit else 'no':>4} "
            f"{u.retrieved_tokens:>6} {u.compressed_tokens:>6} "
            f"{u.prompt_tokens:>7} {u.llm_calls:>4} "
            f"{u.estimated_input_cost_usd:>10.6f} {row['latency']:>6.3f}s"
        )


def summarize(rows: list[dict]) -> dict:
    return {
        "llm_calls": sum(r["usage"].llm_calls for r in rows),
        "cache_hits": sum(1 for r in rows if r["usage"].cache_hit),
        "total_prompt_tokens": sum(r["usage"].prompt_tokens for r in rows),
        "total_cost_usd": round(sum(r["usage"].estimated_input_cost_usd for r in rows), 6),
        "mean_latency_s": round(
            sum(r["latency"] for r in rows) / max(len(rows), 1), 3
        ),
    }


def run_queries(
    pipe: Pipeline,
    queries: list[dict],
    model: MockModel,
    *,
    sleep: bool,
) -> list[dict]:
    rows = []
    clock = 0.0
    for q in queries:
        result = pipe.process(q["text"], history=[], now=clock)
        resp = model.generate(result.usage, result.retrieved, sleep=sleep)
        rows.append({"usage": result.usage, "latency": resp.latency_s, "answer": result.answer})
        clock += resp.latency_s + 1.0
    return rows


def run_conversation(
    pipe: Pipeline,
    turns: list[str],
    model: MockModel,
    *,
    sleep: bool,
) -> list[dict]:
    rows = []
    history: list = []
    clock = 0.0
    for text in turns:
        result = pipe.process(text, history=history, now=clock)
        resp = model.generate(result.usage, result.retrieved, sleep=sleep)
        answer = result.answer or resp.text
        rows.append({"usage": result.usage, "latency": resp.latency_s, "answer": answer})
        history.append(Message("user", text))
        history.append(Message("assistant", answer))
        clock += resp.latency_s + 1.0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--overlap", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--compress-max", type=int, default=40)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()

    docs = load_corpus()
    queries = load_queries()
    turns = load_conversation()
    model = MockModel()
    sleep = not args.no_sleep
    stuffed_tokens = len(tokenize(" ".join(d["text"] for d in docs)))

    print("From-scratch Context OS demo")
    print(f"  Corpus documents:     {len(docs)}")
    print(f"  Stuffed corpus:       ~{stuffed_tokens} tokens")
    print(f"  Query workload:       {len(queries)} questions (includes paraphrases)")
    print(f"  Conversation turns:   {len(turns)}")
    print(f"  RAG k / chunk / overlap: {args.k} / {args.chunk_size} / {args.overlap}")
    print(f"  Cache threshold:      {args.threshold:.2f}")
    print(f"  Compress max tokens:  {args.compress_max}")
    print(f"  Compaction budget:    {args.budget}")

    common = dict(
        docs=docs,
        k=args.k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )
    arms = {
        "A stuffing": build_pipeline(
            stuff=True, cache_enabled=False,
            compress_max_tokens=None, token_budget=None, **common,
        ),
        "B RAG only": build_pipeline(
            stuff=False, cache_enabled=False,
            compress_max_tokens=None, token_budget=None, **common,
        ),
        "C RAG + cache": build_pipeline(
            stuff=False, cache_enabled=True,
            compress_max_tokens=None, token_budget=None, **common,
        ),
        "D full pipeline": build_pipeline(
            stuff=False, cache_enabled=True,
            compress_max_tokens=args.compress_max, token_budget=args.budget, **common,
        ),
    }

    print("\n=== Workload 1: Q&A queries (empty history, paraphrases included) ===")
    summaries = {}
    for name, pipe in arms.items():
        rows = run_queries(pipe, queries, model, sleep=sleep)
        print_table(f"--- {name} ---", rows)
        summaries[name] = summarize(rows)

    print("\n=== Query-workload summary ===")
    print(json.dumps(summaries, indent=2))

    print("\n=== Workload 2: multi-turn conversation (history grows) ===")
    stuffing = build_pipeline(
        stuff=True, cache_enabled=False,
        compress_max_tokens=None, token_budget=None, **common,
    )
    full = build_pipeline(
        stuff=False, cache_enabled=True,
        compress_max_tokens=args.compress_max, token_budget=args.budget, **common,
    )
    base_rows = run_conversation(stuffing, turns, model, sleep=sleep)
    full_rows = run_conversation(full, turns, model, sleep=sleep)
    print_table("--- Arm A: stuffing, no cache, no compact ---", base_rows)
    print_table("--- Arm D: RAG + cache + compress + compact ---", full_rows)

    a, d = summarize(base_rows), summarize(full_rows)
    cost_savings = 1 - d["total_cost_usd"] / a["total_cost_usd"] if a["total_cost_usd"] else 0.0
    tok_savings = (
        1 - d["total_prompt_tokens"] / a["total_prompt_tokens"]
        if a["total_prompt_tokens"] else 0.0
    )
    print("\n=== Conversation summary ===")
    print(json.dumps({"stuffing": a, "full_pipeline": d}, indent=2))
    print(f"\nPrompt-token reduction:  {tok_savings * 100:.1f}%")
    print(f"Input cost reduction:    {cost_savings * 100:.1f}%")
    print(f"LLM calls avoided:       {a['llm_calls'] - d['llm_calls']} of {a['llm_calls']}")


if __name__ == "__main__":
    main()
