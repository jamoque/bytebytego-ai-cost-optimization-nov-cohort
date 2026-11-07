#!/usr/bin/env python3
"""
Four-arm cost experiment over the Northstar knowledge base.

Every request goes through the Lab 1 LiteLLM mesh. The gateway decides
hit vs miss (Qdrant semantic cache). `semantic_cache.enabled` only
controls whether this client sends `cache.no-cache`.

    python run_experiment.py              # queries + conversation
    python run_experiment.py --queries    # Q&A workload only
    python run_experiment.py --conversation

Arms:
  A  stuffing          entire corpus in the prompt, no cache, no compact
  B  RAG only          top-k chunks
  C  RAG + cache       top-k + semantic cache (paraphrases become hits)
  D  full pipeline     RAG + cache + compression + compaction
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pipeline import Message, Pipeline, gateway_similarity_threshold, load_config, log

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"


def load_queries() -> list[dict]:
    return json.loads((FIXTURES / "queries.json").read_text())


def load_conversation() -> list[str]:
    return json.loads((FIXTURES / "conversation.json").read_text())


def arm_config(base: dict, arm: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg["retriever"]["enabled"] = True
    if arm == "A":
        cfg["retriever"]["stuff"] = True
        cfg["retriever"]["k"] = 999
        cfg["retriever"]["min_score"] = 0.0
        cfg["semantic_cache"]["enabled"] = False
        cfg["compression"]["enabled"] = False
        cfg["compaction"]["enabled"] = False
    elif arm == "B":
        cfg["retriever"]["stuff"] = False
        cfg["semantic_cache"]["enabled"] = False
        cfg["compression"]["enabled"] = False
        cfg["compaction"]["enabled"] = False
    elif arm == "C":
        cfg["retriever"]["stuff"] = False
        cfg["semantic_cache"]["enabled"] = True
        cfg["compression"]["enabled"] = False
        cfg["compaction"]["enabled"] = False
    elif arm == "D":
        cfg["retriever"]["stuff"] = False
        cfg["semantic_cache"]["enabled"] = True
        cfg["compression"]["enabled"] = True
        cfg["compaction"]["enabled"] = True
    else:
        raise ValueError(arm)
    cfg.setdefault("mesh", {})
    cfg["mesh"]["session_id"] = f"lab2-arm-{arm}"
    return cfg


ARM_NAMES = {
    "A": "stuffing (whole corpus)",
    "B": "RAG only",
    "C": "RAG + semantic cache",
    "D": "full pipeline (RAG + cache + compress + compact)",
}


def print_table(title: str, rows: list[dict]) -> None:
    log(f"\n{title}")
    header = (
        f"{'#':>3} {'Hit':>4} {'Sim':>6} {'Retr':>6} {'Comp':>6} {'Prompt':>7} "
        f"{'Out':>5} {'LLM':>4} {'Cost$':>10}"
    )
    log(header)
    log("-" * len(header))
    for i, row in enumerate(rows, start=1):
        r = row["result"]
        sim = f"{r.semantic_similarity:.2f}" if r.semantic_similarity is not None else "-"
        log(
            f"{i:>3} {'yes' if r.cache_hit else 'no':>4} {sim:>6} "
            f"{r.retrieved_tokens:>6} {r.compressed_tokens:>6} "
            f"{r.prompt_tokens:>7} {r.completion_tokens:>5} "
            f"{r.llm_calls:>4} {r.estimated_cost_usd:>10.6f}"
        )


def summarize(rows: list[dict]) -> dict:
    return {
        "llm_calls": sum(r["result"].llm_calls for r in rows),
        "cache_hits": sum(1 for r in rows if r["result"].cache_hit),
        "total_prompt_tokens": sum(r["result"].prompt_tokens for r in rows),
        "total_completion_tokens": sum(r["result"].completion_tokens for r in rows),
        "provider_calls": sum(r["result"].llm_calls for r in rows),
        "gateway_requests": len(rows),
        "total_cost_usd": round(sum(r["result"].estimated_cost_usd for r in rows), 6),
        "mean_prompt_tokens": round(
            sum(r["result"].prompt_tokens for r in rows) / max(len(rows), 1), 1
        ),
    }


def run_queries(pipe: Pipeline, queries: list[dict], *, arm: str) -> list[dict]:
    pipe.reset_cache()
    rows = []
    for q in queries:
        result = pipe.ask(q["text"], tags=[f"arm-{arm}"])
        rows.append({"id": q["id"], "text": q["text"], "result": result})
    return rows


def run_conversation(pipe: Pipeline, turns: list[str], *, arm: str) -> list[dict]:
    pipe.reset_cache()
    history: list[Message] = []
    rows = []
    for text in turns:
        result = pipe.ask(text, history=history, tags=[f"arm-{arm}"])
        rows.append({"text": text, "result": result})
        history.append(Message("user", text))
        history.append(Message("assistant", result.answer))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", action="store_true")
    parser.add_argument("--conversation", action="store_true")
    args = parser.parse_args()
    run_both = not args.queries and not args.conversation

    base = load_config()
    queries = load_queries()
    turns = load_conversation()

    log("Northstar cost experiment")
    log(f"  Mesh:                   {base['mesh']['base_url']}  model={base['mesh']['model']}")
    log(f"  Query workload:         {len(queries)} questions")
    log(f"  Conversation turns:     {len(turns)}")
    log(f"  Retriever k:            {base['retriever']['k']}")
    log(f"  Cache threshold:        {gateway_similarity_threshold()} (mesh_config.yaml)")
    log(f"  Compress max tokens:    {base['compression']['max_tokens']}")
    log(f"  Compaction budget:      {base['compaction']['token_budget']}")

    if run_both or args.queries:
        log("\n=== Workload 1: Q&A queries (empty history, paraphrases included) ===")
        q_summaries = {}
        for arm, name in ARM_NAMES.items():
            pipe = Pipeline(arm_config(base, arm))
            if pipe.indexed_chunk_count() == 0:
                log("Vector DB is empty. Run:  python index_kb.py")
                return
            rows = run_queries(pipe, queries, arm=arm)
            print_table(f"--- Arm {arm}: {name} ---", rows)
            q_summaries[f"{arm} {name}"] = summarize(rows)
        log("\n=== Query-workload summary ===")
        log(json.dumps(q_summaries, indent=2))

    if run_both or args.conversation:
        log("\n=== Workload 2: multi-turn conversation (history grows) ===")
        c_summaries = {}
        for arm, name in ARM_NAMES.items():
            pipe = Pipeline(arm_config(base, arm))
            rows = run_conversation(pipe, turns, arm=arm)
            print_table(f"--- Arm {arm}: {name} ---", rows)
            c_summaries[f"{arm} {name}"] = summarize(rows)
        log("\n=== Conversation summary ===")
        log(json.dumps(c_summaries, indent=2))

        a = c_summaries[f"A {ARM_NAMES['A']}"]
        d = c_summaries[f"D {ARM_NAMES['D']}"]
        tok = 1 - d["total_prompt_tokens"] / a["total_prompt_tokens"]
        cost = 1 - d["total_cost_usd"] / a["total_cost_usd"] if a["total_cost_usd"] else 0.0
        log(f"\nPrompt-token reduction (A → D):  {tok * 100:.1f}%")
        log(f"Input cost reduction (A → D):    {cost * 100:.1f}%")
        log(f"Provider calls avoided (A → D):  {a['provider_calls'] - d['provider_calls']} of {a['provider_calls']}")
        log(f"Gateway requests (A / D):        {a['gateway_requests']} / {d['gateway_requests']}")


if __name__ == "__main__":
    main()
