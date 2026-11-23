#!/usr/bin/env python3
"""
Choose the gate's threshold from data (Part 3).

    python calibrate.py                   # score every labeled draft with the decision model
    python calibrate.py --max-risk 0.10   # accept 10% bad drafts among the ones the gate passes
    python calibrate.py --judge           # the same drafts through the chat-model judge (about $0.15)
    python calibrate.py --limit 20        # the first 20 drafts only (for a slow machine)
    python calibrate.py --rescore         # ignore cached scores and ask the model again

fixtures/decisions.json holds 66 first drafts the agent really wrote, each
with the evidence it was written from and a label: did the reply pass the
ticket's checks or not. The gate's job is to pass the good ones (no closing
pass, no extra cost) and stop the bad ones.

A decision model returns a probability, so the gate is a threshold on that
probability, and a threshold is a trade:

    coverage   the share of drafts that pass the gate: closing passes you do not pay for
    risk       among drafts that pass, the share that were bad: errors you ship

This script prints that trade for every threshold and suggests the lowest
one that stays inside a risk budget. Scores are cached in calibration.json,
so changing --max-risk does not ask the model again.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from decisions import confidence, pick_threshold, risk_coverage
from harness import FIXTURES, Harness
from mesh import log

HERE = Path(__file__).parent
CACHE = HERE / "calibration.json"


def load_cache() -> dict[str, Any]:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-risk", type=float, default=0.0, help="share of passed drafts allowed to be bad (default: 0)")
    parser.add_argument("--judge", action="store_true", help="also run the chat-model judge on the same drafts")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N drafts")
    parser.add_argument("--rescore", action="store_true", help="ignore calibration.json")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    harness = Harness.from_config(args.config)
    gate = harness.gate
    drafts = json.loads((FIXTURES / "decisions.json").read_text())["drafts"]
    if args.limit:
        drafts = drafts[: args.limit]
    good = sum(1 for d in drafts if d["good"])
    cache = {} if args.rescore else load_cache()

    # ---- the decision model ------------------------------------------------- #
    alias = harness.config["decision"]["alias"]
    scores: dict[str, Any] = cache.setdefault(f"decision:{alias}", {})
    todo = [d for d in drafts if d["id"] not in scores]
    if todo:
        log(f"Scoring {len(todo)} drafts with {alias} (two questions each, one request per draft)...")
    for i, d in enumerate(todo, start=1):
        decision = gate.score(d["ticket"], d["evidence"], d["reply"])
        scores[d["id"]] = {
            "complete": decision.p("complete"),
            "supported": decision.p("supported"),
            "input_tokens": decision.input_tokens,
            "latency_s": decision.latency_s,
            "cost_usd": decision.total_usd,
        }
        if i % 10 == 0 or i == len(todo):
            log(f"  {i}/{len(todo)}")
            CACHE.write_text(json.dumps(cache, indent=1))

    rows = [scores[d["id"]] for d in drafts]
    scored = [(confidence({"complete": r["complete"], "supported": r["supported"]}), d["good"]) for r, d in zip(rows, drafts)]
    n = len(drafts)
    log()
    log(f"Decision model {alias}: {n} drafts ({good} good, {n - good} bad)")
    log(
        f"  per draft: {sum(r['input_tokens'] for r in rows) / n:,.0f} input tokens, "
        f"{sum(r['latency_s'] for r in rows) / n:.1f}s, ${sum(r['cost_usd'] for r in rows) / n:.5f}"
    )
    log()
    log(f"  {'accept_at':>9}{'passes':>9}{'coverage':>10}{'bad passed':>12}{'risk':>7}{'bad caught':>12}")
    # Round down to three places, so the number you copy into config.yaml still passes the draft it was set by.
    picked = math.floor(pick_threshold(scored, args.max_risk) * 1000) / 1000
    for t in sorted({0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, picked}):
        rc = risk_coverage(scored, t)
        mark = "   <- suggested" if t == picked else ""
        log(
            f"  {t:>9.3f}{rc['passed']:>6.0f}/{n:<3}{rc['coverage']:>9.0%}{rc['bad_passed']:>12.0f}"
            f"{rc['risk']:>7.0%}{rc['caught']:>12.0%}{mark}"
        )
    rc = risk_coverage(scored, picked)
    log()
    log(f"  Lowest threshold whose passed drafts are at most {args.max_risk:.0%} bad: accept_at = {picked:.3f}")
    log(f"  At that threshold {rc['coverage']:.0%} of drafts skip the closing pass and {rc['caught']:.0%} of the bad ones are stopped.")
    log("  To use it: set decision.accept_at in config.yaml.")

    # ---- the chat-model judge, on the same drafts --------------------------- #
    if args.judge:
        tier = harness.config["judge"].get("tier", "mid")
        verdicts: dict[str, Any] = cache.setdefault(f"judge:{harness.mesh.alias(tier)}", {})
        todo = [d for d in drafts if d["id"] not in verdicts]
        if todo:
            log()
            log(f"Judging {len(todo)} drafts with {harness.mesh.alias(tier)}...")
        for i, d in enumerate(todo, start=1):
            verdict = gate.by_llm(d["ticket"], d["evidence"], d["reply"], tags=["calibrate"])
            verdicts[d["id"]] = {"ok": verdict.ok, "latency_s": verdict.latency_s, "cost_usd": verdict.cost_usd}
            if i % 10 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)}")
                CACHE.write_text(json.dumps(cache, indent=1))
        vrows = [verdicts[d["id"]] for d in drafts]
        # A judge has no dial: it says yes or no. That is one point on the table above.
        as_scores = [(1.0 if v["ok"] else 0.0, d["good"]) for v, d in zip(vrows, drafts)]
        rc = risk_coverage(as_scores, 0.5)
        log()
        log(f"Judge {harness.mesh.alias(tier)}: the same {n} drafts")
        log(f"  per draft: {sum(v['latency_s'] for v in vrows) / n:.1f}s, ${sum(v['cost_usd'] for v in vrows) / n:.5f}")
        log()
        log(f"  {'':>9}{'passes':>9}{'coverage':>10}{'bad passed':>12}{'risk':>7}{'bad caught':>12}")
        log(f"  {'verdict':>9}{rc['passed']:>6.0f}/{n:<3}{rc['coverage']:>9.0%}{rc['bad_passed']:>12.0f}{rc['risk']:>7.0%}{rc['caught']:>12.0%}")
        log("  Find the row above with the same coverage and compare the risk, then the price per draft.")

    CACHE.write_text(json.dumps(cache, indent=1))
    log()


if __name__ == "__main__":
    main()
