#!/usr/bin/env python3
"""
The Lab 4 cost experiment. One workload, eight harnesses, every call through the mesh.

    python run_experiment.py                  # all eight arms
    python run_experiment.py --arms A,D       # just these arms
    python run_experiment.py --arms H --tickets refund-pro,sla-credit
    python run_experiment.py --repeats 3      # every ticket three times (steadier success rates)
    python run_experiment.py --replay --fail-cost 3    # re-price the saved runs: a failed ticket now
                                                       # costs $3 to clean up. No model is called.

The workload is fixtures/tickets.json: twelve support tickets, graded on the
facts in the final reply. Eleven can be answered from the records and the
policy. One cannot: its tracking lookup never finishes.

The model, the tools, the playbook, and the tickets are identical in arms
A to G. Only the harness changes, one switch at a time:

  A  naive              nothing engineered
  B  + stable prefix    volatile text leaves the top of the prompt; cache breakpoints are set
  C  + capped tools     every tool result is projected, compressed, and capped
  D  + loop guard       a turn budget, a dollar budget, and "nothing new came back"
  E  + closing pass     every draft is re-derived by the agent before it is sent
  F  + judge gate       a chat model decides which drafts get the closing pass
  G  + decision gate    a decision model decides, at decision.accept_at
  H  open model         arm E's harness with the model swapped for the one on your machine

The number to watch is the last one: dollars per SOLVED ticket. A cheap
attempt that fails is not cheap.

Per-ticket results are written to results.json. Re-running some arms updates
those arms and keeps the rest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness import LINE_ITEMS, Harness, Run, check_ticket, cost_per_solved, ledger, load_tickets, solved
from mesh import log

HERE = Path(__file__).parent

ENGINEERED = {"stable_prefix": True, "cap_tools": True, "loop_guard": True}
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "A": ("naive", {}),
    "B": ("+ stable prefix", {"stable_prefix": True}),
    "C": ("+ capped tools", {"stable_prefix": True, "cap_tools": True}),
    "D": ("+ loop guard", {**ENGINEERED}),
    "E": ("+ closing pass", {**ENGINEERED, "verify": "always"}),
    "F": ("+ judge gate", {**ENGINEERED, "verify": "llm"}),
    "G": ("+ decision gate", {**ENGINEERED, "verify": "decision"}),
    "H": ("open model", {**ENGINEERED, "verify": "always", "tier": "local"}),
}


def run_arm(harness: Harness, arm: str, tickets: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    name, switches = ARMS[arm]
    # Arms A-G run on the agent tier from config.yaml. Arm H names its own.
    switches = {"tier": harness.config["harness"].get("tier", "mid"), "verify": "off", "reasoning": "off", "break_cache": False,
                "stable_prefix": False, "cap_tools": False, "loop_guard": False, **switches}
    log()
    log(f"Arm {arm}: {name}   [{harness.options(**switches).label()}]")
    runs: list[Run] = []
    rows: list[dict[str, Any]] = []
    for attempt in range(repeats):
        for ticket in tickets:
            run = harness.run(ticket["text"], tags=[f"arm-{arm}", ticket["id"]], **switches)
            checks = check_ticket(run.answer, ticket["checks"])
            ok = solved(run, ticket)
            runs.append(run)
            rows.append(
                {
                    "arm": arm,
                    "ticket": ticket["id"],
                    "attempt": attempt + 1,
                    "solved": ok,
                    "failed_checks": [n for n, passed in checks if not passed],
                    "turns": run.turns,
                    "cost_usd": run.cost_usd,
                    "api_usd": run.api_usd,
                    "local_usd": run.local_usd,
                    "gate_usd": run.gate_usd,
                    "check_usd": run.check_usd,
                    "latency_s": run.latency_s,
                    "prompt_tokens": run.prompt_tokens,
                    "cached_tokens": run.cached_tokens,
                    "cache_write_tokens": run.cache_write_tokens,
                    "completion_tokens": run.completion_tokens,
                    "closing_passes": len(run.drafts),
                    "stop_reason": run.stop_reason,
                    "tools": run.tool_log,
                    "events": run.events,
                    "answer": run.answer,
                }
            )
            extra = f"  closing pass" if run.drafts else ""
            log(
                f"  {ticket['id']:<16} {'solved' if ok else 'FAILED':<7} {run.turns:>2} turns  ${run.cost_usd:.4f}  "
                f"{run.latency_s:>5.1f}s  {run.stop_reason}{extra}"
            )
    n = len(runs)
    wins = sum(1 for r in rows if r["solved"])
    total = sum(r.cost_usd for r in runs)
    return {
        "arm": arm,
        "name": name,
        "label": harness.options(**switches).label(),
        "attempts": n,
        "solved": wins,
        "success": wins / n if n else 0.0,
        "total_usd": total,
        "attempt_usd": total / n if n else 0.0,
        "latency_s": sum(r.latency_s for r in runs) / n if n else 0.0,
        "closing_passes": sum(len(r.drafts) for r in runs),
        "gate_usd": sum(r.gate_usd for r in runs),
        "hardware_usd": sum(r.hardware_usd for r in runs),
        "line_items": ledger(harness.mesh, runs),
        "rows": rows,
    }


def print_summary(results: list[dict[str, Any]], fail_cost: float) -> None:
    log()
    log("=" * 118)
    log("Summary. Same tickets, same tools, same playbook. Only the harness changes (and, in arm H, the model).")
    log("=" * 118)
    header = f"  {'Arm':<20}{'Solved':>8}{'Success':>9}{'$/attempt':>11}{'sec':>7}{'$/solved':>10}"
    if fail_cost:
        header += f"{f'at ${fail_cost:g} fail':>14}"
    log(header + "    vs A")
    base = None
    for r in results:
        per_solved = cost_per_solved(r["attempt_usd"], r["success"])
        with_fail = cost_per_solved(r["attempt_usd"], r["success"], fail_cost)
        focus = with_fail if fail_cost else per_solved
        if r["arm"] == "A":
            base = focus
        line = (
            f"  {r['arm'] + '  ' + r['name']:<20}{str(r['solved']) + '/' + str(r['attempts']):>8}{r['success']:>9.0%}"
            f"{r['attempt_usd']:>11.4f}{r['latency_s']:>7.1f}{per_solved:>10.4f}"
        )
        if fail_cost:
            line += f"{with_fail:>14.4f}"
        if base and focus != float("inf"):
            line += f"{focus / base:>8.0%}"
        log(line)
    log()
    log("Where the money went (share of each arm's spend):")
    log(f"  {'Arm':<20}" + "".join(f"{name:>22}" for _, name in LINE_ITEMS) + f"{'Gate':>10}{'Your hardware':>16}")
    for r in results:
        total = r["total_usd"] or 1.0
        items = r["line_items"]
        # The four ledger lines are the agent's own calls. The gate (judge or decision model) and the
        # seconds a local agent spent on your machine are separate columns.
        line = f"  {r['arm'] + '  ' + r['name']:<20}" + "".join(f"{items[key] / total:>22.0%}" for key, _ in LINE_ITEMS)
        log(line + f"{r['gate_usd'] / total:>10.0%}{r['hardware_usd'] / total:>16.0%}")
    log()
    log("Closing passes run:  " + "   ".join(f"{r['arm']} {r['closing_passes']}/{r['attempts']}" for r in results))
    log("(In arm H the ledger lines are $0: an open model has no meter. The cost is seconds of your machine.)")
    log()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm letters (default: all)")
    parser.add_argument("--tickets", default="", help="comma-separated ticket ids (default: all twelve)")
    parser.add_argument("--repeats", type=int, default=1, help="times to run every ticket (default: 1)")
    parser.add_argument("--fail-cost", type=float, default=None, help="dollars to clean up one failed ticket (default: economics.fail_cost_usd)")
    parser.add_argument("--replay", action="store_true", help="re-print the summary from results.json without calling any model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    harness = Harness.from_config(args.config)
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
    tickets = load_tickets()
    if args.tickets:
        wanted = [t.strip() for t in args.tickets.split(",") if t.strip()]
        missing = [w for w in wanted if w not in {t["id"] for t in tickets}]
        if missing:
            parser.error(f"unknown ticket(s) {missing}; choose from {[t['id'] for t in tickets]}")
        tickets = [t for t in tickets if t["id"] in wanted]
    fail_cost = args.fail_cost if args.fail_cost is not None else float(harness.config.get("economics", {}).get("fail_cost_usd", 0.0))

    out = HERE / "results.json"
    saved = {r["arm"]: r for r in json.loads(out.read_text())["arms"]} if out.exists() else {}

    if args.replay:
        # No model is called: the saved runs are re-priced. A fail cost is arithmetic, not an experiment.
        results = [saved[arm] for arm in arms if arm in saved]
        if not results:
            parser.error("nothing to replay: run the experiment once first")
        log(f"Replaying {out.name} ({', '.join(r['arm'] for r in results)}). No model calls.")
        print_summary(results, fail_cost)
        return

    log(f"{len(tickets)} tickets x {args.repeats} attempt(s) x {len(arms)} arm(s), agent tier '{harness.config['harness'].get('tier', 'mid')}'")
    if "G" in arms:
        log(f"Arm G gates at decision.accept_at = {harness.config['decision'].get('accept_at')} (Part 3 chooses this number)")
    results = [run_arm(harness, arm, tickets, args.repeats) for arm in arms]
    print_summary(results, fail_cost)

    # Arms that were not re-run keep their last result, so `--arms G` after a full run updates one row.
    saved.update({r["arm"]: r for r in results})
    out.write_text(json.dumps({"fail_cost_usd": fail_cost, "arms": [saved[a] for a in ARMS if a in saved]}, indent=2))
    log(f"Per-ticket results: {out.name}")


if __name__ == "__main__":
    main()
