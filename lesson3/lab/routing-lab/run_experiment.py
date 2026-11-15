#!/usr/bin/env python3
"""
The Lab 3 cost experiment. Three workloads, every call through the mesh.

    python run_experiment.py                 # all three workloads
    python run_experiment.py --routing       # Workload 1 only
    python run_experiment.py --agents        # Workload 2 only
    python run_experiment.py --memory        # Workload 3 only
    python run_experiment.py --routing --arms B,E    # just these arms

Workload 1: model routing. 32 graded questions (fixtures/queries.json), each
with retrieval from the Lab 2 index. The only thing that changes between
arms is which model reads the context.
  A  all-strong       every question to the strong tier (quality ceiling, cost ceiling)
  B  all-weak         every question to the weak tier (cost floor)
  C  gateway rules    the mesh's rule-based router picks (mesh-auto)
  D  vector routes    nearest route by embedding; the route names the tier
  E  cascade          weak first; a verifier gate decides whether to escalate
  F  RouteLLM         preference router; above the threshold goes to strong

Workload 2: agent architectures. 4 tickets (fixtures/tickets.json), graded
on the facts in the final reply.
  react | plan | team-llm | team-vector

Workload 3: memory. One 10-turn chat (fixtures/session.json) with two
callback turns.
  full | compact | memory

Quality is graded with regexes, so every arm reports cost AND accuracy.
A full run makes a few hundred model calls and takes several minutes.
Per-item results are written to results.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agents import Desk, Session, check_ticket
from routing import Router, final_line, grade, load_config, log

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"

ROUTING_ARMS = {
    "A": ("all-strong", "strong"),
    "B": ("all-weak", "weak"),
    "C": ("gateway rules", "gateway"),
    "D": ("vector routes", "deterministic"),
    "E": ("cascade", "cascade"),
    "F": ("RouteLLM", "preference"),
}
AGENT_ARMS = {
    "react": ("react", None),
    "plan": ("plan", None),
    "team-llm": ("team", "llm"),
    "team-vector": ("team", "vector"),
}
MEMORY_ARMS = ("full", "compact", "memory")
TIER_LETTER = {"weak": "w", "mid": "m", "strong": "S"}


def pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


# --------------------------------------------------------------------------- #
# Workload 1: model routing
# --------------------------------------------------------------------------- #
def run_routing(config: dict, arms: list[str]) -> dict:
    queries = json.loads((FIXTURES / "queries.json").read_text())
    router = Router(config)
    hard = sum(1 for q in queries if q["difficulty"] == "hard")
    log(f"\n=== Workload 1: model routing ({len(queries)} questions: {len(queries) - hard} easy, {hard} hard) ===")
    log(f"  tiers: " + ", ".join(f"{t}={spec['alias']}" for t, spec in config["tiers"].items()))
    log(f"  cascade: ladder={config['cascade']['ladder']} gate={config['cascade'].get('gate_tier', 'weak')} threshold={config['cascade']['threshold']}")
    weights = f"your checkpoint {config['preference']['checkpoint']}" if router.preference.is_custom else "published weights"
    log(f"  RouteLLM threshold: {config['preference']['threshold']} ({weights})")

    rows: dict[str, list[dict]] = {}
    for arm in arms:
        name, strategy = ROUTING_ARMS[arm]
        log(f"\n--- Arm {arm}: {name} ---")
        rows[arm] = []
        for q in queries:
            result = router.ask(q["text"], strategy=strategy, tags=[f"arm-{arm}"])
            ok = grade(result.answer, q["expect"], q.get("reject"))
            rows[arm].append(
                {
                    "id": q["id"],
                    "difficulty": q["difficulty"],
                    "correct": ok,
                    "tier": result.tier,
                    "reason": result.reason,
                    "llm_calls": result.llm_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                    "latency_s": result.latency_s,
                    "answer": final_line(result.answer),
                }
            )
            log(f"  {q['id']:<11} {'ok   ' if ok else 'WRONG'} {result.tier:<6} ${result.cost_usd:.5f}  {result.reason}")

    # One row per question, one column per arm: the tier that answered, and whether it was right.
    log("\nWho answered each question (w=weak m=mid S=strong, ✗ = wrong answer)")
    log(f"  {'question':<11} " + " ".join(f"{arm:>3}" for arm in arms))
    for i, q in enumerate(queries):
        cells = []
        for arm in arms:
            r = rows[arm][i]
            cells.append(f"{TIER_LETTER.get(r['tier'], '?')}{' ' if r['correct'] else '✗'}".rjust(3))
        log(f"  {q['id']:<11} " + " ".join(cells))

    summary: dict[str, dict] = {}
    base_cost = sum(r["cost_usd"] for r in rows["A"]) if "A" in rows else None
    log("\n=== Routing summary ===")
    header = f"  {'arm':<17} {'correct':>8} {'easy':>6} {'hard':>6} {'to strong':>10} {'calls':>6} {'in tok':>8} {'out tok':>8} {'cost $':>9} {'vs A':>6}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for arm in arms:
        rs = rows[arm]
        cost = sum(r["cost_usd"] for r in rs)
        easy = [r for r in rs if r["difficulty"] == "easy"]
        hard_rows = [r for r in rs if r["difficulty"] == "hard"]
        s = {
            "correct": sum(r["correct"] for r in rs),
            "questions": len(rs),
            "easy_correct": f"{sum(r['correct'] for r in easy)}/{len(easy)}",
            "hard_correct": f"{sum(r['correct'] for r in hard_rows)}/{len(hard_rows)}",
            "answered_by_strong": sum(1 for r in rs if r["tier"] == "strong"),
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "completion_tokens": sum(r["completion_tokens"] for r in rs),
            "cost_usd": round(cost, 5),
            "cost_vs_all_strong": round(cost / base_cost, 3) if base_cost else None,
            "mean_latency_s": round(sum(r["latency_s"] for r in rs) / len(rs), 2),
        }
        summary[f"{arm} {ROUTING_ARMS[arm][0]}"] = s
        log(
            f"  {arm + ' ' + ROUTING_ARMS[arm][0]:<17} {s['correct']:>5}/{s['questions']:<2} {s['easy_correct']:>6} {s['hard_correct']:>6} "
            f"{pct(s['answered_by_strong'], len(rs)):>10} {s['llm_calls']:>6} {s['prompt_tokens']:>8} {s['completion_tokens']:>8} "
            f"{s['cost_usd']:>9.4f} {pct(cost, base_cost) if base_cost else '-':>6}"
        )
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# Workload 2: agent architectures
# --------------------------------------------------------------------------- #
def run_agents(config: dict, arms: list[str]) -> dict:
    tickets = json.loads((FIXTURES / "tickets.json").read_text())["tickets"]
    desk = Desk(config)
    total_checks = sum(len(t["checks"]) for t in tickets)
    log(f"\n=== Workload 2: agent architectures ({len(tickets)} tickets, {total_checks} graded facts) ===")
    log("  specialists: " + ", ".join(f"{name}={spec.tier}" for name, spec in desk.specialists().items()))

    rows: dict[str, list[dict]] = {}
    for arm in arms:
        mode, supervisor = AGENT_ARMS[arm]
        log(f"\n--- {arm} ---")
        rows[arm] = []
        for t in tickets:
            result = desk.handle(t["text"], mode=mode, supervisor=supervisor, tags=[f"arm-{arm}"])
            checks = check_ticket(result.answer, t["checks"])
            passed = sum(ok for _, ok in checks)
            by_tier = {k: round(v, 5) for k, v in result.cost_by_tier().items()}
            rows[arm].append(
                {
                    "id": t["id"],
                    "issues": t["issues"],
                    "facts_passed": passed,
                    "facts": len(checks),
                    "failed": [name for name, ok in checks if not ok],
                    "llm_calls": result.llm_calls,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost_usd": result.cost_usd,
                    "cost_by_tier": by_tier,
                    "latency_s": result.latency_s,
                    "agents": [f"{run.agent}[{run.tier}] x{len(run.calls)}" for run in result.runs],
                    "notes": result.notes,
                    "answer": result.answer,
                }
            )
            log(
                f"  {t['id']:<14} facts {passed}/{len(checks)}  calls {result.llm_calls:>2}  in {result.prompt_tokens:>6}  "
                f"out {result.completion_tokens:>5}  ${result.cost_usd:.4f}  {by_tier}"
            )
            log(f"      agents: {', '.join(rows[arm][-1]['agents'])}")
            for name, ok in checks:
                if not ok:
                    log(f"      MISSED: {name}")

    summary: dict[str, dict] = {}
    base_cost = sum(r["cost_usd"] for r in rows["react"]) if "react" in rows else None
    log("\n=== Agent summary ===")
    header = f"  {'arm':<12} {'facts':>7} {'calls':>6} {'in tok':>8} {'out tok':>8} {'strong $':>9} {'other $':>9} {'cost $':>9} {'vs react':>9}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for arm in arms:
        rs = rows[arm]
        cost = sum(r["cost_usd"] for r in rs)
        strong = sum(r["cost_by_tier"].get("strong", 0.0) for r in rs)
        s = {
            "facts_passed": sum(r["facts_passed"] for r in rs),
            "facts": sum(r["facts"] for r in rs),
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
            "completion_tokens": sum(r["completion_tokens"] for r in rs),
            "strong_cost_usd": round(strong, 5),
            "cost_usd": round(cost, 5),
            "cost_vs_react": round(cost / base_cost, 3) if base_cost else None,
        }
        summary[arm] = s
        log(
            f"  {arm:<12} {s['facts_passed']:>4}/{s['facts']:<2} {s['llm_calls']:>6} {s['prompt_tokens']:>8} {s['completion_tokens']:>8} "
            f"{strong:>9.4f} {max(cost - strong, 0.0):>9.4f} {cost:>9.4f} {pct(cost, base_cost) if base_cost else '-':>9}"
        )
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- #
# Workload 3: memory
# --------------------------------------------------------------------------- #
def run_memory(config: dict, arms: list[str]) -> dict:
    turns = json.loads((FIXTURES / "session.json").read_text())["turns"]
    callbacks = sum(1 for t in turns if t.get("expect"))
    log(f"\n=== Workload 3: memory ({len(turns)}-turn chat, {callbacks} callback turns) ===")
    log(f"  token budget {config['memory']['token_budget']}, tier {config['memory']['tier']}")

    rows: dict[str, list[dict]] = {}
    for arm in arms:
        log(f"\n--- {arm} ---")
        log(f"  {'turn':>4} {'prompt':>7} {'history':>8} {'calls':>6} {'cost $':>9}  result")
        session = Session(config, mode=arm)
        rows[arm] = []
        for i, t in enumerate(turns, start=1):
            turn = session.send(t["text"], tags=[f"arm-{arm}"])
            passed = None
            if t.get("expect"):
                passed = all(re.search(p, turn.answer, re.I) for p in t["expect"])
            rows[arm].append(
                {
                    "turn": i,
                    "callback_passed": passed,
                    "answer_prompt_tokens": turn.calls[0].prompt_tokens,
                    "history_messages_sent": turn.history_messages_sent,
                    "llm_calls": len(turn.calls),
                    "cost_usd": turn.cost_usd,
                    "recalled": turn.recalled,
                    "remembered": turn.remembered,
                    "answer": turn.answer,
                }
            )
            mark = "" if passed is None else ("callback ok" if passed else "callback FAILED")
            log(
                f"  {i:>4} {turn.calls[0].prompt_tokens:>7} {turn.history_messages_sent:>8} {len(turn.calls):>6} "
                f"{turn.cost_usd:>9.6f}  {mark}"
            )
            if passed is not None:
                log(f"        “{turn.answer[:110]}”")

    summary: dict[str, dict] = {}
    base_cost = sum(r["cost_usd"] for r in rows["full"]) if "full" in rows else None
    log("\n=== Memory summary ===")
    header = f"  {'arm':<8} {'callbacks':>10} {'calls':>6} {'turn 1 prompt':>14} {'turn 10 prompt':>15} {'cost $':>10} {'vs full':>8}"
    log(header)
    log("  " + "-" * (len(header) - 2))
    for arm in arms:
        rs = rows[arm]
        cost = sum(r["cost_usd"] for r in rs)
        s = {
            "callbacks_passed": sum(1 for r in rs if r["callback_passed"]),
            "callbacks": callbacks,
            "llm_calls": sum(r["llm_calls"] for r in rs),
            "first_turn_prompt_tokens": rs[0]["answer_prompt_tokens"],
            "last_turn_prompt_tokens": rs[-1]["answer_prompt_tokens"],
            "cost_usd": round(cost, 6),
            "cost_vs_full": round(cost / base_cost, 3) if base_cost else None,
        }
        summary[arm] = s
        log(
            f"  {arm:<8} {s['callbacks_passed']:>7}/{callbacks:<2} {s['llm_calls']:>6} {s['first_turn_prompt_tokens']:>14} "
            f"{s['last_turn_prompt_tokens']:>15} {cost:>10.5f} {pct(cost, base_cost) if base_cost else '-':>8}"
        )
    return {"summary": summary, "rows": rows}


def pick(requested: str | None, available: list[str], label: str) -> list[str]:
    if not requested:
        return available
    wanted = [a.strip() for a in requested.split(",") if a.strip()]
    chosen = [a for a in available if a in wanted or a.upper() in [w.upper() for w in wanted]]
    if not chosen:
        raise SystemExit(f"--arms matched no {label} arm. Choose from: {', '.join(available)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routing", action="store_true", help="Workload 1: model routing")
    parser.add_argument("--agents", action="store_true", help="Workload 2: agent architectures")
    parser.add_argument("--memory", action="store_true", help="Workload 3: memory")
    parser.add_argument("--arms", default=None, help="comma-separated arms, e.g. A,B,E or react,plan or full,memory")
    args = parser.parse_args()
    run_all = not (args.routing or args.agents or args.memory)
    if args.arms and run_all:
        raise SystemExit("--arms needs exactly one of --routing, --agents, --memory")

    config = load_config()
    log("Northstar routing experiment")
    log(f"  Mesh: {config['mesh']['base_url']}")

    out: dict[str, dict] = {}
    if run_all or args.routing:
        out["routing"] = run_routing(config, pick(args.arms if args.routing else None, list(ROUTING_ARMS), "routing"))
    if run_all or args.agents:
        out["agents"] = run_agents(config, pick(args.arms if args.agents else None, list(AGENT_ARMS), "agent"))
    if run_all or args.memory:
        out["memory"] = run_memory(config, pick(args.arms if args.memory else None, list(MEMORY_ARMS), "memory"))

    (HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    log("\nPer-item results written to results.json")


if __name__ == "__main__":
    main()
