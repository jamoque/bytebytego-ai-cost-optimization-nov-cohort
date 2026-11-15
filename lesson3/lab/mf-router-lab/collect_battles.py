#!/usr/bin/env python3
"""
Collect preference battles for the Northstar desk (PROVIDED).

A preference router is trained on BATTLES: the same prompt answered by two
models, plus a verdict on which answer was better. RouteLLM's published
router learned from Chatbot Arena votes between dozens of models on public
chatbot traffic. This script builds the same kind of data for YOUR two
models on YOUR traffic, which is the only part of training a router that
costs real money.

`fixtures/battles.json` was produced by this script, so you do not need to
run it to do the exercise. Run it when you want more data, or to label a
different set of questions.

    # 1. Write candidate questions from the Lab 2 policy documents (mid tier).
    #    Anything too close to a graded question, an existing battle, or an
    #    earlier candidate is dropped. Writes questions.json.
    python collect_battles.py generate

    # 2. Answer each question with the weak and the strong tier, then have
    #    the strong tier judge whether the weak answer reaches the same conclusion.
    python collect_battles.py label

    # Label your own questions instead (a JSON list of strings):
    python collect_battles.py label --questions my_questions.json --out my_battles.json

    # Changed JUDGE_PROMPT? Re-judge the stored answers without paying for them again.
    python collect_battles.py rejudge

How a battle is judged
----------------------
Nobody votes here, and there is no answer key. Both tiers answer through
the same retrieval pipeline as Part 1 (`Router.answer`). The strong tier's
final "Answer:" line is treated as the reference, and a judge reads the
weak tier's whole answer against it:

    same conclusion       → winner "tie"       the weak answer was good enough
    different conclusion  → winner "model_b"   assume the strong tier was right

The second line is an assumption, not a measurement. On the graded
workload the strong tier was right far more often than the weak one, which
makes it a reasonable prior, but some of these labels are wrong. Label
noise is a property of every preference dataset, including the Arena.

Check your judge before you trust your labels. The first version of this
script used the mid tier as judge and compared only the two final lines.
It called identical answers different, and it punished the weak tier for
terse final lines under perfectly good answers. Both problems only showed
up by reading the verdicts.

Spend: step 2 costs one strong-tier answer plus one strong-tier verdict per
question (roughly $0.004 together at November 2026 prices). Labeling is
resumable: re-running skips prompts already in the output file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ROUTING_LAB = HERE.parent / "routing-lab"

# The hands-on lab is a dependency: same mesh client, same retrieval, same
# system prompt. Loaded by file path so nothing here can shadow it.
_spec = importlib.util.spec_from_file_location("routing", ROUTING_LAB / "routing.py")
routing = importlib.util.module_from_spec(_spec)
sys.modules["routing"] = routing
_spec.loader.exec_module(routing)

log = routing.log

LOOKUP_PROMPT = """Here is one of Northstar Commerce's support policies.

{doc}

Write {n} different questions a customer (or a customer's developer) might send to support that are answered directly by ONE sentence of this policy. Vary the phrasing: some terse, some chatty, some that describe a situation before asking. Do not repeat a fact more than twice.

Return only a JSON array of {n} strings."""

DERIVE_PROMPT = """Here is one of Northstar Commerce's support policies.

{doc}

Write {n} different customer questions that can only be answered by combining this policy with specifics the customer states in the question (amounts, dates, times of day, counts, plan names). Each must need at least two steps: arithmetic, comparing a number against a threshold, counting days or minutes, or applying an exception. Each must have exactly one correct final answer (an amount, a date or time, a duration, or yes/no) that follows from the policy and the question alone.

Vary the scenarios and the numbers. Include cases that fall just on either side of a threshold, and cases where an exception changes the answer. Do not include the answer.

Return only a JSON array of {n} strings."""

JUDGE_PROMPT = """Two support agents answered the same customer question from the same policy excerpts. Agent B's answer is the reference.

Question: {question}

Agent A's full answer:
{a}

Agent B's answer: {b}

Would the customer come away from Agent A's answer with the same conclusion as from Agent B's: the same amount, date, time, duration, yes/no, or instruction? Judge A's answer as a whole, not just its last line. Ignore wording, length, formatting, and extra detail on either side.

Reply SAME if they agree, or if both decline to answer. Reply DIFFERENT if the amounts, dates, times, or yes/no differ, if one answers and the other declines, or if A never states what B concludes.

Reply with exactly one word: SAME or DIFFERENT."""


def parse_string_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of a model reply (tolerates code fences)."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [re.sub(r"\s+", " ", s).strip() for s in items if isinstance(s, str) and s.strip()]


def near_duplicates(candidates: list[str], existing: list[str], threshold: float) -> list[bool]:
    """For each candidate: is it within `threshold` cosine of an existing text or an earlier candidate?"""
    lab2 = routing.lab2
    cand = lab2.embed(candidates)                    # MiniLM, unit-norm (Lab 2's embedder)
    kept: list[list[float]] = list(lab2.embed(existing)) if existing else []
    flags: list[bool] = []
    for vec in cand:
        dup = any(sum(a * b for a, b in zip(vec, other)) >= threshold for other in kept)
        flags.append(dup)
        if not dup:
            kept.append(vec)
    return flags


# --------------------------------------------------------------------------- #
# Step 1: candidate questions.
# --------------------------------------------------------------------------- #
def generate(args: argparse.Namespace) -> None:
    router = routing.Router.from_config()
    kb = sorted((routing.LAB2_DIR / "knowledge_base").glob("*.txt"))
    half = max(1, args.per_doc // 2)
    jobs: list[tuple[str, str, str]] = []            # (kind, source, prompt)
    for path in kb:
        doc = path.read_text().strip()
        jobs.append(("lookup", path.name, LOOKUP_PROMPT.format(doc=doc, n=half)))
        jobs.append(("derive", path.name, DERIVE_PROMPT.format(doc=doc, n=half)))

    def run(job: tuple[str, str, str]) -> list[dict[str, str]]:
        kind, source, prompt = job
        call = router.mesh.chat(
            args.tier,
            [{"role": "user", "content": prompt}],
            purpose="battle-questions",
            tags=["battles"],
            max_tokens=4000,
            temperature=0.8,
        )
        log(f"  {kind:<8} {source:<20} {len(parse_string_array(call.text)):>3} questions  ${call.cost_usd:.4f}")
        return [{"text": t, "kind": kind, "source": source, "cost": call.cost_usd} for t in parse_string_array(call.text)]

    log(f"Writing questions from {len(kb)} policy documents with the {args.tier} tier...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batches = list(pool.map(run, jobs))
    cost = sum(b[0]["cost"] for b in batches if b)
    rows = [{k: v for k, v in row.items() if k != "cost"} for batch in batches for row in batch]

    # A router evaluated on questions it trained on tells you nothing. Drop
    # anything too close to the graded workload, then anything already
    # battled, then repeats.
    existing = [q["text"] for q in json.loads((ROUTING_LAB / "fixtures" / "queries.json").read_text())]
    if Path(args.battles).exists():
        existing += [b["prompt"] for b in json.loads(Path(args.battles).read_text())]
    flags = near_duplicates([r["text"] for r in rows], existing, args.dedupe)
    kept = [r for r, dup in zip(rows, flags) if not dup]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=2) + "\n")
    log(
        f"\n{len(rows)} written, {len(rows) - len(kept)} dropped as near-duplicates "
        f"(cosine ≥ {args.dedupe} to a graded question, an existing battle, or an earlier one), {len(kept)} kept."
    )
    log(f"Generation cost ${cost:.2f}. Saved {out}")


# --------------------------------------------------------------------------- #
# Step 2: battles.
# --------------------------------------------------------------------------- #
def judge(router: Any, tier: str, question: str, a: str, b: str) -> tuple[str, float]:
    """(winner, judge cost). "tie" when answer A reaches reference B's conclusion, else "model_b"."""
    verdict = router.mesh.chat(
        tier,
        [{"role": "user", "content": JUDGE_PROMPT.format(question=question, a=a, b=b)}],
        purpose="battle-judge",
        tags=["battles"],
        max_tokens=5,
    )
    same = verdict.text.strip().upper().startswith("SAME")
    return ("tie" if same else "model_b"), verdict.cost_usd


def label(args: argparse.Namespace) -> None:
    router = routing.Router.from_config()
    tiers = router.config["tiers"]
    if not Path(args.questions).exists():
        log(f"{args.questions} not found. Run `python collect_battles.py generate` first, or pass --questions.")
        sys.exit(1)
    raw = json.loads(Path(args.questions).read_text())
    questions: list[dict[str, Any]] = [{"text": q} if isinstance(q, str) else dict(q) for q in raw]
    if args.limit:
        questions = questions[: args.limit]

    out = Path(args.out)
    battles: list[dict[str, Any]] = json.loads(out.read_text()) if out.exists() else []
    done = {b["prompt"] for b in battles}
    todo = [q for q in questions if q["text"] not in done]
    log(f"{len(questions)} questions, {len(done)} already labeled in {out.name}, {len(todo)} to go.")
    if not todo:
        return

    # Retrieval runs here, on one thread (the embedder and Chroma are local).
    prepared = []
    for q in todo:
        hits = router.retrieve(q["text"])
        prepared.append((q, router.context_for(q["text"], hits)))

    def battle(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        q, context = item
        weak = router.answer("weak", q["text"], context, tags=["battles"])
        strong = router.answer("strong", q["text"], context, tags=["battles"])
        a, b = weak.text, routing.final_line(strong.text)   # whole weak answer vs the strong conclusion
        winner, judge_cost = judge(router, args.judge, q["text"], a, b)
        return {
            "prompt": q["text"],
            "kind": q.get("kind", ""),
            "source": q.get("source", ""),
            "model_a": tiers["weak"]["alias"],
            "model_b": tiers["strong"]["alias"],
            "answer_a": a,
            "answer_b": b,
            "winner": winner,
            "cost_a": round(weak.cost_usd, 6),
            "cost_b": round(strong.cost_usd, 6),
            "judge_cost": round(judge_cost, 6),
        }

    spent = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(battle, prepared), 1):
            battles.append(row)
            spent += row["cost_a"] + row["cost_b"] + row["judge_cost"]
            if i % 10 == 0 or i == len(prepared):
                out.write_text(json.dumps(battles, indent=2) + "\n")
                strong_wins = sum(1 for b in battles if b["winner"] == "model_b")
                log(f"  {i:>4}/{len(prepared)}  strong wins so far: {strong_wins}/{len(battles)}  spent ${spent:.2f}")
    log(f"Saved {len(battles)} battles to {out}")


def rejudge(args: argparse.Namespace) -> None:
    """Re-run only the judge over stored answers (after editing JUDGE_PROMPT or --judge)."""
    router = routing.Router.from_config()
    out = Path(args.out)
    battles: list[dict[str, Any]] = json.loads(out.read_text())

    def one(row: dict[str, Any]) -> tuple[str, float]:
        return judge(router, args.judge, row["prompt"], row["answer_a"], row["answer_b"])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(one, battles))
    flipped = 0
    for row, (winner, cost) in zip(battles, verdicts):
        flipped += winner != row["winner"]
        row["winner"], row["judge_cost"] = winner, round(cost, 6)
    out.write_text(json.dumps(battles, indent=2) + "\n")
    strong_wins = sum(1 for b in battles if b["winner"] == "model_b")
    log(
        f"Re-judged {len(battles)} battles with the {args.judge} tier for ${sum(c for _, c in verdicts):.2f}: "
        f"{flipped} verdicts changed, strong now wins {strong_wins}/{len(battles)}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write candidate questions from the policy documents")
    g.add_argument("--per-doc", type=int, default=40, help="questions per policy document (half lookups, half derivations)")
    g.add_argument("--tier", default="mid", help="tier that writes the questions")
    g.add_argument("--dedupe", type=float, default=0.9, help="cosine at or above this counts as a duplicate")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--battles", default=str(HERE / "fixtures" / "battles.json"), help="existing battles to de-duplicate against")
    g.add_argument("--out", default=str(HERE / "questions.json"))
    g.set_defaults(func=generate)

    lab = sub.add_parser("label", help="answer with both tiers and judge")
    lab.add_argument("--questions", default=str(HERE / "questions.json"))
    lab.add_argument("--out", default=str(HERE / "fixtures" / "battles.json"))
    lab.add_argument("--judge", default="strong", help="tier that compares the two answers")
    lab.add_argument("--limit", type=int, default=0, help="label only the first N questions")
    lab.add_argument("--workers", type=int, default=4)
    lab.set_defaults(func=label)

    rj = sub.add_parser("rejudge", help="re-run only the judge over stored answers")
    rj.add_argument("--out", default=str(HERE / "fixtures" / "battles.json"))
    rj.add_argument("--judge", default="strong")
    rj.add_argument("--workers", type=int, default=4)
    rj.set_defaults(func=rejudge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
