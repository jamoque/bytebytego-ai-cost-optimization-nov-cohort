#!/usr/bin/env python3
"""
Decisions inside the harness (PROVIDED).

A harness makes small, bounded decisions all day: does this draft need a
second look, is this result relevant, should a person see this. The usual
way to make one is to ask a chat model and parse what it writes. That pays
generation prices for what is really a yes/no, and the answer comes back as
prose with no number attached.

A decision model (TypeSafe's Jev is the hosted one; Kev, Tev1, Nimble and
others are open-weight models you can run yourself) takes the same content
plus typed questions and returns a probability for each permitted answer in
one forward pass. It writes nothing. That changes three things:

    cost        input tokens only, at a small fraction of a chat model's price
    latency     one forward pass, no generation
    control     the answer is a number, so YOU pick the threshold, and you
                can pick it from data (see calibrate.py)

It also takes something away. A decision model cannot tell you what is
wrong, only that the answer to your question is probably no. And it does
not reason: arithmetic, date comparisons, and multi-step inference are
where its probabilities are least trustworthy.

This file is the gate in front of the harness's closing pass. The closing
pass is one more model turn in which the agent re-derives its own figures.
It catches mistakes and it costs money, so the question for every draft is
whether it needs one:

    always     no gate: every draft gets the closing pass
    llm        a chat model judges the draft and writes a verdict
    decision   a decision model answers two yes/no questions about the draft
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mesh import Call, Decision, Mesh, StateTooLarge

JUDGE_SYSTEM = (
    "You review replies for the Northstar Commerce support desk before they are sent. You are given "
    "a customer ticket, the records and policy passages the agent looked up, and the agent's draft "
    "reply.\n\n"
    "Approve the draft unless you can point to one of two faults.\n"
    "INCOMPLETE: a question in the ticket has no specific final answer in the reply (an amount, a "
    "date, a time, a count, or a clear yes or no).\n"
    "UNSUPPORTED: an amount, date, time, or count in the reply differs from what the evidence gives "
    "when you work it out yourself, or the rule it depends on is nowhere in the evidence.\n\n"
    "Judge only the figures the ticket asks for. Do not reject a draft for wording, tone, formatting, "
    "process details the customer did not ask about, or anything you would merely have added.\n\n"
    "{conventions}\n\n"
    'Reply with JSON only, no prose: {{"complete": true or false, "supported": true or false, '
    '"problem": "one sentence naming the wrong figure and the right one, or the unanswered question; '
    'an empty string if you approve"}}'
)

# The same two checks, as typed questions. Names in backticks refer to fields of the state.
GATE_QUESTIONS: dict[str, dict[str, Any]] = {
    "complete": {
        "type": "noul",
        "instructions": (
            "Does `reply` give a specific final answer (an amount, a date, a time, a count, or a "
            "clear yes or no) to every question the customer asks in `ticket`?"
        ),
        "criteria": {
            "true": "Every question in the ticket gets its own specific final answer in the reply.",
            "false": "At least one question in the ticket is skipped, deferred, or answered without a specific value.",
        },
    },
    "supported": {
        "type": "noul",
        "instructions": "Is every amount, date, and number stated in `reply` supported by `evidence`?",
        "criteria": {
            "true": "Each figure in the reply appears in the evidence or follows directly from it.",
            "false": "The reply states a figure that the evidence does not support or contradicts.",
        },
    },
}

# A decision model cannot say what is wrong. This is all the agent can be told.
DOUBTS = {
    "complete": "it may not give a specific final answer to every question in the ticket.",
    "supported": "at least one amount, date, or number in it may not follow from the records and policy.",
}


@dataclass
class Verdict:
    ok: bool                                       # True: send the draft as it is
    problem: str                                   # what the agent is told when the draft is sent back
    path: str                                      # llm | decision
    scores: dict[str, float] = field(default_factory=dict)   # question name -> P(yes), from a decision model
    calls: list[Call] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    note: str = ""                                 # why the gate could not judge the draft, if it could not

    @property
    def cost_usd(self) -> float:
        return sum(c.total_usd for c in self.calls) + sum(d.total_usd for d in self.decisions)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls) + sum(d.latency_s for d in self.decisions)

    def summary(self) -> str:
        scores = " ".join(f"P({name})={p:.2f}" for name, p in self.scores.items())
        outcome = "passed" if self.ok else f"closing pass: {self.problem or self.note}"
        return f"{scores}  {outcome}".strip()


def conventions(playbook: str) -> str:
    """What a reviewer needs from the agent's playbook to redo the agent's sums: today's date and the counting rules."""
    today = re.search(r"^Today is.*$", playbook, re.M)
    counting = re.search(r"^## \d+\. Counting and arithmetic\n+(.*?)\n+## ", playbook, re.S | re.M)
    return "\n\n".join(m.group(m.lastindex or 0).strip() for m in (today, counting) if m)


class Gate:
    def __init__(self, mesh: Mesh, config: dict[str, Any], playbook: str = "") -> None:
        self.mesh = mesh
        self.config = config
        self.system = JUDGE_SYSTEM.format(conventions=conventions(playbook))

    def check(self, ticket: str, evidence: list[str], draft: str, *, mode: str, tags: list[str] | None = None) -> Verdict:
        if mode == "llm":
            return self.by_llm(ticket, evidence, draft, tags=tags)
        if mode == "decision":
            return self.by_decision(ticket, evidence, draft)
        raise ValueError(f"Unknown gate '{mode}'")

    # ---- a chat model as the gate ------------------------------------------ #
    def by_llm(self, ticket: str, evidence: list[str], draft: str, *, tags: list[str] | None = None) -> Verdict:
        """A chat model reads everything and writes a verdict. You pay for the reading and the writing."""
        cfg = self.config["judge"]
        tier = cfg.get("tier", "mid")
        params: dict[str, Any] = {}
        if self.mesh.tiers[tier].get("thinks_by_default"):
            params["reasoning_effort"] = self.config.get("reasoning", {}).get("off_value", "none")
        call = self.mesh.chat(
            tier,
            [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self._packet(ticket, evidence, draft)},
            ],
            purpose="judge",
            tags=[*(tags or []), "gate-llm"],
            max_tokens=int(cfg.get("max_tokens", 300)),
            **params,
        )
        verdict = parse_verdict(call.text)
        if verdict is None:
            # An unreadable verdict is not an approval. The draft gets the closing pass.
            return Verdict(False, "", "llm (unreadable verdict)", calls=[call])
        ok = bool(verdict.get("complete")) and bool(verdict.get("supported"))
        return Verdict(ok, "" if ok else str(verdict.get("problem") or "").strip(), "llm", calls=[call])

    # ---- a decision model as the gate --------------------------------------- #
    def score(self, ticket: str, evidence: list[str], draft: str) -> Decision:
        """One System One request: the draft, what it was written from, and two typed questions."""
        return self.mesh.decide(
            {"ticket": ticket, "evidence": "\n\n".join(evidence), "reply": draft},
            GATE_QUESTIONS,
            purpose="gate",
        )

    def by_decision(self, ticket: str, evidence: list[str], draft: str) -> Verdict:
        """Pass the draft if the decision model is confident enough on both questions.

        The draft is only as good as its weakest check, so the lowest P(yes)
        decides. `decision.accept_at` is the whole policy: at or above it the
        draft goes out, below it the agent takes a second look. Where that
        number should sit is an empirical question (calibrate.py).
        """
        accept_at = float(self.config["decision"].get("accept_at", 0.5))
        try:
            decision = self.score(ticket, evidence, draft)
        except StateTooLarge as exc:
            # The evidence does not fit the decision model's context window (uncapped tool results
            # usually do not). A gate that could not read the evidence cannot vouch for the draft.
            return Verdict(False, "", "decision", note=f"the decision model could not read this draft's evidence: {exc}")
        scores = {name: decision.p(name) for name in GATE_QUESTIONS}
        if confidence(scores) >= accept_at:
            return Verdict(True, "", "decision", scores, [], [decision])
        doubts = [DOUBTS[name] for name in GATE_QUESTIONS if scores[name] < accept_at]
        return Verdict(False, " Also, ".join(doubts), "decision", scores, [], [decision])

    @staticmethod
    def _packet(ticket: str, evidence: list[str], draft: str) -> str:
        looked_up = "\n\n".join(evidence) if evidence else "(the agent looked nothing up)"
        return f"TICKET:\n{ticket}\n\nEVIDENCE THE AGENT LOOKED UP:\n{looked_up}\n\nDRAFT REPLY:\n{draft}"


def confidence(scores: dict[str, float]) -> float:
    """One number for a draft: the lowest P(yes) across the questions asked about it."""
    return min(scores.values())


def parse_verdict(text: str) -> dict[str, Any] | None:
    """The judge's verdict: the last JSON object in its output that has both fields."""
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text or "")):   # tolerate fences and stray prose
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "complete" in data and "supported" in data:
            return data
    return None


# --------------------------------------------------------------------------- #
# Choosing a threshold from data (used by calibrate.py and Part 3).
# --------------------------------------------------------------------------- #
def risk_coverage(scored: list[tuple[float, bool]], accept_at: float) -> dict[str, float]:
    """What a threshold does to a set of labeled drafts.

    `scored` is [(the gate's confidence in the draft, whether the draft was
    actually good)]. Drafts at or above `accept_at` pass the gate and are
    sent as they are. The rest get the closing pass.

    coverage   the share of drafts that pass: closing passes you do not pay for
    risk       among drafts that pass, the share that were bad: errors you ship
    caught     the share of all bad drafts that the gate stopped
    """
    passed = [good for p, good in scored if p >= accept_at]
    bad_total = sum(1 for _, good in scored if not good)
    bad_passed = sum(1 for good in passed if not good)
    return {
        "accept_at": accept_at,
        "coverage": len(passed) / len(scored) if scored else 0.0,
        "risk": bad_passed / len(passed) if passed else 0.0,
        "caught": (bad_total - bad_passed) / bad_total if bad_total else 1.0,
        "passed": float(len(passed)),
        "bad_passed": float(bad_passed),
    }


def pick_threshold(scored: list[tuple[float, bool]], max_risk: float = 0.0) -> float:
    """The lowest `accept_at` whose passed drafts stay within `max_risk`, at this threshold and every higher one.

    Lower thresholds pass more drafts (more coverage, more savings) and let
    more bad ones through. This walks down from the most confident draft and
    stops just before the risk budget is first exceeded.
    """
    best = 1.0
    for p in sorted({p for p, _ in scored}, reverse=True):
        if risk_coverage(scored, p)["risk"] > max_risk:
            break
        best = p
    return best


__all__ = [
    "Gate",
    "Verdict",
    "GATE_QUESTIONS",
    "JUDGE_SYSTEM",
    "confidence",
    "parse_verdict",
    "risk_coverage",
    "pick_threshold",
]
