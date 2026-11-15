#!/usr/bin/env python3
"""
Northstar support agents (PROVIDED).

This is the working system you observe in Parts 2, 3, and 4. You configure
it through `config.yaml` (`agents:` and `memory:`); you do not have to
implement it.

Part 1 routed a QUESTION to a model. A ticket is harder: it can hold
several issues, and answering one takes tool calls, not a single prompt.
Three ways to spend money on it:

    react   one strong agent, all six tools, the whole KB, ONE growing context.
            Every tool result is re-sent on every later step.

    plan    a strong planner writes the steps once. Each step runs in a FRESH
            context on a cheap generalist worker. A cheap model stitches the
            results together.

    team    the same decomposition, but each step is ROUTED to a specialist
            from fixtures/routes.json: its own model tier, two or three tools,
            and only its department's slice of the KB. The supervisor that
            does the routing is either a model (llm) or Part 1's vector
            router (vector, zero model calls).

And for a long chat (Part 2.2), three ways to carry the past:

    full      send the whole conversation every turn
    compact   Lab 2's compaction: keep what fits the budget, drop the rest
    memory    compaction plus a small fact store that survives it

Every model call goes through `routing.Mesh`, so it is traced, tagged, and
priced by the Lab 1 gateway exactly like the calls in Part 1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from routing import HERE, Call, Mesh, VectorRouter, lab2, load_config, load_routes, log, retrieve, token_count

AGENT_SYSTEM = (
    "You are {role} at Northstar Commerce support. Use your tools to look up "
    "policy and records; never guess a policy, a number, or a record. Call one "
    "tool at a time. When you have what you need, reply to the customer in plain "
    "text (no markdown, no tables, no headings), in at most six sentences, and "
    "state each final amount, date, or yes/no plainly. Answer only what was asked."
)

PLANNER_SYSTEM = (
    "You are the planner for Northstar Commerce support. Split the customer's "
    "ticket into the smallest number of steps that fully resolves it: one step "
    "per distinct issue. You do not answer the ticket yourself.\n\n"
    "The worker for a step sees ONLY that step's task text, so copy into it "
    "every identifier, amount, date, and detail the worker will need.\n\n"
    "{roster}"
    "Reply with JSON only, no prose:\n"
    '{{"steps": [{{"id": 1, "task": "...", {agent_field}"depends_on": []}}]}}\n'
    "Use depends_on only when a step needs an earlier step's result."
)

SYNTH_SYSTEM = (
    "You are writing the final reply to a Northstar Commerce customer. Combine "
    "the findings below into one short reply that addresses every issue in the "
    "ticket, in the order the customer raised them. Copy every amount, date, "
    "and number exactly as found. Do not add facts."
)

SESSION_SYSTEM = (
    "You are a Northstar Commerce support agent in a live chat. Answer using the "
    "policy context, what you know about this customer, and the conversation so "
    "far. If you do not have a piece of information, say so. Be concise."
)

EXTRACT_SYSTEM = (
    "You maintain the memory of a customer-support chat. From the exchange below, "
    "extract facts worth remembering for the rest of the conversation: identifiers "
    "(order numbers, account IDs, emails), who the customer is, plan and price "
    "details, any amount or date the agent quoted, and decisions made. Do not "
    "store general policy. Each fact must be a short sentence that stands on its "
    "own. Reply with a JSON array of strings only; reply [] if there is nothing new."
)


# --------------------------------------------------------------------------- #
# Tools. Read-only lookups over fixtures/crm.json plus the Lab 2 index.
# --------------------------------------------------------------------------- #
class Toolbox:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        spec = json.loads((HERE / "fixtures" / "tools.json").read_text())
        self._schemas = {t["function"]["name"]: t for t in spec["tools"]}
        self._crm = json.loads((HERE / "fixtures" / "crm.json").read_text())

    @property
    def names(self) -> list[str]:
        return list(self._schemas)

    def schemas(self, names: list[str]) -> list[dict]:
        return [self._schemas[n] for n in names]

    def call(self, name: str, arguments: str, *, sources: list[str] | None = None) -> str:
        """Run one tool call. Always returns a string; errors are returned, not raised,
        so the model can read them and recover."""
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments were not valid JSON"})
        if name == "search_kb":
            hits = retrieve(
                str(args.get("query", "")),
                k=int(self.config["agents"].get("search_k", 3)),
                min_score=float(self.config["rag"].get("min_score", 0.0)),
                sources=sources,  # a specialist searches only its department's documents
            )
            if not hits:
                return "No matching policy passages."
            return "\n\n".join(f"[{h.source}] {h.text}" for h in hits)
        if name == "get_order":
            return self._lookup("orders", str(args.get("order_id", "")).upper())
        if name == "get_account":
            return self._lookup("accounts", str(args.get("account_id", "")).upper())
        if name == "get_api_usage":
            return self._lookup("api_usage", str(args.get("account_id", "")).upper())
        if name == "get_login_status":
            return self._lookup("logins", str(args.get("email", "")).lower())
        if name == "get_uptime":
            months = self._crm["uptime"].get(str(args.get("account_id", "")).upper())
            if months is None:
                return json.dumps({"error": "account not found"})
            record = months.get(str(args.get("month", "")))
            return json.dumps(record or {"error": "no data for that month", "available_months": sorted(months)})
        return json.dumps({"error": f"unknown tool {name}"})

    def _lookup(self, table: str, key: str) -> str:
        record = self._crm[table].get(key)
        return json.dumps(record if record is not None else {"error": f"{key or '(empty)'} not found"})


# --------------------------------------------------------------------------- #
# One agent = a model tier + a tool list + a slice of the KB.
# --------------------------------------------------------------------------- #
@dataclass
class AgentSpec:
    name: str
    tier: str
    tools: list[str]
    sources: list[str] | None      # None = the whole knowledge base
    description: str


@dataclass
class AgentRun:
    agent: str
    tier: str
    task: str
    answer: str
    calls: list[Call]
    tool_log: list[str]            # "get_order({...})" in call order
    hit_cap: bool = False

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)


def run_agent(
    mesh: Mesh,
    toolbox: Toolbox,
    spec: AgentSpec,
    task: str,
    *,
    config: dict[str, Any],
    tags: list[str] | None = None,
) -> AgentRun:
    """The tool loop: call the model, run the tools it asks for, append the
    results, call again. Stops when the model answers without a tool call.

    The cost shape is in `messages`: nothing is ever removed, so call N pays
    for the system prompt, the tool schemas, and every earlier tool result.
    """
    cfg = config["agents"]
    tools = toolbox.schemas(spec.tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM.format(role=spec.description)},
        {"role": "user", "content": task},
    ]
    calls: list[Call] = []
    tool_log: list[str] = []
    max_steps = int(cfg.get("max_steps", 8))
    max_tokens = int(cfg.get("max_tokens", 500))
    purpose = f"agent-{spec.name}"
    tags = [f"agent-{spec.name}", *(tags or [])]

    for _ in range(max_steps):
        call = mesh.chat(spec.tier, messages, purpose=purpose, tags=tags, tools=tools, max_tokens=max_tokens)
        calls.append(call)
        if not call.tool_calls:
            return AgentRun(spec.name, spec.tier, task, call.text, calls, tool_log)
        messages.append(
            {
                "role": "assistant",
                "content": call.text or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in call.tool_calls
                ],
            }
        )
        for tc in call.tool_calls:
            allowed = tc["name"] in spec.tools
            result = (
                toolbox.call(tc["name"], tc["arguments"], sources=spec.sources)
                if allowed
                else json.dumps({"error": f"{tc['name']} is not one of your tools"})
            )
            tool_log.append(f"{tc['name']}({tc['arguments']})")
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    # Step budget spent. One last call with the tools taken away forces an answer.
    messages.append({"role": "user", "content": "You are out of tool calls. Answer now with what you have."})
    call = mesh.chat(spec.tier, messages, purpose=purpose, tags=[*tags, "step-cap"], max_tokens=max_tokens)
    calls.append(call)
    return AgentRun(spec.name, spec.tier, task, call.text, calls, tool_log, hit_cap=True)


# --------------------------------------------------------------------------- #
# Planning.
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    id: int
    task: str
    agent: str | None = None
    depends_on: list[int] = field(default_factory=list)


def parse_plan(text: str) -> list[Step]:
    """Parse the planner's JSON. Returns [] if it cannot be read (the caller falls back)."""
    match = re.search(r"\{.*\}", text, re.S)  # tolerate ```json fences and stray prose
    if not match:
        return []
    try:
        raw = json.loads(match.group(0)).get("steps", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    steps: list[Step] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not str(item.get("task", "")).strip():
            continue
        deps = [int(d) for d in item.get("depends_on") or [] if str(d).isdigit()]
        steps.append(Step(id=int(item.get("id", i)), task=str(item["task"]).strip(), agent=item.get("agent"), depends_on=deps))
    return steps


# --------------------------------------------------------------------------- #
# The desk: one entry point for a ticket.
# --------------------------------------------------------------------------- #
@dataclass
class TicketResult:
    ticket: str
    mode: str                       # react | plan | team-llm | team-vector
    answer: str
    calls: list[Call]               # every model call, in order
    runs: list[AgentRun]            # one per agent invocation
    plan: list[Step]
    notes: list[str]                # routing decisions and fallbacks, for the trace

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def llm_calls(self) -> int:
        return len(self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    def cost_by_tier(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.calls:
            out[c.tier] = out.get(c.tier, 0.0) + c.cost_usd
        return out


class Desk:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mesh = Mesh(config)
        self.toolbox = Toolbox(config)
        self.routes = load_routes(config)
        self._vector: VectorRouter | None = None

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Desk":
        return cls(load_config(path))

    # ---- the cast --------------------------------------------------------- #
    def generalist(self, tier: str, name: str = "generalist") -> AgentSpec:
        """Every tool, the whole KB. The monolith in react mode; the workers in plan mode."""
        return AgentSpec(name, tier, self.toolbox.names, None, "a support generalist who handles every kind of request")

    def specialists(self) -> dict[str, AgentSpec]:
        """One agent per route in fixtures/routes.json: its tier, its tools, its documents."""
        return {
            r["name"]: AgentSpec(
                name=r["name"],
                tier=r["tier"],
                tools=list(r["tools"]),
                sources=list(r["sources"]),
                description=f"the {r['name']} specialist. Your area: {r['description']}",
            )
            for r in self.routes
        }

    @property
    def vector_router(self) -> VectorRouter:
        if self._vector is None:
            self._vector = VectorRouter(
                self.routes,
                min_score=float(self.config["agents"].get("supervisor_min_score", 0.35)),
                default_tier=self.config["deterministic"].get("default_tier", "strong"),
            )
        return self._vector

    # ---- entry point ------------------------------------------------------ #
    def handle(self, ticket: str, *, mode: str | None = None, supervisor: str | None = None, tags: list[str] | None = None) -> TicketResult:
        cfg = self.config["agents"]
        mode = mode or cfg.get("mode", "react")
        supervisor = supervisor or cfg.get("supervisor", "llm")
        if mode == "react":
            return self._react(ticket, [*(tags or []), "mode-react"])
        if mode == "plan":
            return self._plan(ticket, [*(tags or []), "mode-plan"])
        if mode == "team" and supervisor == "llm":
            return self._team_llm(ticket, [*(tags or []), "mode-team-llm"])
        if mode == "team" and supervisor == "vector":
            return self._team_vector(ticket, [*(tags or []), "mode-team-vector"])
        raise ValueError(f"Unknown agents.mode '{mode}' / supervisor '{supervisor}'")

    # ---- react: one agent, one context ------------------------------------ #
    def _react(self, ticket: str, tags: list[str]) -> TicketResult:
        spec = self.generalist(self.config["agents"].get("planner_tier", "strong"), name="monolith")
        run = run_agent(self.mesh, self.toolbox, spec, ticket, config=self.config, tags=tags)
        return TicketResult(ticket, "react", run.answer, list(run.calls), [run], [], [])

    # ---- shared plan-and-execute machinery -------------------------------- #
    def _make_plan(self, ticket: str, tags: list[str], *, roster: dict[str, AgentSpec] | None) -> tuple[list[Step], Call, list[str]]:
        if roster:
            lines = "\n".join(f"- {name}: {spec.description.split('Your area: ')[-1]}" for name, spec in roster.items())
            roster_text = f"Assign each step to exactly one of these agents:\n{lines}\n\n"
            agent_field = '"agent": "<agent name>", '
        else:
            roster_text, agent_field = "", ""
        system = PLANNER_SYSTEM.format(roster=roster_text, agent_field=agent_field)
        call = self.mesh.chat(
            self.config["agents"].get("planner_tier", "strong"),
            [{"role": "system", "content": system}, {"role": "user", "content": ticket}],
            purpose="plan",
            tags=tags,
            max_tokens=int(self.config["agents"].get("max_tokens", 500)),
        )
        steps = parse_plan(call.text)
        notes: list[str] = []
        if not steps:
            notes.append("planner output was not valid JSON; falling back to one step with the whole ticket")
            steps = [Step(id=1, task=ticket)]
        return steps, call, notes

    def _execute(self, steps: list[Step], pick: Any, tags: list[str]) -> list[AgentRun]:
        """Run steps in order. Each worker starts from a FRESH context: its own
        task plus only the results of the steps it depends on."""
        results: dict[int, str] = {}
        runs: list[AgentRun] = []
        for step in steps:
            task = step.task
            deps = [d for d in step.depends_on if d in results]
            if deps:
                task += "\n\nResults from earlier steps:\n" + "\n".join(f"- step {d}: {results[d]}" for d in deps)
            run = run_agent(self.mesh, self.toolbox, pick(step), task, config=self.config, tags=[*tags, f"step-{step.id}"])
            results[step.id] = run.answer
            runs.append(run)
        return runs

    def _synthesize(self, ticket: str, runs: list[AgentRun], tags: list[str]) -> tuple[str, Call | None]:
        if len(runs) == 1:
            return runs[0].answer, None  # nothing to combine; do not pay a model to copy text
        findings = "\n\n".join(f"Finding {i} ({r.agent}):\n{r.answer}" for i, r in enumerate(runs, start=1))
        call = self.mesh.chat(
            self.config["agents"].get("synth_tier", "weak"),
            [
                {"role": "system", "content": SYNTH_SYSTEM},
                {"role": "user", "content": f"Ticket:\n{ticket}\n\n{findings}"},
            ],
            purpose="synthesize",
            tags=tags,
            max_tokens=int(self.config["agents"].get("max_tokens", 500)),
        )
        return call.text, call

    def _finish(self, ticket: str, mode: str, plan_call: Call | None, steps: list[Step], runs: list[AgentRun], notes: list[str], tags: list[str]) -> TicketResult:
        answer, synth_call = self._synthesize(ticket, runs, tags)
        calls: list[Call] = [plan_call] if plan_call else []
        for run in runs:
            calls.extend(run.calls)
        if synth_call:
            calls.append(synth_call)
        return TicketResult(ticket, mode, answer, calls, runs, steps, notes)

    # ---- plan: strong planner, cheap generalist workers ------------------- #
    def _plan(self, ticket: str, tags: list[str]) -> TicketResult:
        steps, plan_call, notes = self._make_plan(ticket, tags, roster=None)
        worker = self.generalist(self.config["agents"].get("worker_tier", "weak"), name="worker")
        runs = self._execute(steps, lambda step: worker, tags)
        return self._finish(ticket, "plan", plan_call, steps, runs, notes, tags)

    # ---- team, llm supervisor: the planner also assigns the specialist ---- #
    def _team_llm(self, ticket: str, tags: list[str]) -> TicketResult:
        team = self.specialists()
        steps, plan_call, notes = self._make_plan(ticket, tags, roster=team)
        fallback = self.generalist(self.config["deterministic"].get("default_tier", "strong"), name="fallback")

        def pick(step: Step) -> AgentSpec:
            if step.agent in team:
                notes.append(f"step {step.id} → {step.agent} ({team[step.agent].tier}), assigned by the planner")
                return team[step.agent]
            notes.append(f"step {step.id}: planner named unknown agent {step.agent!r} → fallback generalist")
            return fallback

        runs = self._execute(steps, pick, tags)
        return self._finish(ticket, "team-llm", plan_call, steps, runs, notes, tags)

    # ---- team, vector supervisor: no model call to decide who works ------- #
    def _team_vector(self, ticket: str, tags: list[str]) -> TicketResult:
        """Route each sentence of the ticket with Part 1's vector router. Every
        department that wins a sentence gets the whole ticket and is told to
        handle only its own part. Zero model calls are spent on routing."""
        team = self.specialists()
        notes: list[str] = []
        chosen: list[str] = []
        for sentence in lab2.split_sentences(ticket):
            decision = self.vector_router.route(sentence)
            if decision.route and decision.route not in chosen:
                chosen.append(decision.route)
                notes.append(f"{decision.route} ← {decision.score:.2f} “{sentence[:60]}”")
        if not chosen:
            notes.append("no sentence matched a route → fallback generalist")
            fallback = self.generalist(self.config["deterministic"].get("default_tier", "strong"), name="fallback")
            runs = [run_agent(self.mesh, self.toolbox, fallback, ticket, config=self.config, tags=tags)]
            return self._finish(ticket, "team-vector", None, [Step(1, ticket, "fallback")], runs, notes, tags)

        steps: list[Step] = []
        for i, name in enumerate(chosen, start=1):
            if len(chosen) == 1:
                task = ticket
            else:
                task = (
                    f"Customer ticket:\n{ticket}\n\n"
                    f"Handle ONLY the part of this ticket in your area ({team[name].description.split('Your area: ')[-1]}) "
                    "Other specialists are handling the rest; do not address it."
                )
            steps.append(Step(id=i, task=task, agent=name))
        runs = self._execute(steps, lambda step: team[step.agent], tags)
        return self._finish(ticket, "team-vector", None, steps, runs, notes, tags)


def check_ticket(answer: str, checks: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """[(check name, passed)] for a final reply."""
    out = []
    for check in checks:
        ok = all(re.search(p, answer, re.I) for p in check["expect"]) and not any(
            re.search(p, answer, re.I) for p in check.get("reject", [])
        )
        out.append((check["name"], ok))
    return out


# --------------------------------------------------------------------------- #
# Memory (Part 2.2).
#
# Lab 2's compaction keeps the newest turns that fit a token budget and drops
# the rest. That bounds cost, and it also forgets the order number the
# customer typed in turn 1. A memory is a second, much smaller store: short
# facts, embedded with the same MiniLM model, recalled by similarity to the
# current message. History is what was said. Memory is what is still true.
# --------------------------------------------------------------------------- #
class MemoryStore:
    def __init__(self, *, dedupe_score: float = 0.9) -> None:
        self.facts: list[str] = []
        self._vectors: list[list[float]] = []
        self.dedupe_score = dedupe_score

    def remember(self, facts: list[str]) -> int:
        """Store new facts. A fact nearly identical to a stored one replaces it. Returns how many were new."""
        added = 0
        for fact in (f.strip() for f in facts):
            if not fact:
                continue
            vec = lab2.embed([fact])[0]
            sims = [sum(a * b for a, b in zip(vec, v)) for v in self._vectors]
            if sims and max(sims) >= self.dedupe_score:
                i = sims.index(max(sims))
                self.facts[i], self._vectors[i] = fact, vec
                continue
            self.facts.append(fact)
            self._vectors.append(vec)
            added += 1
        return added

    def recall(self, query: str, *, k: int, max_tokens: int) -> list[str]:
        """The k stored facts most similar to `query`, within a token budget."""
        if not self.facts:
            return []
        q = lab2.embed([query])[0]
        ranked = sorted(
            zip(self.facts, self._vectors),
            key=lambda fv: -sum(a * b for a, b in zip(q, fv[1])),
        )
        out: list[str] = []
        used = 0
        for fact, _ in ranked[:k]:
            t = token_count(fact)
            if used + t > max_tokens:
                continue
            out.append(fact)
            used += t
        return out


@dataclass
class Turn:
    text: str
    answer: str
    calls: list[Call]               # the answer call, plus the memory-write call in memory mode
    history_messages_sent: int      # prior-turn messages that survived into this prompt
    recalled: list[str]
    remembered: list[str]

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)


class Session:
    """One multi-turn chat. `memory.mode` in config.yaml picks how the past is carried."""

    def __init__(self, config: dict[str, Any], *, mode: str | None = None) -> None:
        self.config = config
        self.cfg = config["memory"]
        self.mode = mode or self.cfg.get("mode", "full")
        if self.mode not in ("full", "compact", "memory"):
            raise ValueError(f"Unknown memory.mode '{self.mode}'")
        self.mesh = Mesh(config)
        self.history: list[lab2.Message] = []
        self.store = MemoryStore()

    def send(self, text: str, *, tags: list[str] | None = None) -> Turn:
        tags = [f"memory-{self.mode}", *(tags or [])]
        hits = retrieve(
            text,
            k=int(self.cfg.get("retrieve_k", 3)),
            min_score=float(self.config["rag"].get("min_score", 0.0)),
        )
        system = SESSION_SYSTEM
        recalled: list[str] = []
        if self.mode == "memory":
            recalled = self.store.recall(
                text, k=int(self.cfg.get("recall_k", 4)), max_tokens=int(self.cfg.get("recall_max_tokens", 80))
            )
            if recalled:
                system += "\n\nWhat you know about this customer:\n" + "\n".join(f"- {f}" for f in recalled)
        system += "\n\nContext:\n" + "\n\n".join(h.text for h in hits)

        user = lab2.Message("user", text)
        prompt = [lab2.Message("system", system), *self.history, user]
        if self.mode in ("compact", "memory"):
            prompt = lab2.compact(prompt, int(self.cfg.get("token_budget", 400)))  # Lab 2, unchanged
            if not prompt or prompt[-1] is not user:
                prompt.append(user)  # the budget never drops the question being asked
        kept = sum(1 for m in prompt if m in self.history)

        calls = [
            self.mesh.chat(
                self.cfg.get("tier", "weak"),
                [{"role": m.role, "content": m.content} for m in prompt],
                purpose="session",
                tags=tags,
            )
        ]
        answer = calls[0].text
        self.history += [user, lab2.Message("assistant", answer)]

        remembered: list[str] = []
        if self.mode == "memory":
            # The price of remembering: one small extra call per turn.
            write = self.mesh.chat(
                self.cfg.get("tier", "weak"),
                [
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": f"Customer: {text}\nAgent: {answer}"},
                ],
                purpose="memory",
                tags=tags,
                max_tokens=150,
            )
            calls.append(write)
            remembered = parse_facts(write.text)
            self.store.remember(remembered)
        return Turn(text, answer, calls, kept, recalled, remembered)


def parse_facts(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(i).strip() for i in items if isinstance(i, str) and i.strip()]


__all__ = [
    "Desk",
    "TicketResult",
    "AgentSpec",
    "AgentRun",
    "Step",
    "Toolbox",
    "run_agent",
    "parse_plan",
    "check_ticket",
    "Session",
    "Turn",
    "MemoryStore",
    "parse_facts",
]
