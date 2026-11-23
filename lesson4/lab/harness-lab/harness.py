#!/usr/bin/env python3
"""
The Northstar agent harness (PROVIDED).

This is the working system you observe in every part of the lab. You
configure it through `config.yaml` (`harness:`); you do not have to
implement it. Read it anyway: it is the lecture's harness ledger with the
mock model and the mock provider replaced by real ones.

A harness is everything in an agent except the model: what goes into the
prompt and in what order, what a tool result is allowed to add to the
context, when the loop stops, and whether anything checks the answer before
it leaves. One support ticket goes through this loop:

    ticket
      │
      ▼
    assemble the prompt ──────────────── stable_prefix   what is identical between calls goes first,
      │                                                  and the provider is told where the prefix ends
      ▼
    model call through the mesh ──────── reasoning       how much thinking each turn is allowed to buy
      │
      ├─ tool call ─► run it ─► result ─ cap_tools       what one result may add to every later call
      │                │
      │                └──────────────── loop_guard      turn budget, dollar budget, and "nothing new came back"
      │
      └─ draft reply ─────────────────── verify          a closing pass before the reply leaves: the agent
                                                          re-derives its own figures. Run it always, or let a
                                                          gate (a chat model, or a decision model) say when

With every switch off, this is the agent most teams ship first. The model,
the tools, and the tickets are the same in every configuration. Only the
harness changes, and with it the cost of a solved ticket.

Building on the earlier labs:
    Lab 1   every call goes through the mesh; prompt caching is the mechanism behind stable_prefix
    Lab 2   `search_kb` reads Lab 2's index; cap_tools uses Lab 2's `compress`; the loop guard
            uses Lab 2's embedder to notice that a tool result says nothing new
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from mesh import HERE, LAB2_DIR, Call, Decision, Mesh, lab2, load_config, log, token_count

FIXTURES = HERE / "fixtures"

HANDOFF = (
    "[harness] Stop calling tools: {reason}. Write the final reply to the customer now. "
    "Answer every part of the ticket you can from what you already have. For any part you could "
    "not resolve, say plainly that it is still open and that a specialist will follow up."
)

FINAL_MARK = "FINAL REPLY:"

CLOSING = (
    "[harness] Before this reply is sent, check it. {concern}For every question in the ticket, "
    "re-derive the amount, date, time, or count from the records and policy passages in this "
    "conversation, following the playbook's counting rules step by step. If a rule or a record you "
    "need is not in this conversation, look it up now. Show your working, one short line per "
    f"figure. Then write {FINAL_MARK} on a line of its own, followed by the complete reply to the "
    "customer, corrected wherever your check disagrees with the draft."
)


def final_reply(text: str) -> str:
    """A closing pass shows its working first. The customer gets only what follows the marker."""
    parts = re.split(re.escape(FINAL_MARK) + r"(?:\*\*)?", text)   # models like to write **FINAL REPLY:**
    tail = parts[-1].strip()
    return tail if len(parts) > 1 and tail else text.strip()


# --------------------------------------------------------------------------- #
# The switches. Defaults come from `harness:` in config.yaml; any of them can
# be overridden per run: harness.run(ticket, stable_prefix=True).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Options:
    tier: str = "mid"               # which rung of config.yaml's `tiers:` runs the agent
    stable_prefix: bool = False     # volatile text out of the prefix; cache breakpoints set
    break_cache: bool = False       # experiment: breakpoints set, the volatile line still on top
    cap_tools: bool = False         # cut each tool result down before it enters the context
    tool_cap_tokens: int = 400      # the ceiling for one tool result when cap_tools is on
    loop_guard: bool = False        # stop a run that has stopped making progress
    max_turns: int = 12             # loop guard: model calls allowed per ticket
    max_usd: float = 0.10           # loop guard: dollars allowed per ticket
    repeat_limit: int = 2           # loop guard: lookups that return nothing new before the run is stopped
    converge_score: float = 0.97    # loop guard: similarity above which a tool result counts as "nothing new"
    hard_turn_limit: int = 30       # where a run ends when there is no loop guard
    verify: str = "off"             # off | always | llm | decision: when a draft gets a closing pass
    max_checks: int = 1             # closing passes allowed per ticket
    reasoning: str = "off"          # off | on | sandwich: which turns are allowed to think

    def label(self) -> str:
        on = lambda b: "on" if b else "off"  # noqa: E731
        text = (
            f"tier={self.tier} stable-prefix={on(self.stable_prefix)} cap-tools={on(self.cap_tools)} "
            f"loop-guard={on(self.loop_guard)} verify={self.verify} reasoning={self.reasoning}"
        )
        return text + ("  [break-cache]" if self.break_cache else "")


# --------------------------------------------------------------------------- #
# Tools. Read-only lookups over fixtures/crm.json plus the Lab 2 index.
# --------------------------------------------------------------------------- #
class Toolbox:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        spec = json.loads((FIXTURES / "tools.json").read_text())
        # What the model sees: the function schema only. `returns` stays with the harness.
        self._schemas = [{"type": t["type"], "function": t["function"]} for t in spec["tools"]]
        self._returns = {t["function"]["name"]: t.get("returns") for t in spec["tools"]}
        self._crm = json.loads((FIXTURES / "crm.json").read_text())

    @property
    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self._schemas]

    def schemas(self) -> list[dict]:
        return self._schemas

    # ---- running a tool ---------------------------------------------------- #
    def call(self, name: str, arguments: str, *, attempt: int = 0) -> tuple[Any, str]:
        """Run one tool call. Returns (record, query).

        `record` is what the backend returned: a dict for the record tools, a
        list of passages for search_kb. Errors are returned, not raised, so
        the model can read them and recover. `query` is the text the result
        should be relevant to, used when the result is capped. `attempt` is
        how many times this run has already made this exact call; only the
        one lookup that never finishes (see fixtures/crm.json) looks at it.
        """
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "arguments were not valid JSON"}, ""
        if not isinstance(args, dict):
            return {"error": "arguments must be a JSON object"}, ""
        if name == "search_kb":
            query = str(args.get("query", ""))
            return self._search(query), query
        if name == "get_order":
            return self._lookup("orders", str(args.get("order_id", "")).upper()), ""
        if name == "get_account":
            return self._lookup("accounts", str(args.get("account_id", "")).upper()), ""
        if name == "get_api_usage":
            return self._lookup("api_usage", str(args.get("account_id", "")).upper()), ""
        if name == "get_login_status":
            return self._lookup("logins", str(args.get("email", "")).lower()), ""
        if name == "get_return":
            return self._lookup("returns", str(args.get("order_id", "")).upper()), ""
        if name == "get_shipment":
            record = self._lookup("shipments", str(args.get("order_id", "")).upper())
            progress = record.get("sync_percent_by_attempt")
            if progress:
                # A sync that creeps forward and never completes: each retry looks like progress.
                record = {k: v for k, v in record.items() if k != "sync_percent_by_attempt"}
                record["sync_percent"] = progress[min(attempt, len(progress) - 1)]
            return record, ""
        if name == "get_uptime":
            months = self._crm["uptime"].get(str(args.get("account_id", "")).upper())
            if months is None:
                return {"error": "account not found"}, ""
            record = months.get(str(args.get("month", "")))
            return record or {"error": "no data for that month", "available_months": sorted(months)}, ""
        return {"error": f"unknown tool {name}"}, ""

    def _lookup(self, table: str, key: str) -> dict[str, Any]:
        record = self._crm[table].get(key)
        return record if record is not None else {"error": f"{key or '(empty)'} not found"}

    def _search(self, query: str) -> list[str]:
        collection = lab2.get_collection()
        if collection.count() == 0:
            log(f"Vector DB is empty. Run:  (cd {LAB2_DIR} && python index_kb.py)")
            return []
        k = min(int(self.config["harness"].get("search_k", 8)), collection.count())
        result = collection.query(
            query_embeddings=[lab2.embed([query])[0]], n_results=k, include=["documents", "metadatas"]
        )
        return [
            f"[{(meta or {}).get('source', '')}] {doc}"
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]

    # ---- what a result is allowed to add to the context --------------------- #
    def render(self, name: str, record: Any, query: str, *, cap: int | None) -> str:
        """Turn a tool's return value into the text that enters the context.

        cap=None is the naive harness: whatever the backend returned, the
        model gets, and pays for again on every later turn.

        With a cap, three things happen, cheapest first:
          1. projection   a record keeps only the fields the tool is documented
                          to return (`returns` in fixtures/tools.json)
          2. compression  policy passages go through Lab 2's `compress`, keyed
                          on the agent's own search query
          3. a ceiling    anything still over the cap is cut, and the model is
                          told that it was
        """
        if cap is None:
            return "\n\n".join(record) if isinstance(record, list) else json.dumps(record)
        if isinstance(record, list):
            if not record:
                return "No matching policy passages."
            return lab2.compress("\n\n".join(record), query, cap)
        keep = self._returns.get(name)
        if keep and "error" not in record:
            record = {k: record[k] for k in keep if k in record}
        text = json.dumps(record)
        if token_count(text) > cap:
            words = text.split(" ")
            kept: list[str] = []
            for word in words:
                if token_count(" ".join([*kept, word])) > cap:
                    break
                kept.append(word)
            dropped = token_count(text) - token_count(" ".join(kept))
            text = " ".join(kept) + f" ...[capped by the harness: about {dropped} more tokens not shown]"
        return text


# --------------------------------------------------------------------------- #
# The loop guard. Three independent reasons to stop a run:
#   a turn budget, a dollar budget, and semantic convergence.
#
# Convergence is the interesting one. A stuck agent rarely repeats itself
# byte for byte, so comparing strings misses it. Lab 2's embedder does not:
# when a tool keeps returning text that means what an earlier result from
# the same tool already said, the run has stopped learning anything, however
# busy it looks. Lab 2 used this similarity to reuse an answer. Here it is
# used to stop paying for one.
# --------------------------------------------------------------------------- #
class LoopGuard:
    def __init__(self, options: Options) -> None:
        self.options = options
        self._seen: list[tuple[str, list[float]]] = []
        self.stale = 0                      # consecutive results that added nothing

    def observe(self, tool: str, result: str) -> float:
        """Record one tool result. Returns its similarity to the closest earlier result of the same tool."""
        vec = lab2.embed([result[:2000]])[0]
        best = max(
            (sum(a * b for a, b in zip(vec, v)) for name, v in self._seen if name == tool),
            default=0.0,
        )
        self._seen.append((tool, vec))
        self.stale = self.stale + 1 if best >= self.options.converge_score else 0
        return best

    def verdict(self, turns: int, spent_usd: float) -> str | None:
        """A reason to stop, or None to keep going."""
        opt = self.options
        if self.stale >= opt.repeat_limit:
            return f"the last {self.stale} lookups returned nothing new"
        if turns >= opt.max_turns:
            return f"the turn budget of {opt.max_turns} model calls is spent"
        if spent_usd >= opt.max_usd:
            return f"the budget of ${opt.max_usd:.2f} for one ticket is spent"
        return None


# --------------------------------------------------------------------------- #
# One run of the harness on one ticket.
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    ticket: str
    options: Options
    answer: str = ""
    stop_reason: str = ""
    steps: list[Call | Decision] = field(default_factory=list)   # every model request, in the order it was made
    tool_log: list[str] = field(default_factory=list)        # "get_order({...}) -> 132 tok" in call order
    evidence: list[str] = field(default_factory=list)        # each tool result as the model saw it (what a gate checks a draft against)
    events: list[str] = field(default_factory=list)          # what the harness itself did, for the trace
    verdicts: list[Any] = field(default_factory=list)        # decisions.Verdict, one per gated draft
    drafts: list[str] = field(default_factory=list)          # drafts that were sent back for a closing pass

    @property
    def calls(self) -> list[Call]:
        """Every chat completion: the agent's turns and, if a chat model is the gate, its verdicts."""
        return [s for s in self.steps if isinstance(s, Call)]

    @property
    def decisions(self) -> list[Decision]:
        """Every decision-model request."""
        return [s for s in self.steps if isinstance(s, Decision)]

    @property
    def turns(self) -> int:
        return sum(1 for c in self.calls if c.purpose in ("agent", "check", "handoff"))

    @property
    def api_usd(self) -> float:
        """What the mesh metered: provider spend."""
        return sum(c.cost_usd for c in self.calls) + sum(d.cost_usd for d in self.decisions)

    @property
    def local_usd(self) -> float:
        """What your own hardware cost, at local.usd_per_hour."""
        return sum(c.local_usd for c in self.calls) + sum(d.local_usd for d in self.decisions)

    @property
    def cost_usd(self) -> float:
        return self.api_usd + self.local_usd

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls) + sum(d.latency_s for d in self.decisions)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def cached_tokens(self) -> int:
        return sum(c.cached_tokens for c in self.calls)

    @property
    def cache_write_tokens(self) -> int:
        return sum(c.cache_write_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def reasoning_tokens(self) -> int:
        return sum(c.reasoning_tokens for c in self.calls)

    @property
    def hardware_usd(self) -> float:
        """Seconds of your own machine spent running the agent (the gate's seconds are in gate_usd)."""
        return sum(c.local_usd for c in self.calls if c.purpose != "judge")

    @property
    def gate_usd(self) -> float:
        """What deciding cost: judge calls plus decision-model requests."""
        return sum(c.total_usd for c in self.calls if c.purpose == "judge") + sum(d.total_usd for d in self.decisions)

    @property
    def check_usd(self) -> float:
        """What the closing passes cost."""
        return sum(c.total_usd for c in self.calls if c.purpose == "check")


class Harness:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.mesh = Mesh(config)
        self.toolbox = Toolbox(config)
        self.playbook = (FIXTURES / "playbook.md").read_text().strip()
        self._gate = None

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Harness":
        return cls(load_config(path))

    def options(self, **overrides: Any) -> Options:
        known = {f.name for f in fields(Options)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown harness option(s) {sorted(unknown)}. Known: {sorted(known)}")
        base = {k: v for k, v in (self.config.get("harness") or {}).items() if k in known}
        merged = {**base, **overrides}
        for key in ("verify", "reasoning"):
            if merged.get(key) is False:      # an unquoted `off` in YAML parses as False
                merged[key] = "off"
        opt = replace(Options(), **merged)
        if opt.verify not in ("off", "always", "llm", "decision"):
            raise ValueError(f"harness.verify must be off | always | llm | decision, not '{opt.verify}'")
        if opt.reasoning not in ("off", "on", "sandwich"):
            raise ValueError(f"harness.reasoning must be off | on | sandwich, not '{opt.reasoning}'")
        return opt

    @property
    def gate(self):
        """What decides whether a draft needs the closing pass (see decisions.py)."""
        if self._gate is None:
            from decisions import Gate

            self._gate = Gate(self.mesh, self.config, self.playbook)
        return self._gate

    # ---- prompt assembly ---------------------------------------------------- #
    @staticmethod
    def _stamp() -> str:
        """The volatile line: a request id and the wall clock.

        Somebody added it for debugging. It is different on every call, and
        in the naive harness it is the first line of the prompt.
        """
        return f"[request {uuid.uuid4().hex[:8]} | agent clock {time.strftime('%H:%M:%S')}]"

    def _breakpoints(self, opt: Options) -> bool:
        """Does this run mark cache breakpoints? Only providers that need them get them."""
        return (opt.stable_prefix or opt.break_cache) and self.mesh.tiers[opt.tier].get("cache") == "breakpoints"

    def _system(self, opt: Options) -> dict[str, Any]:
        """The system message for ONE model call. Called again on every turn."""
        volatile_on_top = opt.break_cache or not opt.stable_prefix
        text = f"{self._stamp()}\n\n{self.playbook}" if volatile_on_top else self.playbook
        if self._breakpoints(opt):
            # Breakpoint 1: everything up to here (tool schemas + playbook) is one cacheable prefix.
            return {"role": "system", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}
        return {"role": "system", "content": text}

    def _with_tail_breakpoint(self, messages: list[dict[str, Any]], opt: Options) -> list[dict[str, Any]]:
        """Breakpoint 2: the end of the conversation so far.

        The next call re-sends all of this plus one more tool result. Marking
        the tail lets the provider read everything before it from cache, so a
        tool result is paid for once at the write price and then at the read
        price on every later turn, instead of at full price every time.
        """
        if not self._breakpoints(opt) or not messages:
            return messages
        last = dict(messages[-1])
        if isinstance(last.get("content"), str) and last["content"]:
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]
        return [*messages[:-1], last]

    def _effort(self, opt: Options, *, edge: bool, spent: bool = False) -> dict[str, Any]:
        """Reasoning budget for one agent turn.

        `sandwich` thinks at the two edges of a run, the opening turn (the
        plan) and the closing pass (the check), and not in the tool-calling
        middle, where the model is mostly choosing which record to fetch.
        `spent` is set once a turn has burned its whole output budget without
        writing anything: the rest of the run does not think.
        """
        cfg = self.config.get("reasoning", {})
        if not spent and (opt.reasoning == "on" or (opt.reasoning == "sandwich" and edge)):
            return {"reasoning_effort": cfg.get("effort", "low")}
        if self.mesh.tiers[opt.tier].get("thinks_by_default"):
            return {"reasoning_effort": cfg.get("off_value", "none")}   # open models often think unless told not to
        return {}

    # ---- the loop ------------------------------------------------------------ #
    def run(self, ticket: str, *, tags: list[str] | None = None, **overrides: Any) -> Run:
        """Work one ticket to a final reply (or to the point where something stops the run)."""
        opt = self.options(**overrides)
        cfg = self.config["harness"]
        run = Run(ticket=ticket, options=opt)
        tags = [f"tier-{opt.tier}", *(tags or [])]
        tools = self.toolbox.schemas()
        thinking_room = int(self.config.get("reasoning", {}).get("extra_tokens", 0)) if opt.reasoning != "off" else 0
        cap = int(opt.tool_cap_tokens) if opt.cap_tools else None
        guard = LoopGuard(opt) if opt.loop_guard else None

        # The conversation after the system message. With a stable prefix the
        # volatile line is stamped once, at the end of the ticket, where it
        # cannot invalidate anything in front of it.
        opening = f"{ticket}\n\n{self._stamp()}" if opt.stable_prefix and not opt.break_cache else ticket
        history: list[dict[str, Any]] = [{"role": "user", "content": opening}]
        evidence = run.evidence            # tool results as the model saw them, for the gate
        seen: dict[tuple[str, str], int] = {}   # (tool, arguments) -> times called in this run
        checks = 0                         # closing passes run so far
        empties = 0                        # turns that came back with nothing written
        purpose = "agent"

        def model_turn(**params: Any) -> Call:
            messages = [self._system(opt), *self._with_tail_breakpoint(history, opt)]
            call = self.mesh.chat(
                opt.tier,
                messages,
                purpose=purpose,
                tags=tags,
                tools=tools,
                # A closing pass writes its working and then the reply; thinking is billed as output too.
                max_tokens=int(cfg.get("check_max_tokens" if purpose == "check" else "max_tokens", 700)) + thinking_room,
                **self._effort(opt, edge=not run.steps or purpose == "check", spent=bool(empties)),
                **params,
            )
            run.steps.append(call)
            return call

        def hand_off(reason: str) -> None:
            """The guard's exit: one last call that may not use tools, so the customer gets a reply.

            The tool schemas stay in the request and `tool_choice` forbids
            using them. Dropping the schemas would be the obvious way to do
            this, and it would change the first bytes of the prompt: the whole
            cached prefix would be paid for again on the way out.
            """
            nonlocal purpose
            run.events.append(f"loop guard: {reason}")
            history.append({"role": "user", "content": HANDOFF.format(reason=reason)})
            purpose = "handoff"
            run.answer = model_turn(tool_choice="none").text
            run.stop_reason = f"loop guard: {reason}"

        while True:
            if guard:
                reason = guard.verdict(run.turns, run.cost_usd)
                if reason:
                    hand_off(reason)
                    return run
            elif run.turns >= opt.hard_turn_limit:
                # Nothing engineered: the run is cut off mid-task and no reply exists.
                run.stop_reason = f"hard limit of {opt.hard_turn_limit} turns, nothing stopped it"
                run.events.append(run.stop_reason)
                return run

            call = model_turn()

            if call.tool_calls:
                turn: dict[str, Any] = {
                    "role": "assistant",
                    "content": call.text or None,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in call.tool_calls
                    ],
                }
                if call.thinking_blocks:
                    turn["thinking_blocks"] = call.thinking_blocks   # Anthropic requires them back inside a tool loop
                history.append(turn)
                for tc in call.tool_calls:
                    key = (tc["name"], tc["arguments"])
                    record, query = self.toolbox.call(tc["name"], tc["arguments"], attempt=seen.get(key, 0))
                    seen[key] = seen.get(key, 0) + 1
                    result = self.toolbox.render(tc["name"], record, query, cap=cap)
                    history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    evidence.append(f"{tc['name']}({tc['arguments']}) returned:\n{result}")
                    note = f"{tc['name']}({tc['arguments']}) -> {token_count(result)} tok"
                    if guard:
                        similarity = guard.observe(tc["name"], result)
                        if similarity >= opt.converge_score:
                            note += f"  [nothing new: {similarity:.2f} similar to an earlier result]"
                    run.tool_log.append(note)
                continue

            if not call.text:
                # Neither a tool call nor a reply: the output budget ran out, usually on thinking.
                empties += 1
                run.events.append("empty turn: the output budget was spent before anything was written")
                if empties > 1:
                    run.stop_reason = "no reply: the model returned nothing twice"
                    return run
                continue   # ask again, without thinking this time

            # No tool call: the model believes it is done. This is a draft.
            draft = call.text
            if opt.verify != "off" and checks < opt.max_checks:
                concern = ""
                if opt.verify != "always":
                    # A gate decides whether this draft is worth a second look.
                    verdict = self.gate.check(ticket, evidence, draft, mode=opt.verify, tags=tags)
                    run.verdicts.append(verdict)
                    run.steps.extend([*verdict.decisions, *verdict.calls])
                    run.events.append(f"gate[{verdict.path}]: {verdict.summary()}")
                    if verdict.ok:
                        run.answer = draft
                        run.stop_reason = "done, passed the gate"
                        return run
                    concern = f"A reviewer flagged it: {verdict.problem} " if verdict.problem else ""
                checks += 1
                run.drafts.append(draft)
                history.append({"role": "assistant", "content": draft})
                history.append({"role": "user", "content": CLOSING.format(concern=concern)})
                purpose = "check"
                continue
            run.answer = final_reply(draft) if checks else draft
            run.stop_reason = "done, checked" if checks else "done, unverified"
            return run


# --------------------------------------------------------------------------- #
# Grading and the one number that matters.
# --------------------------------------------------------------------------- #
def load_tickets(path: str | Path | None = None) -> list[dict[str, Any]]:
    return json.loads(Path(path or FIXTURES / "tickets.json").read_text())["tickets"]


def check_ticket(answer: str, checks: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """[(check name, passed)] for a final reply. `expect` and `reject` are regexes."""
    out = []
    for check in checks:
        ok = all(re.search(p, answer, re.I) for p in check["expect"]) and not any(
            re.search(p, answer, re.I) for p in check.get("reject", [])
        )
        out.append((check["name"], ok))
    return out


def solved(run: Run, ticket: dict[str, Any]) -> bool:
    """A ticket is solved only if every check passes. Half a reply is a reopened ticket."""
    return bool(run.answer) and all(ok for _, ok in check_ticket(run.answer, ticket["checks"]))


def cost_per_solved(attempt_usd: float, success: float, fail_cost: float = 0.0) -> float:
    """The lecture's metric.

    Retry until solved: 1/success attempts on average, and each failure also
    costs `fail_cost` to clean up (a human reopens the ticket).
    """
    if success <= 0:
        return float("inf")
    return (attempt_usd + (1.0 - success) * fail_cost) / success


# --------------------------------------------------------------------------- #
# Printing: the lecture's ledger and trace, from real usage.
# --------------------------------------------------------------------------- #
LINE_ITEMS = [
    ("input", "Input at full price"),
    ("cache_write", "Cache writes"),
    ("cache_read", "Cache reads"),
    ("output", "Output"),
]


def ledger(mesh: Mesh, runs: list[Run]) -> dict[str, float]:
    """Dollars by line item across runs, priced from config.yaml. The agent's own calls only: a judge is not the agent."""
    totals = {key: 0.0 for key, _ in LINE_ITEMS}
    for run in runs:
        for c in run.calls:
            if c.purpose == "judge":
                continue
            for key, value in mesh.line_items(c.tier, c.fresh_tokens, c.cached_tokens, c.cache_write_tokens, c.completion_tokens).items():
                totals[key] += value
    return totals


def _bar(value: float, scale: float, width: int = 30) -> str:
    if scale <= 0 or value <= 0:
        return ""
    cells = value / scale * width
    return "█" * int(cells) + ("▌" if cells - int(cells) >= 0.5 else ("▏" if int(cells) == 0 else ""))


def print_ledger(harness: Harness, run: Run) -> None:
    """What one run cost, by line item."""
    items = ledger(harness.mesh, [run])
    total = run.cost_usd

    def line(name: str, value: float, note: str = "") -> None:
        share = f"{value / total:4.0%}" if total > 0 and value > 0 else "    "
        log(f"    {name:<24}${value:>9.4f}  {share}  {note or _bar(value, total)}")

    log()
    log(f"Harness  {run.options.label()}")
    log()
    log(f"  {'This run':<26}${total:>9.4f}   {run.turns} turns, {run.latency_s:.1f}s, ended by: {run.stop_reason}")
    for key, name in LINE_ITEMS:
        line(name, items[key])
    if run.hardware_usd:
        seconds = sum(c.latency_s for c in run.calls if c.purpose != "judge")
        line("Your hardware", run.hardware_usd, f"{seconds:.1f}s of your machine at ${harness.config.get('local', {}).get('usd_per_hour', 0)}/hour")
    if run.verdicts:
        tokens = sum(d.input_tokens for d in run.decisions) + sum(c.prompt_tokens for c in run.calls if c.purpose == "judge")
        judged = sum(1 for v in run.verdicts if v.decisions or v.calls)
        refused = len(run.verdicts) - judged       # a draft whose evidence did not fit the decision model's window
        note = f"{judged} draft(s) judged, {tokens:,} input tokens" + (f"; {refused} too large to read" if refused else "")
        line("Gate", run.gate_usd, note)
    log()


def print_trace(run: Run) -> None:
    """One run, request by request: where each call's input tokens were billed."""
    log()
    log(f"  {'call':>4} {'purpose':<8}{'prompt':>8}{'full price':>12}{'cache write':>13}{'cache read':>12}{'output':>8}{'reasoning':>11}{'call $':>10}{'total $':>10}{'sec':>7}")
    running = 0.0
    for i, step in enumerate(run.steps, start=1):
        running += step.total_usd
        if isinstance(step, Decision):     # a decision model reads and answers; it has no output tokens
            log(f"  {i:>4} {'gate':<8}{step.input_tokens:>8,}{'':>12}{'':>13}{'':>12}{'-':>8}{'-':>11}{step.total_usd:>10.5f}{running:>10.4f}{step.latency_s:>7.1f}")
            continue
        log(
            f"  {i:>4} {step.purpose:<8}{step.prompt_tokens:>8,}{step.fresh_tokens:>12,}{step.cache_write_tokens:>13,}{step.cached_tokens:>12,}"
            f"{step.completion_tokens:>8,}{step.reasoning_tokens:>11,}{step.total_usd:>10.5f}{running:>10.4f}{step.latency_s:>7.1f}"
        )
    log()
    for entry in run.tool_log:
        log(f"  tool  {entry[:150]}")
    for entry in run.events:
        log(f"  note  {entry[:220]}")
    log(f"  {run.turns} turns, ${run.cost_usd:.4f}, ended by: {run.stop_reason}")
    log()


__all__ = [
    "Harness",
    "Options",
    "Run",
    "Toolbox",
    "LoopGuard",
    "load_tickets",
    "check_ticket",
    "solved",
    "cost_per_solved",
    "ledger",
    "print_ledger",
    "print_trace",
    "LINE_ITEMS",
]
