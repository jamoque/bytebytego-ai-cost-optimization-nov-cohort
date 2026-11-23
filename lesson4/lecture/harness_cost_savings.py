#!/usr/bin/env python3
"""Harness ledger: what one solved agent task costs, and which harness switch moved it.

Runs a small agent harness against a scripted mock model and a mock provider that bills tokens, including a real
prefix cache. Nothing calls a network. Every run prints the same numbers.

    python3 harness_ledger.py                      # nothing engineered: $28.34 per solved task
    python3 harness_ledger.py --stable-prefix      # flip one switch
    python3 harness_ledger.py --all --reasoning high
    python3 harness_ledger.py ladder --pause       # the Part I.2 walk, one Enter per switch
    python3 harness_ledger.py modes                # the three reasoning modes (Part I.3)
    python3 harness_ledger.py failcost             # where the best mode flips
    python3 harness_ledger.py trace --stuck        # one run, turn by turn
    python3 harness_ledger.py frontier             # all 48 configurations
    python3 harness_ledger.py --break-cache        # caching on, timestamp still in the prompt

Python 3.8 or later, standard library only.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, List, NamedTuple

# --------------------------------------------------------------------------------------
# The world: prices and the scripted task
# --------------------------------------------------------------------------------------

# US dollars per million tokens. A mid-tier closed model, list price, November 2026.
PRICE = {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00}

SYSTEM_TOKENS = 12_000        # system prompt plus tool definitions
OUTPUT_TOKENS = 300           # visible output per turn, mostly a tool call
TOOL_RESULT_RAW = 4_000       # what a tool returns when nothing limits it
TOOL_RESULT_CAPPED = 1_200    # what the harness lets through when it caps tool output
REASONING = {"high": 2_000, "max": 6_000}   # reasoning tokens per step, billed as output

TASK_TURNS = 30               # tool calls the mock model needs to finish the task
VERIFY_TURNS = 4              # extra turns for a verification pass before exit
HARD_TURN_LIMIT = 120         # where a run ends if the harness never stops it
GUARD_TURN_BUDGET = 45        # where the loop guard stops it
STUCK_SHARE = 0.08            # share of runs that get stuck and never finish

# Published: Terminal Bench 2.0 scores for one fixed model under a full harness
# (LangChain, February 2026). Only the reasoning allocation differs.
SUCCESS_FULL_HARNESS = {"max": 0.539, "high": 0.636, "sandwich": 0.665}
# Illustrative: what removing a harness feature costs in success rate.
NO_VERIFY_PENALTY = 0.10
NO_GUARD_PENALTY = 0.03

LINE_ITEMS = [
    ("input", "Input at full price"),
    ("cache_write", "Cache writes"),
    ("cache_read", "Cache reads"),
    ("output", "Output"),
    ("reasoning", "Reasoning"),
]


class Segment(NamedTuple):
    """One piece of a prompt. Two segments are the same bytes only if their keys match."""
    key: str
    tokens: int


@dataclass(frozen=True)
class Config:
    stable_prefix: bool = False   # no timestamp in the system prompt, cache breakpoints set
    cap_tools: bool = False       # truncate tool results before they enter context
    loop_guard: bool = False      # stop a run at a turn budget
    verify: bool = False          # run a verification pass before the agent may exit
    reasoning: str = "max"        # "max", "high" or "sandwich"
    break_cache: bool = False     # experiment: breakpoints set, timestamp still in the prompt
    fail_cost: float = 0.0        # what one failed task costs to clean up, in dollars

    def label(self) -> str:
        on = lambda b: "on " if b else "off"
        text = (f"stable-prefix={on(self.stable_prefix)} cap-tools={on(self.cap_tools)} "
                f"loop-guard={on(self.loop_guard)} verify={on(self.verify)} reasoning={self.reasoning}")
        return text + ("  [break-cache]" if self.break_cache else "")


# --------------------------------------------------------------------------------------
# The mock provider: bills a request, with prefix caching
# --------------------------------------------------------------------------------------

class MockProvider:
    """Bills one request at a time.

    Prefix caching works the way it does at real providers: only the leading part of
    the prompt that is identical to the previous request is read from cache. Everything
    after the first difference is written again at the cache-write price. If the harness
    sets no cache breakpoints, the whole prompt is billed at the plain input price.
    """

    def __init__(self, caching: bool) -> None:
        self.caching = caching
        self.previous: List[Segment] = []

    def bill(self, prompt: List[Segment], output_tokens: int, reasoning_tokens: int) -> Dict[str, int]:
        total = sum(s.tokens for s in prompt)
        if not self.caching:
            usage = {"input": total, "cache_write": 0, "cache_read": 0}
        else:
            hit = 0
            for old, new in zip(self.previous, prompt):
                if old.key != new.key:
                    break                      # first difference ends the cached prefix
                hit += new.tokens
            usage = {"input": 0, "cache_write": total - hit, "cache_read": hit}
            self.previous = list(prompt)
        usage["output"] = output_tokens
        usage["reasoning"] = reasoning_tokens
        return usage


def dollars(usage: Dict[str, int]) -> Dict[str, float]:
    price = dict(PRICE, reasoning=PRICE["output"])
    return {k: usage[k] * price[k] / 1e6 for k in usage}


# --------------------------------------------------------------------------------------
# The harness: everything in the agent except the model
# --------------------------------------------------------------------------------------

@dataclass
class Run:
    turns: int
    stop_reason: str
    cost: Dict[str, float]           # dollars by line item
    trace: List[dict]

    @property
    def total(self) -> float:
        return sum(self.cost.values())


class Harness:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # -- prompt assembly ---------------------------------------------------------------
    def system_segment(self, turn: int) -> Segment:
        """The naive harness stamps the current time into the system prompt.
        That one line makes the first 12,000 tokens different on every turn."""
        timestamp_in_prompt = self.cfg.break_cache or not self.cfg.stable_prefix
        key = f"system@10:42:{turn:02d}" if timestamp_in_prompt else "system"
        return Segment(key, SYSTEM_TOKENS)

    def tool_result(self, turn: int) -> Segment:
        tokens = TOOL_RESULT_CAPPED if self.cfg.cap_tools else TOOL_RESULT_RAW
        return Segment(f"tool#{turn}", tokens)

    # -- reasoning budget --------------------------------------------------------------
    def reasoning_tokens(self, turn: int, last_turn: int) -> int:
        mode = self.cfg.reasoning
        if mode in REASONING:
            return REASONING[mode]
        # Sandwich: max effort for the two planning turns and the two closing turns.
        edge = turn <= 2 or turn > last_turn - 2
        return REASONING["max"] if edge else REASONING["high"]

    # -- the loop ----------------------------------------------------------------------
    def run(self, stuck: bool) -> Run:
        cfg = self.cfg
        provider = MockProvider(caching=cfg.stable_prefix or cfg.break_cache)
        if stuck:
            # The mock model repeats a failing edit forever. Only the harness can end it.
            last_turn = GUARD_TURN_BUDGET if cfg.loop_guard else HARD_TURN_LIMIT
            stop = "loop guard: turn budget" if cfg.loop_guard else "hard limit, nothing stopped it"
        else:
            last_turn = TASK_TURNS + (VERIFY_TURNS if cfg.verify else 0)
            stop = "done, verified" if cfg.verify else "done, unverified"

        history: List[Segment] = []
        cost = {k: 0.0 for k, _ in LINE_ITEMS}
        trace = []
        running = 0.0
        for turn in range(1, last_turn + 1):
            prompt = [self.system_segment(turn)] + history
            usage = provider.bill(prompt, OUTPUT_TOKENS, self.reasoning_tokens(turn, last_turn))
            spent = dollars(usage)
            for k in cost:
                cost[k] += spent[k]
            running += sum(spent.values())
            trace.append({"turn": turn, "prompt": sum(s.tokens for s in prompt), **usage,
                          "turn_cost": sum(spent.values()), "running": running})
            # The model's tool call and the tool's result join the context for the next turn.
            history.append(Segment(f"assistant#{turn}", OUTPUT_TOKENS))
            history.append(self.tool_result(turn))
        return Run(last_turn, stop, cost, trace)


# --------------------------------------------------------------------------------------
# Expected cost of a solved task
# --------------------------------------------------------------------------------------

@dataclass
class Result:
    cfg: Config
    normal: Run
    stuck: Run
    parts: Dict[str, float]
    per_attempt: float
    success: float
    per_success: float


def evaluate(cfg: Config) -> Result:
    harness = Harness(cfg)
    normal, stuck = harness.run(stuck=False), harness.run(stuck=True)
    q = STUCK_SHARE
    parts = {k: (1 - q) * normal.cost[k] + q * stuck.cost[k] for k, _ in LINE_ITEMS}
    per_attempt = sum(parts.values())
    success = (SUCCESS_FULL_HARNESS[cfg.reasoning]
               - (0 if cfg.verify else NO_VERIFY_PENALTY)
               - (0 if cfg.loop_guard else NO_GUARD_PENALTY))
    # Retry until solved: 1/success attempts on average, and each failure also costs cleanup.
    per_success = (per_attempt + (1 - success) * cfg.fail_cost) / success
    return Result(cfg, normal, stuck, parts, per_attempt, success, per_success)


NAIVE = Config()
FULL = Config(stable_prefix=True, cap_tools=True, loop_guard=True, verify=True)


# --------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if USE_COLOR else text


def dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if USE_COLOR else text


def usd(v: float) -> str:
    return f"${v:,.2f}"


def bar(value: float, scale: float, width: int = 36) -> str:
    if scale <= 0 or value <= 0:
        return ""
    cells = value / scale * width
    full, frac = int(cells), cells - int(cells)
    return "█" * full + ("▌" if frac >= 0.5 else ("▏" if full == 0 else ""))


def print_ledger(r: Result) -> None:
    base = evaluate(replace(NAIVE, fail_cost=r.cfg.fail_cost))
    print()
    print(bold("Harness  ") + r.cfg.label())
    print()
    print(f"  {'One attempt':<28}{usd(r.per_attempt):>9}")
    for key, name in LINE_ITEMS:
        v = r.parts[key]
        share = f"{v / r.per_attempt:4.0%}" if v > 0 else "    "
        print(f"    {name:<26}{usd(v):>9}  {share}  {bar(v, r.per_attempt)}")
    print()
    print(f"  {'Normal run':<28}{usd(r.normal.total):>9}  {r.normal.turns} turns, {r.normal.stop_reason}")
    print(f"  {'Stuck run':<28}{usd(r.stuck.total):>9}  {r.stuck.turns} turns, {r.stuck.stop_reason}"
          f"  ({STUCK_SHARE:.0%} of runs)")
    print(f"  {'Success rate':<28}{r.success:>9.1%}")
    if r.cfg.fail_cost:
        print(f"  {'Cleanup per failed task':<28}{usd(r.cfg.fail_cost):>9}")
    print()
    print(bold(f"  {'Cost per solved task':<28}{usd(r.per_success):>9}"))
    ratio = base.per_success / r.per_success
    if abs(ratio - 1) > 0.005:
        word = "cheaper" if ratio > 1 else "dearer"
        factor = ratio if ratio > 1 else 1 / ratio
        print(dim(f"  {factor:.1f}x {word} than the same model with nothing engineered ({usd(base.per_success)})"))
    print()


def cmd_ladder(args) -> None:
    steps = [
        ("Nothing engineered", NAIVE),
        ("+ stable prefix", replace(NAIVE, stable_prefix=True)),
        ("+ cap tool output", replace(NAIVE, stable_prefix=True, cap_tools=True)),
        ("+ loop guard", replace(NAIVE, stable_prefix=True, cap_tools=True, loop_guard=True)),
        ("+ verification", replace(FULL, reasoning="max")),
        ("reasoning: high everywhere", replace(FULL, reasoning="high")),
        ("reasoning: sandwich", replace(FULL, reasoning="sandwich")),
    ]
    steps = [(name, replace(cfg, fail_cost=args.fail_cost)) for name, cfg in steps]
    first = evaluate(steps[0][1]).per_success
    print()
    print(bold(f"  {'Step':<28}{'Attempt':>9}{'Success':>9}{'Solved task':>13}{'vs first':>10}   Largest line item"))
    for i, (name, cfg) in enumerate(steps):
        if args.pause and i and sys.stdin.isatty():
            input(dim("  [Enter for the next switch] "))
        r = evaluate(cfg)
        top_key = max(r.parts, key=r.parts.get)
        top = dict(LINE_ITEMS)[top_key]
        print(f"  {name:<28}{usd(r.per_attempt):>9}{r.success:>9.1%}{usd(r.per_success):>13}"
              f"{first / r.per_success:>9.1f}x   {top.lower()}, {r.parts[top_key] / r.per_attempt:.0%}")
    print()


def cmd_modes(args) -> None:
    print()
    print(bold("  Full harness, three ways to spend reasoning") + (f"  (cleanup {usd(args.fail_cost)} per failure)" if args.fail_cost else ""))
    print()
    print(bold(f"  {'Reasoning':<20}{'Success':>9}{'Attempt':>10}{'Solved task':>13}"))
    rows = [evaluate(replace(FULL, reasoning=m, fail_cost=args.fail_cost)) for m in ("max", "high", "sandwich")]
    best = min(rows, key=lambda r: r.per_success)
    names = {"max": "Max everywhere", "high": "High everywhere", "sandwich": "Sandwich"}
    for r in rows:
        mark = "  <- lowest cost per solved task" if r is best else ""
        print(f"  {names[r.cfg.reasoning]:<20}{r.success:>9.1%}{usd(r.per_attempt):>10}{usd(r.per_success):>13}{mark}")
    print()


def cmd_failcost(args) -> None:
    print()
    print(bold("  Cost per solved task as a failure gets more expensive (full harness)"))
    print()
    print(bold(f"  {'Cleanup':>8}{'Max':>10}{'High':>10}{'Sandwich':>10}   Lowest"))
    flip = None
    previous = None
    cents = 0
    while cents <= 1000:
        f = cents / 100
        vals = {m: evaluate(replace(FULL, reasoning=m, fail_cost=f)).per_success for m in ("max", "high", "sandwich")}
        best = min(vals, key=vals.get)
        if previous and best != previous and flip is None:
            flip = f
        previous = best
        if cents % 50 == 0:
            print(f"  {usd(f):>8}{usd(vals['max']):>10}{usd(vals['high']):>10}{usd(vals['sandwich']):>10}   {best}")
        cents += 1
    print()
    if flip is not None:
        print(f"  The lowest-cost mode changes at about {usd(flip)} per failed task.")
    print()


def all_configs(fail_cost: float = 0.0) -> List[Result]:
    out = []
    for sp, cap, guard, ver in itertools.product([False, True], repeat=4):
        for mode in ("max", "high", "sandwich"):
            out.append(evaluate(Config(sp, cap, guard, ver, mode, fail_cost=fail_cost)))
    return out


def frontier(results: List[Result]) -> List[Result]:
    """Configurations that nothing else beats on both cost per attempt and success."""
    keep = []
    for p in results:
        beaten = any(o is not p and o.per_attempt <= p.per_attempt and o.success >= p.success
                     and (o.per_attempt < p.per_attempt or o.success > p.success) for o in results)
        if not beaten:
            keep.append(p)
    return sorted(keep, key=lambda r: r.per_attempt)


def cmd_frontier(args) -> None:
    results = all_configs(args.fail_cost)
    front = frontier(results)
    shown = sorted(results, key=lambda r: r.per_success)
    if not args.every:
        shown = shown[:10]
    print()
    print(bold(f"  {len(results)} configurations, {len(front)} on the frontier, {len(results) - len(front)} dominated"))
    print()
    print(bold(f"  {'':2}{'prefix':>7}{'cap':>5}{'guard':>7}{'verify':>8}  {'reasoning':<10}{'Attempt':>9}{'Success':>9}{'Solved task':>13}"))
    yn = lambda b: "on" if b else "-"
    for r in shown:
        c = r.cfg
        star = "*" if r in front else " "
        print(f"  {star:2}{yn(c.stable_prefix):>7}{yn(c.cap_tools):>5}{yn(c.loop_guard):>7}{yn(c.verify):>8}  "
              f"{c.reasoning:<10}{usd(r.per_attempt):>9}{r.success:>9.1%}{usd(r.per_success):>13}")
    print()
    print(dim("  * on the frontier: nothing else is both cheaper per attempt and at least as successful."))
    if not args.every:
        print(dim("  Showing the 10 cheapest per solved task. Add --every for all of them."))
    print()


def cmd_trace(args, cfg: Config) -> None:
    run = Harness(cfg).run(stuck=args.stuck)
    rows = run.trace
    if not args.full and len(rows) > 10:
        rows = rows[:5] + [None] + rows[-4:]
    print()
    print(bold("Harness  ") + cfg.label())
    print(f"  One {'stuck' if args.stuck else 'normal'} run, tokens per turn")
    print()
    print(bold(f"  {'turn':>4}{'prompt':>9}{'full price':>12}{'cache write':>13}{'cache read':>12}{'reasoning':>11}{'turn $':>9}{'total $':>9}"))
    for t in rows:
        if t is None:
            print(dim(f"  {'...':>4}"))
            continue
        print(f"  {t['turn']:>4}{t['prompt']:>9,}{t['input']:>12,}{t['cache_write']:>13,}{t['cache_read']:>12,}"
              f"{t['reasoning']:>11,}{t['turn_cost']:>9.3f}{t['running']:>9.2f}")
    print()
    print(f"  {run.turns} turns, {usd(run.total)}, ended by: {run.stop_reason}")
    if not args.full:
        print(dim("  Add --full for every turn."))
    print()


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def add_switches(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("harness switches")
    g.add_argument("--stable-prefix", action="store_true", help="no timestamp in the system prompt; set cache breakpoints")
    g.add_argument("--cap-tools", action="store_true", help="cap each tool result at 1,200 tokens")
    g.add_argument("--loop-guard", action="store_true", help="stop a run at 45 turns")
    g.add_argument("--verify", action="store_true", help="verification pass before exit")
    g.add_argument("--all", action="store_true", help="all four switches on")
    g.add_argument("--reasoning", choices=["max", "high", "sandwich"], default="max", help="reasoning effort per step (default: max)")
    g.add_argument("--break-cache", action="store_true", help="experiment: cache breakpoints set, timestamp still in the prompt")


def config_from(args) -> Config:
    every = getattr(args, "all", False)
    return Config(
        stable_prefix=every or args.stable_prefix,
        cap_tools=every or args.cap_tools,
        loop_guard=every or args.loop_guard,
        verify=every or args.verify,
        reasoning=args.reasoning,
        break_cache=args.break_cache,
        fail_cost=args.fail_cost,
    )


def main(argv: List[str]) -> None:
    commands = {"run", "ladder", "modes", "failcost", "frontier", "trace"}
    if not argv or argv[0] not in commands and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv       # bare switches mean "run"

    fail = argparse.ArgumentParser(add_help=False)
    fail.add_argument("--fail-cost", type=float, default=0.0, metavar="USD", help="cleanup cost of one failed task (default: 0)")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    p_run = sub.add_parser("run", parents=[fail], help="ledger for one configuration (the default)")
    add_switches(p_run)
    p_ladder = sub.add_parser("ladder", parents=[fail], help="turn the switches on one at a time")
    p_ladder.add_argument("--pause", action="store_true", help="wait for Enter before each step")
    sub.add_parser("modes", parents=[fail], help="compare the three reasoning modes")
    sub.add_parser("failcost", help="sweep the cleanup cost of a failure")
    p_front = sub.add_parser("frontier", parents=[fail], help="all 48 configurations")
    p_front.add_argument("--every", action="store_true", help="print all 48, not the 10 cheapest")
    p_trace = sub.add_parser("trace", parents=[fail], help="one run, turn by turn")
    add_switches(p_trace)
    p_trace.add_argument("--stuck", action="store_true", help="trace a stuck run")
    p_trace.add_argument("--full", action="store_true", help="print every turn")

    args = parser.parse_args(argv)
    if args.command == "run":
        print_ledger(evaluate(config_from(args)))
    elif args.command == "ladder":
        cmd_ladder(args)
    elif args.command == "modes":
        cmd_modes(args)
    elif args.command == "failcost":
        cmd_failcost(args)
    elif args.command == "frontier":
        cmd_frontier(args)
    elif args.command == "trace":
        cmd_trace(args, config_from(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (KeyboardInterrupt, BrokenPipeError):
        pass