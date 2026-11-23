#!/usr/bin/env python3
"""
What your decision model is worth (PROVIDED).

    python run_demo.py                                   # your functions on the recorded questions
    python run_demo.py --escalate-usd 0.02               # a more expensive thing to escalate to
    python run_demo.py --live "My card was declined"     # one new question to the model on your machine

Runs your `decision_model.py` over fixtures/recorded.json (114 real
first-token recordings from the Part 2 chat model) and prints three things:

  1. how well its yes/no probabilities separate good drafts from bad ones,
     next to the purpose-built decision model from Part 3 on the same drafts
  2. how well it sorts messages into categories, and whether its confidence
     can be trusted before and after you fit a temperature
  3. what a threshold on that confidence buys: coverage, errors shipped, and
     dollars per decision

`--live` needs Ollama running with the lab's chat model. Everything else is
offline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
impl = importlib.import_module(os.environ.get("DECISION_MODEL_IMPL", "decision_model"))


def pct(x: float) -> str:
    return f"{x:.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usd-per-hour", type=float, default=0.24, help="what an hour of your machine costs (config.yaml's local.usd_per_hour)")
    parser.add_argument("--escalate-usd", type=float, default=0.004, help="cost of what an unsure decision is escalated to (default: one closing pass on the mid tier)")
    parser.add_argument("--live", metavar="MESSAGE", help="classify one new customer message with the model on your machine")
    parser.add_argument("--model", default="lab4-agent")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    recorded = impl.load_recorded()
    kev = json.loads((HERE / "fixtures" / "kev_scores.json").read_text())
    letters = recorded["letters"]

    if args.live:
        import httpx

        from record_logprobs import CHOICE_SYSTEM, choice_prompt, first_token

        prompt, letters = choice_prompt(args.live, recorded["options"])
        with httpx.Client(base_url=args.base_url, timeout=600, headers={"Authorization": "Bearer ollama"}) as client:
            top, seconds = first_token(client, args.model, CHOICE_SYSTEM, prompt, 20)
        answer = impl.choice(top, letters)
        print(f"\n{args.live}\n")
        for name, p in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1]):
            print(f"  {name:<16}{p:>7.3f}  {'█' * round(p * 40)}")
        print(f"\n  choice: {answer['choice']}   confidence: {answer['confidence']:.2f}   one token, {seconds:.2f}s\n")
        return

    gate_usd = recorded["seconds_per_question"] * args.usd_per_hour / 3600.0

    # ---- 1. yes/no: is the draft supported? --------------------------------- #
    noul_items = recorded["noul"]
    mine = [(impl.noul(item["top"]), item["label"]) for item in noul_items]
    theirs = [(kev["noul"][item["id"]], item["label"]) for item in noul_items]
    good = sum(1 for _, label in mine if label)
    print()
    print(f"1. Yes/no: {len(mine)} labeled drafts ({good} good, {len(mine) - good} bad). Question: is every figure in the reply supported?")
    print()
    print(f"   {'':<34}{'AUC':>6}{'sec/question':>15}")
    print(f"   {'Your model (' + recorded['model'] + ', one token)':<34}{impl.auc(mine):>6.2f}{recorded['seconds_per_question']:>15.2f}")
    print(f"   {'Purpose-built (' + kev['model'].split(' ')[0] + ')':<34}{impl.auc(theirs):>6.2f}{kev['seconds_per_question']['noul']:>15.2f}")
    print("   AUC 0.50 is a coin toss. Both models are reading the same drafts and the same evidence.")

    # ---- 2. choice: which category? ------------------------------------------ #
    items = recorded["choice"]
    answers = [impl.choice(item["top"], letters) for item in items]
    right = [a["choice"] == item["label"] for a, item in zip(answers, items)]
    kev_right = sum(1 for item in items if kev["choice"][item["id"]]["choice"] == item["label"])
    conf_right = [a["confidence"] for a, ok in zip(answers, right) if ok]
    conf_wrong = [a["confidence"] for a, ok in zip(answers, right) if not ok]
    print()
    print(f"2. Choice: {len(items)} labeled messages, {len(letters)} categories.")
    print()
    print(f"   {'':<34}{'Correct':>9}{'sec/question':>15}")
    print(f"   {'Your model':<34}{str(sum(right)) + '/' + str(len(items)):>9}{recorded['seconds_per_question']:>15.2f}")
    print(f"   {'Purpose-built':<34}{str(kev_right) + '/' + str(len(items)):>9}{kev['seconds_per_question']['choice']:>15.2f}")
    if conf_wrong:
        print(f"   Your mean confidence when right: {sum(conf_right) / len(conf_right):.2f}   when wrong: {sum(conf_wrong) / len(conf_wrong):.2f}")

    # Fit a temperature on every other message, judge it on the rest.
    examples = [(a["probabilities"], item["label"]) for a, item in zip(answers, items)]
    fit, held = examples[0::2], examples[1::2]
    t = impl.fit_temperature(fit)

    def calibration(temperature: float) -> tuple[float, float]:
        scored = []
        for item in items[1::2]:
            a = impl.choice(item["top"], letters, temperature)
            scored.append((max(a["probabilities"].values()), a["choice"] == item["label"]))
        return impl.expected_calibration_error(scored), impl.nll(held, temperature)

    ece0, nll0 = calibration(1.0)
    ece1, nll1 = calibration(t)
    print()
    print(f"   Temperature fit on {len(fit)} messages: T = {t:.2f}   ({'overconfident' if t > 1 else 'underconfident' if t < 1 else 'calibrated'} before the fit)")
    print(f"   {'On the other ' + str(len(held)) + ' messages':<34}{'ECE':>9}{'NLL':>9}")
    print(f"   {'as recorded (T = 1.00)':<34}{ece0:>9.3f}{nll0:>9.3f}")
    print(f"   {'after the fit (T = ' + format(t, '.2f') + ')':<34}{ece1:>9.3f}{nll1:>9.3f}")

    # ---- 3. the cascade ------------------------------------------------------- #
    print()
    print("3. A threshold on the yes/no probability, as a gate in front of something expensive.")
    print(f"   Your decision costs ${gate_usd:.5f} ({recorded['seconds_per_question']}s of your machine). Escalating costs ${args.escalate_usd:.4f} more.")
    print()
    print(f"   {'accept_at':>9}{'coverage':>10}{'errors shipped':>16}{'$/decision':>12}{'vs always escalating':>22}")
    always = args.escalate_usd          # no gate: every draft gets the expensive treatment
    for accept_at in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85):
        row = impl.cascade(mine, accept_at, gate_usd, args.escalate_usd)
        print(f"   {accept_at:>9.2f}{pct(row['coverage']):>10}{pct(row['error_rate']):>16}{row['usd_per_decision']:>12.5f}{pct(row['usd_per_decision'] / always):>22}")
    print()
    print("   Coverage is free money and errors shipped is what it costs. Which row you want depends on")
    print("   what a wrong decision costs you, and that number is not in this table.")
    print()


if __name__ == "__main__":
    main()
