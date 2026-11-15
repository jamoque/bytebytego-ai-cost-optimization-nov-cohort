#!/usr/bin/env python3
"""
Calibrate the RouteLLM threshold on your own traffic. No labels needed.

RouteLLM scores a prompt with P(the strong model's answer is preferred).
A threshold turns that score into a decision. The right threshold depends
on the distribution of scores in YOUR traffic, so RouteLLM's published
thresholds (calibrated on public chatbot traffic) rarely give the split
you expect.

    python calibrate.py                 # score the sample, print the table
    python calibrate.py --strong 0.30   # threshold that sends ~30% to strong

The sample is fixtures/traffic_sample.json: 40 unlabeled support messages
that are NOT in the graded workload. The only spend is one small embedding
call per message, through the mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from routing import PreferenceScorer, calibrate_threshold, load_config, log

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strong", type=float, default=None, help="target share of traffic for the strong tier, 0-1")
    parser.add_argument("--sample", default=str(HERE / "fixtures" / "traffic_sample.json"))
    args = parser.parse_args()

    config = load_config()
    scorer = PreferenceScorer(config)
    sample: list[str] = json.loads(Path(args.sample).read_text())
    weights = f"your checkpoint {config['preference']['checkpoint']}" if scorer.is_custom else "published weights"
    log(f"Scoring {len(sample)} messages with RouteLLM ({config['preference']['router']}, {weights})...")
    scored = sorted(((scorer.win_rate(text), text) for text in sample), reverse=True)
    rates = [rate for rate, _ in scored]

    log("\nP(strong wins) across the sample")
    log(f"  min {min(rates):.3f}   median {rates[len(rates) // 2]:.3f}   max {max(rates):.3f}")
    lo, hi = min(rates), max(rates)
    width = (hi - lo) / 8 or 1.0
    for b in range(8):
        left = lo + b * width
        n = sum(1 for r in rates if left <= r < left + width or (b == 7 and r == hi))
        log(f"  {left:.3f}-{left + width:.3f}  {'#' * n} {n}")

    log("\nHighest scores (RouteLLM thinks these need the strong model):")
    for rate, text in scored[:5]:
        log(f"  {rate:.3f}  {text[:90]}")
    log("Lowest scores:")
    for rate, text in scored[-5:]:
        log(f"  {rate:.3f}  {text[:90]}")

    log("\nThreshold for a target strong share")
    log(f"  {'strong share':>12}  {'threshold':>9}")
    for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        log(f"  {share:>11.0%}  {calibrate_threshold(rates, share):>9.5f}")

    current = float(config["preference"]["threshold"])
    actual = sum(1 for r in rates if r >= current) / len(rates)
    log(f"\nconfig.yaml preference.threshold = {current:.5f} sends {actual:.0%} of this sample to strong.")

    if args.strong is not None:
        threshold = calibrate_threshold(rates, args.strong)
        got = sum(1 for r in rates if r >= threshold) / len(rates)
        log(f"\nFor ~{args.strong:.0%} strong: set preference.threshold: {threshold:.5f}  (sends {got:.0%} of the sample)")


if __name__ == "__main__":
    main()
