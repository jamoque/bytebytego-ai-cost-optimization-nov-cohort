#!/usr/bin/env python3
"""
REFERENCE SOLUTION: a decision model built from a chat model's first token.

Don't peek until you have tried `decision_model.py` yourself. The tests can
be pointed at this file with:

    DECISION_MODEL_IMPL=reference_solution pytest -v

See `decision_model.py` for the full explanation of each function.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).parent

Candidates = Sequence[dict]            # [{"token": "yes", "logprob": -0.23}, ...] most likely first
Probs = dict[str, float]               # permitted answer -> probability, sums to 1


# --------------------------------------------------------------------------- #
# PROVIDED helpers
# --------------------------------------------------------------------------- #
def load_recorded(path: str | Path | None = None) -> dict:
    """The recorded first-token candidates (see record_logprobs.py)."""
    return json.loads(Path(path or HERE / "fixtures" / "recorded.json").read_text())


def auc(scored: Sequence[tuple[float, bool]]) -> float:
    """Probability that a random true item is scored above a random false one (0.5 = no signal)."""
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return 0.5
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def nll(examples: Sequence[tuple[Probs, str]], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the true answers after temperature scaling. Lower is better."""
    if not examples:
        return 0.0
    return -sum(math.log(temper(probs, temperature)[truth]) for probs, truth in examples) / len(examples)


# --------------------------------------------------------------------------- #
# 1. From tokens to answers
# --------------------------------------------------------------------------- #
def option_probs(top: Candidates, options: Sequence[str], floor: float = 1e-6) -> Probs:
    mass = {option: 0.0 for option in options}
    by_key = {option.strip().casefold(): option for option in options}
    for candidate in top:
        option = by_key.get(candidate["token"].strip().casefold())
        if option is not None:
            mass[option] += math.exp(candidate["logprob"])
    mass = {option: max(m, floor) for option, m in mass.items()}
    total = sum(mass.values())
    return {option: m / total for option, m in mass.items()}


# --------------------------------------------------------------------------- #
# 2. Temperature
# --------------------------------------------------------------------------- #
def temper(probs: Probs, temperature: float) -> Probs:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logs = {option: math.log(p) / temperature for option, p in probs.items()}
    peak = max(logs.values())
    scaled = {option: math.exp(value - peak) for option, value in logs.items()}
    total = sum(scaled.values())
    return {option: value / total for option, value in scaled.items()}


# --------------------------------------------------------------------------- #
# 3. The three question types
# --------------------------------------------------------------------------- #
def noul(top: Candidates, temperature: float = 1.0) -> float:
    return temper(option_probs(top, ["yes", "no"]), temperature)["yes"]


def choice(top: Candidates, letters: dict[str, str], temperature: float = 1.0) -> dict:
    by_letter = temper(option_probs(top, list(letters)), temperature)
    probabilities = {letters[letter]: p for letter, p in by_letter.items()}
    best = max(probabilities, key=probabilities.get)
    k = len(probabilities)
    confidence = (probabilities[best] - 1.0 / k) / (1.0 - 1.0 / k) if k > 1 else 1.0
    return {"choice": best, "probabilities": probabilities, "confidence": max(0.0, confidence)}


def score(top: Candidates, levels: int, temperature: float = 1.0) -> dict:
    labels = [str(i) for i in range(1, levels + 1)]
    by_label = temper(option_probs(top, labels), temperature)
    probabilities = {int(label): p for label, p in by_label.items()}
    mode = max(probabilities, key=probabilities.get)
    expected = sum(level * p for level, p in probabilities.items())
    spread = sum(abs(level - mode) * p for level, p in probabilities.items())
    baseline = (levels * levels // 4) / levels      # the spread of equal mass on every level, from the middle
    confidence = max(0.0, 1.0 - spread / baseline) if levels > 1 else 1.0
    return {"score": mode, "expected": expected, "probabilities": probabilities, "confidence": confidence}


# --------------------------------------------------------------------------- #
# 4. Is the confidence worth anything?
# --------------------------------------------------------------------------- #
def expected_calibration_error(scored: Sequence[tuple[float, bool]], bins: int = 5) -> float:
    if not scored:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in scored:
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        error += len(bucket) / len(scored) * abs(accuracy - mean_confidence)
    return error


def fit_temperature(examples: Sequence[tuple[Probs, str]], grid: Sequence[float] | None = None) -> float:
    grid = list(grid) if grid else [round(0.25 * 1.1**i, 4) for i in range(40)]   # 0.25 ... about 10
    return min(grid, key=lambda t: (nll(examples, t), t))


# --------------------------------------------------------------------------- #
# 5. What a decision is worth
# --------------------------------------------------------------------------- #
def cascade(
    scored: Sequence[tuple[float, bool]],
    accept_at: float,
    gate_usd: float,
    escalate_usd: float,
) -> dict:
    n = len(scored)
    if n == 0:
        return {"coverage": 0.0, "error_rate": 0.0, "usd_per_decision": 0.0}
    taken = [correct for confidence, correct in scored if confidence >= accept_at]
    coverage = len(taken) / n
    return {
        "coverage": coverage,
        "error_rate": sum(1 for correct in taken if not correct) / n,
        "usd_per_decision": gate_usd + (1.0 - coverage) * escalate_usd,
    }


__all__ = [
    "load_recorded",
    "auc",
    "nll",
    "option_probs",
    "temper",
    "noul",
    "choice",
    "score",
    "expected_calibration_error",
    "fit_temperature",
    "cascade",
]
