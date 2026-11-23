#!/usr/bin/env python3
"""
EXERCISE: Build a decision model out of a chat model's first token.

In Part 3 you sent typed questions to a decision model and got back
probabilities. It looked like a different kind of model. It is mostly a
different way of READING a model.

Every time a language model produces a token, it first computes a
probability for every token in its vocabulary, and then one of them is
picked. A chat API normally shows you only the pick. Ask for `logprobs` and
it shows you the runners-up as well:

    "Is every figure in this reply supported by the evidence? yes or no."

      token       logprob     probability
      "yes"       -0.23         0.795          <- what a chat API would have printed
      "no"        -1.60         0.201
      "Yes"       -5.98         0.003
      "<think>"   -7.35         0.001
      "No"        -8.60         0.000
      " yes"      -8.60         0.000
      ...

Stop the model after that one token and you have what a decision model
returns: a probability for each permitted answer, in one forward pass, with
nothing generated.

A purpose-built decision model keeps that shape and replaces the crude
parts. Kev, the one you ran in Part 3, publishes how: the same kind of
language-model backbone, read once and never asked to generate; a small
trained head that scores the permitted options directly instead of reading
them off the vocabulary; an adapter fit on labeled decisions; and one
calibration temperature applied when the model loads. You will build the
untrained version of each of those here, including the temperature.

    recorded.json ─► option_probs ─► temper ─► noul / choice / score ─► a typed answer
     114 questions,   tokens to       fix the    the three System One      with a confidence
     20 candidates    permitted       model's    question types
     each             answers         over- or
                                      under-
                                      confidence
                                          ▲
                                          │
                              fit_temperature, scored by
                              expected_calibration_error
                                                                  cascade ─► coverage, errors
                                                                             shipped, $ per decision

The recorded candidates in `fixtures/recorded.json` are real: the open chat
model from Part 2 answered 66 yes/no questions about the labeled drafts
from Part 3 and 48 choice questions about labeled customer messages, one
token each (`record_logprobs.py` is the script). Everything below runs
offline on that recording.

------------------------------------------------------------------------------
WHAT YOU IMPLEMENT
------------------------------------------------------------------------------
You only need to edit the eight functions marked with `TODO(student)` below:

    option_probs                  candidate tokens -> a probability per permitted answer
    temper                        temperature scaling of a distribution
    noul                          a yes/no question -> P(yes)
    choice                        a multiple-choice question -> answer, probabilities, confidence
    score                         an ordered scale -> level, expected level, confidence
    expected_calibration_error    does "80% confident" mean right 80% of the time?
    fit_temperature               the one number that repairs a model's confidence
    cascade                       what a threshold does to coverage, errors, and cost

Everything else (`load_recorded`, `auc`, `nll`) is provided.

Run the tests to check your work:

    pytest -v

Then see what your decision model is worth on the recorded questions, next
to a purpose-built one:

    python run_demo.py

A reference implementation lives in `reference_solution.py`. Try not to
peek until you've given it a real shot.
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
    """Mean negative log-likelihood of the true answers after temperature scaling. Lower is better.

    `examples` is [(probabilities over the permitted answers, the true answer)].
    A model that puts probability 1 on every true answer scores 0. A model
    that is confidently wrong is punished hard: -log(0.01) is 4.6.
    """
    if not examples:
        return 0.0
    return -sum(math.log(temper(probs, temperature)[truth]) for probs, truth in examples) / len(examples)


# --------------------------------------------------------------------------- #
# 1. From tokens to answers
# --------------------------------------------------------------------------- #
def option_probs(top: Candidates, options: Sequence[str], floor: float = 1e-6) -> Probs:
    """Turn first-token candidates into a probability for each permitted answer.

    The model was asked for "yes" or "no" and its probability is spread over
    "yes", "Yes", " yes", "no", "No", plus tokens that are not answers at
    all ("<think>", "The"). A decision keeps only the permitted answers:

      1. A candidate counts for an option when its token equals the option
         after stripping surrounding whitespace and ignoring case. " Yes"
         counts for "yes". "y" and "yesterday" do not.
      2. An option's mass is the sum of exp(logprob) over its candidates.
      3. An option with less mass than `floor` (including one that never
         appeared among the candidates) gets `floor`. A probability of
         exactly 0 would make log-likelihoods infinite later.
      4. Divide by the total so the permitted answers sum to 1. The mass
         that sat on junk tokens is discarded by this step.

    Return a dict with one key per option, in the order of `options`.

    Example:
        top = [{"token": "yes", "logprob": math.log(0.6)},
               {"token": "no",  "logprob": math.log(0.2)},
               {"token": " Yes", "logprob": math.log(0.1)},
               {"token": "The", "logprob": math.log(0.1)}]
        option_probs(top, ["yes", "no"])  ->  {"yes": 0.7 / 0.9, "no": 0.2 / 0.9}
    """
    # TODO(student): implement the four steps above.
    raise NotImplementedError("Implement option_probs")


# --------------------------------------------------------------------------- #
# 2. Temperature
# --------------------------------------------------------------------------- #
def temper(probs: Probs, temperature: float) -> Probs:
    """Temperature-scale a distribution.

    Each probability is raised to the power 1/T and the result is
    renormalized:

        p_i' = p_i^(1/T) / sum_j p_j^(1/T)

    T = 1 changes nothing. T > 1 flattens the distribution (less
    confident). T < 1 sharpens it (more confident). The ranking of the
    answers never changes, so accuracy never changes: temperature only
    moves the confidence.

    Work in log space (log p / T), subtract the largest value before
    exponentiating, and the arithmetic stays safe for tiny probabilities.

    Raise ValueError if `temperature` is not positive.
    Return a new dict with the same keys in the same order.
    """
    # TODO(student): implement temperature scaling.
    raise NotImplementedError("Implement temper")


# --------------------------------------------------------------------------- #
# 3. The three question types
# --------------------------------------------------------------------------- #
def noul(top: Candidates, temperature: float = 1.0) -> float:
    """A yes/no question: P(yes).

    The permitted answers are "yes" and "no". Use `option_probs`, then
    `temper`, and return the probability of "yes" as a float in (0, 1).
    """
    # TODO(student): implement noul using option_probs and temper.
    raise NotImplementedError("Implement noul")


def choice(top: Candidates, letters: dict[str, str], temperature: float = 1.0) -> dict:
    """A multiple-choice question: the answer, the probabilities, and a confidence.

    The model was shown lettered options and asked for one letter.
    `letters` maps each letter to the option's name, for example
    {"A": "refunds", "B": "returns", ...}. The permitted first tokens are
    the letters; the answer is reported by name.

    Return a dict:
        "choice"          the name of the most probable option (on a tie,
                          the one that comes first in `letters`)
        "probabilities"   {name: probability}, in the order of `letters`
        "confidence"      how far the winner is above a coin toss among K
                          options, rescaled to 0..1:

                              (p_max - 1/K) / (1 - 1/K)

                          A uniform distribution has confidence 0. All the
                          mass on one option has confidence 1. This is the
                          formula the System One API uses for choice
                          questions, so 0.90 on four options and 0.90 on
                          two options do not mean the same thing.

    Apply `temper` to the letter probabilities before anything else.
    """
    # TODO(student): implement choice.
    raise NotImplementedError("Implement choice")


def score(top: Candidates, levels: int, temperature: float = 1.0) -> dict:
    """An ordered scale from 1 to `levels`: the level, the expected level, and a confidence.

    The permitted first tokens are the digits "1" .. str(levels).

    Return a dict:
        "score"           the most probable level, as an int (on a tie, the
                          lower level)
        "expected"        the probability-weighted mean level, a float
        "probabilities"   {level (int): probability}, lowest level first
        "confidence"      how tightly the mass sits around the chosen level,
                          compared with a model that knows nothing:

                              max(0, 1 - spread / baseline)

                          spread     the probability-weighted mean distance
                                     from the chosen level, E|level - score|
                          baseline   the same mean distance when every level
                                     has equal mass, measured from the middle
                                     of the scale:

                                         (levels * levels // 4) / levels

                                     which is 0.5 for 2 levels, 0.667 for 3,
                                     1.0 for 4, and 1.2 for 5.

                          All the mass on one level has confidence 1. Unlike
                          a choice question, being wrong by one level costs
                          less than being wrong by four: the answers are
                          ordered and the confidence knows it. This is the
                          formula the System One API uses for score
                          questions.

    Two differences from the API are only labeling: it numbers levels from 0
    and it reports the expected level as its "score". Here levels start at 1
    (the digits the chat model was asked for) and the most probable level is
    "score".

    Apply `temper` to the level probabilities before anything else.
    `levels` is at least 2.
    """
    # TODO(student): implement score.
    raise NotImplementedError("Implement score")


# --------------------------------------------------------------------------- #
# 4. Is the confidence worth anything?
# --------------------------------------------------------------------------- #
def expected_calibration_error(scored: Sequence[tuple[float, bool]], bins: int = 5) -> float:
    """How far "N% confident" is from "right N% of the time".

    `scored` is [(the model's confidence in the answer it gave, whether that
    answer was correct)]. A threshold on a confidence is only as good as the
    confidence. If the model says 0.9 and is right 60% of the time, a gate
    at 0.9 ships far more errors than its owner believes.

      1. Split [0, 1] into `bins` equal-width bins. An item with confidence
         c belongs to bin int(c * bins); a confidence of exactly 1.0 goes in
         the last bin.
      2. For each bin that has items, compare the mean confidence of its
         items with the share of them that were correct.
      3. ECE is the average of those gaps |accuracy - mean confidence|,
         weighted by the share of all items that fell in each bin.

    0.0 is perfect calibration. Return 0.0 for an empty list.
    """
    # TODO(student): implement expected calibration error.
    raise NotImplementedError("Implement expected_calibration_error")


def fit_temperature(examples: Sequence[tuple[Probs, str]], grid: Sequence[float] | None = None) -> float:
    """Find the temperature that makes the model's confidence fit labeled data.

    `examples` is [(probabilities over the permitted answers, the true
    answer)], as `nll` takes. Try every temperature in `grid` and return the
    one with the lowest `nll(examples, t)`. On a tie, return the lowest
    temperature.

    When `grid` is None use 40 points from 0.25 to about 10:

        [round(0.25 * 1.1**i, 4) for i in range(40)]

    A result above 1 means the model was overconfident. Below 1, it was
    underconfident. This is one number learned from a few dozen labeled
    examples, and it is the cheapest calibration method there is: it cannot
    change which answer the model picks, only how sure it claims to be.
    """
    # TODO(student): search the grid for the temperature with the lowest NLL.
    raise NotImplementedError("Implement fit_temperature")


# --------------------------------------------------------------------------- #
# 5. What a decision is worth
# --------------------------------------------------------------------------- #
def cascade(
    scored: Sequence[tuple[float, bool]],
    accept_at: float,
    gate_usd: float,
    escalate_usd: float,
) -> dict:
    """What one threshold does to a stream of decisions.

    `scored` is [(confidence, whether acting on it would have been right)].
    The decision model answers every item (each costs `gate_usd`). Items
    with confidence >= `accept_at` are taken on its word. The rest are
    escalated to something that costs `escalate_usd` more and is assumed to
    get them right (a stronger model, a closing pass, a person).

    Return a dict:
        "coverage"           the share of items taken on the model's word
        "error_rate"         the share of ALL items that were taken and
                             were wrong: errors shipped per decision
        "usd_per_decision"   gate_usd + (1 - coverage) * escalate_usd

    Return zeros for an empty list.

    Lowering the threshold buys coverage (cheaper) with error rate (worse).
    There is no correct threshold, only the one that fits what an error
    costs you, which is the same trade the lecture made with fail cost.
    """
    # TODO(student): implement the cascade economics.
    raise NotImplementedError("Implement cascade")


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
