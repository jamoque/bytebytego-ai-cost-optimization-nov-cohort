# Decision Model Lab

Build a **decision model out of a chat model's first token**: the three
System One question types (`noul`, `choice`, `score`), the confidence
formulas behind them, a calibration fit, and the cascade arithmetic that
turns a confidence into dollars. Then measure it against the purpose-built
decision model you used in Part 3, on the same questions.

The hands-on system (a real agent harness behind the Lab 1 LiteLLM mesh,
with an open chat model and an open decision model on your machine) lives in
[`../harness-lab/`](../harness-lab/). Work through Part 3 of that first:
this exercise takes apart the thing you called there.

## The big idea

Before a language model produces a token, it computes a probability for
every token it could produce. A chat API shows you the one that was picked.
Ask for `logprobs` and stop after one token, and you get the whole
distribution:

```
"Is every figure in this reply supported by the evidence? yes or no."

  token       probability
  "yes"         0.795        <- what a chat API would have printed
  "no"          0.201
  "Yes"         0.003
  "<think>"     0.001
```

That is what a decision model returns: a probability for each permitted
answer, in one forward pass, with nothing generated.

```
recorded.json ─► option_probs ─► temper ─► noul / choice / score ─► a typed answer
 114 questions,   tokens to       repair    the three System One      with a confidence
 20 candidates    permitted       the       question types
 each             answers         confidence
                                      ▲
                          fit_temperature, scored by
                          expected_calibration_error
                                                          cascade ─► coverage, errors
                                                                     shipped, $ per decision
```

- **A decision is a read, not a generation.** One input pass, one token's
  worth of output. That is why it is fast and why it is priced on input
  only.
- **Only permitted answers count.** The model's probability is spread over
  `yes`, `Yes`, ` yes`, and tokens that are not answers at all. A decision
  keeps the permitted answers and renormalizes.
- **A confidence is a formula, and you can check yours.** The tests rebuild
  candidates from the probabilities a real decision model returned through
  `/v1/systemone` and require your `choice` and `score` to report the
  confidence the API reported, to four decimal places.
- **A confidence is only useful if it is calibrated.** A model that says
  0.9 and is right 60% of the time makes every threshold a lie. Temperature
  scaling is one number, fit on a few dozen labeled examples, that repairs
  this without changing a single answer.
- **A threshold is a price.** Lower it and more decisions are taken on the
  cheap model's word (coverage) and more of them are wrong (errors
  shipped). The cascade function puts both next to dollars per decision.
- **Training is the difference.** Your one-token decision model and the
  purpose-built one both read the input once and generate nothing. The
  purpose-built one (Kev publishes its design) scores the options with a
  small trained head instead of reading them off the vocabulary, was fit
  on labeled decisions, and ships with its own calibration temperature.
  The demo shows you how much that buys.

## Setup

Use the hands-on lab's environment:

```bash
cd lesson4/lab/decision-model-lab
source ../harness-lab/.venv/bin/activate
pip install -r requirements.txt     # adds pytest

# 1. Fill in the TODOs in decision_model.py, then:
pytest -v                           # your definition of "done"; offline, under a second

# 2. See what your decision model is worth on the recorded questions:
python run_demo.py
```

Nothing here calls a paid API. The tests and the demo run offline on
`fixtures/recorded.json`. Two optional steps talk to Ollama on your machine:

```bash
# One new message, classified live by the Part 2 chat model through your functions:
python run_demo.py --live "We were double charged and then locked out of the dashboard."

# Record your own model instead of ours (about two minutes), then rerun the demo:
python record_logprobs.py --model lab4-agent
```

## What we measured

One machine (Apple M1 Max), November 2026, `python run_demo.py` with the
reference solution. "One token" is `qwen3.5:4b`, the Part 2 chat model, read
through the functions you implement. "Purpose-built" is `kev:4b`, the Part 3
decision model, on the same 114 questions.

| | One token of a chat model | Purpose-built decision model |
| --- | --- | --- |
| Yes/no, 66 labeled drafts: AUC | 0.78 | 0.80 |
| Yes/no: seconds per question | 1.03 | 2.37 |
| Choice, 48 messages, 8 categories: correct | 41/48 | 47/48 |
| Choice: seconds per question | 1.03 | 0.40 |

On the choice questions the chat model was overconfident. It was 0.99
confident on average when right and 0.70 confident when wrong. A
temperature fit on half the messages gave T = 1.39, and on the other half
that moved the calibration error from 0.113 to 0.079 and the negative
log-likelihood from 0.411 to 0.360.

Two things to take from the table. On a hard question (is this reply
supported by four pages of evidence?) neither small model is a reliable
judge, and training buys little. On an easy, well-posed one (which of eight
categories?) the purpose-built model is clearly better and its one mistake
came with a confidence of 0.30, which a threshold would have caught.

## Layout

```
decision-model-lab/
├── README.md                # you are here
├── decision_model.py        # <- start here (stubs to implement)
├── test_decision_model.py   # test suite (offline)
├── reference_solution.py    # complete solution (don't peek early)
├── run_demo.py              # your functions on the recording, next to Kev
├── record_logprobs.py       # how recorded.json was made; run it to record your own model
├── fixtures/
│   ├── recorded.json           # 114 questions, the 20 most likely first tokens for each
│   ├── kev_scores.json         # the purpose-built model's answers to the same questions
│   ├── systemone_answers.json  # 54 real /v1/systemone answers; the tests check your confidence against theirs
│   └── messages.json           # 48 labeled customer messages, 8 categories
└── requirements.txt
```

## Prerequisites

- The hands-on lab's virtual environment (`../harness-lab/.venv`).
- For `--live` and `record_logprobs.py` only: Ollama running with the
  `lab4-agent` model from Part 2. The mesh is not needed, because these two
  talk to Ollama's OpenAI-compatible endpoint directly: that is where
  `logprobs` lives.

## References

- Hands-on lab: [`../harness-lab/`](../harness-lab/)
- Hands-on overview: [`../exercise_overview.md`](../exercise_overview.md)
- On Calibration of Modern Neural Networks (temperature scaling, ECE): https://arxiv.org/abs/1706.04599
- Ollama's OpenAI-compatible API (`logprobs`, `top_logprobs`): https://docs.ollama.com/api/openai-compatibility
