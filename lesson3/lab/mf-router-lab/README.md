# MF Router Lab

Train a **custom matrix-factorization router for RouteLLM**: the same model
architecture as the `mf` router you downloaded in Part 1.6, fit to battles
between *your* two tiers on *your* traffic, saved in RouteLLM's checkpoint
format, and loaded back into the hands-on lab's `preference` strategy.

The hands-on system (three priced tiers behind the Lab 1 LiteLLM mesh,
retrieval from the Lab 2 index, RouteLLM's published router) lives in
[`../routing-lab/`](../routing-lab/). Work through Part 1.6 of that first:
this exercise replaces the weights of the router you calibrated there.

## The big idea

RouteLLM's published router learned from Chatbot Arena votes between models
such as GPT-4 and Mixtral on public chatbot traffic. It predicts
P(strong wins) for a prompt it has never seen, a domain it has never seen,
and two models it has never seen. A router is a small supervised model, and
you can train one on your own data:

```
battles.json ──► (winner, loser, prompt) pairs ──► train ──► checkpoint ──► RouteLLM
 307 Northstar questions,                          a 1536→128 projection,
 weak vs strong, judged                            two model vectors, one classifier
```

- **A battle** is one prompt answered by two models, plus a verdict. Here
  the weak and strong tiers both answered through the Part 1 pipeline and a
  judge decided whether the weak answer reached the strong answer's
  conclusion. That labeling is the only part of training a router that costs
  real money, and it is already done (`collect_battles.py` is the script).
- **The model** scores a (model, prompt) pair as
  `w · (normalize(P[model]) ⊙ W·e(prompt))`. P(strong wins) is the sigmoid of
  the difference between two models' scores on the same prompt.
- **Training** pushes `score(winner) − score(loser)` up on every battle.
- **A tie is a win for the cheaper model.** On a support desk most battles
  are ties, and "the cheap answer was fine" is what a cost router has to
  learn.
- **AUC first, threshold second.** AUC says whether the router ranks prompts
  correctly. The threshold is a budget decision you make afterwards, exactly
  as in Part 1.6.

## Setup

Use the hands-on lab's environment (it already has PyTorch, which arrives
with `routellm`):

```bash
cd lesson3/lab/mf-router-lab
source ../routing-lab/.venv/bin/activate
pip install -r requirements.txt     # adds pytest

# 1. Fill in the TODOs in mf_router.py, then:
pytest -v                           # your definition of "done"; offline, a few seconds

# 2. Train on the real battles (the Lab 3 mesh must be running for the
#    first run: one embedding request, well under a cent, then cached):
python train.py
```

A correct implementation prints a held-out AUC for your router next to the
published router's AUC on the same battles, a table of what each
strong-tier budget catches, and the path of the checkpoint it saved. Then
follow the last three lines it prints to route real questions with it.

Training is seeded, but floating-point results differ slightly between
machines, so expect your numbers to be close to ours and not identical.

## What we measured

One machine, November 2026, `python train.py` with seeds 0 to 3:

| Router | Held-out AUC on the 307 battles |
| --- | --- |
| A router that knows nothing | 0.50 |
| RouteLLM published `mf` | 0.60 |
| Trained here with RouteLLM's hyperparameters (`--lr 3e-4 --alpha 0.1`) | 0.51 to 0.56 |
| Trained here from random weights (the defaults) | 0.68 to 0.73 |
| Trained here starting from the published weights (`--init published`) | 0.70 to 0.74 |

That is a real improvement and a modest one. A question's text only partly
determines whether the weak model gets it wrong, and 307 battles is a small
dataset. Read the per-fold AUCs that `train.py` prints before you believe
any single number.

## Layout

```
mf-router-lab/
├── README.md                # you are here
├── mf_router.py             # <- start here (stubs to implement)
├── test_mf_router.py        # test suite (offline)
├── reference_solution.py    # complete solution (don't peek early)
├── train.py                 # embeds, cross-validates, trains, exports
├── collect_battles.py       # how battles.json was made; run it to make more
├── fixtures/battles.json    # 307 judged battles, weak tier vs strong tier
└── requirements.txt
```

`train.py` writes `.cache/` (prompt embeddings) and `checkpoints/` (your
routers). Both are git-ignored.

## Prerequisites

- The hands-on lab's virtual environment (`../routing-lab/.venv`).
- The Lab 3 mesh running, for the first `train.py` run and for routing with
  your checkpoint afterwards.

## References

- Hands-on lab: [`../routing-lab/`](../routing-lab/)
- Hands-on overview: [`../exercise_overview.md`](../exercise_overview.md)
- RouteLLM paper: https://arxiv.org/abs/2406.18665
- RouteLLM's matrix-factorization model and training script: https://github.com/lm-sys/RouteLLM/tree/main/routellm/routers/matrix_factorization
- The published checkpoint: https://huggingface.co/routellm/mf_gpt4_augmented
