#!/usr/bin/env python3
"""
EXERCISE: Train a custom matrix-factorization router for RouteLLM.

In Part 1.6 you downloaded RouteLLM's `mf` router and used it as a black
box: a prompt went in, P(strong wins) came out. That model was trained on
Chatbot Arena votes between models such as GPT-4 and Mixtral, on public
chatbot traffic. It has never seen a Northstar policy, a Northstar ticket,
or either of the two models your mesh actually serves.

Here you train the same architecture on battles between YOUR two tiers on
YOUR traffic (`fixtures/battles.json`), and save it in RouteLLM's checkpoint
format so the hands-on lab's `preference` strategy can load it unchanged.

    battles.json ──► battle_pairs ──► (winner, loser, prompt) pairs
                                           │
    prompt ──► mesh-embed ──► e(prompt) ───┤   embedded once, then frozen
                                           ▼
                                      train_epoch × N      BCE on score(winner) − score(loser)
                                           │
                                           ▼
                          checkpoints/northstar-mf/model.safetensors
                                           │
                                           ▼
        routing-lab/config.yaml  preference.checkpoint  ──►  arm F of the experiment

------------------------------------------------------------------------------
THE MODEL
------------------------------------------------------------------------------
"Matrix factorization" is the recommender-system idea: predict an unseen
(user, item) rating from the dot product of a learned user vector and a
learned item vector. Here the "users" are models, the "items" are prompts,
and the "rating" is how well that model handles that prompt:

    score(model, prompt) = w · ( normalize(P[model]) ⊙ (W · e(prompt)) )

      e(prompt)   the prompt's text-embedding-3-small vector (1536 numbers),
                  computed once by the mesh's `mesh-embed` alias and frozen
      W           `text_proj`: a 1536 → 128 linear map, learned
      P[model]    one learned 128-number vector per model, L2-normalized
      ⊙           element-wise product
      w           `classifier`: a 128 → 1 linear map, learned, no bias

A router never needs the score itself, only a comparison of two models on
the same prompt:

    P(strong wins | prompt) = sigmoid( score(strong, prompt) − score(weak, prompt) )

Training data is a list of battles. Each one says "on this prompt, this
model won and that one lost". Training pushes
`score(winner) − score(loser)` up on every battle (binary cross-entropy
against a target of 1). That is the whole algorithm.

------------------------------------------------------------------------------
COMPATIBILITY WITH ROUTELLM
------------------------------------------------------------------------------
`MFModel` below has the same three parameter tensors, with the same names
and shapes, as RouteLLM's own `MFModel`. RouteLLM keeps a table of 64 model
vectors and, by default, scores row 24 ("gpt-4-1106-preview") against row
36 ("mixtral-8x7b-instruct-v0.1"). This lab keeps the 64-row table and
reuses those two rows for your strong and weak tiers, so that

  * a checkpoint you save here loads in RouteLLM with no code changes, and
  * you can start training from RouteLLM's published weights instead of
    from random ones (`python train.py --init published`).

------------------------------------------------------------------------------
WHAT YOU IMPLEMENT
------------------------------------------------------------------------------
You only need to edit the seven functions marked with `TODO(student)` below:

  1. MFModel.score      -> how well a model handles a prompt, as a logit
  2. MFModel.win_logit  -> score(winner) − score(loser)
  3. battle_pairs       -> judged battles → (winner, loser) pairs; what a tie means
  4. train_epoch        -> shuffle, add noise, BCE against "the winner won", step
  5. strong_win_rate    -> P(strong wins) for each prompt: the router's output
  6. auc                -> does the router rank prompts correctly? no threshold needed
  7. budget_table       -> what each strong-tier budget catches, and what it misses

Everything else is provided: the constants, the model's constructor, the
multi-epoch `fit` loop, threshold calibration (the same function as Part
1.6), and checkpoint export. This file needs PyTorch, which the hands-on
lab's environment already has (it arrives with `routellm`).

Run `pytest` to check your work: the tests train tiny models on synthetic
data, so they need no mesh and no network. Then run `python train.py` to
train on the real battles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Constants (PROVIDED)
# --------------------------------------------------------------------------- #
TEXT_DIM = 1536      # text-embedding-3-small, the embedding RouteLLM's mf router uses
DIM = 128            # size of a model vector and of a projected prompt vector
NUM_MODELS = 64      # rows in RouteLLM's model table
STRONG_ID = 24       # RouteLLM's row for "gpt-4-1106-preview"; your strong tier lives here
WEAK_ID = 36         # RouteLLM's row for "mixtral-8x7b-instruct-v0.1"; your weak tier lives here


# ========================================================================== #
# TODO(student) #1 + #2: the model's forward pass
# ========================================================================== #
class MFModel(torch.nn.Module):
    """RouteLLM's matrix-factorization router, minus the embedding API call.

    RouteLLM's inference model embeds the prompt inside `forward`. Training
    cannot afford one API call per example per epoch, so this model takes
    prompt embeddings that were computed once, up front.
    """

    def __init__(self, dim: int = DIM, num_models: int = NUM_MODELS, text_dim: int = TEXT_DIM) -> None:
        # PROVIDED. The attribute names and the Sequential wrappers match
        # RouteLLM's MFModel exactly, so `state_dict()` here IS a RouteLLM
        # checkpoint: P.weight, text_proj.0.weight, classifier.0.weight.
        super().__init__()
        self.P = torch.nn.Embedding(num_models, dim)
        self.text_proj = torch.nn.Sequential(torch.nn.Linear(text_dim, dim, bias=False))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(dim, 1, bias=False))

    def score(self, model_ids: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """How well each model is expected to handle each prompt, as a logit.

        Args:
            model_ids:     LongTensor of shape (batch,). Row i names the model
                           to score on prompt i.
            prompt_embeds: FloatTensor of shape (batch, text_dim).

        Returns:
            FloatTensor of shape (batch,):

                w · ( normalize(P[model]) ⊙ (W · e) )

        Steps:
          1. Look up the model vectors: `self.P(model_ids)` → (batch, dim).
          2. L2-normalize each model vector (`F.normalize(..., p=2, dim=1)`).
             Only a model vector's direction matters, never its length.
          3. Project the prompt embeddings with `self.text_proj` → (batch, dim).
          4. Multiply the two element-wise, apply `self.classifier`
             → (batch, 1), and drop the last dimension.
        """
        raise NotImplementedError("Implement MFModel.score")

    def win_logit(
        self,
        winner_ids: torch.Tensor,
        loser_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Logit that `winner` beats `loser` on each prompt.

        Returns `score(winner) − score(loser)`, shape (batch,). A sigmoid of
        this is a win probability. It is 0 (a coin flip) when the two ids
        are the same model, and it flips sign when the ids are swapped.

        Because the classifier is linear and has no bias, this equals
        `classifier((normalize(P[winner]) − normalize(P[loser])) ⊙ W·e)`,
        which is how RouteLLM's training script writes it. A bias term
        would cancel in the subtraction and never be learned.
        """
        raise NotImplementedError("Implement MFModel.win_logit")


# ========================================================================== #
# TODO(student) #3: battles → training pairs
# ========================================================================== #
def battle_pairs(
    battles: Sequence[dict[str, Any]],
    model_ids: dict[str, int],
    tie_winner: str | None,
) -> tuple[list[int], list[int], list[int]]:
    """Turn judged battles into (winner, loser) training pairs.

    Each battle is a dict with at least:

        {"model_a": "mesh-gpt-mini", "model_b": "mesh-claude-sonnet",
         "winner": "model_a" | "model_b" | "tie"}

    Args:
        battles:    the judged battles, in order.
        model_ids:  model name → row in the model table,
                    e.g. {"mesh-gpt-mini": WEAK_ID, "mesh-claude-sonnet": STRONG_ID}.
        tie_winner: what to do with a tie.
                      a model name → that model is counted as the winner of
                                     every tie it took part in
                      None         → ties are dropped

    Returns:
        (keep, winners, losers), three lists of equal length:
          keep[i]     index into `battles` of the i-th training pair
          winners[i]  model-table row of the winner
          losers[i]   model-table row of the loser

    Rules:
      - "model_a" → model_a won. "model_b" → model_b won.
      - "tie" with `tie_winner` naming one of the two models → that model
        won, the other lost.
      - Drop the battle (it appears in none of the three lists) when: it is
        a tie and `tie_winner` is None or is neither of its models; its
        `winner` is any other string; or model_a and model_b are the same.

    Why ties matter: RouteLLM's training script drops them. On a support
    desk most battles ARE ties (the cheap answer was fine). Dropping them
    throws away three quarters of this dataset, and every remaining pair
    says the same thing: "strong won". A model trained that way can still
    learn what a strong-win prompt looks like, but it never sees a prompt
    where the cheap answer was enough, so its "probabilities" all drift
    toward 1. This lab scores a tie as a win for the cheaper model.
    """
    raise NotImplementedError("Implement battle_pairs")


# ========================================================================== #
# TODO(student) #4: one epoch of training
# ========================================================================== #
def train_epoch(
    model: MFModel,
    optimizer: torch.optim.Optimizer,
    embeds: torch.Tensor,
    winners: torch.Tensor,
    losers: torch.Tensor,
    *,
    batch_size: int,
    alpha: float,
    generator: torch.Generator,
) -> float:
    """One pass over the training pairs. Returns the mean loss per pair.

    Args:
        embeds:     FloatTensor (n, text_dim), the prompt embedding of each pair.
        winners:    LongTensor (n,), model-table row of each pair's winner.
        losers:     LongTensor (n,), model-table row of each pair's loser.
        batch_size: pairs per optimizer step (the last batch may be smaller).
        alpha:      standard deviation of the Gaussian noise added to each
                    prompt embedding. Noise is the regularizer: the model
                    cannot memorize a prompt it never sees the same way twice.
        generator:  the random source. Use it for BOTH the shuffle and the
                    noise so a seed reproduces a run.

    Steps:
      1. `model.train()`.
      2. Shuffle: `torch.randperm(n, generator=generator)`.
      3. For each batch of indices, in order:
           a. Take that batch's embeddings. If `alpha > 0`, add
              `torch.randn(batch.shape, generator=generator) * alpha`.
              Build a NEW tensor: never modify `embeds` in place.
           b. `logits = model.win_logit(winners[idx], losers[idx], noisy)`.
           c. Every pair is "the winner won", so the target is all ones:
              `F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))`.
           d. `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
      4. Return the mean loss over all n pairs: weight each batch's mean
         loss by the number of pairs in it, then divide by n.
    """
    raise NotImplementedError("Implement train_epoch")


# --------------------------------------------------------------------------- #
# The multi-epoch loop (PROVIDED)
# --------------------------------------------------------------------------- #
def fit(
    embeds: torch.Tensor,
    winners: Sequence[int],
    losers: Sequence[int],
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    alpha: float = 0.02,
    batch_size: int = 64,
    seed: int = 0,
    init: dict[str, torch.Tensor] | None = None,
    dim: int = DIM,
    num_models: int = NUM_MODELS,
) -> tuple[MFModel, list[float]]:
    """PROVIDED: build a model and train it for `epochs` epochs.

    `init` is an optional RouteLLM state dict to start from (fine-tuning).
    Returns the trained model and the loss after each epoch.
    """
    torch.manual_seed(seed)
    model = MFModel(dim=dim, num_models=num_models, text_dim=embeds.shape[1])
    if init is not None:
        model.load_state_dict(init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    win = torch.as_tensor(winners, dtype=torch.long)
    lose = torch.as_tensor(losers, dtype=torch.long)
    losses = [
        train_epoch(model, optimizer, embeds, win, lose, batch_size=batch_size, alpha=alpha, generator=generator)
        for _ in range(epochs)
    ]
    return model, losses


# ========================================================================== #
# TODO(student) #5: inference
# ========================================================================== #
def strong_win_rate(
    model: MFModel,
    embeds: torch.Tensor,
    strong_id: int = STRONG_ID,
    weak_id: int = WEAK_ID,
) -> list[float]:
    """P(the strong model's answer is preferred) for each prompt.

    This is the number RouteLLM's router returns and the number
    `preference.threshold` in config.yaml is compared against.

    Args:
        embeds: FloatTensor (n, text_dim).

    Returns:
        A list of n Python floats in (0, 1):
        sigmoid(score(strong) − score(weak)).

    Run it with `model.eval()` and inside `torch.no_grad()`: this is
    inference, so no noise and no gradients.
    """
    raise NotImplementedError("Implement strong_win_rate")


# ========================================================================== #
# TODO(student) #6: is the router any good?
# ========================================================================== #
def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: does the router RANK prompts correctly?

    AUC is the probability that a randomly chosen positive (a prompt where
    the strong model won) gets a higher score than a randomly chosen
    negative. 0.5 is a coin flip, 1.0 is a perfect ranking. It does not
    depend on any threshold, which is what makes it the right first number
    to look at for a router: the threshold is a budget decision you make
    later.

    Compute it by counting. Over every (positive, negative) pair:
        1    if the positive's score is higher
        0.5  if the two scores are equal
        0    otherwise
    and return the mean.

    Raises:
        ValueError: if there are no positives or no negatives.
    """
    raise NotImplementedError("Implement auc")


# --------------------------------------------------------------------------- #
# Threshold calibration (PROVIDED)
# --------------------------------------------------------------------------- #
def calibrate_threshold(win_rates: Sequence[float], strong_fraction: float) -> float:
    """PROVIDED: the threshold that sends about `strong_fraction` of these prompts to strong.

    The same function as `routing.calibrate_threshold` from Part 1.6.
    A prompt goes to the strong tier when its win rate is >= the threshold.
    """
    if not win_rates:
        raise ValueError("Need at least one win rate to calibrate")
    ranked = sorted(win_rates, reverse=True)
    n_strong = round(strong_fraction * len(ranked))
    if n_strong <= 0:
        return min(1.0, ranked[0] + 1e-6)
    return ranked[min(n_strong, len(ranked)) - 1]


# ========================================================================== #
# TODO(student) #7: what a strong-tier budget buys
# ========================================================================== #
@dataclass(frozen=True)
class BudgetRow:
    """What one strong-tier budget buys (PROVIDED)."""

    strong_fraction: float   # the budget asked for
    threshold: float         # the threshold that budget calibrates to
    sent: float              # share of prompts actually sent to strong (score >= threshold)
    caught: float            # share of the strong-needed prompts that were sent to strong
    quality: float           # share of ALL prompts that got an answer as good as strong's


def budget_table(
    scores: Sequence[float],
    strong_needed: Sequence[bool],
    fractions: Sequence[float],
) -> list[BudgetRow]:
    """Turn router scores into the table a cost engineer actually reads.

    Args:
        scores:        the router's P(strong wins) for each prompt.
        strong_needed: for each prompt, did the strong model win the battle?
        fractions:     strong-tier budgets to evaluate, e.g. [0.1, 0.2, 0.3].

    For each fraction, in order:
      1. `threshold = calibrate_threshold(scores, fraction)`.
      2. A prompt is sent to strong when `score >= threshold`.
      3. `sent`    = prompts sent to strong / all prompts.
      4. `caught`  = strong-needed prompts sent to strong / strong-needed
                     prompts. (If no prompt is strong-needed, 1.0.)
      5. `quality` = 1 − (strong-needed prompts left on weak) / all prompts.
                     A prompt gets a good answer if the weak tier was enough
                     or if it was sent to strong.

    A router that knows nothing catches `fraction` of the strong-needed
    prompts at a budget of `fraction`. Everything above that is what the
    router is worth.
    """
    raise NotImplementedError("Implement budget_table")


# --------------------------------------------------------------------------- #
# Checkpoints (PROVIDED)
# --------------------------------------------------------------------------- #
def export_checkpoint(model: MFModel, directory: str | Path) -> Path:
    """Save `model` as a RouteLLM checkpoint directory.

    RouteLLM loads a matrix-factorization router with

        MatrixFactorizationRouter(checkpoint_path=<directory or Hugging Face repo>)

    and expects a `model.safetensors` holding P.weight, text_proj.0.weight
    and classifier.0.weight. That is this model's state dict, unchanged.
    `config.json` records the shapes RouteLLM passes when it builds the model.
    """
    import json

    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    config = {
        "dim": model.P.embedding_dim,
        "num_models": model.P.num_embeddings,
        "text_dim": model.text_proj[0].in_features,
        "num_classes": 1,
        "use_proj": True,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return directory


def load_checkpoint(directory: str | Path) -> MFModel:
    """Load a checkpoint directory written by `export_checkpoint` (or downloaded from RouteLLM)."""
    from safetensors.torch import load_file

    state = load_file(str(Path(directory) / "model.safetensors"))
    num_models, dim = state["P.weight"].shape
    model = MFModel(dim=dim, num_models=num_models, text_dim=state["text_proj.0.weight"].shape[1])
    model.load_state_dict(state)
    return model.eval()
