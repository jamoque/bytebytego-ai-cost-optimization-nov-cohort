#!/usr/bin/env python3
"""
Train a Northstar router on the battles (PROVIDED).

Run this after `pytest` is green. It drives the seven functions you wrote
in `mf_router.py`:

    python train.py                          # train from random weights, cross-validate, export
    python train.py --init published         # start from RouteLLM's published weights instead
    python train.py --lr 3e-4 --alpha 0.1    # RouteLLM's own hyperparameters
    python train.py --curve                  # how much data do you need? (25/50/75/100% of the battles)
    python train.py --ties drop              # what RouteLLM's training script does with ties

What it does
------------
1. Loads `fixtures/battles.json` and embeds each prompt ONCE with the
   mesh's `mesh-embed` alias (text-embedding-3-small, the embedding
   RouteLLM's router uses at inference time). The vectors are cached in
   `.cache/`, so only the first run needs the mesh. Cost: well under a cent.
2. Scores RouteLLM's published router on the same battles, as the baseline.
3. Cross-validates YOUR model: train on four fifths of the battles, score
   the held-out fifth, five times. Every number in the tables comes from
   prompts the model that scored them never trained on.
4. Trains one final model on all the battles and exports it as a RouteLLM
   checkpoint directory that `../routing-lab` can load.

With 307 battles, a single 80/20 split would leave about 14 strong wins in
the held-out set, and the AUC would swing by ±0.1 depending on which 14.
Cross-validation uses every battle as held-out data exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ROUTING_LAB = HERE.parent / "routing-lab"
PUBLISHED_REPO = "routellm/mf_gpt4_augmented"

try:
    import numpy as np
    import torch
    import yaml
except ImportError:
    print("Install dependencies first: pip install -r requirements.txt", flush=True)
    sys.exit(1)

impl = importlib.import_module(os.environ.get("MF_ROUTER_IMPL", "mf_router"))


def log(message: str = "") -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_lab_config() -> dict[str, Any]:
    """The hands-on lab's config: mesh address, key, embedding alias, tier aliases."""
    config = yaml.safe_load((ROUTING_LAB / "config.yaml").read_text())
    if os.environ.get("MESH_BASE_URL"):
        config["mesh"]["base_url"] = os.environ["MESH_BASE_URL"]
    return config


def embed_prompts(prompts: list[str], config: dict[str, Any]) -> torch.Tensor:
    """One embedding per prompt through the mesh, cached on disk."""
    alias = config["preference"].get("embedding_alias", "mesh-embed")
    key = hashlib.sha1(json.dumps([alias, prompts]).encode()).hexdigest()[:12]
    cache = HERE / ".cache" / f"embeddings-{key}.npy"
    if cache.exists():
        log(f"Embeddings: {len(prompts)} prompts, loaded from {cache.relative_to(HERE)}")
        return torch.from_numpy(np.load(cache))

    from openai import OpenAI

    mesh = config["mesh"]
    client = OpenAI(base_url=mesh["base_url"], api_key=mesh["api_key"])
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(prompts), 100):
            response = client.embeddings.create(model=alias, input=prompts[start : start + 100])
            vectors.extend(item.embedding for item in response.data)
    except Exception as exc:
        log(
            f"Could not embed the prompts. Is the Lab 3 mesh running at {mesh['base_url']}?\n"
            f"  Embedding alias: {alias}\n  Original error: {exc}"
        )
        sys.exit(1)
    array = np.asarray(vectors, dtype=np.float32)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, array)
    log(f"Embeddings: {len(prompts)} prompts × {array.shape[1]} numbers via {alias}, cached in {cache.relative_to(HERE)}")
    return torch.from_numpy(array)


def published_state() -> dict[str, torch.Tensor] | None:
    """RouteLLM's published mf weights (already on disk if you ran Part 1.6)."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        try:  # already on disk? then stay offline
            path = hf_hub_download(PUBLISHED_REPO, "model.safetensors", local_files_only=True)
        except Exception:
            path = hf_hub_download(PUBLISHED_REPO, "model.safetensors")
        return load_file(path)
    except Exception as exc:  # offline and never downloaded
        log(f"(Could not load {PUBLISHED_REPO}: {exc})")
        return None


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #
def train_on(
    rows: list[int],
    battles: list[dict[str, Any]],
    embeds: torch.Tensor,
    args: argparse.Namespace,
    ids: dict[str, int],
    tie_winner: str | None,
    init: dict[str, torch.Tensor] | None,
) -> tuple[Any, list[float], int]:
    """Fit a model on the battles at `rows`. Returns (model, losses, number of training pairs)."""
    subset = [battles[i] for i in rows]
    keep, winners, losers = impl.battle_pairs(subset, ids, tie_winner)
    if not keep:
        log("No training pairs left after applying the tie rule.")
        sys.exit(1)
    index = torch.tensor([rows[k] for k in keep])
    model, losses = impl.fit(
        embeds[index],
        winners,
        losers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        init=init,
    )
    return model, losses, len(keep)


def cross_validate(
    battles: list[dict[str, Any]],
    embeds: torch.Tensor,
    labels: list[bool],
    args: argparse.Namespace,
    ids: dict[str, int],
    tie_winner: str | None,
    init: dict[str, torch.Tensor] | None,
    fraction: float,
) -> tuple[list[float], list[float], int]:
    """Out-of-fold scores for every battle, per-fold AUCs, and the training size used."""
    order = list(range(len(battles)))
    random.Random(args.seed).shuffle(order)
    folds = [order[k :: args.folds] for k in range(args.folds)]
    scores = [0.0] * len(battles)
    fold_aucs: list[float] = []
    trained_on = 0
    for k, held_out in enumerate(folds):
        train_rows = [i for j, fold in enumerate(folds) if j != k for i in fold]
        train_rows = train_rows[: max(1, int(len(train_rows) * fraction))]
        model, _, trained_on = train_on(train_rows, battles, embeds, args, ids, tie_winner, init)
        fold_scores = impl.strong_win_rate(model, embeds[torch.tensor(held_out)])
        for i, s in zip(held_out, fold_scores):
            scores[i] = s
        fold_aucs.append(impl.auc(fold_scores, [labels[i] for i in held_out]))
    return scores, fold_aucs, trained_on


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--battles", default=str(HERE / "fixtures" / "battles.json"))
    parser.add_argument("--init", choices=["random", "published"], default="random", help="starting weights")
    parser.add_argument("--ties", choices=["weak", "drop"], default="weak", help="a tie is a win for the weak tier, or is dropped")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (RouteLLM's script: 3e-4)")
    parser.add_argument("--alpha", type=float, default=0.02, help="embedding noise (RouteLLM's script: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curve", action="store_true", help="cross-validate at 25/50/75/100%% of the training data, then stop")
    parser.add_argument("--out", default=str(HERE / "checkpoints" / "northstar-mf"))
    args = parser.parse_args()

    config = load_lab_config()
    weak_alias = config["tiers"]["weak"]["alias"]
    strong_alias = config["tiers"]["strong"]["alias"]
    ids = {weak_alias: impl.WEAK_ID, strong_alias: impl.STRONG_ID}
    tie_winner = weak_alias if args.ties == "weak" else None

    battles: list[dict[str, Any]] = json.loads(Path(args.battles).read_text())
    labels = [b["winner"] == "model_b" for b in battles]   # did the strong tier win?
    n, wins = len(battles), sum(labels)
    log(f"Battles: {n} prompts, {weak_alias} vs {strong_alias}")
    log(f"  strong wins {wins} ({wins / n:.0%}), ties {n - wins} ({'counted as wins for the weak tier' if tie_winner else 'DROPPED'})")
    embeds = embed_prompts([b["prompt"] for b in battles], config)

    published = published_state()
    init = None
    if args.init == "published":
        if published is None:
            log("--init published needs RouteLLM's checkpoint. Run Part 1.6 once (python calibrate.py) to download it.")
            sys.exit(1)
        init = published
    log(
        f"Training: {args.init} start, {args.epochs} epochs, lr {args.lr:g}, noise alpha {args.alpha:g}, "
        f"batch {args.batch_size}, {args.folds}-fold cross-validation"
    )

    if args.curve:
        log("\nHow much data does it take?   (held-out AUC; 0.5 = coin flip)")
        log(f"  {'battles trained on':>18}  {'AUC':>6}  per fold")
        for fraction in (0.25, 0.5, 0.75, 1.0):
            scores, fold_aucs, trained_on = cross_validate(battles, embeds, labels, args, ids, tie_winner, init, fraction)
            log(f"  {trained_on:>18}  {impl.auc(scores, labels):>6.3f}  {' '.join(f'{a:.2f}' for a in fold_aucs)}")
        return

    scores, fold_aucs, trained_on = cross_validate(battles, embeds, labels, args, ids, tie_winner, init, 1.0)
    published_scores = None
    if published is not None:
        num_models, dim = published["P.weight"].shape
        reference = impl.MFModel(dim=dim, num_models=num_models, text_dim=published["text_proj.0.weight"].shape[1])
        reference.load_state_dict(published)
        published_scores = impl.strong_win_rate(reference, embeds)

    log("\nCan the question alone predict a strong win?   (AUC: 0.5 = coin flip, 1.0 = perfect)")
    log(f"  {'a router that knows nothing':<34} 0.500")
    if published_scores is not None:
        log(f"  {'RouteLLM published mf':<34} {impl.auc(published_scores, labels):.3f}")
    log(
        f"  {'yours, held-out':<34} {impl.auc(scores, labels):.3f}   "
        f"(per fold: {' '.join(f'{a:.2f}' for a in fold_aucs)}; each trained on {trained_on} battles)"
    )

    def spread(values: list[float]) -> str:
        ranked = sorted(values)
        return f"min {ranked[0]:.2f}, median {ranked[len(ranked) // 2]:.2f}, max {ranked[-1]:.2f}"

    log("\nP(strong wins) across the battles")
    if published_scores is not None:
        log(f"  {'RouteLLM published mf':<34} {spread(published_scores)}")
    log(f"  {'yours, held-out':<34} {spread(scores)}")

    weak_cost = sum(b["cost_a"] for b in battles) / n
    strong_cost = sum(b["cost_b"] for b in battles) / n
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    mine = impl.budget_table(scores, labels, fractions)
    theirs = impl.budget_table(published_scores, labels, fractions) if published_scores is not None else None
    log("\nWhat a strong-tier budget buys   (held-out scores)")
    log(f"  All-weak: {1 - wins / n:.0%} of answers match the strong tier's, at {weak_cost / strong_cost:.0%} of the all-strong cost.")
    log(f"  {'budget':>6}  {'strong wins caught':^28}  {'answers as good as strong':^28}  {'cost vs':>10}")
    log(f"  {'':>6}  {'random':>8} {'published':>10} {'yours':>8}  {'random':>8} {'published':>10} {'yours':>8}  {'all-strong':>10}")
    for i, row in enumerate(mine):
        f = row.strong_fraction
        random_quality = 1 - (wins - f * wins) / n
        pub_caught = f"{theirs[i].caught:>10.0%}" if theirs else f"{'-':>10}"
        pub_quality = f"{theirs[i].quality:>10.0%}" if theirs else f"{'-':>10}"
        cost = (row.sent * strong_cost + (1 - row.sent) * weak_cost) / strong_cost
        log(
            f"  {f:>6.0%}  {f:>8.0%} {pub_caught} {row.caught:>8.0%}  "
            f"{random_quality:>8.0%} {pub_quality} {row.quality:>8.0%}  {cost:>10.0%}"
        )

    model, losses, pairs = train_on(list(range(n)), battles, embeds, args, ids, tie_winner, init)
    out = impl.export_checkpoint(model, args.out)
    log(f"\nFinal model: trained on all {pairs} pairs, loss {losses[0]:.3f} → {losses[-1]:.3f}")
    log(f"Saved a RouteLLM checkpoint to {out}")
    try:
        shown = Path(os.path.relpath(out, ROUTING_LAB))
    except ValueError:
        shown = Path(out).resolve()
    log("\nTo route with it, in ../routing-lab/config.yaml set:")
    log("  preference:")
    log(f"    checkpoint: {shown}")
    log("then recalibrate the threshold on traffic the model has not trained on:")
    log("  (cd ../routing-lab && python calibrate.py --strong 0.30)")


if __name__ == "__main__":
    main()
