#!/usr/bin/env python3
"""
Test suite for the matrix-factorization router exercise.

Run with:   pytest -v

These tests are your definition of "done". They import your implementation
from `mf_router.py` by default. (Instructors can point the tests at a
reference implementation with `MF_ROUTER_IMPL=reference_solution pytest`.)

Everything here runs offline: the models are tiny, the "embeddings" are
random vectors with a planted pattern, and the one test that involves
RouteLLM replaces its embedding client with a stub.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

impl = importlib.import_module(os.environ.get("MF_ROUTER_IMPL", "mf_router"))

MFModel = impl.MFModel
battle_pairs = impl.battle_pairs
train_epoch = impl.train_epoch
fit = impl.fit
strong_win_rate = impl.strong_win_rate
auc = impl.auc
calibrate_threshold = impl.calibrate_threshold
budget_table = impl.budget_table
export_checkpoint = impl.export_checkpoint
load_checkpoint = impl.load_checkpoint
STRONG_ID = impl.STRONG_ID
WEAK_ID = impl.WEAK_ID

HERE = Path(__file__).parent

# A small model for fast tests: 6 models, 8-dim vectors, 16-dim "embeddings".
SMALL = dict(dim=8, num_models=6, text_dim=16)
S, W = 1, 4  # strong and weak rows in the small model table


def small_model(seed: int = 0) -> MFModel:
    torch.manual_seed(seed)
    return MFModel(**SMALL)


def embeds(n: int, seed: int = 0, text_dim: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, text_dim, generator=g), dim=1)


def planted(n: int, seed: int = 0) -> tuple[torch.Tensor, list[bool]]:
    """Embeddings where the strong model wins exactly when coordinate 0 is positive."""
    x = embeds(n, seed)
    return x, [bool(v > 0) for v in x[:, 0]]


def pairs_for(labels: list[bool]) -> tuple[list[int], list[int]]:
    winners = [S if y else W for y in labels]
    losers = [W if y else S for y in labels]
    return winners, losers


WEAK_NAME, STRONG_NAME = "mesh-gpt-mini", "mesh-claude-sonnet"
IDS = {WEAK_NAME: WEAK_ID, STRONG_NAME: STRONG_ID}


def battle(winner: str, a: str = WEAK_NAME, b: str = STRONG_NAME) -> dict:
    return {"prompt": "q", "model_a": a, "model_b": b, "winner": winner}


# --------------------------------------------------------------------------- #
# 1. MFModel.score
# --------------------------------------------------------------------------- #
class TestScore:
    def test_returns_one_logit_per_row(self):
        model = small_model()
        out = model.score(torch.tensor([0, 1, 2]), embeds(3))
        assert out.shape == (3,)

    def test_single_row_keeps_batch_dimension(self):
        model = small_model()
        out = model.score(torch.tensor([2]), embeds(1))
        assert out.shape == (1,)

    def test_matches_the_formula(self):
        model = small_model()
        ids, x = torch.tensor([0, 3, 5, 3]), embeds(4)
        p = model.P.weight[ids]
        p = p / p.norm(dim=1, keepdim=True)
        projected = x @ model.text_proj[0].weight.T
        expected = (p * projected) @ model.classifier[0].weight.squeeze(0)
        assert torch.allclose(model.score(ids, x), expected, atol=1e-6)

    def test_model_vector_length_does_not_matter(self):
        """Model vectors are L2-normalized: only their direction is used."""
        model = small_model()
        ids, x = torch.tensor([2, 2]), embeds(2)
        before = model.score(ids, x).detach().clone()
        with torch.no_grad():
            model.P.weight[2] *= 10.0
        assert torch.allclose(model.score(ids, x), before, atol=1e-5)

    def test_no_bias_anywhere(self):
        """A zero prompt embedding scores exactly zero for every model."""
        model = small_model()
        out = model.score(torch.tensor([0, 1]), torch.zeros(2, 16))
        assert torch.equal(out, torch.zeros(2))

    def test_different_models_score_the_same_prompt_differently(self):
        model = small_model()
        x = embeds(1).repeat(2, 1)
        out = model.score(torch.tensor([0, 1]), x)
        assert out[0].item() != pytest.approx(out[1].item())

    def test_gradients_reach_all_three_parameter_tensors(self):
        model = small_model()
        model.score(torch.tensor([0, 1]), embeds(2)).sum().backward()
        assert model.P.weight.grad is not None and model.P.weight.grad.abs().sum() > 0
        assert model.text_proj[0].weight.grad.abs().sum() > 0
        assert model.classifier[0].weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 2. MFModel.win_logit
# --------------------------------------------------------------------------- #
class TestWinLogit:
    def test_is_the_difference_of_two_scores(self):
        model = small_model()
        w, l, x = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), embeds(3)
        assert torch.allclose(model.win_logit(w, l, x), model.score(w, x) - model.score(l, x), atol=1e-6)

    def test_swapping_the_models_flips_the_sign(self):
        model = small_model()
        w, l, x = torch.tensor([0, 1]), torch.tensor([3, 4]), embeds(2)
        assert torch.allclose(model.win_logit(w, l, x), -model.win_logit(l, w, x), atol=1e-6)

    def test_a_model_against_itself_is_a_coin_flip(self):
        model = small_model()
        ids, x = torch.tensor([2, 5]), embeds(2)
        assert torch.allclose(model.win_logit(ids, ids, x), torch.zeros(2), atol=1e-6)

    def test_equals_routellm_training_form(self):
        """classifier((P[winner] − P[loser]) ⊙ W·e), which is how RouteLLM's trainer writes it."""
        model = small_model()
        w, l, x = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), embeds(3)
        pw = F.normalize(model.P(w), dim=1)
        pl = F.normalize(model.P(l), dim=1)
        expected = model.classifier((pw - pl) * model.text_proj(x)).squeeze(-1)
        assert torch.allclose(model.win_logit(w, l, x), expected, atol=1e-6)

    def test_shape(self):
        model = small_model()
        assert model.win_logit(torch.tensor([0]), torch.tensor([1]), embeds(1)).shape == (1,)


# --------------------------------------------------------------------------- #
# 3. battle_pairs
# --------------------------------------------------------------------------- #
class TestBattlePairs:
    def test_model_b_wins(self):
        keep, winners, losers = battle_pairs([battle("model_b")], IDS, WEAK_NAME)
        assert (keep, winners, losers) == ([0], [STRONG_ID], [WEAK_ID])

    def test_model_a_wins(self):
        keep, winners, losers = battle_pairs([battle("model_a")], IDS, WEAK_NAME)
        assert (keep, winners, losers) == ([0], [WEAK_ID], [STRONG_ID])

    def test_a_tie_goes_to_the_tie_winner(self):
        keep, winners, losers = battle_pairs([battle("tie")], IDS, WEAK_NAME)
        assert (keep, winners, losers) == ([0], [WEAK_ID], [STRONG_ID])

    def test_tie_winner_works_from_either_side(self):
        flipped = battle("tie", a=STRONG_NAME, b=WEAK_NAME)
        assert battle_pairs([flipped], IDS, WEAK_NAME) == ([0], [WEAK_ID], [STRONG_ID])
        assert battle_pairs([flipped], IDS, STRONG_NAME) == ([0], [STRONG_ID], [WEAK_ID])

    def test_model_order_is_read_from_the_battle(self):
        flipped = battle("model_a", a=STRONG_NAME, b=WEAK_NAME)
        assert battle_pairs([flipped], IDS, WEAK_NAME) == ([0], [STRONG_ID], [WEAK_ID])

    def test_ties_are_dropped_without_a_tie_winner(self):
        battles = [battle("tie"), battle("model_b"), battle("tie"), battle("model_a")]
        keep, winners, losers = battle_pairs(battles, IDS, None)
        assert keep == [1, 3]
        assert winners == [STRONG_ID, WEAK_ID]
        assert losers == [WEAK_ID, STRONG_ID]

    def test_tie_is_dropped_when_tie_winner_did_not_play(self):
        assert battle_pairs([battle("tie")], IDS, "some-other-model") == ([], [], [])

    def test_unknown_verdicts_are_dropped(self):
        battles = [battle("tie (bothbad)"), battle(""), battle("model_b")]
        assert battle_pairs(battles, IDS, WEAK_NAME) == ([2], [STRONG_ID], [WEAK_ID])

    def test_a_model_against_itself_is_dropped(self):
        battles = [battle("model_a", a=WEAK_NAME, b=WEAK_NAME), battle("tie", a=WEAK_NAME, b=WEAK_NAME)]
        assert battle_pairs(battles, IDS, WEAK_NAME) == ([], [], [])

    def test_keep_indexes_the_original_list(self):
        battles = [battle("model_b"), battle("nonsense"), battle("tie"), battle("model_a")]
        keep, winners, losers = battle_pairs(battles, IDS, WEAK_NAME)
        assert keep == [0, 2, 3]
        assert len(winners) == len(losers) == 3

    def test_empty(self):
        assert battle_pairs([], IDS, WEAK_NAME) == ([], [], [])

    def test_on_the_real_battles_every_one_becomes_a_pair(self):
        battles = json.loads((HERE / "fixtures" / "battles.json").read_text())
        keep, winners, losers = battle_pairs(battles, IDS, WEAK_NAME)
        assert keep == list(range(len(battles)))
        assert set(winners) == {WEAK_ID, STRONG_ID}
        assert all({w, l} == {WEAK_ID, STRONG_ID} for w, l in zip(winners, losers))

    def test_on_the_real_battles_dropping_ties_leaves_only_one_kind_of_winner(self):
        """RouteLLM's script drops ties. On this data every pair that is left says "strong won"."""
        battles = json.loads((HERE / "fixtures" / "battles.json").read_text())
        keep, winners, _ = battle_pairs(battles, IDS, None)
        assert 0 < len(keep) < len(battles)
        assert set(winners) == {STRONG_ID}


# --------------------------------------------------------------------------- #
# 4. train_epoch
# --------------------------------------------------------------------------- #
class TestTrainEpoch:
    def setup(self, n: int = 12, seed: int = 0):
        model = small_model(seed)
        x, labels = planted(n, seed)
        winners, losers = pairs_for(labels)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        return model, optimizer, x, torch.tensor(winners), torch.tensor(losers)

    def test_returns_the_mean_loss_of_the_epoch(self):
        """One full batch, no noise: the loss is BCE against all-ones at the starting weights."""
        model, optimizer, x, w, l = self.setup()
        with torch.no_grad():
            logits = model.win_logit(w, l, x)
            expected = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).item()
        got = train_epoch(model, optimizer, x, w, l, batch_size=100, alpha=0.0, generator=torch.Generator().manual_seed(0))
        assert isinstance(got, float)
        assert got == pytest.approx(expected, rel=1e-5)

    def test_untrained_loss_is_near_log_2(self):
        """A model that knows nothing gives every pair about a 50% win probability."""
        model, optimizer, x, w, l = self.setup(n=64)
        got = train_epoch(model, optimizer, x, w, l, batch_size=64, alpha=0.0, generator=torch.Generator().manual_seed(0))
        assert got == pytest.approx(math.log(2), abs=0.1)

    def test_updates_the_weights(self):
        model, optimizer, x, w, l = self.setup()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.0, generator=torch.Generator().manual_seed(0))
        after = model.state_dict()
        assert not torch.equal(before["text_proj.0.weight"], after["text_proj.0.weight"])
        assert not torch.equal(before["classifier.0.weight"], after["classifier.0.weight"])
        assert not torch.equal(before["P.weight"], after["P.weight"])

    def test_only_the_models_in_the_battles_move(self):
        """Rows of P for models that never played keep their starting values."""
        model, optimizer, x, w, l = self.setup()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
        before = model.P.weight.detach().clone()
        train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.0, generator=torch.Generator().manual_seed(0))
        after = model.P.weight.detach()
        for row in range(SMALL["num_models"]):
            moved = not torch.equal(before[row], after[row])
            assert moved == (row in (S, W)), f"row {row}"

    def test_one_step_per_batch_and_every_pair_once(self):
        model, optimizer, x, w, l = self.setup(n=10)
        steps: list[int] = []
        seen: list[int] = []
        original_step = optimizer.step
        original_win_logit = model.win_logit

        def counting_step(*args, **kwargs):
            steps.append(1)
            return original_step(*args, **kwargs)

        def recording_win_logit(winner_ids, loser_ids, prompt_embeds):
            seen.append(len(winner_ids))
            return original_win_logit(winner_ids, loser_ids, prompt_embeds)

        optimizer.step = counting_step
        model.win_logit = recording_win_logit
        train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.0, generator=torch.Generator().manual_seed(0))
        assert len(steps) == 3            # 4 + 4 + 2
        assert sorted(seen) == [2, 4, 4]
        assert sum(seen) == 10

    def test_shuffles_with_the_generator(self):
        """Batches are drawn in a shuffled order, and the same seed gives the same order."""
        orders = []
        for seed in (0, 0, 1):
            model, optimizer, x, w, l = self.setup(n=16)
            seen: list[torch.Tensor] = []
            original = model.win_logit

            def recording(winner_ids, loser_ids, prompt_embeds, _original=original, _seen=seen):
                _seen.append(prompt_embeds.detach().clone())
                return _original(winner_ids, loser_ids, prompt_embeds)

            model.win_logit = recording
            train_epoch(model, optimizer, x, w, l, batch_size=16, alpha=0.0, generator=torch.Generator().manual_seed(seed))
            orders.append(seen[0])
        assert torch.equal(orders[0], orders[1])
        assert not torch.equal(orders[0], orders[2])
        assert not torch.equal(orders[0], x), "pairs were not shuffled"
        # Same rows, different order.
        assert torch.allclose(orders[0].sum(dim=0), x.sum(dim=0), atol=1e-5)

    def test_noise_is_added_when_alpha_is_positive(self):
        model, optimizer, x, w, l = self.setup(n=8)
        seen: list[torch.Tensor] = []
        original = model.win_logit

        def recording(winner_ids, loser_ids, prompt_embeds):
            seen.append(prompt_embeds.detach().clone())
            return original(winner_ids, loser_ids, prompt_embeds)

        model.win_logit = recording
        train_epoch(model, optimizer, x, w, l, batch_size=8, alpha=0.5, generator=torch.Generator().manual_seed(0))
        noisy = seen[0]
        # No row the model saw is an exact row of the clean embeddings...
        assert all(not any(torch.equal(row, clean) for clean in x) for row in noisy)
        # ...and the noise has roughly the requested scale.
        spread = (noisy.sum(dim=0) - x.sum(dim=0)).std().item() / math.sqrt(8)
        assert 0.2 < spread < 1.0

    def test_no_noise_when_alpha_is_zero(self):
        model, optimizer, x, w, l = self.setup(n=8)
        seen: list[torch.Tensor] = []
        original = model.win_logit

        def recording(winner_ids, loser_ids, prompt_embeds):
            seen.append(prompt_embeds.detach().clone())
            return original(winner_ids, loser_ids, prompt_embeds)

        model.win_logit = recording
        train_epoch(model, optimizer, x, w, l, batch_size=8, alpha=0.0, generator=torch.Generator().manual_seed(0))
        assert all(any(torch.equal(row, clean) for clean in x) for row in seen[0])

    def test_does_not_modify_the_embeddings(self):
        model, optimizer, x, w, l = self.setup()
        before = x.clone()
        train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.5, generator=torch.Generator().manual_seed(0))
        assert torch.equal(x, before)

    def test_same_seed_same_result(self):
        losses = []
        for _ in range(2):
            model, optimizer, x, w, l = self.setup()
            losses.append(
                train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.3, generator=torch.Generator().manual_seed(7))
            )
        assert losses[0] == losses[1]

    def test_loss_goes_down_over_epochs(self):
        model, optimizer, x, w, l = self.setup(n=64)
        g = torch.Generator().manual_seed(0)
        losses = [train_epoch(model, optimizer, x, w, l, batch_size=16, alpha=0.0, generator=g) for _ in range(40)]
        assert losses[-1] < losses[0] * 0.5

    def test_leaves_the_model_in_training_mode(self):
        model, optimizer, x, w, l = self.setup()
        model.eval()
        train_epoch(model, optimizer, x, w, l, batch_size=4, alpha=0.0, generator=torch.Generator().manual_seed(0))
        assert model.training


# --------------------------------------------------------------------------- #
# 5. strong_win_rate
# --------------------------------------------------------------------------- #
class TestStrongWinRate:
    def test_one_probability_per_prompt(self):
        model = small_model()
        rates = strong_win_rate(model, embeds(5), strong_id=S, weak_id=W)
        assert isinstance(rates, list) and len(rates) == 5
        assert all(isinstance(r, float) and 0.0 < r < 1.0 for r in rates)

    def test_is_the_sigmoid_of_the_score_difference(self):
        model = small_model()
        x = embeds(4)
        with torch.no_grad():
            strong = model.score(torch.full((4,), S), x)
            weak = model.score(torch.full((4,), W), x)
            expected = torch.sigmoid(strong - weak).tolist()
        assert strong_win_rate(model, x, strong_id=S, weak_id=W) == pytest.approx(expected, abs=1e-6)

    def test_swapping_the_tiers_gives_the_complement(self):
        model = small_model()
        x = embeds(4)
        forward = strong_win_rate(model, x, strong_id=S, weak_id=W)
        backward = strong_win_rate(model, x, strong_id=W, weak_id=S)
        assert [a + b for a, b in zip(forward, backward)] == pytest.approx([1.0] * 4, abs=1e-6)

    def test_defaults_to_the_routellm_rows(self):
        torch.manual_seed(0)
        model = MFModel(dim=8, num_models=64, text_dim=16)
        x = embeds(3)
        assert strong_win_rate(model, x) == pytest.approx(
            strong_win_rate(model, x, strong_id=STRONG_ID, weak_id=WEAK_ID), abs=1e-9
        )

    def test_does_not_track_gradients_or_change_weights(self):
        model = small_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        strong_win_rate(model, embeds(3), strong_id=S, weak_id=W)
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    def test_empty(self):
        assert strong_win_rate(small_model(), torch.zeros(0, 16), strong_id=S, weak_id=W) == []


# --------------------------------------------------------------------------- #
# 6. auc
# --------------------------------------------------------------------------- #
class TestAUC:
    def test_perfect_ranking(self):
        assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0

    def test_inverted_ranking(self):
        assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0

    def test_constant_scores_are_a_coin_flip(self):
        assert auc([0.5] * 6, [True, False, True, False, False, True]) == 0.5

    def test_counts_pairs(self):
        # positives 0.8, 0.4; negatives 0.6, 0.2 → 3 of 4 pairs ranked correctly
        assert auc([0.8, 0.4, 0.6, 0.2], [True, True, False, False]) == pytest.approx(0.75)

    def test_a_tied_pair_counts_half(self):
        # positive 0.5 vs negatives 0.5 (tie) and 0.1 (win) → (0.5 + 1) / 2
        assert auc([0.5, 0.5, 0.1], [True, False, False]) == pytest.approx(0.75)

    def test_does_not_depend_on_the_scale_of_the_scores(self):
        scores = [0.31, 0.12, 0.27, 0.05, 0.22]
        labels = [True, False, True, False, False]
        assert auc(scores, labels) == auc([s * 100 - 3 for s in scores], labels)

    def test_order_of_inputs_does_not_matter(self):
        scores = [0.8, 0.4, 0.6, 0.2]
        labels = [True, True, False, False]
        assert auc(scores[::-1], labels[::-1]) == pytest.approx(auc(scores, labels))

    def test_needs_both_classes(self):
        with pytest.raises(ValueError):
            auc([0.1, 0.2], [True, True])
        with pytest.raises(ValueError):
            auc([0.1, 0.2], [False, False])


# --------------------------------------------------------------------------- #
# 7. budget_table
# --------------------------------------------------------------------------- #
class TestBudgetTable:
    SCORES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    #          needed       needed            needed
    NEEDED = [True, False, True, False, False, True, False, False, False, False]

    def test_one_row_per_fraction_in_order(self):
        rows = budget_table(self.SCORES, self.NEEDED, [0.5, 0.1, 0.3])
        assert [r.strong_fraction for r in rows] == [0.5, 0.1, 0.3]

    def test_known_values(self):
        row = budget_table(self.SCORES, self.NEEDED, [0.3])[0]
        assert row.threshold == pytest.approx(0.7)
        assert row.sent == pytest.approx(0.3)              # 0.9, 0.8, 0.7
        assert row.caught == pytest.approx(2 / 3)          # two of the three needed prompts
        assert row.quality == pytest.approx(0.9)           # one needed prompt left on weak, of ten

    def test_zero_budget_sends_nothing(self):
        row = budget_table(self.SCORES, self.NEEDED, [0.0])[0]
        assert row.sent == 0.0
        assert row.caught == 0.0
        assert row.quality == pytest.approx(0.7)           # the weak tier alone

    def test_full_budget_sends_everything(self):
        row = budget_table(self.SCORES, self.NEEDED, [1.0])[0]
        assert row.sent == 1.0
        assert row.caught == 1.0
        assert row.quality == 1.0

    def test_threshold_is_the_calibrated_one(self):
        for row in budget_table(self.SCORES, self.NEEDED, [0.1, 0.2, 0.5]):
            assert row.threshold == calibrate_threshold(self.SCORES, row.strong_fraction)
            assert row.sent == pytest.approx(sum(s >= row.threshold for s in self.SCORES) / len(self.SCORES))

    def test_perfect_router_catches_everything_at_the_base_rate(self):
        scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.05, 0.04, 0.03]
        needed = [True, True, True] + [False] * 7
        row = budget_table(scores, needed, [0.3])[0]
        assert row.caught == 1.0 and row.quality == 1.0

    def test_tied_scores_can_overshoot_the_budget(self):
        """Everything at or above the threshold goes to strong, even if that is more than asked."""
        scores = [0.9, 0.5, 0.5, 0.5, 0.1]
        row = budget_table(scores, [False, True, False, False, False], [0.4])[0]
        assert row.threshold == 0.5
        assert row.sent == pytest.approx(0.8)
        assert row.caught == 1.0

    def test_quality_never_goes_down_as_the_budget_grows(self):
        rows = budget_table(self.SCORES, self.NEEDED, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        qualities = [r.quality for r in rows]
        assert qualities == sorted(qualities)

    def test_nothing_needed(self):
        row = budget_table([0.4, 0.2], [False, False], [0.5])[0]
        assert row.caught == 1.0 and row.quality == 1.0


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def train_planted(self):
        x, labels = planted(240, seed=1)
        winners, losers = pairs_for(labels[:160])
        model, losses = fit(
            x[:160], winners, losers, epochs=60, lr=1e-2, alpha=0.02, batch_size=32, seed=0, **{k: SMALL[k] for k in ("dim", "num_models")}
        )
        return model, losses, x, labels

    def test_learns_a_planted_pattern_and_generalizes(self):
        model, losses, x, labels = self.train_planted()
        assert losses[-1] < losses[0]
        held_out = strong_win_rate(model, x[160:], strong_id=S, weak_id=W)
        assert auc(held_out, labels[160:]) > 0.9

    def test_budget_table_beats_a_router_that_knows_nothing(self):
        model, _, x, labels = self.train_planted()
        held_out = strong_win_rate(model, x[160:], strong_id=S, weak_id=W)
        row = budget_table(held_out, labels[160:], [0.3])[0]
        assert row.caught > 0.3 + 0.15

    def test_fit_is_reproducible(self):
        x, labels = planted(60, seed=2)
        winners, losers = pairs_for(labels)
        kwargs = dict(epochs=5, lr=1e-2, alpha=0.1, batch_size=16, seed=3, dim=8, num_models=6)
        a, _ = fit(x, winners, losers, **kwargs)
        b, _ = fit(x, winners, losers, **kwargs)
        assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())

    def test_checkpoint_round_trip(self, tmp_path):
        model, _, x, _ = self.train_planted()
        export_checkpoint(model, tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "model.safetensors").exists()
        loaded = load_checkpoint(tmp_path / "ckpt")
        assert strong_win_rate(loaded, x[:8], strong_id=S, weak_id=W) == pytest.approx(
            strong_win_rate(model, x[:8], strong_id=S, weak_id=W), abs=1e-7
        )

    def test_routellm_loads_the_checkpoint_and_agrees(self, tmp_path, monkeypatch):
        """The point of the exercise: RouteLLM's own router class, your weights, the same number."""
        monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "sk-not-used"))
        pytest.importorskip("routellm", reason="pip install -r requirements.txt")
        import routellm.routers.matrix_factorization.model as routellm_mf
        from routellm.routers.routers import MatrixFactorizationRouter

        torch.manual_seed(0)
        model = MFModel()  # full size: 64 models, 128 dims, 1536-dim embeddings
        with torch.no_grad():  # untrained weights sit at a coin flip; push them away from it
            model.P.weight[STRONG_ID] += 1.0
            model.text_proj[0].weight *= 400.0
        export_checkpoint(model, tmp_path / "northstar-mf")
        prompt_embedding = embeds(1, seed=5, text_dim=impl.TEXT_DIM)

        class FakeEmbeddings:
            @staticmethod
            def create(input, model):  # noqa: A002 - mirrors the OpenAI client
                from types import SimpleNamespace

                return SimpleNamespace(data=[SimpleNamespace(embedding=prompt_embedding[0].tolist())])

        class FakeClient:
            embeddings = FakeEmbeddings

        monkeypatch.setattr(routellm_mf, "OPENAI_CLIENT", FakeClient)
        router = MatrixFactorizationRouter(checkpoint_path=str(tmp_path / "northstar-mf"))
        theirs = router.calculate_strong_win_rate("any prompt: the embedding is stubbed")
        ours = strong_win_rate(model, prompt_embedding)[0]
        assert theirs == pytest.approx(ours, abs=1e-6)
        assert abs(ours - 0.5) > 0.01, "the test model should not sit at a coin flip"
