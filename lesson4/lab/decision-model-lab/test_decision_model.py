#!/usr/bin/env python3
"""
Test suite for the decision-model exercise.

Run with:   pytest -v

These tests are your definition of "done". They import your implementation
from `decision_model.py` by default. (Instructors can point the tests at a
reference implementation with `DECISION_MODEL_IMPL=reference_solution pytest`.)

Everything here runs offline. Most tests use small hand-made candidate
lists. The last two groups use real data: answers a decision model returned
through `POST /v1/systemone` (your confidence has to match the API's), and
the recording of a chat model's first tokens in `fixtures/recorded.json`.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import string
from pathlib import Path

import pytest

impl = importlib.import_module(os.environ.get("DECISION_MODEL_IMPL", "decision_model"))

option_probs = impl.option_probs
temper = impl.temper
noul = impl.noul
choice = impl.choice
score = impl.score
expected_calibration_error = impl.expected_calibration_error
fit_temperature = impl.fit_temperature
cascade = impl.cascade
nll = impl.nll
auc = impl.auc
load_recorded = impl.load_recorded


def cands(**mass: float) -> list[dict]:
    """Candidates from {token: probability}, most likely first. Use `tok(...)` for tokens that are not identifiers."""
    return tok(*mass.items())


def tok(*pairs: tuple[str, float]) -> list[dict]:
    return [{"token": t, "logprob": math.log(p)} for t, p in sorted(pairs, key=lambda tp: -tp[1])]


LETTERS = {"A": "refunds", "B": "returns", "C": "shipping", "D": "billing"}


# --------------------------------------------------------------------------- #
# option_probs
# --------------------------------------------------------------------------- #
class TestOptionProbs:
    def test_two_clean_tokens(self):
        probs = option_probs(cands(yes=0.75, no=0.25), ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.75, "no": 0.25})

    def test_sums_case_and_whitespace_variants(self):
        top = tok(("yes", 0.5), ("no", 0.2), (" yes", 0.1), ("Yes", 0.1), (" No", 0.1))
        probs = option_probs(top, ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.7, "no": 0.3})

    def test_junk_tokens_are_discarded_and_the_rest_renormalized(self):
        top = tok(("yes", 0.6), ("no", 0.2), (" Yes", 0.1), ("The", 0.1))
        probs = option_probs(top, ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.7 / 0.9, "no": 0.2 / 0.9})
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_partial_and_longer_tokens_do_not_count(self):
        top = tok(("yes", 0.4), ("no", 0.4), ("y", 0.1), ("yesterday", 0.05), ("not", 0.05))
        probs = option_probs(top, ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.5, "no": 0.5})

    def test_think_token_is_not_an_answer(self):
        top = tok(("<think>", 0.5), ("no", 0.3), ("yes", 0.2))
        probs = option_probs(top, ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.4, "no": 0.6})

    def test_missing_option_gets_the_floor_not_zero(self):
        probs = option_probs(cands(yes=1.0), ["yes", "no"], floor=1e-6)
        assert probs["no"] > 0
        assert probs["no"] == pytest.approx(1e-6 / (1.0 + 1e-6))
        assert probs["yes"] == pytest.approx(1.0 / (1.0 + 1e-6))

    def test_floor_is_a_parameter(self):
        probs = option_probs(cands(yes=1.0), ["yes", "no"], floor=0.25)
        assert probs == pytest.approx({"yes": 0.8, "no": 0.2})

    def test_no_candidate_matches_gives_uniform(self):
        probs = option_probs(tok(("The", 0.9), ("<think>", 0.1)), ["yes", "no"])
        assert probs == pytest.approx({"yes": 0.5, "no": 0.5})

    def test_letter_options(self):
        top = tok(("B", 0.5), ("A", 0.25), (" B", 0.125), ("b", 0.125))
        probs = option_probs(top, ["A", "B", "C"])
        assert probs["B"] == pytest.approx(0.75, abs=1e-5)
        assert probs["A"] == pytest.approx(0.25, abs=1e-5)
        assert probs["C"] == pytest.approx(1e-6, abs=1e-6)

    def test_digit_options(self):
        probs = option_probs(tok(("3", 0.5), ("4", 0.3), ("2", 0.2)), ["1", "2", "3", "4", "5"])
        assert max(probs, key=probs.get) == "3"
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_keys_follow_the_order_of_options(self):
        probs = option_probs(cands(no=0.9, yes=0.1), ["yes", "no"])
        assert list(probs) == ["yes", "no"]

    def test_input_is_not_modified(self):
        top = cands(yes=0.75, no=0.25)
        before = [dict(c) for c in top]
        option_probs(top, ["yes", "no"])
        assert top == before


# --------------------------------------------------------------------------- #
# temper
# --------------------------------------------------------------------------- #
class TestTemper:
    P = {"a": 0.7, "b": 0.2, "c": 0.1}

    def test_temperature_one_is_the_identity(self):
        assert temper(self.P, 1.0) == pytest.approx(self.P)

    def test_matches_the_formula(self):
        t = 2.0
        raw = {k: v ** (1 / t) for k, v in self.P.items()}
        want = {k: v / sum(raw.values()) for k, v in raw.items()}
        assert temper(self.P, t) == pytest.approx(want)

    def test_high_temperature_flattens(self):
        out = temper(self.P, 4.0)
        assert out["a"] < self.P["a"]
        assert out["c"] > self.P["c"]

    def test_low_temperature_sharpens(self):
        out = temper(self.P, 0.5)
        assert out["a"] > self.P["a"]
        assert out["c"] < self.P["c"]

    def test_ranking_never_changes(self):
        for t in (0.3, 0.7, 1.0, 2.5, 9.0):
            out = temper(self.P, t)
            assert sorted(out, key=out.get, reverse=True) == ["a", "b", "c"]

    def test_always_sums_to_one(self):
        for t in (0.25, 1.0, 3.0, 10.0):
            assert sum(temper(self.P, t).values()) == pytest.approx(1.0)

    def test_very_high_temperature_approaches_uniform(self):
        out = temper(self.P, 1000.0)
        assert all(v == pytest.approx(1 / 3, abs=0.01) for v in out.values())

    def test_tiny_probabilities_do_not_overflow_or_vanish(self):
        out = temper({"a": 1.0 - 1e-12, "b": 1e-12}, 0.25)
        assert math.isfinite(out["a"]) and math.isfinite(out["b"])
        assert out["a"] == pytest.approx(1.0)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_keys_and_order_are_kept(self):
        assert list(temper(self.P, 2.0)) == ["a", "b", "c"]

    def test_input_is_not_modified(self):
        p = dict(self.P)
        temper(p, 3.0)
        assert p == self.P

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_temperature(self, bad):
        with pytest.raises(ValueError):
            temper(self.P, bad)


# --------------------------------------------------------------------------- #
# noul
# --------------------------------------------------------------------------- #
class TestNoul:
    def test_returns_probability_of_yes(self):
        assert noul(cands(yes=0.8, no=0.2)) == pytest.approx(0.8)

    def test_merges_variants_and_ignores_junk(self):
        top = tok(("no", 0.4), ("yes", 0.3), ("Yes", 0.1), ("The", 0.2))
        assert noul(top) == pytest.approx(0.5)

    def test_is_a_float_strictly_between_zero_and_one(self):
        for top in (cands(yes=1.0), cands(no=1.0)):
            p = noul(top)
            assert isinstance(p, float)
            assert 0.0 < p < 1.0

    def test_temperature_pulls_toward_one_half(self):
        top = cands(yes=0.9, no=0.1)
        assert 0.5 < noul(top, temperature=3.0) < noul(top) < noul(top, temperature=0.5)

    def test_temperature_is_applied_after_renormalizing(self):
        top = tok(("yes", 0.45), ("no", 0.05), ("The", 0.5))      # 0.9 / 0.1 once junk is dropped
        want = 0.9**0.5 / (0.9**0.5 + 0.1**0.5)
        assert noul(top, temperature=2.0) == pytest.approx(want)


# --------------------------------------------------------------------------- #
# choice
# --------------------------------------------------------------------------- #
class TestChoice:
    def test_reports_the_winner_by_name(self):
        out = choice(cands(B=0.7, A=0.2, D=0.1), LETTERS)
        assert out["choice"] == "returns"

    def test_probabilities_are_keyed_by_name_in_letter_order(self):
        out = choice(cands(B=0.7, A=0.2, D=0.1), LETTERS)
        assert list(out["probabilities"]) == ["refunds", "returns", "shipping", "billing"]
        assert sum(out["probabilities"].values()) == pytest.approx(1.0)
        assert out["probabilities"]["returns"] == pytest.approx(0.7, abs=1e-5)

    def test_confidence_formula(self):
        out = choice(cands(B=0.7, A=0.2, D=0.1), LETTERS)
        p = out["probabilities"]["returns"]
        assert out["confidence"] == pytest.approx((p - 0.25) / 0.75)

    def test_uniform_has_zero_confidence(self):
        out = choice(cands(A=0.25, B=0.25, C=0.25, D=0.25), LETTERS)
        assert out["confidence"] == pytest.approx(0.0, abs=1e-9)

    def test_certain_has_confidence_near_one(self):
        out = choice(cands(C=1.0), LETTERS)
        assert out["choice"] == "shipping"
        assert out["confidence"] == pytest.approx(1.0, abs=1e-4)

    def test_same_probability_means_less_with_fewer_options(self):
        two = choice(cands(A=0.6, B=0.4), {"A": "x", "B": "y"})
        four = choice(cands(A=0.6, B=0.2, C=0.1, D=0.1), LETTERS)
        assert two["confidence"] == pytest.approx(0.2)
        assert four["confidence"] == pytest.approx((0.6 - 0.25) / 0.75)
        assert four["confidence"] > two["confidence"]

    def test_tie_goes_to_the_first_letter(self):
        out = choice(cands(C=0.4, B=0.4, A=0.2), LETTERS)
        assert out["choice"] == "returns"

    def test_lowercase_and_spaced_letters_count(self):
        out = choice(tok(("d", 0.4), (" D", 0.3), ("A", 0.3)), LETTERS)
        assert out["choice"] == "billing"
        assert out["probabilities"]["billing"] == pytest.approx(0.7, abs=1e-5)

    def test_temperature_lowers_confidence_but_not_the_choice(self):
        top = cands(B=0.9, A=0.05, C=0.03, D=0.02)
        cold, warm = choice(top, LETTERS), choice(top, LETTERS, temperature=3.0)
        assert cold["choice"] == warm["choice"] == "returns"
        assert warm["confidence"] < cold["confidence"]
        assert sum(warm["probabilities"].values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
class TestScore:
    def test_mode_expected_and_probabilities(self):
        out = score(tok(("4", 0.6), ("5", 0.2), ("3", 0.2)), levels=5)
        assert out["score"] == 4
        assert isinstance(out["score"], int)
        assert out["expected"] == pytest.approx(4.0, abs=1e-4)
        assert list(out["probabilities"]) == [1, 2, 3, 4, 5]
        assert out["probabilities"][4] == pytest.approx(0.6, abs=1e-5)

    def test_confidence_formula(self):
        out = score(tok(("4", 0.6), ("5", 0.2), ("3", 0.2)), levels=5)
        # mean distance from level 4 is 0.2*1 + 0.2*1 = 0.4; a know-nothing answer on 5 levels has 1.2
        assert out["confidence"] == pytest.approx(1 - 0.4 / 1.2, abs=1e-4)

    def test_baseline_depends_on_the_number_of_levels(self):
        two = score(tok(("1", 0.8), ("2", 0.2)), levels=2)        # spread 0.2, baseline 0.5
        three = score(tok(("2", 0.8), ("3", 0.2)), levels=3)      # spread 0.2, baseline 2/3
        four = score(tok(("2", 0.7), ("3", 0.3)), levels=4)       # spread 0.3, baseline 1.0
        assert two["confidence"] == pytest.approx(0.6, abs=1e-4)
        assert three["confidence"] == pytest.approx(0.7, abs=1e-4)
        assert four["confidence"] == pytest.approx(0.7, abs=1e-4)

    def test_all_mass_on_one_level_is_confident(self):
        out = score(tok(("2", 1.0)), levels=5)
        assert out["score"] == 2
        assert out["confidence"] == pytest.approx(1.0, abs=1e-4)

    def test_near_misses_cost_less_than_far_misses(self):
        near = score(tok(("3", 0.6), ("2", 0.2), ("4", 0.2)), levels=5)
        far = score(tok(("3", 0.6), ("1", 0.2), ("5", 0.2)), levels=5)
        assert near["score"] == far["score"] == 3
        assert near["confidence"] > far["confidence"]
        assert near["confidence"] == pytest.approx(1 - 0.4 / 1.2, abs=1e-4)
        assert far["confidence"] == pytest.approx(1 - 0.8 / 1.2, abs=1e-4)

    def test_expected_can_sit_between_levels(self):
        out = score(tok(("1", 0.5), ("5", 0.5)), levels=5)
        assert out["expected"] == pytest.approx(3.0, abs=1e-4)

    def test_tie_goes_to_the_lower_level(self):
        out = score(tok(("1", 0.5), ("5", 0.5)), levels=5)
        assert out["score"] == 1
        # half the mass is four levels away: worse than knowing nothing, so the confidence bottoms out
        assert out["confidence"] == pytest.approx(0.0, abs=1e-9)

    def test_confidence_is_never_negative(self):
        for top in (tok(("1", 0.34), ("5", 0.33), ("3", 0.33)), tok(("The", 1.0))):
            assert score(top, levels=5)["confidence"] >= 0.0

    def test_tokens_outside_the_scale_are_ignored(self):
        out = score(tok(("2", 0.5), ("7", 0.3), ("3", 0.2)), levels=3)
        assert out["score"] == 2
        assert out["probabilities"][2] == pytest.approx(0.5 / 0.7, abs=1e-5)

    def test_temperature_keeps_the_level(self):
        top = tok(("4", 0.8), ("3", 0.1), ("5", 0.1))
        cold, warm = score(top, levels=5), score(top, levels=5, temperature=4.0)
        assert cold["score"] == warm["score"] == 4
        assert warm["confidence"] < cold["confidence"]


# --------------------------------------------------------------------------- #
# expected_calibration_error
# --------------------------------------------------------------------------- #
class TestECE:
    def test_empty_is_zero(self):
        assert expected_calibration_error([]) == 0.0

    def test_perfectly_calibrated_bin(self):
        scored = [(0.8, True)] * 8 + [(0.8, False)] * 2
        assert expected_calibration_error(scored) == pytest.approx(0.0, abs=1e-12)

    def test_overconfident_model(self):
        scored = [(0.9, True)] * 6 + [(0.9, False)] * 4       # says 90%, is right 60% of the time
        assert expected_calibration_error(scored) == pytest.approx(0.3)

    def test_underconfident_model(self):
        scored = [(0.55, True)] * 10                          # says 55%, is always right
        assert expected_calibration_error(scored) == pytest.approx(0.45)

    def test_bins_are_weighted_by_size(self):
        scored = [(0.9, True)] * 3 + [(0.9, False)] + [(0.3, False)] * 4       # gaps 0.15 and 0.30, half each
        assert expected_calibration_error(scored) == pytest.approx(0.5 * 0.15 + 0.5 * 0.30)

    def test_uses_the_mean_confidence_of_each_bin(self):
        scored = [(0.82, True), (0.98, True)]                 # one bin, mean confidence 0.90, accuracy 1.0
        assert expected_calibration_error(scored, bins=5) == pytest.approx(0.10)

    def test_confidence_of_exactly_one_goes_in_the_last_bin(self):
        assert expected_calibration_error([(1.0, True), (1.0, True)]) == pytest.approx(0.0)
        assert expected_calibration_error([(1.0, False)]) == pytest.approx(1.0)

    def test_number_of_bins_matters(self):
        scored = [(0.55, False), (0.95, True)]
        assert expected_calibration_error(scored, bins=1) == pytest.approx(0.25)    # 0.75 claimed, 0.5 right
        assert expected_calibration_error(scored, bins=5) == pytest.approx(0.5 * 0.55 + 0.5 * 0.05)

    def test_is_between_zero_and_one(self):
        scored = [(0.99, False)] * 5 + [(0.51, True)] * 5
        assert 0.0 <= expected_calibration_error(scored) <= 1.0


# --------------------------------------------------------------------------- #
# fit_temperature
# --------------------------------------------------------------------------- #
def examples(confidence: float, n_right: int, n_wrong: int) -> list[tuple[dict, str]]:
    """A model that always says "yes" with `confidence` and is right `n_right` times out of n_right + n_wrong."""
    probs = {"yes": confidence, "no": 1 - confidence}
    return [(probs, "yes")] * n_right + [(probs, "no")] * n_wrong


class TestFitTemperature:
    def test_overconfident_model_needs_temperature_above_one(self):
        assert fit_temperature(examples(0.95, 7, 3)) > 1.0

    def test_underconfident_model_needs_temperature_below_one(self):
        assert fit_temperature(examples(0.6, 19, 1)) < 1.0

    def test_calibrated_model_stays_near_one(self):
        t = fit_temperature(examples(0.8, 80, 20))
        assert 0.85 <= t <= 1.15

    def test_returns_a_value_from_the_grid(self):
        grid = [0.5, 1.0, 2.0, 4.0]
        assert fit_temperature(examples(0.95, 7, 3), grid) in grid

    def test_picks_the_lowest_nll_on_the_grid(self):
        ex = examples(0.95, 7, 3)
        grid = [0.5, 1.0, 2.0, 4.0]
        best = fit_temperature(ex, grid)
        assert all(nll(ex, best) <= nll(ex, t) + 1e-12 for t in grid)

    def test_recovers_the_right_confidence(self):
        ex = examples(0.95, 70, 30)                       # should be saying 0.70
        t = fit_temperature(ex)
        tempered = temper({"yes": 0.95, "no": 0.05}, t)
        assert tempered["yes"] == pytest.approx(0.70, abs=0.03)

    def test_fitting_lowers_calibration_error(self):
        ex = examples(0.95, 70, 30)
        t = fit_temperature(ex)
        before = [(p["yes"], truth == "yes") for p, truth in ex]
        after = [(temper(p, t)["yes"], truth == "yes") for p, truth in ex]
        assert expected_calibration_error(after) < expected_calibration_error(before)

    def test_tie_goes_to_the_lowest_temperature(self):
        uniform = [({"yes": 0.5, "no": 0.5}, "yes")] * 4      # temperature cannot move a uniform distribution
        assert fit_temperature(uniform, [3.0, 1.0, 2.0]) == 1.0

    def test_default_grid_spans_a_quarter_to_about_ten(self):
        assert fit_temperature(examples(0.999999, 1, 1)) == pytest.approx(0.25 * 1.1**39, abs=1e-3)
        assert fit_temperature(examples(0.51, 50, 0)) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# cascade
# --------------------------------------------------------------------------- #
SCORED = [(0.95, True), (0.90, True), (0.85, False), (0.70, True), (0.60, False), (0.40, True), (0.30, False), (0.10, False)]


class TestCascade:
    def test_empty(self):
        assert cascade([], 0.5, 0.001, 0.01) == {"coverage": 0.0, "error_rate": 0.0, "usd_per_decision": 0.0}

    def test_coverage_counts_items_at_or_above_the_threshold(self):
        assert cascade(SCORED, 0.85, 0.0, 0.0)["coverage"] == pytest.approx(3 / 8)
        assert cascade(SCORED, 0.86, 0.0, 0.0)["coverage"] == pytest.approx(2 / 8)

    def test_error_rate_is_over_all_items(self):
        out = cascade(SCORED, 0.60, 0.0, 0.0)             # five taken, two of them wrong
        assert out["coverage"] == pytest.approx(5 / 8)
        assert out["error_rate"] == pytest.approx(2 / 8)

    def test_escalated_items_are_never_errors(self):
        assert cascade(SCORED, 0.99, 0.0, 0.0) == pytest.approx({"coverage": 0.0, "error_rate": 0.0, "usd_per_decision": 0.0})

    def test_cost_is_the_gate_plus_the_escalated_share(self):
        out = cascade(SCORED, 0.85, gate_usd=0.0001, escalate_usd=0.004)
        assert out["usd_per_decision"] == pytest.approx(0.0001 + (5 / 8) * 0.004)

    def test_taking_everything_costs_only_the_gate(self):
        out = cascade(SCORED, 0.0, gate_usd=0.0001, escalate_usd=0.004)
        assert out["coverage"] == 1.0
        assert out["usd_per_decision"] == pytest.approx(0.0001)
        assert out["error_rate"] == pytest.approx(4 / 8)

    def test_taking_nothing_costs_the_gate_and_the_escalation(self):
        out = cascade(SCORED, 1.01, gate_usd=0.0001, escalate_usd=0.004)
        assert out["usd_per_decision"] == pytest.approx(0.0041)

    def test_lower_threshold_is_cheaper_and_never_safer(self):
        rows = [cascade(SCORED, t, 0.0001, 0.004) for t in (0.95, 0.85, 0.60, 0.30)]
        costs = [r["usd_per_decision"] for r in rows]
        errors = [r["error_rate"] for r in rows]
        assert costs == sorted(costs, reverse=True)
        assert errors == sorted(errors)


# --------------------------------------------------------------------------- #
# The real API: your confidence against the one a decision model reported
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def api_answers() -> dict:
    return json.loads((Path(__file__).parent / "fixtures" / "systemone_answers.json").read_text())


class TestAgainstTheRealAPI:
    """fixtures/systemone_answers.json holds answers from POST /v1/systemone with the probabilities
    and the confidence the API returned. Candidates are rebuilt from those probabilities, so your
    functions see what the API saw. They have to report the confidence it reported."""

    def test_fixture_shape(self, api_answers):
        assert len(api_answers["choice"]) == 24
        assert len(api_answers["score"]) == 30

    def test_choice_confidence_matches_the_api(self, api_answers):
        for answer in api_answers["choice"]:
            letters = dict(zip(string.ascii_uppercase, answer["options"]))
            top = tok(*[(letter, answer["probabilities"][name]) for letter, name in letters.items()])
            mine = choice(top, letters)
            assert mine["choice"] == answer["choice"]
            assert mine["confidence"] == pytest.approx(answer["confidence"], abs=1e-4), answer["options"]

    def test_score_confidence_matches_the_api(self, api_answers):
        for answer in api_answers["score"]:
            levels = len(answer["levels"])
            top = tok(*[(str(i + 1), p) for i, p in enumerate(answer["probabilities"])])
            mine = score(top, levels)
            assert mine["confidence"] == pytest.approx(answer["confidence"], abs=1e-4), answer["levels"]

    def test_expected_level_matches_the_api_score(self, api_answers):
        for answer in api_answers["score"]:
            top = tok(*[(str(i + 1), p) for i, p in enumerate(answer["probabilities"])])
            mine = score(top, len(answer["levels"]))
            assert mine["expected"] - 1 == pytest.approx(answer["score"], abs=1e-4)   # the API numbers levels from 0


# --------------------------------------------------------------------------- #
# The real recording: your functions on a chat model's actual first tokens
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def recorded() -> dict:
    return load_recorded()


class TestOnTheRecording:
    def test_fixture_shape(self, recorded):
        assert len(recorded["noul"]) == 66
        assert len(recorded["choice"]) == 48
        assert set(recorded["letters"].values()) == set(recorded["options"])

    def test_noul_probabilities_are_valid(self, recorded):
        for item in recorded["noul"]:
            assert 0.0 < noul(item["top"]) < 1.0

    def test_first_token_carries_real_signal_on_yes_no(self, recorded):
        scored = [(noul(item["top"]), item["label"]) for item in recorded["noul"]]
        assert auc(scored) > 0.70          # 0.5 is a coin; the labels are good draft / bad draft

    def test_choice_is_mostly_right(self, recorded):
        answers = [choice(item["top"], recorded["letters"]) for item in recorded["choice"]]
        right = sum(1 for a, item in zip(answers, recorded["choice"]) if a["choice"] == item["label"])
        assert right / len(answers) > 0.80

    def test_choice_confidence_is_higher_when_right(self, recorded):
        answers = [choice(item["top"], recorded["letters"]) for item in recorded["choice"]]
        right = [a["confidence"] for a, item in zip(answers, recorded["choice"]) if a["choice"] == item["label"]]
        wrong = [a["confidence"] for a, item in zip(answers, recorded["choice"]) if a["choice"] != item["label"]]
        assert wrong, "the recording contains mistakes; a perfect score means labels leaked"
        assert sum(right) / len(right) > sum(wrong) / len(wrong)

    def test_the_chat_model_is_overconfident_on_choice(self, recorded):
        ex = [(choice(item["top"], recorded["letters"])["probabilities"], item["label"]) for item in recorded["choice"]]
        assert fit_temperature(ex) > 1.0

    def test_temperature_fit_on_half_helps_the_other_half(self, recorded):
        ex = [(choice(item["top"], recorded["letters"])["probabilities"], item["label"]) for item in recorded["choice"]]
        fit, held_out = ex[0::2], ex[1::2]
        t = fit_temperature(fit)
        assert nll(held_out, t) < nll(held_out, 1.0)

    def test_a_threshold_trades_coverage_for_errors(self, recorded):
        scored = [(noul(item["top"]), item["label"]) for item in recorded["noul"]]
        loose, strict = cascade(scored, 0.5, 0.0001, 0.004), cascade(scored, 0.8, 0.0001, 0.004)
        assert loose["coverage"] > strict["coverage"]
        assert loose["error_rate"] > strict["error_rate"]
        assert loose["usd_per_decision"] < strict["usd_per_decision"]
