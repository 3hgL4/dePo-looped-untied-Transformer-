"""Checks for matched-cohort probe isolation; no model training or extraction."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.probe_h2 import (
    LinearProbe,
    copy_query_state,
    complete_answer_spans,
    hash_evaluation_texts,
    make_probe_split,
    pointer_query_record,
    run_probe_grid,
    score_complete_answer_span,
    seed_probe_training,
    shuffle_training_labels,
    summarize_query_accuracy,
)


class MatchedProbeProtocolTests(unittest.TestCase):
    def test_prompt_split_has_no_overlap_even_with_skipped_queries(self):
        # Unequal group sizes and gaps model skipped prompts or query positions.
        prompt_ids = np.array([0, 0, 0, 2, 2, 5, 8, 8, 9, 9, 9, 9])
        train, test = make_probe_split(
            len(prompt_ids), 0.8, np.random.default_rng(7), prompt_ids,
        )
        self.assertEqual(np.intersect1d(prompt_ids[train], prompt_ids[test]).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, test])),
                                      np.arange(len(prompt_ids)))
        self.assertEqual(np.unique(prompt_ids[train]).size, 4)
        repeated = make_probe_split(
            len(prompt_ids), 0.8, np.random.default_rng(7), prompt_ids,
        )
        np.testing.assert_array_equal(train, repeated[0])
        np.testing.assert_array_equal(test, repeated[1])

    def test_default_split_preserves_legacy_query_indices(self):
        train, test = make_probe_split(20, 0.8, np.random.default_rng(3))
        legacy = np.random.default_rng(3).permutation(20)
        np.testing.assert_array_equal(train, legacy[:16])
        np.testing.assert_array_equal(test, legacy[16:])

    def test_grouped_control_cannot_take_labels_from_test_queries(self):
        targets = np.array([1, 2, 3, 4, 100, 101])
        train = np.arange(4)
        control = shuffle_training_labels(targets, train, np.random.default_rng(9))
        np.testing.assert_array_equal(np.sort(control), targets[train])
        self.assertFalse(np.isin(control, targets[4:]).any())

    def test_query_copy_owns_only_query_storage(self):
        batch_states = torch.arange(3 * 2 * 7 * 4).reshape(3, 2, 7, 4).float()
        expected = batch_states[:, 1, 5, :].numpy().copy()
        query = copy_query_state(batch_states, 1, 5)
        self.assertTrue(query.flags.owndata)
        self.assertIsNone(query.base)
        self.assertEqual(query.nbytes, 3 * 4 * 4)
        batch_states.zero_()
        np.testing.assert_array_equal(query, expected)

    def test_probe_seed_resets_initialization_and_minibatch_order(self):
        seed_probe_training(19)
        first = LinearProbe(4, 3)
        order = torch.randperm(23)
        torch.rand(100)  # An earlier model's construction must not influence us.
        seed_probe_training(19)
        second = LinearProbe(4, 3)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(order, torch.randperm(23)))

    def test_degenerate_group_split_fails_loudly(self):
        with self.assertRaises(ValueError):
            make_probe_split(5, 0.8, np.random.default_rng(0), np.zeros(5))

    def test_complete_node_requires_all_digits_and_boundary(self):
        # '?', space, '1', '0', ' |': logits[i] predicts tokens[i+1].
        tokens = [1, 2, 3, 4, 5]
        self.assertEqual(score_complete_answer_span([2, 3, 4, 5, 0], tokens, 1, 4),
                         (True, True, True))
        self.assertEqual(score_complete_answer_span([2, 3, 3, 5, 0], tokens, 1, 4),
                         (False, True, False))
        # Gold node 1 must not count when the model continues with a zero.
        self.assertEqual(score_complete_answer_span([2, 3, 4, 0], [1, 2, 3, 5], 1, 3),
                         (True, False, False))

    def test_complete_spans_reject_truncated_boundary_and_wrong_metadata(self):
        class Tokenizer:
            def decode(self, tokens):
                return "".join({0: "<eos>", 1: "?", 2: " ", 3: "1", 4: "0", 5: " |"}[t]
                               for t in tokens)

        tokenizer = Tokenizer()
        tokens = [1, 2, 3, 4, 5, 1, 2, 3, 5, 0]
        valid = [True] * 9 + [False]
        self.assertEqual(complete_answer_spans(tokens, valid, {1}, {5}, tokenizer, [10, 1]),
                         [(1, 4), (6, 8)])
        with self.assertRaisesRegex(ValueError, "separator"):
            complete_answer_spans(tokens[:8] + [0], [True] * 8 + [False],
                                  {1}, {5}, tokenizer, [10, 1])
        with self.assertRaisesRegex(ValueError, "metadata"):
            complete_answer_spans(tokens, valid, {1}, {5}, tokenizer, [11, 1])

    def test_query_strata_use_earlier_original_queries_even_when_not_retained(self):
        meta = {"queries": [{"start": 7, "answer": 2}, {"start": 4, "answer": 1},
                            {"start": 7, "answer": 2}]}
        first = pointer_query_record(meta, 8, 0)
        self.assertTrue(first["is_first_query"])
        self.assertTrue(first["is_fresh_start"])
        repeated = pointer_query_record(meta, 8, 2)
        self.assertTrue(repeated["is_repeated_start"])
        self.assertFalse(repeated["is_fresh_start"])
        self.assertEqual(repeated["query_indices"], 2)
        fresh = pointer_query_record(meta, 8, 1)
        self.assertFalse(fresh["is_first_query"])
        self.assertTrue(fresh["is_fresh_start"])

    def test_empty_model_stratum_is_undefined(self):
        metadata = {"is_first_query": np.array([True, False]),
                    "is_fresh_start": np.array([True, True]),
                    "is_repeated_start": np.array([False, False])}
        summary = summarize_query_accuracy(np.array([True, False]), metadata)
        self.assertEqual(summary["all"]["accuracy"], 0.5)
        self.assertEqual(summary["first_query"]["accuracy"], 1.0)
        self.assertEqual(summary["repeated_start"]["n_evaluated"], 0)
        self.assertIsNone(summary["repeated_start"]["accuracy"])

    def test_stratified_probes_reuse_fits_and_only_score_test_rows(self):
        states = np.zeros((6, 1, 2), dtype=np.float32)
        targets = np.arange(6, dtype=np.int64).reshape(-1, 1)
        _, test = make_probe_split(6, 0.5, np.random.default_rng(0))
        selected = np.zeros(6, dtype=bool)
        selected[test[0]] = True
        output = {}

        def fake_fit(xtr, ytr, xte, yte, **kwargs):
            self.assertTrue(kwargs["return_predictions"])
            predicted = yte.copy()
            predicted[0] = (predicted[0] + 1) % 6
            return float((predicted == yte).mean()), predicted

        with patch("src.probe_h2.train_probe", side_effect=fake_fit) as fit:
            result = run_probe_grid(
                states, targets, 6, 0.5, 0, "cpu", probe_types=("linear",),
                query_strata={"selected": selected, "empty": np.zeros(6, dtype=bool)},
                stratified_accuracies=output,
            )
        self.assertEqual(fit.call_count, 2)  # Real and shuffled, no extra stratum fits.
        self.assertAlmostEqual(float(result["linear_real"][0, 0]), 2 / 3)
        self.assertEqual(float(output["linear_real__selected"][0, 0]), 0.0)
        self.assertTrue(np.isnan(output["linear_real__empty"]).all())

    def test_scoring_suffix_is_distinguished_in_text_provenance(self):
        original = ["hop 4 7 ? 1", "hop 4 2 ? 10"]
        self.assertNotEqual(hash_evaluation_texts(original),
                            hash_evaluation_texts([text + " |" for text in original]))


if __name__ == "__main__":
    unittest.main()
