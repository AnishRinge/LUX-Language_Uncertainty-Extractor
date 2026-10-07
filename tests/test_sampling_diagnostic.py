"""Unit tests for the pure helper logic of the CP4.2.1 sampling diagnostic.

These tests cover ONLY the list-based helper functions in
``run_cp4_2_sampling_diagnostic``; they do not load model weights, call
``model.generate`` or touch any CP4.2 artifact. They can therefore run without
the Qwen3 weights or a GPU.
"""

import math
import unittest

from run_cp4_2_sampling_diagnostic import (
    softmax_list,
    nucleus_size_list,
    topk_list,
    rank_of_token,
    top1_prob_list,
)


class TestSoftmaxList(unittest.TestCase):
    def test_uniform(self):
        probs = softmax_list([0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(all(abs(p - 0.25) < 1e-9 for p in probs))

    def test_peaked(self):
        probs = softmax_list([10.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(probs[0] > 0.99)

    def test_minus_inf_is_zero(self):
        probs = softmax_list([float("-inf"), 0.0, 0.0])
        self.assertEqual(probs[0], 0.0)
        self.assertAlmostEqual(probs[1], 0.5)
        self.assertAlmostEqual(probs[2], 0.5)

    def test_empty(self):
        self.assertEqual(softmax_list([]), [])


class TestNucleusSizeList(unittest.TestCase):
    def test_all_in_nucleus(self):
        self.assertEqual(nucleus_size_list([0.0, 0.0, 0.0]), 3)

    def test_filtered_tokens_excluded(self):
        logits = [float("-inf"), float("-inf"), 0.0, 0.0]
        self.assertEqual(nucleus_size_list(logits), 2)

    def test_single_token_nucleus(self):
        logits = [10.0, float("-inf"), float("-inf")]
        self.assertEqual(nucleus_size_list(logits), 1)


class TestTopkList(unittest.TestCase):
    def test_ordering_and_count(self):
        logits = [0.0, 1.0, 2.0, 3.0]
        top = topk_list(logits, 3)
        self.assertEqual(len(top), 3)
        expected_p3 = math.exp(3) / sum(math.exp(i) for i in range(4))
        self.assertEqual(top[0][0], 3)
        self.assertAlmostEqual(top[0][1], expected_p3, places=12)
        # strictly descending probability
        self.assertTrue(top[0][1] > top[1][1] > top[2][1])

    def test_excludes_zero_prob(self):
        logits = [10.0, float("-inf"), float("-inf")]
        self.assertEqual(topk_list(logits, 5), [(0, 1.0)])

    def test_top1(self):
        top = topk_list([1.0, 5.0, 2.0], 1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][0], 1)


class TestRankOfToken(unittest.TestCase):
    def test_rank_one_for_best(self):
        logits = [0.0, 1.0, 2.0]
        self.assertEqual(rank_of_token(logits, 2), 1)

    def test_rank_none_when_zero_prob(self):
        logits = [10.0, float("-inf")]
        self.assertIsNone(rank_of_token(logits, 1))

    def test_rank_two(self):
        logits = [0.0, 1.0, 2.0]
        self.assertEqual(rank_of_token(logits, 1), 2)


class TestTop1ProbList(unittest.TestCase):
    def test_normalized(self):
        self.assertAlmostEqual(top1_prob_list([0.0, 0.0]), 0.5)

    def test_peaked_near_one(self):
        self.assertGreater(top1_prob_list([20.0, 0.0]), 0.999)


if __name__ == "__main__":
    unittest.main()
